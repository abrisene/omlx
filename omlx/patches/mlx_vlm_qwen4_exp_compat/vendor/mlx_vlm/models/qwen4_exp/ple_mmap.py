"""Shared SSD PLE tensor reads and page advice."""
from __future__ import annotations

import json
import math
import mmap
import os
import struct
from concurrent.futures import wait
from pathlib import Path
from threading import Lock

import numpy as np
import mlx.core as mx

from . import language as runtime

logger = runtime.logger


class _SafeTensorMMap:
    """Read selected dense or affine-packed rows without resident weights."""

    def __init__(self, path: Path, *, layout: dict | None = None):
        runtime._ple_require_owner()
        self._owner_pid = os.getpid()
        self._resource_lock = Lock()
        self.path = path
        self._file = None
        self._mapping = None
        with runtime._PLE_RESOURCE_LOCK:
            try:
                # FileIO has no buffered-reader mutex inherited from another thread.
                self._file = path.open("rb", buffering=0)
                if layout is None:
                    header_size = struct.unpack("<Q", self._file.read(8))[0]
                    self._header = json.loads(self._file.read(header_size))
                    self._data_start = 8 + header_size
                else:
                    self._header = layout
                    self._data_start = 0
                self._mapping = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
                self._seen_pages = bytearray(
                    1 + (max(path.stat().st_size, 1) - 1) // runtime._PLE_PAGE_SIZE
                )
                self._last_rearm = 0.0
                self._rearm_count = 0
                try:
                    self._mapping.madvise(mmap.MADV_RANDOM)
                except (AttributeError, OSError):
                    pass
            except BaseException:
                self._close_resources()
                raise
        runtime.register_ple_resource(self)

    def _require_owner(self):
        runtime._ple_require_owner(self._owner_pid)

    def tensor_shape(self, key: str) -> tuple[int, ...]:
        self._require_owner()
        return tuple(self._header[key]["shape"])

    def tensor_dtype(self, key: str) -> str:
        self._require_owner()
        return str(self._header[key]["dtype"])

    def rows_np(self, key: str, rows) -> tuple[np.ndarray, str]:
        """Copy the requested rows out of the mapping; returns the raw array and the safetensors dtype."""
        self._require_owner()
        with self._resource_lock:
            if self._mapping is None or self._file is None:
                raise RuntimeError("SSD-backed Qwen4 PLE reader is closed")
            return self._rows_np_owned(key, rows)

    def _rows_np_owned(self, key, rows):
        entry = self._header[key]
        shape = tuple(entry["shape"])
        start, end = entry["data_offsets"]
        dtype = entry["dtype"]
        dtype_info = {
            "BF16": (np.dtype("<u2"), 2),
            "F16": (np.dtype("<f2"), 2),
            "F32": (np.dtype("<f4"), 4),
            "U32": (np.dtype("<u4"), 4),
            "U8": (np.dtype("u1"), 1),
            "F8_E4M3": (np.dtype("u1"), 1),
        }.get(dtype)
        if dtype_info is None:
            raise TypeError(f"SSD-backed Qwen4 PLE does not support {dtype}")
        np_dtype, item_size = dtype_info
        if len(shape) != 2 or end - start != math.prod(shape) * item_size:
            raise ValueError(f"Invalid sparse PLE tensor layout for {key}")
        row_indices = np.asarray(rows, dtype=np.intp)
        if row_indices.size == 0:
            return np.empty((0, shape[1]), dtype=np_dtype), dtype
        row_bytes = shape[1] * item_size
        row_stride = entry.get("row_stride", row_bytes)
        gather_start = None
        if row_indices.size > runtime._PLE_POOL_PREFETCH_ROWS:
            fully_seen = self._prefetch_missing_pages(
                row_indices,
                self._data_start + start,
                row_bytes,
                row_stride=row_stride,
            )
            gather_start = runtime.time.perf_counter() if fully_seen else None
        view = np.ndarray(
            shape,
            dtype=np_dtype,
            buffer=self._mapping,
            offset=self._data_start + start,
            strides=(row_stride, item_size),
        )
        # Integer-array indexing already copies the selected rows out of mmap.
        copied = view[row_indices]
        if gather_start is not None:
            self._rearm_if_slow(runtime.time.perf_counter() - gather_start, row_indices.size)
        return copied, dtype

    @staticmethod
    def to_mx(copied: np.ndarray, dtype: str) -> mx.array:
        if dtype == "BF16":
            values = (copied.astype(np.uint32) << np.uint32(16)).view(np.float32)
            return mx.array(values).astype(mx.bfloat16)
        if dtype == "F8_E4M3":
            return mx.from_fp8(mx.array(copied), dtype=mx.bfloat16)
        return mx.array(copied)

    def rows(self, key: str, rows: list[int]) -> mx.array:
        return self.to_mx(*self.rows_np(key, rows))

    def advise_rows(self, key: str, rows: np.ndarray) -> None:
        """Ask the kernel to start reading these rows' pages now.

        Copying the rows faults their missing pages in one at a time. Advised
        first, the kernel reads all of them concurrently; resident pages cost
        one system call.
        """
        self._require_owner()
        with self._resource_lock:
            if self._mapping is None or self._file is None:
                raise RuntimeError("SSD-backed Qwen4 PLE reader is closed")
            entry = self._header[key]
            start, end = entry["data_offsets"]
            row_bytes = (end - start) // entry["shape"][0]
            offsets = self._data_start + start + np.asarray(rows, np.int64) * entry.get("row_stride", row_bytes)
            pages = offsets // runtime._PLE_PAGE_SIZE * runtime._PLE_PAGE_SIZE
            for page, row_end in zip(pages.tolist(), (offsets + row_bytes).tolist()):
                self._mapping.madvise(mmap.MADV_WILLNEED, page, row_end - page)

    def _prefetch_missing_pages(self, row_indices, base_offset, row_bytes, *, row_stride=None) -> bool:
        """Prefetch unmarked pages; return whether all were already marked."""
        self._require_owner()
        offsets = base_offset + row_indices * (row_bytes if row_stride is None else row_stride)
        needed_pages = np.unique(
            np.concatenate(
                (offsets // runtime._PLE_PAGE_SIZE, (offsets + row_bytes - 1) // runtime._PLE_PAGE_SIZE)
            )
        )
        seen = np.frombuffer(self._seen_pages, dtype=np.uint8)
        fresh = needed_pages[seen[needed_pages] == 0]
        if fresh.size == 0:
            return True
        fd = self._file.fileno()

        def touch(pages) -> None:
            for page in pages:
                offset = int(page) * runtime._PLE_PAGE_SIZE
                remaining = runtime._PLE_PAGE_SIZE
                while remaining > 0:
                    chunk = os.pread(
                        fd, remaining, offset + (runtime._PLE_PAGE_SIZE - remaining)
                    )
                    if not chunk:
                        break
                    remaining -= len(chunk)

        # #3602 (Matt Barnson) uses submit/wait to drain failed page batches.
        # Also drain partial submission here before releasing the reader mutex;
        # otherwise close could recycle an FD still used by another page read.
        futures = []
        try:
            # Preserve page-read parallelism without one Future per page.
            task_count = min(fresh.size, runtime._PLE_IO_WORKERS)
            for index in range(task_count):
                futures.append(runtime._PLE_IO_POOL.submit(touch, fresh[index::task_count]))
            for future in futures:
                future.result()
        finally:
            wait(futures)
        seen[fresh] = 1
        return False

    def _rearm_if_slow(self, elapsed: float, row_count: int) -> None:
        """Allow another prefetch after a slow gather, at most once per interval."""
        budget = runtime._PLE_REARM_FLOOR_SECONDS + row_count * runtime._PLE_REARM_PER_ROW_SECONDS
        if elapsed < budget:
            return
        now = runtime.time.monotonic()
        if now - self._last_rearm < runtime._PLE_REARM_MIN_INTERVAL_SECONDS:
            return
        self._last_rearm = now
        self._seen_pages = bytearray(len(self._seen_pages))
        self._rearm_count += 1
        logger.info(
            "PLE: warm gather of %d rows took %.1f ms (memcpy budget %.1f ms); "
            "eviction suspected, re-armed seen-page bitmap for %s (#%d)",
            row_count,
            elapsed * 1e3,
            budget * 1e3,
            self.path.name,
            self._rearm_count,
        )

    def _close_resources(self):
        # Caller owns the module resource lock. On exported views leave the mmap
        # attached for a later retry, but release the file descriptor exactly once.
        try:
            if self._mapping is not None:
                self._mapping.close()
                self._mapping = None
        finally:
            if self._file is not None:
                self._file.close()
                self._file = None

    def close(self):
        if os.getpid() != self._owner_pid:
            with runtime._PLE_RESOURCE_LOCK:
                self._close_resources()
            return
        with self._resource_lock:
            with runtime._PLE_RESOURCE_LOCK:
                self._close_resources()

