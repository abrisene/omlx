"""One prefetch and gather pipeline for disk-backed PLE tables."""
from __future__ import annotations

import json
import os
from bisect import bisect_right
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

import numpy as np
import mlx.core as mx
import mlx.nn as nn

from . import language as runtime
from .ple_mmap import _SafeTensorMMap
from .ple_layout import external_layout, shard_spec


class DiskBackedShardedEmbedding(nn.Module):
    """Dense or quantized PLE tables using one SSD prefetch and gather path."""

    def __init__(
        self,
        model_path: str | Path,
        prefix: str,
        num_embeddings: int,
        dims: int,
        num_shards: int,
        *,
        storage: dict | None = None,
    ):
        runtime._ple_require_owner()
        super().__init__()
        self._owner_pid = os.getpid()
        base, remainder = divmod(num_embeddings, num_shards)
        self.shard_sizes = tuple(
            base + (1 if index < remainder else 0) for index in range(num_shards)
        )
        external = external_layout(model_path, storage, prefix, num_embeddings, dims) if storage else None
        if external is not None:
            self.shard_sizes = external[0]
        offsets = [0]
        for size in self.shard_sizes:
            offsets.append(offsets[-1] + size)
        self.shard_offsets = tuple(offsets)
        self.dims = dims
        self.weight_scale = mx.ones((1,), dtype=mx.bfloat16)
        self._prefix = prefix
        self.rows_read = 0
        self.last_uploads = 0
        self.last_prefetch_hit = False
        self._external_storage = storage is not None
        self.prefill_first_chunk = (storage or {}).get(
            "prefill_first_chunk", "wide" if storage else "narrow"
        )
        self._prefetch_engagement_logged = False
        self._pending: dict[bytes, tuple] = {}
        self._prefetch_lock = Lock()
        self._prefetch_closed = False
        self._prefetch_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="ple-prefetch"
        )
        self.last_touched_shards: tuple[int, ...] = ()
        self._readers: dict[str, _SafeTensorMMap] = {}
        self._tensor_readers: dict[str, _SafeTensorMMap] = {}
        self._shard_specs: dict[
            int, tuple[str, str | None, str | None, int | None, int | None]
        ] = {}

        runtime.register_ple_resource(self, priority=1)
        model_path = Path(model_path)
        index_path = model_path / "model.safetensors.index.json"
        layouts = external[2] if external is not None else {}
        weight_map = external[1] if external is not None else json.loads(index_path.read_text()).get("weight_map", {})

        def register_reader(key: str) -> _SafeTensorMMap:
            filename = weight_map[key]
            reader = self._readers.get(filename)
            if reader is None:
                reader = _SafeTensorMMap(model_path / filename, layout=layouts.get(filename))
                self._readers[filename] = reader
            self._tensor_readers[key] = reader
            return reader

        runtime_prefix = prefix
        if runtime_prefix.startswith("model.language_model."):
            runtime_prefix = (
                "language_model.model." + runtime_prefix[len("model.language_model.") :]
            )
        for shard_index, shard_size in enumerate(self.shard_sizes):
            bases = (
                f"{prefix}.shard_{shard_index}",
                f"{prefix}.shards.{shard_index}",
                f"{runtime_prefix}.shard_{shard_index}",
                f"{runtime_prefix}.shards.{shard_index}",
            )
            base = next(
                (
                    candidate
                    for candidate in bases
                    if f"{candidate}.weight" in weight_map
                ),
                None,
            )
            if base is None:
                raise KeyError(
                    f"SSD-backed PLE shard {shard_index} is absent; "
                    f"checked {', '.join(bases)}"
                )

            self._shard_specs[shard_index] = shard_spec(
                base, shard_size, dims, weight_map, register_reader
            )

    def _plan(self, host: np.ndarray):
        """Shards, local rows and tensor families for a chunk; None when the touched shards differ."""
        offsets = np.asarray(self.shard_offsets, dtype=np.int64)
        shard = np.searchsorted(offsets, host, side="right") - 1
        local = host - offsets[shard]
        touched = [int(index) for index in np.unique(shard)]
        specs = [self._shard_specs[index] for index in touched]
        bits, group_size = specs[0][3], specs[0][4]
        families = [0] if bits is None else [0, 1] if specs[0][2] is None else [0, 1, 2]
        dtypes = {
            family: self._tensor_readers[specs[0][family]].tensor_dtype(specs[0][family])
            for family in families
        }
        if any(spec[3:] != (bits, group_size) for spec in specs) or any(
            self._tensor_readers[spec[family]].tensor_dtype(spec[family]) != dtypes[family]
            for spec in specs
            for family in families
        ):
            return None
        return shard, local, touched, specs, families, dtypes, bits, group_size

    def _assemble(self, host: np.ndarray, plan) -> dict[int, np.ndarray]:
        """Copy every family's rows into one host buffer in index order (runs off the main thread on prefetch)."""
        shard, local, touched, specs, families, _, _, _ = plan
        slices = []
        for shard_index, spec in zip(touched, specs):
            positions = np.flatnonzero(shard == shard_index)
            slices.append((positions, local[positions], spec))
        # Advise every small slice before copying any: a decode token's missing
        # pages are then read together, and the copies wait for one drive
        # latency instead of one per row.
        for _, rows, spec in slices:
            if rows.size <= runtime._PLE_POOL_PREFETCH_ROWS:
                for family in families:
                    key = spec[family]
                    self._tensor_readers[key].advise_rows(key, rows)
        buffers: dict[int, np.ndarray] = {}
        for positions, rows, spec in slices:
            for family in families:
                key = spec[family]
                copied, _ = self._tensor_readers[key].rows_np(key, rows)
                buffer = buffers.get(family)
                if buffer is None:
                    buffer = np.empty((host.size, copied.shape[1]), dtype=copied.dtype)
                    buffers[family] = buffer
                buffer[positions] = copied
        return buffers

    def _require_owner(self):
        runtime._ple_require_owner(self._owner_pid)

    def _host_indices(self, indices: mx.array) -> np.ndarray:
        self._require_owner()
        flat = indices.reshape(-1)
        mx.eval(flat)
        host = np.asarray(flat.astype(mx.int64)).reshape(-1)
        if host.size and (int(host.min()) < 0 or int(host.max()) >= self.shard_offsets[-1]):
            raise IndexError("embedding index is outside the sharded vocabulary")
        return host

    def prefetch(self, indices: mx.array) -> None:
        """Assemble a chunk's rows on the prefetch worker; a later call with the same indices consumes them."""
        host = self._host_indices(indices)
        if host.size == 0:
            return
        with self._prefetch_lock:
            if self._prefetch_closed:
                return
            plan = self._plan(host)
            if plan is None:
                return
            key = host.tobytes()
            if key in self._pending:
                return
            # Keep two upcoming chunks; obsolete queued reads need not run.
            while len(self._pending) >= 2:
                _, future = self._pending.pop(next(iter(self._pending)))
                future.cancel()
            self._pending[key] = (
                plan,
                self._prefetch_executor.submit(self._assemble, host, plan),
            )

    def __call__(self, indices: mx.array) -> mx.array:
        self._require_owner()
        with self._prefetch_lock:
            if self._prefetch_closed:
                raise RuntimeError("SSD-backed Qwen4 PLE embedding is closed")
            return self._call_owned(indices)

    def _call_owned(self, indices):
        shape = indices.shape
        host = self._host_indices(indices)
        if host.size == 0:
            return mx.zeros((*shape, self.dims), dtype=self.weight_scale.dtype)
        pending = self._pending.pop(host.tobytes(), None)
        if pending is not None:
            plan, buffers = pending[0], pending[1].result()
            self.last_prefetch_hit = True
            if self._external_storage and not self._prefetch_engagement_logged:
                self._prefetch_engagement_logged = True
                runtime.logger.info("External PLE row-store prefetch engaged for %s", self._prefix)
        else:
            self.last_prefetch_hit = False
            plan = self._plan(host)
            if plan is None:
                return self._gather_per_shard([int(index) for index in host], shape)
            buffers = self._assemble(host, plan)
        _, _, touched, _, families, dtypes, bits, group_size = plan
        self.last_touched_shards = tuple(touched)
        self.rows_read = int(host.size)
        arrays = [_SafeTensorMMap.to_mx(buffers[family], dtypes[family]) for family in families]
        self.last_uploads = len(arrays)
        values = arrays[0]
        if bits is not None:
            values = mx.dequantize(
                values,
                arrays[1],
                arrays[2] if len(arrays) == 3 else None,
                group_size=group_size,
                bits=bits,
                mode="affine" if len(arrays) == 3 else "nvfp4",
            )
        # Decode in the checkpoint's scale dtype. bf16 rows times an fp16
        # scale promote to fp32.
        values = values.astype(self.weight_scale.dtype) * self.weight_scale
        return values.reshape(*shape, self.dims)

    def _gather_per_shard(self, host_indices: list[int], shape) -> mx.array:
        """Fallback for tables whose touched shards differ in dtype or quantization."""
        self._require_owner()
        shard_indices = [
            bisect_right(self.shard_offsets, index) - 1 for index in host_indices
        ]
        touched = tuple(sorted(set(shard_indices)))
        self.last_touched_shards = touched
        self.rows_read = 0
        result = mx.zeros((len(host_indices), self.dims), dtype=self.weight_scale.dtype)
        for shard_index in touched:
            positions = [
                i
                for i, current_shard in enumerate(shard_indices)
                if current_shard == shard_index
            ]
            local = [
                host_indices[i] - self.shard_offsets[shard_index] for i in positions
            ]
            weight_key, scales_key, biases_key, bits, group_size = self._shard_specs[
                shard_index
            ]
            values = self._tensor_readers[weight_key].rows(weight_key, local)
            if bits is not None:
                assert scales_key is not None
                assert group_size is not None
                scales = self._tensor_readers[scales_key].rows(scales_key, local)
                biases = self._tensor_readers[biases_key].rows(biases_key, local) if biases_key else None
                values = mx.dequantize(
                    values,
                    scales,
                    biases,
                    group_size=group_size,
                    bits=bits,
                    mode="affine" if biases_key else "nvfp4",
                )
            values = values.astype(self.weight_scale.dtype) * self.weight_scale
            self.rows_read += len(local)
            result = result.at[mx.array(positions, dtype=mx.int32)].add(values)
        return result.reshape(*shape, self.dims)

    def close(self):
        if os.getpid() != self._owner_pid:
            # No inherited Future.cancel/result, executor shutdown, or instance
            # lock. The module lock was replaced in the child at fork.
            with runtime._PLE_RESOURCE_LOCK:
                self._prefetch_closed = True
                self._pending.clear()
                for reader in self._readers.values():
                    reader._close_resources()
                self._readers.clear()
                self._tensor_readers.clear()
                self._shard_specs.clear()
            return
        with self._prefetch_lock:
            if not self._prefetch_closed:
                self._prefetch_closed = True
                # Drain workers before closing their mmap views. Never hold the
                # module resource lock while waiting for a worker or row read.
                self._prefetch_executor.shutdown(wait=True, cancel_futures=True)
                self._pending.clear()
            for reader in self._readers.values():
                reader.close()
            self._readers.clear()
            self._tensor_readers.clear()
            self._shard_specs.clear()

    @property
    def _prefix(self):
        return self.__prefix

    @_prefix.setter
    def _prefix(self, value):
        self.__prefix = value

