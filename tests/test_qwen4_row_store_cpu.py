"""CPU-only coverage of the real PLE prefetch and external-layout path."""
import json
import tempfile
import struct
import unittest
from pathlib import Path

import numpy as np

from _ple_cpu_runtime import load_runtime


class HostArrays:
    bfloat16 = np.float32
    int64 = np.int64
    int32 = np.int32
    uint32 = np.uint32
    ones = staticmethod(np.ones)
    zeros = staticmethod(np.zeros)
    array = staticmethod(np.asarray)
    eval = staticmethod(lambda *args: None)

    @staticmethod
    def dequantize(weight, scales, biases, *, bits, group_size, mode):
        if (bits, group_size, mode) != (4, 32, "affine"):
            raise AssertionError("Unexpected quantization in this fixture")
        shifts = np.arange(8, dtype=np.uint32) * 4
        values = ((weight[..., None] >> shifts) & 15).reshape(weight.shape[0], -1)
        return values * np.repeat(scales, 32, axis=1) + np.repeat(biases, 32, axis=1)


class ExternalRowStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.ns = load_runtime()
        self.ns["mx"] = HostArrays
        self.rows, self.width = 1024, 32
        values = np.arange(self.rows * self.width, dtype=np.uint32).reshape(self.rows, self.width) % 16
        groups = values.reshape(self.rows, -1, 8)
        packed = np.sum(groups << (np.arange(8, dtype=np.uint32) * 4), axis=-1, dtype=np.uint32)
        scales = np.full((self.rows, 1), np.float32(0.5).view(np.uint32) >> 16, np.uint16)
        biases = np.full((self.rows, 1), np.float32(-2).view(np.uint32) >> 16, np.uint16)
        self.packed, self.scales, self.biases = packed, scales, biases
        self.expected = values * 0.5 - 2
        self.file = self.root / "ple.rows"
        with self.file.open("wb") as out:
            for i in range(self.rows):
                out.write(packed[i].tobytes() + scales[i].tobytes() + biases[i].tobytes())
        self.manifest = self.root / "ple-store.json"
        self.manifest.write_text(json.dumps({
            "version": 2, "source_root": ".", "layout": "interleaved_q4",
            "data_file": self.file.name, "row_count": self.rows,
            "row_width": self.width, "row_stride": 20,
            "quantization": {"bits": 4, "group_size": 32, "mode": "affine"},
        }))
        self.embedding = None

    def tearDown(self):
        if self.embedding is not None:
            self.embedding.close()
        self.ns["_PLE_IO_POOL"].shutdown(wait=True)
        self.temp.cleanup()

    def open(self, **overrides):
        self.embedding = self.ns["DiskBackedShardedEmbedding"](
            self.root, "model.layers.1.ple.ngram_embedding", self.rows,
            overrides.get("width", self.width), 4,
            storage={"manifest": str(self.manifest)},
        )
        return self.embedding

    def test_prefetch_is_consumed_and_matches_direct_lookup_with_duplicate_rows(self):
        e = self.open()
        ids = np.array([[819, 0, 819, 820, 1023, 2]], dtype=np.int64)
        e.prefetch(ids)
        actual = e(ids)
        self.assertTrue(e.last_prefetch_hit)
        np.testing.assert_array_equal(actual, self.expected[ids])
        np.testing.assert_array_equal(e(ids), actual)
        self.assertFalse(e.last_prefetch_hit)
        e.close()
        np.testing.assert_array_equal(actual, self.expected[ids])
        with self.assertRaisesRegex(RuntimeError, "closed"):
            e(ids)

    def test_prefetch_for_other_tokens_cannot_change_the_result(self):
        e = self.open()
        first = np.array([[1, 2, 1]])
        other = np.array([[900, 901, 900]])
        e.prefetch(first)
        np.testing.assert_array_equal(e(other), self.expected[other])
        self.assertFalse(e.last_prefetch_hit)
        np.testing.assert_array_equal(e(first), self.expected[first])
        self.assertTrue(e.last_prefetch_hit)

    def test_invalid_geometry_and_truncated_store_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "row width"):
            self.open(width=64)
        with self.file.open("r+b") as out:
            out.truncate(self.file.stat().st_size - 1)
        with self.assertRaisesRegex(ValueError, "size"):
            self.open()

    def test_manifest_ranges_share_the_same_prefetch_path(self):
        shards = []
        for index in range(2):
            start, count = index * 512, 512
            tensors = {"weight": (self.packed[start:start + count], "U32"),
                       "scales": (self.scales[start:start + count], "BF16"),
                       "biases": (self.biases[start:start + count], "BF16")}
            header, offset = {}, 0
            for name, (array, dtype) in tensors.items():
                header[name] = {"shape": list(array.shape), "dtype": dtype,
                                "data_offsets": [offset, offset + array.nbytes]}
                offset += array.nbytes
            raw = json.dumps(header).encode()
            path = self.root / f"source-{index}.safetensors"
            path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"".join(a.tobytes() for a, _ in tensors.values()))
            shard = {"row_start": start, "row_count": count}
            for name, meta in header.items():
                shard[name] = {"file": path.name, "dtype": meta["dtype"],
                               "shape": meta["shape"], "offset": 8 + len(raw) + meta["data_offsets"][0]}
            shards.append(shard)
        manifest = json.loads(self.manifest.read_text())
        manifest.update(layout="safetensors_ranges", shards=shards)
        self.manifest.write_text(json.dumps(manifest))
        e = self.open()
        ids = np.array([[900, 2, 900, 511, 512]])
        e.prefetch(ids)
        np.testing.assert_array_equal(e(ids), self.expected[ids])
        self.assertTrue(e.last_prefetch_hit)
        self.assertEqual(e.last_touched_shards, (0, 1))

    def test_nvfp4_prefetch_preserves_packed_weights_and_u8_scales(self):
        scales = np.arange(self.rows * 2, dtype=np.uint8).reshape(self.rows, 2)
        self.file.write_bytes(b"".join(self.packed[i].tobytes() + scales[i].tobytes() for i in range(self.rows)))
        manifest = json.loads(self.manifest.read_text())
        manifest.update(row_stride=18, quantization={"bits": 4, "group_size": 16, "mode": "nvfp4"})
        self.manifest.write_text(json.dumps(manifest))
        e = self.open()
        ids = np.array([[819, 2, 819]])
        e.prefetch(ids)
        plan, future = e._pending[ids.astype(np.int64).tobytes()]
        buffers = future.result(timeout=3)
        np.testing.assert_array_equal(buffers[0], self.packed[ids.reshape(-1)])
        np.testing.assert_array_equal(buffers[1], scales[ids.reshape(-1)])
        self.assertEqual(plan[4], [0, 1])
        self.assertEqual(plan[-2:], (4, 16))


if __name__ == "__main__":
    unittest.main()
