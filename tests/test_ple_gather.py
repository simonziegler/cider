#!/usr/bin/env python3
"""Row-gather correctness for cider's SSD-backed n-gram (PLE) table.

The gather copies bytes, so every check here is exact: the packed 4-bit codes
and the bf16 scales and biases must equal a direct slice of the same tensors,
and the dequantized rows must equal ``mx.dequantize`` of that slice bit for bit.

Two fixtures:

* A synthetic checkpoint written here: 4 uneven shards over 2 files, the same
  key names and affine layout (4-bit, group size 32, 160 dims) as
  Qwen3.8-Flash-Next. Always runs.
* The real checkpoint, when ``CIDER_PLE_MODEL`` names a model directory. The
  direct slice there comes from MLX's own safetensors loader (``mx.load``),
  an implementation independent of cider's header parsing.

When oMLX's vendored qwen4_exp model is importable (run under oMLX's Python),
the embedding is also compared with oMLX's ``DiskBackedShardedEmbedding``.

Usage:
    python tests/test_ple_gather.py
    CIDER_PLE_MODEL=~/.omlx/models/<org>/<model> python tests/test_ple_gather.py
"""

from __future__ import annotations

import json
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path

try:  # an installed or PYTHONPATH-staged cider wins (e.g. the oMLX build)
    import cider  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
import numpy as np

from cider.ple import BACKENDS, CiderPLEEmbedding, PLEStore, read_ple_layout

DIMS = 160
BITS = 4
GROUP = 32
PREFIX = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"
STORED = "language_model.model.layers.1.ple.ple_embedding.ngram_embedding"


def _bf16_bits(values: np.ndarray) -> np.ndarray:
    """float32 -> bf16 bit patterns (truncation is fine for test data)."""
    return (values.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)


def _write_safetensors(path: Path, tensors: dict[str, tuple[str, np.ndarray]]):
    header, blobs, offset = {}, [], 0
    for name, (dtype, array) in tensors.items():
        raw = np.ascontiguousarray(array).tobytes()
        header[name] = {
            "dtype": dtype,
            "shape": list(array.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        blobs.append(raw)
        offset += len(raw)
    header["__metadata__"] = {"format": "mlx"}
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(encoded)))
        f.write(encoded)
        for blob in blobs:
            f.write(blob)


def make_checkpoint(root: Path, num_embeddings: int, num_shards: int, seed=0):
    """Write a sharded affine PLE table; return {shard: (w, s_bits, b_bits)}."""
    rng = np.random.default_rng(seed)
    base, rem = divmod(num_embeddings, num_shards)
    sizes = [base + (1 if i < rem else 0) for i in range(num_shards)]
    shards, files, weight_map = {}, [{}, {}], {}
    for i, rows in enumerate(sizes):
        w = rng.integers(0, 2**32, size=(rows, DIMS * BITS // 32), dtype=np.uint32)
        s = _bf16_bits(rng.uniform(0.001, 0.05, size=(rows, DIMS // GROUP)))
        b = _bf16_bits(rng.uniform(-0.4, 0.4, size=(rows, DIMS // GROUP)))
        shards[i] = (w, s, b)
        # Split the three tensors of shard 1 across both files, as real
        # checkpoints do at file boundaries.
        for part, dtype, array in (("weight", "U32", w), ("scales", "BF16", s), ("biases", "BF16", b)):
            f = (i + (part == "biases")) % 2 if i == 1 else i % 2
            name = f"{STORED}.shards.{i}.{part}"
            files[f][name] = (dtype, array)
            weight_map[name] = f"model-{f + 1:05d}-of-00002.safetensors"
    # A decoy tensor before the table so data offsets are not zero.
    files[0] = {"lm_head.weight": ("U32", np.arange(12, dtype=np.uint32).reshape(3, 4)), **files[0]}
    weight_map["lm_head.weight"] = "model-00001-of-00002.safetensors"
    for f, tensors in enumerate(files):
        _write_safetensors(root / f"model-{f + 1:05d}-of-00002.safetensors", tensors)
    (root / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
    offsets = np.cumsum([0] + sizes)
    return shards, offsets


def direct_rows(shards, offsets, ids):
    """Reference: the rows a direct slice of each shard's tensors yields."""
    shard = np.searchsorted(offsets, ids, side="right") - 1
    w = np.stack([shards[s][0][i - offsets[s]] for s, i in zip(shard, ids)])
    s_ = np.stack([shards[s][1][i - offsets[s]] for s, i in zip(shard, ids)])
    b = np.stack([shards[s][2][i - offsets[s]] for s, i in zip(shard, ids)])
    return w, s_, b


def bits_to_bf16(bits: np.ndarray) -> mx.array:
    return mx.array(bits).view(mx.bfloat16)


def assert_raw_equal(tc, got, want, label):
    w, s, b = got
    tc.assertEqual(w.dtype, mx.uint32, label)
    tc.assertEqual(s.dtype, mx.bfloat16, label)
    tc.assertEqual(b.dtype, mx.bfloat16, label)
    np.testing.assert_array_equal(np.array(w), want[0], err_msg=f"{label}: weight")
    np.testing.assert_array_equal(np.array(s.view(mx.uint16)), want[1], err_msg=f"{label}: scales")
    np.testing.assert_array_equal(np.array(b.view(mx.uint16)), want[2], err_msg=f"{label}: biases")


class SyntheticGather(unittest.TestCase):
    NUM_EMBEDDINGS = 1003  # 4 shards of 251, 251, 251, 250: uneven on purpose
    NUM_SHARDS = 4

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="cider-ple-")
        cls.root = Path(cls.tmp.name)
        cls.shards, cls.offsets = make_checkpoint(cls.root, cls.NUM_EMBEDDINGS, cls.NUM_SHARDS)
        rng = np.random.default_rng(1)
        edges = np.concatenate([cls.offsets[:-1], cls.offsets[1:] - 1])  # first and last row of every shard
        random = rng.integers(0, cls.NUM_EMBEDDINGS, size=3000)
        cls.ids = np.concatenate([edges, random, random[:50]]).astype(np.int64)  # duplicates too

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def store(self, **kw):
        layout = read_ple_layout(self.root, PREFIX, self.NUM_EMBEDDINGS, self.NUM_SHARDS, DIMS)
        return PLEStore(layout, **kw)

    def test_layout_matches_writer(self):
        layout = read_ple_layout(self.root, PREFIX, self.NUM_EMBEDDINGS, self.NUM_SHARDS, DIMS)
        self.assertEqual(list(layout.row_starts), list(self.offsets))
        self.assertEqual((layout.bits, layout.group_size), (BITS, GROUP))

    def test_raw_rows_equal_direct_slice_every_backend(self):
        want = direct_rows(self.shards, self.offsets, self.ids)
        for backend in BACKENDS:
            for threads, min_rows in ((1, 1 << 30), (8, 1)):
                label = f"{backend} threads={threads}"
                with self.subTest(label):
                    store = self.store(backend=backend, threads=threads, parallel_min_rows=min_rows)
                    got = store.gather(mx.array(self.ids))
                    mx.eval(*got)
                    assert_raw_equal(self, got, want, label)
                    self.assertEqual(store.stats()["errors"], 0)
                    store.close()

    def test_gather_is_lazy_and_accepts_device_computed_ids(self):
        # Ids computed on the GPU and never synced by the caller.
        store = self.store(backend="mmap")
        ids = mx.array(self.ids[:64]) * 1 + 0
        got = store.gather(ids.reshape(8, 8))
        self.assertEqual(store.stats()["calls"], 0)  # nothing ran yet
        mx.eval(*got)
        self.assertEqual(store.stats()["calls"], 1)
        want = direct_rows(self.shards, self.offsets, self.ids[:64])
        assert_raw_equal(self, got, want, "lazy")

    def test_out_of_range_ids_are_zeroed_and_counted(self):
        store = self.store(backend="mmap")
        w, s, b = store.gather(mx.array([0, self.NUM_EMBEDDINGS, -1], dtype=mx.int64))
        mx.eval(w, s, b)
        self.assertEqual(store.stats()["errors"], 2)
        self.assertFalse(np.any(np.array(w)[1:]))

    def test_embedding_equals_dequantized_direct_slice(self):
        ids = self.ids[:2048].reshape(128, 16)
        w, s, b = direct_rows(self.shards, self.offsets, ids.reshape(-1))
        want = mx.dequantize(mx.array(w), bits_to_bf16(s), bits_to_bf16(b), group_size=GROUP, bits=BITS)
        want = want.reshape(128, 16, DIMS)
        emb = CiderPLEEmbedding(self.root, PREFIX, self.NUM_EMBEDDINGS, DIMS, self.NUM_SHARDS)
        got = emb(mx.array(ids))
        self.assertEqual(got.dtype, mx.bfloat16)
        self.assertEqual(got.shape, (128, 16, DIMS))
        self.assertTrue(mx.array_equal(got, want).item())
        emb.close()

    def test_embedding_matches_omlx_disk_backed_embedding(self):
        omlx_cls = _omlx_disk_backed_class()
        if omlx_cls is None:
            self.skipTest("oMLX's vendored qwen4_exp is not importable here")
        ids = mx.array(self.ids[:3056].reshape(191, 16))
        ours = CiderPLEEmbedding(self.root, PREFIX, self.NUM_EMBEDDINGS, DIMS, self.NUM_SHARDS)
        theirs = omlx_cls(self.root, PREFIX, self.NUM_EMBEDDINGS, DIMS, self.NUM_SHARDS)
        self.assertTrue(mx.array_equal(ours(ids), theirs(ids)).item())
        self.assertEqual(ours.shard_offsets, theirs.shard_offsets)
        self.assertEqual(ours.shard_sizes, theirs.shard_sizes)
        self.assertEqual(set(ours.parameters()), set(theirs.parameters()))
        ours.close()
        theirs.close()


def _omlx_disk_backed_class():
    try:
        from omlx.patches.mlx_vlm_qwen4_exp_compat import apply_mlx_vlm_qwen4_exp_compat_patch

        apply_mlx_vlm_qwen4_exp_compat_patch()
        from mlx_vlm.models.qwen4_exp import language
    except Exception:
        return None
    from cider.ple.omlx import original_disk_backed_class

    return original_disk_backed_class(language)


@unittest.skipUnless(os.environ.get("CIDER_PLE_MODEL"), "set CIDER_PLE_MODEL to test a real checkpoint")
class RealCheckpointGather(unittest.TestCase):
    """Gathered rows equal ``mx.load`` slices of the real tensors."""

    SAMPLED_SHARDS = 6
    ROWS_PER_SHARD = 400

    @classmethod
    def setUpClass(cls):
        cls.model = Path(os.environ["CIDER_PLE_MODEL"]).expanduser()
        config = json.loads((cls.model / "config.json").read_text())["text_config"]
        heads = (config["ngram_size"] - 1) * config["heads_per_ngram"]
        cls.dims = config["ple_embed_dim"] // heads
        cls.num_shards = config["split_ngram_parts"]
        cls.layer = config["ple_layer_ids"][0] - 1
        cls.prefix = f"model.language_model.layers.{cls.layer}.ple.ple_embedding.ngram_embedding"
        index = json.loads((cls.model / "model.safetensors.index.json").read_text())["weight_map"]
        stored = cls.prefix.replace("model.language_model.", "language_model.model.")
        rows = []
        for i in range(cls.num_shards):
            key = f"{stored}.shards.{i}.weight"
            with (cls.model / index[key]).open("rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                rows.append(json.loads(f.read(n))[key]["shape"][0])
        cls.num_embeddings = sum(rows)
        cls.index, cls.stored = index, stored
        cls.offsets = np.cumsum([0] + rows)

    def test_raw_rows_equal_mx_load_slice(self):
        rng = np.random.default_rng(7)
        picks = sorted({0, self.num_shards - 1, *rng.choice(self.num_shards, self.SAMPLED_SHARDS - 2, replace=False).tolist()})
        layout = read_ple_layout(self.model, self.prefix, self.num_embeddings, self.num_shards, self.dims)
        for backend in BACKENDS:
            store = PLEStore(layout, backend=backend, threads=8, parallel_min_rows=64)
            for shard in picks:
                size = self.offsets[shard + 1] - self.offsets[shard]
                local = np.concatenate([[0, size - 1], rng.integers(0, size, self.ROWS_PER_SHARD)])
                with self.subTest(backend=backend, shard=shard):
                    want = []
                    for part in ("weight", "scales", "biases"):
                        key = f"{self.stored}.shards.{shard}.{part}"
                        tensor = mx.load(str(self.model / self.index[key]))[key]
                        want.append(np.array(tensor[mx.array(local)].view(mx.uint32 if part == "weight" else mx.uint16)))
                    got = store.gather(mx.array(local + self.offsets[shard], dtype=mx.int64))
                    mx.eval(*got)
                    assert_raw_equal(self, got, want, f"{backend} shard {shard}")
            self.assertEqual(store.stats()["errors"], 0)
            store.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
