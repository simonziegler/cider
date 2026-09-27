#!/usr/bin/env python3
"""Microbenchmark: n-gram (PLE) row gather, cider against oMLX's reader.

Runs under oMLX's own Python so the baseline is oMLX's real
``DiskBackedShardedEmbedding``. No model is loaded: the table is read straight
from the checkpoint, with ids hashed from real text by oMLX's own
``Qwen4ExpNGramEmbedding`` and the checkpoint's stored hash constants.

Each variant gets its own token windows, so a first touch is the first read
of those rows by any variant in this run (the page cache may still hold them
from earlier runs; the residency before and after is printed). ``repeat``
reads the same ids again.

Usage: <oMLX python> benchmarks/bench_ple_gather.py <model_dir> <text_file>
         [--lengths 1,16,128,512,2048,4096] [--only omlx,cider_mmap,...]
"""

from __future__ import annotations

import argparse
import ctypes
import json
import statistics
import struct
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("model", type=Path)
parser.add_argument("text", type=Path)
parser.add_argument("--lengths", default="1,16,128,512,2048,4096")
parser.add_argument("--only", default=None)
parser.add_argument("--start", type=int, default=300)
args = parser.parse_args()
MODEL = args.model.expanduser()

from omlx.patches.mlx_vlm_qwen4_exp_compat import apply_mlx_vlm_qwen4_exp_compat_patch  # noqa: E402

apply_mlx_vlm_qwen4_exp_compat_patch()
from mlx_vlm.models.qwen4_exp import language as L  # noqa: E402
from mlx_vlm.models.qwen4_exp.config import TextConfig  # noqa: E402

from cider.ple import CiderPLEEmbedding  # noqa: E402

config = json.loads((MODEL / "config.json").read_text())
text_config = TextConfig.from_dict(config["text_config"])
L.configure_ple_runtime(MODEL, mode="mmap")
layer_idx = text_config.ple_layer_ids[0] - 1
ngram = L.Qwen4ExpNGramEmbedding(text_config, text_config.ple_embed_dim, layer_idx, 0)
omlx_table = ngram.ngram_embedding
table_args = (
    MODEL,
    f"model.language_model.layers.{layer_idx}.ple.ple_embedding.ngram_embedding",
    omlx_table.shard_offsets[-1],
    omlx_table.dims,
    len(omlx_table.shard_sizes),
)


def small_tensor(key: str) -> np.ndarray:
    index = json.loads((MODEL / "model.safetensors.index.json").read_text())["weight_map"]
    with (MODEL / index[key]).open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        entry = json.loads(f.read(n))[key]
        start, end = entry["data_offsets"]
        f.seek(8 + n + start)
        raw = f.read(end - start)
    dtype = {"I64": np.int64, "I32": np.int32, "U32": np.uint32}[entry["dtype"]]
    return np.frombuffer(raw, dtype=dtype).reshape(entry["shape"])


prefix = f"language_model.model.layers.{layer_idx}.ple.ple_embedding."
for name in ("layer_multipliers", "ngram_heads_vocab_sizes", "ngram_heads_offsets"):
    stored = small_tensor(prefix + name)
    assert np.array_equal(stored.astype(np.int64), np.array(getattr(ngram, name)).astype(np.int64)), name
    setattr(ngram, name, mx.array(stored))


class Recorder:
    def __call__(self, ids):
        self.ids = ids
        return mx.zeros((*ids.shape, 1), dtype=mx.bfloat16)


def ngram_ids(tokens: np.ndarray) -> np.ndarray:
    recorder = Recorder()
    ngram.ngram_embedding = recorder
    ngram(mx.array(tokens[None], dtype=mx.int64), None)
    ngram.ngram_embedding = omlx_table
    return np.array(recorder.ids)[0]  # (T, 16)


from transformers import AutoTokenizer  # noqa: E402

tokenizer = AutoTokenizer.from_pretrained(str(MODEL))
all_tokens = np.array(tokenizer.encode(args.text.read_text()), dtype=np.int64)
print(f"tokens available: {len(all_tokens)}")

tables = {"omlx": omlx_table}
for backend in ("mmap", "pread", "pread_nocache"):
    tables[f"cider_{backend}"] = CiderPLEEmbedding(*table_args, backend=backend)


def variant(table):
    return lambda ids: table(mx.array(ids))


def variant_gpu_ids(table):
    # Ids arrive as the lazy output of a GPU op, as they do inside the model.
    return lambda ids: table(mx.array(ids) + 0)


variants = {name: variant(t) for name, t in tables.items()}
variants["cider_mmap_gpu_ids"] = variant_gpu_ids(tables["cider_mmap"])
if args.only:
    variants = {k: v for k, v in variants.items() if k in args.only.split(",")}

libc = ctypes.CDLL(None, use_errno=True)
PAGE = 16384
layout = tables["cider_mmap"].store.layout


def resident_fraction() -> float:
    """Share of the table's pages in the page cache (mincore)."""
    import mmap

    maps, resident, total = {}, 0, 0
    for shard, (files, offsets) in enumerate(zip(layout.tensor_file, layout.tensor_offset)):
        rows = layout.row_starts[shard + 1] - layout.row_starts[shard]
        for part, (fi, off) in enumerate(zip(files, offsets)):
            if fi not in maps:
                with open(layout.files[fi], "rb") as fh:
                    maps[fi] = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
            nbytes = rows * (layout.w_row_bytes if part == 0 else layout.sb_row_bytes)
            address = np.frombuffer(maps[fi], dtype=np.uint8, count=1, offset=off).ctypes.data
            start = address - (address % PAGE)
            length = address + nbytes - start
            pages = (length + PAGE - 1) // PAGE
            vec = (ctypes.c_ubyte * pages)()
            if libc.mincore(ctypes.c_void_p(start), ctypes.c_size_t(length), vec) != 0:
                return float("nan")
            resident += sum(1 for v in bytes(vec) if v & 1)
            total += pages
    return resident / total


print(f"PLE page-cache residency before run: {100 * resident_fraction():.1f}%")

check = ngram_ids(all_tokens[:257])
outs = {name: fn(check) for name, fn in variants.items()}
mx.eval(*outs.values())
for name in variants:
    if name != "omlx":
        same = mx.array_equal(outs[name], outs["omlx"]).item()
        print(f"{name} == omlx: {same}")
        assert same, name

lengths = [int(x) for x in args.lengths.split(",")]
reps = {1: 200, 16: 100, 128: 30, 512: 10, 2048: 3, 4096: 2, 8192: 2}
cursor = args.start
names = list(variants)
for T in lengths:
    windows = {}
    for name in names:
        windows[name] = []
        for _ in range(reps.get(T, 2)):
            if cursor + T > len(all_tokens):
                raise SystemExit(f"text too short at T={T}")
            windows[name].append(ngram_ids(all_tokens[cursor : cursor + T]))
            cursor += T
    shift = lengths.index(T) % len(names)
    for name in names[shift:] + names[:shift]:
        fn = variants[name]
        first, repeat = [], []
        for ids in windows[name]:
            t0 = time.perf_counter(); mx.eval(fn(ids)); first.append(time.perf_counter() - t0)
            t0 = time.perf_counter(); mx.eval(fn(ids)); repeat.append(time.perf_counter() - t0)
        print(
            f"T={T:5d} rows={16 * T:6d} {name:20s} first-touch median {1e3 * statistics.median(first):9.3f} ms"
            f"   repeat median {1e3 * statistics.median(repeat):9.3f} ms   (n={len(first)})",
            flush=True,
        )

print(f"PLE page-cache residency after run: {100 * resident_fraction():.1f}%")
