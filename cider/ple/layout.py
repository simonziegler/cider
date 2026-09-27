"""Where each n-gram (PLE) table row lives on disk.

Reads ``model.safetensors.index.json`` and the safetensors headers of the files
that hold the table (headers only, never tensor data), and resolves the shard
tensors under the same four key spellings oMLX's ``DiskBackedShardedEmbedding``
accepts. Nothing is written anywhere.

Only the affine layout is supported: packed ``U32`` codes with ``BF16`` scales
and biases, which is what both Qwen3.8-Flash-Next MLX builds ship. A dense or
FP8 table raises ``NotImplementedError`` so a caller can fall back.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

PARTS = ("weight", "scales", "biases")


@dataclass(frozen=True)
class PLELayout:
    files: tuple[str, ...]                 # absolute paths
    row_starts: tuple[int, ...]            # num_shards + 1 global row offsets
    tensor_file: tuple[tuple[int, int, int], ...]    # per shard: file index of w, s, b
    tensor_offset: tuple[tuple[int, int, int], ...]  # per shard: absolute byte offset of w, s, b
    w_row_bytes: int
    sb_row_bytes: int
    dims: int
    bits: int
    group_size: int


def shard_sizes(num_embeddings: int, num_shards: int) -> tuple[int, ...]:
    """oMLX's split: the first ``num_embeddings % num_shards`` shards get one extra row."""
    base, remainder = divmod(num_embeddings, num_shards)
    return tuple(base + (1 if i < remainder else 0) for i in range(num_shards))


def _read_header(path: Path) -> tuple[dict, int]:
    with path.open("rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(size)), 8 + size


def read_ple_layout(
    model_path: str | Path,
    prefix: str,
    num_embeddings: int,
    num_shards: int,
    dims: int,
) -> PLELayout:
    model_path = Path(model_path)
    weight_map = json.loads((model_path / "model.safetensors.index.json").read_text())["weight_map"]
    runtime_prefix = prefix
    if prefix.startswith("model.language_model."):
        runtime_prefix = "language_model.model." + prefix[len("model.language_model."):]

    sizes = shard_sizes(num_embeddings, num_shards)
    files: list[str] = []
    file_index: dict[str, int] = {}
    headers: dict[str, tuple[dict, int]] = {}
    tensor_file, tensor_offset = [], []
    w_row_bytes = sb_row_bytes = bits = group_size = None

    for shard, rows in enumerate(sizes):
        candidates = (
            f"{prefix}.shard_{shard}",
            f"{prefix}.shards.{shard}",
            f"{runtime_prefix}.shard_{shard}",
            f"{runtime_prefix}.shards.{shard}",
        )
        base = next((c for c in candidates if f"{c}.weight" in weight_map), None)
        if base is None:
            raise KeyError(f"PLE shard {shard} is absent; checked {', '.join(candidates)}")
        if f"{base}.scales" not in weight_map or f"{base}.biases" not in weight_map:
            raise NotImplementedError(f"{base}: only affine (weight, scales, biases) PLE tables are supported")

        entries = []
        for part in PARTS:
            key = f"{base}.{part}"
            filename = weight_map[key]
            if filename not in headers:
                headers[filename] = _read_header(model_path / filename)
                file_index[filename] = len(files)
                files.append(str((model_path / filename).resolve()))
            header, data_start = headers[filename]
            entry = header[key]
            start, end = entry["data_offsets"]
            entries.append((file_index[filename], data_start + start, entry, end - start))

        (_, _, w, w_bytes), (_, _, s, s_bytes), (_, _, b, b_bytes) = entries
        if w["dtype"] != "U32" or s["dtype"] != "BF16" or b["dtype"] != "BF16":
            raise NotImplementedError(f"{base}: need U32 codes and BF16 scales/biases, got {w['dtype']}/{s['dtype']}/{b['dtype']}")
        if len(w["shape"]) != 2 or w["shape"][0] != rows or s["shape"] != b["shape"] or s["shape"][0] != rows:
            raise ValueError(f"{base}: unexpected shapes {w['shape']}, {s['shape']}, {b['shape']} for {rows} rows")
        groups = s["shape"][1]
        if dims % groups:
            raise ValueError(f"{base}: cannot infer group size from dims={dims}, scales={s['shape']}")
        this_group = dims // groups
        this_bits = w["shape"][1] * 32 // dims
        if w["shape"][1] * 32 != dims * this_bits:
            raise ValueError(f"{base}: cannot infer bits from dims={dims}, weight={w['shape']}")
        if w_bytes != rows * w["shape"][1] * 4 or s_bytes != rows * groups * 2 or b_bytes != rows * groups * 2:
            raise ValueError(f"{base}: byte ranges do not match shapes")
        if bits is None:
            bits, group_size = this_bits, this_group
            w_row_bytes, sb_row_bytes = w["shape"][1] * 4, groups * 2
        elif (bits, group_size) != (this_bits, this_group):
            raise NotImplementedError(f"{base}: mixed bits/group size across shards")
        tensor_file.append(tuple(e[0] for e in entries))
        tensor_offset.append(tuple(e[1] for e in entries))

    starts = [0]
    for rows in sizes:
        starts.append(starts[-1] + rows)
    return PLELayout(
        files=tuple(files),
        row_starts=tuple(starts),
        tensor_file=tuple(tensor_file),
        tensor_offset=tuple(tensor_offset),
        w_row_bytes=w_row_bytes,
        sb_row_bytes=sb_row_bytes,
        dims=dims,
        bits=bits,
        group_size=group_size,
    )
