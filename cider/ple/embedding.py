"""Drop-in replacement for oMLX's ``DiskBackedShardedEmbedding``.

Same constructor, same call contract, same parameters (``weight_scale`` only),
same ``close()``. The differences are all in how rows are read:

* the gather is a lazy MLX primitive on the CPU stream, so the forward pass
  never stops for a host sync (oMLX calls ``mx.eval`` and ``tolist`` here);
* ids are resolved to shards in C++, once per id, instead of a Python loop
  over every id for every touched shard;
* rows are read by a thread pool when a batch is large (prefill);
* one ``mx.dequantize`` covers the whole batch, instead of one per shard
  followed by a scatter-add into a zero tensor.

The values are bit-identical: the same stored bytes go through the same
``mx.dequantize`` and the same ``weight_scale`` multiply.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from .layout import read_ple_layout, shard_sizes
from .store import PLEStore


class CiderPLEEmbedding(nn.Module):
    def __init__(
        self,
        model_path: str | Path,
        prefix: str,
        num_embeddings: int,
        dims: int,
        num_shards: int,
        *,
        backend: str | None = None,
        threads: int | None = None,
        parallel_min_rows: int | None = None,
    ):
        super().__init__()
        self.shard_sizes = shard_sizes(num_embeddings, num_shards)
        offsets = [0]
        for size in self.shard_sizes:
            offsets.append(offsets[-1] + size)
        self.shard_offsets = tuple(offsets)
        self.dims = dims
        # The only parameter, as in oMLX: sanitize() supplies it (ones) when the
        # checkpoint has none, and strict weight loading expects exactly it.
        self.weight_scale = mx.ones((1,), dtype=mx.bfloat16)
        self.rows_read = 0
        layout = read_ple_layout(model_path, prefix, num_embeddings, num_shards, dims)
        self._bits = layout.bits
        self._group_size = layout.group_size
        self._store = PLEStore(layout, backend=backend, threads=threads, parallel_min_rows=parallel_min_rows)

    @property
    def store(self) -> PLEStore:
        return self._store

    def __call__(self, indices: mx.array) -> mx.array:
        shape = indices.shape
        codes, scales, biases = self._store.gather(indices)
        values = mx.dequantize(
            codes,
            scales,
            biases,
            group_size=self._group_size,
            bits=self._bits,
            mode="affine",
        )
        values = values.astype(mx.bfloat16) * self.weight_scale
        self.rows_read = codes.shape[0]
        return values.reshape(*shape, self.dims)

    def close(self):
        self._store.close()
