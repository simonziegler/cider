"""SSD-backed n-gram (PLE) embedding for Qwen3.8-Flash-Next-style models.

    from cider.ple import CiderPLEEmbedding
    table = CiderPLEEmbedding(model_dir, prefix, rows, dims, shards)
    rows = table(ids)                      # lazy, bf16, (..., dims)

To run under oMLX's vendored qwen4_exp model, see ``cider.ple.omlx``.
"""

from .embedding import CiderPLEEmbedding
from .layout import PLELayout, read_ple_layout, shard_sizes
from .store import BACKENDS, PLEStore

__all__ = [
    "BACKENDS",
    "CiderPLEEmbedding",
    "PLELayout",
    "PLEStore",
    "read_ple_layout",
    "shard_sizes",
]
