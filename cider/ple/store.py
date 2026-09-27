"""Python face of the C++ ``PLEStore`` (csrc/src/ple_store.cpp)."""

from __future__ import annotations

import os

import mlx.core as mx

from .layout import PLELayout

BACKENDS = ("mmap", "pread", "pread_nocache")

# Below this many rows a gather runs on the calling thread: at decode (16 rows)
# a thread hand-off costs more than the copy.
DEFAULT_PARALLEL_MIN_ROWS = 256
DEFAULT_THREADS = 16


def _ext():
    from ..ops import _load_ext

    return _load_ext()


class PLEStore:
    """Gathers raw table rows (codes, scales, biases) as lazy MLX arrays.

    backend
        ``mmap``: copy from a read-only mapping (parity with oMLX's reader).
        ``pread``: one pread per byte range; rows still land in the page cache.
        ``pread_nocache``: pread with ``F_NOCACHE``; the page cache is bypassed,
        so the table costs no memory beyond the rows in flight.
    """

    def __init__(
        self,
        layout: PLELayout,
        backend: str | None = None,
        threads: int | None = None,
        parallel_min_rows: int | None = None,
    ):
        backend = backend or os.environ.get("CIDER_PLE_BACKEND", "mmap")
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
        threads = threads or int(os.environ.get("CIDER_PLE_THREADS", DEFAULT_THREADS))
        if parallel_min_rows is None:
            parallel_min_rows = int(os.environ.get("CIDER_PLE_PARALLEL_MIN_ROWS", DEFAULT_PARALLEL_MIN_ROWS))
        self.layout = layout
        self.backend = backend
        self._store = _ext().PLEStore(
            list(layout.files),
            list(layout.row_starts),
            [list(f) for f in layout.tensor_file],
            [list(o) for o in layout.tensor_offset],
            layout.w_row_bytes,
            layout.sb_row_bytes,
            backend,
            threads,
            parallel_min_rows,
        )

    def gather(self, ids: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        """Rows for ``ids`` (any shape), flattened: (N, codes), (N, groups) x 2."""
        w, s, b = self._store.gather(ids)
        return w, s, b

    def stats(self) -> dict[str, int]:
        return dict(self._store.stats())

    def reset_stats(self):
        self._store.reset_stats()

    @property
    def closed(self) -> bool:
        return self._store.closed

    def close(self):
        self._store.close()
