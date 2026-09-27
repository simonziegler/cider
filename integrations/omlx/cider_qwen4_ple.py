"""Read Qwen4-Exp's SSD-offloaded n-gram (PLE) table through cider.

Drop-in for a private oMLX fork, as ``omlx/patches/cider_qwen4_ple.py``.
oMLX's ``qwen4_ple_ssd_offload`` keeps the 32 GB n-gram table on SSD and reads
it with ``DiskBackedShardedEmbedding``, which syncs the host mid-forward and
loops in Python over every id for every touched shard. cider replaces that
one class with a C++ gather that runs as a lazy MLX primitive on the CPU
stream: same bytes, same ``mx.dequantize``, bit-identical output.

Call site, in ``omlx/utils/model_loading.py`` inside
``if for_vlm and model_type == "qwen4_exp":``, right after
``configure_qwen4_exp_runtime(...)``::

    if resolved == "mmap" and getattr(model_settings, "qwen4_ple_backend", "cider") == "cider":
        from ..patches.cider_qwen4_ple import apply_cider_qwen4_ple_patch
        apply_cider_qwen4_ple_patch()
    else:
        from ..patches.cider_qwen4_ple import remove_cider_qwen4_ple_patch
        remove_cider_qwen4_ple_patch()

(``configure_qwen4_exp_runtime`` returns ``resolved``; keep it.) The swap
only affects tables built after it, and only in mmap mode: resident mode
never constructs ``DiskBackedShardedEmbedding``.

Self-arming, like ``m5_gather_qmm.py``: each table's first lookup also runs
oMLX's class and compares exactly; on any difference that table falls back to
oMLX's class and logs why. The patch declines, leaving oMLX untouched, when
the cider extension is missing or was built against another mlx, or when the
vendored constructor's signature has drifted.
Kill switch: ``OMLX_CIDER_PLE=0``.
Backend: ``CIDER_PLE_BACKEND=mmap|pread|pread_nocache`` (default ``mmap``).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def apply_cider_qwen4_ple_patch() -> bool:
    try:
        from cider.ple.omlx import apply_cider_qwen4_ple_patch as apply
    except Exception as exc:  # cider not installed, or its extension fails to load
        logger.warning("cider PLE patch not applied: %s", exc)
        return False
    return apply()


def remove_cider_qwen4_ple_patch() -> bool:
    try:
        from cider.ple.omlx import remove_cider_qwen4_ple_patch as remove
    except Exception:
        return False
    return remove()
