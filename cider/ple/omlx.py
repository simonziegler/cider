"""Put cider's PLE gather under oMLX's Qwen4-Exp model, without forking either.

oMLX's vendored ``mlx_vlm.models.qwen4_exp.language`` builds its SSD-offloaded
n-gram table as ``DiskBackedShardedEmbedding(model_path, prefix, rows, dims,
shards)``, looking the name up in the module at construction time. Replacing
that one module attribute after ``configure_qwen4_exp_runtime(..., mode="mmap")``
and before the model is built swaps the table and nothing else: oMLX's
``sanitize()`` still drops the table tensors at load, and ``Model.close()``
still calls ``close()`` on unload.

Safety, in the style of oMLX's own patches (``m5_gather_qmm.py``):

* ``OMLX_CIDER_PLE=0`` disables it.
* It declines, and leaves oMLX untouched, when the extension will not load or
  the vendored constructor's signature has drifted.
* A table cider cannot read (dense or FP8) is built by oMLX's own class.
* Canary: the first lookup of every table is also run through oMLX's class
  and compared exactly. On any difference that table falls back to oMLX's
  class for the rest of its life, and the reason is logged.
  ``OMLX_CIDER_PLE_CANARY=0`` skips it.

In a private oMLX fork this module is what ``omlx/patches/cider_qwen4_ple.py``
imports; see the README section "N-gram (PLE) offload".
"""

from __future__ import annotations

import inspect
import logging
import os

import mlx.core as mx

from .embedding import CiderPLEEmbedding

logger = logging.getLogger(__name__)

_ORIGINAL = "_cider_original_DiskBackedShardedEmbedding"
_SIGNATURE = ("self", "model_path", "prefix", "num_embeddings", "dims", "num_shards")


def _language_module():
    import mlx_vlm.models.qwen4_exp.language as language

    return language


def original_disk_backed_class(language=None):
    """oMLX's own class, whether or not the patch is applied."""
    language = language or _language_module()
    return getattr(language, _ORIGINAL, language.DiskBackedShardedEmbedding)


def is_applied(language=None) -> bool:
    language = language or _language_module()
    return hasattr(language, _ORIGINAL)


class CanaryPLEEmbedding(CiderPLEEmbedding):
    """CiderPLEEmbedding that proves itself against oMLX's class on first use."""

    def __init__(self, reference_cls, model_path, prefix, num_embeddings, dims, num_shards):
        super().__init__(model_path, prefix, num_embeddings, dims, num_shards)
        # object.__setattr__ keeps these out of the module tree, so the
        # parameters stay exactly {weight_scale}.
        object.__setattr__(self, "_reference", (reference_cls, (model_path, prefix, num_embeddings, dims, num_shards)))
        object.__setattr__(self, "_fallback", None)
        object.__setattr__(self, "canary", "armed")

    def __call__(self, indices: mx.array) -> mx.array:
        if self._fallback is not None:
            return self._fallback(indices)
        ours = super().__call__(indices)
        if self.canary != "armed":
            return ours
        reference_cls, args = self._reference
        reference = reference_cls(*args)
        reference.weight_scale = self.weight_scale
        theirs = reference(indices)
        errors = self.store.stats()["errors"]
        if errors == 0 and ours.shape == theirs.shape and mx.array_equal(ours, theirs).item():
            reference.close()
            object.__setattr__(self, "canary", "passed")
            logger.info("cider PLE canary passed: %d rows equal oMLX's", self.rows_read)
            return ours
        diff = float(mx.max(mx.abs(ours.astype(mx.float32) - theirs.astype(mx.float32)))) if ours.shape == theirs.shape else float("nan")
        logger.error(
            "cider PLE canary FAILED (max |diff| %s, %d read errors); "
            "this table falls back to oMLX's DiskBackedShardedEmbedding",
            diff,
            errors,
        )
        object.__setattr__(self, "canary", "failed")
        object.__setattr__(self, "_fallback", reference)
        return theirs

    def close(self):
        if self._fallback is not None:
            self._fallback.close()
        super().close()


def apply_cider_qwen4_ple_patch(language=None, *, canary: bool | None = None) -> bool:
    """Swap oMLX's SSD-backed PLE table for cider's. Idempotent. True if active.

    ``canary`` defaults to on; ``OMLX_CIDER_PLE_CANARY=0`` turns it off.
    """
    if canary is None:
        canary = os.environ.get("OMLX_CIDER_PLE_CANARY", "1").strip().lower() not in ("0", "false", "no", "off")
    if os.environ.get("OMLX_CIDER_PLE", "1").strip().lower() in ("0", "false", "no", "off"):
        logger.info("cider PLE patch disabled by OMLX_CIDER_PLE")
        return False
    language = language or _language_module()
    if is_applied(language):
        return True
    try:
        from ..ops import _load_ext

        _load_ext().PLEStore  # noqa: B018 - raises if the extension predates it
    except Exception as exc:  # pragma: no cover - depends on the install
        logger.warning("cider PLE patch not applied: extension unavailable (%s)", exc)
        return False
    original = language.DiskBackedShardedEmbedding
    params = tuple(inspect.signature(original.__init__).parameters)
    if params != _SIGNATURE:
        logger.warning("cider PLE patch not applied: %s signature is %s, expected %s", original.__name__, params, _SIGNATURE)
        return False

    def build(model_path, prefix, num_embeddings, dims, num_shards):
        try:
            if canary:
                return CanaryPLEEmbedding(original, model_path, prefix, num_embeddings, dims, num_shards)
            return CiderPLEEmbedding(model_path, prefix, num_embeddings, dims, num_shards)
        except NotImplementedError as exc:
            logger.info("cider PLE: %s; using oMLX's reader for this table", exc)
            return original(model_path, prefix, num_embeddings, dims, num_shards)

    build.__name__ = build.__qualname__ = "DiskBackedShardedEmbedding"
    setattr(language, _ORIGINAL, original)
    language.DiskBackedShardedEmbedding = build
    logger.info("cider PLE patch applied to %s", language.__name__)
    return True


def remove_cider_qwen4_ple_patch(language=None) -> bool:
    language = language or _language_module()
    if not is_applied(language):
        return False
    language.DiskBackedShardedEmbedding = getattr(language, _ORIGINAL)
    delattr(language, _ORIGINAL)
    return True
