#!/usr/bin/env python3
"""Model-level parity: cider's offloaded n-gram table against the resident one.

Runs Qwen3.8-Flash-Next (the REAP build fits in memory either way) through
oMLX's own vendored qwen4_exp model, twice, in two processes so each load has
the machine to itself:

1. ``resident``: oMLX's resident mode; the whole table is loaded as MLX
   weights. This is the reference.
2. ``offload``: oMLX's SSD mode with cider's class swap applied
   (``cider.ple.omlx``), exactly as an oMLX fork would run it. The same
   process then swaps each table for oMLX's own ``DiskBackedShardedEmbedding``
   and runs again, as the baseline the offload has to beat.

For each prompt it records the full next-token logits after prefill, then
greedy-decodes ``PARITY_TOKENS`` tokens, recording each step's token and its
top-k logits.

Tolerances. The two paths read the same stored bytes: the 4-bit affine codes
and the bf16 scales and biases are copied unchanged and dequantized by the
same ``mx.dequantize``, and every other weight and kernel is shared. So the
expected difference is zero, and the unit test proves the table rows are
bit-identical. The comparison still allows one bf16 unit in the last place at
the magnitude of the largest logit (``2**(floor(log2 max|logit|) - 7)``,
bf16 carrying 8 significant bits), because the resident path builds the
table output by a scatter-add into zeros and the offloaded path does not; if
MLX ever fused those differently, one rounding in bf16 is the most that could
move. Greedy tokens must match exactly; a split is accepted only where the
reference's top two logits are within that tolerance (a genuine tie), and the
steps after it are not compared.

Usage (oMLX's interpreter, with a cider build for it first on PYTHONPATH):
    CIDER_PLE_MODEL=~/.omlx/models/<org>/<model> python tests/test_ple_model_parity.py
    python tests/test_ple_model_parity.py capture resident|offload <out_dir>
    python tests/test_ple_model_parity.py compare <out_dir>

Environment: PARITY_TOKENS (default 256), PARITY_TOPK (default 8),
PARITY_DIR (default: a temporary directory), CIDER_PLE_BACKEND.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

TOKENS = int(os.environ.get("PARITY_TOKENS", 256))
TOPK = int(os.environ.get("PARITY_TOPK", 8))
HERE = Path(__file__).resolve()

_README_EXCERPT = (HERE.parent.parent / "README.md").read_text()[:6000]

# Short and long prompts, code and prose, and a multi-turn chat whose
# end-of-turn tokens exercise the n-gram history reset.
CONVERSATIONS = {
    "code": [{"role": "user", "content": "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring and three doctests."}],
    "prose": [{"role": "user", "content": "In three short paragraphs, explain why the sky is blue and why sunsets are red."}],
    "long_context": [{"role": "user", "content": "Summarize the following README in five bullet points.\n\n" + _README_EXCERPT}],
    "multi_turn": [
        {"role": "user", "content": "What is the capital of Australia?"},
        {"role": "assistant", "content": "The capital of Australia is Canberra."},
        {"role": "user", "content": "And what is its population, roughly? Then list three landmarks there."},
    ],
    "tool_style": [{"role": "user", "content": "List the steps to reverse a singly linked list in place, then give the time and space complexity."}],
}


# ── capture (runs inside one process per mode) ──────────────────────


def _load(model_dir: Path, mode: str):
    import mlx.core as mx
    from omlx.patches.mlx_vlm_qwen4_exp_compat import (
        apply_mlx_vlm_qwen4_exp_compat_patch,
        configure_qwen4_exp_runtime,
    )

    apply_mlx_vlm_qwen4_exp_compat_patch()
    resolved = configure_qwen4_exp_runtime(str(model_dir), mode="resident" if mode == "resident" else "mmap", mtp_enabled=False)
    if mode == "offload":
        from cider.ple.omlx import apply_cider_qwen4_ple_patch

        assert apply_cider_qwen4_ple_patch(), "cider PLE patch did not apply"
    from mlx_vlm.utils import load as vlm_load
    from omlx.engine.vlm import _force_qwen4_exp_sanitize_on_load
    from omlx.utils.model_loading import materialize_lazy_state

    t0 = time.perf_counter()
    with _force_qwen4_exp_sanitize_on_load(model_dir):
        model, processor = vlm_load(str(model_dir), lazy=True)
    materialize_lazy_state(model)
    mx.synchronize()
    load_s = time.perf_counter() - t0
    tokenizer = getattr(processor, "tokenizer", processor)
    return model, tokenizer, resolved, load_s


def _ple_layers(model):
    for layer in model.language_model.model.layers:
        ple = getattr(layer, "ple", None)
        if ple is not None:
            yield ple.ple_embedding


def _encode(tokenizer, messages) -> list[int]:
    ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
    if not isinstance(ids, list):  # transformers 5 returns a BatchEncoding
        ids = ids["input_ids"]
    return [int(i) for i in ids]


def _warmup(model, tokenizer):
    """One short forward: arms the canary and compiles, outside the timings."""
    import mlx.core as mx

    lm = model.language_model
    ids = _encode(tokenizer, [{"role": "user", "content": "Hello there."}])
    cache = lm.make_cache()
    logits = lm(mx.array([ids]), cache=cache).logits[0, -1]
    for _ in range(4):
        logits = lm(mx.array([[int(mx.argmax(logits).item())]]), cache=cache).logits[0, -1]
    mx.eval(logits)


def _run(model, tokenizer, label: str, out_dir: Path, save: bool = True) -> dict:
    import mlx.core as mx

    lm = model.language_model
    _warmup(model, tokenizer)
    summary = {}
    arrays = {}
    for name, messages in CONVERSATIONS.items():
        ids = _encode(tokenizer, messages)
        cache = lm.make_cache()
        t0 = time.perf_counter()
        logits = lm(mx.array([ids]), cache=cache).logits[0, -1].astype(mx.float32)
        mx.eval(logits)
        prefill_s = time.perf_counter() - t0
        arrays[f"{name}.prefill_logits"] = np.array(logits)
        tokens, top_ids, top_vals, step_s = [], [], [], []
        for _ in range(TOKENS):
            order = mx.argsort(-logits)[: TOPK + 1]
            top_ids.append(np.array(order))
            top_vals.append(np.array(logits[order]))
            token = int(order[0].item())
            tokens.append(token)
            t0 = time.perf_counter()
            logits = lm(mx.array([[token]]), cache=cache).logits[0, -1].astype(mx.float32)
            mx.eval(logits)
            step_s.append(time.perf_counter() - t0)
        arrays[f"{name}.tokens"] = np.array(tokens)
        arrays[f"{name}.top_ids"] = np.stack(top_ids)
        arrays[f"{name}.top_vals"] = np.stack(top_vals)
        summary[name] = {
            "prompt_tokens": len(ids),
            "prefill_s": prefill_s,
            "prefill_tok_s": len(ids) / prefill_s,
            "decode_median_ms": 1e3 * float(np.median(step_s)),
            "decode_tok_s": 1.0 / float(np.median(step_s)),
            "text": tokenizer.decode(tokens[:48]),
        }
        print(f"[{label}] {name}: {len(ids)} prompt tokens, prefill {prefill_s:.2f}s, decode {summary[name]['decode_median_ms']:.1f} ms/token", flush=True)
    if save:
        np.savez(out_dir / f"{label}.npz", **arrays)
    return summary


def capture(mode: str, out_dir: Path):
    import mlx.core as mx

    model_dir = Path(os.environ["CIDER_PLE_MODEL"]).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer, resolved, load_s = _load(model_dir, mode)
    report = {"mode": mode, "resolved_ple_mode": resolved, "load_s": load_s, "active_gb": mx.get_active_memory() / 1e9}
    tables = list(_ple_layers(model))
    report["tables"] = [type(t.ngram_embedding).__name__ for t in tables]
    print(f"[{mode}] loaded in {load_s:.1f}s, PLE mode {resolved}, tables {report['tables']}, active {report['active_gb']:.1f} GB", flush=True)
    if mode == "resident":
        report["runs"] = {"resident": _run(model, tokenizer, "resident", out_dir)}
    else:
        report["runs"] = {"cider": _run(model, tokenizer, "cider", out_dir)}
        report["canary"] = [getattr(t.ngram_embedding, "canary", None) for t in tables]
        report["cider_store_stats"] = [t.ngram_embedding.store.stats() for t in tables]
        # Baseline in the same process and memory: oMLX's own SSD reader,
        # swapped in and out of the same tables.
        from cider.ple.omlx import original_disk_backed_class

        original = original_disk_backed_class()
        ours = [t.ngram_embedding for t in tables]
        theirs = []
        for table, mine in zip(tables, ours):
            assert hasattr(mine, "_reference"), "offload table was not built by the cider patch"
            other = original(*mine._reference[1])
            other.weight_scale = mine.weight_scale
            theirs.append(other)

        def use(modules):
            for table, module in zip(tables, modules):
                table.ngram_embedding = module

        use(theirs)
        report["runs"]["omlx_mmap"] = _run(model, tokenizer, "omlx_mmap", out_dir)
        # The first pass of each backend read rows the page cache may not
        # have held, and cider went first. Second passes, both warm, give
        # the like-for-like timing.
        use(ours)
        report["runs"]["cider_warm"] = _run(model, tokenizer, "cider_warm", out_dir, save=False)
        use(theirs)
        report["runs"]["omlx_mmap_warm"] = _run(model, tokenizer, "omlx_mmap_warm", out_dir, save=False)
        use(ours)
    report["peak_gb"] = mx.get_peak_memory() / 1e9
    (out_dir / f"{mode}.json").write_text(json.dumps(report, indent=2))


# ── compare ─────────────────────────────────────────────────────────


def bf16_ulp(magnitude: float) -> float:
    return 2.0 ** (math.floor(math.log2(max(magnitude, 1e-30))) - 7)


def compare_runs(ref: dict, other: dict) -> dict:
    """Per-prompt comparison of two captured runs; returns findings."""
    results = {}
    for name in CONVERSATIONS:
        r_logits, o_logits = ref[f"{name}.prefill_logits"], other[f"{name}.prefill_logits"]
        tol = bf16_ulp(float(np.max(np.abs(r_logits))))
        prefill_diff = float(np.max(np.abs(r_logits - o_logits)))
        r_tok, o_tok = ref[f"{name}.tokens"], other[f"{name}.tokens"]
        split = next((i for i, (a, b) in enumerate(zip(r_tok, o_tok)) if a != b), None)
        compared = len(r_tok) if split is None else split
        r_ids, o_ids = ref[f"{name}.top_ids"][:compared], other[f"{name}.top_ids"][:compared]
        r_vals, o_vals = ref[f"{name}.top_vals"][:compared], other[f"{name}.top_vals"][:compared]
        topk_value_diff = float(np.max(np.abs(r_vals - o_vals))) if compared else 0.0
        topk_mismatch = []
        for step in range(compared):
            if set(r_ids[step, :TOPK]) != set(o_ids[step, :TOPK]):
                boundary_gap = float(r_vals[step, TOPK - 1] - r_vals[step, TOPK])
                topk_mismatch.append({"step": step, "boundary_gap": boundary_gap, "tie": boundary_gap <= tol})
        split_info = None
        if split is not None:
            gap = float(ref[f"{name}.top_vals"][split, 0] - ref[f"{name}.top_vals"][split, 1])
            split_info = {"step": split, "reference_top2_gap": gap, "tie": gap <= tol}
        results[name] = {
            "tolerance": tol,
            "prefill_logits_max_abs_diff": prefill_diff,
            "prefill_logits_bit_identical": bool(np.array_equal(r_logits, o_logits)),
            "tokens_compared": compared,
            "tokens_identical": split is None,
            "split": split_info,
            "topk_value_max_abs_diff": topk_value_diff,
            "topk_set_mismatches": topk_mismatch,
            "ok": prefill_diff <= tol
            and topk_value_diff <= tol
            and (split_info is None or split_info["tie"])
            and all(m["tie"] for m in topk_mismatch),
        }
    return results


def compare(out_dir: Path) -> dict:
    ref = np.load(out_dir / "resident.npz")
    verdict = {}
    for label in ("cider", "omlx_mmap"):
        path = out_dir / f"{label}.npz"
        if path.exists():
            verdict[label] = compare_runs(ref, np.load(path))
    (out_dir / "compare.json").write_text(json.dumps(verdict, indent=2))
    for label, results in verdict.items():
        for name, r in results.items():
            print(
                f"{label:10s} {name:13s} ok={r['ok']!s:5s} prefill max|diff|={r['prefill_logits_max_abs_diff']:.3g} "
                f"(tol {r['tolerance']:.3g}, bit-identical={r['prefill_logits_bit_identical']}) "
                f"tokens identical={r['tokens_identical']} ({r['tokens_compared']}/{TOKENS}) "
                f"top-{TOPK} max|diff|={r['topk_value_max_abs_diff']:.3g} set mismatches={len(r['topk_set_mismatches'])}"
            )
    return verdict


# ── unittest entry ──────────────────────────────────────────────────


@unittest.skipUnless(os.environ.get("CIDER_PLE_MODEL"), "set CIDER_PLE_MODEL to a qwen4_exp model directory")
class ModelParity(unittest.TestCase):
    def test_offloaded_matches_resident(self):
        try:
            import omlx  # noqa: F401
        except ImportError:
            self.skipTest("needs oMLX's interpreter (its vendored qwen4_exp model)")
        out_dir = Path(os.environ.get("PARITY_DIR") or tempfile.mkdtemp(prefix="cider-ple-parity-"))
        for mode in ("resident", "offload"):
            if not (out_dir / f"{mode}.json").exists():
                subprocess.run([sys.executable, str(HERE), "capture", mode, str(out_dir)], check=True)
        offload = json.loads((out_dir / "offload.json").read_text())
        self.assertEqual(offload["resolved_ple_mode"], "mmap")
        self.assertTrue(all(c == "passed" for c in offload["canary"]), offload["canary"])
        self.assertTrue(all(s["errors"] == 0 for s in offload["cider_store_stats"]))
        verdict = compare(out_dir)
        for name, r in verdict["cider"].items():
            with self.subTest(prompt=name):
                self.assertTrue(r["ok"], json.dumps(r, indent=2))


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "capture":
        capture(sys.argv[2], Path(sys.argv[3]))
    elif len(sys.argv) >= 2 and sys.argv[1] == "compare":
        compare(Path(sys.argv[2]))
    else:
        unittest.main(verbosity=2)
