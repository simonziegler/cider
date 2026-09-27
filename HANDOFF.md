# Handoff: n-gram (PLE) table offload for oMLX — paused 2026-09-27

**Problem.** Qwen 3.8 Flash (Qwen4-Exp) carries about 51 billion n-gram embedding parameters. Resident, they cost tens of
GB of unified memory. oMLX 0.6.4 already has an SSD mode (`qwen4_ple_ssd_offload`, its `DiskBackedShardedEmbedding`);
this branch builds cider's own SSD-backed gather as an MLX primitive, to beat that baseline and later carry into a
private oMLX fork (oMLX has no plugin system; its extensions are patches under `omlx/patches/`).

## Where it stands (branch `feat/ngram-offload`, pushed)

| Commit | What |
| :----- | :--- |
| `3e5e10d` | Build: nanobind pinned to MLX's version, Python 3.11 allowed, `tools/build_for_omlx.sh` builds against oMLX's interpreter |
| `012994f` | `cider/ple/`: the SSD-backed table gather as a lazy CPU-stream MLX primitive, with unit tests (rows bit-identical to resident) |
| `ec4e965` | WIP, untested end to end: `tests/test_ple_model_parity.py` (resident vs offload, full logits and 256 greedy tokens), `benchmarks/bench_ple_gather.py`, `integrations/omlx/cider_qwen4_ple.py` (the class swap an oMLX fork would apply) |

## Evidence already in hand (the docs MCP benchmark, 2026-09-27, oMLX's own offload)

| Build | Offload | Seconds/task (llms.txt / MCP) | Peak `model_memory_used` |
| :---- | :------ | :---------------------------- | -----------------------: |
| Flash pruned (REAP-288, 4-bit) | off | 100 / 67 | 81 GB |
| Flash pruned | on | 88 / 56 | 80 GB |
| Flash full (oQ4e) | off | 112 / 131 | 80 GB |
| Flash full | on | 138 / 111 | 80 GB |

**Read this before building anything else:** oMLX's offload changed *reported* peak memory by about 1 GB, and speed
followed thinking time, not offload. Either `model_memory_used` does not count what the offload saves (memory-mapped
pages), or the offload saves little. So the first job is a measurement, not more code.

## Next steps, in order

1. **Measure what offload actually saves.** For Flash pruned and full, resident vs oMLX offload vs cider offload:
   process RSS and `footprint`/`vmmap` physical footprint, wired memory, compressed memory, and page-ins during a fixed
   decode (not `model_memory_used`). If oMLX's offload already saves the tens of GB, cider must beat it on speed or
   tail latency; if it does not, find out why before continuing.
2. **Run the parity test** (`tests/test_ple_model_parity.py`, usage in its docstring) resident vs cider offload.
   Greedy tokens must match; logits within one bf16 ulp. This is what makes the primitive trustworthy.
3. **Run `benchmarks/bench_ple_gather.py`**: gather latency and throughput per token, cold vs warm page cache, against
   `DiskBackedShardedEmbedding`.
4. **Only if 1-3 show a win:** finish `integrations/omlx/cider_qwen4_ple.py` as an oMLX patch, then rerun the docs MCP
   benchmark's Flash rows (chameleon repo, `website/scripts/docs-mcp/pilot/`, `--task-set button`) for an end-to-end
   number.

**Expectation to test, not assume:** same accuracy (the tables are bit-identical), memory near oMLX's offload,
speed within a few percent of resident once the page cache is warm; a cold first token slower.

## Rules on this machine

- Never quit or restart the oMLX app (it asks for confirmation); unload models with `POST /v1/models/<id>/unload`.
- Never print or commit the oMLX API key (`auth.api_key` in `~/.omlx/settings.json`).
- Toggle oMLX's offload per model through its admin API (log in, then
  `PUT /admin/api/models/<id>/settings {"qwen4_ple_ssd_offload": true|false}`); oMLX forces it when a model would not fit.
- The Metal wired limit was raised with `sudo sysctl iogpu.wired_limit_mb=122880`; it resets on reboot.
- One model loaded at a time for memory measurements; nothing else heavy running.
