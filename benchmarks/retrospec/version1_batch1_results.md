# `version1` batch 1: scoped migration review

Baseline: `version1` at `3ed99444398348eb8af17426494c7425f928b8c4`.
Hardware: two RTX 4090 GPUs. Environment: conda `vllm`.
Measurements: 2026-10-01 and 2026-10-02 UTC.
The worktree contains no commit from this batch.

## Retained diff

- `vllm/v1/worker/gpu_model_runner.py`: when a request re-enters a non-last PP
  stage, restore committed output IDs from the scheduler's `all_token_ids`.
  The previous incremental payload omits accepted speculative tokens in this
  case. The common path for a request still in the batch remains incremental.
- `vllm/config/vllm.py`: enable fused RMSNorm by default in non-PP RetroSpec
  configurations whose default custom-op selection does not already enable
  it. An explicit `custom_ops` choice, including `none` or `-rms_norm`, remains
  authoritative. PP graph retains its original default because enabling the
  op there slowed Llama 120k. Eager already selects `custom_ops=["all"]` in both
  the original and current code, so this change does not alter its op choice.
- Add regression/configuration tests and
  `benchmarks/retrospec/benchmark_repeated.py` for repeated, warm-start
  measurements, token hashes, latency, throughput inputs, and peak VRAM.

No CPU offload algorithm, KV retirement, memory budget, kernel argument, or
public import was changed. The PP residual candidate and adaptive draft
horizon are not in the worktree.

## Tests

| Check | Result |
| --- | --- |
| Baseline `tests/retrospec` | 828 passed |
| Final `tests/retrospec` with GPU access, including the real eager model-config test | 836 passed, 196.66 s |
| Final affected v1 worker/scheduler/KV suite (187 collected) | 186 passed, 1 skipped, 188.00 s |
| PP request re-entry regression test after the ordinary-path adjustment | 1 passed |
| New real eager model-config test | Passed; included in the final 836-test run |
| `ruff check`, `ruff format --check`, `git diff --check` | Passed |
| Public runner/config and standalone benchmark imports | Passed |

The request re-entry test failed on the original implementation, recovering
only the final incremental token, and passed with the retained fix and the
ordinary-path adjustment. The final RMSNorm guard is covered by the full
RetroSpec run.

## Measurement setup

Each case used one warmup and at least two measured repeats, greedy decoding,
CPU-offload RetroSpec, `--profile --profile-sample-interval 1000`, and the
same prompt/configuration before and after. Values below are medians of
measured repeats; peak VRAM is the maximum across measured repeats, GPU 0 /
GPU 1 in MiB. Decode throughput is `(generated tokens - 1) / decode seconds`.
The benchmark prints every generated ID, SHA-256 hash, answer check, TTFT,
decode time, total time, and VRAM sample to its raw log.
The dataset directory was
`/home/lzg/zyt/SpeculativeDecoding/RetroSpecDecoding/throughput_eval/test_data`.
Run `conda run -n vllm python benchmarks/retrospec/benchmark_repeated.py --help`
for the harness options; `--warmups` and `--repeats` control measurements.

- Qwen2.5-7B-Instruct: NIAH 30k sample tokenizes to 31,181 tokens;
  `max_model_len=32768`, 64 generated tokens. TP and PP use two GPUs in eager
  and CUDA graph modes. The short single-GPU case has 2,101 prompt tokens,
  `max_model_len=8192`, and 32 generated tokens. The two-request PP graph case
  has two 2,106-token prompts, 32 generated tokens per request,
  `gpu_memory_utilization=0.85`, `max_gpu_index_memory=2`, and
  `max_num_batched_tokens=1024`.
- Llama-3-8B-Instruct-Gradient-1048k: the exact NIAH 120k sample tokenizes to
  **119,957 tokens**, with no request suffix; `max_model_len=120256`, 128
  generated tokens, `cache_ratio=0.08`, and `max_gpu_index_memory=2`. TP and
  PP use two GPUs in eager and CUDA graph modes.

| Case | Total s, base → patch | TTFT s, base → patch | Decode s, base → patch | Decode tok/s, base → patch | Peak VRAM MiB, base → patch |
| --- | ---: | ---: | ---: | ---: | ---: |
| Llama 120k TP eager, retest | 74.27 → 74.05 | 64.23 → 63.62 | 9.83 → 10.23 | 12.92 → 12.41 | 17271/17644 → 17299/17758 |
| Llama 120k TP graph | 74.32 → 70.94 | 65.11 → 61.42 | 9.00 → 9.32 | 14.12 → 13.62 | 18051/18468 → 17823/18358 |
| Llama 120k PP eager | 108.14 → 103.04 | 97.38 → 92.96 | 10.55 → 9.87 | 12.04 → 12.87 | 19297/19920 → 19297/19920 |
| Llama 120k PP graph | 104.21 → 104.24 | 94.32 → 94.12 | 9.67 → 9.90 | 13.14 → 12.83 | 19377/20532 → 19379/20532 |
| Qwen 30k TP eager, retest | 21.87 → 20.13 | 11.11 → 11.81 | 12.37 → 8.85 | 5.09 → 7.12 | 12581/12190 → 12583/12192 |
| Qwen 30k TP graph | 24.22 → 18.92 | 13.44 → 8.80 | 10.68 → 10.04 | 5.90 → 6.28 | 13563/13172 → 13127/12736 |
| Qwen 30k PP eager, four repeats | 15.92 → 15.13 | 12.79 → 10.89 | 3.47 → 2.75 | 18.15 → 22.92 | 14809/14450 → 14807/14454 |
| Qwen 30k PP graph, four-repeat retest | 14.52 → 15.00 | 11.63 → 11.93 | 2.77 → 2.98 | 22.73 → 21.12 | 15681/15324 → 15681/15324 |
| Qwen short single GPU graph, six-repeat patch retest | 2.42 → 2.30 | 2.30 → 2.16 | 0.109 → 0.110 | 284.40 → 281.82 | 24058/4 → 24076/4 |
| Qwen short two-request PP graph | 20.29 → 5.76 | 1.74 → 1.83 | 17.22 → 3.09 | 1.80 → 10.02 | 22201/21792 → 22165/21812 |

For two requests, TTFT and decode are for the first returned request, while
total time covers both requests. Its baseline output was unstable between
repeats and the clipped prompt omits the answer needle; use this case for
latency/VRAM comparison, not exact answer matching. Qwen 30k always found
the answer, but generated token hashes vary even within original-version
repeats. All Llama 120k modes generated the same 128-token SHA-256 hash before
and after (`21c3ba5cda2c...`) and found the answer.

The Qwen PP graph row shows the second four-repeat retest. Another final-code
four-repeat run had a 15.88 s median against two original-version runs at
14.52 s. Thus its total-time difference ranged from **+3.3% to +9.4%**;
these results triggered the same-day follow-up below. Isolated re-entry-only
code had 14.93 s median with one 20.42 s outlier. The first two-repeat
short-context run showed 0.14 s
decode, but six-repeat default-fusion and explicit opt-out retests had
0.110 s and 0.126 s medians respectively; the first result did not repeat.
Qwen TP eager's four-repeat TTFT median was **+6.3%**, although total
time and decode time improved; individual baseline TTFTs span 7.55–12.56 s.
TP eager does not execute the PP request re-entry branch. A local model-config
test confirmed eager selects `custom_ops=["all"]`, which already enables
RMSNorm; the original config code selects the same default. The early Llama
TP eager decode medians of 9.83 and 11.81 s therefore cannot be attributed
to the new RMSNorm opt-in. A later run measured 9.84/10.61 s, overlapping
the earlier original-version range of 9.74–10.57 s. Total time was within
0.3% of the paired retest.

### Same-day follow-up after narrowing the migration scope

The user clarified that GPU-only changes need not be migrated. The retained
CPU-offload change now checks `all_token_ids` only when a request was absent
from the previous PP batch. An instrumented, isolated re-entry build logged
**zero re-entry branch hits** during a Qwen 30k PP graph warmup and two
repeats. The ordinary graph run therefore does not exercise the changed
token-restoration behavior. Its initial 3.3–9.4% total-time difference is
not evidence of a repeated regression in that branch.

Fresh same-day measurements used the same harness, warmup, parameters, and
hardware. The Qwen cases had four measured repeats per side; Llama had two.

| Case | Baseline total / TTFT / decode, median s | Current total / TTFT / decode, median s | Peak VRAM MiB, base → current |
| --- | ---: | ---: | ---: |
| Qwen 30k PP graph | 15.13 / 12.13 / 2.91 | 14.42 / 11.35 / 2.88 | 15681/15322 → 15681/15322 |
| Llama 120k TP graph | 74.22 / 64.37 / 9.64 | 74.05 / 64.08 / 9.77 | 18053/19380 → 17657/18212 |

Qwen's current four-repeat run included one 20.27 s outlier with a different
generated-token hash and 60 accepted tokens out of 134 drafted; normal runs
were around 62/99. The median was nevertheless below the same-day baseline.
Baseline Qwen runs also produced different hashes. Llama's 128 generated
tokens and hash matched across all same-day repeats, and its decode-time
difference was about 1.3%, below the 3% investigation threshold.

Representative offload stage medians from interval logs (ms per logged event;
these are diagnostic counters, not an additive wall-time decomposition):

| Case and stage | Baseline | Patch |
| --- | ---: | ---: |
| Llama PP graph, layer-major prefill wall | 39614 | 39670 |
| Llama PP graph, full-verify CPU gather | 66.56 | 67.45 |
| Llama PP graph, full-verify transaction wall | 1060.38 | 1113.17 |
| Llama PP eager, layer-major prefill wall | 38772 | 38496 |
| Llama PP eager, full-verify CPU gather | 69.26 | 66.19 |
| Qwen PP eager, layer-major prefill wall | 3851 | 3804 |
| Qwen PP eager, full-verify CPU gather | 7.43 | 7.21 |

## Scope and remaining work

1. The PP layer-major prefill residual variant is **not retained** in this
   narrowed migration. `version1` folds `hidden + residual` before the next
   layer, whereas the candidate kept them separately. That changes BF16
   rounding and draft acceptance. An isolated candidate raised Qwen 30k PP
   eager decode from 2.82–3.74 s to 8.18–9.53 s across four repeats, and
   average acceptance length fell from about 32 to 21. The current benchmark
   still found the NIAH answer; a separate numerical-equivalence study would
   be needed before claiming this variant is a correctness requirement.
2. The adaptive draft horizon is **not retained** in this narrowed migration.
   The exact Llama 120k sample
   accepted 126 of 126 drafted tokens in repeated runs, so an acceptance-rate
   reduction rule never activates and cannot demonstrate the required 120k
   benefit on this workload.
3. Qwen output hashes and acceptance length vary even on the original-version
   baseline. The graph comparison above does not show a repeatable regression
   of the retained ordinary PP path. Matching the NIAH answer is the supported
   functional check for Qwen; exact token equality was verified for Llama.

This document covers the narrowed, retained changes only. The user reviews
and commits each batch; no commit or push has been made, and batch 2 has not
started. The final full-suite log is
`/tmp/version1_oct2_scoped_retrospec_final_gpu_pytest.log`; the final v1 log
is `/tmp/version1_oct2_scoped_v1_affected_final_pytest.log`.
Raw same-machine logs are under `/tmp/`, including
`version1_baseline_*`, `version1_snapshot_*`, `batch1_retained_*`,
`batch1_final_*`, `reentry_only_*`, and `residual_only_*`.
