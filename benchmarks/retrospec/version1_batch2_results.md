# `version1` batch 2: runtime control-path refactor

Baseline: `d54115184` (`[RetroSpec] Restore accepted tokens on PP reentry and gate RMSNorm fusion.`).
Branch: `version1`. The user reviews and commits this batch; this worktree has no
batch-2 commit or push.

## Scope

The CPU-offload RetroSpec runtime remains active. This batch only relocates
Python methods and their type definitions. It does not change an attention or
index kernel, launch parameters, buffer allocation, CUDA streams or graphs,
memory budgets, KV retirement, or configuration defaults.

| Public class and original module | New internal responsibility | Moved methods |
| --- | --- | ---: |
| `Scheduler`, `vllm.v1.core.sched.scheduler` | `runtime/scheduler.py`: index admission, layer-prefill admission, PP batch state | 16 |
| `GPUModelRunner`, `vllm.v1.worker.gpu_model_runner` | `runtime/runner_prefill.py`: layer-major prefill; `runtime/runner_proposal.py`: PP proposal coordination | 10 |
| `RetroSpecProposer`, `vllm.v1.spec_decode.retrospec.proposer` | `runtime/proposer_draft.py`: draft model steps; `runtime/proposer_verification.py`: pair construction, sampling, verification and transitions | 23 |
| `RetroSpecSparseAttention`, `vllm.v1.spec_decode.retrospec.attention` | `runtime/attention_execution.py`: exact and estimation attention execution | 6 |

All 55 relocated method ASTs match the committed baseline with attributes
excluded. The original public classes and import paths remain available.
`RetroSpecAttentionMode`, the proposer result types, and the runner state types
retain their former qualified module paths for serialization. The original
`merge_attn_states` call sites remain in `attention.py`.

The unit tests that patch module-level helpers now patch the helpers in their
new implementation modules. The public `GPUModelRunner.propose_draft_token_ids`
entry still calls the PP group helper in its original module; its test patches
both the original entry and the new proposal coordinator.

## Correctness checks

| Check | Result |
| --- | --- |
| RetroSpec full suite with two RTX 4090 GPUs | 836 passed in 196.14 s |
| Focused tests for moved runner and verification helpers | 6 passed |
| v1 runner, scheduler, KV, and runner streaming selection | 180 passed, 1 skipped in final run |
| Expanded selection including streaming scheduler tests | 180 passed, 1 skipped; 8 streaming scheduler failures |
| Same eight streaming scheduler tests on the `d54115184` baseline snapshot | 8 failed with the same fake encoder-decoder model assertion |
| Moved-method AST comparison against `d54115184` | 55 identical |
| Public class import, qualified module path, and enum pickle check | Passed |
| `ruff check`, `ruff format --check`, `git diff --check` | Passed after final edits |

The eight streaming failures occur in `Scheduler.__init__` before any moved
method runs. Both baseline and refactor reject the test's mocked
encoder-decoder model at the multimodal budget assertion. They are outside
this batch's scope.

## Performance comparison

The baseline is a read-only `git archive` of
`d54115184` under `/tmp/version1_batch2_baseline`; both sides use the same
installed conda `vllm` environment and compiled extensions. All measurements
use two local RTX 4090 GPUs, one warmup and two measured repeats per mode,
greedy decoding, and the same NIAH sample. The Qwen 30k prompt is 31,181
tokens (`max_model_len=32768`, `max_tokens=64`); the exact Llama 120k prompt is
119,957 tokens (`max_model_len=120256`, `max_tokens=128`). The short Qwen input
is clipped to 8,000 characters (2,101–2,106 tokens; `max_model_len=8192`,
`max_tokens=32`). Raw logs and generated token hashes are in
`/tmp/version1_batch2_benchmarks/`.

All runs set `retrospec_cache_ratio` to 0 for Qwen and 0.08 for Llama.
The Qwen 30k modes use `max_gpu_index_memory=4`, GPU memory utilization
0.94, and `max_num_batched_tokens=2048`; the Llama 120k modes use 2, 0.94,
and 2048 respectively. The short single-GPU mode uses the Qwen 30k values.
The short two-request PP mode uses 2, 0.85, and 1024 respectively. The
repeated benchmark is `benchmarks/retrospec/benchmark_repeated.py`.

The following are **initial** paired medians; flagged differences were retested
below. Each arrow is baseline → refactor. Decode throughput
is the median of each repeat's `(generated tokens - 1) / decode seconds`.

| Case | Total s | TTFT s | Decode s | Decode tok/s | Peak VRAM GPU 0/1 MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen 30k TP eager | 22.90 → 21.72 (-5.2%) | 12.19 → 9.98 | 10.62 → 11.68 | 5.93 → 6.28 | 12583/12192 → unavailable |
| Qwen 30k TP graph | 18.41 → 20.21 (+9.8%) | 12.03 → 11.75 | 6.32 → 8.38 | 14.37 → 8.26 | 13127/12736 → 13127/12736 |
| Qwen 30k PP eager | 15.92 → 15.07 (-5.4%) | 12.78 → 12.08 | 3.07 → 2.92 | 20.52 → 21.73 | 14807/14452 → 14807/14452 |
| Qwen 30k PP graph | 14.90 → 14.43 (-3.1%) | 12.12 → 11.56 | 2.72 → 2.78 | 23.15 → 22.66 | 15679/15322 → 15679/15326 |
| Qwen short single graph | 2.24 → 2.30 (+2.8%) | 2.11 → 2.14 | 0.12 → 0.16 | 259.63 → 198.92 | 24076/4 → 24074/4 |
| Qwen short two-request PP graph | 6.17 → 6.32 (+2.5%) | 1.83 → 2.40 | 4.26 → 2.98 | 7.47 → 10.64 | 22165/21812 → 22165/21812 |
| Llama 120k TP eager | 72.16 → 75.01 (+4.0%) | 62.07 → 64.91 | 9.88 → 9.88 | 12.85 → 12.85 | 17485/17672 → 17245/17646 |
| Llama 120k TP graph | 71.06 → 71.54 (+0.7%) | 61.63 → 62.09 | 9.22 → 9.25 | 13.77 → 13.74 | 17741/18210 → 17797/18212 |
| Llama 120k PP eager | 102.44 → 102.80 (+0.3%) | 92.23 → 92.53 | 9.97 → 10.07 | 12.77 → 12.61 | 19297/19920 → 19297/19920 |
| Llama 120k PP graph | 104.94 → 105.17 (+0.2%) | 94.62 → 94.58 | 10.12 → 10.39 | 12.57 → 12.24 | 19379/20588 → 19377/20648 |

All Llama repeats generated the same 128 token IDs and found the answer.
Qwen 30k found the answer in every repeat, but its output hash sometimes
changed even within the baseline process. The short clipped prompts omit the
needle, so their answer checks are false on both sides; their token hashes
matched by mode. The Qwen TP eager refactor GPU monitor returned no VRAM
samples in its first process. The retest recorded 12,581/12,190 MiB for the
refactor and 12,583/12,192 MiB for the baseline.

The raw logs also contain generated token IDs, hashes, and RetroSpec offload
stage counters. `cpu_avg` and `cuda_avg` are interval summaries whose event
counts vary, so they are diagnostic and cannot be added to reconstruct wall
time or compared as identical samples. The initial Llama TP eager difference
is concentrated in TTFT/prefill; its decode time is unchanged. Qwen TP graph
repeats span 7.2 s on the baseline and 5.8 s on the refactor, so its median
alone is not conclusive.

For the two Llama graph modes, logged stage medians in milliseconds were:

| Stage | TP graph baseline → refactor | PP graph baseline → refactor |
| --- | ---: | ---: |
| Layer-major prefill wall | 46770 → 46989 | 39626 → 39743 |
| Full-verify CPU gather | 27.58 → 27.36 | 66.97 → 66.69 |
| Proposal wall | 6360 → 6189 | 5417 → 5739 |

These stage counters describe sampled work within the request. Their event
counts and overlap can vary, so the end-to-end timing remains the performance
gate; the raw logs retain the other offload counters for review.

### Retests

The Llama 120k TP eager retest reversed the process order (refactor before
baseline) and increased measured repeats to four after one warmup. All eight
measured requests again generated the same 128-token hash and found the answer.

| Metric | Refactor | Baseline | Difference |
| --- | ---: | ---: | ---: |
| Total time, median s | 74.63 | 74.97 | -0.4% |
| TTFT, median s | 64.31 | 64.60 | -0.4% |
| Decode, median s | 10.05 | 9.99 | +0.5% |
| Layer-major prefill wall, median logged ms | 48576 | 48657 | -0.2% |
| Full-verify CPU gather, median logged ms | 27.60 | 27.21 | +1.4% |

The initial +4.0% total-time result did not repeat. Individual refactor
totals rose from 72.58 to 75.99 s over four repeats, while the subsequent
baseline totals were 73.65–75.45 s. The retest distributions overlap, and
both TTFT and prefill stage timings are near equal. This does not support a
repeatable regression from the refactor.

The Qwen retests also reversed the process order and used four measured
repeats after warmup. Median total time in seconds:

| Case | Initial baseline → refactor | Four-repeat baseline → refactor | Further retest |
| --- | ---: | ---: | ---: |
| Qwen 30k TP eager | 22.90 → 21.72 | 18.33 → 18.73 | — |
| Qwen 30k TP graph | 18.41 → 20.21 | 18.34 → 14.64 | — |
| Qwen 30k PP eager | 15.92 → 15.07 | 17.17 → 16.00 | — |
| Qwen 30k PP graph | 14.90 → 14.43 | 15.18 → 15.97 | — |
| Qwen short two-request PP graph | 6.17 → 6.32 | 5.60 → 6.18 | 6.01 → 5.13 (six repeats) |
| Qwen short single graph | 2.24 → 2.30 | 2.19 → 2.24 (six repeats, profiled) | 2.22 → 2.23 (12 repeats, unprofiled) |

The Qwen TP graph and PP graph median ordering changed between runs. The
two-request PP case was slower in the four-repeat run and faster in the
additional six-repeat run. Its number of proposal rounds also changed across
the processes (baseline/refactor: 13/14 initially, 23/25 in the four-repeat
run, 35/33 in the six-repeat run). Output hashes within each short mode still
matched. The variable number of rounds and the large within-process ranges
make a single paired median unreliable for these short workloads.

The profiled short single-GPU case repeatedly showed a roughly 40 ms decode
difference in one run. To check the user-visible total without the profiler,
we ran six more repeats (2.168 → 2.165 s median total), then reversed process
order and ran twelve more (2.216 → 2.229 s; +0.6%). In the latter run, decode
medians were 0.114 → 0.136 s, but the distributions overlapped. A permutation
check of that decode median difference gave p≈0.41, and a bootstrap 95% interval
for the difference included zero. The generated token hash and peak VRAM were
the same on both sides.

For component isolation, a final six-repeat profiled run compared the full
baseline, the full refactor, and two baseline snapshots with only the runner
or proposer relocation removed. All four generated the same token hash and
used 24,076 MiB peak GPU memory. Median total/decode seconds were baseline
2.264/0.146, no-runner 2.205/0.116, no-proposer 2.257/0.132, and full
refactor 2.191/0.091. The full refactor was fastest in this run, reversing
the earlier profiled short decode result. Neither relocation produced a
repeatable slowdown when isolated.

Raw retest logs are in `/tmp/version1_batch2_retests/`,
`/tmp/version1_batch2_short_retest/`,
`/tmp/version1_batch2_short_noprofile/`,
`/tmp/version1_batch2_short_noprofile12/`, and
`/tmp/version1_batch2_isolate_short/`. No repeatable end-to-end regression
was identified in this matrix. The short decode interval is only about
0.1–0.15 s, so its throughput ratio is particularly sensitive to timing
noise. No kernel or CPU-offload behavior was changed in this batch.
