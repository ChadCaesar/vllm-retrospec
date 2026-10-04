# `version1` GPU index type split

Baseline commit: `78d2fe2d5`. This change is uncommitted for review.

## Final scope

The eleven data and allocator types at the start of
`offload/index_residency.py` moved to `index_residency_types.py`. The
`RetroSpecGPUIndexResidencyManager` and all 42 of its methods stay in their
original module and order. The original public import and serialized class
paths remain available. No kernel, stream, arena capacity, memory-budget,
configuration, or CUDA graph logic changed.

The manager file decreased from 1,568 to 1,351 lines. The new types module is
250 lines. Parsed ASTs of all 11 moved type classes and all 42 manager methods
match the committed baseline.

## Validation

- Public import and class-pickle checks: 24 passed.
- Index-residency and import-compatibility selection: `43 passed` during the
  exploratory split; all are included in the final full suite.
- Ruff check and format check, syntax compilation, and `git diff --check`
  passed.
- Final type-only RetroSpec suite: `864 passed` in 207.58 seconds.
- Final type-only vLLM scheduler, KV, and GPU runner selection: `162 passed`,
  `1 skipped` in 171.12 seconds.

## Performance method

The committed baseline runs from an isolated `git archive` snapshot with
the local extension binaries attached; source paths in output confirm which
revision ran. Measurements use conda `vllm`, the two RTX 4090 GPUs, one
warmup, and repeated requests. They compare generated tokens, first-token
latency, decode time, total request time, and device-wide peak VRAM. The
Llama NIAH 120k sample contains 119,957 input tokens.

## Final-scope performance

Values are medians of measured request time in seconds. All Llama runs used
128 generated tokens and had the same generated-token SHA-256. Qwen 30k used
31,181 input tokens and 64 generated tokens; every request found the needle,
but output hashes varied on both versions.

| Workload | Committed baseline | Final type-only split | Observation |
| --- | ---: | ---: | --- |
| Llama 120k TP graph, 2 repeats | 64.49 | 64.28 | First-token 55.50 / 55.12; decode 8.79 / 8.96. |
| Llama 120k PP graph, 2 repeats | 86.39; reverse 87.40 | 87.98 | Reverse baseline narrowed total difference to 0.7%; decode 10.03 / 10.22. |
| Qwen 30k TP graph, 6 repeats | 13.73, 14.61, 14.68 across processes | 14.65 | Within the repeated baseline range; generated hashes vary. |
| Qwen 30k PP graph, 6 repeats | 13.52 | 12.35 | No total-time regression. |
| Qwen 30k TP eager, 3 repeats | 18.61; reverse 20.65 | 19.96; reverse 20.86 | The reverse pair differs by 1.0%; generated hashes vary. |

The first PP decode and Qwen eager differences above 3% triggered reverse
baseline runs. In each case the same-day reverse baseline moved toward the
candidate. Peak device-wide VRAM did not show a repeatable increase: Llama TP
was 18,156 / 18,004 MiB and PP 20,644 / 20,590 MiB for initial baseline /
candidate; Qwen TP graph was 13,127 / 13,129 MiB. These are device-wide
`nvidia-smi` samples at 0.5-second intervals.

The two-request Qwen PP graph case used 2,211 tokens per request. In each of
two measured batches, both requests returned exact text `2387512` and stopped
normally. Batch wall times were 6.25 and 6.79 seconds versus 7.91 and 6.82
seconds for the earlier committed-code baseline; sampled peak VRAM was within
2 MiB.

## Offload-stage counters

A separate Llama 120k TP graph pair enabled coarse RetroSpec profiling. The
committed-code baseline is the profile run saved from the preceding batch,
whose candidate code became `78d2fe2d5`. Rank 0's measured request had the
following counters:

| Counter | Baseline | Final type-only split |
| --- | ---: | ---: |
| Prefill prompt tokens / tile executions | 119,957 / 128 | 119,957 / 128 |
| Verified tokens | 126 | 126 |
| Full verification requests / H2D bytes | 2 / 15,709,765,632 | 2 / 15,709,765,632 |
| Prefetch waves submitted | 7 | 7 |
| Draft tokens / expanded verification tokens | 133 / 1 | 126 / 0 |

The generated-token hash matched. The baseline's warmup request itself had
126 draft and zero expanded tokens, showing that this speculative stage count
varies within unchanged code. Resident hit counts likewise depend on
asynchronous prefetch completion. Profiled runs are excluded from the timing
comparison above because collection adds overhead. Raw logs are in
`/tmp/version1_index_*.log`; the saved stage baseline is
`/tmp/version1_structure_llama120k_tp_profile_direct_binding.log`.

## Scope decision

An exploratory split also moved arena and pinned-summary methods. It passed
the full RetroSpec suite, and Llama 120k TP/PP graph matched the baseline
output and total-time range. Its Qwen 30k TP graph runs had higher aggregate
medians: 17.47, 20.19, and 17.60 seconds, compared with baseline medians
13.73, 14.61, and 14.68 seconds. Qwen output hashes varied on both sides,
and timings for the same output hash overlapped. Restoring arena methods in a
temporary ablation gave 15.88 seconds; restoring summary methods too gave
15.79 seconds, with mixed output hashes in both ablations. These results do
not identify a stable per-output cost regression, but they leave the
aggregate output mix uncertain. The final scope keeps every manager method
in its original location and moves only type definitions.
