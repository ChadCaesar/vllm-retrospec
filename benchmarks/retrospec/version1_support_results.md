# `version1` cluster-store support type split

Baseline commit: `1372692b7`. The candidate is the uncommitted working tree.
The baseline was run from an isolated `git archive` at
`/tmp/version1_support_baseline_1372692b7`, with the same local compiled
extensions and FlashAttention Python sources. A second snapshot containing
the candidate's three changed backend modules was used to compare both
versions under the same `/tmp` source layout. No commit or push was made.

## Scope and compatibility

The 1,205-line `offload/cluster_store_support.py` was separated into
`cluster_staging_types.py` (449 lines), `cluster_verification_types.py` (398
lines), and a 452-line support and prefetch-state module. The support module
re-exports all moved types. Page-store, prefetch, verification, kernel, CUDA
stream, and CUDA graph execution methods remain at their original call sites.
No capacity, memory-budget, or configuration rule changed.

Parsed ASTs of all 32 original classes and three type/constant assignments
match `1372692b7`. Every original class remains identical through the legacy
`retrospec.cluster_store` and `offload.cluster_store_support` imports and
round-trips through class pickling with its original qualified path.

## Tests

| Check | Result |
| --- | --- |
| Page-store and compatibility selection | 142 passed |
| Full RetroSpec suite, conda `vllm` with GPU access | 883 passed in 205.43 s |
| Affected vLLM scheduler, KV cache, and GPU runner selection | 162 passed, 1 skipped in 172.18 s |
| Original support-type annotation resolution | Passed for all annotated classes |
| Ruff check, Ruff format check, AST comparison, and `git diff --check` | Passed |

The compatibility suite gained 19 cases for the moved support types. The
full-suite count increased from the baseline's 864 by those 19 cases.

## End-to-end performance

Both sides used the two RTX 4090 GPUs, conda `vllm`, one warmup, greedy
decoding, the same local model and NIAH sample, and the same model and
RetroSpec settings per row. Qwen 30k used 31,181 input tokens and 64 fixed
generated tokens. Llama 120k used the **full 119,957-token input** and 128
fixed generated tokens. Qwen 30k used `max_model_len=32768`, cache ratio 0,
GPU index budget 4 GiB, and GPU memory utilization 0.94; Llama used
`max_model_len=120256`, cache ratio 0.08, index budget 2 GiB, and GPU memory
utilization 0.94. Both used 2,048 maximum batched tokens. The short single
request had 2,101 input tokens; the two-request PP case had 2,207 tokens per
request and used a 1,024-token batch limit.

Median measured request wall time in seconds:

| Workload | Baseline | Candidate | Repeats per side | Notes |
| --- | ---: | ---: | ---: | --- |
| Qwen 30k TP eager | 20.21 | 19.42 | 3 | All six output hashes match. |
| Qwen 30k TP graph | 18.70 | 18.50 | 6 | Retest after variable initial output/timing mix. |
| Qwen 30k PP eager | 14.85 | 14.59 | 3 | All requests found the needle. |
| Qwen 30k PP graph | 14.04 | 13.59 | 6 | Uses reverse baseline; earlier baseline median was 12.19. |
| Qwen short single graph | 2.21 | 1.98 | 2 | Exact generated-token hashes match. |
| Qwen short two-request PP graph | 7.14 | 7.20 | 8 | All requests found the needle. |
| Llama 120k TP eager | 66.12 | 64.53 | 2 | Exact generated-token hashes match. |
| Llama 120k TP graph | 63.16 | 63.11 | 2 | Reverse pair; candidate in the same `/tmp` layout: 62.30. |
| Llama 120k PP eager | 85.10 | 85.11 | 2 | Exact generated-token hashes match. |
| Llama 120k PP graph | 87.30 | 87.06 | 2 | Reverse pair; candidate in the same `/tmp` layout: 87.16. |

All Llama repeats produced the same generated-token SHA-256 and found the
needle. Qwen 30k output hashes varied on both versions, including within a
single process, so its aggregate wall times must be read with the output mix.
Its TP graph same-hash `fc1bfef3` samples had overlapping times: 18.54–21.32
seconds on the baseline and 18.36–20.63 seconds on the candidate. The short
single-request prefix does not contain the needle; its exact generated
hashes are the output check. In the eight-repeat short multi-request run,
seven baseline batches and all eight candidate batches shared the same first
request hash. The remaining baseline batch still found the needle.

The first Llama TP and PP graph comparisons showed 3–7% differences in
decode-only medians, although total time did not regress. Reverse runs
narrowed the PP difference. Under the same `/tmp` source layout, Llama TP
graph decode medians were 8.83 s baseline versus 8.76 s candidate; PP graph
was 9.75 versus 9.87 s. The initial worktree decode gap did not reproduce
under that layout. The short two-request case also had a 7.83 s reverse
baseline versus 7.78 s candidate snapshot; its later eight-repeat direct
comparison was 7.14 versus 7.20 s. This workload has broad per-repeat
first-token and decode variance, so small phase differences are not
attributed to the split.

Device-wide peak VRAM sampling at 0.5-second intervals showed no consistent
increase. For Qwen TP graph the maxima were 13,129/12,738 MiB on both sides;
for Qwen PP graph they were 15,681/15,322 versus 15,679/15,326 MiB; for the
eight-repeat short multi-request case they were 22,171/21,822 versus
22,171/21,824 MiB. Llama PP graph reverse baseline versus candidate snapshot
was 19,379/20,588 versus 19,379/20,532 MiB. Llama TP graph had larger
run-to-run GPU 0 variation on both versions, so an individual device-wide
peak is not treated as a program allocation change.

All 120 measured repeats, including initial comparisons, retests, snapshot
checks, prompt and output hashes, latency phases, and VRAM samples, are in
`version1_support_runs.tsv`. Raw run logs are under
`/tmp/version1_support_*.log`.

## Offload-stage check

A separate profiled Llama 120k TP graph pair used one warmup and one measured
request; those times are excluded from the latency table. On rank 0:

| Measured-request counter | Baseline | Candidate |
| --- | ---: | ---: |
| Prefill tokens / tile executions | 119,957 / 128 | 119,957 / 128 |
| Draft / sparse verification / verified tokens | 126 / 126 / 126 | 126 / 126 / 126 |
| Full verification requests / H2D bytes | 2 / 15,709,765,632 | 2 / 15,709,765,632 |
| Prefetch waves submitted | 9 | 9 |
| Verification miss H2D bytes | 851,042,304 | 851,902,464 |
| Resident hit rate | 0.937 | 0.937 |

Verification-miss bytes differed by about 0.1% as asynchronous residency
completed at slightly different times. Profiled rank-0 draft model CUDA
averages were 79.27 versus 78.07 ms per sampled token, sparse verification
CUDA averages were 195.03 versus 195.31 ms per launch, and CPU proposal
averages were 4,442 versus 4,384 ms per call. This check found no change in
the measured work or hot-stage cost. The full counters remain in
`/tmp/version1_support_llama120_tp_graph_{baseline,candidate}_profile.log`.
