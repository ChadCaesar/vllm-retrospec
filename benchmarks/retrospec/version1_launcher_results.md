# `version1` resident launcher split

Baseline commit: `c5d7bec31`. The candidate is the uncommitted working tree.
The baseline ran from a `git archive` at
`/tmp/version1_launcher_baseline_c5d7bec31`, with the same local compiled
extensions and FlashAttention Python sources. A candidate snapshot under
`/tmp/version1_launcher_candidate_snapshot` was also used for a same-layout
Qwen comparison. No commit or push was made.

## Scope and compatibility

The 1,195-line `offload/resident_kernels.py` is now a 56-line compatibility
entry. Its nine launcher functions live in `resident_lookup_launchers.py`
(304 lines), `resident_draft_launchers.py` (592 lines), and
`resident_verification_launchers.py` (301 lines). The original module still
exports every launcher, the draft statistics constant, and its imported
Triton definitions. The parent `retrospec.resident_kernels` import remains
valid; every moved launcher resolves to the same function object through all
three paths.

All nine launcher function ASTs match `c5d7bec31`. Caller imports, kernel
bodies, launch arguments, compile-time constants, buffers, streams, graph
settings, cache budget, and configuration rules are unchanged.

## Tests

| Check | Result |
| --- | --- |
| Resident kernels, cache, execution, and import compatibility | 157 passed in 29.54 s |
| Full RetroSpec suite, conda `vllm` with GPU access | 902 passed in 212.03 s |
| Affected vLLM scheduler, KV cache, and GPU runner selection | 162 passed, 1 skipped in 169.33 s |
| Original and moved launcher ASTs; all nine moved import identities | Passed |
| Baseline and candidate `resident_kernels` export name sets | Identical |
| Ruff check, Ruff format check, and `git diff --check` | Passed |

The compatibility suite gained nine parametrized cases for the moved
launchers. The full-suite count increased from 893 by these nine cases.

## End-to-end performance

Both sides used two RTX 4090 GPUs, conda `vllm`, one warmup, greedy decoding,
identical local models, and identical NIAH samples. Qwen 30k used 31,181
input tokens and 64 fixed generated tokens. Llama 120k used the **full
119,957-token input** and 128 fixed generated tokens. Qwen 30k used
`max_model_len=32768`, cache ratio 0, GPU index budget 4 GiB, GPU memory
utilization 0.94, and a 2,048-token batch limit. Llama used
`max_model_len=120256`, cache ratio 0.08, GPU index budget 2 GiB, GPU memory
utilization 0.94, and the same batch limit. The short single request used
2,101 input tokens and `max_model_len=8192`; the two-request PP case used
2,207 per request, `max_model_len=8192`, a 2 GiB GPU index budget, GPU
memory utilization 0.85, and a 1,024-token batch limit. Each TP/PP mode
used both GPUs. Other shared RetroSpec settings were 64 speculative tokens,
draft length 1–8, retrieval ratio 0.018, estimation ratio 0.232, an
8,192-token index segment, and a 32,768-token prefill tile.

Median measured request wall time in seconds. Each cell shows
**baseline / candidate**. The reverse pair ran candidate first and baseline
second to check order and run-to-run variation.

| Workload | Initial pair | Reversed pair | Repeats per side |
| --- | ---: | ---: | ---: |
| Llama 120k TP graph | 65.61 / 66.25 | — | 2 |
| Llama 120k TP eager | 65.31 / 66.10 | 67.82 / 65.43 | 2 + 2 |
| Llama 120k PP graph | 86.88 / 87.12 | — | 2 |
| Llama 120k PP eager | 86.00 / 85.85 | 85.95 / 85.92 | 2 + 2 |
| Qwen 30k TP graph | 14.36 / 14.23 | 18.95 / 18.89 | 6 + 6 |
| Qwen 30k TP eager | 16.67 / 18.01 | 19.92 / 20.01 | 4 + 6 |
| Qwen 30k PP graph | 13.60 / 13.03 | 13.70 / 13.63 | 6 + 6 |
| Qwen 30k PP eager | 13.03 / 13.65 | 14.68 / 15.12 | 4 + 6 |
| Qwen short single graph | 2.07 / 2.04 | — | 6 |
| Qwen short two-request PP graph | 7.46 / 7.50 | 7.45 / 7.49 | 8 + 8 |

All Llama repeats found the needle and produced the same generated-token
SHA-256. The Llama graph pairs differ by at most 1.0% in total time. Initial
TP and PP eager decode medians differed by about 5%, but the reversed pairs
changed direction: TP eager decode was 11.18 s baseline versus 9.75 s
candidate, and PP eager was 10.38 versus 10.15 s. There is no repeated
eager decode regression attributable to this split. The profiled stage
comparison below gives the measured work behind one TP eager pair.

Every Qwen 30k repeat found the needle. Its generated-token hashes and
latency phases varied on both sides, including within one process. Qwen TP
eager had an 8% initial total-time gap, but its reversed pair was 19.92 s
baseline versus 20.01 s candidate. With both source trees under `/tmp`,
eight more repeats yielded 18.29 versus 18.40 s; for the shared
`fc1bfef3` output hash, the medians were 19.16 s across seven baseline
samples and 18.25 s across three candidate samples. The earlier slow
candidate samples did not persist. Qwen PP graph changed from a faster
candidate initial pair to a near-equal reversed pair. Qwen PP eager shared
the `2e680372` output hash in nine of ten baseline and all ten candidate
repeats; in the reversed pair, the same-hash medians were 14.86 versus
15.12 s (about 1.8%).

The short single-request prefix omits the needle, so its exact generated
hash across all repeats is the output check. All two-request repeats found
the needle. A few batches on both sides generated fewer first- or
second-request tokens, while most shared the same hashes. Its total-time
medians were within about 1% in both run orders. Short-request decode spans
can be around 0.1 s, making throughput percentages from a few milliseconds
of timing variation unstable.

Device-wide GPU memory was sampled every 0.5 s. Qwen TP eager maxima were
12,583/12,192 MiB on both sides in the initial pair; Qwen PP eager was
14,809/14,450 versus 14,809/14,452 MiB. Qwen graph and short workloads
differed by at most 10 MiB. Llama PP eager reached 19,297/19,922 MiB on
both sides in the reversed pair. Other Llama peaks varied between runs
without a consistent candidate increase. One Qwen TP eager reversed
baseline run had missing memory samples; its initial and same-layout
baseline runs did capture peaks. These figures include all GPU processes,
not only vLLM.

All 174 measured repeats, including retests, the same-layout check, and
profiled runs, are in `version1_launcher_runs.tsv`. It records prompt and
output hashes, first-token and decode times, per-request decode throughput,
total time, and GPU peak samples. Raw logs are under
`/tmp/version1_launcher_*.log`.

## Offload-stage check

A separate Llama 120k TP eager profile pair used one warmup and one measured
request per side; profiled times are excluded from the latency table. Rank-0
measured-request counters were:

| Counter or sampled stage | Baseline | Candidate |
| --- | ---: | ---: |
| Prefill tokens / tile executions | 119,957 / 128 | 119,957 / 128 |
| Draft / sparse verification / verified tokens | 126 / 126 / 126 | 133 / 133 / 126 |
| Full verification requests / H2D bytes | 2 / 15,709,765,632 | 2 / 15,709,765,632 |
| Prefetch waves submitted | 10 | 9 |
| Verification miss H2D bytes | 849,362,944 | 925,917,184 |
| Resident hit rate | 0.937 | 0.941 |
| Draft model CUDA average | 92.96 ms / 126 | 90.70 ms / 133 |
| Sparse verification CUDA average | 200.96 ms / 16 | 193.00 ms / 17 |

The two profiles took different speculative and prefetch routes, so their
stage averages are not a same-work microbenchmark. The different prefetch
timing and extra draft work also changed verification-miss transfer volume.
Both completed the same verified-token count and full-verification transfer,
with no higher sampled draft or sparse-verification CUDA average on the
candidate. The unprofiled reversed pairs above are the performance
comparison.
