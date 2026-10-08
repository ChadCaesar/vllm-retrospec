# `version1` cluster-handle lifecycle split

Baseline commit: `382e6120c`. The candidate is the uncommitted working tree.
The baseline runs from a `git archive` at
`/tmp/version1_store_baseline_382e6120c`, with the same compiled extensions
and FlashAttention Python sources. No commit or push was made.

## Scope and compatibility

Stable cluster-handle allocation and release, plus infrequent identity,
page-width, and allocation-count queries moved from
`offload/cluster_store.py` into `offload/cluster_store_metadata.py`.
The page-store file shrank from 1,904 to 1,649 lines; the new internal module
has 278 lines.
`RetroSpecClusterPageStore` binds these five methods directly. Its public
class path, MRO, call signatures, and direct method lookup remain in place.
The old `offload.cluster_store` and parent `retrospec.cluster_store` imports
still resolve. All 53 page-store method ASTs match the baseline; 48 remain
on the class, including the decode-time validation, group lookup, and CPU
page-descriptor materialization methods. No kernel, launch argument, CUDA
stream, buffer, CPU backing-page budget, resident-cache capacity formula, or
configuration rule changed.

## Tests

| Check | Result |
| --- | --- |
| Full RetroSpec suite, conda `vllm` with GPU access | 902 passed in 207.18 s |
| Affected vLLM scheduler, KV cache, and GPU runner selection | 162 passed, 1 skipped in 173.94 s |
| Baseline and candidate page-store method ASTs | All 53 identical; five moved |
| Ruff check, Ruff format check, and `git diff --check` | Passed |

## End-to-end performance

Both sides use two RTX 4090 GPUs, conda `vllm`, one warmup, greedy decoding,
identical local models, and identical NIAH samples. Qwen 30k has 31,181
input tokens and 64 fixed generated tokens. Llama 120k has the full
119,957-token input and 128 fixed generated tokens. TP and PP each use both
GPUs; eager and CUDA graph are measured separately. Short single-request
and two-request PP workloads are included. Configurations match
`version1_launcher_results.md`. Final-version per-repeat measurements are in
`version1_store_runs.tsv`; raw logs are under
`/tmp/version1_store_*_{final,final_reverse,final_profile}.log`.

Median request wall time in seconds, **baseline / candidate**. The reverse
pair ran candidate first. Llama used two measured requests per side after one
warmup. Qwen and short requests used four per side initially; the reverse
pair used eight for Qwen TP graph, Qwen PP graph, and the two-request PP case.
Profiled requests are excluded from these latency medians.

| Workload | Initial pair | Reverse pair |
| --- | ---: | ---: |
| Llama 120k TP graph | 65.92 / 64.31 | 66.09 / 65.46 |
| Llama 120k TP eager | 64.95 / 65.89 | — |
| Llama 120k PP graph | 87.40 / 87.36 | — |
| Llama 120k PP eager | 85.73 / 86.06 | — |
| Qwen 30k TP graph | 17.08 / 19.14 | 13.52 / 17.73 |
| Qwen 30k TP eager | 19.96 / 19.59 | 21.74 / 17.77 |
| Qwen 30k PP graph | 12.79 / 13.81 | 13.46 / 13.82 |
| Qwen 30k PP eager | 14.41 / 14.48 | 16.01 / 12.41 |
| Qwen short single graph | 2.05 / 2.02 | 2.09 / 2.03 |
| Qwen short two-request PP graph | 6.96 / 6.26 | 7.09 / 6.85 |

The initial-pair first-token latency, decode throughput, and device-wide
peak-memory samples are below. Throughput is the median generated tokens
after the first token divided by each request's decode seconds. The memory
sampler polls each GPU every 0.5 s and includes all device processes.

| Workload | TTFT s, baseline / candidate | Decode token/s, baseline / candidate | Peak MiB GPU0/GPU1, baseline → candidate |
| --- | ---: | ---: | ---: |
| Llama TP graph | 55.69 / 55.09 | 12.77 / 14.12 | 17,633/18,274 → 17,769/18,212 |
| Llama TP eager | 54.51 / 55.63 | 12.44 / 12.65 | 17,299/18,038 → 17,371/17,870 |
| Llama PP graph | 77.67 / 77.41 | 13.34 / 13.06 | 19,379/20,644 → 19,377/20,532 |
| Llama PP eager | 75.36 / 75.68 | 12.51 / 12.48 | 19,297/19,754 → 19,297/19,920 |
| Qwen TP graph | 8.33 / 9.92 | 7.34 / 6.93 | 13,127/12,736 → 13,127/12,736 |
| Qwen TP eager | 10.27 / 11.03 | 6.04 / 6.84 | 12,581/12,190 → 12,583/12,192 |
| Qwen PP graph | 9.96 / 10.90 | 21.03 / 22.37 | 15,679/15,322 → 15,677/15,322 |
| Qwen PP eager | 10.21 / 9.19 | 22.64 / 14.98 | 14,809/14,452 → 14,809/14,450 |
| Short single graph | 1.94 / 1.90 | 297.68 / 270.50 | 24,074/4 → 24,074/4 |
| Short two-request PP graph | 4.21 / 4.05 | 4.85 / 6.03 | 22,171/21,824 → 22,169/21,822 |

All 22 Llama measured requests used exactly 119,957 prompt tokens, found the
needle, and generated the same 128 token IDs. All 92 Qwen 30k measured
requests used 31,181 prompt tokens, generated 64 tokens, and found the
needle; their output hashes and speculative paths varied on both sides.
All 16 short single-request runs generated the same 32 token IDs. The short
prefix omits the needle, so `answer_found=False` is expected. Both requests
found the needle in all 24 two-request PP runs; completion length sometimes
varied. Short-request decode spans near 0.1 s make throughput percentages
sensitive to a few milliseconds. One candidate PP graph reverse run did not
return GPU peak samples; the initial pair captured them on both sides.

## Controlled cluster-build comparison

The moved allocation and release methods run during cluster construction;
decode-time validation, group lookup, and descriptor materialization remain
in the original class. To isolate construction and prefill, Qwen 30k was also
run with the same 31,181-token input and only **one generated token**. Every
run produced the same token hash. Each side had one warmup and eight measured
requests; the reverse TP pair ran baseline first. These measurements are in
`version1_store_prefill_runs.tsv`.

| Controlled workload | Wall s, baseline / candidate | TTFT s, baseline / candidate | Peak MiB GPU0/GPU1, baseline → candidate |
| --- | ---: | ---: | ---: |
| TP graph | 5.202 / 5.202 | 5.142 / 5.135 | 12,636/12,636 → 13,027/12,636 |
| TP graph, reversed | 5.184 / 5.186 | 5.128 / 5.119 | 13,027/12,636 → 13,027/12,636 |
| PP graph | 7.227 / 7.087 | 7.166 / 7.026 | 14,385/14,028 → 14,385/14,028 |

The first TP pair's 391 MiB GPU0 peak difference appeared on the baseline
side too in the reversed pair, while both sides had the same peak in the
standard 64-token TP graph run. The device-wide sampler therefore does not
show a repeatable memory increase. One-token requests cannot answer the
needle question; `answer_found=False` is expected here.

## Offload-stage comparison

A separate profiled Llama 120k TP eager pair used one warmup and one measured
request per side. Both measured requests found the needle and generated the
same 128 token IDs. Rank-0 measured-request counters were:

| Counter or sampled stage | Baseline | Candidate |
| --- | ---: | ---: |
| Prompt tokens / prefill tiles | 119,957 / 128 | 119,957 / 128 |
| Cluster builds / pages built | 480 / 1,382,228 | 480 / 1,382,227 |
| Draft / verified tokens | 133 / 126 | 126 / 126 |
| Prefetch waves submitted | 7 | 7 |
| Full verification H2D bytes | 15,709,765,632 | 15,709,765,632 |
| Token KV D2H bytes | 7,854,882,816 | 7,854,882,816 |
| Verification miss H2D bytes | 945,192,960 | 856,834,048 |
| Cluster-page allocation CPU average | 1.668 ms / 480 | 1.667 ms / 480 |
| Cluster-page registration CPU average | 17.263 ms / 480 | 17.375 ms / 480 |
| Draft model CUDA average | 91.669 ms / 133 | 91.797 ms / 126 |

The draft count and verification misses followed different speculative paths;
the page-build count differs by one page among about 1.38 million. The
full-layer and token-KV transfer volumes, prefill tile count, and cluster
build count match exactly. The profiled request wall times were 63.28 / 63.54 s.

Qwen 30k TP graph was profiled for six measured requests per side. Rank-0
records show that even the **same output hash** may require different draft
and verification work. For example, `fcd67dc9` used 123 draft tokens and
14.57 s on one baseline request, versus 110 tokens and 13.11 s or 217 tokens
and 21.77 s on candidate requests. For `fc1bfef3`, a baseline request used
180 draft tokens and 19.21 s; a candidate request used 194 and 20.66 s.
The fast `17f84c42` route also occurred on candidate, using 77 draft tokens
and 11.61 s. Across these six requests, the measured cluster bucket resolve
CPU averages were 0.658–0.757 ms (baseline) and 0.692–0.782 ms (candidate);
draft-model CUDA averages were 45.16–49.79 ms and 46.46–50.03 ms. These
ranges overlap. This is evidence that the higher raw Qwen TP graph median
reflects a different mix of speculative work, rather than a measured increase
in per-operation cost. It does not establish a strict ±3% whole-request bound
for this nondeterministic workload.

## Review note

The standard Qwen TP graph medians are higher for candidate in both run
orders, while both sides have 10–22 s ranges, several generated-token hashes,
and different draft/verification counts. The controlled construction
comparison, unchanged method ASTs, unchanged decode-time method placement,
and matched-work stage measurements show no repeatable cost increase in the
code moved by this patch. Whole-request variation remains a limitation of
this benchmark, so the raw medians and all 154 standard/profiled plus 48
controlled repeats are retained for review. The batch should be assessed with
that limitation in mind.
