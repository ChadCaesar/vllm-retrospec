# `version1` resident Triton kernel split

Baseline commit: `d56aa22b8`. The candidate is the uncommitted working tree.
The baseline ran from a `git archive` at
`/tmp/version1_kernel_baseline_d56aa22b8`, with the same local compiled
extensions and FlashAttention Python sources. No commit or push was made.

## Scope and compatibility

The 1,590-line `offload/resident_kernel_impl.py` now contains the five
lookup and table kernels (366 lines). Three shared Triton helpers live in
`resident_kernel_helpers.py` (134 lines), three draft kernels in
`resident_draft_kernel_impl.py` (723 lines), and four verification kernels in
`resident_verification_kernel_impl.py` (403 lines). The original implementation
module re-exports all ten moved definitions. The unchanged `resident_kernels.py`
launchers and the legacy `retrospec.resident_kernels` import still resolve to
the same function objects.

All 15 original Triton function ASTs match the baseline. The split changes
source location and imports only: no kernel expression, signature, launch
argument, compile-time constant, buffer, stream, graph setting, cache budget,
or configuration rule changed.

## Tests

| Check | Result |
| --- | --- |
| Resident kernels, cache, execution, and import compatibility | 148 passed |
| Full RetroSpec suite, conda `vllm` with GPU access | 893 passed in 208.62 s |
| Affected vLLM scheduler, KV cache, and GPU runner selection | 162 passed, 1 skipped in 170.75 s |
| Original and moved kernel ASTs; all ten moved import identities | Passed |
| Ruff check, Ruff format check, and `git diff --check` | Passed |

The compatibility suite gained ten parametrized cases for the moved Triton
definitions. The full-suite count increased from 883 by these ten cases.

## End-to-end performance

Both sides used the same two RTX 4090 GPUs, conda `vllm`, one warmup, greedy
decoding, and identical local models and NIAH samples. Qwen 30k used 31,181
input tokens and 64 fixed generated tokens; Llama 120k used the **full
119,957-token input** and 128 fixed generated tokens. Qwen 30k used
`max_model_len=32768`, cache ratio 0, GPU index budget 4 GiB, GPU memory
utilization 0.94, and a 2,048-token batch limit. Llama used
`max_model_len=120256`, cache ratio 0.08, GPU index budget 2 GiB, GPU memory
utilization 0.94, and the same batch limit. The short single request used
2,101 input tokens. The two-request PP case used 2,207 tokens per request
and a 1,024-token batch limit.

Median measured request wall time, seconds; TP and PP each use both GPUs:

| Workload | Baseline | Candidate | Repeats per side | Candidate difference |
| --- | ---: | ---: | ---: | ---: |
| Qwen 30k TP graph | 18.97 | 18.49 | 6 / 6 | -2.5% |
| Qwen 30k TP eager | 17.79 | 17.74 | 3 / 3 | -0.3% |
| Qwen 30k PP graph | 13.88 | 14.23 | 6 / 6 | +2.5% |
| Qwen 30k PP eager | 15.21 | 14.82 | 3 / 6 | -2.6% |
| Qwen short single graph | 2.02 | 2.04 | 6 / 6 | +0.7% |
| Qwen short two-request PP graph | 7.73 | 7.76 | 8 / 8 | +0.4% |
| Llama 120k TP graph | 64.93 | 65.19 | 2 / 2 | +0.4% |
| Llama 120k TP eager | 63.24 | 64.82 | 2 / 2 | +2.5% |
| Llama 120k PP graph | 87.06 | 87.64 | 2 / 2 | +0.7% |
| Llama 120k PP eager | 85.69 | 85.41 | 2 / 2 | -0.3% |

Every Llama repeat produced the same generated-token SHA-256 and found the
needle. Qwen 30k produced several output hashes on both sides, including
within the same process; all repeats found the needle. The common Qwen TP
graph hash `fcd67dc9` had 18.66 s baseline and 18.84 s candidate median
wall times. Both Qwen PP eager comparison sets used the same `2e680372`
hash. Its initial candidate median was 12.38 s, but the six-repeat retest
was 14.82 s, close to the 15.21 s reverse baseline. The single-request short
prefix omits the needle, so its identical output hashes are the output check.
Its initial 2.01 s baseline versus 2.16 s candidate gap narrowed to 2.02 s
versus 2.04 s in six-repeat tests. Qwen TP eager's initial 22.19 s baseline
median also narrowed to 17.79 s on its three-repeat retest, near the 17.74 s
candidate median.
In the first two-request pair, one of eight baseline batches generated ten
instead of twelve tokens for the first request; all other batches had the
same output hashes. That pair's median was 7.79 s baseline versus 7.58 s
candidate. A reverse-order eight-repeat pair was 7.73 s versus 7.76 s;
two candidate batches generated ten and nine first-request tokens, while
all baseline batches generated twelve. All requests still found the needle.
The first-token median changed direction between pairs, so its phase times
are not attributed to the module split.

Initial Llama PP graph candidate repeats included a slower 94.87 s prefill;
the next two repeats had an 87.64 s median and 77.93 s first-token median,
near the 87.06 s and 77.25 s baseline. Initial Llama TP eager candidate
total-time median was 65.63 s; retest was 64.82 s. Llama PP eager decode
medians were 10.23 s baseline, 10.72 s initial candidate, and 10.33 s
candidate retest. These >3% initial differences did not repeat. The short
single request's decode interval is about 0.1 s, so its few-millisecond
phase variation should not be read as a throughput change.

Device-wide GPU memory was sampled every 0.5 s. The Qwen eager initial
baseline samples were missing, so separate baseline repeats supplied peak
readings: TP eager 12,583/12,192 MiB on both sides; PP eager
14,809/14,450 MiB baseline versus 14,809/14,452 MiB candidate. Qwen graph
and short workloads differed by at most a few MiB in maximum sampled memory.
Llama TP and PP had varying device-wide peaks across runs, without a
consistent increase. Memory numbers include the whole GPU, not only vLLM.

All 121 measured repeats, including initial comparisons, retests, and three
profiled repeats, have their prompts, output hashes, latency phases, and VRAM
samples in `version1_kernel_runs.tsv`. Raw logs are under
`/tmp/version1_kernel_*.log`.

## Offload-stage check

A separately profiled Llama 120k TP graph pair used one warmup and one
measured request. Profiled times are excluded from the latency table. The
rank-0 measured-request counters, using the candidate retest with the same
draft work as the baseline, were:

| Counter or sampled stage | Baseline | Candidate retest |
| --- | ---: | ---: |
| Prefill tokens / tile executions | 119,957 / 128 | 119,957 / 128 |
| Draft / sparse verification / verified tokens | 133 / 133 / 126 | 133 / 133 / 126 |
| Full verification requests / H2D bytes | 2 / 15,709,765,632 | 2 / 15,709,765,632 |
| Prefetch waves submitted | 7 | 7 |
| Verification miss H2D bytes | 930,578,432 | 927,571,968 |
| Resident hit rate | 0.935 | 0.939 |
| Draft model CUDA average | 77.84 ms / 133 | 78.42 ms / 133 |
| Sparse verification CUDA average | 186.98 ms / 17 | 184.41 ms / 17 |
| Draft bucket resolution CPU average | 1.066 ms / 4,256 | 1.011 ms / 4,256 |

The first candidate profile followed a different dynamic route (126 draft
tokens, nine prefetch waves) and sampled the draft model at 99.37 ms. The
same-work retest above did not reproduce that high sample. Verification miss
H2D bytes varied by about 0.3% as asynchronous residency completed at
slightly different times. The profile does not show a repeated hot-stage
regression from moving the kernel definitions.
