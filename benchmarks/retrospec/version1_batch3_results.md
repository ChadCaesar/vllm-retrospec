# `version1` batch 3: CPU offload backend relocation

Baseline: committed `version1` revision `9ab7288e9`. The comparison uses a
`git archive` snapshot under `/tmp/version1_batch3_baseline` and the same
conda `vllm` environment, compiled extensions, local models, NIAH data, and
two RTX 4090 GPUs. No batch-3 commit or push is part of this change.

## Code and compatibility

The active CPU offload backend moved from 14 flat `retrospec/*.py` modules to
`retrospec/offload/`. The former modules remain import-compatible entries.
Runtime attention and capacity code now imports the backend directly. The
6,330-line page store was separated into store, prefetch, verification,
support, CPU page pool, and verification transfer modules. The 5,334-line
segmented index was separated into coordinator, build, selection, verification,
and data type modules. Other backend modules keep their kernel bodies and
launch sites intact.

AST comparison against `9ab7288e9` found 154 matching top-level definitions,
19 matching module assignments, and 209 matching methods in the two large
classes. A compatibility check
resolved 171 original definitions through their former module paths and
round-tripped every original class path through `pickle`. All 81 original
dataclass annotations resolved through their old modules. The original class
identities and qualified paths are retained. No configuration default,
memory-budget formula, KV retirement rule, kernel body, launch argument,
buffer reuse policy, or CUDA stream/graph behavior was intentionally changed.

Backend tests are grouped under `tests/retrospec/offload/`; the page store and
segmented index tests are split by responsibility. The full RetroSpec
collection remains 836 tests. The architecture and lifecycle are documented
in `docs/design/retrospec-offload.md`.

## Test results

| Check | Result |
| --- | --- |
| RetroSpec full suite in conda `vllm` with GPU access | 836 passed in 200.52 s |
| Affected v1 runner, scheduler, KV, and streaming tests | 180 passed, 1 skipped in 176.96 s |
| RetroSpec test collection after relocation | 836 collected |
| Original imports, class qualified paths, and class pickle round trips | 171 definitions passed |
| Original dataclass annotation resolution | 81 passed |
| Backend method, definition, and assignment AST comparison | 209 methods, 154 definitions, 19 assignments matched |
| Ruff check and format check | Passed for all changed Python files |
| `git diff --check` and untracked whitespace check | Passed |

The v1 selection comprises GPU model runner, scheduler, single-type KV cache
manager, KV cache utilities and metrics, and streaming GPU model runner tests.
The one skip is reported by pytest; no test failed.

## Performance comparison

The exact Qwen 30k prompt contains 31,181 tokens. The exact Llama 120k prompt
contains 119,957 tokens; it is not shortened for measurement. Each mode uses
one warmup and two measured repeats with greedy decoding. The short Qwen
prompt is clipped to 8,000 characters; its needle is absent by construction.
The configurations match the batch-2 matrix: Qwen 30k TP/PP eager and CUDA
graph, Qwen short single-GPU graph, Qwen short two-request PP graph, and Llama
120k TP/PP eager and CUDA graph. Raw baseline logs are in
`/tmp/version1_batch3_baseline_benchmarks/`; refactor logs are in
`/tmp/version1_batch3_patch_benchmarks/`.

Qwen 30k uses `max_model_len=32768`, `max_tokens=64`, cache ratio 0,
`max_gpu_index_memory=4`, GPU memory utilization 0.94, and
`max_num_batched_tokens=2048`. The short single-GPU case uses an 8,000
character prompt, `max_model_len=8192`, `max_tokens=32`, and the same cache
settings. The short two-request PP case uses the same short prompt with
`max_gpu_index_memory=2`, GPU memory utilization 0.85, and
`max_num_batched_tokens=1024`. Llama 120k uses `max_model_len=120256`,
`max_tokens=128`, cache ratio 0.08, `max_gpu_index_memory=2`, GPU memory
utilization 0.94, and `max_num_batched_tokens=2048`.

The 168 measured repeats are also saved in
`benchmarks/retrospec/version1_batch3_runs.tsv`, with per-repeat prompt and
generated token counts, answer status, generated-token hashes, total time,
TTFT, decode time and throughput, peak memory, and profiler mode. This TSV
preserves the comparison data independently of the temporary raw logs.

The initial paired medians are below; each arrow is baseline → refactor.
Decode throughput is the median of `(generated tokens - 1) / decode seconds`
per repeat. Modes with a material difference were retested below.

| Case | Total s | TTFT s | Decode s | Decode tok/s | Peak VRAM GPU 0/1 MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen 30k TP eager | 22.34 → 17.23 (-22.9%) | 11.88 → 8.37 | 10.39 → 8.77 | 6.07 → 8.36 | 12583/12192 → 12583/12192 |
| Qwen 30k TP graph | 18.45 → 21.27 (+15.2%) | 6.97 → 11.67 | 11.40 → 9.52 | 5.57 → 6.86 | 13127/12736 → 13127/12736 |
| Qwen 30k PP eager | 16.72 → 18.81 (+12.5%) | 13.74 → 12.93 | 2.92 → 5.80 | 21.59 → 14.72 | 14809/14452 → 14807/14452 |
| Qwen 30k PP graph | 14.77 → 14.08 (-4.6%) | 11.66 → 11.30 | 3.03 → 2.72 | 21.13 → 23.23 | 15677/15320 → 15679/15322 |
| Qwen short single graph | 2.24 → 2.22 (-1.1%) | 2.09 → 2.09 | 0.15 → 0.12 | 220.86 → 259.99 | 24076/4 → 24076/4 |
| Qwen short two-request PP graph | 5.82 → 6.71 (+15.3%) | 2.38 → 2.21 | 3.35 → 3.40 | 9.33 → 9.13 | 22165/21812 → 22165/21814 |
| Llama 120k TP eager | 72.88 → 74.07 (+1.6%) | 62.93 → 64.00 | 9.74 → 9.83 | 13.04 → 12.93 | 17217/17758 → 17299/17756 |
| Llama 120k TP graph | 73.18 → 73.63 (+0.6%) | 63.85 → 64.07 | 9.13 → 9.37 | 13.91 → 13.56 | 17823/18156 → 17715/18276 |
| Llama 120k PP eager | 102.84 → 102.32 (-0.5%) | 92.28 → 92.03 | 10.36 → 10.09 | 12.27 → 12.61 | 19297/19922 → 19297/19918 |
| Llama 120k PP graph | 106.43 → 104.39 (-1.9%) | 95.01 → 94.34 | 11.21 → 9.81 | 11.33 → 12.94 | 19379/21374 → 19379/20644 |

Every Llama repeat generated the same 128-token hash and found the answer.
Every Qwen 30k repeat found the answer, but TP output hashes varied within
the refactor process. The short prompts exclude the needle and therefore
report `answer_found=False` on both sides; their output hashes matched by
mode. Qwen TP eager refactor totals spanned 12.76–21.69 s in one process,
which is too wide for a two-repeat median to establish a speed change.

The logged Llama 120k TP eager layer-prefill wall median was
47,680 → 47,606 ms; full-verification CPU gather was 27.53 → 27.86 ms.
For PP eager, the same stage medians were 38,723 → 38,610 ms and
67.34 → 64.06 ms. These offload counters are interval diagnostics; their
event counts and overlap can vary, so they are not added to reconstruct
end-to-end time. The full logs retain proposal, prefetch, and verification
counters for every mode.

### Qwen retests

We reversed process order (refactor before baseline) and used four measured
repeats after warmup. Median total seconds:

| Case | Initial baseline → refactor | Four-repeat baseline → refactor |
| --- | ---: | ---: |
| Qwen 30k TP eager | 22.34 → 17.23 | 19.07 → 22.41 |
| Qwen 30k TP graph | 18.45 → 21.27 | 17.85 → 21.59 |
| Qwen 30k PP eager | 16.72 → 18.81 | 15.71 → 16.34 |
| Qwen 30k PP graph | 14.77 → 14.08 | 13.91 → 14.78 |
| Qwen short single graph | 2.24 → 2.22 | 2.23 → 2.27 |
| Qwen short two-request PP graph | 5.82 → 6.71 | 5.55 → 5.99 |

The TP eager and PP graph ordering flipped between the initial and reversed
run. TP graph baseline repeats ranged from 11.49 to 21.22 s and produced
three hashes; refactor repeats ranged from 20.39 to 22.38 s with one hash.
The baseline sample with the same hash as the refactor took 21.22 s. In the
TP eager retest, the refactor also ran more proposal rounds than the baseline,
while its layer-prefill stage did not slow. The variable proposal work and
baseline prefill time prevent a reliable speed conclusion from these paired
medians alone.

To remove profiling overhead, we then ran six measured repeats after warmup
for all six Qwen modes. The first two used baseline-before-refactor order;
the other four reversed the order. Median total seconds:

| Case | Baseline → refactor | Change |
| --- | ---: | ---: |
| Qwen 30k TP eager | 19.42 → 19.08 | -1.7% |
| Qwen 30k TP graph | 20.59 → 19.38 | -5.9% |
| Qwen 30k PP eager | 16.41 → 16.06 | -2.1% |
| Qwen 30k PP graph | 14.75 → 13.92 | -5.6% |
| Qwen short single graph | 2.23 → 2.14 | -3.7% |
| Qwen short two-request PP graph | 6.07 → 5.81 | -4.2% |

No unprofiled Qwen mode showed a repeatable total-time regression. The
profiled slowdowns reversed or disappeared, while individual Qwen 30k
repeats remained noisy: PP graph baseline ranged from 12.73 to 24.95 s.
Every 30k repeat found the answer. Token hashes varied on both sides in most
30k modes; the two short-mode hashes matched within and across revisions.
Peak VRAM remained within a few MiB for every paired Qwen mode. Raw logs are
in `/tmp/version1_batch3_retests/`, `/tmp/version1_batch3_unprofiled/`, and
`/tmp/version1_batch3_unprofiled_extra/`.

### Llama 120k PP graph retest

We reversed process order and ran four measured repeats after warmup for the
exact 119,957-token prompt. The generated 128-token hash and answer matched
on every repeat.

| Metric | Baseline → refactor |
| --- | ---: |
| Total time median | 105.03 → 104.09 s (-0.9%) |
| TTFT median | 94.82 → 93.80 s |
| Decode time median | 10.10 → 10.12 s |
| Decode throughput median | 12.59 → 12.56 tok/s |
| Peak VRAM GPU 0/1 | 19379/20588 → 19379/20648 MiB |
| Layer-prefill wall median | 39,796 → 39,888 ms |
| Full-verification CPU gather median | 67.13 → 67.58 ms |

The initial decode-throughput and GPU-1 peak-VRAM gaps did not recur in this
retest. The approximately 60 MiB GPU-1 peak difference is below 0.3% of
the baseline peak. Raw logs are in `/tmp/version1_batch3_llama_retest/`.

## Outcome and limits

All planned Qwen and Llama configurations ran on the baseline and refactor,
including the unshortened Llama 120k input. The RetroSpec and affected v1
tests passed. No retested mode showed a repeatable end-to-end regression;
the unchanged backend method ASTs and retained public imports support the
structural-only scope of this batch. Qwen 30k greedy output hashes varied
across repeats on both revisions, so those modes establish answer agreement
and overlapping token outputs, not bitwise identity on every repeat. The
short modes and all Llama 120k modes produced matching token hashes across
revisions. Profiler-enabled latency varied substantially for Qwen 30k;
unprofiled six-repeat medians are the preferred Qwen performance comparison.
