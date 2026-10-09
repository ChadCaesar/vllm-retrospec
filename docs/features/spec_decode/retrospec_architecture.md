# RetroSpec architecture and regression contract

RetroSpec is selected with `speculative_config={"method": "retrospec", ...}`. The active inference path is:

```text
Scheduler admission and PP batch state
  → GPUModelRunner layer-major prefill and proposal coordination
  → RetroSpecProposer
  → RetroSpecSparseAttention
  → RetroSpecGPUNativeIndex
```

## Active components

| Area | Modules | Responsibility |
| --- | --- | --- |
| Scheduler | `vllm/v1/core/sched/retrospec_admission.py`, `retrospec_pipeline.py` | GPU index tickets, exclusive prefill admission, PP batch ownership and completion |
| Runner | `vllm/v1/worker/retrospec_runner_prefill.py`, `retrospec_runner_pipeline.py`, `retrospec_runner_state.py` | Layer-major prefill, PP proposal handoff and temporary state |
| Drafting | `vllm/v1/spec_decode/retrospec/proposer.py`, `proposal/` | Request state, model execution, draft sampling, feedback and verification |
| Sparse attention | `attention.py`, `gpu_native.py`, `native/` | Attention dispatch, GPU index construction and ranking, sparse attention kernels |
| Shared sizing | `capacity.py` | GPU index memory sizing |

The public classes and import paths stay in their original modules. The original classes compose implementation mixins; calls use the same kernels, launch parameters, buffers, CUDA streams and graph paths as before the split. A method's module globals now live in its implementation module, which matters when tests patch a dependency.

The package layout separates the runtime implementation from stable entry points:

```text
retrospec/
  proposer.py, attention.py, gpu_native.py, ...  public entry points
  proposal/                                   model, draft, sampling, verification, feedback
  native/                                     index, rank, attention, shared types
    kernels/                                  build, rank, attention Triton kernels
```

Private implementation module paths changed as part of this organization; callers should use the current public modules. The tests mirror this split under `tests/retrospec/{native,proposal,integration}/`, with reusable fixtures in `support/`.

## Compatibility and performance gates

For structural changes to the active GPU-native path, preserve configuration defaults and validation, public Python imports, native operator signatures, numerical policy, memory budget formulas, kernel bodies and launch parameters. Check output tokens and phase statistics for every benchmark mode; compare latency, throughput and peak GPU memory on the same model, data, device and configuration. Warm up and repeat measurements. Investigate an approximately 3% difference and revert or repair a repeatable regression.

The pre-refactor baseline on two RTX 4090 GPUs passed all 808 `tests/retrospec` tests. With Llama-3-8B-Instruct-Gradient-1048k and NIAH 120k, TP=2 eager completed a 119,957-token prompt. PP=2 at 120k failed at KV allocation on the baseline: 14.68 GiB required versus 10.12 GiB available per GPU. A shorter input is not evidence for PP=2 at 120k.

Run `pytest -q tests/retrospec` in the `vllm` conda environment with this checkout on `PYTHONPATH`. The focused test files cover the current proposer and GPU index plus scheduler and runner integration. Use `benchmarks/retrospec/benchmark_long_context.py` for the Qwen 30k and Llama 120k comparisons.

## Refactor comparison on 2026-09-30

The comparison used the original `29a3c2cb9` source and this working tree on the same two RTX 4090 GPUs. `benchmarks/retrospec/benchmark_refactor.py` ran one or two warmups followed by repeated requests with prefix caching disabled. Each pair used the same model, prompt, GPU count, parallel mode, eager or graph mode, and memory settings. The 30k NIAH input tokenized to 31,185 tokens with the benchmark's request suffix. The Llama 120k input tokenized to 119,961 tokens with that suffix; the unmodified NIAH input has 119,957 tokens. Times below are medians in seconds. Decode time is the interval between first and last generated token; for the multi-request row, the latency columns refer to request 0.

| Model and mode | Repeats original/refactor | Total original → refactor | First token original → refactor | Decode original → refactor | Peak MiB original → refactor |
| --- | ---: | ---: | ---: | ---: | ---: |
| Llama 120k, TP=2 eager | 3/3 | 28.916 → 29.057 | 27.667 → 27.789 | 1.050 → 1.068 | 23594 → 23594 |
| Qwen 30k, TP=2 eager | 3/3 | 5.842 → 5.828 | 4.941 → 4.944 | 0.816 → 0.815 | 20928 → 20928 |
| Qwen 30k, PP=2 eager | 3/3 | 6.440 → 6.441 | 5.634 → 5.639 | 0.738 → 0.737 | 21236 → 21236 |
| Qwen 30k, TP=2 graph | 3/3 | 5.550 → 5.593 | 4.755 → 4.791 | 0.718 → 0.735 | 20686 → 20686 |
| Qwen 30k, PP=2 graph | 2/3 | 6.483 → 6.491 | 5.675 → 5.681 | 0.742 → 0.743 | 21480 → 21480 |
| Qwen short (2106 tokens), one GPU eager | 5/5 | 3.865 → 3.864 | 1.402 → 1.403 | 2.447 → 2.452 | 19172 → 19172 |
| Qwen short, two requests, PP=2 eager | 5/5 | 5.944 → 5.933 | 1.414 → 1.414 | 4.507 → 4.495 | 19448 → 19448 |

All 96-token output hashes matched between original and refactor for the long-context modes. Short-context outputs were occasionally different across repeated requests on **both** source versions; the most common output hash matched in each pair. Multi-request timing varied even for matching output hashes. A first PP=2 graph measurement differed by 868 MiB because the original run allocated a smaller KV cache (9.86 GiB versus 10.71 GiB). Repeating the original run on idle GPUs allocated 10.71 GiB and matched the refactor's peak memory.

With RetroSpec phase profiling enabled on Qwen 30k TP=2, counters and histograms were identical on both TP ranks in the original and refactor runs. Each request recorded 31,185 prefill tokens, 112 prefill tiles, 2,744 GPU-native cluster-attention layers, 85 draft tokens, 94 accepted proposal tokens, 94 sparse-verification tokens, and 13 verification-sampling launches.

Llama 120k PP=2 still cannot initialize on either source version: it requires 14.68 GiB of KV cache per GPU and has about 10.12 GiB available. No PP=2 120k latency or throughput claim is made.

## Package-layout comparison on 2026-09-30

After moving the implementation into the subpackages shown above, all 808 RetroSpec tests passed. The 717 original functions and methods compared in the refactor retained identical ASTs. The following warmup-plus-repeat measurements compare the same original `29a3c2cb9` source with the nested-package working tree on idle RTX 4090 GPUs. Times are median seconds; all listed cases generated 96 tokens per request.

The affected vLLM scheduler, runner, KV cache and EAGLE DP test selection produced 162 passes and one skip. The remaining EAGLE DP case stopped before inference because its required `meta-llama/Llama-3.1-8B-Instruct` repository returned HTTP 401 in this environment. RetroSpec TP and PP inference was exercised by the benchmark runs below.

| Model and mode | Repeats original/nested | Total original → nested | First token original → nested | Decode original → nested | Peak MiB original → nested |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen 30k, TP=2 eager | 3/3 | 5.842 → 5.772 | 4.941 → 4.901 | 0.816 → 0.805 | 20928 → 20928 |
| Qwen 30k, TP=2 graph, paired retest | 5/5 | 5.577 → 5.542 | 4.773 → 4.757 | 0.730 → 0.720 | 20686 → 20686 |
| Qwen 30k, PP=2 eager | 3/3 | 6.440 → 6.430 | 5.634 → 5.636 | 0.738 → 0.738 | 21236 → 21236 |
| Qwen 30k, PP=2 graph | 2/3 | 6.483 → 6.490 | 5.675 → 5.684 | 0.742 → 0.743 | 21480 → 21480 |
| Llama 120k, TP=2 eager | 3/3 | 28.916 → 28.926 | 27.667 → 27.677 | 1.050 → 1.042 | 23594 → 23594 |
| Qwen short, one GPU eager | 5/5 | 3.865 → 3.855 | 1.402 → 1.400 | 2.447 → 2.444 | 19172 → 19172 |
| Qwen short, two requests, PP=2 eager | 5/5 | 5.944 → 6.044 | 1.414 → 1.414 | 4.507 → 4.606 | 19448 → 19448 |

The long-context token hashes matched the original in every repeat. The short-context cases varied across repeats on both source versions; their most common output hash matched. The two-request median total time differed by 1.7%; individual runs varied up to 7.40 seconds on the original and 8.79 seconds on the nested version. The first three-repeat TP=2 graph run showed a 5% decode-time difference from an older original run. A consecutive five-repeat original/nested retest found 0.730 versus 0.720 seconds, so the apparent regression did not reproduce. Peak memory stayed equal in every paired mode. With the same warmed Qwen 30k TP=2 profiling configuration, all four per-request, per-rank counter sets and histograms were identical to the original source.
