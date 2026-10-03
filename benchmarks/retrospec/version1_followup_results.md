# `version1` RetroSpec follow-up measurements

Committed baseline: `010c82d76`. The working tree contains these follow-up
changes and has not been committed or pushed. All measurements use conda
`vllm`, two RTX 4090 GPUs where TP or PP is specified, local
`Qwen2.5-7B-Instruct` and `Llama-3-8B-Instruct-Gradient-1048k`, and
`RetroSpecDecoding/throughput_eval/test_data`. The baseline runtime is a
`git archive` of the commit at `/tmp/version1_goal_baseline_runtime`; the
benchmark scripts are shared between baseline and candidate. The native
extension and native builder implementation are unchanged.

## Benchmark validity

`benchmark_repeated.py` now has two explicit completion modes: `fixed` retains
the existing `ignore_eos=True` fixed-token throughput workload, while `answer`
stops after the expected answer and includes that answer in returned text.
`--short-needle-chars` selects a bounded document window containing the needle
and appends the original final question. Request suffixes are inserted before
that question. The legacy `--prompt-chars` prefix mode remains available.

The NIAH 30k sample's answer starts at character 54,707, so the former
8,000-character prefix did not contain the needle or final question. The new
8,000-character window has one answer occurrence and the full question. Qwen
single-GPU graph measured a 2,207-token prompt: two answer-mode repeats both
returned `2387512`; two fixed-mode repeats each generated 32 tokens and found
the answer. A two-request PP graph run measured 2,211 prompt tokens per
request; both requests stopped on `2387512` in both measured repeats. Raw
logs: `/tmp/version1_goal_short_answer.log`,
`/tmp/version1_goal_short_fixed.log`, and
`/tmp/version1_goal_short_multi_answer_final_retry.log`.

Seven representative relocated class paths now have repository tests for
old/new identity, original `__module__`, and class pickle round trips. The
benchmark and compatibility selection initially passed 14 tests. A direct
native page-builder test passed on the original binary for full-page writes
and partial-page zero padding. Full-suite results appear below.

One additional PP graph configuration (`max_model_len=4096`,
`max_num_batched_tokens=2048`, two 2,211-token requests) fails in
`sync_and_slice_intermediate_tensors` with a 2048-versus-small-tensor shape
mismatch on both the candidate and unchanged baseline. The previously used
`max_model_len=8192`, `max_num_batched_tokens=1024` configuration succeeds.
The failing configuration is a pre-existing runner issue, outside this
descriptor optimization. Logs:
`/tmp/version1_goal_short_multi_answer_final.log` and
`/tmp/version1_goal_short_multi_answer_base_2048.log`.

## CPU page build diagnosis

The original `_C` binary was copied to
`/tmp/version1_goal_baseline_C.abi3.so` before any native change. The native
page-builder microbenchmark uses eight KV heads, 8,192 tokens per head,
500 clusters per head, 47 CPU slabs, page size 16, and four workers in its
mixed layout. Its original median was 2.81 ms across 20 repeats. The full
layout with 128 clusters per head and 32 slabs had a 2.12 ms median.

Profiling the unshortened 119,957-token Llama TP graph workload split the
`cluster_page_build` average of 44.82 ms per call into 0.32 ms input
preparation, 1.70 ms page allocation, 5.45 ms native packing, 37.31 ms
cluster registration, and 0.03 ms cache resizing. Each of four worker log
records counted 480 page-build calls. These timings overlap other execution
and must not be summed to infer end-to-end latency.

The descriptor arena previously allocated one small `torch.tensor` per
cluster during registration. On the 4,000-cluster synthetic case, batch
construction reduced registration median from 53.11 to 13.77 ms across ten
repeats, with identical native-page and descriptor-state SHA-256 hashes. A
reversed baseline run measured 54.04 ms with the same descriptor hash. The
native packing code has not been changed; the measured native stage is a
minority of the page-build wall time. Raw microbenchmark logs are in
`/tmp/version1_goal_registration_{base,patch,base_retest}.log`.

## End-to-end comparison

Each process warms up once, then repeats the measured request. The comparison
uses the same model, NIAH prompt hash, speculative settings and token budget.
The table shows the median measured request wall time in seconds. The 30k
sample is 31,181 prompt tokens with Qwen; the 120k sample is 119,957 prompt
tokens with Llama. The fixed-length workload generates 64 and 128 tokens,
respectively. Peak memory is sampled device-wide with `nvidia-smi` every
0.5 seconds, so only stable differences are meaningful.

| Workload | Baseline | Candidate | Notes |
| --- | ---: | ---: | --- |
| Llama 120k TP graph, 2 repeats | 72.48 | 64.89 | Reverse order: 72.32 vs 64.15; identical generated token hashes in all eight measured runs. |
| Llama 120k PP graph, 2 repeats | 104.45 | 86.87 | Identical token hashes; peak VRAM 19,378/20,645 MiB vs 19,377/20,646 MiB. |
| Qwen 30k PP graph, 6 repeats | 15.37 | 14.08 | Multiple output hashes in both groups; peak VRAM differs by at most 4 MiB. |
| Qwen 30k TP graph, 6 repeats | 16.65 | 17.50 | Reverse order: 17.33 vs 14.76. Generation and timing vary; the apparent first-pass slowdown does not repeat. |
| Qwen 30k TP eager, 3 repeats | 18.85 | 19.22 | 2% wall-time difference; different generated sequences and large run-to-run variance. |
| Qwen 30k TP graph, answer stop, 6 repeats | 14.60 | 14.82 | All runs returned the exact answer; variable speculative overshoot changes generated-token counts. |

For Llama TP graph, median first-token latency fell from 63.34 to 54.71
seconds (reverse order: 62.79 to 54.68). For Llama PP graph it fell from
94.37 to 77.12 seconds. Decode time did not show a repeatable regression:
TP graph medians were 8.94/9.94 seconds in the first pair and 9.32/9.26 in
the reverse pair; PP graph medians were 9.84/9.54. The TP peak VRAM samples
vary across runs, without a consistent increase. Raw logs use the pattern
`/tmp/version1_goal_{llama120k,qwen30k}_{tp,pp}_{graph,eager}_*.log`.

## Resident handle-table diagnosis

On a profiled Llama 120k TP graph run, the two workers rebuilt the resident
handle table 39 and 36 times, all due to page-width growth. Mean rebuild
wall time was 6.90 and 6.24 ms, respectively; 6.16 and 5.30 ms of that was
entry publication. Summed per worker, rebuilds cost roughly 0.27 and 0.22
seconds during a roughly 65-second request. The candidate's page registration
phase averaged 16–17 ms per call, down from roughly 37 ms in the pre-change
profile. The diagnostic generated the same 128-token hash as the unprofiled
Llama runs. No resident-table width or memory-budget change is justified by
these measurements. Raw log: `/tmp/version1_goal_resident_rebuild_profile.log`.

## Validation and remaining structure

`conda run -n vllm pytest -q tests/retrospec` passed 847 tests. The affected
vLLM scheduler, KV cache and runner selection passed 162 tests with one skip:
`tests/v1/core/test_scheduler.py`, `test_kv_cache_utils.py`,
`test_single_type_kv_cache_manager.py`, and
`tests/v1/worker/test_gpu_model_runner.py`. Ruff check, Ruff format check, and
`git diff --check` passed for the changed files. The two-GPU TP and PP
end-to-end runs above exercise the distributed runtime paths modified in
earlier batches; this follow-up does not change those paths.

The remaining large offload modules are
`resident_cache.py` (~2.8k lines), `resident_kernels.py` (~2.8k), and
`execution.py` (~2.7k). Their natural boundaries are lookup versus admission,
resident kernel families, and attention kernels versus workspace orchestration.
Moving them should be a separate structure-only change with AST and import
compatibility checks; this follow-up changes only the measured descriptor
publication hotspot and profile-only timers.
