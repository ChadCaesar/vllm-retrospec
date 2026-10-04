# `version1` offload module split

Baseline commit: `bee1b2b38`. This is a structure-only change and is left
uncommitted for review.

## Scope

| Original module | New boundary | Original / current line count |
| --- | --- | ---: |
| `offload/resident_cache.py` | Core cache state stays in place; `resident_cache_types.py`, `resident_cache_lookup.py`, and `resident_cache_admission.py` contain the relocated types and method groups. | 2844 / 1483 |
| `offload/resident_kernels.py` | Host launchers stay in place; `resident_kernel_impl.py` contains Triton definitions. | 2757 / 1195 |
| `offload/execution.py` | Source types and reusable workspace stay in place; `execution_kernels.py` contains Triton definitions. | 2702 / 1396 |

The public cache and attention workspace classes remain at their original
import paths. The relocated resident access types retain their original
`__module__` for class pickling. No method body, kernel body, launcher body,
signature, launch argument, buffer allocation, stream order, CUDA graph
setting, configuration default, or memory-budget formula was intentionally
changed.

The first version inherited the two method groups as mixins. A repeated
Llama 120k TP graph slowdown traced to this dispatch change. The final cache
class binds the 19 relocated methods directly, restoring its original
single-base method lookup while keeping the source files separated.

## Verification

- Compared parsed ASTs against `bee1b2b38`: all 68 resident cache methods,
  24 resident kernel functions, 7 attention kernel functions, and relocated
  resident type definitions are identical.
- Public import identity, serialized class path, and class pickle checks:
  `18 passed` in `tests/retrospec/test_offload_compatibility.py`.
- GPU-backed resident cache, resident kernel, attention execution, and import
  compatibility selection: `113 passed` after the direct-binding correction.
- Full RetroSpec suite after that correction: `858 passed` in 196.29 seconds.
- Affected vLLM scheduler, KV cache, and GPU runner tests: `162 passed`,
  `1 skipped` in 172.51 seconds.
- Runtime import smoke check, Ruff check and format check, and
  `git diff --check` passed.
- End-to-end performance comparisons below use the two RTX 4090 GPUs, conda
  `vllm`, one warmup, and the same model, prompt hash, speculative settings,
  token budget, and graph/eager mode within each paired workload. Earlier GPU
  test attempts had been rejected before execution by the automatic approval
  system; no test result was inferred from them.

## Performance comparison

The Llama prompt is the full 119,957-token NIAH sample with 128 generated
tokens. Its generated-token SHA-256 was identical in all baseline, ablation,
and final-candidate runs. The Qwen 30k prompt is 31,181 tokens with 64 fixed
generated tokens; its output hashes vary across repeats on both versions.
Times below are medians of measured request wall time in seconds.

| Workload | `bee1b2b38` baseline | Final direct-binding split | Interpretation |
| --- | ---: | ---: | --- |
| Llama 120k TP graph, 2 repeats per process | 64.02; reverse 63.90 | 63.76 | Output identical; no slowdown. |
| Llama 120k PP graph, 2 repeats per process | 86.92 | 91.72; reverse 87.19 | First candidate process had one 96.73 s prefill outlier. Reverse pair was 86.85/87.53 s versus baseline 87.26/86.57 s. |
| Qwen 30k TP graph, 6 repeats | Earlier baseline runs 17.50 and 14.76 | 15.98 | Falls within the existing output/timing variance. |
| Qwen 30k PP graph, 6 repeats | Earlier baseline 14.08 | 14.36; reverse 12.91 | Decode-stage outliers in the first candidate process did not repeat. |
| Qwen 30k TP eager, 3 repeats | Earlier baseline 19.22 | 17.99 | No total-time regression. |

An isolated kernel-file split with the original resident cache measured
64.12 s on Llama TP. The initial mixin-based full split measured 66.51 and
66.44 s in two processes versus 64.02 and 63.90 s for the original, exceeding
the ~3% gate. Direct method binding removed this repeatable slowdown. The
final Llama TP first-token and decode medians were 54.97 and 8.59 s versus
54.51/9.31 s in the first same-day baseline. The PP reverse pair had similar
first-token and decode times, and no repeatable peak-VRAM increase. VRAM was
sampled device-wide every 0.5 s; small sample differences are not attributed
to this patch.

The short Qwen PP graph case used two 2,211-token requests. Both requests
stopped at the exact answer `2387512` in each of two measured repeats.
Raw logs are `/tmp/version1_structure_*.log`; previous-commit baseline logs
are `/tmp/version1_goal_*_patch*.log` and summarized in
`version1_followup_results.md`.

## Offload-stage counters

A separate full-length Llama 120k TP graph pair enabled RetroSpec's coarse
statistics (`--profile --stats-interval 300`). The table uses rank 0's
measured request, emitted at shutdown after one warmup. Instrumentation adds
overhead, so the unprofiled repetitions above are the latency comparison.

| Measured-request statistic | Baseline | Final split |
| --- | ---: | ---: |
| Prefill prompt tokens / tile executions | 119,957 / 128 | 119,957 / 128 |
| Draft / verified tokens | 133 / 126 | 133 / 126 |
| Sparse / expanded verification tokens | 133 / 1 | 133 / 1 |
| Full verification requests / prefetch waves | 2 / 7 | 2 / 7 |
| Resident hit rate / verification hit rate | 0.932 / 0.940 | 0.934 / 0.940 |
| Full verification H2D bytes | 15,709,765,632 | 15,709,765,632 |

The generated-token hash was identical. Resident hit counts and verification
miss-transfer bytes varied slightly between processes as background prefetch
completed at different times; the listed stage counts and output matched.
Profile logs are
`/tmp/version1_structure_llama120k_tp_profile_{baseline,direct_binding}.log`.
