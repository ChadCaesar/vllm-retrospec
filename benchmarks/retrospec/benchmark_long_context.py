# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Run one reproducible RetroSpec long-context request per process."""

import argparse
import json
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from vllm import LLM, SamplingParams


class GPUMonitor:
    def __init__(self, interval_seconds: float = 0.5) -> None:
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.peak_memory_mib: dict[int, int] = {}
        self.peak_utilization_percent: dict[int, int] = {}

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()

    def reset(self) -> None:
        self.peak_memory_mib.clear()
        self.peak_utilization_percent.clear()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=index,memory.used,utilization.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                for line in result.stdout.splitlines():
                    index, memory, utilization = (
                        int(value.strip()) for value in line.split(",")
                    )
                    self.peak_memory_mib[index] = max(
                        self.peak_memory_mib.get(index, 0), memory
                    )
                    self.peak_utilization_percent[index] = max(
                        self.peak_utilization_percent.get(index, 0), utilization
                    )
            except (OSError, subprocess.SubprocessError, ValueError):
                pass
            self._stop.wait(self.interval_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--context-len", type=int, required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.94)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--num-speculative-tokens", type=int, default=64)
    parser.add_argument("--min-draft-tokens", type=int, default=1)
    parser.add_argument("--max-draft-tokens", type=int, default=16)
    parser.add_argument("--retrieval-ratio", type=float, default=0.018)
    parser.add_argument("--estimation-ratio", type=float, default=0.232)
    parser.add_argument("--cache-ratio", type=float, default=0.0)
    parser.add_argument("--index-segment-size", type=int, default=8192)
    parser.add_argument("--index-update-interval", type=int, default=1024)
    parser.add_argument("--prefill-tile-size", type=int, default=8192)
    parser.add_argument("--blocks-per-cluster", type=int, default=1)
    parser.add_argument("--kmeans-iterations", type=int, default=10)
    parser.add_argument("--cpu-page-build-workers", type=int, default=4)
    parser.add_argument("--full-verify-gather-workers", type=int, default=4)
    parser.add_argument("--max-pinned-memory", type=float, default=1.0)
    parser.add_argument("--max-gpu-index-memory", type=float, default=4.0)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--stats-interval", type=float, default=1.0)
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--native", action="store_true")
    parser.add_argument("--sync-attention-gates", action="store_true")
    return parser.parse_args()


def load_sample(args: argparse.Namespace) -> dict[str, Any]:
    path = args.dataset_dir / f"NIAH_{args.context_len}.json"
    with path.open(encoding="utf-8") as data_file:
        samples = json.load(data_file)
    try:
        return samples[args.sample_index]
    except IndexError as exc:
        raise ValueError(f"Sample index {args.sample_index} is outside {path}") from exc


def build_speculative_config(args: argparse.Namespace) -> dict[str, Any]:
    config: dict[str, Any] = {
        "method": "retrospec",
        "num_speculative_tokens": args.num_speculative_tokens,
        "retrospec_retrieval_ratio": args.retrieval_ratio,
        "retrospec_estimation_ratio": args.estimation_ratio,
        "retrospec_cache_ratio": args.cache_ratio,
        "retrospec_index_segment_size": args.index_segment_size,
        "retrospec_index_update_interval": args.index_update_interval,
        "retrospec_prefill_tile_size": args.prefill_tile_size,
        "retrospec_blocks_per_cluster": args.blocks_per_cluster,
        "retrospec_kmeans_iterations": args.kmeans_iterations,
        "retrospec_cpu_page_build_workers": args.cpu_page_build_workers,
        "retrospec_full_verify_gather_workers": args.full_verify_gather_workers,
        "retrospec_max_pinned_memory": args.max_pinned_memory,
        "retrospec_max_gpu_index_memory": args.max_gpu_index_memory,
        "retrospec_min_draft_tokens": args.min_draft_tokens,
        "retrospec_max_draft_tokens": args.max_draft_tokens,
        "retrospec_stats_interval_seconds": (
            args.stats_interval if args.profile else 0.0
        ),
        "enforce_eager": not args.graph,
    }
    if args.sync_attention_gates:
        config.update(
            {
                "retrospec_hit_attn_threshold": 0.0,
                "retrospec_retrieval_attn_threshold": 0.0,
                "retrospec_expanded_attn_threshold": 0.0,
            }
        )
    return config


def shutdown_llm(llm: LLM) -> None:
    """Stop workers explicitly so terminal RetroSpec samples are flushed."""
    llm.llm_engine.engine_core.shutdown()


def main() -> None:
    args = parse_args()
    sample = load_sample(args)
    monitor = GPUMonitor()
    llm: LLM | None = None
    monitor.start()
    try:
        initialization_started = time.perf_counter()
        llm_args: dict[str, Any] = {
            "model": args.model,
            "dtype": args.dtype,
            "max_model_len": args.max_model_len
            or args.context_len + args.max_tokens + 128,
            "max_num_seqs": args.max_num_seqs,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "tensor_parallel_size": args.tensor_parallel_size,
            "pipeline_parallel_size": args.pipeline_parallel_size,
            "enable_chunked_prefill": True,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "enforce_eager": not args.graph,
            "disable_log_stats": False,
        }
        if not args.native:
            llm_args["speculative_config"] = build_speculative_config(args)
        llm = LLM(**llm_args)
        initialization_seconds = time.perf_counter() - initialization_started
        initialization_peak_memory_mib = dict(monitor.peak_memory_mib)

        monitor.reset()
        started_at = time.perf_counter()
        output = llm.generate(
            [sample["input"]],
            SamplingParams(
                temperature=0.0, max_tokens=args.max_tokens, ignore_eos=True
            ),
            use_tqdm=False,
        )[0]
        elapsed_seconds = time.perf_counter() - started_at

        completion = output.outputs[0]
        output_tokens = len(completion.token_ids)
        metrics = output.metrics
        ttft_seconds = metrics.first_token_latency if metrics is not None else None
        decode_seconds = (
            metrics.last_token_ts - metrics.first_token_ts
            if metrics is not None
            else None
        )
        answer = str(sample["answer"])
        result = {
            "mode": "native" if args.native else "retrospec",
            "profile": args.profile,
            "graph": args.graph,
            "requested_context_len": args.context_len,
            "actual_prompt_tokens": len(output.prompt_token_ids),
            "requested_output_tokens": args.max_tokens,
            "generated_tokens": output_tokens,
            "finish_reason": completion.finish_reason,
            "answer": answer,
            "answer_found": answer in completion.text,
            "generated_text": completion.text,
            "initialization_seconds": initialization_seconds,
            "elapsed_seconds": elapsed_seconds,
            "ttft_seconds": ttft_seconds,
            "decode_seconds": decode_seconds,
            "decode_tokens_per_second": (
                (output_tokens - 1) / decode_seconds
                if decode_seconds and output_tokens > 1
                else None
            ),
            "initialization_peak_memory_mib": initialization_peak_memory_mib,
            "generation_peak_memory_mib": dict(monitor.peak_memory_mib),
            "generation_peak_utilization_percent": dict(
                monitor.peak_utilization_percent
            ),
        }
    finally:
        if llm is not None:
            shutdown_llm(llm)
        monitor.stop()

    print(
        "RETROSPEC_BENCHMARK_RESULT="
        + json.dumps(result, ensure_ascii=False, sort_keys=True),
        flush=True,
    )


if __name__ == "__main__":
    main()
