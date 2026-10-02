# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Repeat a RetroSpec request after warmup for before/after comparisons."""

import argparse
import hashlib
import json
import sys
import time

from benchmark_long_context import (
    GPUMonitor,
    build_speculative_config,
    load_sample,
    parse_args,
    shutdown_llm,
)

import vllm
from vllm import LLM, SamplingParams


def main() -> None:
    extra_parser = argparse.ArgumentParser(add_help=False)
    extra_parser.add_argument("--warmups", type=int, default=1)
    extra_parser.add_argument("--repeats", type=int, default=3)
    extra_parser.add_argument("--num-requests", type=int, default=1)
    extra_parser.add_argument("--prompt-chars", type=int)
    extra_parser.add_argument("--append-request-suffix", action="store_true")
    extra, remaining = extra_parser.parse_known_args()
    if extra.warmups < 1 or extra.repeats < 1 or extra.num_requests < 1:
        raise ValueError("warmups, repeats and num-requests must be positive")
    sys.argv = [sys.argv[0], *remaining]
    args = parse_args()
    sample = load_sample(args)
    base_prompt = sample["input"]
    if extra.prompt_chars is not None:
        base_prompt = base_prompt[: extra.prompt_chars]
    prompts = (
        [base_prompt]
        if extra.num_requests == 1 and not extra.append_request_suffix
        else [base_prompt + f"\nRequest {i}." for i in range(extra.num_requests)]
    )

    config = {
        "model": args.model,
        "dtype": args.dtype,
        "max_model_len": args.max_model_len or args.context_len + args.max_tokens + 128,
        "max_num_seqs": max(args.max_num_seqs, extra.num_requests),
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "tensor_parallel_size": args.tensor_parallel_size,
        "pipeline_parallel_size": args.pipeline_parallel_size,
        "enable_chunked_prefill": True,
        "enable_prefix_caching": False,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": not args.graph,
        "disable_log_stats": False,
    }
    if not args.native:
        config["speculative_config"] = build_speculative_config(args)

    monitor = GPUMonitor()
    monitor.start()
    llm = None
    try:
        llm = LLM(**config)
        sampling = SamplingParams(
            temperature=0.0,
            max_tokens=args.max_tokens,
            ignore_eos=True,
        )
        for _ in range(extra.warmups):
            llm.generate(prompts, sampling, use_tqdm=False)

        for repeat in range(extra.repeats):
            monitor.reset()
            started = time.perf_counter()
            outputs = llm.generate(prompts, sampling, use_tqdm=False)
            elapsed = time.perf_counter() - started
            token_ids = [output.outputs[0].token_ids for output in outputs]
            result = {
                "source_path": vllm.__file__,
                "model": args.model,
                "requested_context_len": args.context_len,
                "max_model_len": config["max_model_len"],
                "max_tokens": args.max_tokens,
                "warmups": extra.warmups,
                "repeats": extra.repeats,
                "retrieval_ratio": args.retrieval_ratio,
                "cache_ratio": args.cache_ratio,
                "max_gpu_index_memory": args.max_gpu_index_memory,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "max_num_batched_tokens": args.max_num_batched_tokens,
                "profile": args.profile,
                "repeat": repeat,
                "tensor_parallel_size": args.tensor_parallel_size,
                "pipeline_parallel_size": args.pipeline_parallel_size,
                "graph": args.graph,
                "num_requests": extra.num_requests,
                "prompt_tokens": [len(output.prompt_token_ids) for output in outputs],
                "generated_tokens": [len(tokens) for tokens in token_ids],
                "generated_token_ids": token_ids,
                "generated_text": [output.outputs[0].text for output in outputs],
                "answer_found": [
                    str(sample["answer"]) in output.outputs[0].text
                    for output in outputs
                ],
                "token_hashes": [
                    hashlib.sha256(
                        json.dumps(tokens, separators=(",", ":")).encode()
                    ).hexdigest()
                    for tokens in token_ids
                ],
                "ttft_seconds": [
                    output.metrics.first_token_latency
                    if output.metrics is not None
                    else None
                    for output in outputs
                ],
                "decode_seconds": [
                    output.metrics.last_token_ts - output.metrics.first_token_ts
                    if output.metrics is not None
                    else None
                    for output in outputs
                ],
                "elapsed_seconds": elapsed,
                "peak_memory_mib": dict(monitor.peak_memory_mib),
            }
            print(
                "RETROSPEC_REPEATED_RESULT=" + json.dumps(result, sort_keys=True),
                flush=True,
            )
    finally:
        if llm is not None:
            shutdown_llm(llm)
        monitor.stop()


if __name__ == "__main__":
    main()
