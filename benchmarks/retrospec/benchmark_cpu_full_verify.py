# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Probe an independent BF16 CPU target model without changing the GPU engine.

Run with conda run -n vllm python benchmarks/retrospec/benchmark_cpu_full_verify.py
--model /path/to/model --contexts 128 1024 --output /tmp/cpu-probe.json.
Add --gpu-reference 2 for long-context KV handoff and same-prefix comparisons.
The two-GPU reference partitions layers; it is not a vLLM TP benchmark.
"""

import argparse
import gc
import hashlib
import json
import os
import platform
import resource
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)
from transformers.cache_utils import DynamicCache, DynamicLayer
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


def grouped_decode_attention(
    module,
    query,
    key,
    value,
    attention_mask,
    dropout=0.0,
    scaling=None,
    **kwargs,
):
    """Share one GQA head's KV across its single-token query heads in CPU SDPA.

    Each group becomes query rows of the same KV head. All rows have the same
    absolute token position, so this is valid only for an unmasked decode step.
    Prefill, masked attention and GPU calls retain the standard implementation.
    """
    if (
        query.device.type != "cpu"
        or query.shape[-2] != 1
        or attention_mask is not None
        or dropout
        or kwargs.get("is_causal")
    ):
        return sdpa_attention_forward(
            module,
            query,
            key,
            value,
            attention_mask,
            dropout=dropout,
            scaling=scaling,
            **kwargs,
        )
    batch, heads, _, dimension = query.shape
    kv_heads, length = key.shape[1:3]
    groups = heads // kv_heads
    output = torch.nn.functional.scaled_dot_product_attention(
        query.reshape(batch * kv_heads, 1, groups, dimension),
        key.reshape(batch * kv_heads, 1, length, dimension),
        value.reshape(batch * kv_heads, 1, length, dimension),
        is_causal=False,
        scale=scaling,
    )
    return output.reshape(batch, heads, 1, dimension).transpose(1, 2).contiguous(), None


class AppendOnlyLayer(DynamicLayer):
    """Bounded full KV with dynamic valid-length views, without history cat()."""

    def __init__(self, capacity: int):
        super().__init__()
        if capacity <= 0:
            raise ValueError("KV capacity must be positive")
        self.capacity = capacity
        self.length = 0

    def lazy_initialization(self, key_states: torch.Tensor):
        self.dtype, self.device = key_states.dtype, key_states.device
        shape = (*key_states.shape[:-2], self.capacity, key_states.shape[-1])
        self.keys = torch.empty(shape, dtype=self.dtype, device=self.device)
        self.values = torch.empty_like(self.keys)
        self.is_initialized = True

    def update(self, key_states, value_states, cache_kwargs=None):
        end = self.length + key_states.shape[-2]
        if end > self.capacity:
            raise ValueError(f"KV capacity exceeded: {end} > {self.capacity}")
        if not self.is_initialized:
            self.lazy_initialization(key_states)
        self.keys[..., self.length : end, :].copy_(key_states)
        self.values[..., self.length : end, :].copy_(value_states)
        self.length = end
        return self.keys[..., :end, :], self.values[..., :end, :]

    def get_seq_length(self):
        return self.length

    def crop(self, max_length):
        if max_length < 0:
            max_length = max(0, self.length + max_length)
        self.length = min(self.length, max_length)

    def reset(self):
        self.length = 0


def make_cache(config, capacity: int) -> DynamicCache:
    cache = DynamicCache(config=config)
    cache.layers = [AppendOnlyLayer(capacity) for _ in cache.layers]
    return cache


def copy_cache_to_cpu(source, config, capacity: int):
    target = make_cache(config, capacity)
    for src, dst in zip(source.layers, target.layers, strict=True):
        length = src.get_seq_length()
        dst.update(
            src.keys[..., :length, :].to("cpu"),
            src.values[..., :length, :].to("cpu"),
        )
    return target


def cache_bytes(cache, allocated: bool = False) -> int:
    total = 0
    for layer in cache.layers:
        if layer.is_initialized:
            for tensor in (layer.keys, layer.values):
                if not allocated:
                    tensor = tensor[..., : layer.get_seq_length(), :]
                total += tensor.numel() * tensor.element_size()
    return total


def compare_logits(actual: torch.Tensor, expected: torch.Tensor):
    actual, expected = actual.float().cpu(), expected.float().cpu()
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise ValueError("Non-finite full-model logits")
    error = actual - expected
    expected_top = expected.topk(2).values
    normalized_rmse = (
        error.square().mean().sqrt() / expected.square().mean().sqrt().clamp_min(1e-8)
    ).item()
    # A gross-error check, not a cross-backend quality certification.
    if normalized_rmse > 0.1:
        raise AssertionError(f"Full-model logits error is too large: {normalized_rmse}")
    return {
        "max_abs": error.abs().max().item(),
        "mean_abs": error.abs().mean().item(),
        "normalized_rmse": normalized_rmse,
        "actual_token": actual.argmax(-1).item(),
        "reference_token": expected.argmax(-1).item(),
        "reference_margin": (expected_top[..., 0] - expected_top[..., 1]).item(),
    }


def emit(event: str, **values):
    print(json.dumps({"event": event, **values}, ensure_ascii=False), flush=True)


def synchronize(gpu_count: int):
    for device in range(gpu_count):
        torch.cuda.synchronize(device)


def forward(model, ids, cache=None, *, position: int = 0):
    device = model.get_input_embeddings().weight.device
    kwargs = {}
    if cache is not None:
        kwargs = {
            "past_key_values": cache,
            "cache_position": torch.arange(
                position, position + ids.shape[1], device=device
            ),
        }
    output = model(
        input_ids=ids.to(device),
        use_cache=cache is not None,
        logits_to_keep=1,
        **kwargs,
    )
    return output.logits[:, -1].detach()


def prefill(model, ids, cache, chunk_size: int, gpu_count: int):
    started = time.perf_counter()
    for start in range(0, ids.shape[1], chunk_size):
        logits = forward(
            model, ids[:, start : start + chunk_size], cache, position=start
        )
        if start and start % (chunk_size * 32) == 0:
            emit("prefill_progress", tokens=start, total=ids.shape[1])
    synchronize(gpu_count)
    return logits, time.perf_counter() - started


def check_stopping(model, tokenizer):
    """Check the CPU backend's ordinary greedy generation separately from timing."""
    inputs = tokenizer("The sum of two and two is", return_tensors="pt")
    length = inputs.input_ids.shape[1]
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    generation = GenerationConfig(
        do_sample=False,
        max_new_tokens=3,
        eos_token_id=None,
        pad_token_id=pad_id,
    )
    baseline = model.generate(**inputs, generation_config=generation)
    if baseline.shape[1] != length + 3:
        raise AssertionError("CPU max_new_tokens rule failed")
    eos_id = baseline[0, length].item()
    stopped = model.generate(
        **inputs,
        generation_config=generation,
        eos_token_id=[eos_id, (eos_id + 1) % model.config.vocab_size],
    )
    if stopped.shape[1] != length + 1 or stopped[0, -1].item() != eos_id:
        raise AssertionError("CPU EOS rule failed")
    stop_text = tokenizer.decode(baseline[0, length:], skip_special_tokens=False)
    if not stop_text:
        raise AssertionError("Stopping probe produced an empty stop string")
    stopped = model.generate(
        **inputs,
        generation_config=generation,
        max_new_tokens=4,
        stop_strings=[stop_text],
        tokenizer=tokenizer,
    )
    if stopped.shape[1] != length + 3:
        raise AssertionError("CPU multi-token stop-string rule failed")
    return {"max_tokens": True, "eos": True, "multi_token_stop_string": True}


def run_context(args, config, tokenizer, cpu, gpu, context: int):
    # A natural-text prefix with an answerable final question; exact token budget.
    document = tokenizer.encode(
        "The observatory recorded clear skies. The access code is 314159. ",
        add_special_tokens=False,
    )
    question = tokenizer.encode(
        "\nWhat is the access code? Answer:", add_special_tokens=False
    )
    prefix = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    remaining = context - len(prefix) - len(question)
    if remaining < 1:
        raise ValueError("Context cannot fit the question")
    ids = torch.tensor(
        [prefix + (document * (remaining // len(document) + 1))[:remaining] + question]
    )
    capacity = context + args.decode_tokens + 2
    if capacity > config.max_position_embeddings:
        raise ValueError(f"Context exceeds model positions: {capacity}")
    record: dict[str, Any] = {"context_tokens": context, "cpu_threads": args.threads}
    record["cpu_attention"] = args.cpu_attention
    emit("context_start", **record)
    if gpu is None or context <= args.cpu_prefill_limit:
        cpu_cache = make_cache(config, capacity)
        cpu_logits, record["cpu_prefill_seconds"] = prefill(
            cpu, ids, cpu_cache, args.prefill_chunk, 0
        )
    else:
        cpu_cache = None
    gpu_cache = None
    if gpu is not None:
        gpu_cache = make_cache(config, capacity)
        gpu_logits, record["gpu_prefill_seconds"] = prefill(
            gpu, ids, gpu_cache, args.prefill_chunk, args.gpu_reference
        )
        if cpu_cache is None:
            started = time.perf_counter()
            cpu_cache = copy_cache_to_cpu(gpu_cache, config, capacity)
            record["initial_kv_d2h_seconds"] = time.perf_counter() - started
            record["cpu_prefix_source"] = "gpu_full_prefill"
            cpu_logits = gpu_logits.cpu()
        else:
            record["prefill_logits_comparison"] = compare_logits(cpu_logits, gpu_logits)
            record["cpu_prefix_source"] = "independent_cpu_prefill"
    else:
        record["cpu_prefix_source"] = "independent_cpu_prefill"
    if context <= args.cpu_prefill_limit:
        reference = forward(cpu, ids)
        record["cpu_prefill_vs_uncached"] = compare_logits(cpu_logits, reference)

    cpu_times, gpu_times, comparisons, generated = [], [], [], []
    for step in range(args.decode_tokens):
        # Feed both backends the same token, even if their argmax differs.
        token = cpu_logits.argmax(-1).cpu().reshape(1, 1)
        generated.append(token.item())
        started = time.perf_counter()
        cpu_logits = forward(cpu, token, cpu_cache, position=context + step)
        cpu_times.append(time.perf_counter() - started)
        if gpu is not None:
            synchronize(args.gpu_reference)
            started = time.perf_counter()
            gpu_logits = forward(gpu, token, gpu_cache, position=context + step)
            synchronize(args.gpu_reference)
            gpu_times.append(time.perf_counter() - started)
            comparisons.append(compare_logits(cpu_logits, gpu_logits))
        if step < args.validation_tokens and context <= args.cpu_prefill_limit:
            reference_ids = torch.cat([ids, torch.tensor([generated])], dim=1)
            comparison = compare_logits(cpu_logits, forward(cpu, reference_ids))
            record.setdefault("cpu_incremental_vs_uncached", []).append(comparison)
        emit("decode_step", context=context, step=step, cpu_ms=cpu_times[-1] * 1000)

    if gpu is not None:
        # Publish only CPU's newly computed KV; the existing GPU history stays put.
        started = time.perf_counter()
        for src, dst in zip(cpu_cache.layers, gpu_cache.layers, strict=True):
            end = src.get_seq_length()
            for name in ("keys", "values"):
                cpu_delta = getattr(src, name)[..., context:end, :]
                gpu_delta = getattr(dst, name)[..., context:end, :]
                gpu_delta.copy_(cpu_delta)
                if not torch.equal(cpu_delta, gpu_delta.cpu()):
                    raise AssertionError("KV delta handoff changed tensor contents")
        synchronize(args.gpu_reference)
        record["kv_delta_roundtrip_seconds"] = time.perf_counter() - started
        token = cpu_logits.argmax(-1).cpu().reshape(1, 1)
        position = context + args.decode_tokens
        cpu_logits = forward(cpu, token, cpu_cache, position=position)
        record["after_cpu_kv_handoff"] = compare_logits(
            cpu_logits,
            forward(gpu, token, gpu_cache, position=position),
        )
        record["gpu_step_ms"] = [value * 1000 for value in gpu_times]
        record["same_prefix_comparisons"] = comparisons
        record["gpu_peak_allocated_gib"] = [
            torch.cuda.max_memory_allocated(i) / 2**30
            for i in range(args.gpu_reference)
        ]

    expected_length = context + args.decode_tokens + int(gpu is not None)
    if any(layer.get_seq_length() != expected_length for layer in cpu_cache.layers):
        raise AssertionError("CPU KV watermark does not match consumed tokens")
    record.update(
        cpu_step_ms=[value * 1000 for value in cpu_times],
        cpu_median_step_ms=statistics.median(cpu_times) * 1000,
        cpu_steady_median_step_ms=statistics.median(cpu_times[1:] or cpu_times) * 1000,
        cpu_uninterrupted_median_step_ms=statistics.median(
            cpu_times[max(1, args.validation_tokens + 1) :]
            if context <= args.cpu_prefill_limit
            and len(cpu_times) > max(1, args.validation_tokens + 1)
            else cpu_times[1:] or cpu_times
        )
        * 1000,
        cpu_kv_valid_gib=cache_bytes(cpu_cache) / 2**30,
        cpu_kv_allocated_gib=cache_bytes(cpu_cache, allocated=True) / 2**30,
        process_peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
        generated_token_ids=generated,
        generated_text=tokenizer.decode(generated),
    )
    if args.profile_cpu_step:
        length = cpu_cache.get_seq_length()
        token = cpu_logits.argmax(-1).cpu().reshape(1, 1)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
        ) as profiler:
            forward(cpu, token, cpu_cache, position=length)
        cpu_cache.crop(length)
        events = sorted(
            profiler.key_averages(),
            key=lambda event: event.self_cpu_time_total,
            reverse=True,
        )
        record["cpu_profile_top_ops"] = [
            {
                "operator": event.key,
                "self_cpu_ms": event.self_cpu_time_total / 1000,
                "total_cpu_ms": event.cpu_time_total / 1000,
                "calls": event.count,
            }
            for event in events[:15]
        ]
    emit("context_complete", **record)
    del cpu_cache, gpu_cache
    gc.collect()
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--contexts", type=int, nargs="+", default=[128, 1024])
    parser.add_argument("--decode-tokens", type=int, default=8)
    parser.add_argument("--validation-tokens", type=int, default=2)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--thread-sweep", type=int, nargs="*", default=[])
    parser.add_argument("--gpu-reference", type=int, choices=[0, 1, 2], default=0)
    parser.add_argument("--cpu-prefill-limit", type=int, default=1024)
    parser.add_argument("--prefill-chunk", type=int, default=512)
    parser.add_argument("--profile-cpu-step", action="store_true")
    parser.add_argument(
        "--cpu-attention", choices=["sdpa", "grouped_sdpa"], default="sdpa"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        min(
            args.contexts
            + args.thread_sweep
            + [args.decode_tokens, args.threads, args.prefill_chunk]
        )
        <= 0
    ):
        parser.error("Contexts, decode tokens, threads and chunk size must be positive")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    if config.model_type not in ("llama", "qwen2") or getattr(
        config, "use_sliding_window", False
    ):
        raise ValueError("Probe supports dense full-attention Llama/Qwen2 only")
    if getattr(config, "quantization_config", None):
        raise ValueError("Probe requires non-quantized weights")
    if max(args.contexts) + args.decode_tokens + 2 > config.max_position_embeddings:
        raise ValueError("Requested context and decode budget exceed model positions")
    if args.gpu_reference and torch.cuda.device_count() < args.gpu_reference:
        raise RuntimeError("Requested GPU reference devices are not available")
    if args.cpu_attention == "grouped_sdpa":
        ALL_ATTENTION_FUNCTIONS.register("sdpa", grouped_decode_attention)
    report: dict[str, Any] = {
        "args": {**vars(args), "output": str(args.output)},
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "platform": platform.platform(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "reference_parallelism": "layer_partition"
        if args.gpu_reference == 2
        else "single_device",
        "cases": [],
    }
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    with torch.inference_mode():
        started = time.perf_counter()
        cpu = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            local_files_only=True,
        ).eval()
        report["cpu_load_seconds"] = time.perf_counter() - started
        if any(parameter.device.type != "cpu" for parameter in cpu.parameters()):
            raise AssertionError("CPU model contains non-CPU parameters")
        emit("cpu_loaded", seconds=report["cpu_load_seconds"])
        report["cpu_stopping_checks"] = check_stopping(cpu, tokenizer)
        emit("cpu_stopping_checks", **report["cpu_stopping_checks"])
        gpu = None
        if args.gpu_reference:
            placement: Any = "cuda:0"
            if args.gpu_reference == 2:
                placement = {"model.embed_tokens": 0, "model.norm": 1, "lm_head": 1}
                placement.update(
                    {
                        f"model.layers.{i}": int(i >= config.num_hidden_layers // 2)
                        for i in range(config.num_hidden_layers)
                    }
                )
            gpu = AutoModelForCausalLM.from_pretrained(
                args.model,
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
                device_map=placement,
                local_files_only=True,
            ).eval()
            emit("gpu_loaded")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        for threads in [args.threads, *args.thread_sweep]:
            args.threads = threads
            torch.set_num_threads(threads)
            for context in args.contexts:
                try:
                    report["cases"].append(
                        run_context(args, config, tokenizer, cpu, gpu, context)
                    )
                except Exception as exc:
                    report["failure"] = {
                        "context_tokens": context,
                        "cpu_threads": threads,
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
                    args.output.write_text(
                        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
                    )
                    raise
                args.output.write_text(
                    json.dumps(report, ensure_ascii=False, indent=2) + "\n"
                )
    emit("complete", output=str(args.output))


if __name__ == "__main__":
    main()
