# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Summarize RetroSpec benchmark results and per-rank profile intervals."""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

RESULT_MARKER = "RETROSPEC_BENCHMARK_RESULT="
COUNTER_PATTERN = re.compile(r"counters=\{([^}]*)\}")
CPU_PATTERN = re.compile(r"cpu_avg=\{([^}]*)\}")
CUDA_PATTERN = re.compile(r"cuda_avg=\{([^}]*)\}")
HISTOGRAM_PATTERN = re.compile(r"histograms=\{(.*?)\}; derived=")
HISTOGRAM_ENTRY_PATTERN = re.compile(r"([A-Za-z0-9_]+)=\[([^]]*)\]")
REASON_PATTERN = re.compile(r"reason=([^)]+)")
RANK_PATTERN = re.compile(r"Worker_TP(\d+)")
TIME_PATTERN = re.compile(r"([0-9.]+)ms/(\d+)")


def parse_named_pairs(payload: str) -> dict[str, str]:
    if payload == "none" or not payload:
        return {}
    return dict(item.split("=", 1) for item in payload.split(", "))


def parse_time(value: str) -> tuple[float, int]:
    match = TIME_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError(f"Invalid RetroSpec timing: {value}")
    return float(match.group(1)), int(match.group(2))


def parse_histograms(payload: str) -> dict[str, Counter[int]]:
    result: dict[str, Counter[int]] = {}
    for name, bins in HISTOGRAM_ENTRY_PATTERN.findall(payload):
        histogram: Counter[int] = Counter()
        if bins:
            for entry in bins.split(","):
                value, count = entry.split(":", 1)
                histogram[int(value)] += int(count)
        result[name] = histogram
    return result


def summarize(path: Path) -> dict[str, Any]:
    counters: dict[int, Counter[str]] = defaultdict(Counter)
    totals_ms: dict[int, Counter[str]] = defaultdict(Counter)
    timing_counts: dict[int, Counter[str]] = defaultdict(Counter)
    histograms: dict[int, dict[str, Counter[int]]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    reasons: dict[int, Counter[str]] = defaultdict(Counter)
    benchmark_result: dict[str, Any] | None = None

    with path.open(encoding="utf-8", errors="replace") as profile:
        for line in profile:
            if RESULT_MARKER in line:
                benchmark_result = json.loads(line.split(RESULT_MARKER, 1)[1])
            if "RetroSpec performance" not in line:
                continue
            rank_match = RANK_PATTERN.search(line)
            rank = int(rank_match.group(1)) if rank_match is not None else 0
            if match := COUNTER_PATTERN.search(line):
                counters[rank].update(
                    {
                        key: int(value)
                        for key, value in parse_named_pairs(match.group(1)).items()
                    }
                )
            for pattern in (CPU_PATTERN, CUDA_PATTERN):
                if match := pattern.search(line):
                    for key, value in parse_named_pairs(match.group(1)).items():
                        average_ms, count = parse_time(value)
                        totals_ms[rank][key] += average_ms * count
                        timing_counts[rank][key] += count
            if match := HISTOGRAM_PATTERN.search(line):
                for name, histogram in parse_histograms(match.group(1)).items():
                    histograms[rank][name].update(histogram)
            if match := REASON_PATTERN.search(line):
                reasons[rank][match.group(1)] += 1

    ranks = sorted(set(counters) | set(totals_ms) | set(histograms) | set(reasons))
    return {
        "path": str(path),
        "result": benchmark_result,
        "ranks": {
            str(rank): {
                "counters": dict(sorted(counters[rank].items())),
                "timings": {
                    name: {
                        "count": timing_counts[rank][name],
                        "total_ms": round(total, 3),
                        "average_ms": round(total / timing_counts[rank][name], 3),
                    }
                    for name, total in sorted(totals_ms[rank].items())
                },
                "histograms": {
                    name: {
                        str(value): count for value, count in sorted(histogram.items())
                    }
                    for name, histogram in sorted(histograms[rank].items())
                },
                "flush_reasons": dict(sorted(reasons[rank].items())),
            }
            for rank in ranks
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    args = parser.parse_args()
    output = [summarize(path) for path in args.logs]
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
