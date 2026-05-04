"""Benchmark: Naive vs Continuous Batching engine comparison.

Runs the same set of prompts through both engines and compares:
  - Total throughput (tokens/sec)
  - Average latency per request
  - Time to complete all requests

Usage:
    python -m benchmarks.compare_engines
    python -m benchmarks.compare_engines --num-prompts 20 --max-tokens 64
"""

import argparse
import statistics
import time
from typing import List, Tuple

import torch
from loguru import logger

from src.cache.manager import CacheManager
from src.engine.batching import ContinuousBatchingEngine
from src.engine.naive import GenerationRequest, NaiveEngine
from src.engine.scheduler import RequestScheduler, ScheduledRequest
from src.models.config import ModelConfig
from src.models.loader import ModelLoader


SAMPLE_PROMPTS = [
    "The future of artificial intelligence is",
    "In a world where robots can think,",
    "The most important scientific discovery was",
    "Once upon a time in a digital realm,",
    "The key to efficient distributed systems is",
    "Machine learning models are trained by",
    "The transformer architecture revolutionized NLP because",
    "When designing scalable infrastructure, one must consider",
    "The relationship between mathematics and computer science",
    "In the year 2050, technology will",
    "The fundamental theorem of calculus states that",
    "Deep learning has achieved remarkable success in",
    "The principles of good software engineering include",
    "Natural language processing enables computers to",
    "The history of computing began with",
    "Quantum computing promises to solve problems that",
    "The ethics of artificial intelligence require us to",
    "Operating systems manage hardware resources by",
    "The internet was originally designed to",
    "Reinforcement learning differs from supervised learning in that",
]


def bench_naive(
    model, tokenizer, device: str, prompts: List[str], max_tokens: int
) -> dict:
    """Benchmark the naive engine."""
    engine = NaiveEngine(model=model, tokenizer=tokenizer, device=device)

    latencies = []
    total_tokens = 0
    start = time.perf_counter()

    for i, prompt in enumerate(prompts):
        req = GenerationRequest(
            request_id=f"naive-{i}",
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.8,
        )
        result = engine.generate(req)
        latencies.append(result.total_time * 1000)
        total_tokens += result.num_generated_tokens

    wall_time = time.perf_counter() - start

    return {
        "engine": "naive",
        "num_requests": len(prompts),
        "total_tokens": total_tokens,
        "wall_time_sec": round(wall_time, 2),
        "throughput_tok_per_sec": round(total_tokens / wall_time, 1),
        "avg_latency_ms": round(statistics.mean(latencies), 1),
        "p50_latency_ms": round(statistics.median(latencies), 1),
        "p95_latency_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 1),
        "max_latency_ms": round(max(latencies), 1),
    }


def bench_batching(
    model, tokenizer, device: str, prompts: List[str], max_tokens: int,
    batch_size: int = 4,
) -> dict:
    """Benchmark the continuous batching engine."""
    scheduler = RequestScheduler(max_queue_size=256)
    cache_manager = CacheManager(num_blocks=512, block_size=16)
    engine = ContinuousBatchingEngine(
        model=model,
        tokenizer=tokenizer,
        device=device,
        scheduler=scheduler,
        cache_manager=cache_manager,
        max_batch_size=batch_size,
    )

    # Submit all requests
    for i, prompt in enumerate(prompts):
        req = ScheduledRequest(
            request_id=f"batch-{i}",
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.8,
        )
        engine.submit_request(req)

    # Run until all complete
    start = time.perf_counter()
    completed = engine.run_until_empty()
    wall_time = time.perf_counter() - start

    total_tokens = sum(r.generated_tokens for r in completed)
    stats = engine.get_stats()

    return {
        "engine": f"batching (bs={batch_size})",
        "num_requests": len(completed),
        "total_tokens": total_tokens,
        "wall_time_sec": round(wall_time, 2),
        "throughput_tok_per_sec": round(total_tokens / wall_time, 1) if wall_time > 0 else 0,
        "avg_tokens_per_iteration": round(stats["tokens_per_iteration"], 2),
        "total_iterations": stats["total_iterations"],
        "cache_peak_utilization": stats["cache"]["utilization"],
    }


def print_comparison(naive_result: dict, batching_result: dict):
    """Print a formatted comparison table."""
    print("\n" + "=" * 65)
    print("  ENGINE COMPARISON BENCHMARK")
    print("=" * 65)

    def row(label, naive_val, batch_val, unit=""):
        n = f"{naive_val}{unit}"
        b = f"{batch_val}{unit}"
        print(f"  {label:<30} {n:>14}  {b:>14}")

    print(f"  {'Metric':<30} {'Naive':>14}  {'Batching':>14}")
    print("-" * 65)

    row("Requests", naive_result["num_requests"], batching_result["num_requests"])
    row("Total tokens", naive_result["total_tokens"], batching_result["total_tokens"])
    row("Wall time", naive_result["wall_time_sec"], batching_result["wall_time_sec"], "s")
    row("Throughput", naive_result["throughput_tok_per_sec"],
        batching_result["throughput_tok_per_sec"], " tok/s")

    if "avg_latency_ms" in naive_result:
        print(f"  {'Avg latency (naive)':<30} {naive_result['avg_latency_ms']:>13}ms")
        print(f"  {'P50 latency (naive)':<30} {naive_result['p50_latency_ms']:>13}ms")
        print(f"  {'P95 latency (naive)':<30} {naive_result['p95_latency_ms']:>13}ms")

    if "total_iterations" in batching_result:
        print(f"  {'Iterations (batching)':<30} {batching_result['total_iterations']:>14}")
        print(f"  {'Tokens/iteration (batching)':<30} {batching_result['avg_tokens_per_iteration']:>14}")

    # Speedup
    if naive_result["throughput_tok_per_sec"] > 0:
        speedup = batching_result["throughput_tok_per_sec"] / naive_result["throughput_tok_per_sec"]
        print("-" * 65)
        print(f"  {'Throughput speedup':<30} {speedup:>13.2f}x")

    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Compare naive vs batching engine")
    parser.add_argument("--model", default="gpt2", help="HuggingFace model name")
    parser.add_argument("--num-prompts", type=int, default=10, help="Number of prompts")
    parser.add_argument("--max-tokens", type=int, default=32, help="Max tokens per request")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for batching engine")
    parser.add_argument("--device", default="auto", help="Device (auto/cpu/cuda/mps)")
    args = parser.parse_args()

    config = ModelConfig(name=args.model, device=args.device)
    loader = ModelLoader(config)
    model, tokenizer = loader.load()
    device = loader.device

    prompts = SAMPLE_PROMPTS[:args.num_prompts]

    print(f"\nModel: {args.model} on {device}")
    print(f"Prompts: {len(prompts)}, max_tokens: {args.max_tokens}")

    print("\nRunning naive engine...")
    naive_result = bench_naive(model, tokenizer, device, prompts, args.max_tokens)

    print("Running batching engine...")
    batching_result = bench_batching(
        model, tokenizer, device, prompts, args.max_tokens, args.batch_size
    )

    print_comparison(naive_result, batching_result)


if __name__ == "__main__":
    main()
