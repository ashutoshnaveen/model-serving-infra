"""Baseline benchmark for the naive inference engine.

Measures latency and throughput for single-request inference
to establish a baseline for comparison with optimized engines.
"""

import time
import statistics
from typing import List

from loguru import logger

from src.engine.naive import GenerationRequest, GenerationResponse, NaiveEngine
from src.models.config import ModelConfig
from src.models.loader import ModelLoader


BENCHMARK_PROMPTS = [
    "The future of artificial intelligence is",
    "In distributed systems, consistency and availability",
    "Machine learning models can be served efficiently by",
    "The key challenge in scaling large language models is",
    "When designing a high-throughput inference server,",
    "Attention mechanisms in transformers work by",
    "Memory management for KV-cache requires",
    "The difference between static and dynamic batching is",
]


def run_single_benchmark(
    engine: NaiveEngine,
    prompt: str,
    max_tokens: int = 50,
    request_id: str = "bench",
) -> GenerationResponse:
    """Run a single generation and return the response."""
    request = GenerationRequest(
        request_id=request_id,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0.1,  # Low temp for reproducibility
    )
    return engine.generate(request)


def run_benchmark_suite(
    engine: NaiveEngine,
    prompts: List[str] = None,
    max_tokens: int = 50,
    warmup_runs: int = 2,
    benchmark_runs: int = 5,
):
    """Run full benchmark suite and print results."""
    prompts = prompts or BENCHMARK_PROMPTS

    logger.info("=" * 60)
    logger.info("NAIVE ENGINE BENCHMARK")
    logger.info("=" * 60)
    logger.info(f"Prompts: {len(prompts)}")
    logger.info(f"Max tokens per request: {max_tokens}")
    logger.info(f"Warmup runs: {warmup_runs}")
    logger.info(f"Benchmark runs: {benchmark_runs}")
    logger.info("-" * 60)

    # Warmup
    logger.info("Warming up...")
    for i in range(warmup_runs):
        run_single_benchmark(engine, prompts[0], max_tokens, f"warmup-{i}")

    # Benchmark
    latencies = []
    ttfts = []
    throughputs = []

    for run in range(benchmark_runs):
        for i, prompt in enumerate(prompts):
            result = run_single_benchmark(
                engine, prompt, max_tokens, f"bench-{run}-{i}"
            )
            latencies.append(result.total_time)
            ttfts.append(result.time_to_first_token)
            throughputs.append(result.tokens_per_second)

    # Results
    logger.info("")
    logger.info("RESULTS")
    logger.info("=" * 60)
    logger.info(f"Total requests: {len(latencies)}")
    logger.info("")
    logger.info("Latency (seconds):")
    logger.info(f"  Mean:   {statistics.mean(latencies):.3f}s")
    logger.info(f"  Median: {statistics.median(latencies):.3f}s")
    logger.info(f"  P95:    {sorted(latencies)[int(len(latencies) * 0.95)]:.3f}s")
    logger.info(f"  P99:    {sorted(latencies)[int(len(latencies) * 0.99)]:.3f}s")
    logger.info("")
    logger.info("Time to First Token:")
    logger.info(f"  Mean:   {statistics.mean(ttfts) * 1000:.1f}ms")
    logger.info(f"  Median: {statistics.median(ttfts) * 1000:.1f}ms")
    logger.info("")
    logger.info("Throughput (tokens/sec):")
    logger.info(f"  Mean:   {statistics.mean(throughputs):.1f} tok/s")
    logger.info(f"  Median: {statistics.median(throughputs):.1f} tok/s")
    logger.info("=" * 60)

    return {
        "latency_mean": statistics.mean(latencies),
        "latency_median": statistics.median(latencies),
        "latency_p95": sorted(latencies)[int(len(latencies) * 0.95)],
        "ttft_mean": statistics.mean(ttfts),
        "throughput_mean": statistics.mean(throughputs),
        "throughput_median": statistics.median(throughputs),
        "total_requests": len(latencies),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run inference benchmarks")
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()

    # Load model
    config = ModelConfig(name=args.model)
    loader = ModelLoader(config)
    model, tokenizer = loader.load()

    # Create engine
    engine = NaiveEngine(model=model, tokenizer=tokenizer, device=loader.device)

    # Run benchmarks
    results = run_benchmark_suite(
        engine,
        max_tokens=args.max_tokens,
        warmup_runs=args.warmup,
        benchmark_runs=args.runs,
    )
