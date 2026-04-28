"""Prometheus-style metrics for monitoring the inference server.

Tracks key performance indicators:
  - Request latency (p50, p95, p99)
  - Throughput (tokens/sec, requests/sec)
  - Queue depth
  - KV-cache utilization
  - Batch size distribution

Exposes a /metrics endpoint compatible with Prometheus scraping.
"""

import time
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List


@dataclass
class LatencyRecord:
    """A single latency measurement."""
    timestamp: float
    total_ms: float
    ttft_ms: float
    tokens_generated: int
    tokens_per_second: float


class MetricsCollector:
    """Collects and aggregates server metrics.

    Uses a sliding window (default 5 minutes) for percentile calculations.
    Thread-safe for concurrent request recording.
    """

    def __init__(self, window_seconds: float = 300.0):
        self.window_seconds = window_seconds
        self._lock = threading.Lock()

        # Sliding window of recent records
        self._records: Deque[LatencyRecord] = deque()

        # Counters (monotonically increasing)
        self.total_requests: int = 0
        self.total_tokens_generated: int = 0
        self.total_errors: int = 0
        self._start_time = time.monotonic()

        # Current gauge values
        self.current_batch_size: int = 0
        self.current_queue_depth: int = 0
        self.cache_utilization: float = 0.0

    def record_request(
        self,
        total_ms: float,
        ttft_ms: float,
        tokens_generated: int,
        tokens_per_second: float,
    ):
        """Record a completed request's metrics."""
        record = LatencyRecord(
            timestamp=time.monotonic(),
            total_ms=total_ms,
            ttft_ms=ttft_ms,
            tokens_generated=tokens_generated,
            tokens_per_second=tokens_per_second,
        )

        with self._lock:
            self._records.append(record)
            self.total_requests += 1
            self.total_tokens_generated += tokens_generated
            self._prune_old_records()

    def record_error(self):
        """Record a failed request."""
        with self._lock:
            self.total_errors += 1

    def update_gauges(
        self,
        batch_size: int = None,
        queue_depth: int = None,
        cache_utilization: float = None,
    ):
        """Update current gauge values."""
        if batch_size is not None:
            self.current_batch_size = batch_size
        if queue_depth is not None:
            self.current_queue_depth = queue_depth
        if cache_utilization is not None:
            self.cache_utilization = cache_utilization

    def get_metrics(self) -> dict:
        """Get all metrics as a dictionary.

        Returns a snapshot of all tracked metrics including
        percentiles computed over the sliding window.
        """
        with self._lock:
            self._prune_old_records()
            records = list(self._records)

        uptime = time.monotonic() - self._start_time

        # Compute percentiles
        latencies = sorted(r.total_ms for r in records) if records else []
        ttfts = sorted(r.ttft_ms for r in records) if records else []
        tps_values = [r.tokens_per_second for r in records] if records else []

        return {
            # Counters
            "total_requests": self.total_requests,
            "total_tokens_generated": self.total_tokens_generated,
            "total_errors": self.total_errors,
            "uptime_seconds": round(uptime, 1),

            # Request rate
            "requests_per_second": round(
                self.total_requests / uptime if uptime > 0 else 0, 2
            ),

            # Latency percentiles (ms)
            "latency_p50_ms": self._percentile(latencies, 0.50),
            "latency_p95_ms": self._percentile(latencies, 0.95),
            "latency_p99_ms": self._percentile(latencies, 0.99),

            # Time to first token (ms)
            "ttft_p50_ms": self._percentile(ttfts, 0.50),
            "ttft_p95_ms": self._percentile(ttfts, 0.95),

            # Throughput
            "avg_tokens_per_second": round(
                sum(tps_values) / len(tps_values) if tps_values else 0, 1
            ),

            # Gauges
            "current_batch_size": self.current_batch_size,
            "current_queue_depth": self.current_queue_depth,
            "cache_utilization": round(self.cache_utilization, 3),

            # Window info
            "window_seconds": self.window_seconds,
            "samples_in_window": len(records),
        }

    def format_prometheus(self) -> str:
        """Format metrics in Prometheus exposition format.

        Returns a string compatible with Prometheus scraping.
        """
        m = self.get_metrics()
        lines = [
            "# HELP inference_requests_total Total number of inference requests",
            "# TYPE inference_requests_total counter",
            f"inference_requests_total {m['total_requests']}",
            "",
            "# HELP inference_errors_total Total number of failed requests",
            "# TYPE inference_errors_total counter",
            f"inference_errors_total {m['total_errors']}",
            "",
            "# HELP inference_tokens_total Total tokens generated",
            "# TYPE inference_tokens_total counter",
            f"inference_tokens_total {m['total_tokens_generated']}",
            "",
            "# HELP inference_latency_ms Request latency in milliseconds",
            "# TYPE inference_latency_ms summary",
            f'inference_latency_ms{{quantile="0.5"}} {m["latency_p50_ms"]}',
            f'inference_latency_ms{{quantile="0.95"}} {m["latency_p95_ms"]}',
            f'inference_latency_ms{{quantile="0.99"}} {m["latency_p99_ms"]}',
            "",
            "# HELP inference_ttft_ms Time to first token in milliseconds",
            "# TYPE inference_ttft_ms summary",
            f'inference_ttft_ms{{quantile="0.5"}} {m["ttft_p50_ms"]}',
            f'inference_ttft_ms{{quantile="0.95"}} {m["ttft_p95_ms"]}',
            "",
            "# HELP inference_batch_size Current active batch size",
            "# TYPE inference_batch_size gauge",
            f"inference_batch_size {m['current_batch_size']}",
            "",
            "# HELP inference_queue_depth Current pending request count",
            "# TYPE inference_queue_depth gauge",
            f"inference_queue_depth {m['current_queue_depth']}",
            "",
            "# HELP inference_cache_utilization KV-cache memory utilization ratio",
            "# TYPE inference_cache_utilization gauge",
            f"inference_cache_utilization {m['cache_utilization']}",
        ]
        return "\n".join(lines) + "\n"

    def _percentile(self, sorted_values: List[float], p: float) -> float:
        """Compute percentile from sorted values."""
        if not sorted_values:
            return 0.0
        idx = int(len(sorted_values) * p)
        idx = min(idx, len(sorted_values) - 1)
        return round(sorted_values[idx], 2)

    def _prune_old_records(self):
        """Remove records outside the sliding window."""
        cutoff = time.monotonic() - self.window_seconds
        while self._records and self._records[0].timestamp < cutoff:
            self._records.popleft()
