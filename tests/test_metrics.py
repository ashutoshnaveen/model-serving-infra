"""Tests for metrics collector."""

from src.api.metrics import MetricsCollector


class TestMetricsCollector:
    def test_initial_state(self):
        mc = MetricsCollector()
        metrics = mc.get_metrics()
        assert metrics["total_requests"] == 0
        assert metrics["total_errors"] == 0

    def test_record_request(self):
        mc = MetricsCollector()
        mc.record_request(
            total_ms=150.0,
            ttft_ms=50.0,
            tokens_generated=20,
            tokens_per_second=133.3,
        )
        metrics = mc.get_metrics()
        assert metrics["total_requests"] == 1
        assert metrics["total_tokens_generated"] == 20
        assert metrics["samples_in_window"] == 1

    def test_percentiles(self):
        mc = MetricsCollector()
        for i in range(100):
            mc.record_request(
                total_ms=float(i),
                ttft_ms=float(i) / 2,
                tokens_generated=10,
                tokens_per_second=100.0,
            )
        metrics = mc.get_metrics()
        assert metrics["latency_p50_ms"] == 50.0
        assert metrics["latency_p95_ms"] == 95.0

    def test_record_error(self):
        mc = MetricsCollector()
        mc.record_error()
        mc.record_error()
        assert mc.get_metrics()["total_errors"] == 2

    def test_update_gauges(self):
        mc = MetricsCollector()
        mc.update_gauges(batch_size=4, queue_depth=12, cache_utilization=0.75)
        metrics = mc.get_metrics()
        assert metrics["current_batch_size"] == 4
        assert metrics["current_queue_depth"] == 12
        assert metrics["cache_utilization"] == 0.75

    def test_prometheus_format(self):
        mc = MetricsCollector()
        mc.record_request(100.0, 30.0, 10, 100.0)
        output = mc.format_prometheus()
        assert "inference_requests_total 1" in output
        assert "inference_tokens_total 10" in output
        assert "# TYPE" in output
