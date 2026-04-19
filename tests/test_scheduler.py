"""Tests for request scheduler."""

import time

from src.engine.scheduler import (
    RequestPriority,
    RequestScheduler,
    RequestStatus,
    ScheduledRequest,
)


def _make_request(request_id="req1", **kwargs):
    return ScheduledRequest(request_id=request_id, prompt="test", **kwargs)


class TestScheduledRequest:
    def test_initial_state(self):
        req = _make_request()
        assert req.status == RequestStatus.PENDING
        assert req.is_active

    def test_start(self):
        req = _make_request()
        req.start()
        assert req.status == RequestStatus.RUNNING
        assert req.started_at is not None

    def test_complete(self):
        req = _make_request()
        req.start()
        req.complete("output text", generated_tokens=10)
        assert req.status == RequestStatus.COMPLETED
        assert req.output_text == "output text"
        assert not req.is_active

    def test_cancel(self):
        req = _make_request()
        req.cancel()
        assert req.status == RequestStatus.CANCELLED


class TestRequestScheduler:
    def test_submit_and_get_batch(self):
        sched = RequestScheduler(max_queue_size=10)
        sched.submit(_make_request("r1"))
        sched.submit(_make_request("r2"))

        batch = sched.get_next_batch(max_batch_size=2)
        assert len(batch) == 2
        assert batch[0].request_id == "r1"
        assert batch[1].request_id == "r2"
        assert sched.running_count == 2
        assert sched.pending_count == 0

    def test_batch_respects_max_size(self):
        sched = RequestScheduler()
        for i in range(5):
            sched.submit(_make_request(f"r{i}"))

        batch = sched.get_next_batch(max_batch_size=2)
        assert len(batch) == 2
        assert sched.pending_count == 3

    def test_queue_full_rejection(self):
        sched = RequestScheduler(max_queue_size=2)
        assert sched.submit(_make_request("r1")) is True
        assert sched.submit(_make_request("r2")) is True
        assert sched.submit(_make_request("r3")) is False

    def test_priority_scheduling(self):
        sched = RequestScheduler(scheduling_policy="priority")
        sched.submit(_make_request("low", priority=RequestPriority.LOW))
        sched.submit(_make_request("high", priority=RequestPriority.HIGH))
        sched.submit(_make_request("normal", priority=RequestPriority.NORMAL))

        batch = sched.get_next_batch(max_batch_size=3)
        assert batch[0].request_id == "high"
        assert batch[1].request_id == "normal"
        assert batch[2].request_id == "low"

    def test_complete_request(self):
        sched = RequestScheduler()
        sched.submit(_make_request("r1"))
        batch = sched.get_next_batch(1)
        sched.complete_request("r1", "output", 5)
        assert sched.running_count == 0

    def test_cancel_pending(self):
        sched = RequestScheduler()
        sched.submit(_make_request("r1"))
        assert sched.cancel_request("r1") is True
        assert sched.pending_count == 0

    def test_cancel_running(self):
        sched = RequestScheduler()
        sched.submit(_make_request("r1"))
        sched.get_next_batch(1)
        assert sched.cancel_request("r1") is True
        assert sched.running_count == 0

    def test_timeout_enforcement(self):
        sched = RequestScheduler()
        req = _make_request("r1", timeout_seconds=0.01)
        sched.submit(req)
        time.sleep(0.02)
        batch = sched.get_next_batch(1)
        assert len(batch) == 0  # request should have been timed out
        assert sched.pending_count == 0

    def test_get_request_status(self):
        sched = RequestScheduler()
        sched.submit(_make_request("r1"))
        found = sched.get_request_status("r1")
        assert found is not None
        assert found.request_id == "r1"
        assert sched.get_request_status("nonexistent") is None

    def test_stats(self):
        sched = RequestScheduler()
        sched.submit(_make_request("r1"))
        stats = sched.get_stats()
        assert stats["pending"] == 1
        assert stats["total_submitted"] == 1
