"""Request scheduler for the inference engine.

Manages incoming generation requests with support for:
  - FIFO queue (default)
  - Priority-based scheduling
  - Request timeouts and cancellation
  - Admission control based on available memory

The scheduler sits between the API layer and the batching engine,
deciding which requests to run and in what order.
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Callable, Dict, List, Optional

from loguru import logger


class RequestStatus(Enum):
    """Lifecycle states of a generation request."""
    PENDING = "pending"       # in queue, waiting to be scheduled
    RUNNING = "running"       # actively generating tokens
    COMPLETED = "completed"   # generation finished
    CANCELLED = "cancelled"   # cancelled by client or timeout
    FAILED = "failed"         # generation failed with error


class RequestPriority(IntEnum):
    """Priority levels (lower number = higher priority)."""
    HIGH = 0
    NORMAL = 1
    LOW = 2


@dataclass
class ScheduledRequest:
    """A generation request with scheduling metadata."""
    request_id: str
    prompt: str
    max_tokens: int = 256
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    stream: bool = False

    # Scheduling metadata
    priority: RequestPriority = RequestPriority.NORMAL
    status: RequestStatus = RequestStatus.PENDING
    created_at: float = field(default_factory=time.monotonic)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    timeout_seconds: float = 60.0

    # Token tracking
    prompt_tokens: int = 0
    generated_tokens: int = 0

    # Result
    output_text: str = ""
    error: Optional[str] = None

    @property
    def wait_time(self) -> float:
        """Time spent waiting in queue."""
        if self.started_at is not None:
            return self.started_at - self.created_at
        return time.monotonic() - self.created_at

    @property
    def is_timed_out(self) -> bool:
        """Check if this request has exceeded its timeout."""
        return (time.monotonic() - self.created_at) > self.timeout_seconds

    @property
    def is_active(self) -> bool:
        return self.status in (RequestStatus.PENDING, RequestStatus.RUNNING)

    def start(self):
        self.status = RequestStatus.RUNNING
        self.started_at = time.monotonic()

    def complete(self, output_text: str, generated_tokens: int):
        self.status = RequestStatus.COMPLETED
        self.completed_at = time.monotonic()
        self.output_text = output_text
        self.generated_tokens = generated_tokens

    def cancel(self):
        self.status = RequestStatus.CANCELLED
        self.completed_at = time.monotonic()

    def fail(self, error: str):
        self.status = RequestStatus.FAILED
        self.completed_at = time.monotonic()
        self.error = error


class RequestScheduler:
    """Manages the request queue and scheduling decisions.

    Supports FCFS (first-come-first-served) and priority-based scheduling.
    Handles timeout enforcement and admission control.
    """

    def __init__(
        self,
        max_queue_size: int = 128,
        default_timeout: float = 60.0,
        scheduling_policy: str = "fcfs",
    ):
        self.max_queue_size = max_queue_size
        self.default_timeout = default_timeout
        self.scheduling_policy = scheduling_policy

        # Request storage
        self._pending: List[ScheduledRequest] = []
        self._running: Dict[str, ScheduledRequest] = {}
        self._completed: Dict[str, ScheduledRequest] = {}

        # Stats
        self._total_submitted = 0
        self._total_completed = 0
        self._total_cancelled = 0
        self._total_timed_out = 0

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def running_count(self) -> int:
        return len(self._running)

    @property
    def queue_full(self) -> bool:
        return self.pending_count >= self.max_queue_size

    def submit(self, request: ScheduledRequest) -> bool:
        """Submit a new request to the queue.

        Returns:
            True if accepted, False if queue is full.
        """
        if self.queue_full:
            logger.warning(
                f"Queue full ({self.max_queue_size}), rejecting {request.request_id}"
            )
            return False

        request.timeout_seconds = (
            request.timeout_seconds or self.default_timeout
        )
        self._pending.append(request)
        self._total_submitted += 1

        logger.debug(
            f"Submitted request {request.request_id} "
            f"(queue depth: {self.pending_count})"
        )
        return True

    def get_next_batch(self, max_batch_size: int) -> List[ScheduledRequest]:
        """Select the next batch of requests to process.

        Applies the scheduling policy and timeout enforcement.

        Args:
            max_batch_size: Maximum requests to include in the batch.

        Returns:
            List of requests to run in this iteration.
        """
        # First, clean up timed-out requests
        self._enforce_timeouts()

        if not self._pending:
            return []

        # Sort by scheduling policy
        if self.scheduling_policy == "priority":
            self._pending.sort(key=lambda r: (r.priority, r.created_at))
        # else: FCFS — already in order

        # Select batch
        batch_size = min(max_batch_size, len(self._pending))
        batch = self._pending[:batch_size]
        self._pending = self._pending[batch_size:]

        # Mark as running
        for req in batch:
            req.start()
            self._running[req.request_id] = req

        return batch

    def complete_request(
        self,
        request_id: str,
        output_text: str,
        generated_tokens: int,
    ):
        """Mark a request as completed."""
        if request_id not in self._running:
            logger.warning(f"Completing unknown request: {request_id}")
            return

        req = self._running.pop(request_id)
        req.complete(output_text, generated_tokens)
        self._completed[request_id] = req
        self._total_completed += 1

    def cancel_request(self, request_id: str) -> bool:
        """Cancel a pending or running request.

        Returns:
            True if the request was found and cancelled.
        """
        # Check pending
        for i, req in enumerate(self._pending):
            if req.request_id == request_id:
                req.cancel()
                self._pending.pop(i)
                self._completed[request_id] = req
                self._total_cancelled += 1
                return True

        # Check running
        if request_id in self._running:
            req = self._running.pop(request_id)
            req.cancel()
            self._completed[request_id] = req
            self._total_cancelled += 1
            return True

        return False

    def get_request_status(self, request_id: str) -> Optional[ScheduledRequest]:
        """Look up a request by ID across all states."""
        for req in self._pending:
            if req.request_id == request_id:
                return req
        if request_id in self._running:
            return self._running[request_id]
        return self._completed.get(request_id)

    def _enforce_timeouts(self):
        """Cancel requests that have exceeded their timeout."""
        timed_out = [r for r in self._pending if r.is_timed_out]
        for req in timed_out:
            req.cancel()
            req.error = "Request timed out in queue"
            self._pending.remove(req)
            self._completed[req.request_id] = req
            self._total_timed_out += 1
            logger.info(f"Request {req.request_id} timed out after {req.wait_time:.1f}s")

    def get_stats(self) -> dict:
        """Return scheduler statistics."""
        wait_times = [
            r.wait_time for r in self._completed.values()
            if r.started_at is not None
        ]
        return {
            "pending": self.pending_count,
            "running": self.running_count,
            "completed": self._total_completed,
            "cancelled": self._total_cancelled,
            "timed_out": self._total_timed_out,
            "total_submitted": self._total_submitted,
            "avg_wait_time_ms": (
                sum(wait_times) / len(wait_times) * 1000
                if wait_times else 0
            ),
            "scheduling_policy": self.scheduling_policy,
            "queue_capacity": self.max_queue_size,
        }
