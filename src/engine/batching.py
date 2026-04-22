"""Continuous batching engine.

Unlike static batching (where all requests in a batch must finish before
new ones start), continuous batching operates at the iteration level:

  Static batching:
    [req1 ████████████]
    [req2 ████████    ]  ← wastes GPU cycles waiting for req1
    [req3 ████        ]  ← even more waste

  Continuous batching:
    [req1 ████████████]
    [req2 ████████][req4 ████]  ← req4 joins as soon as req2 finishes
    [req3 ████][req5 ██████]    ← req5 joins as soon as req3 finishes

At each decode step, the engine:
  1. Runs one forward pass for all active sequences in the batch
  2. Checks if any sequence has finished (hit EOS or max_tokens)
  3. Removes finished sequences, freeing their KV-cache blocks
  4. Admits new sequences from the queue if there's capacity

This maximizes GPU utilization by keeping the batch full at all times.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
from loguru import logger

from src.cache.manager import CacheManager
from src.engine.scheduler import RequestScheduler, ScheduledRequest


@dataclass
class SequenceState:
    """Tracks the generation state of a single sequence in the batch."""
    request: ScheduledRequest
    input_ids: torch.Tensor  # current token ids [seq_len]
    generated_ids: List[int] = field(default_factory=list)
    is_finished: bool = False
    finish_reason: str = ""  # "length", "eos", "cancelled"


class ContinuousBatchingEngine:
    """Inference engine with continuous (iteration-level) batching.

    This engine processes multiple sequences simultaneously, running
    one forward pass per iteration across all active sequences.
    New sequences can join the batch at any iteration boundary.

    Args:
        model: The loaded HuggingFace model.
        tokenizer: The tokenizer.
        device: Device string (cpu/cuda/mps).
        scheduler: Request scheduler for queue management.
        cache_manager: KV-cache block manager.
        max_batch_size: Maximum sequences in a batch.
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cpu",
        scheduler: Optional[RequestScheduler] = None,
        cache_manager: Optional[CacheManager] = None,
        max_batch_size: int = 8,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.scheduler = scheduler or RequestScheduler()
        self.cache_manager = cache_manager or CacheManager()
        self.max_batch_size = max_batch_size

        # Active batch
        self._active_sequences: Dict[str, SequenceState] = {}

        # Stats
        self._total_requests = 0
        self._total_tokens_generated = 0
        self._total_iterations = 0

    @property
    def batch_size(self) -> int:
        return len(self._active_sequences)

    @property
    def has_capacity(self) -> bool:
        return self.batch_size < self.max_batch_size

    def submit_request(self, request: ScheduledRequest) -> bool:
        """Submit a request to the scheduler queue."""
        return self.scheduler.submit(request)

    def step(self) -> List[ScheduledRequest]:
        """Run one iteration of the batching engine.

        This is the core loop. Each call:
          1. Admits new requests if batch has capacity
          2. Runs one forward pass for all active sequences
          3. Processes outputs (append tokens, check completion)
          4. Returns list of completed requests in this step

        Returns:
            List of requests that completed in this iteration.
        """
        completed = []

        # Step 1: Admit new sequences from queue
        self._admit_new_sequences()

        if not self._active_sequences:
            return completed

        self._total_iterations += 1

        # Step 2: Prepare batch inputs
        batch_input_ids, seq_ids = self._prepare_batch_inputs()

        # Step 3: Forward pass
        with torch.no_grad():
            outputs = self.model(
                input_ids=batch_input_ids.to(self.device),
                use_cache=False,  # we manage cache ourselves in Phase 2+
            )

        # Step 4: Sample next tokens and update sequences
        next_tokens = self._sample_tokens(outputs.logits, seq_ids)

        # Step 5: Process results
        for seq_id, next_token in zip(seq_ids, next_tokens):
            seq_state = self._active_sequences[seq_id]
            token_id = next_token.item()

            seq_state.generated_ids.append(token_id)
            # Append to input_ids for next iteration
            seq_state.input_ids = torch.cat([
                seq_state.input_ids,
                torch.tensor([token_id]),
            ])
            self._total_tokens_generated += 1

            # Check completion
            if token_id == self.tokenizer.eos_token_id:
                seq_state.is_finished = True
                seq_state.finish_reason = "eos"
            elif len(seq_state.generated_ids) >= seq_state.request.max_tokens:
                seq_state.is_finished = True
                seq_state.finish_reason = "length"

        # Step 6: Remove finished sequences
        finished_ids = [
            sid for sid, s in self._active_sequences.items() if s.is_finished
        ]
        for seq_id in finished_ids:
            seq_state = self._active_sequences.pop(seq_id)
            output_text = self.tokenizer.decode(
                seq_state.generated_ids, skip_special_tokens=True
            )
            self.scheduler.complete_request(
                seq_id, output_text, len(seq_state.generated_ids)
            )
            self.cache_manager.free_sequence(seq_id)
            completed.append(seq_state.request)
            self._total_requests += 1

        return completed

    def run_until_empty(self) -> List[ScheduledRequest]:
        """Keep stepping until all pending and active requests are done.

        Returns:
            All completed requests.
        """
        all_completed = []
        max_iterations = 10000  # safety limit

        for _ in range(max_iterations):
            completed = self.step()
            all_completed.extend(completed)

            # Stop if nothing is pending or active
            if (self.scheduler.pending_count == 0 and
                    self.batch_size == 0):
                break

        return all_completed

    def _admit_new_sequences(self):
        """Pull requests from the scheduler and add to active batch."""
        available_slots = self.max_batch_size - self.batch_size
        if available_slots <= 0:
            return

        batch = self.scheduler.get_next_batch(available_slots)

        for request in batch:
            # Tokenize prompt
            encoded = self.tokenizer(
                request.prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
            )
            input_ids = encoded["input_ids"].squeeze(0)  # [seq_len]
            request.prompt_tokens = len(input_ids)

            # Allocate KV-cache blocks
            try:
                self.cache_manager.allocate_sequence(
                    request.request_id, len(input_ids)
                )
            except MemoryError:
                # Try evicting to make room
                evicted = self.cache_manager.evict_lru()
                if evicted and evicted in self._active_sequences:
                    evicted_state = self._active_sequences.pop(evicted)
                    evicted_state.request.fail("Evicted due to memory pressure")
                try:
                    self.cache_manager.allocate_sequence(
                        request.request_id, len(input_ids)
                    )
                except MemoryError:
                    request.fail("Insufficient KV-cache memory")
                    continue

            self._active_sequences[request.request_id] = SequenceState(
                request=request,
                input_ids=input_ids,
            )

    def _prepare_batch_inputs(self) -> Tuple[torch.Tensor, List[str]]:
        """Pad active sequences into a batch tensor.

        For simplicity, we right-pad shorter sequences. In production,
        you'd use FlashAttention or custom kernels that handle
        variable-length sequences without padding.

        Returns:
            (batch_input_ids [batch, max_seq_len], list of sequence IDs)
        """
        seq_ids = list(self._active_sequences.keys())
        all_input_ids = [
            self._active_sequences[sid].input_ids for sid in seq_ids
        ]

        # Pad to same length
        max_len = max(len(ids) for ids in all_input_ids)
        pad_id = self.tokenizer.pad_token_id or 0

        padded = []
        for ids in all_input_ids:
            padding = torch.full((max_len - len(ids),), pad_id, dtype=ids.dtype)
            padded.append(torch.cat([ids, padding]))

        batch_tensor = torch.stack(padded)  # [batch_size, max_len]
        return batch_tensor, seq_ids

    def _sample_tokens(
        self,
        logits: torch.Tensor,
        seq_ids: List[str],
    ) -> List[torch.Tensor]:
        """Sample next tokens from model output logits.

        Takes the logits for the last position of each sequence
        and samples according to the request's temperature/top_k/top_p.

        Args:
            logits: Model output [batch_size, seq_len, vocab_size]
            seq_ids: Sequence IDs in batch order

        Returns:
            List of next token tensors, one per sequence.
        """
        next_tokens = []
        for i, seq_id in enumerate(seq_ids):
            seq_state = self._active_sequences[seq_id]
            request = seq_state.request

            # Get logits for the last real token position
            seq_len = len(seq_state.input_ids)
            last_logits = logits[i, seq_len - 1, :]  # [vocab_size]

            # Apply temperature
            temp = max(request.temperature, 1e-7)
            scaled_logits = last_logits / temp

            # Apply top-k
            if request.top_k > 0 and request.top_k < scaled_logits.size(-1):
                top_k_vals, _ = torch.topk(scaled_logits, request.top_k)
                threshold = top_k_vals[-1]
                scaled_logits[scaled_logits < threshold] = float('-inf')

            # Apply top-p (nucleus sampling)
            if request.top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(
                    scaled_logits, descending=True
                )
                cumulative_probs = torch.cumsum(
                    torch.softmax(sorted_logits, dim=-1), dim=-1
                )
                mask = cumulative_probs - torch.softmax(sorted_logits, dim=-1) >= request.top_p
                sorted_logits[mask] = float('-inf')
                scaled_logits = torch.zeros_like(scaled_logits).scatter(
                    0, sorted_indices, sorted_logits
                )

            # Sample
            probs = torch.softmax(scaled_logits, dim=-1)
            if temp < 0.01:  # greedy
                next_token = torch.argmax(probs)
            else:
                next_token = torch.multinomial(probs, num_samples=1).squeeze()

            next_tokens.append(next_token)

        return next_tokens

    def get_stats(self) -> dict:
        """Return engine statistics."""
        return {
            "engine": "continuous_batching",
            "active_batch_size": self.batch_size,
            "max_batch_size": self.max_batch_size,
            "total_requests": self._total_requests,
            "total_tokens_generated": self._total_tokens_generated,
            "total_iterations": self._total_iterations,
            "tokens_per_iteration": (
                self._total_tokens_generated / self._total_iterations
                if self._total_iterations > 0 else 0
            ),
            "scheduler": self.scheduler.get_stats(),
            "cache": self.cache_manager.get_stats(),
        }
