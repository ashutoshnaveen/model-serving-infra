"""Naive inference engine — single-request, no batching.

This is the baseline implementation. It processes one request at a time
with no KV-cache reuse between requests and no batching.

Purpose: establish baseline latency/throughput numbers to compare against
optimized implementations (continuous batching, PagedAttention).
"""

import time
from dataclasses import dataclass
from typing import Optional

import torch
from loguru import logger


@dataclass
class GenerationRequest:
    """A single text generation request."""

    request_id: str
    prompt: str
    max_tokens: int = 256
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0
    repetition_penalty: float = 1.0


@dataclass
class GenerationResponse:
    """Response from text generation."""

    request_id: str
    generated_text: str
    prompt: str
    num_prompt_tokens: int
    num_generated_tokens: int
    time_to_first_token: float  # seconds
    total_time: float  # seconds
    tokens_per_second: float


class NaiveEngine:
    """Naive single-request inference engine.

    No batching, no KV-cache management. Each request is processed
    independently from scratch. This is intentionally simple — it's
    the baseline we'll optimize against.
    """

    def __init__(self, model, tokenizer, device: str = "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self._request_count = 0

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate text for a single request.

        This is a synchronous, blocking call. The model generates tokens
        one at a time (autoregressive decoding) using HuggingFace's
        built-in generate() method.
        """
        self._request_count += 1
        start_time = time.perf_counter()

        # Tokenize input
        inputs = self.tokenizer(
            request.prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(self.device)

        num_prompt_tokens = inputs["input_ids"].shape[1]

        # Generate
        with torch.no_grad():
            # Track time to first token
            ttft_start = time.perf_counter()

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=max(request.temperature, 1e-7),
                top_k=request.top_k,
                top_p=request.top_p,
                repetition_penalty=request.repetition_penalty,
                do_sample=request.temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            ttft = time.perf_counter() - ttft_start

        # Decode output (only the generated tokens, not the prompt)
        generated_ids = outputs[0][num_prompt_tokens:]
        generated_text = self.tokenizer.decode(
            generated_ids, skip_special_tokens=True
        )

        total_time = time.perf_counter() - start_time
        num_generated = len(generated_ids)
        tps = num_generated / total_time if total_time > 0 else 0

        logger.debug(
            f"Request {request.request_id}: "
            f"{num_prompt_tokens} prompt tokens, "
            f"{num_generated} generated tokens, "
            f"{tps:.1f} tok/s"
        )

        return GenerationResponse(
            request_id=request.request_id,
            generated_text=generated_text,
            prompt=request.prompt,
            num_prompt_tokens=num_prompt_tokens,
            num_generated_tokens=num_generated,
            time_to_first_token=ttft,
            total_time=total_time,
            tokens_per_second=tps,
        )

    @property
    def stats(self) -> dict:
        """Return engine statistics."""
        return {
            "engine": "naive",
            "total_requests_served": self._request_count,
            "batching": False,
            "kv_cache_reuse": False,
        }
