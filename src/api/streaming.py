"""Server-Sent Events (SSE) streaming for token-by-token generation.

Instead of waiting for the full response, clients receive tokens
as they're generated. This gives a much better user experience
for long generations (ChatGPT-style streaming).

Protocol: SSE (text/event-stream)
  data: {"token": "The", "index": 0, "finish_reason": null}
  data: {"token": " future", "index": 1, "finish_reason": null}
  ...
  data: {"token": "", "index": 42, "finish_reason": "length"}
  data: [DONE]
"""

import asyncio
import json
import time
import uuid
from typing import AsyncGenerator

import torch
from loguru import logger


class StreamingGenerator:
    """Generates tokens one at a time, yielding SSE-formatted events.

    This wraps the model to provide a streaming interface. Each call
    to generate_stream() returns an async generator that yields
    individual tokens as they're produced.
    """

    def __init__(self, model, tokenizer, device: str = "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
    ) -> AsyncGenerator[str, None]:
        """Stream generated tokens as SSE events.

        Yields:
            SSE-formatted strings: 'data: {"token": "...", ...}\n\n'
        """
        request_id = str(uuid.uuid4())[:8]
        start_time = time.perf_counter()

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(self.device)

        input_ids = inputs["input_ids"]
        past_key_values = None
        generated_count = 0

        for i in range(max_tokens):
            with torch.no_grad():
                if past_key_values is not None:
                    # Only feed the last token when using KV-cache
                    model_input = input_ids[:, -1:]
                    outputs = self.model(
                        input_ids=model_input,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                else:
                    # First pass: feed entire prompt
                    outputs = self.model(
                        input_ids=input_ids,
                        use_cache=True,
                    )

                past_key_values = outputs.past_key_values

            # Sample next token
            logits = outputs.logits[:, -1, :]
            next_token_id = self._sample(logits, temperature, top_k, top_p)

            # Check for EOS
            if next_token_id == self.tokenizer.eos_token_id:
                event = self._format_event(
                    token="",
                    index=i,
                    finish_reason="eos",
                    request_id=request_id,
                )
                yield event
                break

            # Decode token
            token_text = self.tokenizer.decode(
                [next_token_id], skip_special_tokens=False
            )
            generated_count += 1

            # Yield SSE event
            finish_reason = None
            if i == max_tokens - 1:
                finish_reason = "length"

            event = self._format_event(
                token=token_text,
                index=i,
                finish_reason=finish_reason,
                request_id=request_id,
            )
            yield event

            # Append token for next iteration
            input_ids = torch.cat([
                input_ids,
                torch.tensor([[next_token_id]], device=self.device),
            ], dim=1)

            # Small yield to allow event loop to process other tasks
            await asyncio.sleep(0)

        # Send final DONE event
        elapsed = time.perf_counter() - start_time
        done_event = self._format_done_event(
            request_id=request_id,
            generated_tokens=generated_count,
            elapsed_ms=elapsed * 1000,
        )
        yield done_event

    def _sample(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> int:
        """Sample a token from logits."""
        temp = max(temperature, 1e-7)
        scaled = logits / temp

        # Top-k filtering
        if top_k > 0 and top_k < scaled.size(-1):
            top_k_vals, _ = torch.topk(scaled, top_k, dim=-1)
            threshold = top_k_vals[:, -1].unsqueeze(-1)
            scaled[scaled < threshold] = float('-inf')

        # Top-p filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(scaled, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(
                torch.softmax(sorted_logits, dim=-1), dim=-1
            )
            mask = cumulative_probs - torch.softmax(sorted_logits, dim=-1) >= top_p
            sorted_logits[mask] = float('-inf')
            scaled = torch.zeros_like(scaled).scatter(1, sorted_indices, sorted_logits)

        probs = torch.softmax(scaled, dim=-1)
        if temperature < 0.01:
            return torch.argmax(probs, dim=-1).item()
        return torch.multinomial(probs, num_samples=1).squeeze().item()

    def _format_event(
        self,
        token: str,
        index: int,
        finish_reason: str = None,
        request_id: str = "",
    ) -> str:
        """Format a single SSE event."""
        data = {
            "request_id": request_id,
            "token": token,
            "index": index,
            "finish_reason": finish_reason,
        }
        return f"data: {json.dumps(data)}\n\n"

    def _format_done_event(
        self,
        request_id: str,
        generated_tokens: int,
        elapsed_ms: float,
    ) -> str:
        """Format the final DONE event with summary."""
        data = {
            "request_id": request_id,
            "generated_tokens": generated_tokens,
            "elapsed_ms": round(elapsed_ms, 1),
        }
        return f"data: {json.dumps(data)}\n\ndata: [DONE]\n\n"
