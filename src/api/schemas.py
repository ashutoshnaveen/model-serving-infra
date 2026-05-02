"""Pydantic schemas for API request/response validation."""

from typing import Optional

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    """Request body for /generate endpoint."""

    prompt: str = Field(..., description="Input text prompt", min_length=1)
    max_tokens: int = Field(
        default=256,
        description="Maximum number of tokens to generate",
        ge=1,
        le=2048,
    )
    temperature: float = Field(
        default=1.0,
        description="Sampling temperature. 0 = greedy, higher = more random",
        ge=0.0,
        le=2.0,
    )
    top_k: int = Field(
        default=50,
        description="Top-k sampling. Only sample from top k tokens",
        ge=1,
    )
    top_p: float = Field(
        default=1.0,
        description="Nucleus sampling. Sample from tokens with cumulative prob <= top_p",
        ge=0.0,
        le=1.0,
    )
    repetition_penalty: float = Field(
        default=1.0,
        description="Penalize repeated tokens. 1.0 = no penalty",
        ge=1.0,
        le=2.0,
    )
    stream: bool = Field(
        default=False,
        description="Stream the response token by token via SSE",
    )

    model_config = {"json_schema_extra": {
        "examples": [
            {
                "prompt": "The future of AI is",
                "max_tokens": 50,
                "temperature": 0.8,
            }
        ]
    }}


class GenerateResponse(BaseModel):
    """Response body for /generate endpoint."""

    request_id: str
    generated_text: str
    prompt: str
    usage: "UsageInfo"
    timing: "TimingInfo"


class UsageInfo(BaseModel):
    """Token usage information."""

    prompt_tokens: int
    generated_tokens: int
    total_tokens: int


class TimingInfo(BaseModel):
    """Timing information for the request."""

    time_to_first_token_ms: float
    total_time_ms: float
    tokens_per_second: float


class ModelInfoResponse(BaseModel):
    """Response body for /model/info endpoint."""

    name: str
    parameters: int
    parameters_human: str
    device: str
    dtype: str
    max_sequence_length: int


class HealthResponse(BaseModel):
    """Response body for /health endpoint."""

    status: str
    model_loaded: bool
    engine: str
    uptime_seconds: float


# Needed for forward references
GenerateResponse.model_rebuild()
