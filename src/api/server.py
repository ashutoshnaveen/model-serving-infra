"""FastAPI server for model inference.

This is the main entry point for the API. It initializes the model,
creates the inference engine, and exposes HTTP endpoints for generation.
"""

import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from loguru import logger

from src.api.schemas import (
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    ModelInfoResponse,
    TimingInfo,
    UsageInfo,
)
from src.engine.naive import GenerationRequest, NaiveEngine
from src.models.config import ModelConfig, ServerConfig
from src.models.loader import ModelLoader

# Global state
_engine: NaiveEngine = None
_loader: ModelLoader = None
_start_time: float = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize model and engine on startup."""
    global _engine, _loader, _start_time

    _start_time = time.time()
    config = ServerConfig()

    # Load model
    _loader = ModelLoader(config.model)
    model, tokenizer = _loader.load()

    # Create naive engine (will be replaced with batching engine later)
    _engine = NaiveEngine(
        model=model,
        tokenizer=tokenizer,
        device=_loader.device,
    )

    logger.info("Server ready to accept requests")
    yield

    logger.info("Shutting down server")


app = FastAPI(
    title="Model Serving Infrastructure",
    description=(
        "A production-quality LLM inference server inspired by vLLM. "
        "Built from scratch to understand efficient model serving."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=_engine is not None,
        engine=_engine.stats["engine"] if _engine else "none",
        uptime_seconds=time.time() - _start_time if _start_time else 0,
    )


@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    """Return information about the loaded model."""
    if _loader is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    info = _loader.get_model_info()
    return ModelInfoResponse(**info)


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate text from a prompt.

    This endpoint currently uses the naive (non-batched) engine.
    Each request is processed independently and synchronously.

    Future versions will support:
    - Continuous batching for higher throughput
    - KV-cache reuse across requests
    - Streaming via SSE
    """
    if _engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    request_id = str(uuid.uuid4())[:8]

    # Convert API request to engine request
    engine_request = GenerationRequest(
        request_id=request_id,
        prompt=request.prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_k=request.top_k,
        top_p=request.top_p,
        repetition_penalty=request.repetition_penalty,
    )

    try:
        result = _engine.generate(engine_request)
    except Exception as e:
        logger.error(f"Generation failed for request {request_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

    return GenerateResponse(
        request_id=result.request_id,
        generated_text=result.generated_text,
        prompt=result.prompt,
        usage=UsageInfo(
            prompt_tokens=result.num_prompt_tokens,
            generated_tokens=result.num_generated_tokens,
            total_tokens=result.num_prompt_tokens + result.num_generated_tokens,
        ),
        timing=TimingInfo(
            time_to_first_token_ms=result.time_to_first_token * 1000,
            total_time_ms=result.total_time * 1000,
            tokens_per_second=result.tokens_per_second,
        ),
    )


@app.get("/engine/stats")
async def engine_stats():
    """Return engine statistics."""
    if _engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    return _engine.stats
