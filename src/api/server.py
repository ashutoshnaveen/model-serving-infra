"""FastAPI server for model inference.

This is the main entry point for the API. It initializes the model,
creates the inference engine, and exposes HTTP endpoints for generation.

Supports two engine modes:
  - "naive": Single-request, no batching (baseline)
  - "batching": Continuous batching with KV-cache management
"""

import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import PlainTextResponse, StreamingResponse
from loguru import logger

from src.api.metrics import MetricsCollector
from src.api.schemas import (
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    ModelInfoResponse,
    TimingInfo,
    UsageInfo,
)
from src.api.streaming import StreamingGenerator
from src.cache.manager import CacheManager
from src.engine.batching import ContinuousBatchingEngine
from src.engine.naive import GenerationRequest, NaiveEngine
from src.engine.scheduler import RequestScheduler, ScheduledRequest
from src.models.config import ModelConfig, ServerConfig
from src.models.loader import ModelLoader

# Global state
_naive_engine: NaiveEngine = None
_batching_engine: ContinuousBatchingEngine = None
_streaming_gen: StreamingGenerator = None
_loader: ModelLoader = None
_metrics: MetricsCollector = None
_start_time: float = None
_engine_mode: str = "naive"  # "naive" or "batching"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize model and engines on startup."""
    global _naive_engine, _batching_engine, _streaming_gen
    global _loader, _metrics, _start_time, _engine_mode

    _start_time = time.time()
    _metrics = MetricsCollector()
    config = ServerConfig()

    # Engine mode from env var
    _engine_mode = os.environ.get("ENGINE_MODE", "naive")

    # Load model
    _loader = ModelLoader(config.model)
    model, tokenizer = _loader.load()

    # Always create naive engine (used as fallback)
    _naive_engine = NaiveEngine(
        model=model,
        tokenizer=tokenizer,
        device=_loader.device,
    )

    # Create batching engine
    scheduler = RequestScheduler(
        max_queue_size=128,
        scheduling_policy=config.engine.scheduling_policy,
    )
    cache_manager = CacheManager(
        num_blocks=256,
        block_size=config.cache.block_size,
    )
    _batching_engine = ContinuousBatchingEngine(
        model=model,
        tokenizer=tokenizer,
        device=_loader.device,
        scheduler=scheduler,
        cache_manager=cache_manager,
        max_batch_size=config.engine.max_batch_size,
    )

    # Create streaming generator
    _streaming_gen = StreamingGenerator(
        model=model,
        tokenizer=tokenizer,
        device=_loader.device,
    )

    logger.info(f"Server ready (engine_mode={_engine_mode})")
    yield

    logger.info("Shutting down server")


app = FastAPI(
    title="Model Serving Infrastructure",
    description=(
        "A production-quality LLM inference server inspired by vLLM. "
        "Built from scratch to understand efficient model serving."
    ),
    version="0.2.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    engine = _batching_engine if _engine_mode == "batching" else _naive_engine
    return HealthResponse(
        status="healthy",
        model_loaded=engine is not None,
        engine=_engine_mode,
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

    Uses the configured engine mode (naive or batching).
    Set ENGINE_MODE=batching to use continuous batching.
    """
    request_id = str(uuid.uuid4())[:8]

    if _engine_mode == "batching" and not request.stream:
        return await _generate_batching(request, request_id)
    elif request.stream:
        return await _generate_stream(request)
    else:
        return await _generate_naive(request, request_id)


async def _generate_naive(request: GenerateRequest, request_id: str):
    """Generate using the naive (non-batched) engine."""
    if _naive_engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

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
        result = _naive_engine.generate(engine_request)
    except Exception as e:
        _metrics.record_error()
        logger.error(f"Generation failed for request {request_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

    # Record metrics
    _metrics.record_request(
        total_ms=result.total_time * 1000,
        ttft_ms=result.time_to_first_token * 1000,
        tokens_generated=result.num_generated_tokens,
        tokens_per_second=result.tokens_per_second,
    )

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


async def _generate_batching(request: GenerateRequest, request_id: str):
    """Generate using the continuous batching engine."""
    if _batching_engine is None:
        raise HTTPException(status_code=503, detail="Batching engine not initialized")

    sched_request = ScheduledRequest(
        request_id=request_id,
        prompt=request.prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_k=request.top_k,
        top_p=request.top_p,
        repetition_penalty=request.repetition_penalty,
    )

    if not _batching_engine.submit_request(sched_request):
        raise HTTPException(status_code=429, detail="Request queue is full")

    # Run engine until this request completes
    start = time.perf_counter()
    completed = _batching_engine.run_until_empty()
    elapsed = time.perf_counter() - start

    # Find our request's result
    our_result = None
    for req in completed:
        if req.request_id == request_id:
            our_result = req
            break

    if our_result is None or our_result.error:
        _metrics.record_error()
        detail = our_result.error if our_result else "Request not found"
        raise HTTPException(status_code=500, detail=detail)

    tps = our_result.generated_tokens / elapsed if elapsed > 0 else 0

    _metrics.record_request(
        total_ms=elapsed * 1000,
        ttft_ms=0,  # not tracked per-token in batching mode yet
        tokens_generated=our_result.generated_tokens,
        tokens_per_second=tps,
    )

    return GenerateResponse(
        request_id=our_result.request_id,
        generated_text=our_result.output_text,
        prompt=our_result.prompt,
        usage=UsageInfo(
            prompt_tokens=our_result.prompt_tokens,
            generated_tokens=our_result.generated_tokens,
            total_tokens=our_result.prompt_tokens + our_result.generated_tokens,
        ),
        timing=TimingInfo(
            time_to_first_token_ms=0,
            total_time_ms=elapsed * 1000,
            tokens_per_second=tps,
        ),
    )


async def _generate_stream(request: GenerateRequest):
    """Generate with SSE token streaming."""
    if _streaming_gen is None:
        raise HTTPException(status_code=503, detail="Streaming not initialized")

    return StreamingResponse(
        _streaming_gen.generate_stream(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
        ),
        media_type="text/event-stream",
    )


@app.get("/engine/stats")
async def engine_stats():
    """Return engine statistics."""
    if _engine_mode == "batching" and _batching_engine:
        return _batching_engine.get_stats()
    if _naive_engine:
        return _naive_engine.stats
    raise HTTPException(status_code=503, detail="Engine not initialized")


@app.get("/metrics")
async def metrics(format: str = Query("json", enum=["json", "prometheus"])):
    """Return server metrics.

    Supports JSON (default) and Prometheus exposition format.
    """
    if _metrics is None:
        raise HTTPException(status_code=503, detail="Metrics not initialized")

    # Update gauges from engine state
    if _batching_engine:
        _metrics.update_gauges(
            batch_size=_batching_engine.batch_size,
            queue_depth=_batching_engine.scheduler.pending_count,
            cache_utilization=_batching_engine.cache_manager.utilization,
        )

    if format == "prometheus":
        return PlainTextResponse(
            _metrics.format_prometheus(),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )
    return _metrics.get_metrics()
