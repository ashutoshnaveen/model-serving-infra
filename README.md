# Model Serving Infrastructure

A production-quality LLM inference server inspired by [vLLM](https://github.com/vllm-project/vllm), built from scratch to understand the internals of efficient model serving.

## Why This Project?

Large Language Models are expensive to serve. Naive inference wastes GPU memory and compute through:
- **Memory fragmentation** in KV-cache management
- **Static batching** that blocks on the slowest request
- **No request-level scheduling** for concurrent users

This project implements key optimizations from recent research (PagedAttention, continuous batching) to build an efficient inference server, progressing from a naive baseline to a production-grade system.

## Architecture

```
┌─────────────────────────────────────────────┐
│                  API Layer                   │
│            (FastAPI + SSE Streaming)         │
├─────────────────────────────────────────────┤
│               Request Scheduler              │
│        (Priority Queue + Rate Limiting)      │
├─────────────────────────────────────────────┤
│            Continuous Batching Engine         │
│      (Iteration-level batch scheduling)      │
├─────────────────────────────────────────────┤
│              KV-Cache Manager                │
│     (PagedAttention-style block allocation)  │
├─────────────────────────────────────────────┤
│              Model Backend                   │
│     (HuggingFace Transformers + PyTorch)     │
└─────────────────────────────────────────────┘
```

## Features

### Phase 1: Naive Baseline ✅
- Single-request synchronous inference
- FastAPI server with `/generate` endpoint
- GPT-2 (124M) as default model
- Baseline latency and throughput measurement

### Phase 2: KV-Cache & Continuous Batching ✅
- [x] Block-based KV-cache with PagedAttention-style allocation
- [x] LRU cache eviction under memory pressure
- [x] Per-sequence block tables (logical → physical mapping)
- [x] Continuous batching with iteration-level scheduling
- [x] Dynamic request admission at every decode step

### Phase 3: Advanced Features ✅
- [x] Server-Sent Events (SSE) streaming (`stream: true`)
- [x] Priority-based request scheduling (HIGH/NORMAL/LOW)
- [x] Request timeout and cancellation
- [x] Prometheus-compatible metrics endpoint (`/metrics`)
- [x] Dual engine support (naive / batching via `ENGINE_MODE` env var)

### Phase 4: Benchmarking & Deployment ✅
- [x] Engine comparison benchmark (naive vs continuous batching)
- [x] Docker and docker-compose support
- [x] Comprehensive test suite (config, cache, scheduler, API, metrics)

## Quick Start

```bash
# Clone and setup
git clone https://github.com/ashutoshnaveen/model-serving-infra.git
cd model-serving-infra
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Download model (first run only)
python -m src.models.loader --model gpt2

# Start the server (naive mode)
make serve

# Start with continuous batching
ENGINE_MODE=batching make serve

# Generate text
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The future of AI is", "max_tokens": 50}'

# Stream tokens
curl -N -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The future of AI is", "max_tokens": 50, "stream": true}'

# Check metrics
curl http://localhost:8000/metrics
curl http://localhost:8000/metrics?format=prometheus
```

### Docker

```bash
# Build and run (naive mode)
docker-compose up

# Run with batching engine
docker-compose --profile batching up
```

### Benchmarks

```bash
# Run baseline benchmark
python -m benchmarks.run_benchmark

# Compare naive vs batching engine
python -m benchmarks.compare_engines --num-prompts 10 --max-tokens 32 --batch-size 4
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check with engine status |
| GET | `/model/info` | Loaded model metadata |
| POST | `/generate` | Text generation (supports `stream: true`) |
| GET | `/engine/stats` | Engine-specific statistics |
| GET | `/metrics` | Server metrics (JSON or Prometheus format) |

## Project Structure

```
model-serving-infra/
├── src/
│   ├── api/
│   │   ├── server.py      # FastAPI app with all endpoints
│   │   ├── schemas.py     # Pydantic request/response models
│   │   ├── streaming.py   # SSE token streaming
│   │   └── metrics.py     # Prometheus-style metrics collector
│   ├── engine/
│   │   ├── naive.py       # Baseline single-request engine
│   │   ├── batching.py    # Continuous batching engine
│   │   └── scheduler.py   # Priority queue + timeout handling
│   ├── cache/
│   │   ├── block.py       # Physical block and block table structures
│   │   └── manager.py     # Block allocator with LRU eviction
│   ├── models/
│   │   ├── config.py      # Typed config dataclasses
│   │   └── loader.py      # HuggingFace model loader
│   └── utils/
├── tests/
│   ├── test_config.py     # Config dataclass tests
│   ├── test_cache.py      # KV-cache block manager tests
│   ├── test_scheduler.py  # Request scheduler tests
│   ├── test_metrics.py    # Metrics collector tests
│   └── test_api.py        # API endpoint integration tests
├── benchmarks/
│   ├── run_benchmark.py   # Baseline latency/throughput benchmark
│   └── compare_engines.py # Naive vs batching comparison
├── configs/default.yaml   # Default server configuration
├── Dockerfile
├── docker-compose.yml
└── Makefile
```

## Documentation

- **[Architecture](docs/architecture.md)** — System diagrams (Mermaid), request flow, KV-cache visualization
- **[Design Decisions](docs/design-decisions.md)** — Why GPT-2, block-based cache, SSE over WebSockets, etc.

## Key Concepts

### PagedAttention-style Memory Management
Instead of allocating contiguous memory for each sequence's KV-cache, we use fixed-size **blocks** (like OS memory pages). Sequences map logical block indices to physical blocks via a **block table**, enabling non-contiguous allocation and efficient memory sharing.

### Continuous Batching
Unlike static batching (where all sequences must complete before new ones start), continuous batching operates at the **iteration level**. At each decode step, finished sequences leave and new sequences join — maximizing GPU utilization.

### Request Scheduling
Supports FCFS and priority-based scheduling with timeout enforcement. The scheduler decouples request admission from execution, allowing the engine to maintain high throughput under load.

## Key References

- [Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180) (vLLM)
- [Orca: A Distributed Serving System for Transformer-Based Models](https://www.usenix.org/conference/osdi22/presentation/yu)
- [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135)

## License

MIT
