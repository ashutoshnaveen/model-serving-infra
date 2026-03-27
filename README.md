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

### Phase 2: KV-Cache & Continuous Batching (In Progress)
- [ ] KV-cache storage and reuse
- [ ] Memory tracking per request
- [ ] Cache eviction policies
- [ ] Continuous batching with iteration-level scheduling
- [ ] Dynamic request joining

### Phase 3: Advanced Features
- [ ] Server-Sent Events (SSE) streaming
- [ ] Priority-based request scheduling
- [ ] Request timeout and cancellation
- [ ] INT8 quantization support
- [ ] Prometheus-style metrics (`/metrics`)

### Phase 4: Benchmarking & Docs
- [ ] Benchmark suite (naive vs static vs continuous batching)
- [ ] Performance charts and analysis
- [ ] Architecture decision records
- [ ] Docker support

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

# Start the server
make serve

# Test it
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The future of AI is", "max_tokens": 50}'
```

## Project Structure

```
model-serving-infra/
├── src/
│   ├── api/              # FastAPI routes and middleware
│   ├── engine/           # Core inference engine
│   │   ├── scheduler.py  # Request scheduling
│   │   └── batching.py   # Continuous batching logic
│   ├── cache/            # KV-cache management
│   │   ├── manager.py    # Cache allocation/eviction
│   │   └── paged.py      # PagedAttention implementation
│   ├── models/           # Model loading and config
│   └── utils/            # Shared utilities
├── tests/                # Unit and integration tests
├── benchmarks/           # Performance benchmarking scripts
├── configs/              # Server and model configurations
└── docs/                 # Architecture docs and diagrams
```

## Key References

- [Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180) (vLLM)
- [Orca: A Distributed Serving System for Transformer-Based Models](https://www.usenix.org/conference/osdi22/presentation/yu)
- [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135)

## License

MIT
