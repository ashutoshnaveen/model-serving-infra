# Design Decisions

This document records the key architectural decisions made during development and the reasoning behind them.

---

## DD-1: GPT-2 as Default Model

**Decision**: Use GPT-2 (124M parameters) as the default model.

**Context**: We needed a model small enough to run on a laptop CPU/MPS but large enough to demonstrate real inference patterns (attention, KV-cache, autoregressive generation).

**Alternatives considered**:
- **TinyLlama (1.1B)**: Better quality outputs but too slow on CPU for rapid iteration. Would need GPU for reasonable dev cycles.
- **DistilGPT-2 (82M)**: Smaller but less representative of production model behavior.
- **GPT-2 Medium (355M)**: Good middle ground but slower dev cycles on CPU.

**Outcome**: GPT-2 124M runs in ~1s per request on M1 Mac CPU, allowing fast iteration while still exercising the full inference pipeline.

---

## DD-2: Block-Based KV-Cache (PagedAttention-style)

**Decision**: Implement KV-cache as fixed-size blocks with a block table per sequence, inspired by vLLM's PagedAttention.

**Context**: Standard KV-cache allocates contiguous memory per sequence up to `max_seq_len`. This causes:
1. Memory waste (short sequences reserve max memory)
2. Fragmentation (completed sequences leave gaps)
3. No sharing between sequences

**Our approach**: Fixed 16-token blocks allocated from a pool. Each sequence has a block table mapping logical indices to physical blocks. LRU eviction frees blocks when memory is full.

**Tradeoff**: More complex memory management code, but eliminates fragmentation and enables fine-grained memory control. The block table indirection adds minimal overhead.

---

## DD-3: Continuous Batching at Iteration Level

**Decision**: Implement iteration-level batching where sequences can join/leave the batch at every decode step.

**Context**: Static batching is simple but wastes GPU compute — the batch is blocked until the longest sequence finishes. For inference servers handling many concurrent requests, this is unacceptable.

**Implementation**: At each `step()`:
1. Run forward pass for all active sequences
2. Remove finished sequences and free their cache blocks
3. Admit new sequences from the queue

**Tradeoff**: Requires padding shorter sequences in the batch (wastes some compute). Production systems like vLLM solve this with custom CUDA kernels; we accept the padding overhead for simplicity.

---

## DD-4: Dual Engine Architecture

**Decision**: Support both naive and batching engines, selected via `ENGINE_MODE` environment variable.

**Reasoning**:
1. The naive engine is essential as a **baseline** for benchmarking
2. Easier to **debug** issues by switching engines
3. **Incremental migration** — start with naive, switch to batching when ready
4. Demonstrates **software engineering maturity** (abstraction, backward compat)

---

## DD-5: FCFS + Priority Scheduling

**Decision**: Default to FCFS (first-come-first-served) with optional priority-based scheduling.

**Context**: Real inference services need to handle different SLA tiers (premium vs standard users). FCFS is fair but doesn't support differentiated service.

**Implementation**: `RequestScheduler` supports both policies. Priority scheduling sorts by `(priority, created_at)` — same priority falls back to FCFS. Timeout enforcement runs before each batch selection.

---

## DD-6: SSE Streaming (not WebSockets)

**Decision**: Use Server-Sent Events (SSE) for token streaming, not WebSockets.

**Reasoning**:
1. **Unidirectional**: Token streaming is server → client only. WebSockets are bidirectional and add unnecessary complexity.
2. **HTTP native**: SSE works over regular HTTP, through proxies and load balancers without special configuration.
3. **Industry standard**: OpenAI, Anthropic, and other LLM APIs use SSE for streaming.
4. **Simple client**: `EventSource` in browsers, `curl -N` in terminals.

---

## DD-7: Dynamic Quantization (INT8)

**Decision**: Implement PyTorch dynamic quantization for INT8 support.

**Alternatives**:
- **Static quantization**: Better accuracy but requires calibration data and is more complex to set up.
- **GPTQ/AWQ**: State-of-the-art weight-only quantization, but requires external libraries and GPU.
- **bitsandbytes**: Good for GPU inference but adds a heavy dependency.

**Our choice**: `torch.quantization.quantize_dynamic` is built into PyTorch, works on CPU, requires no calibration, and clearly demonstrates the memory/latency tradeoff. Sufficient for portfolio demonstration.

---

## DD-8: Prometheus-Compatible Metrics

**Decision**: Implement a custom metrics collector with Prometheus exposition format support.

**Alternatives**:
- **prometheus_client library**: Full-featured but adds a dependency for a simple use case.
- **Custom JSON only**: Simpler but not compatible with standard monitoring stacks.

**Our choice**: Custom collector with both JSON and Prometheus text format. Uses a sliding window for percentile calculation (5-minute default). Thread-safe for concurrent request recording.

---

## DD-9: No External Dependencies for Scheduling

**Decision**: Implement the request scheduler from scratch rather than using Celery, Redis queues, or similar.

**Reasoning**: The goal of this project is to **understand and implement** the internals. Using an off-the-shelf queue would hide the scheduling logic which is a core learning objective. Our scheduler is ~250 lines of Python and demonstrates priority queues, timeout handling, and admission control.
