# Architecture

## System Overview

```mermaid
graph TB
    subgraph "Client Layer"
        C1[REST Client]
        C2[Streaming Client SSE]
        C3[Prometheus Scraper]
    end

    subgraph "API Layer"
        API[FastAPI Server]
        STR[SSE Streaming Generator]
        MET[Metrics Collector]
    end

    subgraph "Scheduling Layer"
        SCH[Request Scheduler]
        PQ[Priority Queue]
        TO[Timeout Manager]
    end

    subgraph "Engine Layer"
        NE[Naive Engine]
        BE[Continuous Batching Engine]
        QE[INT8 Quantization]
    end

    subgraph "Memory Management"
        CM[Cache Manager]
        BT[Block Tables]
        PB[Physical Block Pool]
        LRU[LRU Eviction]
    end

    subgraph "Model Backend"
        ML[Model Loader]
        HF[HuggingFace Transformers]
        PT[PyTorch]
    end

    C1 -->|POST /generate| API
    C2 -->|POST /generate stream=true| STR
    C3 -->|GET /metrics| MET

    API --> SCH
    STR --> NE

    SCH --> PQ
    SCH --> TO
    SCH --> BE

    BE --> CM
    BE --> PT
    NE --> PT

    CM --> BT
    CM --> PB
    CM --> LRU

    ML --> HF
    HF --> PT

    QE -.->|optional| NE
    QE -.->|optional| BE
```

## Request Flow

```mermaid
sequenceDiagram
    participant Client
    participant API as FastAPI
    participant Scheduler
    participant Engine as Batching Engine
    participant Cache as Cache Manager
    participant Model as PyTorch Model

    Client->>API: POST /generate {prompt, max_tokens}
    API->>Scheduler: Submit ScheduledRequest
    Scheduler->>Scheduler: Enqueue (priority + FCFS)
    
    loop Each Decode Iteration
        Scheduler->>Engine: get_next_batch(max_batch_size)
        Engine->>Cache: allocate_sequence(seq_id, tokens)
        Cache->>Cache: Assign physical blocks
        Engine->>Model: Forward pass (batched)
        Model-->>Engine: Logits [batch, seq, vocab]
        Engine->>Engine: Sample next tokens
        
        alt Sequence finished (EOS or max_tokens)
            Engine->>Cache: free_sequence(seq_id)
            Engine->>Scheduler: complete_request(id, output)
        end
        
        alt New request in queue & batch has capacity
            Engine->>Scheduler: Admit new sequence
        end
    end
    
    API-->>Client: GenerateResponse {text, usage, timing}
```

## KV-Cache Block Management

```mermaid
graph LR
    subgraph "Sequence A (35 tokens, block_size=16)"
        LA0[Logical 0] --> PA5[Physical 5<br/>tokens 0-15]
        LA1[Logical 1] --> PA12[Physical 12<br/>tokens 16-31]
        LA2[Logical 2] --> PA3[Physical 3<br/>tokens 32-34]
    end

    subgraph "Sequence B (20 tokens)"
        LB0[Logical 0] --> PB8[Physical 8<br/>tokens 0-15]
        LB1[Logical 1] --> PB1[Physical 1<br/>tokens 16-19]
    end

    subgraph "Free Pool"
        F1[Physical 0]
        F2[Physical 2]
        F3[Physical 4]
        F4[Physical 6]
        F5[Physical 7]
    end

    style PA5 fill:#2563eb,color:#fff
    style PA12 fill:#2563eb,color:#fff
    style PA3 fill:#2563eb,color:#fff
    style PB8 fill:#7c3aed,color:#fff
    style PB1 fill:#7c3aed,color:#fff
    style F1 fill:#374151,color:#9ca3af
    style F2 fill:#374151,color:#9ca3af
    style F3 fill:#374151,color:#9ca3af
    style F4 fill:#374151,color:#9ca3af
    style F5 fill:#374151,color:#9ca3af
```

**Key insight**: Sequences don't need contiguous memory. Physical blocks are allocated from a free pool and mapped via block tables — exactly like virtual memory pages in an OS. This eliminates fragmentation.

## Continuous Batching vs Static Batching

```
Static Batching:
  Time →  ████████████████████████
  Seq A:  ██████████████████████     (22 tokens)
  Seq B:  ██████████████             (14 tokens, 8 slots wasted)
  Seq C:  ████████                   (8 tokens, 14 slots wasted)
  
  New requests must wait until ALL sequences in the batch finish.

Continuous Batching:
  Time →  ████████████████████████
  Seq A:  ██████████████████████     (22 tokens)
  Seq B:  ██████████████ Seq D: █████████
  Seq C:  ████████ Seq E: ████████████████
  
  Finished sequences are immediately replaced by waiting requests.
  GPU stays fully utilized.
```

## Engine Modes

| Feature | Naive Engine | Batching Engine |
|---------|-------------|-----------------|
| Concurrency | 1 request at a time | Up to max_batch_size |
| KV-Cache | Managed by HuggingFace | Block-level management |
| Scheduling | None (synchronous) | Priority queue + FCFS |
| Memory | Uncontrolled growth | Block allocation + LRU eviction |
| Best for | Debugging, baselines | Production throughput |
