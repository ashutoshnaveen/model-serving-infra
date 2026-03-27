"""Model and server configuration."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for the model backend."""

    name: str = "gpt2"
    device: str = "auto"  # auto, cpu, cuda, mps
    dtype: str = "float32"  # float32, float16, bfloat16
    max_sequence_length: int = 1024

    def resolve_device(self) -> str:
        """Resolve 'auto' device to the best available hardware."""
        if self.device != "auto":
            return self.device

        import torch

        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def resolve_dtype(self):
        """Resolve dtype string to torch dtype."""
        import torch

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        return dtype_map.get(self.dtype, torch.float32)


@dataclass
class GenerationConfig:
    """Default generation parameters."""

    max_tokens: int = 256
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0
    repetition_penalty: float = 1.0


@dataclass
class EngineConfig:
    """Configuration for the inference engine."""

    max_batch_size: int = 32
    max_sequence_length: int = 1024
    scheduling_policy: str = "fcfs"  # fcfs = first-come-first-served


@dataclass
class CacheConfig:
    """Configuration for KV-cache management."""

    max_cache_size_mb: int = 1024
    block_size: int = 16  # tokens per block (PagedAttention)
    eviction_policy: str = "lru"  # lru, fifo


@dataclass
class ServerConfig:
    """Top-level server configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    host: str = "0.0.0.0"
    port: int = 8000
