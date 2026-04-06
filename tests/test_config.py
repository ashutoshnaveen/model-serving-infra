"""Tests for model and server configuration."""

from src.models.config import (
    CacheConfig,
    EngineConfig,
    GenerationConfig,
    ModelConfig,
    ServerConfig,
)


def test_model_config_defaults():
    config = ModelConfig()
    assert config.name == "gpt2"
    assert config.device == "auto"
    assert config.dtype == "float32"
    assert config.max_sequence_length == 1024


def test_model_config_resolve_device():
    config = ModelConfig(device="cpu")
    assert config.resolve_device() == "cpu"

    config_auto = ModelConfig(device="auto")
    device = config_auto.resolve_device()
    assert device in ("cpu", "cuda", "mps")


def test_model_config_resolve_dtype():
    import torch

    config = ModelConfig(dtype="float32")
    assert config.resolve_dtype() == torch.float32

    config16 = ModelConfig(dtype="float16")
    assert config16.resolve_dtype() == torch.float16


def test_generation_config_defaults():
    config = GenerationConfig()
    assert config.max_tokens == 256
    assert config.temperature == 1.0
    assert config.top_k == 50
    assert config.top_p == 1.0


def test_engine_config_defaults():
    config = EngineConfig()
    assert config.max_batch_size == 32
    assert config.scheduling_policy == "fcfs"


def test_cache_config_defaults():
    config = CacheConfig()
    assert config.block_size == 16
    assert config.eviction_policy == "lru"


def test_server_config_composition():
    config = ServerConfig()
    assert isinstance(config.model, ModelConfig)
    assert isinstance(config.generation, GenerationConfig)
    assert isinstance(config.engine, EngineConfig)
    assert isinstance(config.cache, CacheConfig)
    assert config.host == "0.0.0.0"
    assert config.port == 8000
