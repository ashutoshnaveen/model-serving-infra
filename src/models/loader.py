"""Model loading and management.

Handles downloading, caching, and loading HuggingFace models
with proper device placement and dtype configuration.
"""

import time
from typing import Optional, Tuple

from loguru import logger

from src.models.config import ModelConfig


class ModelLoader:
    """Loads and manages a HuggingFace causal language model."""

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.model = None
        self.tokenizer = None
        self._device = None
        self._dtype = None

    @property
    def device(self) -> str:
        if self._device is None:
            self._device = self.config.resolve_device()
        return self._device

    @property
    def dtype(self):
        if self._dtype is None:
            self._dtype = self.config.resolve_dtype()
        return self._dtype

    def load(self) -> Tuple:
        """Load model and tokenizer. Downloads on first run.

        Returns:
            Tuple of (model, tokenizer)
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = self.config.name
        logger.info(f"Loading model: {model_name}")
        logger.info(f"Device: {self.device}, Dtype: {self.dtype}")

        start = time.perf_counter()

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        elapsed = time.perf_counter() - start
        param_count = sum(p.numel() for p in self.model.parameters())

        logger.info(
            f"Model loaded in {elapsed:.2f}s | "
            f"Parameters: {param_count / 1e6:.1f}M | "
            f"Device: {self.device}"
        )

        return self.model, self.tokenizer

    def get_model_info(self) -> dict:
        """Return model metadata."""
        if self.model is None:
            return {"status": "not_loaded"}

        param_count = sum(p.numel() for p in self.model.parameters())
        return {
            "name": self.config.name,
            "parameters": param_count,
            "parameters_human": f"{param_count / 1e6:.1f}M",
            "device": str(self.device),
            "dtype": str(self.dtype),
            "max_sequence_length": self.config.max_sequence_length,
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download and cache a model")
    parser.add_argument("--model", type=str, default="gpt2", help="Model name")
    parser.add_argument("--device", type=str, default="auto", help="Device")
    args = parser.parse_args()

    config = ModelConfig(name=args.model, device=args.device)
    loader = ModelLoader(config)
    model, tokenizer = loader.load()
    print(loader.get_model_info())
