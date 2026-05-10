"""INT8 dynamic quantization support for inference.

Quantization reduces model memory footprint and speeds up inference
by converting FP32 weights to lower precision (INT8).

Two approaches:
  1. **Dynamic quantization** (this file): Quantizes weights at load time.
     Simple to apply, no calibration data needed. Best for CPU inference.

  2. **Static quantization**: Requires calibration dataset to determine
     optimal scale factors. Better accuracy but more complex setup.

We implement dynamic quantization as it's the simplest path to
demonstrating latency/memory tradeoffs for the portfolio project.

Usage:
    from src.engine.quantization import quantize_model, get_model_size_mb

    model_int8 = quantize_model(model)
    print(f"FP32: {get_model_size_mb(model):.1f} MB")
    print(f"INT8: {get_model_size_mb(model_int8):.1f} MB")
"""

import copy
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from loguru import logger


@dataclass
class QuantizationResult:
    """Result of quantizing a model."""
    original_size_mb: float
    quantized_size_mb: float
    compression_ratio: float
    quantization_time_ms: float
    dtype: str


def get_model_size_mb(model: nn.Module) -> float:
    """Calculate model size in megabytes from parameter storage."""
    total_bytes = 0
    for param in model.parameters():
        total_bytes += param.nelement() * param.element_size()
    for buf in model.buffers():
        total_bytes += buf.nelement() * buf.element_size()
    return total_bytes / (1024 * 1024)


def quantize_model(
    model: nn.Module,
    dtype: str = "int8",
    inplace: bool = False,
) -> Tuple[nn.Module, QuantizationResult]:
    """Apply dynamic quantization to a model.

    Quantizes nn.Linear layers (which dominate transformer models)
    to INT8, reducing memory by ~4x for those layers.

    Args:
        model: The PyTorch model to quantize.
        dtype: Quantization dtype ("int8" supported).
        inplace: If False, quantize a copy of the model.

    Returns:
        (quantized_model, QuantizationResult)
    """
    if dtype != "int8":
        raise ValueError(f"Unsupported quantization dtype: {dtype}. Use 'int8'.")

    original_size = get_model_size_mb(model)
    logger.info(f"Original model size: {original_size:.1f} MB")

    start = time.perf_counter()

    if not inplace:
        model = copy.deepcopy(model)

    # Move to CPU for quantization (PyTorch dynamic quant is CPU-only)
    model = model.cpu()

    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear},  # quantize all Linear layers
        dtype=torch.qint8,
    )

    elapsed_ms = (time.perf_counter() - start) * 1000
    quantized_size = get_model_size_mb(quantized_model)
    ratio = original_size / quantized_size if quantized_size > 0 else 0

    result = QuantizationResult(
        original_size_mb=round(original_size, 2),
        quantized_size_mb=round(quantized_size, 2),
        compression_ratio=round(ratio, 2),
        quantization_time_ms=round(elapsed_ms, 1),
        dtype=dtype,
    )

    logger.info(
        f"Quantized to INT8: {quantized_size:.1f} MB "
        f"({ratio:.1f}x compression, {elapsed_ms:.0f}ms)"
    )

    return quantized_model, result


def compare_inference(
    model_fp32: nn.Module,
    model_int8: nn.Module,
    tokenizer,
    prompt: str = "The future of artificial intelligence is",
    max_tokens: int = 50,
    num_runs: int = 3,
) -> dict:
    """Compare FP32 vs INT8 inference speed and output quality.

    Runs generation on both models and reports timing differences.
    Both models must be on CPU for fair comparison.

    Args:
        model_fp32: Original FP32 model.
        model_int8: INT8 quantized model.
        tokenizer: Tokenizer for both models.
        prompt: Test prompt.
        max_tokens: Tokens to generate per run.
        num_runs: Number of runs for averaging.

    Returns:
        Comparison dict with timings and outputs.
    """
    inputs = tokenizer(prompt, return_tensors="pt")

    def _time_generation(model, label):
        model = model.cpu()
        times = []
        output_text = ""
        for _ in range(num_runs):
            start = time.perf_counter()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,  # greedy for reproducibility
                )
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        avg_time = sum(times) / len(times)
        return avg_time, output_text

    fp32_time, fp32_output = _time_generation(model_fp32, "FP32")
    int8_time, int8_output = _time_generation(model_int8, "INT8")

    speedup = fp32_time / int8_time if int8_time > 0 else 0

    return {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "num_runs": num_runs,
        "fp32": {
            "avg_time_ms": round(fp32_time * 1000, 1),
            "output": fp32_output,
            "size_mb": get_model_size_mb(model_fp32),
        },
        "int8": {
            "avg_time_ms": round(int8_time * 1000, 1),
            "output": int8_output,
            "size_mb": get_model_size_mb(model_int8),
        },
        "speedup": round(speedup, 2),
    }
