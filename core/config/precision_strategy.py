# precision_estimation/core/config/precision_strategy.py
"""
Precision strategy utilities for precision_estimation project.

Provides:
- PrecisionStrategy dataclass
- get_precision_strategy(name)
- ulp_like(x, dtype) : returns element-wise ULP estimate with same shape as x (robust across all devices/dtypes)
- quantize_to_dtype(x, dtype) : simulates tensor quantization to given dtype (with IEEE round-to-nearest)
- promote_exact(x, to_dtype) : exact promotion (low->high precision exact conversion)
- demote_with_round(x, to_dtype) : simulates precision demotion and returns demoted value (with rounding error estimate)

Implementation considerations:
- torch.nextafter is unavailable for Half on CUDA in some PyTorch versions; use fallback for compatibility.
- For float16/bfloat16 etc., use relative eps-based estimation: ulp(x) ≈ |x| * eps(dtype)
  and use dtype.tiny (smallest subnormal value) as lower bound for values near 0 to ensure numerical stability.
"""
from dataclasses import dataclass
from typing import Tuple, Dict, Any

import torch
import math


# -------------------------
# Precision strategy type
# -------------------------
@dataclass
class PrecisionStrategy:
    name: str
    input_dtype: torch.dtype
    weight_dtype: torch.dtype
    compute_dtype: torch.dtype
    output_dtype: torch.dtype

    def __repr__(self):
        return f"PrecisionStrategy(name='{self.name}', input_dtype={self.input_dtype}, weight_dtype={self.weight_dtype}, compute_dtype={self.compute_dtype}, output_dtype={self.output_dtype})"


# -------------------------
# Common strategy factory
# -------------------------
def get_precision_strategy(name: str) -> PrecisionStrategy:
    """
    Returns common mixed-precision strategies.
    Supported names:
      - 'FP32'                    : all FP32
      - 'FP16_all'                : all FP16
      - 'FP16_compute_FP32_accum' : FP16 compute, FP32 accumulation (example)
      - 'FP16_input_FP32_weight_FP32_compute_accum' : your default experimental setting
      - 'BF16_compute'            : BF16 compute
    """
    name = name.strip()
    mapping = {
        'FP32': PrecisionStrategy('FP32', torch.float32, torch.float32, torch.float32, torch.float32),
        'FP16_all': PrecisionStrategy('FP16_all', torch.float16, torch.float16, torch.float16, torch.float16),
        'FP16_compute_FP32_accum': PrecisionStrategy('FP16_compute_FP32_accum', torch.float16, torch.float16, torch.float16, torch.float16),
        # Your default experiment (input FP16, weight FP32, compute FP32, output FP16)
        'FP16_input_FP32_weight_FP32_compute_accum': PrecisionStrategy(
            'FP16_input_FP32_weight_FP32_compute_accum',
            torch.float16, torch.float32, torch.float32, torch.float16
        ),
        'BF16_compute': PrecisionStrategy('BF16_compute', torch.bfloat16, torch.bfloat16, torch.bfloat16, torch.bfloat16),
    }

    if name in mapping:
        return mapping[name]
    else:
        raise ValueError(f"Unknown precision strategy {name}")


# -------------------------
# ULP estimation function (robust implementation)
# -------------------------
def _finfo_for_dtype(dtype: torch.dtype) -> torch.finfo:
    # Raise error for non-float types
    if dtype not in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
        raise ValueError(f"ulp_like only supports float dtypes, got {dtype}")
    return torch.finfo(dtype)


def ulp_like(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """
    Returns element-wise ULP estimate (tensor) with same shape as x.
    Implementation strategy:
      1) For dtypes that can safely call torch.nextafter (float32/64 available on mainstream cuda/cpu versions),
         try using nextafter to compute more accurate differences
      2) If nextafter is unavailable for the dtype on current device (e.g. float16 on some CUDA versions),
         use fallback: ulp(x) ≈ max( |x| * eps(dtype), tiny(dtype) )
      3) For positions where x is 0, return tiny(dtype) (smallest subnormal absolute value) to avoid 0
    Note: Return value is on x.device (if x is on GPU), caller doesn't need additional .to(device)
    """
    if not torch.is_tensor(x):
        raise ValueError("ulp_like requires a tensor x")

    target_device = x.device
    finfo = _finfo_for_dtype(dtype)
    eps = finfo.eps
    tiny = finfo.tiny

    # working tensor on same device/shape for broadcasting
    x_tgt = x.to(dtype=torch.float32) if dtype == torch.float16 or dtype == torch.bfloat16 else x.to(dtype=dtype)

    # try to use nextafter when supported for the dtype on that device
    try:
        # nextafter needs both operands same dtype; create +inf tensor in that dtype/device
        if dtype in (torch.float32, torch.float64):
            x_in_dtype = x.to(dtype=dtype)
            inf_tensor = torch.full_like(x_in_dtype, float('inf'), dtype=dtype, device=target_device)
            # torch.nextafter may throw if unsupported; wrap in try
            next_vals = torch.nextafter(x_in_dtype, inf_tensor)
            ulp = (next_vals - x_in_dtype).abs()
            # protect extremely small values: ensure at least tiny
            ulp = torch.clamp(ulp, min=torch.tensor(tiny, device=target_device, dtype=ulp.dtype))
            return ulp.to(device=target_device)
        else:
            # float16 / bfloat16: some platforms / cuda versions don't implement nextafter_cuda for half
            # fallback to relative eps estimate
            raise RuntimeError("fallback to relative estimation for half/bfloat")
    except Exception:
        # fallback: relative eps * |x| with floor at tiny
        # Use float32 arithmetic for stability then cast to target device/dtype
        with torch.no_grad():
            abs_x = x.abs().to(dtype=torch.float32, device=target_device)
            ulp_est = abs_x * float(eps)  # relative measure
            # values near zero: use tiny
            tiny_tensor = torch.full_like(ulp_est, float(tiny), device=target_device, dtype=ulp_est.dtype)
            ulp_est = torch.maximum(ulp_est, tiny_tensor)
            # cast to requested dtype if needed (we return float32 tensor for numerical operations)
            return ulp_est.to(device=target_device)


# -------------------------
# Quantization / promotion / demotion simulation tools
# -------------------------
def quantize_to_dtype(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """
    Quantize tensor x to dtype and return to float32 (simulates storing to that dtype then reading back to compute width).
    For example: quantizing FP32->FP16 then returning FP32 represents that storage error has been introduced 
    but subsequent computation proceeds in FP32.
    Uses PyTorch's dtype cast to approximate IEEE rounding.
    """
    if dtype == x.dtype:
        return x.clone()
    # cast to dtype then back to float32 for further processing
    quant = x.to(dtype=dtype)
    # return in float32 for consistent downstream arithmetic
    return quant.to(dtype=torch.float32)


def promote_exact(x: torch.Tensor, to_dtype: torch.dtype) -> torch.Tensor:
    """
    Exact promotion (low->high): low precision to high precision conversion is exact per IEEE (no additional error).
    Therefore direct cast is sufficient.
    """
    return x.to(dtype=to_dtype)


def demote_with_round(x: torch.Tensor, to_dtype: torch.dtype) -> torch.Tensor:
    """
    Simulate precision demotion (e.g. FP32 -> FP16) and return demoted tensor (with rounding error).
    Uses PyTorch's cast (default rounding) directly, and returns float32 for subsequent comparison.
    """
    dem = x.to(dtype=to_dtype)
    return dem.to(dtype=torch.float32)


# -------------------------
# Other convenience functions (examples)
# -------------------------
def ulp_scalar(value: float, dtype: torch.dtype) -> float:
    """
    Estimate ULP for a single float value (for small examples/testing).
    """
    finfo = _finfo_for_dtype(dtype)
    if value == 0.0:
        return float(finfo.tiny)
    else:
        return max(abs(value) * finfo.eps, float(finfo.tiny))