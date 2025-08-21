# precision_estimation/core/detector/layernorm_error_detector.py
import torch
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional

from core.config.precision_strategy import (
    PrecisionStrategy, quantize_to_dtype, promote_exact, demote_with_round
)
from core.oracle.layernorm_oracle_mc import DataAwareMCLayerNormOracle, OracleResult

class LayerNormErrorDetector:

    def __init__(
        self, 
        strategy: PrecisionStrategy, 
        normalized_shape: Tuple[int, ...],
        eps: float,
        elementwise_affine: bool,
        oracle: DataAwareMCLayerNormOracle
    ):
        self.strategy = strategy
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.oracle = oracle

    @torch.inference_mode()
    def _actual_run(
        self, 
        x: torch.Tensor, 
        weight: Optional[torch.Tensor] = None, 
        bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        s = self.strategy
        x_q = quantize_to_dtype(x, s.input_dtype)
        weight_q = quantize_to_dtype(weight, s.weight_dtype) if weight is not None else None
        bias_q = quantize_to_dtype(bias, s.weight_dtype) if bias is not None else None
        
        x_c = promote_exact(x_q, s.compute_dtype)
        weight_c = promote_exact(weight_q, s.compute_dtype) if weight_q is not None else None
        bias_c = promote_exact(bias_q, s.compute_dtype) if bias_q is not None else None
        
        y_c = F.layer_norm(x_c, self.normalized_shape, weight_c, bias_c, self.eps)
        y_out = demote_with_round(y_c, s.output_dtype)
        return y_out

    @torch.inference_mode()
    def detect(
        self, 
        x: torch.Tensor, 
        weight: Optional[torch.Tensor] = None, 
        bias: Optional[torch.Tensor] = None
    ) -> Tuple[bool, float, float, OracleResult]:
        oracle_result: OracleResult = self.oracle.predict_error_bound(x, weight, bias)
        bound = oracle_result.predicted_bound


        weight_ref = weight.double() if weight is not None else None
        bias_ref = bias.double() if bias is not None else None
        y_ref = F.layer_norm(x.double(), self.normalized_shape, weight_ref, bias_ref, self.eps).float()

        y_act = self._actual_run(x, weight, bias)
        err = (y_act - y_ref).abs().max().item()
        exceeded = bool(err > bound)
        return exceeded, err, bound, oracle_result

