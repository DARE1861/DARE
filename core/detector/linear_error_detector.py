# precision_estimation/core/detector/linear_error_detector.py
import torch
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional

from core.config.precision_strategy import (
    PrecisionStrategy, quantize_to_dtype, promote_exact, demote_with_round
)
from core.oracle.linear_oracle_mc import DataAwareMCLinearOracle, OracleResult

class LinearErrorDetector:

    def __init__(self, strategy: PrecisionStrategy, oracle: DataAwareMCLinearOracle):
        self.strategy = strategy
        self.oracle = oracle

    @torch.inference_mode()
    def _actual_run(self, x: torch.Tensor, w: torch.Tensor, b: Optional[torch.Tensor] = None) -> torch.Tensor:
        s = self.strategy
        x_q = quantize_to_dtype(x, s.input_dtype)
        w_q = quantize_to_dtype(w, s.weight_dtype)
        b_q = quantize_to_dtype(b, s.weight_dtype) if b is not None else None
        
        x_c = promote_exact(x_q, s.compute_dtype)
        w_c = promote_exact(w_q, s.compute_dtype)
        b_c = promote_exact(b_q, s.compute_dtype) if b_q is not None else None
        
        y_c = F.linear(x_c, w_c, b_c)
        y_out = demote_with_round(y_c, s.output_dtype)
        return y_out

    @torch.inference_mode()
    def detect(self, x: torch.Tensor, w: torch.Tensor, b: Optional[torch.Tensor] = None) -> Tuple[bool, float, float, OracleResult]:
        oracle_result: OracleResult = self.oracle.predict_error_bound(x, w, b)
        bound = oracle_result.predicted_bound

        y_ref = F.linear(x.double(), w.double(), b.double() if b is not None else None).float()
        y_act = self._actual_run(x, w, b)
        err = (y_act - y_ref).abs().max().item()
        exceeded = bool(err > bound)
        return exceeded, err, bound, oracle_result


