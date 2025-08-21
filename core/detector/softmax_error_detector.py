
# precision_estimation/core/detector/softmax_error_detector.py
import torch
import torch.nn.functional as F
from typing import Tuple, Dict, Any

from core.config.precision_strategy import (
    PrecisionStrategy, quantize_to_dtype, promote_exact, demote_with_round
)
from core.oracle.softmax_oracle_mc import DataAwareMCSoftmaxOracle, OracleResult

class SoftmaxErrorDetector:


    def __init__(self, strategy: PrecisionStrategy, dim: int, oracle: DataAwareMCSoftmaxOracle):
        self.strategy = strategy
        self.dim = dim
        self.oracle = oracle

    @torch.inference_mode()
    def _actual_run(self, x: torch.Tensor) -> torch.Tensor:
        s = self.strategy
        x_q = quantize_to_dtype(x, s.input_dtype)
        x_c = promote_exact(x_q, s.compute_dtype)
        y_c = F.softmax(x_c, dim=self.dim)
        y_out = demote_with_round(y_c, s.output_dtype)
        return y_out

    @torch.inference_mode()
    def detect(self, x: torch.Tensor) -> Tuple[bool, float, float, OracleResult]:
        oracle_result: OracleResult = self.oracle.predict_error_bound(x)
        bound = oracle_result.predicted_bound

        y_ref = F.softmax(x.double(), dim=self.dim).float()
        y_act = self._actual_run(x)
        err = (y_act - y_ref).abs().max().item()
        exceeded = bool(err > bound)
        return exceeded, err, bound, oracle_result

