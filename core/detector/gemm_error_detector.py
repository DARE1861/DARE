# precision_estimation/core/detector/gemm_error_detector.py
import torch
import torch.nn.functional as F
from typing import Tuple, Dict, Any

from core.config.precision_strategy import (
    PrecisionStrategy, quantize_to_dtype, promote_exact, demote_with_round
)
from core.oracle.gemm_oracle_mc import DataAwareMCGEMMOracle, OracleResult

class GEMMErrorDetector:


    def __init__(self, strategy: PrecisionStrategy, gemm_params: Dict[str, Any], oracle: DataAwareMCGEMMOracle):
        self.strategy = strategy
        self.params = gemm_params
        self.oracle = oracle

    @torch.inference_mode()
    def _actual_run(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        s = self.strategy
        x_q = quantize_to_dtype(x, s.input_dtype)
        w_q = quantize_to_dtype(w, s.weight_dtype)
        x_c = promote_exact(x_q, s.compute_dtype)
        w_c = promote_exact(w_q, s.compute_dtype)
        

        if self.params.get("transpose_a", False):
            x_c = x_c.T
        if self.params.get("transpose_b", False):
            w_c = w_c.T
            
        y_c = torch.mm(x_c, w_c)
        y_out = demote_with_round(y_c, s.output_dtype)
        return y_out

    @torch.inference_mode()
    def detect(self, x: torch.Tensor, w: torch.Tensor) -> Tuple[bool, float, float, OracleResult]:
        oracle_result: OracleResult = self.oracle.predict_error_bound(x, w)
        bound = oracle_result.predicted_bound


        x_ref = x.double()
        w_ref = w.double()
        if self.params.get("transpose_a", False):
            x_ref = x_ref.T
        if self.params.get("transpose_b", False):
            w_ref = w_ref.T
        y_ref = torch.mm(x_ref, w_ref).float()

        y_act = self._actual_run(x, w)
        err = (y_act - y_ref).abs().max().item()
        exceeded = bool(err > bound)
        return exceeded, err, bound, oracle_result
