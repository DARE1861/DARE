# precision_estimation/core/detector/conv2d_error_detector.py
import torch
import torch.nn.functional as F
from typing import Tuple, Dict, Any

from core.config.precision_strategy import (
    PrecisionStrategy, quantize_to_dtype, promote_exact, demote_with_round
)
from core.oracle.conv2d_oracle_mc import DataAwareMCConv2DOracle, OracleResult

class Conv2DErrorDetector:


    def __init__(self, strategy: PrecisionStrategy, conv_params: Dict[str,Any], oracle: DataAwareMCConv2DOracle):
        self.strategy = strategy
        self.params = conv_params
        self.oracle = oracle

    @torch.inference_mode()
    def _actual_run(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        s = self.strategy
        x_q = quantize_to_dtype(x, s.input_dtype)
        w_q = quantize_to_dtype(w, s.weight_dtype)
        x_c = promote_exact(x_q, s.compute_dtype)
        w_c = promote_exact(w_q, s.compute_dtype)
        y_c = F.conv2d(
            x_c, w_c, bias=None,
            stride=self.params.get("stride",1),
            padding=self.params.get("padding",0),
            dilation=self.params.get("dilation",1),
            groups=self.params.get("groups",1),
        )
        y_out = demote_with_round(y_c, s.output_dtype)
        return y_out

    @torch.inference_mode()
    def detect(self, x: torch.Tensor, w: torch.Tensor) -> Tuple[bool, float, float, OracleResult]:
        oracle_result: OracleResult = self.oracle.predict_error_bound(x, w)
        bound = oracle_result.predicted_bound

        y_ref = F.conv2d(x.double(), w.double(),
                         bias=None,
                         stride=self.params.get("stride",1),
                         padding=self.params.get("padding",0),
                         dilation=self.params.get("dilation",1),
                         groups=self.params.get("groups",1)).float()

        y_act = self._actual_run(x, w)
        err = (y_act - y_ref).abs().max().item()
        exceeded = bool(err > bound)
        return exceeded, err, bound, oracle_result
