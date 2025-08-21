# precision_estimation/core/detector/batchnorm_error_detector.py
import torch
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional

from core.config.precision_strategy import (
    PrecisionStrategy, quantize_to_dtype, promote_exact, demote_with_round
)
from core.oracle.batchnorm_oracle_mc import DataAwareMCBatchNormOracle, OracleResult

class BatchNormErrorDetector:

    def __init__(self, strategy: PrecisionStrategy, bn_params: Dict[str, Any], oracle: DataAwareMCBatchNormOracle):
        self.strategy = strategy
        self.params = bn_params
        self.oracle = oracle

    @torch.inference_mode()
    def _actual_run(
        self, 
        x: torch.Tensor, 
        weight: Optional[torch.Tensor], 
        bias: Optional[torch.Tensor],
        running_mean: torch.Tensor,
        running_var: torch.Tensor
    ) -> torch.Tensor:
        s = self.strategy
        x_q = quantize_to_dtype(x, s.input_dtype)
        weight_q = quantize_to_dtype(weight, s.weight_dtype) if weight is not None else None
        bias_q = quantize_to_dtype(bias, s.weight_dtype) if bias is not None else None
        
        x_c = promote_exact(x_q, s.compute_dtype)
        weight_c = promote_exact(weight_q, s.compute_dtype) if weight_q is not None else None
        bias_c = promote_exact(bias_q, s.compute_dtype) if bias_q is not None else None
        running_mean_c = promote_exact(running_mean, s.compute_dtype)
        running_var_c = promote_exact(running_var, s.compute_dtype)
        
        y_c = F.batch_norm(
            x_c, running_mean_c, running_var_c, weight_c, bias_c,
            training=self.params.get("training", True),
            momentum=self.params.get("momentum", 0.1),
            eps=self.params.get("eps", 1e-5)
        )
        y_out = demote_with_round(y_c, s.output_dtype)
        return y_out

    @torch.inference_mode()
    def detect(
        self, 
        x: torch.Tensor, 
        weight: Optional[torch.Tensor], 
        bias: Optional[torch.Tensor],
        running_mean: torch.Tensor,
        running_var: torch.Tensor
    ) -> Tuple[bool, float, float, OracleResult]:
        oracle_result: OracleResult = self.oracle.predict_error_bound(x, weight, bias, running_mean, running_var)
        bound = oracle_result.predicted_bound

        y_ref = F.batch_norm(
            x.double(), 
            running_mean.double(), 
            running_var.double(),
            weight.double() if weight is not None else None,
            bias.double() if bias is not None else None,
            training=self.params.get("training", True),
            momentum=self.params.get("momentum", 0.1),
            eps=self.params.get("eps", 1e-5)
        ).float()

        y_act = self._actual_run(x, weight, bias, running_mean, running_var)
        err = (y_act - y_ref).abs().max().item()
        exceeded = bool(err > bound)
        return exceeded, err, bound, oracle_result


