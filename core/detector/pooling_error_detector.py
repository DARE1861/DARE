import torch
import torch.nn.functional as F
from typing import Tuple, Dict, Any
from core.config.precision_strategy import PrecisionStrategy
from core.oracle.pooling_oracle_mc import DataAwareMCPoolingOracle, OracleResult

class PoolingErrorDetector:
    def __init__(self, strategy:PrecisionStrategy, pool_params:Dict[str,Any], oracle:DataAwareMCPoolingOracle):
        self.strategy = strategy
        self.pool_params = pool_params
        self.oracle = oracle

    @torch.inference_mode()
    def _actual_run(self, x:torch.Tensor) -> torch.Tensor:
        x_c = x.to(dtype=self.strategy.compute_dtype)
        if self.pool_params["pool_type"]=="max":
            y_c = F.max_pool2d(x_c,kernel_size=self.pool_params["kernel_size"],stride=self.pool_params["stride"])
        elif self.pool_params["pool_type"]=="avg":
            y_c = F.avg_pool2d(x_c,kernel_size=self.pool_params["kernel_size"],stride=self.pool_params["stride"])
        y_out = y_c.to(dtype=self.strategy.output_dtype)
        return y_out

    @torch.inference_mode()
    def detect(self,x:torch.Tensor) -> Tuple[bool,float,float,OracleResult]:
        oracle_result: OracleResult = self.oracle.predict_error_bound(x)
        bound = oracle_result.predicted_bound

        if self.pool_params["pool_type"]=="max":
            y_ref = F.max_pool2d(x.double(),kernel_size=self.pool_params["kernel_size"],stride=self.pool_params["stride"]).float()
        elif self.pool_params["pool_type"]=="avg":
            y_ref = F.avg_pool2d(x.double(),kernel_size=self.pool_params["kernel_size"],stride=self.pool_params["stride"]).float()

        y_act = self._actual_run(x)
        err = (y_act - y_ref).abs().max().item()
        exceeded = bool(err > bound)
        return exceeded, err, bound, oracle_result
