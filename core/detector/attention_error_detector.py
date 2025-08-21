# precision_estimation/core/detector/attention_error_detector.py
import torch
import torch.nn.functional as F
from typing import Tuple, Dict, Any

from core.config.precision_strategy import (
    PrecisionStrategy, quantize_to_dtype, promote_exact, demote_with_round
)
from core.oracle.attention_oracle_mc import DataAwareMCAttentionOracle, OracleResult

class AttentionErrorDetector:


    def __init__(self, strategy: PrecisionStrategy, attention_params: Dict[str, Any], oracle: DataAwareMCAttentionOracle):
        self.strategy = strategy
        self.params = attention_params
        self.oracle = oracle

    @torch.inference_mode()
    def _actual_run(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        s = self.strategy
        
        q_q = quantize_to_dtype(q, s.input_dtype)
        k_q = quantize_to_dtype(k, s.input_dtype)
        v_q = quantize_to_dtype(v, s.input_dtype)
        
        q_c = promote_exact(q_q, s.compute_dtype)
        k_c = promote_exact(k_q, s.compute_dtype)
        v_c = promote_exact(v_q, s.compute_dtype)
        

        scale = self.params.get("scale", 1.0)
        is_causal = self.params.get("is_causal", True)
        
        scores = torch.matmul(q_c, k_c.transpose(-2, -1)) * scale
        
        if is_causal:
            scores = scores.masked_fill(~mask, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v_c)
        

        y_out = demote_with_round(output, s.output_dtype)
        return y_out

    @torch.inference_mode()
    def detect(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> Tuple[bool, float, float, OracleResult]:
        oracle_result: OracleResult = self.oracle.predict_error_bound(q, k, v, mask)
        bound = oracle_result.predicted_bound


        scale = self.params.get("scale", 1.0)
        is_causal = self.params.get("is_causal", True)
        
        q_ref = q.double()
        k_ref = k.double()
        v_ref = v.double()
        
        scores_ref = torch.matmul(q_ref, k_ref.transpose(-2, -1)) * scale
        if is_causal:
            scores_ref = scores_ref.masked_fill(~mask, float('-inf'))
        attn_weights_ref = F.softmax(scores_ref, dim=-1)
        y_ref = torch.matmul(attn_weights_ref, v_ref).float()

        y_act = self._actual_run(q, k, v, mask)
        err = (y_act - y_ref).abs().max().item()
        exceeded = bool(err > bound)
        return exceeded, err, bound, oracle_result

