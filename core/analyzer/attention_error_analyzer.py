# precision_estimation/core/analyzer/attention_error_analyzer.py
"""
AttentionErrorAnalyzer

Analyzer for Attention precision errors with token-wise and head-wise analysis.
"""
from typing import Any, Dict, Optional, Tuple, List
import math
import torch
import torch.nn.functional as F

from core.config.precision_strategy import (
    ulp_like, quantize_to_dtype, demote_with_round, PrecisionStrategy,
)

class AttentionErrorAnalyzer:
    def __init__(self):
        pass

    def analyze(
        self,
        oracle_result: Any,
        q: Optional[torch.Tensor] = None,
        k: Optional[torch.Tensor] = None,
        v: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        strategy: Optional[PrecisionStrategy] = None,
        attention_params: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
    ) -> Dict[str, Any]:
        
        comp_est = oracle_result.component_estimates if hasattr(oracle_result, "component_estimates") else {}
        comp_est = {key: float(val) for key, val in comp_est.items()} if comp_est else {}

        total_pred = sum(comp_est.values()) if comp_est else None
        comp_ratios = {}
        primary = None
        if comp_est and total_pred and total_pred > 0:
            for key, val in comp_est.items():
                comp_ratios[key] = val / total_pred
            primary = max(comp_ratios.items(), key=lambda kv: kv[1])[0]
        elif comp_est:
            comp_ratios = {key: 0.0 for key in comp_est}
            primary = max(comp_est.items(), key=lambda kv: kv[1])[0] if comp_est else None

        report: Dict[str, Any] = {
            "component_estimates": comp_est,
            "component_ratios": comp_ratios,
            "primary_source": primary,
            "top_tokens": [],
            "head_analysis": [],
            "attention_pattern_stats": {},
            "suggestion": "",
        }

        if q is None or k is None or v is None or strategy is None:
            report["suggestion"] = self._suggest_from_primary(primary)
            return report

        q_tensor = q.detach().cpu().to(dtype=torch.float32)
        k_tensor = k.detach().cpu().to(dtype=torch.float32)
        v_tensor = v.detach().cpu().to(dtype=torch.float32)
        if mask is not None:
            mask_tensor = mask.detach().cpu()
        else:
            mask_tensor = None

        if attention_params is None:
            attention_params = {"scale": 1.0 / math.sqrt(q_tensor.shape[-1]), "is_causal": True}

        with torch.no_grad():
            y_ref = self._compute_attention_reference(q_tensor, k_tensor, v_tensor, mask_tensor, attention_params)
            
            y_mixed, intermediate_results = self._compute_attention_mixed_precision_with_intermediates(
                q_tensor, k_tensor, v_tensor, mask_tensor, attention_params, strategy
            )

            per_token_err = (y_ref - y_mixed).abs().mean(dim=-1)  # (B, H, S)
            max_err = float(per_token_err.max().item())
            report["max_token_error"] = max_err

            B, H, S = per_token_err.shape
            flat_err = per_token_err.view(-1)
            topk = min(top_k, flat_err.numel())
            
            if topk > 0:
                topk_vals, topk_idx = torch.topk(flat_err, topk)
                
                for i, idx in enumerate(topk_idx.tolist()):
                    b = idx // (H * S)
                    h = (idx % (H * S)) // S
                    s = idx % S
                    
                    token_error = float(per_token_err[b, h, s].item())
                    
                    attn_weights_ref = intermediate_results["attn_weights_ref"][b, h, s, :]
                    attn_weights_mixed = intermediate_results["attn_weights_mixed"][b, h, s, :]
                    
                    entropy_ref = -torch.sum(attn_weights_ref * torch.log(attn_weights_ref + 1e-12)).item()
                    entropy_mixed = -torch.sum(attn_weights_mixed * torch.log(attn_weights_mixed + 1e-12)).item()
                    
                    max_attn_ref = float(attn_weights_ref.max().item())
                    max_attn_mixed = float(attn_weights_mixed.max().item())
                    
                    attn_weight_error = float((attn_weights_ref - attn_weights_mixed).abs().sum().item())
                    
                    stage_errors = self._analyze_token_stage_errors(
                        q_tensor[b:b+1, h:h+1, s:s+1, :], 
                        k_tensor[b:b+1, h:h+1, :, :], 
                        v_tensor[b:b+1, h:h+1, :, :],
                        mask_tensor[b:b+1, h:h+1, s:s+1, :] if mask_tensor is not None else None,
                        attention_params, strategy, s
                    )
                    
                    token_record = {
                        "batch_idx": b,
                        "head_idx": h,
                        "token_idx": s,
                        "token_error": token_error,
                        "entropy_ref": entropy_ref,
                        "entropy_mixed": entropy_mixed,
                        "max_attention_weight_ref": max_attn_ref,
                        "max_attention_weight_mixed": max_attn_mixed,
                        "attention_weight_error": attn_weight_error,
                        "stage_errors": stage_errors,
                    }
                    
                    report["top_tokens"].append(token_record)

            # Head-wise analysis
            head_errors = per_token_err.mean(dim=(0, 2))  # (H,)
            for h in range(H):
                head_err = float(head_errors[h].item())
                
                attn_weights = intermediate_results["attn_weights_mixed"][:, h, :, :]  # (B, S, S)
                
                avg_entropy = float(-torch.sum(attn_weights * torch.log(attn_weights + 1e-12), dim=-1).mean().item())
                avg_max_weight = float(attn_weights.max(dim=-1)[0].mean().item())
                sparsity = float((attn_weights < 0.01).float().mean().item())
                
                head_record = {
                    "head_idx": h,
                    "avg_error": head_err,
                    "avg_entropy": avg_entropy,
                    "avg_max_weight": avg_max_weight,
                    "sparsity": sparsity,
                }
                
                report["head_analysis"].append(head_record)

            # Global attention pattern statistics
            all_attn_weights = intermediate_results["attn_weights_mixed"]
            report["attention_pattern_stats"] = {
                "global_avg_entropy": float(-torch.sum(all_attn_weights * torch.log(all_attn_weights + 1e-12), dim=-1).mean().item()),
                "global_sparsity": float((all_attn_weights < 0.01).float().mean().item()),
                "max_attention_weight": float(all_attn_weights.max().item()),
                "min_attention_weight": float(all_attn_weights.min().item()),
            }

            # Aggregated stage analysis and refined suggestions
            if report["top_tokens"]:
                stage_totals = {}
                for token in report["top_tokens"]:
                    for stage_name, err in token["stage_errors"].items():
                        stage_totals[stage_name] = stage_totals.get(stage_name, 0.0) + err
                
                total_stage = sum(stage_totals.values()) + 1e-20
                stage_ratios = {stage_name: val / total_stage for stage_name, val in stage_totals.items()}
                
                report["aggregated_stage_ratios"] = stage_ratios
                primary_refined = max(stage_ratios.items(), key=lambda kv: kv[1])[0]
                report["primary_source_refined"] = primary_refined
                
                high_entropy_count = sum(1 for t in report["top_tokens"] if t["entropy_ref"] > 3.0)
                low_entropy_count = sum(1 for t in report["top_tokens"] if t["entropy_ref"] < 1.0)
                
                report["attention_behavior_stats"] = {
                    "high_entropy_tokens": high_entropy_count,
                    "low_entropy_tokens": low_entropy_count,
                    "total_analyzed": len(report["top_tokens"]),
                }
                
                report["suggestion"] = self._suggest_from_attention_analysis(primary_refined, report["attention_behavior_stats"], comp_ratios)
            else:
                report["suggestion"] = self._suggest_from_primary(primary)

        return report

    def _compute_attention_reference(self, q, k, v, mask, params):
        """Compute attention with high precision as reference"""
        q_ref = q.double()
        k_ref = k.double()
        v_ref = v.double()
        
        scale = params.get("scale", 1.0)
        is_causal = params.get("is_causal", True)
        
        scores = torch.matmul(q_ref, k_ref.transpose(-2, -1)) * scale
        if is_causal and mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v_ref)
        
        return output.float()

    def _compute_attention_mixed_precision_with_intermediates(self, q_tensor, k_tensor, v_tensor, mask_tensor, params, strategy):
        """Compute attention with mixed precision and return intermediate results"""
        # Input quantization
        q_q = quantize_to_dtype(q_tensor, strategy.input_dtype).to(dtype=torch.float32)
        k_q = quantize_to_dtype(k_tensor, strategy.input_dtype).to(dtype=torch.float32)
        v_q = quantize_to_dtype(v_tensor, strategy.input_dtype).to(dtype=torch.float32)
        
        # Convert to compute dtype
        q_c = q_q.to(dtype=strategy.compute_dtype)
        k_c = k_q.to(dtype=strategy.compute_dtype)
        v_c = v_q.to(dtype=strategy.compute_dtype)
        
        # Compute attention
        scale = params.get("scale", 1.0)
        is_causal = params.get("is_causal", True)
        
        scores = torch.matmul(q_c, k_c.transpose(-2, -1)) * scale
        if is_causal and mask_tensor is not None:
            scores = scores.masked_fill(~mask_tensor, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v_c)
        
        # Output demotion
        y_mixed = demote_with_round(output, strategy.output_dtype)
        
        # Compute reference attention weights for comparison
        q_ref = q_tensor.double()
        k_ref = k_tensor.double()
        scores_ref = torch.matmul(q_ref, k_ref.transpose(-2, -1)) * scale
        if is_causal and mask_tensor is not None:
            scores_ref = scores_ref.masked_fill(~mask_tensor, float('-inf'))
        attn_weights_ref = F.softmax(scores_ref, dim=-1).float()
        
        intermediates = {
            "attn_weights_ref": attn_weights_ref,
            "attn_weights_mixed": attn_weights.float(),
            "scores_ref": scores_ref.float(),
            "scores_mixed": scores.float(),
        }
        
        return y_mixed, intermediates

    def _analyze_token_stage_errors(self, q_token, k_all, v_all, mask_token, params, strategy, token_pos):
        """Analyze errors at different stages of attention computation for a specific token"""
        scale = params.get("scale", 1.0)
        
        # Estimate stage-wise errors (simplified heuristic)
        q_storage_err = (q_token - quantize_to_dtype(q_token, strategy.input_dtype).to(dtype=torch.float32)).abs().mean().item()
        k_storage_err = (k_all - quantize_to_dtype(k_all, strategy.input_dtype).to(dtype=torch.float32)).abs().mean().item()
        v_storage_err = (v_all - quantize_to_dtype(v_all, strategy.input_dtype).to(dtype=torch.float32)).abs().mean().item()

        return {
            "storage_error": q_storage_err + k_storage_err + v_storage_err,
            "matmul1_error": q_storage_err * k_storage_err * scale * k_all.shape[-2],  # Q@K error scaling
            "softmax_error": 0.1 * q_storage_err,  # Approximate softmax sensitivity
            "matmul2_error": v_storage_err * 0.1,  # Attention@V error
            "demote_error": q_storage_err * 0.01,  # Output demotion error
        }

    def _suggest_from_primary(self, primary: Optional[str]) -> str:
        """Generate suggestions based on primary error source"""
        if primary is None:
            return "Unable to determine primary error source; please provide more runtime context for detailed diagnosis."
        if "output" in primary or "demote" in primary:
            return "Primary error source is output demotion. Suggestions: delay precision reduction or increase output precision; check numerical range of attention outputs."
        if "softmax" in primary:
            return "Primary error source is Softmax numerical instability. Suggestions: use numerically stable softmax implementation; check numerical range of attention scores; consider reducing scaling factor."
        if "matmul" in primary:
            return "Primary error source is matrix multiplication accumulation error. Suggestions: increase computation precision; consider block-wise computation; use more precise GEMM implementation."
        if "input" in primary or "storage" in primary:
            return "Primary error source is input storage precision. Suggestions: increase Q/K/V storage precision; perform input normalization; check output range of embedding layer."
        if "scaling" in primary:
            return "Primary error source is scaling operation. Suggestions: adjust scaling factor; use higher precision for scaling; check numerical range after scaling."
        return "Analysis result is unclear; please check input distribution or increase MC sampling."

    def _suggest_from_attention_analysis(self, primary: str, behavior_stats: Dict, comp_ratios: Dict) -> str:
        """Generate suggestions based on detailed attention analysis"""
        base = self._suggest_from_primary(primary)
        
        # Add specific suggestions based on attention patterns
        total_tokens = behavior_stats["total_analyzed"]
        high_entropy = behavior_stats["high_entropy_tokens"]
        low_entropy = behavior_stats["low_entropy_tokens"]
        
        if low_entropy / total_tokens > 0.5:
            base += " Detected many sharp attention patterns (low entropy), which amplify softmax numerical errors. Consider using attention dropout or temperature scaling."
        
        if high_entropy / total_tokens > 0.3:
            base += " Detected diffuse attention patterns (high entropy), which may indicate information dilution. Check if stronger positional encoding or improved attention mechanism is needed."
        
        if "softmax" in primary:
            base += " Special note: softmax is prone to numerical instability with long sequences and extreme attention weights; recommend using stable softmax implementation."
        
        return base