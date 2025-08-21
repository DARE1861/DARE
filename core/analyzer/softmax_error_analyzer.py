# precision_estimation/core/analyzer/softmax_error_analyzer.py
"""
SoftmaxErrorAnalyzer

Analyzer for Softmax precision errors with element-wise critical-path identification.
"""
from typing import Any, Dict, Optional, Tuple, List
import math
import torch
import torch.nn.functional as F

from core.config.precision_strategy import (
    ulp_like, quantize_to_dtype, demote_with_round, PrecisionStrategy,
)

class SoftmaxErrorAnalyzer:
    def __init__(self):
        pass

    def analyze(
        self,
        oracle_result: Any,
        x: Optional[torch.Tensor] = None,
        strategy: Optional[PrecisionStrategy] = None,
        dim: int = -1,
        top_k: int = 10,
    ) -> Dict[str, Any]:
        """
        Analyze Softmax precision errors
        """
        # Basic report
        comp_est = oracle_result.component_estimates if hasattr(oracle_result, "component_estimates") else {}
        comp_est = {k: float(v) for k, v in comp_est.items()} if comp_est else {}

        total_pred = sum(comp_est.values()) if comp_est else None
        comp_ratios = {}
        primary = None
        if comp_est and total_pred and total_pred > 0:
            for k, v in comp_est.items():
                comp_ratios[k] = v / total_pred
            primary = max(comp_ratios.items(), key=lambda kv: kv[1])[0]
        elif comp_est:
            comp_ratios = {k: 0.0 for k in comp_est}
            primary = max(comp_est.items(), key=lambda kv: kv[1])[0] if comp_est else None

        report: Dict[str, Any] = {
            "component_estimates": comp_est,
            "component_ratios": comp_ratios,
            "primary_source": primary,
            "top_elements": [],
            "suggestion": "",
        }

        if x is None or strategy is None:
            report["suggestion"] = self._suggest_from_primary(primary)
            return report

        x = x.detach().cpu().to(dtype=torch.float32)

        # Compute high-precision reference and mixed-precision output
        with torch.no_grad():
            y_ref = F.softmax(x.double(), dim=dim).to(dtype=torch.float32)
            
            # Simulate mixed precision
            x_q = quantize_to_dtype(x, strategy.input_dtype).to(dtype=torch.float32)
            x_c = x_q.to(dtype=strategy.compute_dtype)
            y_c = F.softmax(x_c, dim=dim)
            y_mixed = demote_with_round(y_c, strategy.output_dtype)

            # Per-element error
            per_elem_err = (y_ref - y_mixed).abs()
            max_err = float(per_elem_err.max().item())
            report["max_element_error"] = max_err

            # Softmax-specific statistics
            x_stats = {
                "min": float(x.min().item()),
                "max": float(x.max().item()),
                "mean": float(x.mean().item()),
                "std": float(x.std().item()),
                "range": float((x.max() - x.min()).item()),
            }
            report["input_stats"] = x_stats

            # Check numerical stability issues
            x_max = torch.max(x, dim=dim, keepdim=True)[0]
            x_shifted = x - x_max
            exp_vals = torch.exp(x_shifted)
            
            stability_stats = {
                "max_shifted_logit": float(x_shifted.max().item()),
                "min_shifted_logit": float(x_shifted.min().item()),
                "max_exp_val": float(exp_vals.max().item()),
                "min_exp_val": float(exp_vals.min().item()),
                "exp_sum_max": float(torch.sum(exp_vals, dim=dim).max().item()),
                "exp_sum_min": float(torch.sum(exp_vals, dim=dim).min().item()),
            }
            report["stability_stats"] = stability_stats

            # Find top-k elements with maximum error
            flat = per_elem_err.view(-1)
            num_elems = flat.numel()
            topk = min(top_k, num_elems)
            
            if topk > 0:
                topk_vals, topk_idx = torch.topk(flat, topk)
                
                # Convert to coordinates
                coords = []
                for idx in topk_idx.tolist():
                    coord = self._flat_to_coord(idx, x.shape)
                    coords.append(coord)

                # Analyze each top element
                for k_idx, coord in enumerate(coords):
                    elem_error = float(per_elem_err[coord].item())
                    x_val = float(x[coord].item())
                    x_q_val = float(x_q[coord].item())
                    y_ref_val = float(y_ref[coord].item())
                    y_mixed_val = float(y_mixed[coord].item())
                    
                    # Storage quantization error
                    storage_err = abs(x_val - x_q_val)
                    
                    # Output demotion error
                    y_c_val = float(y_c[coord].item())
                    demote_err = abs(y_c_val - y_mixed_val)
                    
                    # Softmax position analysis
                    # Calculate relative position of this location in the softmax dimension
                    softmax_dim_size = x.shape[dim] if dim >= 0 else x.shape[dim + len(x.shape)]
                    coord_list = list(coord)
                    softmax_pos = coord_list[dim if dim >= 0 else dim + len(x.shape)]
                    
                    # Analyze the relationship between this position's logit value and the max
                    x_at_coord = x[coord]
                    x_max_along_dim = torch.max(x.select(dim, softmax_pos) if dim != len(coord)-1 else x[coord[:-1]], dim=0)[0] if len(x.shape) > 1 else x.max()
                    
                    logit_analysis = {
                        "softmax_position": softmax_pos,
                        "softmax_dim_size": softmax_dim_size,
                        "is_max_logit": bool(abs(x_val - float(x_max_along_dim.item())) < 1e-6),
                        "distance_from_max": float(x_val - float(x_max_along_dim.item())),
                    }
                    
                    elem_record = {
                        "coord": coord,
                        "element_error": elem_error,
                        "x_original": x_val,
                        "x_quantized": x_q_val,
                        "y_ref": y_ref_val,
                        "y_mixed": y_mixed_val,
                        "storage_error": storage_err,
                        "demote_error": demote_err,
                        "logit_analysis": logit_analysis,
                    }
                    
                    report["top_elements"].append(elem_record)

            # Aggregated analysis
            if report["top_elements"]:
                total_storage = sum(e["storage_error"] for e in report["top_elements"])
                total_demote = sum(e["demote_error"] for e in report["top_elements"])
                total_agg = total_storage + total_demote + 1e-20
                
                agg_ratios = {
                    "storage": total_storage / total_agg,
                    "demote": total_demote / total_agg,
                }
                
                report["aggregated_ratios"] = agg_ratios
                primary_refined = max(agg_ratios.items(), key=lambda kv: kv[1])[0]
                report["primary_source_refined"] = primary_refined
                
                # Statistics of max logit position error ratio
                max_logit_errors = sum(e["element_error"] for e in report["top_elements"] 
                                     if e["logit_analysis"]["is_max_logit"])
                total_top_errors = sum(e["element_error"] for e in report["top_elements"])
                
                if total_top_errors > 0:
                    report["max_logit_error_ratio"] = max_logit_errors / total_top_errors
                
                report["suggestion"] = self._suggest_from_softmax_analysis(
                    primary_refined, comp_ratios, stability_stats, report.get("max_logit_error_ratio", 0)
                )
            else:
                report["suggestion"] = self._suggest_from_primary(primary)

        return report

    def _flat_to_coord(self, flat_idx: int, shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Convert flat index to multi-dimensional coordinates"""
        coord = []
        remaining = flat_idx
        for dim in reversed(shape):
            coord.append(remaining % dim)
            remaining //= dim
        return tuple(reversed(coord))

    def _suggest_from_primary(self, primary: Optional[str]) -> str:
        """Generate suggestions based on primary error source"""
        if primary is None:
            return "Unable to determine primary error source; please provide more runtime context for detailed diagnosis."
        if "output" in primary or "storage" in primary and "output" in primary:
            return "Primary error source is output storage precision. Suggestions: increase output precision or delay precision reduction until after critical computations."
        if "exponential" in primary:
            return "Primary error source is exponential computation. Suggestions: check input logit value range to avoid large values causing exp precision loss; consider using more stable implementations."
        if "sum" in primary or "accumulation" in primary:
            return "Primary error source is summation accumulation. Suggestions: use Kahan summation or block-wise summation to improve precision; consider increasing accumulation precision."
        if "max" in primary or "subtraction" in primary:
            return "Primary error source is max subtraction stage. Suggestions: check max value computation precision, which is a critical step for numerical stability."
        if "input" in primary:
            return "Primary error source is input storage precision. Suggestions: increase input precision or perform appropriate input preprocessing (such as normalization)."
        return "Analysis result is unclear; please check input distribution or increase MC sampling."

    def _suggest_from_softmax_analysis(
        self, 
        primary: str, 
        comp_ratios: Dict[str, float], 
        stability_stats: Dict[str, float],
        max_logit_error_ratio: float
    ) -> str:
        """Generate suggestions based on detailed Softmax analysis"""
        base = self._suggest_from_primary(primary)
        
        # Add suggestions based on numerical stability statistics
        if stability_stats.get("max_shifted_logit", 0) > 10.0:
            base += " Large shifted logit values detected, potential numerical instability risk."
        
        if stability_stats.get("max_exp_val", 0) > 1e10:
            base += " Extremely large exp values detected, suggest checking input logit range or adjusting numerical stability strategy."
        
        if stability_stats.get("min_exp_val", 1) < 1e-10:
            base += " Extremely small exp values detected, may cause underflow and precision loss."
        
        # Add suggestions based on max logit error ratio
        if max_logit_error_ratio > 0.5:
            base += f" {max_logit_error_ratio:.1%} of top errors come from max logit positions, which typically have the greatest impact on softmax output."
        
        # Provide additional suggestions based on component ratios
        if "exponential" in comp_ratios and comp_ratios["exponential"] > 0.4:
            base += " Exponential computation error dominates, consider using log-space computation or more conservative logit ranges."
        
        return base