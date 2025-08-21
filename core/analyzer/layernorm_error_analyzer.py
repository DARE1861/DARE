# precision_estimation/core/analyzer/layernorm_error_analyzer.py
"""
LayerNormErrorAnalyzer

Analyzer for LayerNorm precision errors with element-wise critical-path identification.
"""
from typing import Any, Dict, Optional, Tuple, List
import math
import torch
import torch.nn.functional as F

from core.config.precision_strategy import (
    ulp_like, quantize_to_dtype, demote_with_round, PrecisionStrategy,
)

class LayerNormErrorAnalyzer:
    def __init__(self):
        pass

    def analyze(
        self,
        oracle_result: Any,
        x: Optional[torch.Tensor] = None,
        weight: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        strategy: Optional[PrecisionStrategy] = None,
        normalized_shape: Optional[Tuple[int, ...]] = None,
        eps: float = 1e-5,
        top_k: int = 10,
    ) -> Dict[str, Any]:
        """
        Analyze LayerNorm precision errors
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

        if normalized_shape is None:
            normalized_shape = oracle_result.meta.get("normalized_shape") if hasattr(oracle_result, "meta") else (x.shape[-1],)

        x = x.detach().cpu().to(dtype=torch.float32)
        weight = weight.detach().cpu().to(dtype=torch.float32) if weight is not None else None
        bias = bias.detach().cpu().to(dtype=torch.float32) if bias is not None else None

        # Compute high-precision reference and mixed-precision output
        with torch.no_grad():
            # High-precision reference
            weight_ref = weight.double() if weight is not None else None
            bias_ref = bias.double() if bias is not None else None
            y_ref = F.layer_norm(x.double(), normalized_shape, weight_ref, bias_ref, eps).to(dtype=torch.float32)
            
            # Simulate mixed precision
            x_q = quantize_to_dtype(x, strategy.input_dtype).to(dtype=torch.float32)
            weight_q = quantize_to_dtype(weight, strategy.weight_dtype).to(dtype=torch.float32) if weight is not None else None
            bias_q = quantize_to_dtype(bias, strategy.weight_dtype).to(dtype=torch.float32) if bias is not None else None

            x_c = x_q.to(dtype=strategy.compute_dtype)
            weight_c = weight_q.to(dtype=strategy.compute_dtype) if weight_q is not None else None
            bias_c = bias_q.to(dtype=strategy.compute_dtype) if bias_q is not None else None

            y_c = F.layer_norm(x_c, normalized_shape, weight_c, bias_c, eps)
            y_mixed = demote_with_round(y_c, strategy.output_dtype)

            # Per-element error
            per_elem_err = (y_ref - y_mixed).abs()
            max_err = float(per_elem_err.max().item())
            report["max_element_error"] = max_err

            # Compute statistics for analysis
            normalized_ndim = len(normalized_shape)
            normalized_dims = tuple(range(-normalized_ndim, 0))
            
            mean_ref = x.double().mean(dim=normalized_dims, keepdim=True).float()
            var_ref = ((x.double() - mean_ref.double()) ** 2).mean(dim=normalized_dims, keepdim=True).float()
            std_ref = torch.sqrt(var_ref + eps)
            
            mean_mixed = x_c.mean(dim=normalized_dims, keepdim=True).to(dtype=torch.float32)
            var_mixed = ((x_c - mean_mixed.to(dtype=x_c.dtype)) ** 2).mean(dim=normalized_dims, keepdim=True).to(dtype=torch.float32)
            std_mixed = torch.sqrt(var_mixed + eps)

            # Statistics errors
            mean_error = (mean_ref - mean_mixed).abs()
            var_error = (var_ref - var_mixed).abs()
            std_error = (std_ref - std_mixed).abs()

            report["statistics_errors"] = {
                "mean_error": float(mean_error.max().item()),
                "var_error": float(var_error.max().item()),
                "std_error": float(std_error.max().item()),
            }

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
                    y_ref_val = float(y_ref[coord].item())
                    y_mixed_val = float(y_mixed[coord].item())
                    
                    # Storage quantization error
                    x_q_val = float(x_q[coord].item())
                    storage_err = abs(x_val - x_q_val)
                    
                    # Weight/bias errors (if applicable)
                    weight_err = 0.0
                    bias_err = 0.0
                    if weight is not None and weight_q is not None:
                        # For weights, need to find corresponding coordinates (usually last few dimensions)
                        try:
                            weight_coord = coord[-len(normalized_shape):]
                            weight_err = abs(float(weight[weight_coord].item()) - float(weight_q[weight_coord].item()))
                        except (IndexError, RuntimeError):
                            weight_err = 0.0
                    if bias is not None and bias_q is not None:
                        try:
                            bias_coord = coord[-len(normalized_shape):]
                            bias_err = abs(float(bias[bias_coord].item()) - float(bias_q[bias_coord].item()))
                        except (IndexError, RuntimeError):
                            bias_err = 0.0
                    
                    # Output demotion error
                    y_c_val = float(y_c[coord].item())
                    demote_err = abs(y_c_val - y_mixed_val)
                    
                    # Impact of statistics error on this element
                    # Calculate the normalization group this element belongs to
                    coord_stats = self._get_stats_coord(coord, normalized_dims)
                    try:
                        local_mean_err = float(mean_error[coord_stats].item())
                        local_std_err = float(std_error[coord_stats].item())
                    except (IndexError, RuntimeError):
                        # If coordinate is out of range, use maximum value as conservative estimate
                        local_mean_err = float(mean_error.max().item())
                        local_std_err = float(std_error.max().item())
                    
                    # Analyze LayerNorm behavior characteristics
                    try:
                        norm_behavior = self._analyze_norm_behavior(
                            x_val, float(mean_ref[coord_stats].item()), float(std_ref[coord_stats].item()), eps
                        )
                    except (IndexError, RuntimeError):
                        # If coordinate access fails, use global statistics
                        global_mean = float(mean_ref.mean().item())
                        global_std = float(std_ref.mean().item())
                        norm_behavior = self._analyze_norm_behavior(x_val, global_mean, global_std, eps)
                    
                    elem_record = {
                        "coord": coord,
                        "element_error": elem_error,
                        "x_original": x_val,
                        "x_quantized": x_q_val,
                        "y_ref": y_ref_val,
                        "y_mixed": y_mixed_val,
                        "storage_error": storage_err,
                        "weight_error": weight_err,
                        "bias_error": bias_err,
                        "demote_error": demote_err,
                        "local_mean_error": local_mean_err,
                        "local_std_error": local_std_err,
                        "norm_behavior": norm_behavior,
                        "stats_coord": str(coord_stats),  # Convert to string to avoid serialization issues
                    }
                    
                    report["top_elements"].append(elem_record)

            # Aggregated analysis
            if report["top_elements"]:
                total_storage = sum(e["storage_error"] for e in report["top_elements"])
                total_weight = sum(e["weight_error"] for e in report["top_elements"])
                total_bias = sum(e["bias_error"] for e in report["top_elements"])
                total_demote = sum(e["demote_error"] for e in report["top_elements"])
                total_stats = sum(e["local_mean_error"] + e["local_std_error"] for e in report["top_elements"])
                
                total_agg = total_storage + total_weight + total_bias + total_demote + total_stats + 1e-20
                
                agg_ratios = {
                    "storage": total_storage / total_agg,
                    "weight": total_weight / total_agg,
                    "bias": total_bias / total_agg,
                    "statistics": total_stats / total_agg,
                    "demote": total_demote / total_agg,
                }
                
                report["aggregated_ratios"] = agg_ratios
                primary_refined = max(agg_ratios.items(), key=lambda kv: kv[1])[0]
                report["primary_source_refined"] = primary_refined
                
                # Statistics of LayerNorm behavior
                behaviors = {}
                for e in report["top_elements"]:
                    b = e["norm_behavior"]
                    behaviors[b] = behaviors.get(b, 0) + 1
                report["norm_behavior_stats"] = behaviors
                
                report["suggestion"] = self._suggest_from_layernorm_analysis(primary_refined, behaviors, report["statistics_errors"])
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

    def _get_stats_coord(self, coord: Tuple[int, ...], normalized_dims: Tuple[int, ...]) -> Tuple[int, ...]:
        """Get statistics coordinates (remove normalized dimensions)"""
        # Convert negative dimensions to positive
        ndim = len(coord)
        normalized_dims_positive = tuple(ndim + d if d < 0 else d for d in normalized_dims)
        
        stats_coord = []
        for i, c in enumerate(coord):
            if i not in normalized_dims_positive:
                stats_coord.append(c)
        
        # If all dimensions are normalized, return empty tuple corresponding index
        if not stats_coord:
            return ()
        return tuple(stats_coord)

    def _analyze_norm_behavior(self, x_val: float, mean_val: float, std_val: float, eps: float) -> str:
        """Analyze LayerNorm behavior characteristics"""
        normalized_val = (x_val - mean_val) / (std_val + eps)
        
        if abs(normalized_val) > 3.0:
            return "outlier"  # Outlier
        elif abs(x_val - mean_val) < eps:
            return "near_mean"  # Close to mean
        elif std_val < eps * 10:
            return "small_variance"  # Small variance case
        elif abs(normalized_val) < 0.1:
            return "close_to_mean"  # Close to mean (after normalization)
        else:
            return "normal"  # Normal case

    def _suggest_from_primary(self, primary: Optional[str]) -> str:
        """Generate suggestions based on primary error source"""
        if primary is None:
            return "Unable to determine primary error source; please provide more runtime context for detailed diagnosis."
        if "output" in primary or "demote" in primary:
            return "Primary error source is output demotion. Suggestions: delay precision reduction or increase output precision; check numerical range of LayerNorm outputs."
        if "input" in primary or "storage" in primary:
            return "Primary error source is input storage precision. Suggestions: increase input storage precision or perform input preprocessing, pay special attention to input numerical range and distribution."
        if "statistics" in primary:
            return "Primary error source is statistics computation (mean/variance). Suggestions: use higher precision for statistical computation, or consider using numerically stable LayerNorm implementation."
        if "weight" in primary:
            return "Primary error source is weight parameter storage. Suggestions: maintain weights in high precision (FP32) or use more precise quantization strategies."
        if "bias" in primary:
            return "Primary error source is bias parameter storage. Suggestions: maintain bias in high precision or check bias numerical range."
        return "Analysis result is unclear; please check input distribution or increase MC sampling."

    def _suggest_from_layernorm_analysis(
        self, 
        primary: str, 
        behaviors: Dict[str, int], 
        stats_errors: Dict[str, float]
    ) -> str:
        """Generate suggestions based on detailed LayerNorm analysis"""
        base = self._suggest_from_primary(primary)
        
        # Add specific suggestions based on LayerNorm behavior
        total_behaviors = sum(behaviors.values()) + 1e-6
        
        if "small_variance" in behaviors and behaviors["small_variance"] / total_behaviors > 0.3:
            base += " Small variance detected, may cause division instability. Consider increasing eps value or using RMSNorm."
        
        if "outlier" in behaviors and behaviors["outlier"] / total_behaviors > 0.2:
            base += " Outliers detected, may affect statistics computation. Consider adding gradient clipping or input normalization."
        
        if "near_mean" in behaviors and behaviors["near_mean"] / total_behaviors > 0.4:
            base += " Many elements close to mean, mean computation precision is critical. Consider using Kahan summation for mean calculation."
        
        # Add suggestions based on statistics errors
        if stats_errors["var_error"] > stats_errors["mean_error"] * 2:
            base += " Variance computation error significantly larger than mean error, consider using more stable variance computation methods."
        
        return base