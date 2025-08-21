# precision_estimation/core/analyzer/relu_error_analyzer.py
"""
ReLUErrorAnalyzer

Analyzer for ReLU precision errors with element-wise critical-path identification.
"""
from typing import Any, Dict, Optional, Tuple, List
import torch
import torch.nn.functional as F

from core.config.precision_strategy import (
    ulp_like, quantize_to_dtype, demote_with_round, PrecisionStrategy,
)

class ReLUErrorAnalyzer:
    def __init__(self):
        pass

    def analyze(
        self,
        oracle_result: Any,
        x: Optional[torch.Tensor] = None,
        strategy: Optional[PrecisionStrategy] = None,
        top_k: int = 10,
    ) -> Dict[str, Any]:
        """
        Analyze ReLU precision errors
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
            y_ref = F.relu(x.double()).to(dtype=torch.float32)
            
            # Simulate mixed precision
            x_q = quantize_to_dtype(x, strategy.input_dtype).to(dtype=torch.float32)
            x_c = x_q.to(dtype=strategy.compute_dtype)
            y_c = F.relu(x_c)
            y_mixed = demote_with_round(y_c, strategy.output_dtype)

            # Per-element error
            per_elem_err = (y_ref - y_mixed).abs()
            max_err = float(per_elem_err.max().item())
            report["max_element_error"] = max_err

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
                    
                    # Output demotion error
                    y_c_val = float(y_c[coord].item())
                    demote_err = abs(y_c_val - y_mixed_val)
                    
                    # Analyze ReLU behavior
                    relu_behavior = "positive" if x_val > 0 else "negative_to_zero"
                    if abs(x_val) < 1e-6:
                        relu_behavior = "near_zero"
                    
                    elem_record = {
                        "coord": coord,
                        "element_error": elem_error,
                        "x_original": x_val,
                        "x_quantized": x_q_val,
                        "y_ref": y_ref_val,
                        "y_mixed": y_mixed_val,
                        "storage_error": storage_err,
                        "demote_error": demote_err,
                        "relu_behavior": relu_behavior,
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
                
                # Statistics of ReLU behavior
                behaviors = {}
                for e in report["top_elements"]:
                    b = e["relu_behavior"]
                    behaviors[b] = behaviors.get(b, 0) + 1
                report["relu_behavior_stats"] = behaviors
                
                report["suggestion"] = self._suggest_from_relu_analysis(primary_refined, behaviors)
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
        if "output" in primary or "demote" in primary:
            return "Primary error source is output demotion. Suggestions: delay precision reduction or increase output precision; check for precision loss near ReLU boundary."
        if "input" in primary or "storage" in primary:
            return "Primary error source is input storage precision. Suggestions: increase input storage precision or perform input normalization, pay special attention to values close to zero."
        return "Analysis result is unclear; please check input distribution or increase MC sampling."

    def _suggest_from_relu_analysis(self, primary: str, behaviors: Dict[str, int]) -> str:
        """Generate suggestions based on detailed ReLU analysis"""
        base = self._suggest_from_primary(primary)
        
        # Add specific suggestions based on ReLU behavior
        if "near_zero" in behaviors and behaviors["near_zero"] > 0:
            base += " Input values close to zero detected, which are most prone to precision errors in ReLU. Consider adding small bias or using Leaky ReLU."
        
        if "negative_to_zero" in behaviors:
            ratio = behaviors["negative_to_zero"] / sum(behaviors.values())
            if ratio > 0.3:
                base += f" {ratio:.1%} of error elements involve negative values being clipped to zero, potential threshold sensitivity issues."
        
        return base