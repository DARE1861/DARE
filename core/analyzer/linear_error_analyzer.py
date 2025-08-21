# precision_estimation/core/analyzer/linear_error_analyzer.py
"""
LinearErrorAnalyzer

Analyzer for Linear precision errors with neuron-level critical-path identification.
"""
from typing import Any, Dict, Optional, Tuple, List
import math

import torch
import torch.nn.functional as F

from core.config.precision_strategy import (
    ulp_like, quantize_to_dtype, demote_with_round, PrecisionStrategy,
)

# Helper type for coordinate
NeuronCoord = Tuple[int, ...]  # Output neuron coordinate

class LinearErrorAnalyzer:
    def __init__(self):
        pass

    def analyze(
        self,
        oracle_result: Any,
        x: Optional[torch.Tensor] = None,
        w: Optional[torch.Tensor] = None,
        b: Optional[torch.Tensor] = None,
        strategy: Optional[PrecisionStrategy] = None,
        top_k: int = 10,
    ) -> Dict[str, Any]:
        """
        Analyze Linear precision errors
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
            "top_neurons": [],
            "suggestion": "",
        }

        if x is None or w is None or strategy is None:
            report["suggestion"] = self._suggest_from_primary(primary)
            return report

        x = x.detach().cpu().to(dtype=torch.float32)
        w = w.detach().cpu().to(dtype=torch.float32)
        if b is not None:
            b = b.detach().cpu().to(dtype=torch.float32)

        # Compute high-precision reference and mixed-precision output
        with torch.no_grad():
            y_ref = F.linear(x.double(), w.double(), b.double() if b is not None else None).to(dtype=torch.float32)
            
            # Simulate mixed precision
            x_q = quantize_to_dtype(x, strategy.input_dtype).to(dtype=torch.float32)
            w_q = quantize_to_dtype(w, strategy.weight_dtype).to(dtype=torch.float32)
            b_q = quantize_to_dtype(b, strategy.weight_dtype).to(dtype=torch.float32) if b is not None else None
            
            x_c = x_q.to(dtype=strategy.compute_dtype)
            w_c = w_q.to(dtype=strategy.compute_dtype)
            b_c = b_q.to(dtype=strategy.compute_dtype) if b_q is not None else None
            
            y_c = F.linear(x_c, w_c, b_c)
            y_mixed = demote_with_round(y_c, strategy.output_dtype)

            # Per-neuron error
            per_neuron_err = (y_ref - y_mixed).abs()
            max_err = float(per_neuron_err.max().item())
            report["max_neuron_error"] = max_err
            report["y_ref_stats"] = {
                "min": float(y_ref.min().item()), 
                "max": float(y_ref.max().item()), 
                "mean": float(y_ref.mean().item())
            }

            # Find top-k neurons with maximum error
            flat = per_neuron_err.view(-1)
            num_neurons = flat.numel()
            topk = min(top_k, num_neurons)
            
            if topk > 0:
                topk_vals, topk_idx = torch.topk(flat, topk)
                
                # Convert to coordinates
                coords = []
                for idx in topk_idx.tolist():
                    coord = self._flat_to_coord(idx, y_ref.shape)
                    coords.append(coord)

                # Analyze each top neuron
                for k_idx, coord in enumerate(coords):
                    neuron_error = float(per_neuron_err[coord].item())
                    y_ref_val = float(y_ref[coord].item())
                    y_mixed_val = float(y_mixed[coord].item())
                    
                    # Analyze the computation process of this neuron
                    neuron_analysis = self._analyze_neuron(coord, x, w, b, x_q, w_q, b_q, strategy)
                    
                    neuron_record = {
                        "coord": coord,
                        "neuron_error": neuron_error,
                        "y_ref": y_ref_val,
                        "y_mixed": y_mixed_val,
                        **neuron_analysis
                    }
                    
                    report["top_neurons"].append(neuron_record)

            # Aggregated analysis
            if report["top_neurons"]:
                total_input_storage = sum(n["input_storage_error"] for n in report["top_neurons"])
                total_weight_storage = sum(n["weight_storage_error"] for n in report["top_neurons"])
                total_bias_storage = sum(n.get("bias_storage_error", 0) for n in report["top_neurons"])
                total_accum = sum(n["accumulation_estimate"] for n in report["top_neurons"])
                total_demote = sum(n["demote_error"] for n in report["top_neurons"])
                
                total_agg = total_input_storage + total_weight_storage + total_bias_storage + total_accum + total_demote + 1e-20
                
                agg_ratios = {
                    "input_storage": total_input_storage / total_agg,
                    "weight_storage": total_weight_storage / total_agg,
                    "bias_storage": total_bias_storage / total_agg,
                    "accumulation": total_accum / total_agg,
                    "demote": total_demote / total_agg,
                }
                
                report["aggregated_ratios"] = agg_ratios
                primary_refined = max(agg_ratios.items(), key=lambda kv: kv[1])[0]
                report["primary_source_refined"] = primary_refined
                
                report["suggestion"] = self._suggest_from_linear_analysis(primary_refined, comp_ratios)
            else:
                report["suggestion"] = self._suggest_from_primary(primary)

        return report

    def _analyze_neuron(self, coord: NeuronCoord, x: torch.Tensor, w: torch.Tensor, b: Optional[torch.Tensor],
                       x_q: torch.Tensor, w_q: torch.Tensor, b_q: Optional[torch.Tensor], 
                       strategy: PrecisionStrategy) -> Dict[str, Any]:
        """
        Analyze error sources of a single neuron
        """
        # Determine the weight row corresponding to this neuron
        if len(coord) == 1:  # 1D output (out_features,)
            neuron_idx = coord[0]
            batch_idx = 0
        else:  # 2D output (batch_size, out_features)
            batch_idx, neuron_idx = coord
        
        # Extract relevant weights and inputs
        w_row = w[neuron_idx]  # (in_features,)
        w_row_q = w_q[neuron_idx]
        
        if len(x.shape) == 1:  # 1D input
            x_vec = x
            x_vec_q = x_q
        else:  # 2D input
            x_vec = x[batch_idx]
            x_vec_q = x_q[batch_idx]
        
        # Calculate storage errors
        input_storage_err = (x_vec - x_vec_q).abs().sum().item()
        weight_storage_err = (w_row - w_row_q).abs().sum().item()
        
        bias_storage_err = 0.0
        if b is not None and b_q is not None:
            bias_storage_err = abs(b[neuron_idx].item() - b_q[neuron_idx].item())
        
        # Estimate accumulation error (based on square root scaling of input dimension)
        in_features = w_row.shape[0]
        
        # Compute theoretical output of this neuron (compute precision)
        x_c = x_vec_q.to(dtype=strategy.compute_dtype)
        w_c = w_row_q.to(dtype=strategy.compute_dtype)
        y_neuron_c = torch.dot(x_c, w_c)
        if b_q is not None:
            b_c = b_q[neuron_idx].to(dtype=strategy.compute_dtype)
            y_neuron_c = y_neuron_c + b_c
        
        # ULP estimation
        ulp_y = ulp_like(y_neuron_c.unsqueeze(0), strategy.compute_dtype).squeeze(0).item()
        accum_estimate = 0.5 * ulp_y * math.sqrt(in_features)
        
        # Demotion error
        y_neuron_f32 = y_neuron_c.to(dtype=torch.float32).item()
        y_neuron_demoted = demote_with_round(y_neuron_c.unsqueeze(0), strategy.output_dtype).squeeze(0).item()
        demote_err = abs(y_neuron_f32 - y_neuron_demoted)
        
        return {
            "input_storage_error": input_storage_err,
            "weight_storage_error": weight_storage_err,
            "bias_storage_error": bias_storage_err,
            "accumulation_estimate": accum_estimate,
            "demote_error": demote_err,
            "in_features": in_features,
            "neuron_idx": neuron_idx,
            "batch_idx": batch_idx,
        }

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
            return "Primary error source is output demotion. Suggestions: delay precision reduction or increase output precision; check if higher precision output format can be used."
        if "accumulation" in primary or "accum" in primary:
            return "Primary error source is matrix multiplication accumulation error. Suggestions: use higher precision accumulator (e.g., FP32 computation); consider block-wise computation to reduce single accumulation length; or use numerically stable techniques like Kahan summation."
        if "input" in primary or "storage" in primary:
            return "Primary error source is input storage precision. Suggestions: increase input storage precision or perform input normalization/standardization to reduce numerical range."
        if "weight" in primary:
            return "Primary error source is weight storage precision. Suggestions: maintain weights in high precision (FP32); or use more precise weight quantization strategies (e.g., symmetric quantization, calibration)."
        if "bias" in primary:
            return "Primary error source is bias storage precision. Suggestions: use same or higher precision as weights for bias storage; consider whether bias can be separated from main computation."
        return "Analysis result is unclear; please check input/weight distribution or increase MC sampling."

    def _suggest_from_linear_analysis(self, primary: str, comp_ratios: Dict[str, float]) -> str:
        """Generate suggestions based on detailed linear analysis"""
        base = self._suggest_from_primary(primary)
        
        # Add specific suggestions based on Linear layer characteristics
        if "accumulation" in primary:
            base += " Large input dimensions in Linear layers tend to amplify accumulation errors, pay special attention to layers with large in_features."
        
        if "weight" in primary:
            base += " Storage precision of large weight matrices significantly affects Linear layers, consider weight blocking or mixed precision strategies."
            
        # If global estimates are inconsistent with local analysis, add reminder
        if comp_ratios:
            sorted_comp = sorted(comp_ratios.items(), key=lambda kv: kv[1], reverse=True)
            top_name, top_ratio = sorted_comp[0]
            if top_ratio > 0.6 and top_name.replace("_error", "").replace("_storage", "") not in primary:
                base += f" Note: Global estimate indicates `{top_name}` also accounts for a significant proportion ({top_ratio:.2%})."
        
        return base