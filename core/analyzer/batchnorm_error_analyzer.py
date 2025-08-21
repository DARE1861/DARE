# precision_estimation/core/analyzer/batchnorm_error_analyzer.py
"""
BatchNormErrorAnalyzer

Analyzer for BatchNorm precision errors with channel-wise and statistical analysis.
"""
from typing import Any, Dict, Optional, Tuple, List
import math
import torch
import torch.nn.functional as F

from core.config.precision_strategy import (
    ulp_like, quantize_to_dtype, demote_with_round, PrecisionStrategy,
)

class BatchNormErrorAnalyzer:
    def __init__(self):
        pass

    def analyze(
        self,
        oracle_result: Any,
        x: Optional[torch.Tensor] = None,
        weight: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        running_mean: Optional[torch.Tensor] = None,
        running_var: Optional[torch.Tensor] = None,
        strategy: Optional[PrecisionStrategy] = None,
        bn_params: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
    ) -> Dict[str, Any]:
        """
        Analyze BatchNorm precision errors
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
            "channel_analysis": [],
            "batch_statistics": {},
            "suggestion": "",
        }

        if x is None or strategy is None:
            report["suggestion"] = self._suggest_from_primary(primary)
            return report

        # Get BatchNorm parameters
        if bn_params is None:
            bn_params = {"eps": 1e-5, "momentum": 0.1, "training": True}

        x = x.detach().cpu().to(dtype=torch.float32)
        if weight is not None:
            weight = weight.detach().cpu().to(dtype=torch.float32)
        if bias is not None:
            bias = bias.detach().cpu().to(dtype=torch.float32)
        if running_mean is not None:
            running_mean = running_mean.detach().cpu().to(dtype=torch.float32)
        if running_var is not None:
            running_var = running_var.detach().cpu().to(dtype=torch.float32)

        # Compute high-precision reference and mixed-precision output
        with torch.no_grad():
            y_ref = F.batch_norm(
                x.double(),
                running_mean.double() if running_mean is not None else None,
                running_var.double() if running_var is not None else None,
                weight.double() if weight is not None else None,
                bias.double() if bias is not None else None,
                training=bn_params.get("training", True),
                momentum=bn_params.get("momentum", 0.1),
                eps=bn_params.get("eps", 1e-5)
            ).to(dtype=torch.float32)

            # Simulate mixed precision
            x_q = quantize_to_dtype(x, strategy.input_dtype).to(dtype=torch.float32)
            weight_q = quantize_to_dtype(weight, strategy.weight_dtype).to(dtype=torch.float32) if weight is not None else None
            bias_q = quantize_to_dtype(bias, strategy.weight_dtype).to(dtype=torch.float32) if bias is not None else None
            running_mean_q = running_mean if running_mean is not None else None
            running_var_q = running_var if running_var is not None else None

            x_c = x_q.to(dtype=strategy.compute_dtype)
            weight_c = weight_q.to(dtype=strategy.compute_dtype) if weight_q is not None else None
            bias_c = bias_q.to(dtype=strategy.compute_dtype) if bias_q is not None else None
            running_mean_c = running_mean_q.to(dtype=strategy.compute_dtype) if running_mean_q is not None else None
            running_var_c = running_var_q.to(dtype=strategy.compute_dtype) if running_var_q is not None else None

            y_c = F.batch_norm(
                x_c, running_mean_c, running_var_c, weight_c, bias_c,
                training=bn_params.get("training", True),
                momentum=bn_params.get("momentum", 0.1),
                eps=bn_params.get("eps", 1e-5)
            )
            y_mixed = demote_with_round(y_c, strategy.output_dtype)

            # Per-element error
            per_elem_err = (y_ref - y_mixed).abs()
            max_err = float(per_elem_err.max().item())
            report["max_element_error"] = max_err

            # Batch statistics analysis
            N, C, H, W = x.shape
            report["batch_statistics"] = {
                "batch_size": N,
                "num_channels": C,
                "spatial_size": (H, W),
                "input_range": {
                    "min": float(x.min().item()),
                    "max": float(x.max().item()),
                    "mean": float(x.mean().item()),
                    "std": float(x.std().item())
                }
            }

            # Channel-wise error analysis
            if C > 0:
                # Calculate error statistics for each channel
                channel_errors = per_elem_err.mean(dim=(0, 2, 3))  # (C,)
                
                # Calculate input statistics for each channel
                x_channel_stats = {}
                if bn_params.get("training", True):
                    # Training mode: use current batch statistics
                    batch_mean = x.mean(dim=(0, 2, 3))  # (C,)
                    batch_var = x.var(dim=(0, 2, 3), unbiased=False)  # (C,)
                    x_channel_stats = {
                        "batch_mean": batch_mean,
                        "batch_var": batch_var,
                        "batch_std": torch.sqrt(batch_var + bn_params.get("eps", 1e-5))
                    }
                else:
                    # Evaluation mode: use running statistics
                    x_channel_stats = {
                        "running_mean": running_mean if running_mean is not None else torch.zeros(C),
                        "running_var": running_var if running_var is not None else torch.ones(C),
                        "running_std": torch.sqrt((running_var if running_var is not None else torch.ones(C)) + bn_params.get("eps", 1e-5))
                    }

                # Find top-k channels with maximum error
                topk = min(top_k, C)
                if topk > 0:
                    topk_vals, topk_idx = torch.topk(channel_errors, topk)
                    
                    for k_idx, ch_idx in enumerate(topk_idx.tolist()):
                        ch_error = float(channel_errors[ch_idx].item())
                        
                        # Analyze characteristics of this channel
                        ch_input_range = {
                            "min": float(x[:, ch_idx].min().item()),
                            "max": float(x[:, ch_idx].max().item()),
                            "mean": float(x[:, ch_idx].mean().item()),
                            "std": float(x[:, ch_idx].std().item())
                        }
                        
                        # Calculate various error components for this channel
                        if weight is not None:
                            weight_err = abs(float(weight[ch_idx].item()) - float(weight_q[ch_idx].item())) if weight_q is not None else 0.0
                        else:
                            weight_err = 0.0
                            
                        if bias is not None:
                            bias_err = abs(float(bias[ch_idx].item()) - float(bias_q[ch_idx].item())) if bias_q is not None else 0.0
                        else:
                            bias_err = 0.0
                        
                        # Statistical computation error estimation
                        if bn_params.get("training", True) and N > 1:
                            # Estimate numerical instability of statistical computation
                            stats_instability = 1.0 / max(math.sqrt(N), 1.0)  # Instability due to small batch
                            var_magnitude = float(x_channel_stats["batch_var"][ch_idx].item())
                            if var_magnitude < bn_params.get("eps", 1e-5) * 10:
                                stats_instability *= 10  # Instability due to small variance
                        else:
                            stats_instability = 0.0
                        
                        # Analyze numerical features of this channel
                        numerical_features = self._analyze_channel_numerical_features(
                            x[:, ch_idx], bn_params.get("eps", 1e-5)
                        )
                        
                        channel_record = {
                            "channel_index": ch_idx,
                            "channel_error": ch_error,
                            "input_range": ch_input_range,
                            "weight_error": weight_err,
                            "bias_error": bias_err,
                            "stats_instability_estimate": stats_instability,
                            "numerical_features": numerical_features,
                        }
                        
                        # Add statistics information
                        if bn_params.get("training", True):
                            channel_record.update({
                                "batch_mean": float(x_channel_stats["batch_mean"][ch_idx].item()),
                                "batch_var": float(x_channel_stats["batch_var"][ch_idx].item()),
                                "batch_std": float(x_channel_stats["batch_std"][ch_idx].item()),
                            })
                        else:
                            if running_mean is not None and running_var is not None:
                                channel_record.update({
                                    "running_mean": float(running_mean[ch_idx].item()),
                                    "running_var": float(running_var[ch_idx].item()),
                                    "running_std": float(torch.sqrt(running_var[ch_idx] + bn_params.get("eps", 1e-5)).item()),
                                })
                        
                        report["channel_analysis"].append(channel_record)

            # Aggregated analysis
            if report["channel_analysis"]:
                total_weight_err = sum(ch["weight_error"] for ch in report["channel_analysis"])
                total_bias_err = sum(ch["bias_error"] for ch in report["channel_analysis"])
                total_stats_err = sum(ch["stats_instability_estimate"] for ch in report["channel_analysis"])
                
                total_agg = total_weight_err + total_bias_err + total_stats_err + 1e-20
                agg_ratios = {
                    "affine_params": (total_weight_err + total_bias_err) / total_agg,
                    "statistics": total_stats_err / total_agg,
                }
                
                report["aggregated_ratios"] = agg_ratios
                primary_refined = max(agg_ratios.items(), key=lambda kv: kv[1])[0]
                report["primary_source_refined"] = primary_refined
                
                # Statistical numerical features
                feature_stats = {}
                for feature in ["near_zero_variance", "extreme_values", "small_batch", "high_variance"]:
                    count = sum(1 for ch in report["channel_analysis"] if ch["numerical_features"].get(feature, False))
                    feature_stats[feature] = count
                report["numerical_feature_stats"] = feature_stats
                
                report["suggestion"] = self._suggest_from_batchnorm_analysis(primary_refined, feature_stats, N, C)
            else:
                report["suggestion"] = self._suggest_from_primary(primary)

        return report

    def _analyze_channel_numerical_features(self, x_channel: torch.Tensor, eps: float) -> Dict[str, bool]:
        """
        Analyze numerical features of a single channel
        """
        var_val = float(x_channel.var().item())
        mean_val = float(x_channel.mean().item())
        min_val = float(x_channel.min().item())
        max_val = float(x_channel.max().item())
        
        features = {
            "near_zero_variance": var_val < eps * 100,  # Variance close to eps
            "extreme_values": (max_val - min_val) > 1000,  # Large range of extreme values
            "small_batch": x_channel.numel() < 100,  # Few effective samples
            "high_variance": var_val > 10000,  # High variance
            "near_zero_mean": abs(mean_val) < 1e-6,  # Mean close to 0
            "skewed_distribution": abs(mean_val) > 3 * math.sqrt(var_val + eps),  # Skewed distribution
        }
        
        return features

    def _suggest_from_primary(self, primary: Optional[str]) -> str:
        """Generate suggestions based on primary error source"""
        if primary is None:
            return "Unable to determine primary error source; please provide more runtime context for detailed diagnosis."
        if "output" in primary or "storage" in primary:
            return "Primary error source is output storage precision. Suggestions: delay precision reduction or increase output precision; check numerical range of BatchNorm outputs."
        if "input" in primary:
            return "Primary error source is input storage precision. Suggestions: increase input storage precision or perform input normalization preprocessing."
        if "affine" in primary:
            return "Primary error source is learnable parameter (γ,β) precision. Suggestions: maintain affine parameters in high precision (FP32) or use more precise parameter initialization."
        if "statistics" in primary:
            return "Primary error source is statistics computation precision. Suggestions: increase batch size, use more stable statistical algorithms, or consider LayerNorm as alternative."
        return "Analysis result is unclear; please check input distribution or increase MC sampling."

    def _suggest_from_batchnorm_analysis(
        self, 
        primary: str, 
        feature_stats: Dict[str, int], 
        batch_size: int, 
        num_channels: int
    ) -> str:
        """Generate suggestions based on detailed BatchNorm analysis"""
        base = self._suggest_from_primary(primary)
        
        # Add suggestions based on BatchNorm-specific features
        if feature_stats.get("small_batch", 0) > 0:
            base += f" Small batch size detected ({batch_size}), which increases numerical instability in BatchNorm. Consider increasing batch size or using GroupNorm/LayerNorm."
        
        if feature_stats.get("near_zero_variance", 0) > num_channels * 0.2:
            ratio = feature_stats["near_zero_variance"] / num_channels
            base += f" {ratio:.1%} of channels have variance close to eps, risking numerical instability. Consider increasing eps value or checking input data preprocessing."
        
        if feature_stats.get("extreme_values", 0) > 0:
            base += " Extreme values detected, potentially causing unstable statistics computation. Suggest adding input clipping or using more robust normalization methods."
        
        if feature_stats.get("high_variance", 0) > 0:
            base += " High variance channels detected, may require more careful numerical precision management or input scaling."
        
        return base