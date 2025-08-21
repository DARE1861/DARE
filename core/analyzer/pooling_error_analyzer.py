# precision_estimation/core/analyzer/pooling_error_analyzer.py
"""
PoolingErrorAnalyzer

Analyzer for Pooling (MaxPool / AvgPool) precision errors with per-pixel critical-path attribution.

Primary API:
    analyzer = PoolingErrorAnalyzer()
    report = analyzer.analyze(oracle_result, x=None, strategy=None, pool_params=None, top_k=5)

- oracle_result: OracleResult produced by DataAwareMCPoolingOracle
- x: optional input tensor (N,C,H,W)
- strategy: optional PrecisionStrategy instance
- pool_params: dict with keys: kernel_size, stride, padding, mode ('max' or 'avg')
- top_k: number of top output pixels to explain
Returns:
    report: dict with keys:
      - component_estimates
      - component_ratios
      - primary_source
      - top_paths: list of {out_coord, pixel_error, y_ref, y_mixed, contributions: [...]}
      - suggestion (str)
"""
from typing import Any, Dict, Optional, Tuple, List
import math
import torch
import torch.nn.functional as F
from core.config.precision_strategy import (
    ulp_like,
    quantize_to_dtype,
    demote_with_round,
    get_precision_strategy,
    PrecisionStrategy,
)

Coord = Tuple[int, int, int, int]  # (n, c, i, j)

class PoolingErrorAnalyzer:
    def __init__(self):
        pass

    def analyze(
        self,
        oracle_result: Any,
        x: Optional[torch.Tensor] = None,
        strategy: Optional[PrecisionStrategy] = None,
        pool_params: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
    ) -> Dict[str, Any]:

        comp_est = getattr(oracle_result, "component_estimates", {}) or {}
        comp_est = {k: float(v) for k, v in comp_est.items()} if comp_est else {}

        total_pred = sum(comp_est.values()) if comp_est else None
        comp_ratios = {}
        primary = None
        if comp_est and total_pred and total_pred > 0:
            comp_ratios = {k: v / total_pred for k, v in comp_est.items()}
            primary = max(comp_ratios.items(), key=lambda kv: kv[1])[0]
        elif comp_est:
            comp_ratios = {k: 0.0 for k in comp_est}
            primary = max(comp_est.items(), key=lambda kv: kv[1])[0] if comp_est else None

        report: Dict[str, Any] = {
            "component_estimates": comp_est,
            "component_ratios": comp_ratios,
            "primary_source": primary,
            "top_paths": [],
            "suggestion": "",
        }

        if x is None or strategy is None:
            report["suggestion"] = self._suggest_from_primary(primary)
            return report

        x = x.detach().cpu().to(dtype=torch.float32)

        if pool_params is None:
            pool_params = getattr(oracle_result, "meta", {}).get("pool_params", None)
        if pool_params is None:
            pool_params = {"kernel_size": 2, "stride": 2, "padding": 0, "mode": "max"}

        kernel_size = pool_params.get("kernel_size", 2)
        stride = pool_params.get("stride", kernel_size)
        padding = pool_params.get("padding", 0)
        mode = pool_params.get("mode", "max").lower()

        # Reference output in high precision
        with torch.no_grad():
            y_ref = self._pool(x.double(), kernel_size, stride, padding, mode).to(dtype=torch.float32)

            # Simulate mixed precision
            x_q = quantize_to_dtype(x, strategy.input_dtype).to(dtype=torch.float32)
            x_c = x_q.to(dtype=strategy.compute_dtype)
            y_c = self._pool(x_c, kernel_size, stride, padding, mode)
            y_mixed = demote_with_round(y_c, strategy.output_dtype)

            per_pixel_err = (y_ref - y_mixed).abs()
            max_err = float(per_pixel_err.max().item())
            report["max_pixel_error"] = max_err
            report["y_ref_stats"] = {
                "min": float(y_ref.min().item()),
                "max": float(y_ref.max().item()),
                "mean": float(y_ref.mean().item()),
            }

            flat = per_pixel_err.view(-1)
            num_pixels = flat.numel()
            topk = min(top_k, num_pixels)
            if topk == 0:
                report["suggestion"] = "No output pixels found for analysis."
                return report

            topk_vals, topk_idx = torch.topk(flat, topk)
            N, C, H_out, W_out = per_pixel_err.shape
            coords: List[Coord] = []
            for idx in topk_idx.tolist():
                n = idx // (C * H_out * W_out)
                rem = idx % (C * H_out * W_out)
                c_idx = rem // (H_out * W_out)
                rem2 = rem % (H_out * W_out)
                i = rem2 // W_out
                j = rem2 % W_out
                coords.append((n, c_idx, i, j))

            for coord in coords:
                n, c_out, i_out, j_out = coord
                receptive = self._get_receptive_indices(i_out, j_out, x.shape, kernel_size, stride, padding)
                x_patch, coords_patch = self._gather_patch(x, n, c_out, receptive)

                num_elems = x_patch.numel()
                if num_elems == 0:
                    continue

                x_q_elem = quantize_to_dtype(x_patch, strategy.input_dtype).to(dtype=torch.float32)
                eps_x = (x_q_elem - x_patch).abs()

                y_pixel_c = self._pool(x_patch.to(dtype=strategy.compute_dtype).view(1, 1, *self._patch_shape(receptive)), 
                                        kernel_size=kernel_size, stride=kernel_size, padding=0, mode=mode).to(dtype=torch.float32).squeeze()
                y_pixel_demoted = demote_with_round(y_pixel_c.unsqueeze(0), strategy.output_dtype).squeeze(0)
                demote_err_pixel = abs(y_pixel_c - y_pixel_demoted).item()

                # accumulation estimate: sqrt(n) heuristic
                ops = num_elems
                ulp_y = ulp_like(y_pixel_c.unsqueeze(0), strategy.compute_dtype).squeeze(0).item()
                accum_total = 0.5 * ulp_y * math.sqrt(max(1, ops))

                # contributions
                linear_term = eps_x  # only input rounding
                accum_share = torch.ones_like(linear_term) * accum_total / max(1, num_elems)
                demote_share_vec = torch.ones_like(linear_term) * demote_err_pixel / max(1, num_elems)
                total_contrib = linear_term + accum_share + demote_share_vec

                contrib_list = []
                for idx_elem, (in_i, in_j) in enumerate(coords_patch):
                    contrib_value = float(total_contrib[idx_elem].item())
                    percent = contrib_value / (float(per_pixel_err[n, c_out, i_out, j_out].item()) + 1e-20)
                    contrib_list.append({
                        "input_coord": (n, c_out, in_i, in_j),
                        "contribution": contrib_value,
                        "contribution_ratio_of_pixel": percent,
                    })

                contrib_list.sort(key=lambda x: x["contribution"], reverse=True)

                path_record = {
                    "out_coord": (n, c_out, i_out, j_out),
                    "pixel_error": float(per_pixel_err[n, c_out, i_out, j_out].item()),
                    "y_ref": float(y_ref[n, c_out, i_out, j_out].item()),
                    "y_mixed": float(y_mixed[n, c_out, i_out, j_out].item()),
                    "demote_error_pixel": demote_err_pixel,
                    "accum_estimate": accum_total,
                    "num_contributing_elements": num_elems,
                    "top_contributors": contrib_list[: min(10, len(contrib_list))],
                }
                report["top_paths"].append(path_record)

            # Aggregate contributions
            agg = {"linear": 0.0, "accum": 0.0, "demote": 0.0}
            for p in report["top_paths"]:
                for c in p["top_contributors"]:
                    agg["linear"] += c["contribution"]  # approximate
            tot_agg = sum(abs(v) for v in agg.values()) + 1e-20
            agg_ratio = {k: float(abs(v) / tot_agg) for k, v in agg.items()}
            report["aggregated_contributions"] = agg
            report["aggregated_ratios"] = agg_ratio
            primary_refined = max(agg_ratio.items(), key=lambda kv: kv[1])[0] if agg_ratio else primary
            report["primary_source_refined"] = primary_refined
            report["suggestion"] = self._suggest_from_aggregated(primary_refined, comp_ratios)

        return report

    # --------- internal helpers ---------
    def _pool(self, x, kernel_size, stride, padding, mode="max"):
        """Perform pooling operation with specified parameters"""
        if mode == "max":
            return F.max_pool2d(x, kernel_size, stride=stride, padding=padding)
        elif mode == "avg":
            return F.avg_pool2d(x, kernel_size, stride=stride, padding=padding)
        else:
            raise ValueError(f"Unsupported pooling mode: {mode}")

    def _suggest_from_primary(self, primary: Optional[str]) -> str:
        """Generate suggestions based on primary error source"""
        if primary is None:
            return "Unable to determine primary error source. Please provide more context for detailed diagnosis."
        if "demote" in primary or "output" in primary:
            return "Primary error source is output demotion. Suggestions: use higher precision output or delay precision reduction."
        if "accum" in primary:
            return "Primary error source is accumulation rounding. Suggestions: increase accumulation precision."
        if "input" in primary or "storage" in primary:
            return "Primary error source is input storage. Suggestions: increase input precision or normalize inputs."
        return "Analysis result is unclear."

    def _suggest_from_aggregated(self, primary: Optional[str], comp_ratios: Dict[str, float]) -> str:
        """Generate suggestions based on aggregated analysis"""
        base = self._suggest_from_primary(primary)
        if comp_ratios:
            sorted_comp = sorted(comp_ratios.items(), key=lambda kv: kv[1], reverse=True)
            top_name, top_ratio = sorted_comp[0]
            if top_ratio > 0.6 and top_name not in primary:
                base += f" Note: Global estimate indicates `{top_name}` also accounts for a significant proportion ({top_ratio:.2%})."
        return base

    def _get_receptive_indices(self, out_i, out_j, x_shape, kernel_size, stride, padding) -> List[Tuple[int, int]]:
        """Get indices of input elements that contribute to a specific output pixel"""
        _, C, H_in, W_in = x_shape
        receptive = []
        for i in range(kernel_size):
            for j in range(kernel_size):
                in_i = out_i * stride + i - padding
                in_j = out_j * stride + j - padding
                if 0 <= in_i < H_in and 0 <= in_j < W_in:
                    receptive.append((in_i, in_j))
        return receptive

    def _gather_patch(self, x, batch_n, c, receptive: List[Tuple[int, int]]):
        """Gather input patch elements for a specific output pixel"""
        elems = []
        coords_patch = []
        for (i, j) in receptive:
            elems.append(x[batch_n, c, i, j].unsqueeze(0))
            coords_patch.append((i, j))
        if len(elems) == 0:
            return torch.tensor([], dtype=torch.float32), coords_patch
        x_patch = torch.cat(elems, dim=0).to(dtype=torch.float32)
        return x_patch, coords_patch

    def _patch_shape(self, receptive: List[Tuple[int, int]]):
        """Calculate patch shape for pooling computation"""
        size = int(math.sqrt(len(receptive)))
        return size, size