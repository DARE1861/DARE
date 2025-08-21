# precision_estimation/core/analyzer/conv2d_error_analyzer.py
"""
Conv2DErrorAnalyzer

Analyzer for Conv2D precision errors with attention-driven critical-path identification.

Primary API:
    analyzer = Conv2DErrorAnalyzer()
    report = analyzer.analyze(oracle_result, x=None, w=None, strategy=None, conv_params=None, top_k=5)

- oracle_result: OracleResult produced by DataAwareMCConv2DOracle
- x: optional input tensor (N,C,H,W) used for per-pixel analysis
- w: optional weight tensor (C_out,C_in,K_h,K_w)
- strategy: optional PrecisionStrategy instance (used to simulate mixed-precision path)
- conv_params: optional dict with keys stride,padding,dilation,groups (if None, try oracle_result.meta)
- top_k: how many top output pixels to explain
Returns:
    report: dict with keys:
      - component_estimates
      - component_ratios
      - primary_source
      - top_paths: list of {out_coord, pixel_error, y_ref, y_mixed, demote_error, accumulation_est, contributions: [...]}
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

# Helper type for coordinate
Coord = Tuple[int, int, int, int]  # (n, out_c, i, j)


class Conv2DErrorAnalyzer:
    def __init__(self):
        pass

    def analyze(
        self,
        oracle_result: Any,
        x: Optional[torch.Tensor] = None,
        w: Optional[torch.Tensor] = None,
        strategy: Optional[PrecisionStrategy] = None,
        conv_params: Optional[Dict[str, int]] = None,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Perform analysis.

        If x and w and strategy provided, will run per-pixel fine-grained causal attribution.
        Otherwise returns a high-level report based on oracle_result.component_estimates.
        """
        # Basic report from component estimates (always available if OracleResult returned them)
        comp_est = oracle_result.component_estimates if hasattr(oracle_result, "component_estimates") else {}
        # Normalize to float
        comp_est = {k: float(v) for k, v in comp_est.items()} if comp_est else {}

        total_pred = sum(comp_est.values()) if comp_est else None
        comp_ratios = {}
        primary = None
        if comp_est and total_pred and total_pred > 0:
            for k, v in comp_est.items():
                comp_ratios[k] = v / total_pred
            # primary source
            primary = max(comp_ratios.items(), key=lambda kv: kv[1])[0]
        elif comp_est:
            # fallback if total_pred zero
            comp_ratios = {k: 0.0 for k in comp_est}
            primary = max(comp_est.items(), key=lambda kv: kv[1])[0] if comp_est else None

        report: Dict[str, Any] = {
            "component_estimates": comp_est,
            "component_ratios": comp_ratios,
            "primary_source": primary,
            "top_paths": [],
            "suggestion": "",
        }

        # If tensors not provided, return high-level report
        if x is None or w is None or strategy is None:
            # Provide generic suggestion based on primary source
            report["suggestion"] = self._suggest_from_primary(primary)
            return report

        # Ensure tensors are float32 and on CPU for analysis (we will move to device if needed)
        x = x.detach().cpu().to(dtype=torch.float32)
        w = w.detach().cpu().to(dtype=torch.float32)

        # Conv params
        if conv_params is None:
            conv_params = oracle_result.meta.get("conv_params") if getattr(oracle_result, "meta", None) else None
        if conv_params is None:
            # fallback to defaults
            conv_params = {"stride": 1, "padding": 0, "dilation": 1, "groups": 1}

        stride = conv_params.get("stride", 1)
        padding = conv_params.get("padding", 0)
        dilation = conv_params.get("dilation", 1)
        groups = conv_params.get("groups", 1)

        # Compute high-precision reference and mixed-precision outputs for per-pixel errors
        # y_ref: FP64 conv -> float32
        with torch.no_grad():
            device = x.device
            # compute reference with double precision
            y_ref = F.conv2d(x.double(), w.double(), bias=None,
                             stride=stride, padding=padding, dilation=dilation, groups=groups).to(dtype=torch.float32)

            # Simulate mixed precision according to strategy
            # Step 1: quantize storage for input and weight (simulate storage rounding)
            x_q = quantize_to_dtype(x, strategy.input_dtype).to(dtype=torch.float32)
            w_q = quantize_to_dtype(w, strategy.weight_dtype).to(dtype=torch.float32)

            # Step 2: promote to compute dtype and compute conv
            x_c = x_q.to(dtype=strategy.compute_dtype)
            w_c = w_q.to(dtype=strategy.compute_dtype)

            y_c = F.conv2d(x_c, w_c, bias=None, stride=stride, padding=padding, dilation=dilation, groups=groups)

            # Step 3: demote to output dtype (simulate output storage)
            y_mixed = demote_with_round(y_c, strategy.output_dtype)  # returns float32 representation

            # Per-pixel absolute errors
            per_pixel_err = (y_ref - y_mixed).abs()  # shape (N, C_out, H_out, W_out)

            # Global diagnostics
            max_err = float(per_pixel_err.max().item())
            report["max_pixel_error"] = max_err
            report["y_ref_stats"] = {
                "min": float(y_ref.min().item()), "max": float(y_ref.max().item()), "mean": float(y_ref.mean().item())
            }

            # Find top-k output pixels by error (return their coordinates)
            flat = per_pixel_err.view(-1)
            num_pixels = flat.numel()
            topk = min(top_k, num_pixels)
            if topk == 0:
                report["suggestion"] = "No output pixels found for analysis (empty output)."
                return report

            # Get topk indices and coords
            topk_vals, topk_idx = torch.topk(flat, topk)
            # Convert indices to coordinates
            out_shape = per_pixel_err.shape  # (N, C_out, H_out, W_out)
            N, C_out, H_out, W_out = out_shape
            coords: List[Coord] = []
            for idx in topk_idx.tolist():
                n = idx // (C_out * H_out * W_out)
                rem = idx % (C_out * H_out * W_out)
                co = rem // (H_out * W_out)
                rem2 = rem % (H_out * W_out)
                i = rem2 // W_out
                j = rem2 % W_out
                coords.append((n, co, i, j))

            # For each top pixel, compute receptive field and contributions
            for k_idx, coord in enumerate(coords):
                n, out_c, i_out, j_out = coord
                # Extract receptive field indices in input space
                receptive = self._get_receptive_field_indices(
                    i_out, j_out, x.shape, w.shape, stride, padding, dilation
                )
                # receptive: list of (c_in, i_in, j_in, k_h, k_w) mapping to weight indices for out_c
                # We'll gather x_patch and w_patch arrays aligned
                x_patch, w_patch, coords_patch = self._gather_patch(x, w, n, out_c, receptive)
                # x_patch and w_patch shapes: (num_elems,) flatten vectors - easier for contribution math
                num_elems = x_patch.numel()
                if num_elems == 0:
                    continue

                # Compute exact per-element storage rounding errors (data-aware)
                x_q_elem = quantize_to_dtype(x_patch, strategy.input_dtype).to(dtype=torch.float32)
                w_q_elem = quantize_to_dtype(w_patch, strategy.weight_dtype).to(dtype=torch.float32)
                eps_x = (x_q_elem - x_patch).abs()  # storage rounding absolute error for each element
                eps_w = (w_q_elem - w_patch).abs()

                # Compute per-element product and sum
                prod = (x_patch.to(dtype=strategy.compute_dtype) * w_patch.to(dtype=strategy.compute_dtype)).to(dtype=torch.float32)
                sum_abs_prod = prod.abs().sum().clamp(min=1e-12)

                # Compute local y_pixel in compute dtype and demotion error
                y_pixel_c = prod.sum().to(dtype=strategy.compute_dtype).to(dtype=torch.float32)
                y_pixel_demoted = demote_with_round(y_pixel_c.unsqueeze(0), strategy.output_dtype).squeeze(0)
                demote_err_pixel = abs(y_pixel_c - y_pixel_demoted).item()

                # Estimated accumulation error for this pixel:
                # Use sqrt(n) scaling heuristic times 0.5 ULP of compute result
                ops = num_elems
                # ULP estimate for y_pixel_c via ulp_like
                ulp_y = ulp_like(y_pixel_c.unsqueeze(0), strategy.compute_dtype).squeeze(0).item()
                accum_total = 0.5 * ulp_y * math.sqrt(max(1, ops))  # scalar estimate

                # Now compute per-element contributions:
                # linear propagation: |w_i| * eps_x_i + |x_i| * eps_w_i
                # accumulation share: allocate proportionally to |prod_i|
                abs_w = w_patch.abs()
                abs_x = x_patch.abs()
                linear_term = (abs_w * eps_x) + (abs_x * eps_w)  # per element
                abs_prod = prod.abs()
                accum_share = (abs_prod / sum_abs_prod) * accum_total

                # Total per-element contribution to this pixel error
                contrib_vec = linear_term + accum_share

                # We also include demote_err_pixel as extra bucket (not per-element) - attribute fractionally
                # fractional_demote_share = distribute demote error proportional to contrib magnitude
                sum_contrib = contrib_vec.sum().clamp(min=1e-12)
                demote_share_vec = (contrib_vec / sum_contrib) * demote_err_pixel

                total_elem_contrib = contrib_vec + demote_share_vec  # final per-element contributions

                # Summaries
                pixel_error = float(per_pixel_err[n, out_c, i_out, j_out].item())
                yref_val = float(y_ref[n, out_c, i_out, j_out].item())
                ymix_val = float(y_mixed[n, out_c, i_out, j_out].item())

                # Build contribution list sorted by descending contribution
                contrib_list = []
                # coords_patch list contains tuples (c_in, i_in, j_in, kh, kw)
                for idx_elem in range(num_elems):
                    c_in, i_in, j_in, kh, kw = coords_patch[idx_elem]
                    contrib_value = float(total_elem_contrib[idx_elem].item())
                    # also compute percent of pixel error explained
                    percent = contrib_value / (pixel_error + 1e-20)
                    contrib_list.append({
                        "input_coord": (n, c_in, i_in, j_in),
                        "weight_coord": (out_c, c_in, kh, kw),
                        "product": float(prod[idx_elem].item()),
                        "linear_term": float(linear_term[idx_elem].item()),
                        "accum_share": float(accum_share[idx_elem].item()),
                        "demote_share": float(demote_share_vec[idx_elem].item()),
                        "contribution": contrib_value,
                        "contribution_ratio_of_pixel": percent,
                    })

                # sort by contribution desc
                contrib_list.sort(key=lambda x: x["contribution"], reverse=True)

                path_record = {
                    "out_coord": (n, out_c, i_out, j_out),
                    "pixel_error": pixel_error,
                    "y_ref": yref_val,
                    "y_mixed": ymix_val,
                    "demote_error_pixel": demote_err_pixel,
                    "accum_estimate": accum_total,
                    "num_contributing_elements": num_elems,
                    "top_contributors": contrib_list[: min(10, len(contrib_list))],
                }

                report["top_paths"].append(path_record)

            # After analyzing top pixels, craft suggestions based on component decomposition
            # We'll combine both oracle-provided component_ratios and per-pixel observed metrics
            # compute aggregated contributions across top_paths
            agg = {"linear": 0.0, "accum": 0.0, "demote": 0.0}
            for p in report["top_paths"]:
                # sum of linear (linear_term + accum_share + demote_share)
                for c in p["top_contributors"]:
                    agg["linear"] += c["linear_term"]
                    agg["accum"] += c["accum_share"]
                    agg["demote"] += c["demote_share"]

            # normalize
            tot_agg = sum(abs(v) for v in agg.values()) + 1e-20
            agg_ratio = {k: float(abs(v) / tot_agg) for k, v in agg.items()}
            report["aggregated_contributions"] = agg
            report["aggregated_ratios"] = agg_ratio

            # primary source refinement: prefer per-pixel observed if available
            primary_refined = max(agg_ratio.items(), key=lambda kv: kv[1])[0] if agg_ratio else primary
            report["primary_source_refined"] = primary_refined

            # suggestion
            report["suggestion"] = self._suggest_from_aggregated(primary_refined, comp_ratios)

        return report

    # ------------------- internal helpers -------------------
    def _suggest_from_primary(self, primary: Optional[str]) -> str:
        """Generate suggestions based on primary error source"""
        if primary is None:
            return "Unable to determine primary error source; please provide more runtime context (x,w,strategy) for detailed diagnosis."
        if "demote" in primary or "output" in primary:
            return "Primary error source is output demotion. Suggestions: delay precision reduction or increase output precision; if not feasible, consider using FP32 storage for critical layers or investigate weight/activation scales."
        if "accum" in primary:
            return "Primary error source is accumulation rounding. Suggestions: use Kahan summation or increase accumulation precision; or adjust operator implementation (block-wise accumulation)."
        if "input" in primary or "storage" in primary:
            return "Primary error source is input storage. Suggestions: increase input storage precision or perform input normalization/clipping to reduce numerical range."
        if "weight" in primary:
            return "Primary error source is weight storage. Suggestions: maintain weights in high precision (FP32) or apply more precise quantization strategies (symmetric/unbiased quantization)."
        return "Analysis result is unclear; please check input/weight distribution or increase MC sampling for more robust estimates."

    def _suggest_from_aggregated(self, primary: Optional[str], comp_ratios: Dict[str, float]) -> str:
        """Generate suggestions based on aggregated analysis"""
        # More nuanced suggestion using both per-pixel and global component ratios
        base = self._suggest_from_primary(primary)
        # if comp_ratios indicates mismatch, add note
        if comp_ratios:
            sorted_comp = sorted(comp_ratios.items(), key=lambda kv: kv[1], reverse=True)
            top_name, top_ratio = sorted_comp[0]
            if top_ratio > 0.6 and top_name not in primary:
                base += f" Note: Global estimate indicates `{top_name}` also accounts for a significant proportion ({top_ratio:.2%})."
        return base

    def _get_receptive_field_indices(
        self,
        out_i: int,
        out_j: int,
        x_shape: Tuple[int, int, int, int],
        w_shape: Tuple[int, int, int, int],
        stride: int,
        padding: int,
        dilation: int,
    ) -> List[Tuple[int, int, int, int, int]]:
        """
        Return mapping from output coordinate (out_i,out_j) to input indices that contribute,
        and corresponding kernel indices. Returns list of tuples:
            (c_in, in_i, in_j, kh, kw)
        Works for single-batch n and without groups (extension possible).
        """
        _, C_in, H_in, W_in = x_shape
        C_out, C_in_k, K_h, K_w = w_shape

        receptive = []
        # For each input channel and k pos, compute input coordinates
        for c in range(C_in):
            for kh in range(K_h):
                for kw in range(K_w):
                    in_i = out_i * stride + kh - padding
                    in_j = out_j * stride + kw - padding
                    # check bounds
                    if 0 <= in_i < H_in and 0 <= in_j < W_in:
                        receptive.append((c, in_i, in_j, kh, kw))
        return receptive

    def _gather_patch(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        batch_n: int,
        out_c: int,
        receptive: List[Tuple[int, int, int, int, int]],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[int, int, int, int, int]]]:
        """
        Given receptive list, return flattened x_patch and w_patch aligned vectors
        x_patch: tensor shape (num_elems,) consisting of x[batch_n, c, in_i, in_j]
        w_patch: tensor shape (num_elems,) consisting of w[out_c, c, kh, kw]
        coords_patch: repeated receptive entries
        """
        elems = []
        w_elems = []
        coords_patch = []
        for (c, in_i, in_j, kh, kw) in receptive:
            elems.append(x[batch_n, c, in_i, in_j].unsqueeze(0))
            w_elems.append(w[out_c, c, kh, kw].unsqueeze(0))
            coords_patch.append((c, in_i, in_j, kh, kw))
        if len(elems) == 0:
            return torch.tensor([], dtype=torch.float32), torch.tensor([], dtype=torch.float32), coords_patch
        x_patch = torch.cat(elems, dim=0).to(dtype=torch.float32).view(-1)
        w_patch = torch.cat(w_elems, dim=0).to(dtype=torch.float32).view(-1)
        return x_patch, w_patch, coords_patch