# precision_estimation/examples/layernorm_precision_pipeline.py
import time
import json
import math
import torch
import torch.multiprocessing as mp

from core.config.precision_strategy import get_precision_strategy
from core.generator.layernorm_generator import LayerNormInputGenerator
from core.oracle.layernorm_oracle_mc import DataAwareMCLayerNormOracle
from core.detector.layernorm_error_detector import LayerNormErrorDetector
from core.analyzer.layernorm_error_analyzer import LayerNormErrorAnalyzer


def pretty_float(x):
    try:
        return float(x)
    except Exception:
        return x


def main():
    num_batches = 2000

    # 1) Select mixed precision strategy
    strategy = get_precision_strategy("FP16_input_FP32_weight_FP32_compute_accum")
    print("Using strategy:", strategy)

    # 2) Construct input generator
    normalized_shape = (768,)  # Hidden dimension
    gen = LayerNormInputGenerator(
        input_shape=(8, 512, 768),  # (batch, seq_len, hidden_dim)
        normalized_shape=normalized_shape,
        distribution="small_variance",  # Distribution most sensitive to LayerNorm
        device="cpu",
        seed=42,
        eps=1e-5,
        elementwise_affine=True
    )

    # 3) Construct Oracle
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    oracle = DataAwareMCLayerNormOracle(
        strategy=strategy,
        normalized_shape=normalized_shape,
        eps=1e-5,
        elementwise_affine=True,
        num_mc_samples=512,
        quantile=0.999,
        safety_factor=1.10,
        seeded=True,
        devices=devices,
        enable_noise_input=True,
        enable_noise_weight=True,
        enable_noise_bias=True,
        enable_noise_stats=True,
        enable_noise_output=True,
    )
    analyzer = LayerNormErrorAnalyzer()

    for batch_id in range(1, num_batches + 1):
        print(f"\n===== Batch {batch_id}/{num_batches} =====")

        # Generate input data
        x, weight, bias, meta = gen.generate()

        # 4) Construct detector and run detection
        detector = LayerNormErrorDetector(
            strategy, 
            normalized_shape, 
            meta["eps"], 
            meta["elementwise_affine"], 
            oracle
        )

        t0 = time.time()
        exceeded, actual_err, bound, oracle_result = detector.detect(x, weight, bias)
        dt = time.time() - t0

        print(f"Detection time: {dt:.3f}s")
        print(f"Maximum actual error: {actual_err:.6e}")
        print(f"Oracle predicted error bound: ±{bound:.6e}")
        print(f"Exceeded predicted bound: {'Yes' if bool(exceeded) else 'No'}")

        # 5) Call Analyzer
        print("Performing interpretability analysis (element-level)...")
        report = analyzer.analyze(
            oracle_result,
            x=x,
            weight=weight,
            bias=bias,
            strategy=strategy,
            normalized_shape=normalized_shape,
            eps=meta["eps"],
            top_k=10,
        )

        # 6) Print high-level summary
        print("\n-- Error Decomposition (High-level) --")
        for k, v in report.get("component_estimates", {}).items():
            print(f"  {k}: {pretty_float(v):.3e}")
        print("-- Ratios --")
        for k, v in report.get("component_ratios", {}).items():
            print(f"  {k}: {v:.3f}")
        print("Primary error source:", report.get("primary_source") or report.get("primary_source_refined"))
        
        # Statistics errors
        if "statistics_errors" in report:
            print("-- Statistics Errors --")
            for stat, err in report["statistics_errors"].items():
                print(f"  {stat}: {err:.6e}")
        
        # LayerNorm behavior statistics
        if "norm_behavior_stats" in report:
            print("-- LayerNorm Behavior Statistics --")
            for behavior, count in report["norm_behavior_stats"].items():
                print(f"  {behavior}: {count}")
        
        print("Suggestion:", report.get("suggestion"))

        # 7) Print top-k critical elements
        top_elements = report.get("top_elements", [])
        if top_elements:
            print(f"\n-- Top {len(top_elements)} Critical Element Analysis --")
            for idx, elem in enumerate(top_elements):
                print(f"\n[{idx+1}] Coordinate: {elem['coord']}")
                print(f"    element_error = {elem['element_error']:.6e}")
                print(f"    x_original = {elem['x_original']:.6e}")
                print(f"    x_quantized = {elem['x_quantized']:.6e}")
                print(f"    y_ref = {elem['y_ref']:.6e}")
                print(f"    y_mixed = {elem['y_mixed']:.6e}")
                print(f"    storage_error = {elem['storage_error']:.6e}")
                print(f"    weight_error = {elem['weight_error']:.6e}")
                print(f"    bias_error = {elem['bias_error']:.6e}")
                print(f"    demote_error = {elem['demote_error']:.6e}")
                print(f"    local_mean_error = {elem['local_mean_error']:.6e}")
                print(f"    local_std_error = {elem['local_std_error']:.6e}")
                print(f"    norm_behavior = {elem['norm_behavior']}")

        # 8) Save JSON
        out_json = {
            "detector": {
                "detected_exceeded": bool(exceeded),
                "actual_err": actual_err,
                "predicted_bound": bound,
                "elapsed_sec": dt,
            },
            "analyzer_report": {
                "component_estimates": report.get("component_estimates", {}),
                "component_ratios": report.get("component_ratios", {}),
                "primary_source": report.get("primary_source_refined", report.get("primary_source")),
                "suggestion": report.get("suggestion", ""),
                "statistics_errors": report.get("statistics_errors", {}),
                "norm_behavior_stats": report.get("norm_behavior_stats", {}),
                "top_elements_summary": [
                    {
                        "coord": elem["coord"],
                        "element_error": elem["element_error"],
                        "x_original": elem["x_original"],
                        "y_ref": elem["y_ref"],
                        "y_mixed": elem["y_mixed"],
                        "norm_behavior": elem["norm_behavior"],
                        "storage_error": elem["storage_error"],
                        "weight_error": elem["weight_error"],
                        "bias_error": elem["bias_error"],
                        "demote_error": elem["demote_error"],
                        "local_mean_error": elem["local_mean_error"],
                        "local_std_error": elem["local_std_error"],
                    }
                    for elem in top_elements
                ],
            },
        }

        json_filename = f"evaluation/experiment/default_layernorm_analysis_batch{batch_id}.json"
        with open(json_filename, "w") as f:
            json.dump(out_json, f, indent=2)
        print(f"\nAnalysis results saved to {json_filename}")

    print("\nAll batches completed")


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()