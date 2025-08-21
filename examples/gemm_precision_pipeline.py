# precision_estimation/examples/gemm_precision_pipeline.py
import time
import json
import torch
import torch.multiprocessing as mp

from core.config.precision_strategy import get_precision_strategy
from core.generator.gemm_generator import GEMMInputGenerator
from core.oracle.gemm_oracle_mc import DataAwareMCGEMMOracle
from core.detector.gemm_error_detector import GEMMErrorDetector
from core.analyzer.gemm_error_analyzer import GEMMErrorAnalyzer


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
    gen = GEMMInputGenerator(
        input_shape=(256, 1024),   # M x K
        weight_shape=(1024, 512),  # K x N
        distribution="adversarial_sum",  # Options: 'normal', 'boundary', 'adversarial_sum', 'ill_conditioned'
        device="cpu",
        seed=42,
    )

    # 3) Construct Oracle
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    oracle = DataAwareMCGEMMOracle(
        strategy=strategy,
        gemm_params={"transpose_a": False, "transpose_b": False},
        num_mc_samples=512,
        quantile=0.999,
        safety_factor=1.10,
        seeded=True,
        devices=devices,
        enable_noise_input=True,
        enable_noise_weight=True,
        enable_noise_accum=True,
        enable_noise_output=True,
    )
    analyzer = GEMMErrorAnalyzer()

    for batch_id in range(1, num_batches + 1):
        print(f"\n===== Batch {batch_id}/{num_batches} =====")

        # Generate input data
        x, w, meta = gen.generate()
        gemm_params = meta.get("gemm_params", {"transpose_a": False, "transpose_b": False})

        print(f"Matrix dimensions: X{x.shape} @ W{w.shape} -> Y{(x.shape[0], w.shape[1])}")
        print(f"Accumulation dimension K: {x.shape[1]}")

        # 4) Construct detector and run detection
        detector = GEMMErrorDetector(strategy, gemm_params, oracle)

        t0 = time.time()
        exceeded, actual_err, bound, oracle_result = detector.detect(x, w)
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
            w=w,
            strategy=strategy,
            gemm_params=gemm_params,
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
        
        # Numerical stability analysis
        if "numerical_stability" in report:
            stability = report["numerical_stability"]
            print(f"\n-- Numerical Stability Analysis --")
            print(f"  Accumulation dimension: {stability['accumulation_dimension']}")
            print(f"  Average condition indicator: {stability['average_condition_indicator']:.2f}")
            print(f"  Stability assessment: {stability['stability_assessment']}")
        
        print("Suggestion:", report.get("suggestion"))

        # 7) Print top-k critical elements
        top_elements = report.get("top_elements", [])
        if top_elements:
            print(f"\n-- Top {len(top_elements)} Critical Element Analysis --")
            for idx, elem in enumerate(top_elements):
                print(f"\n[{idx+1}] Output coordinate: {elem['coord']}")
                print(f"    element_error = {elem['element_error']:.6e}")
                print(f"    y_ref = {elem['y_ref']:.6e}")
                print(f"    y_mixed = {elem['y_mixed']:.6e}")
                print(f"    x_storage_error = {elem['x_storage_error']:.6e}")
                print(f"    w_storage_error = {elem['w_storage_error']:.6e}")
                print(f"    accumulation_error_estimate = {elem['accumulation_error_estimate']:.6e}")
                print(f"    demote_error = {elem['demote_error']:.6e}")
                print(f"    accumulation_dim = {elem['accumulation_dim']}")
                print(f"    condition_indicator = {elem['condition_indicator']:.2f}")
                
                stats = elem['product_stats']
                print(f"    product_stats: min={stats['min']:.3e}, max={stats['max']:.3e}, mean={stats['mean']:.3e}, std={stats['std']:.3e}")

        # 8) Save JSON
        out_json = {
            "detector": {
                "detected_exceeded": bool(exceeded),
                "actual_err": actual_err,
                "predicted_bound": bound,
                "elapsed_sec": dt,
            },
            "matrix_info": {
                "input_shape": list(x.shape),
                "weight_shape": list(w.shape),
                "output_shape": [x.shape[0], w.shape[1]],
                "accumulation_dim": x.shape[1],
            },
            "analyzer_report": {
                "component_estimates": report.get("component_estimates", {}),
                "component_ratios": report.get("component_ratios", {}),
                "primary_source": report.get("primary_source_refined", report.get("primary_source")),
                "suggestion": report.get("suggestion", ""),
                "numerical_stability": report.get("numerical_stability", {}),
                "top_elements_summary": [
                    {
                        "coord": elem["coord"],
                        "element_error": elem["element_error"],
                        "y_ref": elem["y_ref"],
                        "y_mixed": elem["y_mixed"],
                        "x_storage_error": elem["x_storage_error"],
                        "w_storage_error": elem["w_storage_error"],
                        "accumulation_error_estimate": elem["accumulation_error_estimate"],
                        "demote_error": elem["demote_error"],
                        "condition_indicator": elem["condition_indicator"],
                        "product_stats": elem["product_stats"],
                    }
                    for elem in top_elements
                ],
            },
        }

        json_filename = f"evaluation/experiment/default_gemm_analysis_batch{batch_id}.json"
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