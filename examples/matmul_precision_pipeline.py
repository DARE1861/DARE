import time
import json
import torch
import torch.multiprocessing as mp

from core.config.precision_strategy import get_precision_strategy
from core.generator.matmul_generator import MatmulInputGenerator
from core.oracle.matmul_oracle_mc import DataAwareMCMatmulOracle
from core.detector.matmul_error_detector import MatmulErrorDetector
from core.analyzer.matmul_error_analyzer import MatmulErrorAnalyzer


def pretty_float(x):
    try:
        return float(x)
    except Exception:
        return x


def main():
    # How many input sets to run at once
    num_batches = 2000

    # 1) Select mixed precision strategy
    strategy = get_precision_strategy("FP16_input_FP32_weight_FP32_compute_accum")
    print("Using strategy:", strategy)

    # 2) Construct input generator
    gen = MatmulInputGenerator(
        a_shape=(512, 256),
        b_shape=(256, 512),
        distribution="adversarial_sum",  # Options: 'normal', 'boundary', 'uniform', 'adversarial_sum'
        device="cpu",
        seed=42,
    )

    # 3) Construct Oracle (supports multi-GPU parallelism)
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    oracle = DataAwareMCMatmulOracle(
        strategy=strategy,
        num_mc_samples=512,      # Adjust based on resources
        quantile=0.999,
        safety_factor=1.10,
        seeded=True,
        devices=devices,         # e.g. [0,1,2,3] or []
        enable_noise_input=True,
        enable_noise_weight=True,
        enable_noise_accum=True,
        enable_noise_output=True,
    )
    analyzer = MatmulErrorAnalyzer()

    # Loop through multiple batches
    for batch_id in range(1, num_batches + 1):
        print(f"\n===== Batch {batch_id}/{num_batches} =====")

        # Generate input data
        x, w, meta = gen.generate()

        # 4) Construct detector and run detection
        detector = MatmulErrorDetector(strategy, oracle)

        t0 = time.time()
        exceeded, actual_err, bound, oracle_result = detector.detect(x, w)
        dt = time.time() - t0

        print(f"Detection time: {dt:.3f}s")
        print(f"Maximum actual error: {actual_err:.6e}")
        print(f"Oracle predicted error bound: ±{bound:.6e}")
        print(f"Exceeded predicted bound: {'Yes' if bool(exceeded) else 'No'}")

        # 5) Call Analyzer: pass oracle_result + original tensors x,w + strategy
        print("Performing interpretability analysis (matrix element-level)...")
        report = analyzer.analyze(
            oracle_result,
            A=x,
            B=w,
            strategy=strategy,
            top_k=5,
        )

        # 6) Print high-level summary
        print("\n-- Error Decomposition (High-level) --")
        for k, v in report.get("component_estimates", {}).items():
            print(f"  {k}: {pretty_float(v):.3e}")
        print("-- Ratios --")
        for k, v in report.get("component_ratios", {}).items():
            print(f"  {k}: {v:.3f}")
        print("Primary error source:", report.get("primary_source") or report.get("primary_source_refined"))
        print("Suggestion:", report.get("suggestion"))

        # 7) Print top-k critical path summary (matrix elements)
        top_paths = report.get("top_paths", [])
        if not top_paths:
            print("\nNo top_paths (x/w not provided or output is empty)")
        else:
            print(f"\n-- Top {len(top_paths)} Critical Output Element Contributions --")
            for idx, p in enumerate(top_paths):
                print(f"\n[{idx+1}] Output index: {p['out_coord']}")
                print(f"    element_error = {p['pixel_error']:.6e}")
                print(f"    y_ref = {p['y_ref']:.6e}, y_mixed = {p['y_mixed']:.6e}")
                print(f"    demote_error_element = {p['demote_error_pixel']:.6e}")
                print(f"    accum_estimate = {p['accum_estimate']:.6e}")
                print(f"    Contributing elements: {p['num_contributing_elements']}")
                print(f"    Top contributors:")
                for j, c in enumerate(p['top_contributors'][:10]):
                    in_coord = c['a_coord']
                    w_coord = c['b_coord']
                    contrib = c['contribution']
                    ratio = c['contribution_ratio_of_pixel']
                    print(f"      {j+1}. input{in_coord}, weight{w_coord}, contrib={contrib:.3e}, ratio_of_element={ratio:.3%}")

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
                "top_paths_summary": [
                    {
                        "out_coord": p["out_coord"],
                        "pixel_error": p["pixel_error"],
                        "y_ref": p["y_ref"],
                        "y_mixed": p["y_mixed"],
                        "demote_error_pixel": p["demote_error_pixel"],
                        "accum_estimate": p["accum_estimate"],
                        "num_contributing_elements": p["num_contributing_elements"],
                        "top_contributors": [
                            {
                                "input_coord": c["a_coord"],
                                "weight_coord": c["b_coord"],
                                "contribution": c["contribution"],
                                "contribution_ratio_of_pixel": c["contribution_ratio_of_pixel"],
                            }
                            for c in p["top_contributors"][:10]
                        ],
                    }
                    for p in top_paths
                ],
            },
        }

        json_filename = f"default_matmul_analysis_batch{batch_id}.json"
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