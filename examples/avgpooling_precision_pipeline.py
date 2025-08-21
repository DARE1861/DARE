# precision_estimation/examples/pooling_precision_pipeline.py
import time
import json
import torch
import torch.multiprocessing as mp

from core.config.precision_strategy import get_precision_strategy
from core.generator.pooling_generator import PoolingInputGenerator
from core.oracle.pooling_oracle_mc import DataAwareMCPoolingOracle
from core.detector.pooling_error_detector import PoolingErrorDetector
from core.analyzer.pooling_error_analyzer import PoolingErrorAnalyzer


def pretty_float(x):
    try:
        return float(x)
    except Exception:
        return x


def main():
    num_batches = 2000

    # 1) Mixed precision strategy
    strategy = get_precision_strategy("FP16_input_FP32_weight_FP32_compute_accum")
    print("Using strategy:", strategy)

    # 2) Input generator
    pool_params = {"pool_type": "avg", "kernel_size": 2, "stride": 2}
    gen = PoolingInputGenerator(
        input_shape=(1, 3, 32, 32),
        pool_type=pool_params.get("mode", "avg"),
        kernel_size=pool_params.get("kernel_size", 2),
        stride=pool_params.get("stride", 2),
        seed=42
    )

    # 3) Oracle
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    oracle = DataAwareMCPoolingOracle(
        strategy=strategy,
        pool_params=pool_params,
        num_mc_samples=512,
        quantile=0.999,
        safety_factor=1.10,
        seeded=True,
        devices=devices,
        enable_noise_input=True,
        enable_noise_accum=True,
    )

    analyzer = PoolingErrorAnalyzer()

    for batch_id in range(1, num_batches + 1):
        print(f"\n===== Batch {batch_id}/{num_batches} =====")

        # Generate input
        x, meta = gen.generate()
        print(f"Generated input shape: {x.shape}")

        # 4) Construct detector and run detection
        detector = PoolingErrorDetector(strategy, pool_params, oracle)
        t0 = time.time()
        exceeded, actual_err, bound, oracle_result = detector.detect(x)
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
            strategy=strategy,
            pool_params=pool_params,
            top_k=5,
        )

        # 6) High-level summary
        print("\n-- Error Decomposition (High-level) --")
        for k, v in report.get("component_estimates", {}).items():
            print(f"  {k}: {pretty_float(v):.3e}")
        print("-- Ratios --")
        for k, v in report.get("component_ratios", {}).items():
            print(f"  {k}: {v:.3f}")
        print("Primary error source:", report.get("primary_source") or report.get("primary_source_refined"))
        print("Suggestion:", report.get("suggestion"))

        # 7) Top-k critical elements
        top_paths = report.get("top_paths", [])
        if not top_paths:
            print("\nNo top_paths (x not provided or output is empty)")
        else:
            print(f"\n-- Top {len(top_paths)} Critical Element Contributions --")
            for idx, p in enumerate(top_paths):
                print(f"\n[{idx+1}] Output coordinate: {p['out_coord']}")
                print(f"    element_error = {p['pixel_error']:.6e}")
                print(f"    y_ref = {p['y_ref']:.6e}, y_mixed = {p['y_mixed']:.6e}")
                print(f"    accum_estimate = {p['accum_estimate']:.6e}")
                print(f"    Contributing elements: {p['num_contributing_elements']}")
                print(f"    Top contributors:")
                for j, c in enumerate(p['top_contributors'][:10]):
                    print(f"      {j+1}. input{c['input_coord']}, contrib={c['contribution']:.3e}, ratio_of_element={c['contribution_ratio_of_pixel']:.3%}")

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
                "suggestion": report.get("suggestion"),
                "top_paths_summary": top_paths,
            },
        }

        json_filename = f"evaluation/experiment/default_avg_pooling_analysis_batch{batch_id}.json"
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