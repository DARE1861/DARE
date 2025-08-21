# precision_estimation/examples/batchnorm_precision_pipeline.py
import time
import json
import torch
import torch.multiprocessing as mp
import math

from core.config.precision_strategy import get_precision_strategy
from core.generator.batchnorm_generator import BatchNormInputGenerator
from core.oracle.batchnorm_oracle_mc import DataAwareMCBatchNormOracle
from core.detector.batchnorm_error_detector import BatchNormErrorDetector
from core.analyzer.batchnorm_error_analyzer import BatchNormErrorAnalyzer


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
    gen = BatchNormInputGenerator(
        input_shape=(4, 32, 16, 16),  # Smaller batch to test BatchNorm sensitivity
        distribution="boundary",  # Distribution most sensitive to BatchNorm
        device="cpu",
        seed=42,
        eps=1e-5,
        momentum=0.1,
        affine=True
    )

    # 3) Construct Oracle
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    
    for batch_id in range(1, num_batches + 1):
        print(f"\n===== Batch {batch_id}/{num_batches} =====")

        # Generate input data
        x, weight, bias, running_mean, running_var, meta = gen.generate()
        bn_params = meta.get("bn_params", {
            "eps": 1e-5, "momentum": 0.1, "affine": True, 
            "track_running_stats": True, "training": True
        })

        oracle = DataAwareMCBatchNormOracle(
            strategy=strategy,
            bn_params=bn_params,
            num_mc_samples=256,  # BatchNorm computation is relatively simple, can use fewer samples
            quantile=0.999,
            safety_factor=1.10,
            seeded=True,
            devices=devices,
            enable_noise_input=True,
            enable_noise_weight=True,
            enable_noise_stats=True,
            enable_noise_output=True,
        )
        analyzer = BatchNormErrorAnalyzer()

        # 4) Construct detector and run detection
        detector = BatchNormErrorDetector(strategy, bn_params, oracle)

        t0 = time.time()
        exceeded, actual_err, bound, oracle_result = detector.detect(x, weight, bias, running_mean, running_var)
        dt = time.time() - t0

        print(f"Detection time: {dt:.3f}s")
        print(f"Maximum actual error: {actual_err:.6e}")
        print(f"Oracle predicted error bound: ±{bound:.6e}")
        print(f"Exceeded predicted bound: {'Yes' if bool(exceeded) else 'No'}")

        # 5) Call Analyzer
        print("Performing interpretability analysis (channel-level)...")
        report = analyzer.analyze(
            oracle_result,
            x=x,
            weight=weight,
            bias=bias,
            running_mean=running_mean,
            running_var=running_var,
            strategy=strategy,
            bn_params=bn_params,
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
        
        # BatchNorm specific statistics
        batch_stats = report.get("batch_statistics", {})
        if batch_stats:
            print("-- Batch Statistics --")
            print(f"  batch_size: {batch_stats.get('batch_size')}")
            print(f"  num_channels: {batch_stats.get('num_channels')}")
            print(f"  spatial_size: {batch_stats.get('spatial_size')}")
            print(f"  input_range: min={batch_stats['input_range']['min']:.3e}, max={batch_stats['input_range']['max']:.3e}")
        
        # Numerical feature statistics
        if "numerical_feature_stats" in report:
            print("-- Numerical Feature Statistics --")
            for feature, count in report["numerical_feature_stats"].items():
                print(f"  {feature}: {count}")
        
        print("Suggestion:", report.get("suggestion"))

        # 7) Print top-k critical channels
        channel_analysis = report.get("channel_analysis", [])
        if channel_analysis:
            print(f"\n-- Top {len(channel_analysis)} Critical Channel Analysis --")
            for idx, ch in enumerate(channel_analysis):
                print(f"\n[{idx+1}] Channel: {ch['channel_index']}")
                print(f"    channel_error = {ch['channel_error']:.6e}")
                print(f"    input_range: min={ch['input_range']['min']:.3e}, max={ch['input_range']['max']:.3e}")
                print(f"    weight_error = {ch['weight_error']:.6e}")
                print(f"    bias_error = {ch['bias_error']:.6e}")
                print(f"    stats_instability = {ch['stats_instability_estimate']:.6e}")
                
                # Statistics information
                if "batch_mean" in ch:
                    print(f"    batch_mean = {ch['batch_mean']:.6e}")
                    print(f"    batch_var = {ch['batch_var']:.6e}")
                elif "running_mean" in ch:
                    print(f"    running_mean = {ch['running_mean']:.6e}")
                    print(f"    running_var = {ch['running_var']:.6e}")
                
                # Numerical features
                features = ch.get("numerical_features", {})
                problematic_features = [k for k, v in features.items() if v]
                if problematic_features:
                    print(f"    problematic_features: {', '.join(problematic_features)}")

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
                "batch_statistics": report.get("batch_statistics", {}),
                "numerical_feature_stats": report.get("numerical_feature_stats", {}),
                "channel_analysis_summary": [
                    {
                        "channel_index": ch["channel_index"],
                        "channel_error": ch["channel_error"],
                        "weight_error": ch["weight_error"],
                        "bias_error": ch["bias_error"],
                        "stats_instability_estimate": ch["stats_instability_estimate"],
                        "numerical_features": ch["numerical_features"],
                    }
                    for ch in channel_analysis
                ],
            },
        }

        json_filename = f"evaluation/experiment/default_batchnorm_analysis_batch{batch_id}.json"
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