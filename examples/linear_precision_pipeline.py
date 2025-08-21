# precision_estimation/examples/linear_precision_pipeline.py
import time
import json
import torch
import torch.multiprocessing as mp

from core.config.precision_strategy import get_precision_strategy
from core.generator.linear_generator import LinearInputGenerator
from core.oracle.linear_oracle_mc import DataAwareMCLinearOracle
from core.detector.linear_error_detector import LinearErrorDetector
from core.analyzer.linear_error_analyzer import LinearErrorAnalyzer


def pretty_float(x):
    try:
        return float(x)
    except Exception:
        return x


def main():
    num_batches = 200

    # 1) Select mixed precision strategy
    strategy = get_precision_strategy("FP16_all")
    print("Using strategy:", strategy)

    # 2) Construct input generator
    gen = LinearInputGenerator(
        input_shape=(32, 1024),      # (batch_size, in_features)
        weight_shape=(4096, 1024),   # (out_features, in_features)
        bias=True,                   # Include bias
        distribution="adversarial_sum",  # Distribution sensitive to accumulation errors
        device="cpu",
        seed=42,
    )

    # 3) Construct Oracle
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    oracle = DataAwareMCLinearOracle(
        strategy=strategy,
        num_mc_samples=512,
        quantile=0.999,
        safety_factor=1.10,
        seeded=True,
        devices=devices,
        enable_noise_input=True,
        enable_noise_weight=True,
        enable_noise_bias=True,
        enable_noise_accum=True,
        enable_noise_output=True,
    )
    analyzer = LinearErrorAnalyzer()

    for batch_id in range(1, num_batches + 1):
        print(f"\n===== Batch {batch_id}/{num_batches} =====")

        # Generate input data
        x, w, b, meta = gen.generate()
        print(f"Input shape: {x.shape}, Weight shape: {w.shape}")
        if b is not None:
            print(f"Bias shape: {b.shape}")
        print(f"Distribution type: {meta['distribution']}")

        # 4) Construct detector and run detection
        detector = LinearErrorDetector(strategy, oracle)

        t0 = time.time()
        exceeded, actual_err, bound, oracle_result = detector.detect(x, w, b)
        dt = time.time() - t0

        print(f"Detection time: {dt:.3f}s")
        print(f"Maximum actual error: {actual_err:.6e}")
        print(f"Oracle predicted error bound: ±{bound:.6e}")
        print(f"Exceeded predicted bound: {'Yes' if bool(exceeded) else 'No'}")

        # 5) Call Analyzer
        print("Performing interpretability analysis (neuron-level)...")
        report = analyzer.analyze(
            oracle_result,
            x=x,
            w=w,
            b=b,
            strategy=strategy,
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
        print("Suggestion:", report.get("suggestion"))

        # 7) Print top-k critical neurons
        top_neurons = report.get("top_neurons", [])
        if top_neurons:
            print(f"\n-- Top {len(top_neurons)} Critical Neuron Analysis --")
            for idx, neuron in enumerate(top_neurons):
                print(f"\n[{idx+1}] Neuron coordinate: {neuron['coord']}")
                print(f"    neuron_error = {neuron['neuron_error']:.6e}")
                print(f"    y_ref = {neuron['y_ref']:.6e}")
                print(f"    y_mixed = {neuron['y_mixed']:.6e}")
                print(f"    input_storage_error = {neuron['input_storage_error']:.6e}")
                print(f"    weight_storage_error = {neuron['weight_storage_error']:.6e}")
                if neuron.get('bias_storage_error', 0) > 0:
                    print(f"    bias_storage_error = {neuron['bias_storage_error']:.6e}")
                print(f"    accumulation_estimate = {neuron['accumulation_estimate']:.6e}")
                print(f"    demote_error = {neuron['demote_error']:.6e}")
                print(f"    in_features = {neuron['in_features']}")
                print(f"    neuron_idx = {neuron['neuron_idx']}, batch_idx = {neuron['batch_idx']}")

        # 8) Aggregated analysis
        if "aggregated_ratios" in report:
            print("\n-- Aggregated Error Analysis --")
            for error_type, ratio in report["aggregated_ratios"].items():
                print(f"  {error_type}: {ratio:.3f}")

        # 9) Save JSON
        out_json = {
            "detector": {
                "detected_exceeded": bool(exceeded),
                "actual_err": actual_err,
                "predicted_bound": bound,
                "elapsed_sec": dt,
            },
            "meta": meta,
            "analyzer_report": {
                "component_estimates": report.get("component_estimates", {}),
                "component_ratios": report.get("component_ratios", {}),
                "primary_source": report.get("primary_source_refined", report.get("primary_source")),
                "suggestion": report.get("suggestion", ""),
                "aggregated_ratios": report.get("aggregated_ratios", {}),
                "top_neurons_summary": [
                    {
                        "coord": neuron["coord"],
                        "neuron_error": neuron["neuron_error"],
                        "y_ref": neuron["y_ref"],
                        "y_mixed": neuron["y_mixed"],
                        "input_storage_error": neuron["input_storage_error"],
                        "weight_storage_error": neuron["weight_storage_error"],
                        "bias_storage_error": neuron.get("bias_storage_error", 0),
                        "accumulation_estimate": neuron["accumulation_estimate"],
                        "demote_error": neuron["demote_error"],
                        "in_features": neuron["in_features"],
                        "neuron_idx": neuron["neuron_idx"],
                        "batch_idx": neuron["batch_idx"],
                    }
                    for neuron in top_neurons
                ],
            },
        }

        json_filename = f"evaluation/experiment/fp16_linear_analysis_batch{batch_id}.json"
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