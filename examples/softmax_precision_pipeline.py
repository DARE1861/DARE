# precision_estimation/examples/softmax_precision_pipeline.py
import time
import json
import torch
import torch.multiprocessing as mp

from core.config.precision_strategy import get_precision_strategy
from core.generator.softmax_generator import SoftmaxInputGenerator
from core.oracle.softmax_oracle_mc import DataAwareMCSoftmaxOracle
from core.detector.softmax_error_detector import SoftmaxErrorDetector
from core.analyzer.softmax_error_analyzer import SoftmaxErrorAnalyzer


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
    gen = SoftmaxInputGenerator(
        input_shape=(8, 512),  # (batch_size, vocab_size)
        distribution="large_logits",  # Test numerical stability
        device="cpu",
        seed=42,
        temperature=1.0,
        dim=-1,  # Apply softmax on the last dimension
    )

    # 3) Construct Oracle
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    oracle = DataAwareMCSoftmaxOracle(
        strategy=strategy,
        dim=-1,
        num_mc_samples=512,
        quantile=0.999,
        safety_factor=1.10,
        seeded=True,
        devices=devices,
        enable_noise_input=True,
        enable_noise_max_sub=True,
        enable_noise_exp=True,
        enable_noise_sum=True,
        enable_noise_output=True,
    )
    analyzer = SoftmaxErrorAnalyzer()

    for batch_id in range(1, num_batches + 1):
        print(f"\n===== Batch {batch_id}/{num_batches} =====")

        # Generate input data
        x, meta = gen.generate()
        dim = meta.get("dim", -1)

        # 4) Construct detector and run detection
        detector = SoftmaxErrorDetector(strategy, dim, oracle)

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
            dim=dim,
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
        
        # Softmax specific statistics
        if "input_stats" in report:
            print("-- Input Statistics --")
            for k, v in report["input_stats"].items():
                print(f"  {k}: {v:.6f}")
        
        if "stability_stats" in report:
            print("-- Numerical Stability Statistics --")
            for k, v in report["stability_stats"].items():
                print(f"  {k}: {v:.6e}")
        
        if "max_logit_error_ratio" in report:
            print(f"Max logit error ratio: {report['max_logit_error_ratio']:.3f}")
        
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
                print(f"    demote_error = {elem['demote_error']:.6e}")
                
                logit_info = elem['logit_analysis']
                print(f"    logit_analysis:")
                print(f"      softmax_position = {logit_info['softmax_position']}")
                print(f"      is_max_logit = {logit_info['is_max_logit']}")
                print(f"      distance_from_max = {logit_info['distance_from_max']:.6e}")

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
                "input_stats": report.get("input_stats", {}),
                "stability_stats": report.get("stability_stats", {}),
                "max_logit_error_ratio": report.get("max_logit_error_ratio", 0),
                "top_elements_summary": [
                    {
                        "coord": elem["coord"],
                        "element_error": elem["element_error"],
                        "x_original": elem["x_original"],
                        "y_ref": elem["y_ref"],
                        "y_mixed": elem["y_mixed"],
                        "storage_error": elem["storage_error"],
                        "demote_error": elem["demote_error"],
                        "logit_analysis": elem["logit_analysis"],
                    }
                    for elem in top_elements
                ],
            },
        }

        json_filename = f"evaluation/experiment/default_softmax_analysis_batch{batch_id}.json"
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