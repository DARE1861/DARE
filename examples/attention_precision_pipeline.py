# precision_estimation/examples/attention_precision_pipeline.py
import time
import json
import torch
import torch.multiprocessing as mp

from core.config.precision_strategy import get_precision_strategy
from core.generator.attention_generator import AttentionInputGenerator
from core.oracle.attention_oracle_mc import DataAwareMCAttentionOracle
from core.detector.attention_error_detector import AttentionErrorDetector
from core.analyzer.attention_error_analyzer import AttentionErrorAnalyzer


def pretty_float(x):
    try:
        return float(x)
    except Exception:
        return x


def main():
    num_batches = 2000  # Attention computation is heavy, reduce batch count

    # 1) Select mixed precision strategy
    strategy = get_precision_strategy("FP16_input_FP32_weight_FP32_compute_accum")
    print("Using strategy:", strategy)

    # 2) Construct input generator
    gen = AttentionInputGenerator(
        batch_size=1,
        seq_len=128,  # Moderate sequence length
        d_model=256,
        num_heads=8,
        distribution="attention_specific",  # Distribution specifically for attention
        device="cpu",
        seed=42,
    )

    # 3) Construct Oracle
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    # Attention computation is complex, reduce MC sample count
    oracle = DataAwareMCAttentionOracle(
        strategy=strategy,
        attention_params={"scale": 1.0/16.0, "is_causal": True},  # scale = 1/sqrt(d_head)
        num_mc_samples=256,  # Fewer samples
        quantile=0.999,
        safety_factor=1.10,
        seeded=True,
        devices=devices,
        enable_noise_input=True,
        enable_noise_matmul1=True,
        enable_noise_scaling=True,
        enable_noise_softmax=True,
        enable_noise_matmul2=True,
        enable_noise_output=True,
    )
    analyzer = AttentionErrorAnalyzer()

    for batch_id in range(1, num_batches + 1):
        print(f"\n===== Batch {batch_id}/{num_batches} =====")

        # Generate input data
        q, k, v, mask, meta = gen.generate()
        attention_params = meta.get("attention_params", {"scale": 1.0/16.0, "is_causal": True})

        print(f"Input shapes: Q{q.shape}, K{k.shape}, V{v.shape}")

        # 4) Construct detector and run detection
        detector = AttentionErrorDetector(strategy, attention_params, oracle)

        t0 = time.time()
        exceeded, actual_err, bound, oracle_result = detector.detect(q, k, v, mask)
        dt = time.time() - t0

        print(f"Detection time: {dt:.3f}s")
        print(f"Maximum actual error: {actual_err:.6e}")
        print(f"Oracle predicted error bound: ±{bound:.6e}")
        print(f"Exceeded predicted bound: {'Yes' if bool(exceeded) else 'No'}")

        # 5) Call Analyzer
        print("Performing interpretability analysis (token-level and head-level)...")
        report = analyzer.analyze(
            oracle_result,
            q=q,
            k=k,
            v=v,
            mask=mask,
            strategy=strategy,
            attention_params=attention_params,
            top_k=8,
        )

        # 6) Print high-level summary
        print("\n-- Error Decomposition (High-level) --")
        for k, v in report.get("component_estimates", {}).items():
            print(f"  {k}: {pretty_float(v):.3e}")
        print("-- Ratios --")
        for k, v in report.get("component_ratios", {}).items():
            print(f"  {k}: {v:.3f}")
        print("Primary error source:", report.get("primary_source") or report.get("primary_source_refined"))

        # Attention pattern statistics
        if "attention_pattern_stats" in report:
            print("-- Attention Pattern Statistics --")
            stats = report["attention_pattern_stats"]
            print(f"  Global average entropy: {stats.get('global_avg_entropy', 0):.3f}")
            print(f"  Global sparsity: {stats.get('global_sparsity', 0):.3f}")
            print(f"  Max attention weight: {stats.get('max_attention_weight', 0):.6f}")

        if "attention_behavior_stats" in report:
            print("-- Attention Behavior Statistics --")
            behavior = report["attention_behavior_stats"]
            total = behavior["total_analyzed"]
            print(f"  High entropy tokens: {behavior['high_entropy_tokens']}/{total}")
            print(f"  Low entropy tokens: {behavior['low_entropy_tokens']}/{total}")

        print("Suggestion:", report.get("suggestion"))

        # 7) Print top-k critical tokens
        top_tokens = report.get("top_tokens", [])
        if top_tokens:
            print(f"\n-- Top {len(top_tokens)} Critical Token Analysis --")
            for idx, token in enumerate(top_tokens):
                print(f"\n[{idx+1}] Batch{token['batch_idx']}, Head{token['head_idx']}, Token{token['token_idx']}")
                print(f"    token_error = {token['token_error']:.6e}")
                print(f"    entropy_ref = {token['entropy_ref']:.3f}")
                print(f"    max_attention_weight_ref = {token['max_attention_weight_ref']:.6f}")
                print(f"    attention_weight_error = {token['attention_weight_error']:.6e}")
                print(f"    Main stage errors:")
                for stage, err in token['stage_errors'].items():
                    print(f"      {stage}: {err:.6e}")

        # 8) Print head analysis
        head_analysis = report.get("head_analysis", [])
        if head_analysis:
            print(f"\n-- Per-Head Error Analysis --")
            for head in head_analysis:
                print(f"Head {head['head_idx']}: avg_error={head['avg_error']:.6e}, "
                      f"avg_entropy={head['avg_entropy']:.3f}, "
                      f"sparsity={head['sparsity']:.3f}")

        # 9) Save JSON
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
                "attention_pattern_stats": report.get("attention_pattern_stats", {}),
                "attention_behavior_stats": report.get("attention_behavior_stats", {}),
                "top_tokens_summary": [
                    {
                        "batch_idx": t["batch_idx"],
                        "head_idx": t["head_idx"],
                        "token_idx": t["token_idx"],
                        "token_error": t["token_error"],
                        "entropy_ref": t["entropy_ref"],
                        "max_attention_weight_ref": t["max_attention_weight_ref"],
                        "attention_weight_error": t["attention_weight_error"],
                        "stage_errors": t["stage_errors"],
                    }
                    for t in top_tokens
                ],
                "head_analysis": head_analysis,
            },
        }

        json_filename = f"evaluation/experiment/default_attention_analysis_batch{batch_id}.json"
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