"""
Variance Runner — Multi-Seed Ablation & Threshold Sweep.

Runs ablation and/or threshold sweep across multiple seeds to produce
publishable mean ± std results with significance tests.

Usage:
    python -m csam_project.evaluation.variance_runner
    python -m csam_project.evaluation.variance_runner --seeds 42 43 44 45 46 --mode both
    python -m csam_project.evaluation.variance_runner --mode ablation --seeds 42 99 --quiet
"""

import sys
import os
import json
import time
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

import numpy as np

# Project path setup
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(project_root), ".env"), override=True)

logger = logging.getLogger(__name__)

DEFAULT_SEEDS = [42, 43, 44, 45, 46]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _aggregate_ablation(result_files: List[str]) -> Dict[str, Any]:
    """Aggregate per-seed ablation JSONs into mean/std/min/max per strategy."""
    per_seed = [_load_json(p) for p in result_files]

    # Collect strategy names from the first seed
    strategy_names = [r["strategy"] for r in per_seed[0]["results"]]

    aggregated: Dict[str, Dict[str, Any]] = {}
    for name in strategy_names:
        f1_values = []
        mem_values = []
        for seed_data in per_seed:
            for r in seed_data["results"]:
                if r["strategy"] == name:
                    f1_values.append(r["overall_f1"])
                    mem_values.append(r["memory_count"])
                    break

        aggregated[name] = {
            "f1_mean": float(np.mean(f1_values)),
            "f1_std": float(np.std(f1_values, ddof=1)) if len(f1_values) > 1 else 0.0,
            "f1_min": float(np.min(f1_values)),
            "f1_max": float(np.max(f1_values)),
            "f1_values": f1_values,
            "memory_mean": float(np.mean(mem_values)),
            "n_seeds": len(f1_values),
        }

    return aggregated


def _aggregate_sweep(result_files: List[str]) -> Dict[str, Any]:
    """Aggregate per-seed sweep JSONs into mean/std per threshold."""
    per_seed = [_load_json(p) for p in result_files]

    # No-Forgetting reference
    nf_f1s = [s["no_forgetting"]["overall_f1"] for s in per_seed]
    nf_agg = {
        "f1_mean": float(np.mean(nf_f1s)),
        "f1_std": float(np.std(nf_f1s, ddof=1)) if len(nf_f1s) > 1 else 0.0,
        "f1_values": nf_f1s,
    }

    # Per-threshold
    thresholds = [e["threshold"] for e in per_seed[0]["sweep"]]
    thresh_agg = {}
    for thresh in thresholds:
        f1s = []
        for seed_data in per_seed:
            for entry in seed_data["sweep"]:
                if entry["threshold"] == thresh:
                    f1s.append(entry["overall_f1"])
                    break
        thresh_agg[str(thresh)] = {
            "f1_mean": float(np.mean(f1s)),
            "f1_std": float(np.std(f1s, ddof=1)) if len(f1s) > 1 else 0.0,
            "f1_values": f1s,
        }

    return {"no_forgetting": nf_agg, "thresholds": thresh_agg}


def _significance_tests(
    aggregated: Dict[str, Dict[str, Any]],
    target: str = "Consolidation-Aware (Ours)",
) -> Dict[str, Dict[str, float]]:
    """Paired significance tests: target vs each baseline."""
    try:
        from scipy.stats import wilcoxon, ttest_rel
        has_scipy = True
    except ImportError:
        has_scipy = False

    if target not in aggregated:
        return {}

    target_vals = aggregated[target]["f1_values"]
    results: Dict[str, Dict[str, float]] = {}

    for name, data in aggregated.items():
        if name == target:
            continue
        baseline_vals = data["f1_values"]
        if len(target_vals) != len(baseline_vals):
            continue

        entry: Dict[str, float] = {}
        if not has_scipy:
            entry["note"] = "scipy not installed; skipping significance tests"
            results[name] = entry
            continue

        n = len(target_vals)
        try:
            if n >= 5:
                stat, p = wilcoxon(target_vals, baseline_vals)
                entry["test"] = "wilcoxon"
            else:
                stat, p = ttest_rel(target_vals, baseline_vals)
                entry["test"] = "paired_t"
            entry["statistic"] = round(float(stat), 6)
            entry["p_value"] = round(float(p), 6)
        except Exception as e:
            entry["error"] = str(e)

        results[name] = entry

    return results


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_variance(
    seeds: List[int],
    mode: str = "both",
    output_dir: str = None,
    llm_model: str = "llama-3.1-8b-instant",
    num_conversations: int = 5,
    interactions_per_conversation: int = 100,
    forget_threshold: int = 80,
    use_llm: bool = True,
    verbose: bool = True,
) -> None:
    """Run ablation/sweep across multiple seeds and aggregate results."""
    output_dir = output_dir or os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("CSAM VARIANCE RUNNER: Multi-Seed Evaluation")
    print("=" * 70)
    print(f"  Seeds:          {seeds}")
    print(f"  Mode:           {mode}")
    print(f"  Conversations:  {num_conversations}")
    print(f"  Interactions:   {interactions_per_conversation}")
    print(f"  Forget thresh:  {forget_threshold}")
    print(f"  LLM:            {llm_model if use_llm else 'disabled (word-overlap)'}")
    print(f"  Output dir:     {output_dir}")

    do_ablation = mode in ("ablation", "both")
    do_sweep = mode in ("sweep", "both")

    # ── Ablation across seeds ─────────────────────────────────
    ablation_files: List[str] = []
    if do_ablation:
        from evaluation.run_ablation import run_ablation_study

        print("\n" + "─" * 70)
        print("ABLATION RUNS")
        print("─" * 70)

        for i, seed in enumerate(seeds, 1):
            out_file = os.path.join(output_dir, f"ablation_seed{seed}.json")
            print(f"\n[{i}/{len(seeds)}] Ablation seed={seed}")
            t0 = time.time()

            run_ablation_study(
                num_conversations=num_conversations,
                interactions_per_conversation=interactions_per_conversation,
                forget_threshold=forget_threshold,
                output_file=out_file,
                use_llm=use_llm,
                llm_model=llm_model,
                verbose=verbose,
                seed=seed,
            )

            elapsed = time.time() - t0
            print(f"  Seed {seed} done in {elapsed:.0f}s → {out_file}")
            ablation_files.append(out_file)

        # Aggregate
        print("\n" + "─" * 70)
        print("ABLATION AGGREGATION")
        print("─" * 70)
        agg = _aggregate_ablation(ablation_files)
        sig = _significance_tests(agg)

        # Print summary table
        print(f"\n{'Strategy':<30} {'Mean F1':>10} {'± Std':>8} "
              f"{'Min':>8} {'Max':>8} {'p-value':>10}")
        print("-" * 74)
        for name, data in agg.items():
            p_str = ""
            if name in sig and "p_value" in sig[name]:
                p_str = f"{sig[name]['p_value']:.4f}"
            print(f"{name:<30} {data['f1_mean']:>10.4f} "
                  f"{'±':>1}{data['f1_std']:>7.4f} "
                  f"{data['f1_min']:>8.4f} {data['f1_max']:>8.4f} "
                  f"{p_str:>10}")

        # Check CA wins
        ca_key = "Consolidation-Aware (Ours)"
        if ca_key in agg:
            ca_vals = agg[ca_key]["f1_values"]
            wins = {}
            for name, data in agg.items():
                if name == ca_key:
                    continue
                w = sum(1 for c, b in zip(ca_vals, data["f1_values"]) if c > b)
                wins[name] = f"{w}/{len(ca_vals)}"
            print(f"\nCA wins per seed: {wins}")

        # Save
        variance_file = os.path.join(output_dir, "variance_ablation_results.json")
        variance_data = {
            "timestamp": datetime.now().isoformat(),
            "seeds": seeds,
            "config": {
                "num_conversations": num_conversations,
                "interactions_per_conversation": interactions_per_conversation,
                "forget_threshold": forget_threshold,
                "llm_model": llm_model if use_llm else None,
            },
            "aggregated": {k: {kk: vv for kk, vv in v.items()}
                          for k, v in agg.items()},
            "significance_tests": sig,
            "per_seed_files": ablation_files,
        }
        with open(variance_file, "w") as f:
            json.dump(variance_data, f, indent=2)
        print(f"\nVariance results saved to: {variance_file}")

    # ── Sweep across seeds ────────────────────────────────────
    sweep_files: List[str] = []
    if do_sweep:
        from evaluation.threshold_sweep import run_threshold_sweep

        print("\n" + "─" * 70)
        print("THRESHOLD SWEEP RUNS")
        print("─" * 70)

        for i, seed in enumerate(seeds, 1):
            out_file = os.path.join(output_dir, f"sweep_seed{seed}.json")
            print(f"\n[{i}/{len(seeds)}] Sweep seed={seed}")
            t0 = time.time()

            run_threshold_sweep(
                num_conversations=num_conversations,
                interactions_per_conversation=interactions_per_conversation,
                output_file=out_file,
                llm_model=llm_model,
                verbose=verbose,
                seed=seed,
            )

            elapsed = time.time() - t0
            print(f"  Seed {seed} done in {elapsed:.0f}s → {out_file}")
            sweep_files.append(out_file)

        # Aggregate
        print("\n" + "─" * 70)
        print("SWEEP AGGREGATION")
        print("─" * 70)
        sweep_agg = _aggregate_sweep(sweep_files)

        nf = sweep_agg["no_forgetting"]
        print(f"\nNo-Forgetting: F1 = {nf['f1_mean']:.4f} ± {nf['f1_std']:.4f}")
        print(f"\n{'Threshold':<12} {'Mean F1':>10} {'± Std':>8}")
        print("-" * 30)
        for thresh, data in sweep_agg["thresholds"].items():
            print(f"{thresh:<12} {data['f1_mean']:>10.4f} ±{data['f1_std']:>7.4f}")

        # Save
        variance_file = os.path.join(output_dir, "variance_sweep_results.json")
        variance_data = {
            "timestamp": datetime.now().isoformat(),
            "seeds": seeds,
            "config": {
                "num_conversations": num_conversations,
                "interactions_per_conversation": interactions_per_conversation,
                "llm_model": llm_model,
            },
            "aggregated": sweep_agg,
            "per_seed_files": sweep_files,
        }
        with open(variance_file, "w") as f:
            json.dump(variance_data, f, indent=2)
        print(f"\nVariance sweep results saved to: {variance_file}")

    print("\n" + "=" * 70)
    print("VARIANCE RUNNER COMPLETE")
    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="CSAM Multi-Seed Variance Runner"
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
        help=f"Seeds to run (default: {DEFAULT_SEEDS})",
    )
    parser.add_argument(
        "--mode", type=str, default="both",
        choices=["ablation", "sweep", "both"],
        help="Which evaluation to run (default: both)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory for output JSONs (default: evaluation/)",
    )
    parser.add_argument(
        "--model", type=str, default="llama-3.1-8b-instant",
        help="Groq model for QA (default: llama-3.1-8b-instant)",
    )
    parser.add_argument(
        "--conversations", type=int, default=5,
        help="Number of conversations (default: 5)",
    )
    parser.add_argument(
        "--interactions", type=int, default=100,
        help="Interactions per conversation (default: 100)",
    )
    parser.add_argument(
        "--threshold", type=int, default=80,
        help="Forget threshold for ablation (default: 80)",
    )
    parser.add_argument(
        "--no-llm", action="store_true",
        help="Disable LLM, use word-overlap fallback (for smoke testing)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Reduce per-question output",
    )

    args = parser.parse_args()

    run_variance(
        seeds=args.seeds,
        mode=args.mode,
        output_dir=args.output_dir,
        llm_model=args.model,
        num_conversations=args.conversations,
        interactions_per_conversation=args.interactions,
        forget_threshold=args.threshold,
        use_llm=not args.no_llm,
        verbose=not args.quiet,
    )
