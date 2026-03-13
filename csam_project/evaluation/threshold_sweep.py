"""
Threshold Sweep — Memory-Quality Tradeoff Curve.

Runs the Consolidation-Aware (CA) strategy at multiple forget thresholds
to map out how memory capacity affects QA accuracy (F1).

The sweep also runs No-Forgetting once as an upper-bound reference.

Usage:
    python -m csam_project.evaluation.threshold_sweep
    python -m csam_project.evaluation.threshold_sweep --thresholds 40 60 80 100 150
    python -m csam_project.evaluation.threshold_sweep --output sweep.json
"""

import sys
import os
import json
import time
import logging
from datetime import datetime
from typing import List

# Project path setup
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(project_root), ".env"), override=True)

from csam_core.forgetting_engine import (
    NoForgetting,
    ConsolidationAwareForgetting,
)
from csam_core.services.embedding import EmbeddingService
from csam_core.services.llm_hosted import HostedLLMService
from evaluation.npc_locomo import BenchmarkGenerator
from evaluation.run_ablation import evaluate_strategy

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLDS: List[int] = [50, 80, 100, 120, 150]


def run_threshold_sweep(
    thresholds: List[int] | None = None,
    num_conversations: int = 5,
    interactions_per_conversation: int = 100,
    output_file: str | None = None,
    llm_model: str = "llama-3.1-8b-instant",
    verbose: bool = True,
    seed: int = 42,
) -> None:
    """Run CA strategy at multiple thresholds + a No-Forgetting baseline."""
    thresholds = thresholds or DEFAULT_THRESHOLDS
    thresholds_sorted = sorted(thresholds)

    print("=" * 70)
    print("CSAM THRESHOLD SWEEP: Memory-Quality Tradeoff")
    print("=" * 70)
    print(f"  Conversations:  {num_conversations}")
    print(f"  Interactions:   {interactions_per_conversation}")
    print(f"  Thresholds:     {thresholds_sorted}")
    print(f"  CA weights:     α=0.25 β=0.25 γ=0.25 δ=0.25")
    print(f"  Model:          {llm_model}")

    # ── Dataset (single generation, reused) ───────────────────
    print("\nGenerating benchmark dataset...")
    generator = BenchmarkGenerator(seed=seed)
    dataset = generator.generate_benchmark_dataset(
        num_conversations=num_conversations,
        interactions_per_conversation=interactions_per_conversation,
    )
    total_qa = sum(len(h.qa_pairs) for h in dataset)
    print(f"  {len(dataset)} conversations, {total_qa} Q&A pairs")

    # ── Services ──────────────────────────────────────────────
    print("\nLoading embedding model...")
    embedding_service = EmbeddingService()
    _ = embedding_service.dimension

    llm_service = HostedLLMService(provider="groq", model=llm_model)
    if not llm_service.is_available():
        print("  [ERROR] Groq not available. Set GROQ_API_KEY in .env")
        return
    n_keys = len(llm_service._api_keys)
    print(f"  [OK] Groq connected ({llm_model}), {n_keys} API key(s)")

    # ── No-Forgetting reference ───────────────────────────────
    print("\n[0] Running No-Forgetting baseline (unbounded memory)...")
    t0 = time.time()
    nf_result = evaluate_strategy(
        strategy_name="No-Forgetting",
        forgetting_strategy=NoForgetting(),
        dataset=dataset,
        embedding_service=embedding_service,
        forget_threshold=99999,
        llm_service=llm_service,
        verbose=verbose,
        random_seed=seed,
        llm_seed=seed,
    )
    nf_elapsed = time.time() - t0
    print(f"  → F1={nf_result.overall_f1:.4f}  mem={nf_result.memory_count}  "
          f"({nf_elapsed:.0f}s)")

    # ── Sweep CA across thresholds ────────────────────────────
    sweep_results = []
    for idx, thresh in enumerate(thresholds_sorted, 1):
        print(f"\n[{idx}/{len(thresholds_sorted)}] "
              f"CA threshold={thresh}")
        t0 = time.time()

        strategy = ConsolidationAwareForgetting(
            alpha=0.25, beta=0.25, gamma=0.25, delta=0.25,
            consolidation_threshold=0.3,
        )

        result = evaluate_strategy(
            strategy_name=f"CA(thresh={thresh})",
            forgetting_strategy=strategy,
            dataset=dataset,
            embedding_service=embedding_service,
            forget_threshold=thresh,
            llm_service=llm_service,
            verbose=verbose,
            random_seed=seed,
            llm_seed=seed,
        )

        elapsed = time.time() - t0
        entry = {
            "threshold": thresh,
            "overall_f1": round(result.overall_f1, 4),
            "f1_scores": {k: round(v, 4) for k, v in result.f1_scores.items()},
            "memory_count": result.memory_count,
            "consolidation_ratio": round(result.consolidation_ratio, 4),
            "avg_latency_ms": round(result.avg_latency_ms, 2),
            "elapsed_s": round(elapsed, 1),
        }
        sweep_results.append(entry)

        print(f"  → F1={result.overall_f1:.4f}  "
              f"single-hop={result.f1_scores.get('single-hop', 0):.3f}  "
              f"mem={result.memory_count}  ({elapsed:.0f}s)")

    # ── Summary table ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("THRESHOLD SWEEP RESULTS")
    print("=" * 70)

    header = (f"{'Threshold':<12}{'F1':>8}{'S-Hop':>8}{'M-Hop':>8}"
              f"{'Temp':>8}{'Adv':>8}{'Mem':>8}{'Lat(ms)':>10}")
    print(f"\n{header}")
    print("-" * 70)

    # No-Forgetting row
    nf_f1 = nf_result.f1_scores
    print(f"{'∞ (no-fgt)':<12}"
          f"{nf_result.overall_f1:>8.4f}"
          f"{nf_f1.get('single-hop', 0):>8.4f}"
          f"{nf_f1.get('multi-hop', 0):>8.4f}"
          f"{nf_f1.get('temporal', 0):>8.4f}"
          f"{nf_f1.get('adversarial', 0):>8.4f}"
          f"{nf_result.memory_count:>8}"
          f"{nf_result.avg_latency_ms:>10.2f}")

    for entry in sweep_results:
        f1s = entry["f1_scores"]
        print(f"{entry['threshold']:<12}"
              f"{entry['overall_f1']:>8.4f}"
              f"{f1s.get('single-hop', 0):>8.4f}"
              f"{f1s.get('multi-hop', 0):>8.4f}"
              f"{f1s.get('temporal', 0):>8.4f}"
              f"{f1s.get('adversarial', 0):>8.4f}"
              f"{entry['memory_count']:>8}"
              f"{entry['avg_latency_ms']:>10.2f}")

    # Retention ratio
    if nf_result.overall_f1 > 0:
        print("\nRetention ratio (CA F1 / No-Forgetting F1):")
        for entry in sweep_results:
            ratio = entry["overall_f1"] / nf_result.overall_f1
            mem_ratio = entry["memory_count"] / nf_result.memory_count
            print(f"  thresh={entry['threshold']:<5} "
                  f"F1-retention={ratio:.1%}  "
                  f"mem-reduction={1 - mem_ratio:.1%}")

    # Token usage
    usage = llm_service.get_usage_stats()
    print("\nAPI USAGE:")
    print(f"  Requests:  {usage['total_requests']}")
    print(f"  Tokens:    {usage['total_tokens']:,}")

    # ── Save ──────────────────────────────────────────────────
    if output_file is None:
        output_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "threshold_sweep_results.json",
        )

    output_data = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "num_conversations": num_conversations,
            "interactions_per_conversation": interactions_per_conversation,
            "thresholds": thresholds_sorted,
            "ca_weights": {"alpha": 0.25, "beta": 0.25,
                           "gamma": 0.25, "delta": 0.25},
            "consolidation_threshold": 0.3,
            "llm_model": llm_model,
            "seed": seed,
        },
        "no_forgetting": {
            "overall_f1": round(nf_result.overall_f1, 4),
            "f1_scores": {k: round(v, 4)
                          for k, v in nf_result.f1_scores.items()},
            "memory_count": nf_result.memory_count,
        },
        "sweep": sweep_results,
        "api_usage": usage,
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="CSAM threshold sweep for memory-quality tradeoff"
    )
    parser.add_argument(
        "--thresholds", type=int, nargs="+",
        default=DEFAULT_THRESHOLDS,
        help=f"Threshold values to test (default: {DEFAULT_THRESHOLDS})",
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
        "--output", type=str, default=None,
        help="Output JSON file",
    )
    parser.add_argument(
        "--model", type=str, default="llama-3.1-8b-instant",
        help="Groq model for QA (default: llama-3.1-8b-instant)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Reduce per-question output",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    run_threshold_sweep(
        thresholds=args.thresholds,
        num_conversations=args.conversations,
        interactions_per_conversation=args.interactions,
        output_file=args.output,
        llm_model=args.model,
        verbose=not args.quiet,
        seed=args.seed,
    )
