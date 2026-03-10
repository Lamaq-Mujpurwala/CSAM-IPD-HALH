"""
Grid Search for Forgetting Formula Weights (α, β, γ, δ).

Searches over strategic weight combinations for ConsolidationAwareForgetting
to find the optimal formula:
    ForgetScore(m) = α·R(m) + β·(1-I(m)) + γ·C(m) + δ·D(m)

Only evaluates the CA strategy (not all 4), so each combo costs ~68 QA
calls (~22K tokens). With 20 combos and 2 API keys the full search fits
in a single free-tier Groq session.

Usage:
    python -m csam_project.evaluation.grid_search_weights
    python -m csam_project.evaluation.grid_search_weights --conversations 2 --threshold 40
    python -m csam_project.evaluation.grid_search_weights --output grid_results.json
"""

import sys
import os
import json
import time
import logging
from datetime import datetime
from typing import List, Tuple

# Project path setup
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(project_root), ".env"), override=True)

from csam_core.forgetting_engine import ConsolidationAwareForgetting
from csam_core.services.embedding import EmbeddingService
from csam_core.services.llm_hosted import HostedLLMService
from evaluation.npc_locomo import BenchmarkGenerator
from evaluation.run_ablation import evaluate_strategy

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────
# Strategic weight combinations (α, β, γ, δ) — all sum to 1.0
# α = recency, β = inverse importance, γ = consolidation, δ = redundancy
# ────────────────────────────────────────────────────────────
WEIGHT_GRID: List[Tuple[float, float, float, float]] = [
    # --- Baseline ---
    (0.20, 0.20, 0.30, 0.30),  # current default

    # --- Equal ---
    (0.25, 0.25, 0.25, 0.25),

    # --- Heavy consolidation (γ dominant) ---
    (0.10, 0.10, 0.50, 0.30),
    (0.10, 0.10, 0.60, 0.20),
    (0.05, 0.05, 0.50, 0.40),

    # --- Heavy redundancy (δ dominant) ---
    (0.10, 0.10, 0.30, 0.50),
    (0.10, 0.10, 0.20, 0.60),

    # --- Heavy importance (β dominant) ---
    (0.10, 0.40, 0.25, 0.25),
    (0.10, 0.50, 0.20, 0.20),

    # --- Heavy recency (α dominant) ---
    (0.40, 0.10, 0.25, 0.25),
    (0.50, 0.10, 0.20, 0.20),

    # --- Balanced novel (γ+δ emphasised) ---
    (0.15, 0.15, 0.35, 0.35),
    (0.10, 0.10, 0.40, 0.40),
    (0.05, 0.15, 0.40, 0.40),
    (0.15, 0.05, 0.40, 0.40),
    (0.05, 0.05, 0.45, 0.45),

    # --- Traditional-heavy (α+β emphasised) ---
    (0.30, 0.30, 0.20, 0.20),

    # --- Cross combinations ---
    (0.10, 0.30, 0.40, 0.20),  # importance + consolidation
    (0.30, 0.10, 0.20, 0.40),  # recency + redundancy
    (0.20, 0.10, 0.40, 0.30),  # low β, high γ
]


def run_grid_search(
    num_conversations: int = 5,
    interactions_per_conversation: int = 100,
    forget_threshold: int = 80,
    consolidation_threshold: float = 0.3,
    output_file: str | None = None,
    llm_model: str = "llama-3.1-8b-instant",
    verbose: bool = True,
) -> None:
    """Run grid search over weight combinations."""
    print("=" * 70)
    print("CSAM GRID SEARCH: Forgetting Weight Optimisation")
    print("=" * 70)
    print(f"  Conversations:  {num_conversations}")
    print(f"  Interactions:   {interactions_per_conversation}")
    print(f"  Forget thresh:  {forget_threshold}")
    print(f"  Consol thresh:  {consolidation_threshold}")
    print(f"  Weight combos:  {len(WEIGHT_GRID)}")
    print(f"  Model:          {llm_model}")

    # ── Generate dataset (once) ───────────────────────────────
    print("\nGenerating benchmark dataset...")
    generator = BenchmarkGenerator(seed=42)
    dataset = generator.generate_benchmark_dataset(
        num_conversations=num_conversations,
        interactions_per_conversation=interactions_per_conversation,
    )
    total_qa = sum(len(h.qa_pairs) for h in dataset)
    print(f"  {len(dataset)} conversations, {total_qa} Q&A pairs")

    # ── Load shared services ──────────────────────────────────
    print("\nLoading embedding model...")
    embedding_service = EmbeddingService()
    _ = embedding_service.dimension  # force load

    llm_service = HostedLLMService(provider="groq", model=llm_model)
    if not llm_service.is_available():
        print("  [ERROR] Groq not available. Set GROQ_API_KEY in .env")
        return
    n_keys = len(llm_service._api_keys)
    print(f"  [OK] Groq connected ({llm_model}), {n_keys} API key(s)")

    # ── Evaluate each weight combo ────────────────────────────
    combo_results = []
    best_f1 = -1.0
    best_combo = WEIGHT_GRID[0]

    for idx, (alpha, beta, gamma, delta) in enumerate(WEIGHT_GRID, 1):
        label = f"α={alpha} β={beta} γ={gamma} δ={delta}"
        print(f"\n[{idx}/{len(WEIGHT_GRID)}] {label}")

        strategy = ConsolidationAwareForgetting(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            delta=delta,
            consolidation_threshold=consolidation_threshold,
        )

        result = evaluate_strategy(
            strategy_name=f"CA({alpha},{beta},{gamma},{delta})",
            forgetting_strategy=strategy,
            dataset=dataset,
            embedding_service=embedding_service,
            forget_threshold=forget_threshold,
            llm_service=llm_service,
            verbose=verbose,
        )

        entry = {
            "weights": {"alpha": alpha, "beta": beta, "gamma": gamma, "delta": delta},
            "overall_f1": round(result.overall_f1, 4),
            "f1_scores": {k: round(v, 4) for k, v in result.f1_scores.items()},
            "memory_count": result.memory_count,
            "consolidation_ratio": round(result.consolidation_ratio, 4),
            "avg_latency_ms": round(result.avg_latency_ms, 2),
        }
        combo_results.append(entry)

        if result.overall_f1 > best_f1:
            best_f1 = result.overall_f1
            best_combo = (alpha, beta, gamma, delta)

        print(f"  → F1={result.overall_f1:.4f}  "
              f"single-hop={result.f1_scores.get('single-hop', 0):.3f}  "
              f"mem={result.memory_count}")

    # ── Summary ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("GRID SEARCH RESULTS (sorted by F1)")
    print("=" * 70)
    ranked = sorted(combo_results, key=lambda x: x["overall_f1"], reverse=True)
    print(f"\n{'Rank':<6}{'α':>6}{'β':>6}{'γ':>6}{'δ':>6}  {'F1':>8}  "
          f"{'S-Hop':>8}  {'Mem':>6}")
    print("-" * 62)
    for rank, entry in enumerate(ranked, 1):
        w = entry["weights"]
        print(f"{rank:<6}{w['alpha']:>6.2f}{w['beta']:>6.2f}"
              f"{w['gamma']:>6.2f}{w['delta']:>6.2f}  "
              f"{entry['overall_f1']:>8.4f}  "
              f"{entry['f1_scores'].get('single-hop', 0):>8.4f}  "
              f"{entry['memory_count']:>6}")

    a, b, g, d = best_combo
    print(f"\n★ Best: α={a} β={b} γ={g} δ={d}  →  F1={best_f1:.4f}")

    # ── Token usage ───────────────────────────────────────────
    usage = llm_service.get_usage_stats()
    print("\nAPI USAGE:")
    print(f"  Requests:  {usage['total_requests']}")
    print(f"  Tokens in: {usage['total_tokens_in']:,}")
    print(f"  Tokens out:{usage['total_tokens_out']:,}")
    print(f"  Total:     {usage['total_tokens']:,}")

    # ── Save ──────────────────────────────────────────────────
    if output_file is None:
        output_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "grid_search_results.json",
        )

    output_data = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "num_conversations": num_conversations,
            "interactions_per_conversation": interactions_per_conversation,
            "forget_threshold": forget_threshold,
            "consolidation_threshold": consolidation_threshold,
            "llm_model": llm_model,
            "num_combos": len(WEIGHT_GRID),
        },
        "best_weights": {
            "alpha": best_combo[0],
            "beta": best_combo[1],
            "gamma": best_combo[2],
            "delta": best_combo[3],
        },
        "best_f1": round(best_f1, 4),
        "results": ranked,
        "api_usage": usage,
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Grid search for CSAM forgetting weights"
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
        help="Forget threshold (default: 80)",
    )
    parser.add_argument(
        "--consol-threshold", type=float, default=0.3,
        help="Consolidation protection threshold (default: 0.3)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON file (default: grid_search_results.json)",
    )
    parser.add_argument(
        "--model", type=str, default="llama-3.1-8b-instant",
        help="Groq model for QA (default: llama-3.1-8b-instant)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Reduce per-question output",
    )

    args = parser.parse_args()

    run_grid_search(
        num_conversations=args.conversations,
        interactions_per_conversation=args.interactions,
        forget_threshold=args.threshold,
        consolidation_threshold=args.consol_threshold,
        output_file=args.output,
        llm_model=args.model,
        verbose=not args.quiet,
    )
