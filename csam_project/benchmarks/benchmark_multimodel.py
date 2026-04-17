"""
Multi-Model LoCoMo Benchmark - Compare CSAM across different LLM backends.

Runs the LoCoMo benchmark across multiple conversations with multiple hosted
LLM providers/models to prove CSAM's architecture is model-agnostic.

Key fixes (PB-02, PB-04):
  - Outer conversation loop replaces data[0] hardcode
  - --max-conversations, --seed, --questions-per-conv args
  - Per-conversation checkpointing for Colab/Kaggle resilience
  - Bootstrap 95% CI across question-level F1
  - Protocol fingerprint block in every output JSON

Usage:
    python benchmarks/benchmark_multimodel.py \\
        --provider groq --model llama-3.1-8b-instant \\
        --max-conversations 5 --seed 42

    python benchmarks/benchmark_multimodel.py --all --max-conversations 5 --seed 42
"""

import json
import os
import sys
import re
import time
import random
import argparse
import numpy as np
from collections import Counter
from datetime import datetime
import logging

# Add project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from simulation.npc import NPC, NPCPersonality
from csam_core.services.embedding import EmbeddingService
from csam_core.services.llm_hosted import HostedLLMService, PROVIDERS, PUBLICATION_MODELS
from benchmarks.checkpoint import BenchmarkCheckpoint

# Load env from .env file if present
from dotenv import load_dotenv
env_path = os.path.join(os.path.dirname(project_root), ".env")
if os.path.exists(env_path):
    load_dotenv(env_path)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ── Model configurations ─────────────────────────────────────────────────────
# Sourced from PUBLICATION_MODELS in llm_hosted.py for consistency.
ALL_MODELS = [
    (m["provider"], m["model"], m["label"], f"{m['size_b']}B")
    for m in PUBLICATION_MODELS
]


# ── Evaluation helpers ───────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    text = text.lower()
    return re.sub(r"[^\w\s]", "", text)


def calculate_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(ground_truth).split()
    if not pred_tokens or not truth_tokens:
        return float(pred_tokens == truth_tokens)
    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def calculate_bleu1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(ground_truth).split()
    if not pred_tokens or not truth_tokens:
        return 0.0
    truth_counter = Counter(truth_tokens)
    clipped = sum(
        1 for t in pred_tokens
        if truth_counter[t] > 0 and not truth_counter.__setitem__(t, truth_counter[t] - 1)  # noqa: WPS220
    )
    # simpler version without the side-effect trick above:
    clipped = 0
    tc = Counter(truth_tokens)
    for t in pred_tokens:
        if tc[t] > 0:
            clipped += 1
            tc[t] -= 1
    return clipped / len(pred_tokens)


def sort_session_keys(keys: list) -> list:
    def extract_num(k: str) -> float:
        m = re.search(r"session_(\d+)", k)
        return int(m.group(1)) if m else float("inf")
    return sorted(
        [k for k in keys if "session_" in k and "date_time" not in k],
        key=extract_num,
    )


def get_session_date(conv_dict: dict, session_key: str) -> str:
    date_str = conv_dict.get(f"{session_key}_date_time", "")
    if date_str:
        match = re.search(r"on\s+(.+)", date_str)
        return match.group(1).strip() if match else date_str.strip()
    return ""


def bootstrap_ci(
    scores: list[float],
    n_samples: int = 1000,
    ci: float = 0.95,
    rng_seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap confidence interval for the mean of scores."""
    if not scores:
        return 0.0, 0.0
    rng = np.random.default_rng(rng_seed)
    boot_means = [
        float(np.mean(rng.choice(scores, size=len(scores), replace=True)))
        for _ in range(n_samples)
    ]
    lo = (1 - ci) / 2
    return float(np.quantile(boot_means, lo)), float(np.quantile(boot_means, 1 - lo))


# ── Single-conversation benchmark ────────────────────────────────────────────

def run_single_conversation(
    conv_data: dict,
    conv_index: int,
    embedding_service: EmbeddingService,
    llm_service: HostedLLMService,
    questions_per_conv: int | None = None,
    skip_consolidation: bool = True,
) -> dict | None:
    """
    Run LoCoMo QA for a single conversation.

    Returns a dict of per-conversation results, or None on fatal error.
    Services are passed in — callers share the embedding service across
    conversations for efficiency.
    """
    user_name = conv_data["conversation"].get("speaker_a", "User")
    npc_name = conv_data["conversation"].get("speaker_b", "Assistant")

    personality = NPCPersonality(
        name=npc_name,
        role="Helpful Assistant",
        traits=["friendly", "supportive", "good memory"],
        background=f"Assistant to {user_name}",
    )
    npc = NPC(
        personality=personality,
        embedding_service=embedding_service,
        llm_service=llm_service,
        max_memories=100_000,
    )

    # ── Phase 1: Ingest conversation history ────────────────────────────────
    conv_dict = conv_data["conversation"]
    session_keys = sort_session_keys(conv_dict.keys())
    total_turns = 0

    for session_key in session_keys:
        session_date = get_session_date(conv_dict, session_key)
        for turn in conv_dict[session_key]:
            speaker = turn.get("speaker", "Unknown")
            content = turn.get("text", "")
            if not content:
                continue
            total_turns += 1
            date_prefix = f"[{session_date}] " if session_date else ""
            memory_text = f"{date_prefix}{speaker}: {content}"
            importance = 0.6 if speaker == user_name else 0.5
            npc.add_memory(memory_text, importance=importance, metadata={
                "player_name": user_name,
                "speaker": speaker,
                "session": session_key,
                "date": session_date,
            })

    # ── Phase 1.5: Optional L3 consolidation ────────────────────────────────
    if not skip_consolidation and hasattr(npc, "consolidation_pipeline"):
        npc.consolidation_pipeline.min_memories_per_batch = 1
        npc.consolidation_pipeline.consolidation_threshold_hours = 0.0001
        npc.run_consolidation()

    # Clear L1 before QA to prevent ingestion turns from polluting context
    npc.working_memory.clear_all()

    # ── Phase 2: QA evaluation ───────────────────────────────────────────────
    qa_pairs = [qa for qa in conv_data.get("qa", []) if "answer" in qa]
    if questions_per_conv is not None:
        qa_pairs = qa_pairs[:questions_per_conv]

    f1_scores: list[float] = []
    bleu1_scores: list[float] = []
    latencies: list[float] = []
    qa_details: list[dict] = []

    for i, qa in enumerate(qa_pairs):
        question = qa["question"]
        truth = str(qa["answer"])
        t0 = time.time()

        query_embedding = embedding_service.encode(question)
        l2_results = npc.memory_repo.retrieve(query_embedding, k=20, update_access=False)

        context_parts = [f"- {mem.text}" for mem, _ in l2_results[:10]]
        context = "\n".join(context_parts) if context_parts else "No relevant memories."

        prediction = llm_service.generate_response(
            context=context,
            user_message=question,
            persona=None,
            mode="qa",
        )

        latency = (time.time() - t0) * 1000
        f1 = calculate_f1(prediction, truth)
        bleu1 = calculate_bleu1(prediction, truth)

        f1_scores.append(f1)
        bleu1_scores.append(bleu1)
        latencies.append(latency)

        top_ctx = l2_results[0][0].text[:80] if l2_results else "n/a"
        top_sim = f"{l2_results[0][1]:.3f}" if l2_results else "n/a"

        status = "[OK]" if f1 > 0.3 else "~" if f1 > 0 else "[FAIL]"
        print(
            f"    {status} Q{i+1} F1={f1:.3f} BLEU={bleu1:.3f} | {latency:.0f}ms"
            f"\n         Truth: '{truth[:60]}'"
            f"\n         Pred:  '{prediction[:60]}'"
            f"\n         Top:   '{top_ctx}' (sim={top_sim})"
        )

        qa_details.append({
            "question": question,
            "ground_truth": truth,
            "prediction": prediction,
            "f1": f1,
            "bleu1": bleu1,
            "latency_ms": latency,
            "context_preview": context[:300],
        })

    avg_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0
    ci_lo, ci_hi = bootstrap_ci(f1_scores)

    return {
        "conv_index": conv_index,
        "total_turns_ingested": total_turns,
        "l2_memories": len(npc.memory_repo),
        "l3_nodes": len(npc.knowledge_graph),
        "num_questions": len(f1_scores),
        "avg_f1": avg_f1,
        "avg_bleu1": float(np.mean(bleu1_scores)) if bleu1_scores else 0.0,
        "avg_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
        "f1_ci_95": [ci_lo, ci_hi],
        "f1_scores": f1_scores,
        "bleu1_scores": bleu1_scores,
        "qa_details": qa_details,
    }


# ── Multi-conversation orchestrator ─────────────────────────────────────────

def run_multi_conversation_benchmark(
    dataset_path: str,
    provider: str,
    model: str,
    display_name: str,
    max_conversations: int | None = None,
    questions_per_conv: int | None = None,
    seed: int = 42,
    checkpoint_dir: str | None = None,
    skip_consolidation: bool = True,
) -> dict | None:
    """Run LoCoMo benchmark over multiple conversations, with checkpointing."""
    print(f"\n{'='*70}")
    print(f"BENCHMARK: CSAM + {display_name} ({provider}) — LoCoMo")
    print(f"Model: {model} | seed={seed} | max_conv={max_conversations}")
    print(f"{'='*70}")

    with open(dataset_path, encoding="utf-8") as f:
        data: list = json.load(f)

    random.seed(seed)
    np.random.seed(seed)
    random.shuffle(data)

    if max_conversations is not None:
        data = data[:max_conversations]
    n_conv = len(data)
    print(f"Loaded {n_conv} conversations")

    # Services shared across all conversations
    print("Initializing services...")
    embedding_service = EmbeddingService(model_name="all-MiniLM-L6-v2")
    llm_service = HostedLLMService(provider=provider, model=model)

    if not llm_service.is_available():
        print(f"[FAIL] Cannot connect to {provider}/{model}")
        print(f"   Check: ${PROVIDERS[provider]['env_key']}")
        return None
    print(f"[OK] Connected to {provider} ({model})")

    ckpt = BenchmarkCheckpoint.for_run(
        benchmark="locomo",
        provider=provider,
        model=model,
        seed=seed,
        questions=n_conv,  # checkpoint granularity = conversation
        checkpoint_dir=checkpoint_dir,
    )
    if ckpt.num_completed() > 0:
        print(f"[RESUME] {ckpt.num_completed()}/{n_conv} conversations already done")

    conv_results: list[dict] = []

    for idx, conv_data in enumerate(data):
        conv_id = str(idx)
        if ckpt.is_done(conv_id):
            saved = ckpt.get_result(conv_id)
            conv_results.append(saved)
            print(f"  [SKIP] Conv {idx+1}/{n_conv} F1={saved['avg_f1']:.3f} (checkpoint)")
            continue

        print(f"\n  --- Conversation {idx+1}/{n_conv} ---")
        result = run_single_conversation(
            conv_data=conv_data,
            conv_index=idx,
            embedding_service=embedding_service,
            llm_service=llm_service,
            questions_per_conv=questions_per_conv,
            skip_consolidation=skip_consolidation,
        )
        if result is None:
            logger.warning("Skipping conversation %d — run_single_conversation returned None", idx)
            continue

        ckpt.add_result(conv_id, result)
        conv_results.append(result)

    if not conv_results:
        print("[FAIL] No conversations completed")
        return None

    # ── Aggregate across conversations ──────────────────────────────────────
    all_question_f1 = [f1 for r in conv_results for f1 in r["f1_scores"]]
    conv_f1_means = [r["avg_f1"] for r in conv_results]

    macro_f1 = float(np.mean(conv_f1_means))
    micro_f1 = float(np.mean(all_question_f1))
    ci_lo, ci_hi = bootstrap_ci(all_question_f1, rng_seed=seed)

    usage = llm_service.get_usage_stats()

    print(f"\n{'='*60}")
    print(f"RESULTS: CSAM + {display_name} — LoCoMo")
    print(f"{'='*60}")
    print(f"  Conversations:   {len(conv_results)}")
    print(f"  Total questions: {len(all_question_f1)}")
    print(f"  Macro F1:        {macro_f1:.4f}")
    print(f"  Micro F1:        {micro_f1:.4f}  [{ci_lo:.4f}, {ci_hi:.4f}] 95% CI")
    print(f"  API tokens:      {usage['total_tokens']}")
    print(f"{'='*60}")

    protocol = {
        "dataset": os.path.basename(dataset_path),
        "num_conversations": len(conv_results),
        "questions_per_conv": questions_per_conv,
        "retrieval_k": 20,
        "context_top_k": 10,
        "seed": seed,
        "skip_consolidation": skip_consolidation,
        "timestamp": datetime.now().isoformat(),
        "model_id": model,
        "provider": provider,
    }

    results = {
        "benchmark": "locomo",
        "timestamp": datetime.now().isoformat(),
        "provider": provider,
        "model": model,
        "display_name": display_name,
        "seed": seed,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "f1_ci_95": [ci_lo, ci_hi],
        "avg_bleu1": float(np.mean([r["avg_bleu1"] for r in conv_results])),
        "avg_latency_ms": float(np.mean([r["avg_latency_ms"] for r in conv_results])),
        "api_usage": usage,
        "protocol": protocol,
        "per_conversation": conv_results,
    }

    safe_model = model.replace("/", "_").replace(":", "_")
    out_path = os.path.join(
        project_root, "benchmarks",
        f"results_locomo_{provider}_{safe_model}_s{seed}.json",
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out_path}")

    ckpt.delete()
    return results


# ── Multi-model runner ───────────────────────────────────────────────────────

def run_all_models(
    dataset_path: str,
    max_conversations: int | None = 5,
    questions_per_conv: int | None = None,
    seed: int = 42,
    checkpoint_dir: str | None = None,
) -> list:
    """Run LoCoMo benchmark across all PUBLICATION_MODELS."""
    all_results = []

    print("\n" + "=" * 70)
    print("CSAM MULTI-MODEL COMPARATIVE BENCHMARK — LoCoMo")
    print(f"Models: {len(ALL_MODELS)} | convs: {max_conversations} | seed: {seed}")
    print("=" * 70)

    for provider, model, display_name, size in ALL_MODELS:
        try:
            result = run_multi_conversation_benchmark(
                dataset_path=dataset_path,
                provider=provider,
                model=model,
                display_name=display_name,
                max_conversations=max_conversations,
                questions_per_conv=questions_per_conv,
                seed=seed,
                checkpoint_dir=checkpoint_dir,
            )
            if result:
                all_results.append(result)
        except Exception:
            logger.exception("Benchmark failed for %s (%s/%s)", display_name, provider, model)

    if all_results:
        print("\n" + "=" * 80)
        print("COMPARATIVE SUMMARY — LoCoMo")
        print("=" * 80)
        print(f"{'Model':<28} {'Size':<6} {'Macro F1':>10} {'Micro F1':>10} {'CI 95%':>18} {'Tokens':>10}")
        print("-" * 80)
        for r in all_results:
            size = next((s for _, m, _, s in ALL_MODELS if m == r["model"]), "?")
            ci = r.get("f1_ci_95", [0, 0])
            print(
                f"{r['display_name']:<28} {size:<6}"
                f" {r['macro_f1']:>10.4f} {r['micro_f1']:>10.4f}"
                f" [{ci[0]:.4f},{ci[1]:.4f}]"
                f" {r['api_usage']['total_tokens']:>10}"
            )
        print("=" * 80)

        summary_path = os.path.join(project_root, "benchmarks", f"results_locomo_summary_s{seed}.json")
        summary = {
            "timestamp": datetime.now().isoformat(),
            "benchmark": "locomo_multimodel",
            "max_conversations": max_conversations,
            "seed": seed,
            "results": [
                {
                    "model": r["display_name"],
                    "provider": r["provider"],
                    "model_id": r["model"],
                    "macro_f1": r["macro_f1"],
                    "micro_f1": r["micro_f1"],
                    "f1_ci_95": r.get("f1_ci_95"),
                    "latency_ms": r["avg_latency_ms"],
                    "total_tokens": r["api_usage"]["total_tokens"],
                    "protocol": r.get("protocol", {}),
                }
                for r in all_results
            ],
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary saved: {summary_path}")

    return all_results


# ── Entry point ──────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="CSAM Multi-Model LoCoMo Benchmark")
    parser.add_argument("--provider", type=str, default="groq",
                        choices=list(PROVIDERS.keys()),
                        help="API provider (default: groq)")
    parser.add_argument("--model", type=str, default="llama-3.1-8b-instant",
                        help="Model ID (default: llama-3.1-8b-instant)")
    parser.add_argument("--all", action="store_true",
                        help="Run all PUBLICATION_MODELS sequentially")
    parser.add_argument("--dataset", type=str, default="benchmarks/data/locomo10.json",
                        help="Path to LoCoMo dataset")
    parser.add_argument("--max-conversations", type=int, default=5,
                        help="Max conversations to evaluate (default: 5)")
    parser.add_argument("--questions-per-conv", type=int, default=None,
                        help="QA questions per conversation (default: all)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--consolidate", action="store_true",
                        help="Enable L3 consolidation phase (slower)")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Directory for checkpoint files")

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    dataset_path = args.dataset
    if not os.path.exists(dataset_path):
        dataset_path = os.path.join(project_root, args.dataset)
    if not os.path.exists(dataset_path):
        print(f"[FAIL] Dataset not found: {args.dataset}")
        print("   Expected at: benchmarks/data/locomo10.json")
        return 1

    if args.all:
        run_all_models(
            dataset_path,
            max_conversations=args.max_conversations,
            questions_per_conv=args.questions_per_conv,
            seed=args.seed,
            checkpoint_dir=args.checkpoint_dir,
        )
    else:
        display_name = args.model.split("/")[-1] if "/" in args.model else args.model
        run_multi_conversation_benchmark(
            dataset_path=dataset_path,
            provider=args.provider,
            model=args.model,
            display_name=display_name,
            max_conversations=args.max_conversations,
            questions_per_conv=args.questions_per_conv,
            seed=args.seed,
            checkpoint_dir=args.checkpoint_dir,
            skip_consolidation=not args.consolidate,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
