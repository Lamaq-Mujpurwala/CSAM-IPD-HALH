"""
Baseline RAG Benchmark — Hosted LLM, L2-only flat retrieval.

This is the apples-to-apples comparison agent for PB-03.
It uses IDENTICAL settings to benchmark_multimodel.py:
  - Same embedding model (all-MiniLM-L6-v2)
  - Same hosted LLM provider and model (Groq)
  - Same retrieval budget (k=20, top-10 context)
  - Same ingestion format ([date] Speaker: content)
  - Same QA prompt (mode="qa")
  - Same seed-based conversation shuffle

The ONLY architectural difference from CSAM:
  - No L1 working memory
  - No L3 knowledge graph
  - No consolidation pipeline
  - No forgetting engine  ← flat HNSW-only retrieval

Protocol fingerprint is identical to benchmark_multimodel.py output so
compare_csam_vs_baseline.py can validate parity before computing deltas.

Usage:
    python benchmarks/benchmark_baseline_rag_hosted.py \\
        --provider groq --model llama-3.1-8b-instant \\
        --max-conversations 5 --seed 42
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import random
import argparse
import numpy as np
from collections import Counter
from datetime import datetime
import logging

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from csam_core.memory_repository import MemoryRepository
from csam_core.services.embedding import EmbeddingService
from csam_core.services.llm_hosted import HostedLLMService, PROVIDERS, PUBLICATION_MODELS
from benchmarks.checkpoint import BenchmarkCheckpoint
import benchmarks.metrics as metrics_module

from dotenv import load_dotenv
env_path = os.path.join(os.path.dirname(project_root), ".env")
if os.path.exists(env_path):
    load_dotenv(env_path, override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

ALL_MODELS = [
    (m["provider"], m["model"], m["label"], f"{m['size_b']}B")
    for m in PUBLICATION_MODELS
]

# ── Metrics (duplicated from benchmark_multimodel for script independence) ───

def normalize_text(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text.lower())


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
    tc = Counter(truth_tokens)
    clipped = 0
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
    if not scores:
        return 0.0, 0.0
    rng = np.random.default_rng(rng_seed)
    boot_means = [
        float(np.mean(rng.choice(scores, size=len(scores), replace=True)))
        for _ in range(n_samples)
    ]
    lo = (1 - ci) / 2
    return float(np.quantile(boot_means, lo)), float(np.quantile(boot_means, 1 - lo))


# ── Flat RAG agent (no L1/L3/consolidation/forgetting) ──────────────────────

class FlatRAGAgent:
    """
    Minimal flat-HNSW retrieval agent — the baseline for PB-03.

    Architecturally identical to CSAM except:
    - No L1 working memory cache
    - No L3 knowledge graph
    - No consolidation pipeline
    - No forgetting engine
    All memories added go straight to L2 (HNSW) and stay there.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        max_memories: int = 100_000,
    ) -> None:
        self.embedding_service = embedding_service
        self.memory_repo = MemoryRepository(
            embedding_dim=embedding_service.dimension,
            max_memories=max_memories,
        )

    def add_memory(self, text: str, importance: float = 0.5) -> None:
        embedding = self.embedding_service.encode(text)
        self.memory_repo.add(text, embedding, importance)

    def retrieve(self, query: str, k: int = 20) -> list:
        query_embedding = self.embedding_service.encode(query)
        return self.memory_repo.retrieve(query_embedding, k=k, update_access=False)

    def reset(self) -> None:
        """Clear all memories between conversations."""
        self.memory_repo = MemoryRepository(
            embedding_dim=self.embedding_service.dimension,
            max_memories=self.memory_repo.max_memories,
        )


# ── Single-conversation benchmark ───────────────────────────────────────────

def run_single_conversation(
    conv_data: dict,
    conv_index: int,
    embedding_service: EmbeddingService,
    llm_service: HostedLLMService,
    questions_per_conv: int | None = None,
) -> dict | None:
    """Run baseline RAG QA for one LoCoMo conversation."""
    user_name = conv_data["conversation"].get("speaker_a", "User")
    agent = FlatRAGAgent(embedding_service=embedding_service)

    # ── Ingest (same format as CSAM benchmark) ──────────────────────────────
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
            importance = 0.6 if speaker == user_name else 0.5
            agent.add_memory(
                f"{date_prefix}{speaker}: {content}",
                importance=importance,
            )

    # ── QA (same retrieval budget as CSAM benchmark) ─────────────────────────
    qa_pairs = [qa for qa in conv_data.get("qa", []) if "answer" in qa]
    if questions_per_conv is not None:
        qa_pairs = qa_pairs[:questions_per_conv]

    f1_scores: list[float] = []
    sem_scores: list[float] = []
    bleu1_scores: list[float] = []
    latencies: list[float] = []
    qa_details: list[dict] = []

    for i, qa in enumerate(qa_pairs):
        question = qa["question"]
        truth = str(qa["answer"])
        t0 = time.time()

        l2_results = agent.retrieve(question, k=20)
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
        sem = metrics_module.semantic_f1(prediction, truth, embedding_service.encode)
        f1_scores.append(f1)
        sem_scores.append(sem)
        bleu1_scores.append(bleu1)
        latencies.append(latency)

        status = "[OK]" if f1 > 0.3 else "~" if f1 > 0 else "[FAIL]"
        print(
            f"    {status} Q{i+1} F1={f1:.3f} Sem={sem:.3f} | {latency:.0f}ms"
            f"\n         Truth: '{truth[:60]}'"
            f"\n         Pred:  '{prediction[:60]}'"
        )

        qa_details.append({
            "question": question,
            "ground_truth": truth,
            "prediction": prediction,
            "f1": f1,
            "semantic_sim": sem,
            "bleu1": bleu1,
            "latency_ms": latency,
            "context_preview": context[:300],
        })

    avg_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0
    avg_sem = float(np.mean(sem_scores)) if sem_scores else 0.0
    ci_lo, ci_hi = bootstrap_ci(f1_scores)

    return {
        "conv_index": conv_index,
        "total_turns_ingested": total_turns,
        "l2_memories": len(agent.memory_repo),
        "num_questions": len(f1_scores),
        "avg_f1": avg_f1,
        "avg_semantic_sim": avg_sem,
        "avg_bleu1": float(np.mean(bleu1_scores)) if bleu1_scores else 0.0,
        "avg_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
        "f1_ci_95": [ci_lo, ci_hi],
        "f1_scores": f1_scores,
        "semantic_sim_scores": sem_scores,
        "bleu1_scores": bleu1_scores,
        "qa_details": qa_details,
    }


# ── Multi-conversation orchestrator ─────────────────────────────────────────

def run_baseline_benchmark(
    dataset_path: str,
    provider: str,
    model: str,
    display_name: str,
    max_conversations: int | None = None,
    questions_per_conv: int | None = None,
    seed: int = 42,
    checkpoint_dir: str | None = None,
    output_dir: str | None = None,
) -> dict | None:
    """Run flat-RAG baseline over multiple LoCoMo conversations."""
    print(f"\n{'='*70}")
    print(f"BASELINE RAG: {display_name} ({provider}) — LoCoMo")
    print(f"Model: {model} | seed={seed} | max_conv={max_conversations}")
    print(f"Architecture: L2-only flat HNSW (no L1/L3/consolidation/forgetting)")
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

    print("Initializing services...")
    embedding_service = EmbeddingService(model_name="all-MiniLM-L6-v2")
    llm_service = HostedLLMService(provider=provider, model=model)

    if not llm_service.is_available():
        print(f"[FAIL] Cannot connect to {provider}/{model}")
        return None
    print(f"[OK] Connected to {provider} ({model})")

    ckpt = BenchmarkCheckpoint.for_run(
        benchmark="locomo_baseline",
        provider=provider,
        model=model,
        seed=seed,
        questions=n_conv,
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
        )
        if result is None:
            logger.warning("Skipping conversation %d", idx)
            continue

        ckpt.add_result(conv_id, result)
        conv_results.append(result)

    if not conv_results:
        print("[FAIL] No conversations completed")
        return None

    all_question_f1 = [f1 for r in conv_results for f1 in r["f1_scores"]]
    conv_f1_means = [r["avg_f1"] for r in conv_results]
    macro_f1 = float(np.mean(conv_f1_means))
    micro_f1 = float(np.mean(all_question_f1))
    ci_lo, ci_hi = bootstrap_ci(all_question_f1, rng_seed=seed)

    usage = llm_service.get_usage_stats()

    print(f"\n{'='*60}")
    print(f"BASELINE RESULTS: {display_name} — LoCoMo")
    print(f"{'='*60}")
    print(f"  Conversations:   {len(conv_results)}")
    print(f"  Total questions: {len(all_question_f1)}")
    print(f"  Macro F1:        {macro_f1:.4f}")
    print(f"  Micro F1:        {micro_f1:.4f}  [{ci_lo:.4f}, {ci_hi:.4f}] 95% CI")
    print(f"{'='*60}")

    # Protocol fingerprint — MUST match benchmark_multimodel.py structure exactly
    protocol = {
        "dataset": os.path.basename(dataset_path),
        "num_conversations": len(conv_results),
        "questions_per_conv": questions_per_conv,
        "retrieval_k": 20,
        "context_top_k": 10,
        "seed": seed,
        "skip_consolidation": True,   # N/A for baseline, set True for parity check
        "timestamp": datetime.now().isoformat(),
        "model_id": model,
        "provider": provider,
        "architecture": "flat_rag_l2_only",
    }

    results = {
        "benchmark": "locomo_baseline",
        "timestamp": datetime.now().isoformat(),
        "provider": provider,
        "model": model,
        "display_name": display_name,
        "seed": seed,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "f1_ci_95": [ci_lo, ci_hi],
        "avg_bleu1": float(np.mean([r["avg_bleu1"] for r in conv_results])),
        "avg_semantic_sim": float(np.mean([r["avg_semantic_sim"] for r in conv_results])),
        "avg_latency_ms": float(np.mean([r["avg_latency_ms"] for r in conv_results])),
        "api_usage": usage,
        "protocol": protocol,
        "per_conversation": conv_results,
    }

    safe_model = model.replace("/", "_").replace(":", "_")
    save_dir = output_dir if output_dir else os.path.join(project_root, "benchmarks")
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"results_locomo_baseline_{provider}_{safe_model}_s{seed}.json")
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
    output_dir: str | None = None,
) -> list:
    all_results = []
    print("\n" + "=" * 70)
    print("BASELINE RAG MULTI-MODEL BENCHMARK — LoCoMo")
    print(f"Models: {len(ALL_MODELS)} | convs: {max_conversations} | seed: {seed}")
    print("=" * 70)

    for provider, model, display_name, size in ALL_MODELS:
        try:
            result = run_baseline_benchmark(
                dataset_path=dataset_path,
                provider=provider,
                model=model,
                display_name=display_name,
                max_conversations=max_conversations,
                questions_per_conv=questions_per_conv,
                seed=seed,
                checkpoint_dir=checkpoint_dir,
                output_dir=output_dir,
            )
            if result:
                all_results.append(result)
        except Exception:
            logger.exception("Baseline failed for %s (%s/%s)", display_name, provider, model)

    if all_results:
        print("\n" + "=" * 70)
        print("BASELINE SUMMARY")
        print("=" * 70)
        print(f"{'Model':<28} {'Size':<6} {'Macro F1':>10} {'Micro F1':>10}")
        print("-" * 70)
        for r in all_results:
            size = next((s for _, m, _, s in ALL_MODELS if m == r["model"]), "?")
            print(f"{r['display_name']:<28} {size:<6} {r['macro_f1']:>10.4f} {r['micro_f1']:>10.4f}")

    return all_results


# ── Entry point ──────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Baseline RAG LoCoMo Benchmark")
    parser.add_argument("--provider", type=str, default="groq",
                        choices=list(PROVIDERS.keys()))
    parser.add_argument("--model", type=str, default="llama-3.1-8b-instant")
    parser.add_argument("--all", action="store_true",
                        help="Run all PUBLICATION_MODELS")
    parser.add_argument("--dataset", type=str,
                        default="benchmarks/data/locomo10.json")
    parser.add_argument("--max-conversations", type=int, default=5)
    parser.add_argument("--questions-per-conv", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save results (default: benchmarks/)")

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    dataset_path = args.dataset
    if not os.path.exists(dataset_path):
        dataset_path = os.path.join(project_root, args.dataset)
    if not os.path.exists(dataset_path):
        print(f"[FAIL] Dataset not found: {args.dataset}")
        return 1

    if args.all:
        run_all_models(
            dataset_path,
            max_conversations=args.max_conversations,
            questions_per_conv=args.questions_per_conv,
            seed=args.seed,
            checkpoint_dir=args.checkpoint_dir,
            output_dir=args.output_dir,
        )
    else:
        display_name = args.model.split("/")[-1] if "/" in args.model else args.model
        run_baseline_benchmark(
            dataset_path=dataset_path,
            provider=args.provider,
            model=args.model,
            display_name=display_name,
            max_conversations=args.max_conversations,
            questions_per_conv=args.questions_per_conv,
            seed=args.seed,
            checkpoint_dir=args.checkpoint_dir,
            output_dir=args.output_dir,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
