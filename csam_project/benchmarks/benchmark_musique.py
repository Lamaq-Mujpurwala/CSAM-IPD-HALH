"""
MuSiQue Multi-Hop QA Benchmark - CSAM Architecture Validation

Tests CSAM's ability to perform multi-hop reasoning over factual paragraphs.
MuSiQue (Multi-hop Sequential Question Answering) requires chaining 2-4 reasoning
steps across multiple paragraphs to arrive at the final answer.

Architecture Pattern (from benchmark_multimodel.py critical fixes):
  1. Ingest paragraphs as memories with "[Title] content" prefix (temporal grounding analog)
  2. Clear L1 working memory before QA phase
  3. Use direct retrieval (k=20) — do NOT use npc.respond()
  4. Skip MMR diversity for factual QA (high lambda = pure relevance)

Usage:
    python benchmarks/benchmark_musique.py --provider groq --model llama-3.1-8b-instant
    python benchmarks/benchmark_musique.py --provider groq --model llama-3.3-70b-versatile
    python benchmarks/benchmark_musique.py --all
"""

import json
import os
import sys
import re
import time
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
from csam_core.services.llm_hosted import HostedLLMService, PROVIDERS

# Load env from .env file if present
from dotenv import load_dotenv
env_path = os.path.join(os.path.dirname(project_root), ".env")
if os.path.exists(env_path):
    load_dotenv(env_path)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ── Evaluation Metrics ──────────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    """Normalize text for F1 evaluation (lowercase, strip punctuation)."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text


def calculate_f1(prediction: str, ground_truth: str) -> float:
    """Calculate word-level F1 score."""
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(ground_truth).split()
    if not pred_tokens or not truth_tokens:
        return int(pred_tokens == truth_tokens)
    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def calculate_em(prediction: str, ground_truth: str) -> float:
    """Calculate Exact Match score."""
    return float(normalize_text(prediction).strip() == normalize_text(ground_truth).strip())


def calculate_f1_with_aliases(prediction: str, ground_truth: str, aliases: list) -> float:
    """Calculate best F1 across ground truth and all aliases."""
    candidates = [ground_truth] + (aliases or [])
    return max(calculate_f1(prediction, c) for c in candidates)


def calculate_em_with_aliases(prediction: str, ground_truth: str, aliases: list) -> float:
    """Calculate best EM across ground truth and all aliases."""
    candidates = [ground_truth] + (aliases or [])
    return max(calculate_em(prediction, c) for c in candidates)


# ── Model Configurations ────────────────────────────────────────────────────────

ALL_MODELS = [
    # (provider, model_id, display_name, model_size)
    ("groq", "llama-3.1-8b-instant", "Llama 3.1 8B", "8B"),
    ("groq", "meta-llama/llama-4-scout-17b-16e-instruct", "Llama 4 Scout 17B", "17B"),
    ("groq", "llama-3.3-70b-versatile", "Llama 3.3 70B", "70B"),
]


# ── Core Benchmark Logic ────────────────────────────────────────────────────────

def run_musique_benchmark(
    dataset_path: str,
    provider: str,
    model: str,
    display_name: str,
    limit_questions: int = 50,
):
    """
    Run MuSiQue multi-hop QA benchmark with a specific hosted model.

    For EACH question:
      1. Create a fresh NPC (isolated memory per question — MuSiQue provides
         separate paragraph sets per question, unlike LoCoMo's shared conversation).
      2. Ingest all paragraphs as L2 memories with "[Title] content" format.
      3. Clear L1, then do direct k=20 retrieval + LLM QA.

    Returns:
        Dictionary of results.
    """
    print(f"\n{'='*70}")
    print(f"BENCHMARK: CSAM + {display_name} ({provider}) — MuSiQue")
    print(f"Model: {model}")
    print(f"{'='*70}")

    # Load dataset
    print(f"Loading dataset from {dataset_path}...")
    entries = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    print(f"Loaded {len(entries)} entries")

    # Filter to answerable only
    entries = [e for e in entries if e.get('answerable', True)]
    if limit_questions and limit_questions < len(entries):
        entries = entries[:limit_questions]
    print(f"Evaluating {len(entries)} answerable questions")

    # Initialize shared services
    print("Initializing services...")
    embedding_service = EmbeddingService(model_name="all-MiniLM-L6-v2")
    llm_service = HostedLLMService(provider=provider, model=model)

    if not llm_service.is_available():
        print(f"[FAIL] Cannot connect to {provider} with model {model}")
        print(f"   Check API key: ${PROVIDERS[provider]['env_key']}")
        return None
    print(f"[OK] Connected to {provider} ({model})")

    # ── Run QA ──────────────────────────────────────────────────────────────────
    print(f"\nRunning QA Evaluation ({len(entries)} questions)...")

    f1_scores = []
    em_scores = []
    latencies = []
    qa_details = []
    hop_f1 = {}  # Track F1 by hop count

    for i, entry in enumerate(entries):
        question = entry['question']
        truth = str(entry['answer'])
        aliases = entry.get('answer_aliases', [])
        paragraphs = entry.get('paragraphs', [])
        entry_id = entry.get('id', f'q_{i}')

        # Determine hop count from ID (e.g., "2hop__..." -> "2hop")
        hop_key = entry_id.split('__')[0] if '__' in entry_id else 'unknown'

        t0 = time.time()

        # Create a fresh NPC per question (MuSiQue gives separate paragraph sets)
        personality = NPCPersonality(
            name="MuSiQue_Agent",
            role="Knowledge Base",
            traits=["precise", "factual"],
            background="Multi-hop QA agent"
        )
        npc = NPC(
            personality=personality,
            embedding_service=embedding_service,
            llm_service=llm_service,
            max_memories=100000,
        )

        # ── Ingest paragraphs as L2 memories ──
        # CRITICAL FIX 1 (temporal grounding analog): prefix with [Title] for context
        # CRITICAL FIX 2 (uniform ingestion): all paragraphs ingested uniformly
        for para in paragraphs:
            title = para.get('title', '')
            text = para.get('paragraph_text', '')
            if not text:
                continue
            memory_text = f"[{title}] {text}" if title else text
            is_supporting = para.get('is_supporting', False)
            npc.add_memory(memory_text, importance=0.7 if is_supporting else 0.5, metadata={
                "title": title,
                "is_supporting": is_supporting,
                "paragraph_idx": para.get('idx', -1),
            })

        # CRITICAL FIX 3: Clear L1 working memory before QA
        npc.working_memory.clear_all()

        # CRITICAL FIX 4: Direct retrieval with k=20, skip npc.respond()
        query_embedding = embedding_service.encode(question)
        l2_results = npc.memory_repo.retrieve(query_embedding, k=20, update_access=False)

        # Build context from top-10 most relevant memories
        context_parts = []
        for mem, score in l2_results[:10]:
            context_parts.append(f"- {mem.text}")
        context = "\n".join(context_parts) if context_parts else "No relevant memories."

        # Generate answer using hosted LLM
        prediction = llm_service.generate_response(
            context=context,
            user_message=question,
            persona=None,
            mode="qa"
        )

        latency = (time.time() - t0) * 1000
        latencies.append(latency)

        f1 = calculate_f1_with_aliases(prediction, truth, aliases)
        em = calculate_em_with_aliases(prediction, truth, aliases)
        f1_scores.append(f1)
        em_scores.append(em)

        # Track by hop count
        if hop_key not in hop_f1:
            hop_f1[hop_key] = []
        hop_f1[hop_key].append(f1)

        qa_details.append({
            "id": entry_id,
            "hop": hop_key,
            "question": question,
            "ground_truth": truth,
            "aliases": aliases,
            "prediction": prediction,
            "f1": f1,
            "em": em,
            "latency_ms": latency,
            "num_paragraphs": len(paragraphs),
            "num_supporting": sum(1 for p in paragraphs if p.get('is_supporting', False)),
            "context_preview": context[:300],
        })

        status_icon = "[OK]" if f1 > 0.5 else "⚠️" if f1 > 0 else "[FAIL]"
        print(f"  {status_icon} Q{i+1}/{len(entries)} [{hop_key}] F1={f1:.3f} EM={em:.0f} | {latency:.0f}ms")
        print(f"       Truth: '{truth[:60]}'")
        print(f"       Pred:  '{prediction[:60]}'")

    # ── Results ─────────────────────────────────────────────────────────────────
    avg_f1 = float(np.mean(f1_scores)) if f1_scores else 0
    avg_em = float(np.mean(em_scores)) if em_scores else 0
    avg_latency = float(np.mean(latencies)) if latencies else 0

    hop_averages = {k: float(np.mean(v)) for k, v in hop_f1.items()}

    usage = llm_service.get_usage_stats()

    print(f"\n{'='*60}")
    print(f"RESULTS: CSAM + {display_name} — MuSiQue Multi-Hop QA")
    print(f"{'='*60}")
    print(f"  Average F1:      {avg_f1:.4f}")
    print(f"  Average EM:      {avg_em:.4f}")
    print(f"  Average Latency: {avg_latency:.0f}ms")
    print(f"  Total API Calls: {usage['total_requests']}")
    print(f"  Total Tokens:    {usage['total_tokens']}")
    print(f"\n  F1 by Hop Count:")
    for hop, avg in sorted(hop_averages.items()):
        count = len(hop_f1[hop])
        print(f"    {hop}: {avg:.4f} (n={count})")
    print(f"{'='*60}")

    results = {
        "benchmark": "musique",
        "timestamp": datetime.now().isoformat(),
        "provider": provider,
        "model": model,
        "display_name": display_name,
        "dataset": os.path.basename(dataset_path),
        "num_questions": len(f1_scores),
        "avg_f1": avg_f1,
        "avg_em": avg_em,
        "avg_latency_ms": avg_latency,
        "hop_f1": hop_averages,
        "hop_counts": {k: len(v) for k, v in hop_f1.items()},
        "api_usage": usage,
        "f1_scores": f1_scores,
        "em_scores": em_scores,
        "qa_details": qa_details,
    }

    # Save results
    safe_model_name = model.replace("/", "_").replace(":", "_")
    output_file = os.path.join(project_root, "benchmarks", f"results_musique_{provider}_{safe_model_name}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    return results


def run_all_models(dataset_path: str, limit_questions: int = 50):
    """Run MuSiQue benchmark across all configured models."""
    all_results = []

    print("\n" + "=" * 70)
    print("CSAM MULTI-MODEL COMPARATIVE BENCHMARK — MuSiQue")
    print(f"Testing {len(ALL_MODELS)} models ({limit_questions} questions each)")
    print("=" * 70)

    for provider, model, display_name, size in ALL_MODELS:
        try:
            result = run_musique_benchmark(
                dataset_path=dataset_path,
                provider=provider,
                model=model,
                display_name=display_name,
                limit_questions=limit_questions,
            )
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"\n[FAIL] FAILED: {display_name} ({provider}/{model})")
            print(f"   Error: {e}")
            logger.exception(f"Benchmark failed for {model}")

    # Summary table
    if all_results:
        print("\n" + "=" * 80)
        print("COMPARATIVE SUMMARY — MuSiQue Multi-Hop QA")
        print("=" * 80)
        print(f"{'Model':<25} {'Size':<8} {'F1':>8} {'EM':>8} {'Latency':>10} {'Tokens':>10}")
        print("-" * 80)

        for r in all_results:
            size = next((s for _, m, _, s in ALL_MODELS if m == r['model']), "?")
            print(f"{r['display_name']:<25} {size:<8} {r['avg_f1']:>8.4f} {r['avg_em']:>8.4f} {r['avg_latency_ms']:>8.0f}ms {r['api_usage']['total_tokens']:>10}")

        print("=" * 80)

        # Save combined summary
        summary_path = os.path.join(project_root, "benchmarks", "results_musique_summary.json")
        summary = {
            "timestamp": datetime.now().isoformat(),
            "benchmark": "musique_multimodel",
            "num_questions": limit_questions,
            "results": [
                {
                    "model": r['display_name'],
                    "provider": r['provider'],
                    "model_id": r['model'],
                    "f1": r['avg_f1'],
                    "em": r['avg_em'],
                    "latency_ms": r['avg_latency_ms'],
                    "total_tokens": r['api_usage']['total_tokens'],
                    "hop_f1": r.get('hop_f1', {}),
                }
                for r in all_results
            ]
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary saved to: {summary_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="CSAM MuSiQue Multi-Hop QA Benchmark")
    parser.add_argument("--provider", type=str, default="groq",
                        choices=list(PROVIDERS.keys()),
                        help="API provider (default: groq)")
    parser.add_argument("--model", type=str, default=None,
                        help="Model ID (provider-specific)")
    parser.add_argument("--questions", type=int, default=50,
                        help="Number of questions to evaluate (default: 50)")
    parser.add_argument("--all", action="store_true",
                        help="Run all preconfigured models")
    parser.add_argument("--dataset", type=str, default="benchmarks/data/musique_dev.jsonl",
                        help="Path to MuSiQue dataset")

    args = parser.parse_args()

    # Resolve dataset path
    dataset_path = args.dataset
    if not os.path.exists(dataset_path):
        dataset_path = os.path.join(project_root, args.dataset)
    if not os.path.exists(dataset_path):
        print(f"[FAIL] Dataset not found: {args.dataset}")
        print("   Expected at: benchmarks/data/musique_dev.jsonl")
        return 1

    if args.all:
        run_all_models(dataset_path, limit_questions=args.questions)
    else:
        model = args.model or "llama-3.1-8b-instant"
        display_name = model.split("/")[-1] if "/" in model else model
        run_musique_benchmark(
            dataset_path=dataset_path,
            provider=args.provider,
            model=model,
            display_name=display_name,
            limit_questions=args.questions,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
