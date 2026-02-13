"""
Multi-Model LoCoMo Benchmark - Compare CSAM across different LLM backends.

Runs the same LoCoMo benchmark with different hosted LLM providers/models
to prove CSAM's architecture is model-agnostic and scales with model quality.

Usage:
    # Set API keys first (PowerShell):
    # $env:GROQ_API_KEY = "your_key"
    # $env:CEREBAS_API_KEY = "your_key"
    # $env:SAMBANOVA_API_KEY = "your_key"
    
    python benchmarks/benchmark_multimodel.py --provider groq --model llama-3.1-8b-instant
    python benchmarks/benchmark_multimodel.py --provider groq --model llama-3.3-70b-versatile
    python benchmarks/benchmark_multimodel.py --provider groq --model openai/gpt-oss-120b
    python benchmarks/benchmark_multimodel.py --provider groq --model qwen/qwen3-32b
    python benchmarks/benchmark_multimodel.py --provider cerebras --model llama-3.3-70b
    python benchmarks/benchmark_multimodel.py --all   # Run all models sequentially
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def normalize_text(text):
    """Normalize text for F1 evaluation."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text


def calculate_f1(prediction, ground_truth):
    """Calculate word-level F1 score."""
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(ground_truth).split()
    if not pred_tokens or not truth_tokens:
        return int(pred_tokens == truth_tokens)
    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def calculate_bleu1(prediction, ground_truth):
    """Calculate BLEU-1 (unigram precision) score."""
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(ground_truth).split()
    if not pred_tokens or not truth_tokens:
        return 0.0
    truth_counter = Counter(truth_tokens)
    clipped_count = 0
    for token in pred_tokens:
        if truth_counter[token] > 0:
            clipped_count += 1
            truth_counter[token] -= 1
    return clipped_count / len(pred_tokens) if pred_tokens else 0.0


def sort_session_keys(keys):
    """Sort session keys numerically."""
    def extract_num(k):
        m = re.search(r'session_(\d+)', k)
        return int(m.group(1)) if m else float('inf')
    return sorted([k for k in keys if 'session_' in k and 'date_time' not in k], key=extract_num)


def get_session_date(conv_dict, session_key):
    """Extract date string from session metadata."""
    # LoCoMo stores dates as e.g., 'session_1_date_time': '2:31 pm on 17 July, 2023'
    date_key = f"{session_key}_date_time"
    date_str = conv_dict.get(date_key, "")
    if date_str:
        # Extract just the date part: '17 July, 2023' from '2:31 pm on 17 July, 2023'
        match = re.search(r'on\s+(.+)', date_str)
        if match:
            return match.group(1).strip()
        return date_str.strip()
    return ""


# Define the model configurations to test
ALL_MODELS = [
    # (provider, model_id, display_name, model_size)
    ("groq", "llama-3.1-8b-instant", "Llama 3.1 8B", "8B"),
    ("groq", "meta-llama/llama-4-scout-17b-16e-instruct", "Llama 4 Scout 17B", "17B"),
    ("groq", "llama-3.3-70b-versatile", "Llama 3.3 70B", "70B"),
    ("groq", "openai/gpt-oss-120b", "GPT OSS 120B", "120B"),
]


def run_single_benchmark(
    dataset_path: str,
    provider: str,
    model: str,
    display_name: str,
    limit_questions: int = 10,
    skip_consolidation: bool = True
):
    """
    Run LoCoMo benchmark with a specific hosted model.
    
    Args:
        dataset_path: Path to locomo10.json
        provider: API provider name
        model: Model ID
        display_name: Human-readable model name
        limit_questions: Number of QA questions to evaluate
        skip_consolidation: If True, skip L3 consolidation (faster, tests L2 only)
    
    Returns:
        Dictionary of results
    """
    print(f"\n{'='*70}")
    print(f"BENCHMARK: CSAM + {display_name} ({provider})")
    print(f"Model: {model}")
    print(f"{'='*70}")
    
    # Load dataset
    print(f"Loading dataset from {dataset_path}...")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    conv_data = data[0]
    user_name = conv_data['conversation'].get('speaker_a', 'User')
    npc_name = conv_data['conversation'].get('speaker_b', 'Assistant')
    
    # Initialize services
    print("Initializing services...")
    embedding_service = EmbeddingService(model_name="all-MiniLM-L6-v2")
    
    # Create hosted LLM service
    llm_service = HostedLLMService(provider=provider, model=model)
    
    if not llm_service.is_available():
        print(f"[FAIL] Cannot connect to {provider} with model {model}")
        print(f"   Check API key: ${PROVIDERS[provider]['env_key']}")
        return None
    
    print(f"[OK] Connected to {provider} ({model})")
    
    # Create NPC with hosted LLM
    personality = NPCPersonality(
        name=npc_name,
        role="Helpful Assistant",
        traits=["friendly", "supportive", "good memory"],
        background=f"Assistant to {user_name}"
    )
    
    npc = NPC(
        personality=personality,
        embedding_service=embedding_service,
        llm_service=llm_service,
        max_memories=100000
    )
    
    # Phase 1: Ingest History
    # GENERAL APPROACH: Ingest ALL turns from both speakers with uniform format
    # - Prepend session date for temporal grounding
    # - Use "Speaker: content" format so both speakers' facts are searchable
    # - This is architecture-general: CSAM works regardless of who speaks
    print(f"\nPhase 1: Ingesting History for User: {user_name}")
    start_ingest = time.time()
    
    conv_dict = conv_data['conversation']
    npc_name = conv_data['conversation'].get('speaker_b', 'Melanie')
    session_keys = sort_session_keys(conv_dict.keys())
    total_turns = 0
    player_turns = 0
    
    for session_key in session_keys:
        session_date = get_session_date(conv_dict, session_key)
        turns = conv_dict[session_key]
        for turn in turns:
            speaker = turn.get('speaker', 'Unknown')
            content = turn.get('text', '')
            if not content:
                continue
            total_turns += 1
            
            # Uniform format: [date] Speaker: content
            date_prefix = f"[{session_date}] " if session_date else ""
            memory_text = f"{date_prefix}{speaker}: {content}"
            
            if speaker == user_name:
                npc.add_memory(memory_text, importance=0.6, metadata={
                    "player_name": user_name, "speaker": user_name,
                    "session": session_key, "date": session_date
                })
                player_turns += 1
            else:
                npc.add_memory(memory_text, importance=0.5, metadata={
                    "player_name": user_name, "speaker": npc_name,
                    "session": session_key, "date": session_date
                })
            
            if total_turns % 100 == 0:
                print(f"    Processed {total_turns} turns ({len(npc.memory_repo)} memories stored)...")
    
    ingest_time = time.time() - start_ingest
    print(f"Ingestion Complete. {total_turns} total turns, {player_turns} player memories stored")
    print(f"  Time: {ingest_time:.2f}s ({player_turns/ingest_time:.1f} memories/s)")
    print(f"  L2 Memories: {len(npc.memory_repo)}")
    
    # Phase 1.5: Optional Consolidation
    if not skip_consolidation and hasattr(npc, 'consolidation_pipeline'):
        print("\nPhase 1.5: Running Consolidation...")
        npc.consolidation_pipeline.min_memories_per_batch = 1
        npc.consolidation_pipeline.consolidation_threshold_hours = 0.0001
        result = npc.run_consolidation()
        print(f"Consolidation Complete. L3 Nodes: {len(npc.knowledge_graph)}")
    else:
        print("\nPhase 1.5: Skipping Consolidation (L2-only retrieval)")
    
    # FIX 3: Clear L1 working memory before QA — it's full of irrelevant recent ingestion turns
    npc.working_memory.clear_all()
    print("  L1 Working Memory cleared for QA phase")
    
    # Phase 2: Run QA
    # FIX 4: Use direct retrieval with k=10 instead of npc.respond() which uses k=5
    #         and also saves QA turns as memories (polluting the store mid-benchmark)
    print(f"\nPhase 2: Running QA Evaluation ({limit_questions} questions)")
    qa_pairs = conv_data['qa']
    if limit_questions:
        qa_pairs = qa_pairs[:limit_questions]
    
    f1_scores = []
    bleu1_scores = []
    latencies = []
    qa_details = []
    
    for i, qa in enumerate(qa_pairs):
        question = qa['question']
        truth = str(qa['answer'])
        
        t0 = time.time()
        
        # Direct retrieval with larger k — bypasses L1 pollution and npc.respond() side-effects
        query_embedding = embedding_service.encode(question)
        
        # Get more L2 candidates (k=20) for better recall
        l2_results = npc.memory_repo.retrieve(query_embedding, k=20, update_access=False)
        
        # Build context from top-10 most relevant memories (skip MMR diversity for factual QA)
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
        
        f1 = calculate_f1(prediction, truth)
        bleu1 = calculate_bleu1(prediction, truth)
        f1_scores.append(f1)
        bleu1_scores.append(bleu1)
        
        qa_details.append({
            "question": question,
            "ground_truth": truth,
            "prediction": prediction,
            "f1": f1,
            "bleu1": bleu1,
            "latency_ms": latency,
            "context_preview": context[:300]
        })
        
        status_icon = "[OK]" if f1 > 0.3 else "⚠️" if f1 > 0 else "[FAIL]"
        print(f"  {status_icon} Q{i+1}: F1={f1:.3f} BLEU1={bleu1:.3f} | {latency:.0f}ms")
        print(f"       Truth: '{truth[:60]}'")
        print(f"       Pred:  '{prediction[:60]}'")
        print(f"       Top context: '{l2_results[0][0].text[:80]}' (sim={l2_results[0][1]:.3f})")
    
    # Results
    avg_f1 = float(np.mean(f1_scores)) if f1_scores else 0
    avg_bleu1 = float(np.mean(bleu1_scores)) if bleu1_scores else 0
    avg_latency = float(np.mean(latencies)) if latencies else 0
    
    usage = llm_service.get_usage_stats()
    
    print(f"\n{'='*50}")
    print(f"RESULTS: CSAM + {display_name}")
    print(f"{'='*50}")
    print(f"  Average F1:      {avg_f1:.4f}")
    print(f"  Average BLEU-1:  {avg_bleu1:.4f}")
    print(f"  Average Latency: {avg_latency:.0f}ms")
    print(f"  Total API Calls: {usage['total_requests']}")
    print(f"  Total Tokens:    {usage['total_tokens']}")
    print(f"  L2 Memories:     {len(npc.memory_repo)}")
    print(f"  L3 Nodes:        {len(npc.knowledge_graph)}")
    print(f"{'='*50}")
    
    results = {
        "benchmark": "locomo",
        "timestamp": datetime.now().isoformat(),
        "provider": provider,
        "model": model,
        "display_name": display_name,
        "dataset": "locomo10.json",
        "conversation_id": 0,
        "num_questions": len(f1_scores),
        "avg_f1": avg_f1,
        "avg_bleu1": avg_bleu1,
        "avg_latency_ms": avg_latency,
        "total_turns_ingested": total_turns,
        "player_memories_stored": player_turns,
        "l2_memories": len(npc.memory_repo),
        "l3_nodes": len(npc.knowledge_graph),
        "api_usage": usage,
        "f1_scores": f1_scores,
        "bleu1_scores": bleu1_scores,
        "qa_details": qa_details
    }
    
    # Save results
    safe_model_name = model.replace("/", "_").replace(":", "_")
    output_file = os.path.join(project_root, "benchmarks", f"results_hosted_{provider}_{safe_model_name}.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")
    
    return results


def run_all_models(dataset_path: str, limit_questions: int = 10):
    """Run benchmark across all configured models."""
    all_results = []
    
    print("\n" + "=" * 70)
    print("CSAM MULTI-MODEL COMPARATIVE BENCHMARK")
    print(f"Testing {len(ALL_MODELS)} models on LoCoMo ({limit_questions} questions)")
    print("=" * 70)
    
    for provider, model, display_name, size in ALL_MODELS:
        try:
            result = run_single_benchmark(
                dataset_path=dataset_path,
                provider=provider,
                model=model,
                display_name=display_name,
                limit_questions=limit_questions
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
        print("COMPARATIVE SUMMARY")
        print("=" * 80)
        print(f"{'Model':<25} {'Size':<8} {'F1':>8} {'BLEU-1':>8} {'Latency':>10} {'API Calls':>10}")
        print("-" * 80)
        
        for r in all_results:
            size = next((s for _, m, _, s in ALL_MODELS if m == r['model']), "?")
            print(f"{r['display_name']:<25} {size:<8} {r['avg_f1']:>8.4f} {r['avg_bleu1']:>8.4f} {r['avg_latency_ms']:>8.0f}ms {r['api_usage']['total_requests']:>10}")
        
        # Also add our existing local results for comparison
        local_result_path = os.path.join(project_root, "benchmarks", "results_locomo_test.json")
        if os.path.exists(local_result_path):
            with open(local_result_path) as f:
                local = json.load(f)
            print(f"{'Llama 3.2 3B (Local)':<25} {'3B':<8} {local.get('avg_f1', 0):>8.4f} {'N/A':>8} {local.get('avg_latency_ms', 0):>8.0f}ms {'Local':>10}")
        
        print("=" * 80)
        
        # Save combined summary
        summary_path = os.path.join(project_root, "benchmarks", "results_multimodel_summary.json")
        summary = {
            "timestamp": datetime.now().isoformat(),
            "benchmark": "locomo_multimodel",
            "num_questions": limit_questions,
            "results": [
                {
                    "model": r['display_name'],
                    "provider": r['provider'],
                    "model_id": r['model'],
                    "f1": r['avg_f1'],
                    "bleu1": r['avg_bleu1'],
                    "latency_ms": r['avg_latency_ms'],
                    "total_tokens": r['api_usage']['total_tokens'],
                }
                for r in all_results
            ]
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary saved to: {summary_path}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="CSAM Multi-Model LoCoMo Benchmark")
    parser.add_argument("--provider", type=str, default="groq",
                       choices=list(PROVIDERS.keys()),
                       help="API provider (default: groq)")
    parser.add_argument("--model", type=str, default=None,
                       help="Model ID (provider-specific)")
    parser.add_argument("--questions", type=int, default=10,
                       help="Number of QA questions to evaluate (default: 10)")
    parser.add_argument("--all", action="store_true",
                       help="Run all preconfigured models")
    parser.add_argument("--consolidate", action="store_true",
                       help="Enable L3 consolidation (slower, more API calls)")
    parser.add_argument("--dataset", type=str, default="benchmarks/data/locomo10.json",
                       help="Path to LoCoMo dataset")
    
    args = parser.parse_args()
    
    # Check dataset
    dataset_path = args.dataset
    if not os.path.exists(dataset_path):
        # Try relative to project root
        dataset_path = os.path.join(project_root, args.dataset)
    
    if not os.path.exists(dataset_path):
        print(f"[FAIL] Dataset not found: {args.dataset}")
        print("   Expected at: benchmarks/data/locomo10.json")
        return 1
    
    if args.all:
        run_all_models(dataset_path, limit_questions=args.questions)
    else:
        model = args.model or PROVIDERS[args.provider]["models"].get("llama-8b", "llama-3.1-8b-instant")
        display_name = model.split("/")[-1] if "/" in model else model
        run_single_benchmark(
            dataset_path=dataset_path,
            provider=args.provider,
            model=model,
            display_name=display_name,
            limit_questions=args.questions,
            skip_consolidation=not args.consolidate
        )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
