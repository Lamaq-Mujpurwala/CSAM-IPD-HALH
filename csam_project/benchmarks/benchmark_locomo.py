
import json
import os
import sys
import re
import time
import numpy as np
from collections import Counter
import logging

# Add project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from simulation.npc import NPC, NPCPersonality
from csam_core.services.embedding import EmbeddingService
from csam_core.services.llm import LLMService

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MockLLMService:
    def is_available(self):
        return True # Pretend to be available so NPC calls generate
        
    def generate(self, prompt, system_prompt=None, temperature=0.7, max_tokens=150):
        # Return dummy response
        return "I processed that."

    def summarize(self, memories):
        # Return dummy summary
        return "Summary of memories."

    def extract_entities(self, text):
        # Return dummy entities
        return {"entities": [], "relationships": []}

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
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def sort_session_keys(keys):
    """Sort session keys numerically (session_1, session_2, ...)."""
    def extract_num(k):
        m = re.search(r'session_(\d+)', k)
        return int(m.group(1)) if m else float('inf')
    return sorted([k for k in keys if 'session_' in k and 'date_time' not in k], key=extract_num)

def run_benchmark(dataset_path, limit_questions=None):
    print(f"Loading dataset from {dataset_path}...")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # Process First Conversation Only (for testing)
    conv_data = data[0] 
    
    # 1. Setup NPC
    print("Initializing NPC...")
    embedding_service = EmbeddingService(model_name="all-MiniLM-L6-v2")
    
    # User is Caroline (from dataset inspection)
    user_name = conv_data['conversation'].get('speaker_a', 'User')
    npc_name = conv_data['conversation'].get('speaker_b', 'Assistant')
    
    personality = NPCPersonality(
        name=npc_name,
        role="Helpful Assistant",
        traits=["friendly", "supportive", "good memory"],
        background=f"Assistant to {user_name}"
    )
    
    # Try to connect to real LLM service (Ollama)
    # Configure here:
    OLLAMA_URL = "http://localhost:11434"
    OLLAMA_MODEL = "llama3.2:3b" # Default to 3b if available, or 1b
    
    real_llm = LLMService(base_url=OLLAMA_URL, model=OLLAMA_MODEL)
    
    if real_llm.is_available():
        print(f"[OK] Connected to Ollama at {OLLAMA_URL} using model {OLLAMA_MODEL}")
        llm_service = real_llm
    else:
        print(f"⚠ Could not connect to Ollama at {OLLAMA_URL}. Using Mock LLM.")
        print("  To enable Real LLM: Ensure Ollama is running and 'ollama serve' is active.")
        llm_service = MockLLMService()

    npc = NPC(
        personality=personality,
        embedding_service=embedding_service,
        llm_service=llm_service,
        max_memories=100000 
    )
    
    # 2. Ingest History
    print(f"\nPhase 1: Ingesting History for User: {user_name}")
    start_ingest = time.time()
    
    conv_dict = conv_data['conversation']
    session_keys = sort_session_keys(conv_dict.keys())
    
    total_turns = 0
    
    for session_key in session_keys:
        turns = conv_dict[session_key]
        
        for turn in turns:
            speaker = turn.get('speaker', 'Unknown')
            content = turn.get('text', '')
            
            if not content: continue
            
            if speaker == user_name: # User
                # Direct memory injection (skipping retrieval/generation for speed)
                npc.add_memory(content, importance=0.5, metadata={"player_name": user_name, "speaker": user_name})
            else: # Agent/Speaker_B
                # Inject agent's past response
                npc.add_memory(f"I said: {content}", importance=0.3, metadata={"player_name": user_name, "speaker": "self"})
            
            total_turns += 1
            if total_turns % 100 == 0:
                print(f"    Ingested {total_turns} turns...")
                
    ingest_time = time.time() - start_ingest
    ingest_time = time.time() - start_ingest
    print(f"Ingestion Complete. {total_turns} turns in {ingest_time:.2f}s ({total_turns/ingest_time:.1f} turns/s)")
    
    # 2.5 Force Aggressive Consolidation (L3 Enabler)
    print("\nPhase 1.5: Aggressive Consolidation (Creating Knowledge Graph)...")
    # 2.5 Force Aggressive Consolidation (L3 Enabler)
    print("\nPhase 1.5: Aggressive Consolidation (Creating Knowledge Graph)...")
    if hasattr(npc, 'consolidation_pipeline'):
        # Lower threshold to force consolidation of EVERYTHING
        npc.consolidation_pipeline.min_memories_per_batch = 1
        npc.consolidation_pipeline.consolidation_threshold_hours = 0.0001 # Force immediate consolidation
        
        # Run consolidation
        result = npc.run_consolidation()
        print(f"Consolidation Triggered: {result}")
        
        # Wait until graph is populated? (run_consolidation is synchronous in this implementation)
        print(f"Consolidation Complete. L3 Nodes: {len(npc.knowledge_graph)}")
    else:
        print("⚠ NPC has no consolidation pipeline.")
    
    # 3. Run QA
    print(f"\nPhase 2: Running QA Evaluation")
    qa_pairs = conv_data['qa']
    if limit_questions:
        qa_pairs = qa_pairs[:limit_questions]
        
    scores = []
    latencies = []
    
    print(f"Running {len(qa_pairs)} questions...")
    
    for i, qa in enumerate(qa_pairs):
        question = qa['question']
        truth = str(qa['answer'])
        
        # Measure Latency
        t0 = time.time()
        # Use QA mode for concise, factual answers (Optimization)
        response_data = npc.respond(question, player_name=user_name, mode="qa")
        latency = (time.time() - t0) * 1000
        latencies.append(latency)
        
        prediction = response_data['response']
        
        # Score
        f1 = calculate_f1(prediction, truth)
        scores.append(f1)
        
        if i % 5 == 0:
            print(f"  Q{i+1}: F1={f1:.2f} | Latency={latency:.0f}ms | Truth='{truth}' | Pred='{prediction[:50]}...'")

    # 4. Report
    avg_f1 = np.mean(scores) if scores else 0
    avg_latency = np.mean(latencies) if latencies else 0
    
    print("\n" + "="*40)
    print("BENCHMARK RESULTS (LoCoMo - Conversation 1)")
    print("="*40)
    print(f"Total Questions: {len(scores)}")
    print(f"Average F1 Score: {avg_f1:.4f}")
    print(f"Average Latency:  {avg_latency:.2f} ms")
    print(f"Ingestion Rate:   {total_turns/ingest_time:.1f} turns/s")
    print(f"L1 Cache Usage:   {len(npc.working_memory)} items")
    print(f"L2 Memories:      {len(npc.memory_repo)}")
    print(f"L3 Nodes:         {len(npc.knowledge_graph)}")
    print("="*40)
    
    # Save to file
    results = {
        "dataset": "locomo10.json",
        "conversation_id": 0,
        "avg_f1": avg_f1,
        "avg_latency_ms": avg_latency,
        "total_turns": total_turns,
        "metrics": scores
    }
    with open("benchmarks/results_locomo_test.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    dataset_file = "benchmarks/data/locomo10.json"
    if os.path.exists(dataset_file):
        run_benchmark(dataset_file, limit_questions=10) # Limit to 10 for quick test
    else:
        print(f"Dataset not found at {dataset_file}")
