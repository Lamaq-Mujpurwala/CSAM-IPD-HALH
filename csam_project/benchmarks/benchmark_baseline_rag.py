
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

from csam_core.services.embedding import EmbeddingService
from csam_core.services.llm import LLMService
# Use MemoryRepository as our "Vector DB"
from csam_core.memory_repository import MemoryRepository

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BaselineAgent:
    """
    A "Standard RAG" agent.
    - No L1 (Working Memory)
    - No L3 (Knowledge Graph)
    - No Consolidation
    - Just Vector Search + LLM Generation
    """
    def __init__(self, embedding_service, llm_service):
        self.embedding_service = embedding_service
        self.llm_service = llm_service
        self.memory_repo = MemoryRepository(
            embedding_dim=embedding_service.dimension,
            max_memories=100000
        )
        
    def add_memory(self, text):
        embedding = self.embedding_service.encode(text)
        self.memory_repo.add(text, embedding, importance=0.5)
        
    def respond(self, query):
        # 1. Standard Vector Retrieval (k=5)
        query_embedding = self.embedding_service.encode(query)
        results = self.memory_repo.retrieve(query_embedding, k=5)
        
        # 2. Construct Prompt (Standard RAG Style)
        context_text = "\n".join([f"- {r[0].text}" for r in results])
        
        # Use strict specific QA prompt like the optimized NPC to keep comparison fair on the prompt side
        prompt = f"""Answer the question based ONLY on the context below. Be extremely concise.

Context:
{context_text}

Question: {query}

Answer:"""
        
        system = "You are a precise database. Output only the requested date, name, or fact."
        
        if self.llm_service and self.llm_service.is_available():
            return self.llm_service.generate(prompt, system_prompt=system, temperature=0.1, max_tokens=150)
        else:
            return "LLM Unavailable"

class MockLLMService:
    def is_available(self): return True
    def generate(self, *args, **kwargs): return "Mock Response"

def normalize_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def calculate_f1(prediction, ground_truth):
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(ground_truth).split()
    if not pred_tokens or not truth_tokens: return int(pred_tokens == truth_tokens)
    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())
    if num_same == 0: return 0
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(truth_tokens)
    return (2 * precision * recall) / (precision + recall)

def sort_session_keys(keys):
    def extract_num(k):
        m = re.search(r'session_(\d+)', k)
        return int(m.group(1)) if m else float('inf')
    return sorted([k for k in keys if 'session_' in k and 'date_time' not in k], key=extract_num)

def run_baseline(dataset_path, limit_questions=None):
    print(f"Loading dataset from {dataset_path}...")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    conv_data = data[0]
    user_name = conv_data['conversation'].get('speaker_a', 'User')
    
    # 1. Setup Baseline Agent
    print("Initializing Baseline RAG Agent...")
    embedding_service = EmbeddingService(model_name="all-MiniLM-L6-v2")
    
    # Use same Real LLM configuration
    OLLAMA_URL = "http://localhost:11434"
    OLLAMA_MODEL = "llama3.2:3b"
    real_llm = LLMService(base_url=OLLAMA_URL, model=OLLAMA_MODEL)
    
    if real_llm.is_available():
        print(f"[OK] Connected to Ollama ({OLLAMA_MODEL})")
        llm_service = real_llm
    else:
        print("⚠ Using Mock LLM")
        llm_service = MockLLMService()
        
    agent = BaselineAgent(embedding_service, llm_service)
    
    # 2. Ingest History
    print(f"\nPhase 1: Ingesting History (Standard RAG)...")
    start_ingest = time.time()
    
    conv_dict = conv_data['conversation']
    session_keys = sort_session_keys(conv_dict.keys())
    total_turns = 0
    
    for session_key in session_keys:
        turns = conv_dict[session_key]
        for turn in turns:
            content = turn.get('text', '')
            if content:
                # Naive ingestion: just add everything to vector store
                agent.add_memory(content)
                total_turns += 1
                
    ingest_time = time.time() - start_ingest
    print(f"Ingestion Complete. {total_turns} turns in {ingest_time:.2f}s ({total_turns/ingest_time:.1f} turns/s)")
    
    # 3. Run QA
    print(f"\nPhase 2: Running QA (Baseline)...")
    qa_pairs = conv_data['qa']
    if limit_questions: qa_pairs = qa_pairs[:limit_questions]
    
    scores = []
    latencies = []
    
    for i, qa in enumerate(qa_pairs):
        question = qa['question']
        truth = str(qa['answer'])
        
        t0 = time.time()
        prediction = agent.respond(question)
        latency = (time.time() - t0) * 1000
        latencies.append(latency)
        
        f1 = calculate_f1(prediction, truth)
        scores.append(f1)
        
        if i % 5 == 0:
            print(f"  Q{i+1}: F1={f1:.2f} | Latency={latency:.0f}ms | Truth='{truth}' | Pred='{prediction[:50]}...'")
            
    # 4. Report
    avg_f1 = np.mean(scores) if scores else 0
    avg_latency = np.mean(latencies) if latencies else 0
    
    print("\n" + "="*40)
    print("BASELINE RESULTS (Standard RAG)")
    print("="*40)
    print(f"Average F1 Score: {avg_f1:.4f}")
    print(f"Average Latency:  {avg_latency:.2f} ms")
    print("="*40)
    
    # Save results
    results = {
        "dataset": "locomo10.json",
        "avg_f1": avg_f1,
        "avg_latency_ms": avg_latency,
        "metrics": scores
    }
    with open("benchmarks/results_baseline_rag.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    dataset_file = "benchmarks/data/locomo10.json"
    if os.path.exists(dataset_file):
        run_baseline(dataset_file, limit_questions=10)
    else:
        print("Dataset not found")
