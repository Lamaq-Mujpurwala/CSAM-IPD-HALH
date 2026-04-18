# CSAM Benchmark Testing Guide

**Repo:** https://github.com/Lamaq-Mujpurwala/CSAM-IPD-HALH.git  
**Purpose:** Run all publication benchmarks across 6 independent notebooks and collect JSON result files.  
**Split:** Two developers can each take 3 notebooks and run them in parallel on separate Kaggle/Colab accounts.

---

## Quick Reference

| Notebook | What it tests | API intensity | Time est. |
|---|---|---|---|
| NB-1 `nb1_locomo_benchmark.ipynb` | CSAM vs Flat-RAG, 4 models | HIGH (4+4 model runs) | 3–6 h |
| NB-2 `nb2_hotpotqa.ipynb` | HotPotQA 100Q, 4 models + 8B seeds | HIGH | 2–3 h |
| NB-3 `nb3_musique.ipynb` | MuSiQue 100Q, 4 models + 8B seeds + no-L3 | HIGH | 2–3 h |
| NB-4 `nb4_ablation.ipynb` | 5 forgetting strategies, 5 model/seed combos | VERY HIGH | 3–5 h |
| NB-5 `nb5_scaling.ipynb` | NPC scaling sweep, mostly no API | LOW | 15–60 min |
| NB-6 `nb6_question_scaling.ipynb` | 50/100/200/500Q sweep (8B only) | MEDIUM | 2–3 h |

---

## Recommended Developer Split

### Developer A — NB-1, NB-2, NB-4
Covers the core CSAM vs baseline comparison, multi-hop QA, and the ablation study.  
Needs accounts for all 4 models (8B is the main workhorse).

### Developer B — NB-3, NB-5, NB-6
Covers MuSiQue multi-hop + L3 isolation, scaling, and question-count robustness.  
Lighter on big-model API calls (NB-5 uses no API at all).

> All 6 notebooks are completely independent — there are no shared output files or dependencies between them. You can assign any split as long as each developer takes 3.

---

## Groq Account Setup

### Models and Rate Limits (Groq Free Tier)

| Model | Groq model ID | RPM | RPD | TPM |
|---|---|---|---|---|
| Llama 3.1 8B | `llama-3.1-8b-instant` | 30 | 14,400 | 6,000 |
| Llama 4 Scout 17B | `meta-llama/llama-4-scout-17b-16e-instruct` | 30 | 1,000 | 30,000 |
| Llama 3.3 70B | `llama-3.3-70b-versatile` | 30 | 1,000 | 12,000 |
| GPT-OSS 120B | `openai/gpt-oss-120b` | 30 | 1,000 | 8,000 |

**RPM** = requests per minute · **RPD** = requests per day · **TPM** = tokens per minute

### Critical: Limits Are Per Organization, Not Per Key

Multiple API keys from the **same Groq account** share the same daily pool. Adding 5 keys from one account does NOT multiply your RPD — it only helps with RPM bursting (parallel requests within a minute).

To get more daily quota, you need **separate Groq accounts** (different email addresses).

### How Many Accounts You Need

| Model | RPD | Calls needed (full suite) | Accounts needed |
|---|---|---|---|
| 8B | 14,400 | ~2,500 (all 8B runs) | 1 account |
| Scout 17B | 1,000 | ~600 | 1 account (may need 2 days) |
| Llama 70B | 1,000 | ~500 | 1 account (may need 2 days) |
| GPT-OSS 120B | 1,000 | ~400 | 1 account |

**Minimum recommended:** 1 primary account + 1 backup account per developer.  
**Ideal:** 2–3 accounts per developer for the 70B/Scout/120B models, which hit the 1K RPD limit fast.

### Getting API Keys

1. Sign up at https://console.groq.com (free, no credit card)
2. Go to **API Keys** → **Create API Key**
3. Copy the key (starts with `gsk_...`)
4. For multiple keys on the same account: create up to 5 keys (they share the org quota but help with RPM)

### Key Rotation in Notebooks

The notebooks auto-discover multiple keys in this priority order:

```
GROQ_API_KEY        ← primary key (required)
GROQ_API_KEY_2      ← rotated to on 429
GROQ_API_KEY_3
...
GROQ_API_KEY_9
```

When a 429 rate-limit error hits, the service automatically rotates to the next available key and respects the `Retry-After` header. You don't need to manually handle this.

---

## Per-Notebook Configuration

Each notebook has a **Step 3 — Configure** cell with all the parameters you need to set. Default values are already publication-quality — only change them if needed.

---

### NB-1: LoCoMo — CSAM vs Flat-RAG Baseline

**File:** `csam_project/nb1_locomo_benchmark.ipynb`  
**Output dir:** `results/nb1_locomo/`  
**Dataset:** `csam_project/benchmarks/data/locomo10.json` (already in repo)

#### Config cell parameters

```python
MAX_CONVERSATIONS = 5      # 3 = quick test (~10 min), 5 = default, 10 = publication quality
QUESTIONS_PER_CONV = None  # None = all questions per conversation
SEED = 42
```

#### What it runs (automatically, in sequence)

1. CSAM benchmark on all 4 models (`benchmark_multimodel --all --consolidate`)
2. Flat-RAG baseline on all 4 models (`benchmark_baseline_rag_hosted --all`)
3. Results summary table

#### Expected output files

```
results/nb1_locomo/results_locomo_csam_groq_llama-3.1-8b-instant_s42.json
results/nb1_locomo/results_locomo_csam_groq_meta-llama_llama-4-scout-17b-16e-instruct_s42.json
results/nb1_locomo/results_locomo_csam_groq_llama-3.3-70b-versatile_s42.json
results/nb1_locomo/results_locomo_csam_groq_openai_gpt-oss-120b_s42.json
results/nb1_locomo/results_locomo_baseline_groq_llama-3.1-8b-instant_s42.json
results/nb1_locomo/results_locomo_baseline_groq_meta-llama_llama-4-scout-17b-16e-instruct_s42.json
results/nb1_locomo/results_locomo_baseline_groq_llama-3.3-70b-versatile_s42.json
results/nb1_locomo/results_locomo_baseline_groq_openai_gpt-oss-120b_s42.json
```

#### Keys needed

| Model | Key requirement |
|---|---|
| 8B | GROQ_API_KEY |
| Scout 17B | GROQ_API_KEY (same or different) |
| 70B | GROQ_API_KEY |
| GPT-OSS 120B | GROQ_API_KEY |

Run 8B and 120B first (they have higher RPD), then Scout/70B (1K RPD limit — may need a second day or second account).

---

### NB-2: HotPotQA — Multi-Model + 8B Variance Seeds

**File:** `csam_project/nb2_hotpotqa.ipynb`  
**Output dir:** `results/nb2_hotpotqa/`  
**Dataset:** `csam_project/benchmarks/data/hotpotqa_dev.json` (already in repo, 7405 Q)

#### Config cell parameters

```python
N_QUESTIONS = 100   # 50 = quick test, 100 = publication quality
```

#### What it runs

1. All 4 models at 100Q, seed 42
2. 8B at 100Q, seed 123
3. 8B at 100Q, seed 456

Total API calls: ~600 on 8B, ~100 each on Scout/70B/120B.

#### Expected output files

```
results/nb2_hotpotqa/results_hotpotqa_groq_llama-3.1-8b-instant_s42_n100.json
results/nb2_hotpotqa/results_hotpotqa_groq_meta-llama_llama-4-scout-17b-16e-instruct_s42_n100.json
results/nb2_hotpotqa/results_hotpotqa_groq_llama-3.3-70b-versatile_s42_n100.json
results/nb2_hotpotqa/results_hotpotqa_groq_openai_gpt-oss-120b_s42_n100.json
results/nb2_hotpotqa/results_hotpotqa_groq_llama-3.1-8b-instant_s123_n100.json
results/nb2_hotpotqa/results_hotpotqa_groq_llama-3.1-8b-instant_s456_n100.json
```

---

### NB-3: MuSiQue — Multi-Model + L3 Isolation

**File:** `csam_project/nb3_musique.ipynb`  
**Output dir:** `results/nb3_musique/`  
**Dataset:** `csam_project/benchmarks/data/musique_dev.jsonl` (already in repo, ~500 Q)

#### Config cell parameters

```python
N_QUESTIONS = 100   # 50 = quick test, 100 = publication quality
```

#### What it runs

1. All 4 models at 100Q WITH L3, seed 42
2. 8B at 100Q WITH L3, seed 123
3. 8B at 100Q WITH L3, seed 456
4. 8B at 100Q WITHOUT L3, seed 42  ← isolates knowledge-graph contribution

The L3 ablation delta (run 1 8B vs run 4) proves the value of the knowledge graph for multi-hop QA.

#### Expected output files

```
results/nb3_musique/results_musique_groq_llama-3.1-8b-instant_s42_n100.json
results/nb3_musique/results_musique_groq_meta-llama_llama-4-scout-17b-16e-instruct_s42_n100.json
results/nb3_musique/results_musique_groq_llama-3.3-70b-versatile_s42_n100.json
results/nb3_musique/results_musique_groq_openai_gpt-oss-120b_s42_n100.json
results/nb3_musique/results_musique_groq_llama-3.1-8b-instant_s123_n100.json
results/nb3_musique/results_musique_groq_llama-3.1-8b-instant_s456_n100.json
results/nb3_musique/results_musique_groq_llama-3.1-8b-instant_nol3_s42_n100.json
```

---

### NB-4: Ablation Study — 5 Forgetting Strategies

**File:** `csam_project/nb4_ablation.ipynb`  
**Output dir:** `results/nb4_ablation/`  
**Dataset:** Synthetic (generated internally — no external dataset file needed)

#### Config cell parameters

```python
CONVERSATIONS = 5    # conversations per run (5 = default)
INTERACTIONS  = 50   # memory events per conversation (50 = fast, 100 = paper quality)
THRESHOLD     = 80   # eviction trigger: forget when L2 has > 80 entries
```

#### What it runs

| Run tag | Model | Seed | Est. API calls |
|---|---|---|---|
| `8b_s42` | Llama 3.1 8B | 42 | ~250 |
| `8b_s123` | Llama 3.1 8B | 123 | ~250 |
| `8b_s456` | Llama 3.1 8B | 456 | ~250 |
| `scout17b_s42` | Llama 4 Scout 17B | 42 | ~250 |
| `70b_s42` | Llama 3.3 70B | 42 | ~250 |

Each run tests all 5 strategies: No-Forgetting, LRU, Importance-Only, CA-Formula-Only, CA-Ours.  
A run is skipped if its output file already exists (safe to re-run after interruption).

#### Expected output files

```
results/nb4_ablation/ablation_8b_s42.json
results/nb4_ablation/ablation_8b_s123.json
results/nb4_ablation/ablation_8b_s456.json
results/nb4_ablation/ablation_scout17b_s42.json
results/nb4_ablation/ablation_70b_s42.json
```

#### Key metric to verify

Each output file has `"false_forgetting_rate"` per strategy. Expected result:

```
Strategy                             F1      FFR
No-Forgetting                       X.XX    N/A
LRU                                 X.XX    > 0
Importance-Only                     X.XX    > 0
CA-Formula-Only (gate disabled)     X.XX    > 0
Consolidation-Aware (Ours)          X.XX    ≈ 0   ← OURS MUST WIN
```

If CA-Ours FFR is not near 0, something is wrong — check that git pull picked up the latest code.

---

### NB-5: Scaling Benchmark

**File:** `csam_project/nb5_scaling.ipynb`  
**Output dir:** `results/nb5_scaling/`  
**API required:** No (runs with `--no-llm` flag; API optional for richer results)

#### Config cell parameters

```python
MAX_NPCS = 50     # number of NPC agents (50 = default, 100 = paper quality)
MEMORIES  = 100   # memories per NPC
QUERIES   = 10    # queries per NPC
USE_LLM   = False # auto-detected from Step 2 — set GROQ_API_KEY to enable
MODEL     = 'llama-3.1-8b-instant'
```

#### What it runs

1. Single scaling run at configured NPC count (with or without LLM)
2. Sweep: 10, 25, 50, 100 NPCs (always `--no-llm` to avoid API costs)

#### Expected output files

```
results/nb5_scaling/scaling_npcs50_mem100_q10.json
results/nb5_scaling/scaling_sweep_npcs10.json
results/nb5_scaling/scaling_sweep_npcs25.json
results/nb5_scaling/scaling_sweep_npcs50.json
results/nb5_scaling/scaling_sweep_npcs100.json
```

---

### NB-6: Question-Count Scaling (50 / 100 / 200 / 500Q)

**File:** `csam_project/nb6_question_scaling.ipynb`  
**Output dir:** `results/nb6_qscaling/`  
**Datasets:** Both HotPotQA and MuSiQue (already in repo)

#### Config cell parameters

```python
QUESTION_COUNTS = [50, 100, 200, 500]  # modify if you want fewer runs
MODEL = 'llama-3.1-8b-instant'
SEED  = 42
```

#### What it runs

HotPotQA at 50Q, 100Q, 200Q, 500Q then MuSiQue at 50Q, 100Q, 200Q, 500Q.  
Total: 8 runs × ~avg 200Q = ~1,600 API calls on 8B only.  
All within the 14,400 RPD 8B limit — can complete in a single day from one account.

#### Expected output files

```
results/nb6_qscaling/results_hotpotqa_groq_llama-3.1-8b-instant_s42_n50.json
results/nb6_qscaling/results_hotpotqa_groq_llama-3.1-8b-instant_s42_n100.json
results/nb6_qscaling/results_hotpotqa_groq_llama-3.1-8b-instant_s42_n200.json
results/nb6_qscaling/results_hotpotqa_groq_llama-3.1-8b-instant_s42_n500.json
results/nb6_qscaling/results_musique_groq_llama-3.1-8b-instant_s42_n50.json
results/nb6_qscaling/results_musique_groq_llama-3.1-8b-instant_s42_n100.json
results/nb6_qscaling/results_musique_groq_llama-3.1-8b-instant_s42_n200.json
results/nb6_qscaling/results_musique_groq_llama-3.1-8b-instant_s42_n500.json
```

---

## Running on Kaggle / Colab

### Kaggle (recommended — 30h/week GPU, persistent secrets)

1. Create a new notebook at kaggle.com/code
2. Go to **Settings → Add-ons → Secrets** and add:
   - `GROQ_API_KEY` = your primary key
   - `GROQ_API_KEY_2`, `GROQ_API_KEY_3` = additional keys (optional)
3. Upload the `.ipynb` file (or connect to GitHub)
4. Set **Accelerator** to None (CPU is fine — no GPU needed)
5. Set **Internet** to On
6. Run all cells (**Run All**)
7. After completion, download from **Output** tab on the right

### Colab

1. Open colab.research.google.com → Upload notebook
2. Click the **key icon** in the left sidebar → add `GROQ_API_KEY` secret
3. Runtime → Run all
4. Use the download cell at the end of each notebook (Step 7/8/9)

### Important: `git pull` runs automatically

Every notebook's Step 1 does a `git pull origin main` before running anything. This means if code fixes are pushed to GitHub, re-running Step 1 picks them up automatically without needing to re-upload the notebook file.

---

## Sharing Results

### What to collect

Each developer downloads all `.json` files from their output directories:

```
Developer A collects:
  results/nb1_locomo/    → 8 JSON files
  results/nb2_hotpotqa/  → 6 JSON files
  results/nb4_ablation/  → 5 JSON files

Developer B collects:
  results/nb3_musique/   → 7 JSON files
  results/nb5_scaling/   → 5 JSON files
  results/nb6_qscaling/  → 8 JSON files
```

### Sharing method

Share via Google Drive folder or email. Create a folder per notebook:

```
CSAM_Results/
  nb1_locomo/
  nb2_hotpotqa/
  nb3_musique/
  nb4_ablation/
  nb5_scaling/
  nb6_qscaling/
```

Both developers upload their results to the same shared folder. The chart generation and paper writing scripts read all JSON files from these directories.

### Verifying a result file is complete

Open any result file and check it has these top-level keys:

```json
{
  "avg_f1": 0.XXXX,
  "avg_semantic_sim": 0.XXXX,
  "num_questions": 100,
  "provider": "groq",
  "model": "llama-3.1-8b-instant",
  "seed": 42
}
```

For ablation files, check for:

```json
{
  "results": [
    { "strategy": "...", "overall_f1": 0.XXXX, "false_forgetting_rate": 0.XX }
  ]
}
```

---

## Troubleshooting

### 429 Rate Limited / Quota Exceeded

- **RPM hit:** The key rotation handles this automatically. If you have multiple keys, rotation kicks in. Otherwise the script waits and retries.
- **RPD hit (daily quota exhausted):** Stop the notebook. Resume tomorrow, or switch to a key from a different Groq account. Checkpointing means progress is saved — the notebook will skip already-completed questions on the next run.

### "Dataset not found"

The config cell prints `Dataset OK: True/False`. If False:

```
csam_project/benchmarks/data/hotpotqa_dev.json   ← HotPotQA
csam_project/benchmarks/data/musique_dev.jsonl   ← MuSiQue
csam_project/benchmarks/data/locomo10.json       ← LoCoMo
```

These files are in the repo. Check that `git clone` completed successfully in Step 1.

### Results file not created after run

- Check for `[FAIL]` in the cell output — the benchmark script printed an error
- Re-run that single cell; checkpointing picks up from the last completed question
- If repeating failures, check `GROQ_API_KEY` is set correctly (Step 2 output should say "API key loaded")

### Sem= not appearing in progress output (NB-4 ablation)

If progress lines show `F1=X.XX EM=X.XX` but no `Sem=X.XX`, the old code is running. Fix:

```
Step 1 cell: verify git pull says "Already up to date" OR shows new commits
If git pull fails: restart kernel and re-run from Step 1
```

### Scout 17B or 70B hit 1K RPD limit mid-run

The checkpoint system saves after every question. Stop the notebook, wait until midnight UTC (Groq resets daily limits at midnight UTC), then re-run — the notebook will resume from the checkpoint automatically.

---

## Fallback Providers

If Groq limits are a bottleneck, the codebase supports fallback providers. These require separate API keys added to `.env`:

| Provider | Models | Sign-up |
|---|---|---|
| Cerebras | Llama 3.1 8B (high throughput) | cloud.cerebras.ai |
| SambaNova | Llama 3.3 70B | cloud.sambanova.ai |
| Fireworks | All sizes | fireworks.ai |

To use a fallback, change the `--provider` argument in the config cell from `groq` to `cerebras`, `sambanova`, or `fireworks`, and add the corresponding `CEREBRAS_API_KEY` / `SAMBANOVA_API_KEY` / `FIREWORKS_API_KEY` to Secrets.

---

## All Model IDs (Copy-Paste Ready)

```
llama-3.1-8b-instant
meta-llama/llama-4-scout-17b-16e-instruct
llama-3.3-70b-versatile
openai/gpt-oss-120b
```

---

## Checklist Before Starting

- [ ] Groq account created and API key copied
- [ ] Kaggle or Colab account ready
- [ ] API key added to Kaggle/Colab Secrets as `GROQ_API_KEY`
- [ ] Agreed on which 3 notebooks each developer runs
- [ ] Shared folder created for result exchange (Google Drive or similar)
- [ ] Read the "Key metric to verify" section for NB-4 (ablation FFR check)
