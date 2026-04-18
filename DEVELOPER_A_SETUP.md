# Developer A Setup Guide — Kaggle Notebook Execution

**Your Notebooks:** NB-1 (LoCoMo), NB-2 (HotPotQA), NB-4 (Ablation)  
**Total Time:** 8–14 hours across all 3 notebooks  
**API Keys Needed:** 2–3 Groq keys from separate accounts for 70B/Scout (1K RPD each)

---

## Part 1: Kaggle Account Setup (One-Time)

### Step 1.1 — Create/Login to Kaggle

1. Go to **kaggle.com** (create account if needed)
2. Click your **profile icon** → **Settings**
3. Left sidebar: **API**
4. Click **Create New API Token** (downloads `kaggle.json`)
5. Save it — Kaggle CLI uses this for authentication

### Step 1.2 — Add Groq API Keys as Secrets

These are how notebooks access Groq API **without hardcoding keys** in the notebook file.

1. In any Kaggle notebook: **Settings** (gear icon, top right)
2. **Add-ons** → **Secrets**
3. Add secrets (**all required for Developer A**):

```
GROQ_API_KEY        = gsk_...your primary key...
GROQ_API_KEY_2      = gsk_...second key (70B account)...
GROQ_API_KEY_3      = gsk_...third key (Scout account)...
```

**Why 3 keys?**
- 8B: RPD = 14,400 (fits all NB-4 and NB-2 8B runs in 1 day)
- Scout 17B: RPD = 1,000 (needs 2 days OR separate account)
- 70B: RPD = 1,000 (needs 2 days OR separate account)

**Recommended:** Get 3 Groq accounts (3 different emails). Each has 1 API key.

### Step 1.3 — Get Your Groq Keys

Create separate Groq accounts for 8B, Scout, and 70B:

```
Email 1 (8B account):      api-key-1 = GROQ_API_KEY
Email 2 (Scout account):   api-key-2 = GROQ_API_KEY_2  
Email 3 (70B account):     api-key-3 = GROQ_API_KEY_3
```

Go to **https://console.groq.com** for each email:
- Sign in → **API Keys** → **Create API Key** → copy the `gsk_...` key

---

## Part 2: Running Each Notebook

Each notebook follows the same structure:
1. **Step 1** — Install deps & clone repo (auto-pulls latest code)
2. **Step 2** — Load API keys from Kaggle Secrets
3. **Step 3** — Configure parameters (we'll do this below)
4. **Steps 4+** — Run benchmarks automatically
5. **Final Step** — Download results

**Total notebook cells:** 13–17 cells per notebook (just hit "Run All")

---

## NB-1: LoCoMo — CSAM vs Baseline

**File:** `csam_project/nb1_locomo_benchmark.ipynb`  
**Expected Time:** 3–6 hours  
**API Calls:** ~8 CSAM runs + 8 baseline runs (all 4 models)

### Setup & Run

1. Create new Kaggle notebook at **kaggle.com/code**
2. **Settings** → **Secrets** → add all 3 keys (GROQ_API_KEY, GROQ_API_KEY_2, GROQ_API_KEY_3)
3. **Settings** → **Add Custom Code**:
   ```
   # Paste the full nb1_locomo_benchmark.ipynb content here
   # OR upload the .ipynb file directly
   ```
4. Go to **Step 3 — Configure** cell:
   ```python
   MAX_CONVERSATIONS = 5      # DON'T CHANGE (publication quality)
   QUESTIONS_PER_CONV = None  # DON'T CHANGE
   SEED = 42
   ```
5. Click **Run All** (⏯️ button, top)
6. Wait for completion (~4–6 hours)

### What It Does (Auto-Runs)

| Step | What | Time | Model |
|---|---|---|---|
| 4 | CSAM benchmark all 4 models | 2–3h | 8B, Scout, 70B, 120B |
| 5 | Flat-RAG baseline all 4 models | 2–3h | 8B, Scout, 70B, 120B |
| 6 | Results summary table | <1m | N/A |

### Expected Output Files (Download These)

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

**Keys Used:** 
- 8B uses GROQ_API_KEY (primary)
- Scout uses GROQ_API_KEY or GROQ_API_KEY_2 (rotates on 429)
- 70B uses GROQ_API_KEY or GROQ_API_KEY_3 (rotates on 429)
- 120B uses primary or any key

---

## NB-2: HotPotQA — Multi-Model + Variance Seeds

**File:** `csam_project/nb2_hotpotqa.ipynb`  
**Expected Time:** 2–3 hours  
**API Calls:** ~600 on 8B, ~100 each on Scout/70B/120B

### Setup & Run

1. Create new Kaggle notebook
2. **Settings** → **Secrets** → add all 3 keys
3. **Step 3 — Configure**:
   ```python
   N_QUESTIONS = 100   # DON'T CHANGE (publication quality)
   ```
4. **Run All**
5. Wait ~2–3 hours

### What It Does (Auto-Runs)

| Run | Model | Seed | Questions | Time |
|---|---|---|---|---|
| 1 | All 4 models | 42 | 100 | 1–1.5h |
| 2 | 8B only | 123 | 100 | 15–20m |
| 3 | 8B only | 456 | 100 | 15–20m |

**Why 3 runs?** Seeds 123 and 456 let you compute error bars (standard deviation) across different random initializations.

### Expected Output Files

```
results/nb2_hotpotqa/results_hotpotqa_groq_llama-3.1-8b-instant_s42_n100.json
results/nb2_hotpotqa/results_hotpotqa_groq_meta-llama_llama-4-scout-17b-16e-instruct_s42_n100.json
results/nb2_hotpotqa/results_hotpotqa_groq_llama-3.3-70b-versatile_s42_n100.json
results/nb2_hotpotqa/results_hotpotqa_groq_openai_gpt-oss-120b_s42_n100.json
results/nb2_hotpotqa/results_hotpotqa_groq_llama-3.1-8b-instant_s123_n100.json
results/nb2_hotpotqa/results_hotpotqa_groq_llama-3.1-8b-instant_s456_n100.json
```

---

## NB-4: Ablation Study — 5 Forgetting Strategies

**File:** `csam_project/nb4_ablation.ipynb`  
**Expected Time:** 3–5 hours (5 model/seed combos × 5 strategies each)  
**API Calls:** ~1,250 on 8B, ~250 each on Scout/70B  
**Key Metric:** False Forgetting Rate (FFR) — **MUST be ~0 for CA-Ours, >0 for all others**

### Setup & Run

1. Create new Kaggle notebook
2. **Settings** → **Secrets** → add all 3 keys
3. **Step 3 — Configure**:
   ```python
   CONVERSATIONS = 5      # DON'T CHANGE
   INTERACTIONS  = 50     # DON'T CHANGE (50=fast, 100=paper quality)
   THRESHOLD     = 80     # DON'T CHANGE
   ```
4. **Run All**
5. Wait ~4 hours

### What It Does (Auto-Runs)

The notebook runs **5 separate ablation studies**, each testing **all 5 strategies**:

```
Run 1: Llama 3.1 8B,   seed 42  → ablation_8b_s42.json
Run 2: Llama 3.1 8B,   seed 123 → ablation_8b_s123.json
Run 3: Llama 3.1 8B,   seed 456 → ablation_8b_s456.json
Run 4: Llama 4 Scout,  seed 42  → ablation_scout17b_s42.json
Run 5: Llama 3.3 70B,  seed 42  → ablation_70b_s42.json
```

Each run takes ~45 min and tests:
1. **No-Forgetting** (unbounded memory baseline)
2. **LRU** (least-recently-used)
3. **Importance-Only** (forget least important)
4. **CA-Formula-Only** (consolidation formula, gate=θ=0, i.e., disabled)
5. **Consolidation-Aware (Ours)** (full formula + gate=θ=0.3)

### Expected Output Files

```
results/nb4_ablation/ablation_8b_s42.json
results/nb4_ablation/ablation_8b_s123.json
results/nb4_ablation/ablation_8b_s456.json
results/nb4_ablation/ablation_scout17b_s42.json
results/nb4_ablation/ablation_70b_s42.json
```

### Critical Verification Step ⚠️

After NB-4 completes, **open one of the output JSON files** and verify:

```json
{
  "results": [
    {
      "strategy": "Consolidation-Aware (Ours)",
      "overall_f1": 0.XXXX,
      "false_forgetting_rate": 0.0,   ← MUST BE ~0
      "memory_count": XXX,
      "avg_latency_ms": XX.X
    },
    {
      "strategy": "CA-Formula-Only",
      "false_forgetting_rate": > 0.0   ← MUST BE > 0
    },
    {
      "strategy": "LRU",
      "false_forgetting_rate": > 0.0   ← MUST BE > 0
    }
  ]
}
```

**If CA-Ours FFR is NOT ~0:**
- Stop immediately
- Restart Kaggle kernel (forces fresh Python environment)
- Re-run Step 1 (forces git pull)
- If still broken, message the team — code may have reverted

---

## Part 3: Timing & Memory Metrics (What We're Collecting)

### We ARE Already Calculating:

| Metric | Where | What It Measures |
|---|---|---|
| `latency_ms` | NB-1, NB-2, NB-4 | Time per QA pair (embedding + retrieval + LLM) in milliseconds |
| `avg_latency_ms` | All notebooks | Average latency across all questions |
| `estimated_memory_mb` | NB-5 (scaling) | Total L2 HNSW index + metadata in MB |
| `memory_count` | NB-4 (ablation) | Number of memories stored at end of run |
| `memory_bytes` | NB-4 (ablation) | Estimated memory usage of all memories (rough) |

### What Each Output JSON Contains:

```json
{
  "provider": "groq",
  "model": "llama-3.1-8b-instant",
  "avg_f1": 0.5123,              ← Token-level F1 (Accuracy)
  "avg_semantic_sim": 0.6234,    ← Semantic similarity (embedding-based)
  "avg_latency_ms": 312.4,       ← Speed: milliseconds per question
  "num_questions": 100,          ← Test set size
  "seed": 42,                    ← Reproducibility
  "per_question": [
    {
      "latency_ms": 287.5,       ← Individual question time
      "f1": 0.65,
      "semantic_sim": 0.71,
      "exact_match": 1.0
    }
  ]
}
```

### Key Speed Metrics Summary:

Your results will have this structure across all 3 notebooks:

**NB-1 (LoCoMo):**
- **Speed:** CSAM vs Baseline latency per conversation
- **Memory:** Not explicitly logged (but L2 HNSW index is capped at 200 memories)

**NB-2 (HotPotQA):**
- **Speed:** ~300–400ms per question (8B), ~150–200ms (70B) — 2-hop retrieval
- **Memory:** Implicit (200-memory cap means fixed memory usage)

**NB-4 (Ablation):**
- **Speed:** `avg_latency_ms` per strategy
- **Memory:** `memory_count` + `memory_bytes_mb` per strategy
- **Efficiency:** Which strategy achieves best F1 with LEAST memory usage?

### How to Interpret the Metrics:

```
FAST + ACCURATE = Good
  Example: 70B model at 200ms with F1=0.75

SLOW + ACCURATE = Acceptable (bigger model, more power)
  Example: 120B model at 400ms with F1=0.80

FAST + INACCURATE = Bad (might be worthless)
  Example: 8B model at 100ms with F1=0.30

CA-Ours should have:
  - Lower FFR than all baselines (memory efficiency)
  - Similar or better F1 than baselines (accuracy)
  - Similar latency to baselines (no speed penalty)
```

---

## Part 4: Downloading Results from Kaggle

After each notebook completes:

### Option A: Kaggle Download Button (Easiest)

1. In the completed notebook, go to **Output** tab (right side)
2. You'll see folder: `results/nb1_locomo/` (or nb2, nb4)
3. Click the folder → **Download** (blue button)
4. All JSON files auto-zip and download

### Option B: Step 7 — Download Cell (Built-In)

Every notebook's last cell auto-downloads files to your computer:

```python
# This cell is in every notebook (Step 7)
if os.path.exists('/kaggle'):
    # If on Kaggle: copy to /kaggle/working (auto-saved as output)
    for fp in all_files:
        shutil.copy(fp, '/kaggle/working/')
else:
    # If local: trigger browser download
    from google.colab import files
    for fp in all_files:
        files.download(fp)
```

Just run the final cell and files download automatically.

### Where Files Go

```
Your Downloads/
  nb1_locomo_benchmark.ipynb           ← Step 3 setup
  nb2_hotpotqa.ipynb                   ← Step 3 setup
  nb4_ablation.ipynb                   ← Step 3 setup
  
  results_locomo_csam_groq_llama-3.1-8b-instant_s42.json
  results_locomo_baseline_groq_llama-3.1-8b-instant_s42.json
  results_hotpotqa_groq_llama-3.1-8b-instant_s42_n100.json
  results_hotpotqa_groq_llama-3.1-8b-instant_s123_n100.json
  results_hotpotqa_groq_llama-3.1-8b-instant_s456_n100.json
  ablation_8b_s42.json
  ablation_8b_s123.json
  ablation_8b_s456.json
  ablation_scout17b_s42.json
  ablation_70b_s42.json
```

---

## Part 5: Sharing Results with Developer B

Create a shared folder (**Google Drive or GitHub**) with this structure:

```
CSAM_Results_Apr2026/
  Developer_A/
    nb1_locomo/
      results_locomo_csam_groq_*.json
      results_locomo_baseline_groq_*.json
    nb2_hotpotqa/
      results_hotpotqa_groq_*.json
    nb4_ablation/
      ablation_*.json
      
  Developer_B/
    (will upload their results here)
    nb3_musique/
    nb5_scaling/
    nb6_qscaling/
```

Both developers upload to the same folder so chart generation scripts can find all results in one place.

---

## Troubleshooting

### Q: Notebook starts but then says "GROQ_API_KEY not found"
**A:** You added the secret AFTER creating the notebook. **Restart the kernel** (top-right, "Run All" button → stop → restart).

### Q: 429 Rate Limited error mid-run
**A:** Notebook auto-rotates to the next key. But if you only have 1 key, you hit the daily limit. Wait until tomorrow (Groq resets at midnight UTC) or add another key.

### Q: One model takes way longer than expected
**A:** It might have hit RPM (requests per minute) limit. Groq throttles to 30 RPM. Notebook waits and retries automatically. Just let it run.

### Q: Results file is empty or incomplete
**A:** Stop the notebook and check the cell output for `[FAIL]`. If a question fails:
1. Restart kernel
2. Re-run Step 1 (git pull)
3. Re-run just that notebook from the top

Checkpointing saves progress, so it will skip already-completed questions.

### Q: Ablation FFR is not 0 for CA-Ours
**CRITICAL:** This is a code issue.
1. Restart kernel
2. Force git pull: **Stop** → **Run Step 1** only → check "Commit: ..." output
3. If commit is old: `git pull failed`. Try again from Step 1.
4. If commit is new: Contact team immediately.

---

## Timeline Estimate

| Notebook | Start | Finish | Notes |
|---|---|---|---|
| NB-1 | Day 1, 9am | Day 1, 3pm | 6 hours (8B + Scout fast, 70B/120B slower) |
| NB-2 | Day 1, 3pm OR Day 2, 9am | Day 1, 6pm OR Day 2, 12pm | 2–3 hours (mostly 8B) |
| NB-4 | Day 2, 1pm OR Day 3, 9am | Day 2, 5pm OR Day 3, 1pm | 4–5 hours (5 runs sequential) |

**Parallel Option:** Start NB-1 and NB-2 on separate Kaggle notebooks simultaneously (they share 8B key, but checkpointing handles that gracefully).

---

## Files You Need

Make sure you have **either**:
- **GitHub** with latest code (notebooks do `git pull` automatically), OR
- Downloaded `.ipynb` files from repo directly

Both work. GitHub is safer (always up-to-date).

```bash
# To download all 3 notebooks from GitHub:
git clone https://github.com/Lamaq-Mujpurwala/CSAM-IPD-HALH.git
# Then upload to Kaggle:
#   csam_project/nb1_locomo_benchmark.ipynb
#   csam_project/nb2_hotpotqa.ipynb
#   csam_project/nb4_ablation.ipynb
```

---

**You're all set!** Start with NB-1 and work through in order. Reach out if any step fails or metrics look wrong.
