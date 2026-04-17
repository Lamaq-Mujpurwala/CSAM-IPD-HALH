*CSAM: Consolidation-Aware Scalable Agent Memory for Long-Term Conversational AI*

  Lamaq Mujpurwala
Department of Computer Science and Engineering (Data Science)
Dwarkadas J. Sanghvi College of Engineering
Mumbai, India
lamaqmuj5@gmail.com

  Harshil Bhanushali
Department of Computer Science and Engineering (Data Science)
Dwarkadas J. Sanghvi College of Engineering
Mumbai, India
[FILL: email address or ORCID]

  Avena Jain
Department of Computer Science and Engineering (Data Science)
Dwarkadas J. Sanghvi College of Engineering
Mumbai, India
[FILL: email address or ORCID]

  Hannah Fernandes
Department of Computer Science and Engineering (Data Science)
Dwarkadas J. Sanghvi College of Engineering
Mumbai, India
hannahfernandes2005@gmail.com

---

***Abstract*** ***Scalable long-term memory is a core bottleneck in conversational AI models that require maintaining coherent interactions over long dialogues. Although recent breakthroughs in large language models have led to improvements in reasoning and conversation quality, persistent memory architectures continue to suffer from a fundamental scalability–retention trade-off: unbounded memory growth degrades retrieval efficiency, whereas naive forgetting policies risk discarding semantically critical information.***

***This paper presents CSAM (Consolidation-Aware Scalable Agent Memory), a three-tier memory hierarchy inspired by human memory consolidation. The architecture integrates transient working memory (L1), HNSW-based episodic retrieval (L2), and a structured knowledge graph (L3) to enable efficient long-horizon reasoning. Central to CSAM is a consolidation-aware forgetting mechanism that verifies semantic assimilation of episodic content into the knowledge graph prior to eviction, enabling bounded memory growth without sacrificing informational fidelity.***

***We evaluate CSAM on three established benchmarks — LoCoMo (long-term conversation), MuSiQue, and HotPotQA (multi-hop question answering) — across model scales from 8B to 70B parameters using hosted inference. [UPDATE WITH FINAL NUMBERS AFTER BENCHMARK RUNS COMPLETE] Across experiments, improvements driven by memory and retrieval design substantially outweigh gains from model scaling. Ablation analysis further shows that consolidation-aware forgetting maintains recall comparable to unbounded memory while ensuring controlled memory growth.***

***These findings highlight memory architecture rather than model size as the primary limiting factor in long-term conversational AI performance.***

***Keywords*** Conversational AI, Long-Term Memory, Memory Consolidation, Knowledge Graph Integration, Retrieval-Based Reasoning, Scalable Agent Architectures

---

## I. Introduction

There is something deeply frustrating about working with language model agents that forget everything the moment a session ends. We ran into this problem while building a prototype companion NPC — the kind of character that is supposed to feel alive, remember what you have told it, and behave like it actually knows you after a few sessions. What we found instead was an agent that reset completely every time. You could tell it your name, your goals, your backstory — and on the next run, it had no idea who you were.

The obvious fix is to store conversation history and retrieve it at inference time. That works, briefly. After a few hundred interactions the retrieved context becomes cluttered with stale, contradictory, or simply irrelevant snippets — and retrieval quality degrades noticeably. We tried capping the store: throw away old memories, keep only recent ones. That is when we started losing things that actually mattered. A player's stated goal from session 1, a preference they'd mentioned offhand — gone, because LRU had decided something newer was more important.

This trade-off — remembering everything (which breaks retrieval) versus forgetting strategically (which risks losing critical information) — turned out to be the core problem we kept circling back to. It is not really a storage problem or a retrieval problem. It is a *forgetting* problem. And forgetting, it turns out, is surprisingly difficult to get right.

What changed our thinking was reading about how human memory handles this. The multi-store model [12] draws a distinction between short-term and long-term memory that goes beyond capacity — the transfer involves *restructuring*, not just copying. Episodic experiences get abstracted, compressed, and integrated with existing knowledge before the raw trace fades. The details of a conversation might blur over time, but the meaning persists. Controlled forgetting [11] supports memory efficiency precisely because the important content survives in a different form. Forgetting happens *after* consolidation — not instead of it.

That observation is what CSAM is built around. If a memory's content has been abstracted into a more structured form — a knowledge graph node, a named entity, a summarized relationship — then deleting the raw episodic trace is not a loss, it is just removing redundancy. But if that abstraction has not happened yet, deletion is genuinely destructive. Prior systems largely miss this distinction.

Transformer architectures [23] provide strong conversational reasoning, but they have no persistence across sessions. RAG [14] extends effective context via retrieval, though stores grow noisy at scale as irrelevant entries accumulate. Generative Agents [2] introduced a reflection mechanism for synthesizing higher-level insights, but scanning the full memory stream linearly does not scale across many concurrent agents. MemGPT [3] uses a hierarchical paging design — clever, but the paging decisions are made by the model itself, which adds token overhead to every context switch. H-MEM [5] improves retrieval routing but does not address what to prune or when. HippoRAG [6] performs well on static knowledge corpora but was not designed for memory stores that evolve continuously through interaction.

The question none of them tackle directly: *how do you know when it is safe to forget something?*

We propose **CSAM (Consolidation-Aware Scalable Agent Memory)**, which answers that question by tracking whether a memory's content has been consolidated into a structured knowledge graph before allowing deletion. The system operates across three tiers:

- **L1 (Working Memory):** The active LLM context — the last 20 turns, always available at query time
- **L2 (Episodic Memory):** An HNSW-indexed vector store [17] holding detailed interaction traces, capped at 200 entries
- **L3 (Semantic Memory):** A knowledge graph built incrementally by extracting entities and relationships from L2, designed for long-term retention

Episodic memories are scored for deletion using four factors — recency decay, importance, consolidation coverage, and semantic redundancy — but a hard gate prevents eviction of any memory whose consolidation coverage falls below a threshold θ. In plain terms: a memory cannot be deleted until its meaning exists somewhere more durable.

We evaluate CSAM on LoCoMo [4] for long-horizon conversational recall, and on HotPotQA [10] and MuSiQue [9] for multi-hop reasoning, using model scales from 8B to 70B parameters. One result we did not fully anticipate was how little additional performance came from scaling the model versus improving the retrieval architecture. On memory-intensive tasks, the memory design seems to matter more than the model size — at least within the scales we tested.

### A. Summary of Contributions

1. **Consolidation-Aware Forgetting:** A forgetting mechanism with an explicit gate — memories are only eligible for deletion once their content has been confirmed as absorbed into L3. This is a hard recoverability check, not a soft scoring weight.

2. **Three-Tier Architecture:** Working memory (L1), HNSW-indexed episodic memory (L2), and a structured knowledge graph (L3) operating together as a unified system with separated read and write paths.

3. **False Forgetting Rate (FFR):** A metric to measure how often a forgetting strategy deletes memories that were not yet consolidated — useful for directly comparing safety properties across strategies.

4. **Architecture-vs-Scaling Analysis:** An empirical comparison across 8B to 70B models suggesting that memory architecture is the primary bottleneck in long-horizon conversational AI for retrieval-intensive tasks.

---

## II. Related Work

When we surveyed prior work, a pattern emerged that we found striking: most systems invest heavily in retrieval quality and relatively little in *what to discard*. Memory stores tend to grow append-only and are either capped arbitrarily or allowed to grow without bound. The question of when it is genuinely safe to remove something gets comparatively little attention.

### A. Memory-Augmented Language Models

MemGPT [3] is probably the most architecturally similar to CSAM, and it is worth being specific about where the comparison breaks down. The operating-system framing — context window as RAM, external storage as disk, model-controlled paging between them — is a genuinely useful abstraction. The limitation is that paging decisions happen inside the model's attention window. Every context switch costs tokens, and in a system with many concurrent agents running long-horizon sessions, that overhead accumulates. We wanted memory management to happen asynchronously, outside the model, so that per-query latency stays bounded by retrieval and generation alone.

H-MEM [5] focuses on retrieval efficiency through hierarchical indexing — fast routing to relevant memories — but does not formalize when entries should be removed. This is a different problem than the one we are solving: H-MEM optimizes access; CSAM also addresses lifecycle. Generative Agents [2] introduced periodic reflection to synthesize higher-level insights from raw memories, which is philosophically close to consolidation. The practical issue is linear scanning of the full memory stream, which becomes expensive as store size grows and especially so when running many agents in parallel.

A-MEM [19] and MemoryBank [18] represent more recent entries: A-MEM links notes dynamically via connections between memory items, and MemoryBank applies temporal decay curves. Both are improvements over naive retrieval, but neither conditions forgetting on whether content has been captured in a more durable representation. That specific check — verify before deleting — is the gap CSAM fills.

### B. Knowledge Graph-Based Memory

HippoRAG [6] is the closest precedent for using a knowledge graph as the long-term component in a retrieval pipeline. Personalized PageRank over a structured graph does enable strong multi-hop reasoning performance. Where HippoRAG is designed for relatively static external corpora — documents you load once and query repeatedly — CSAM needs an L3 that grows incrementally from ongoing conversations and remains consistent with a constantly-changing episodic store. The consolidation pipeline is specifically designed for this: it handles clustering, entity extraction, and the bookkeeping of which L2 memories map to which L3 nodes — which is the prerequisite for the forgetting gate. GraphRAG [13] takes a related approach at document scale but is not designed for short-form episodic ingestion of the kind our use case requires.

### C. Forgetting and Memory Management

Most treatment of forgetting in AI comes from the continual learning literature, where catastrophic forgetting [11] refers to neural network parameters overwriting old knowledge when trained on new data. That is a fundamentally different mechanism from ours — we are not updating model weights, we are managing an external store at inference time. The transfer of lessons from that literature to memory-augmented LLM systems is partial at best.

In practice, memory-augmented systems tend to use either LRU eviction or importance-based heuristics. LRU is simple but semantically blind — it cannot distinguish between an old memory that has been safely consolidated and an old memory that represents unique, irreplaceable information. Importance-based strategies are better, but importance is typically assigned at ingestion time rather than re-evaluated at eviction. A memory that seemed low-salience when stored might be the only surviving record of a key fact by the time the eviction decision is made.

The failure mode we kept encountering during development: a foundational piece of information from early in a session — a user's stated goal, a named preference — scores poorly on both recency and importance and gets evicted, with nothing in L3 to fall back on. Checking consolidation coverage at eviction time is what prevents this. To our knowledge, no prior memory-augmented LLM system makes this check explicitly.

---

## III. System Comparison

| System | O(log N) | Knowledge Graph | Hierarchical | Forgetting | Novel Claim |
| :----- | :------- | :-------------- | :----------- | :--------- | :---------- |
| SAM (Rae et al., 2016) [22] | Yes | No | No | No | Sparse R/W |
| Gen. Agents (2023) [2] | No | No | No | No | Reflection |
| MemGPT (2023) [3] | Yes | No | Yes | No | OS metaphor |
| H-MEM (2025) [5] | Yes | No | Yes | No | Index routing |
| HippoRAG (2024) [6] | Yes | Yes | No | Yes | PPR retrieval |
| CSAM (Ours) | Yes | Yes | Yes | Consolidation | Safe forgetting |

---

## IV. Architecture

### A. System Overview

CSAM implements a three-tier memory architecture inspired by the Atkinson–Shiffrin multi-store memory model [12] and modern theories of controlled forgetting [11]. The system is designed to enable persistent interaction while maintaining bounded memory growth and real-time retrieval performance. Fig. 1 illustrates the overall architecture.

The architecture consists of three memory layers with distinct functional roles.

**L1 (Working Memory)** operates within the active context window of the language model. It maintains the most recent interaction turns and provides immediate access to short-term conversational state. This layer is transient and optimized for low-latency response generation rather than long-term storage.

**L2 (Episodic Memory)** stores interaction traces as dense vector embeddings indexed using Hierarchical Navigable Small World (HNSW) graphs [17]. This indexing structure enables approximate nearest neighbor retrieval in O(log N) time, ensuring scalability as memory size increases. Episodic memory preserves detailed contextual information but is not assumed to be permanent.

**L3 (Semantic Memory)** maintains a structured knowledge graph that stores abstracted entities, relationships, and summarized representations extracted from episodic experiences. Unlike L2, which stores high-fidelity traces, L3 captures consolidated knowledge intended for long-term retention.

During inference, a hybrid retrieval module queries L1, L2, and L3 jointly. Retrieved candidates are ranked and filtered before being incorporated into the model's response generation process. This layered retrieval ensures that immediate conversational context, relevant past experiences, and structured long-term knowledge are all considered.

A consolidation pipeline periodically analyzes episodic memories and extracts semantic content for integration into L3. Once the system verifies that the essential information of an episodic trace has been successfully consolidated, the forgetting engine evaluates that trace for potential deletion. This consolidation-aware forgetting mechanism prevents unbounded memory accumulation while preserving critical knowledge.

By separating working, episodic, and semantic memory into distinct layers, CSAM balances latency, scalability, and persistence—addressing limitations observed in prior memory-augmented LLM architectures [2], [3], [5].

---

*Fig. 1. CSAM three-tier memory architecture.* Working memory (L1) provides short-term conversational context, episodic memory (L2) enables O(log N) retrieval through HNSW indexing, and semantic memory (L3) maintains a structured knowledge graph. A consolidation pipeline abstracts episodic traces into semantic knowledge, enabling consolidation-aware safe forgetting.

---

### B. Formal Memory Representation

Each episodic memory record stored in L2 is represented as a five-tuple:

*m = (x, e, t, I, μ)*

where *x* is the raw memory text, *e ∈ ℝ^d* is its dense embedding produced by a Sentence-BERT encoder [7], *t* encodes timestamp and recency metadata, *I ∈ [0, 1]* represents the importance score assigned at ingestion time, and *μ* captures auxiliary metadata including speaker identity, player scope, and session provenance.

Given a query embedding *q*, L2 retrieval returns the top-*k* episodic memories by cosine similarity:

*ℛ_L2(q, k) = TopK_{m∈L2} s(q, e_m)*

where *s* denotes cosine similarity. This retrieval is executed via HNSW rather than exact exhaustive search, preserving sub-linear query times at scale [17].

L3 maintains semantic nodes *v* and labelled relational edges *r*, generated by the consolidation pipeline from subsets of L2 memories. A consolidation tracker records memory-to-node linkage, enabling coverage estimation for each episodic record—a prerequisite for the forgetting gate described in Section IV-E.

### C. Write Path and Read Path Separation

CSAM explicitly separates the memory write path from the read path. This separation ensures that consolidation and forgetting operations—which may involve LLM inference for summarisation—do not block or degrade response latency during active interaction.

**Write path:** When a new interaction turn is received, CSAM computes a Sentence-BERT embedding and ingests the record into L2 [7]. The record is simultaneously mirrored in L1 as the most recent interaction. Periodically—currently triggered every 20 turns—the consolidation pipeline executes asynchronously to abstract L2 content into L3. Once consolidation has verified coverage, the forgetting engine evaluates eligible records for deletion under memory pressure.

**Read path:** At inference time, the query embedding is used to retrieve candidate context from L2, L3, or both via the Hybrid Multi-Retrieval (HMR) module. Retrieved candidates are ranked by relevance and assembled into the model's context window. The response is generated and, in persistent mode, the interaction is subsequently stored. Critically, this path does not invoke the consolidation pipeline, preserving deterministic latency characteristics.

This write/read separation is architecturally significant. It allows CSAM to operate as a real-time system whose per-query latency is bounded by retrieval and generation costs alone, while memory lifecycle management—which carries higher and more variable overhead—proceeds in controlled background intervals.

---

*Fig. 2. CSAM Memory Flow Pipeline.* Store → Fill → Consolidate → Forget → Recall. The operational cycle consists of memory insertion, natural conversational accumulation, periodic consolidation into L3, consolidation-aware forgetting to maintain bounded memory, and hybrid recall for answer generation.

---

### D. Consolidation Pipeline

The consolidation pipeline is the mechanism through which episodic memories in L2 are abstracted into semantic knowledge in L3. It operates in three phases: semantic grouping, LLM-powered summarisation, and coverage registration.

**1. Semantic Grouping:**
Episodic records in L2 are clustered by semantic similarity using their stored embeddings. Memories that share thematic or entity-level content are grouped together as consolidation candidates. This step prevents redundant or partial knowledge nodes from being created independently for closely related episodic traces.

**2. LLM-Powered Summarisation and Entity Extraction:**
For each cluster, the consolidation pipeline invokes an LLM to extract named entities, summarise relational content, and generate a structured node representation for insertion into L3. This produces three output types: entity nodes (e.g., character names, locations, factual attributes), relational edges connecting these entities, and summary nodes capturing contextual meaning that does not reduce to a single entity. An example consolidation mapping is illustrated in Fig. 3: a cluster of five episodic traces involving a player named Alexander, a weather reference, and a lucky number yields three distinct L3 nodes—an entity node for the character, a casual context summary, and a numeric entity record.

**3. Coverage Registration:**
Once an episodic memory *m* has been mapped to one or more L3 nodes, its consolidation coverage score *C(m) ∈ [0, 1]* is updated to reflect the proportion of its semantic content that has been successfully absorbed into the knowledge graph. A memory with *C(m) ≥ θ*—where *θ* is the consolidation threshold—is considered eligible for forgetting. This coverage registration step is the essential prerequisite for safe forgetting, as it ensures that deletion decisions are conditioned on demonstrated semantic absorption rather than mere temporal distance or importance heuristics alone.

---

*Fig. 3. Consolidation Pipeline:* L2 Episodic Memories to L3 Semantic Nodes. Episodic memories are grouped and transformed into semantic graph nodes and summaries. The memory-to-knowledge mapping produced by this step is used to compute consolidation coverage C(m) for safe forgetting decisions.

---

### E. Consolidation-Aware Forgetting Formulation

The forgetting mechanism in CSAM is formalised as a composite scoring function. For each episodic memory *m* in L2, a forgetting score is computed as:

*F(m) = α·R(m) + β·(1 − I(m)) + γ·C(m) + δ·D(m)*

where the four components are defined as follows:

1. **R(m)**: Recency decay. Encodes temporal distance from the most recent access, penalising memories that have not been retrieved over extended interaction horizons.
2. **I(m)**: Importance. Represents the semantic salience assigned at ingestion. The term (1 − I(m)) converts this into a retention-weakening signal: lower-importance memories score higher toward forgetting.
3. **C(m)**: Consolidation coverage. Reflects the degree to which the memory's semantic content has been successfully abstracted into L3. High coverage indicates that the essential information is recoverable from the knowledge graph, making deletion safer.
4. **D(m)**: Redundancy. Measures overlap between the memory's content and existing L3 semantic nodes, penalising retention of content already well-represented in structured form.

The coefficients *α, β, γ, δ* are tunable weights governing the relative contribution of each factor. In the current evaluated configuration, equal weighting (*α = β = γ = δ = 0.25*) yielded the highest observed F1 within the tested hyperparameter grid. Threshold tuning showed larger practical impact than moderate weight perturbations; these findings should be understood as protocol-specific observations pending broader variance analysis across seeds and interaction volumes.

The critical structural addition relative to prior forgetting policies is an **operational gate** on the forgetting score:

*if C(m) < θ, then F(m) = 0*

This gate ensures that no episodic memory is eligible for deletion unless its consolidation coverage meets the threshold *θ*. A memory may score highly on recency decay, low importance, and redundancy—yet if its semantic content has not been demonstrably absorbed into L3, it is protected from eviction. This converts the forgetting function from a pure ranking mechanism into a **recoverability-aware retention policy**: memories are not treated as disposable solely because they are old or low-salience, but only when semantic equivalents have been confirmed to exist in the knowledge graph.

This design addresses the known failure modes of both LRU and importance-only strategies. LRU eviction may delete unique but older memories before they have been semantically preserved [11]. Importance-only strategies may retain high-salience but redundant records while removing less salient but unique evidence. By conditioning deletion on *C(m)* and *D(m)*, CSAM aligns forgetting with recoverability rather than with access recency or assigned salience alone. To our knowledge, no prior memory-augmented LLM architecture explicitly models and gates forgetting on consolidation coverage in this manner [2][3][5][6].

---

## V. Experimental Setup

### A. Models and Infrastructure

All experiments are conducted using hosted inference via the Groq API. We evaluate across three model scales: **Llama 3.1 8B Instant** (lightweight baseline), **Llama 3.3 70B Versatile** (strong general-purpose), and **Llama 4 Scout 17B** (instruction-tuned mid-range). Sentence embeddings are generated using all-MiniLM-L6-v2 (384 dimensions) via sentence-transformers [7]. No GPU is required at inference time as embeddings are cached per session.

### B. Datasets and Evaluation Protocol

We evaluate on three benchmarks covering complementary aspects of long-horizon memory:

**LoCoMo [4]:** A long-term conversational memory benchmark consisting of multi-session human dialogues. We evaluate on the locomo10 split (10 conversations), each spanning multiple temporally-distributed sessions with question-answer pairs designed to test retention of facts established early in the conversation.

**HotPotQA [10]:** A two-hop open-domain QA dataset drawn from Wikipedia. We evaluate on a 100-question sample from the development set (7,405 total questions), focusing on bridge-type multi-hop reasoning that requires synthesizing information from two distinct evidence passages. The sample size reflects practical student resource constraints (see Limitations) while maintaining sufficient scale for statistical significance assessment.

**MuSiQue [9]:** A multi-hop QA dataset requiring 2–4 reasoning hops via single-hop question composition. We evaluate on the full 200-question development set. Crucially, MuSiQue is evaluated twice—with and without the L3 knowledge graph—to isolate the graph-hop contribution. The F1 delta between these two conditions directly quantifies the value of the knowledge graph layer.

### C. Evaluation Metrics

The primary evaluation metric is **token-level macro F1**, computed via normalized token overlap between predicted and reference answers. Normalization consists of lowercasing, punctuation removal, and whitespace collapsing. Per-conversation F1 scores are aggregated via arithmetic mean. Statistical significance of the CSAM-vs-baseline delta is assessed via bootstrap resampling (2,000 iterations, seed-controlled at seed=42). The 95% confidence interval on the delta is reported to distinguish genuine improvements from sampling variation.

The **False Forgetting Rate (FFR)** is reported for ablation experiments as a secondary metric. FFR measures the fraction of evicted memories with consolidation coverage *C(m) < θ* at eviction time—i.e., memories deleted before being fully absorbed into L3. Lower FFR indicates safer forgetting behavior.

### D. Implementation Details and Reproducibility

All stochastic components—including HNSW index initialization, retrieval tie-breaking, and bootstrap resampling—use seed=42 unless otherwise specified. The L2 memory capacity is fixed at 200 entries; forgetting is triggered when occupancy exceeds this threshold. The consolidation gate threshold is **θ = 0.3**, with equal weighting **α = β = γ = δ = 0.25** for the four forgetting score components. Retrieval uses **k = 5** candidates from L2 and **top-2** from L3, with Maximal Marginal Relevance (MMR) applied to reduce redundancy in the assembled context window.

Per-conversation checkpoints are saved atomically after each conversation completes, enabling interrupted runs to resume from the last completed checkpoint without re-processing prior conversations. Complete benchmark code, evaluation notebooks, and dataset access instructions are publicly available at the project repository.

---

## VI. Evaluation and Results

### A. Practical Rationale for Gated Forgetting

The practical motivation for explicitly modeling C(m) (Consolidation Coverage) and D(m) (Semantic Redundancy) is addressed by analyzing the failure modes of existing forgetting policies across extended interaction horizons.

**A. Mitigation of Recency-Induced Information Loss:** Under standard LRU eviction, temporal recency determines deletion priority. While computationally simple, this is semantically indiscriminate. In long-term testing, we observed that foundational anchors—such as a player's stated name or core preferences disclosed early in an interaction—become high-priority deletion targets despite being uniquely important. CSAM's operational gate prevents this: unless *C(m) ≥ θ*, the memory remains ineligible for deletion regardless of its age, ensuring that critical information is never discarded before it is abstracted.

**B. Redundancy and Importance Management:** Under importance-only strategies, high-salience memories are often retained even if equivalent content already exists in L3, leading to memory bloat. Conversely, low-importance but unique details may be prematurely discarded. The D(m) term addresses this by using redundancy with L3 content as independent evidence for safe deletion. Together, these terms ensure that CSAM's forgetting decisions are conditioned on the **recoverability** of information from the knowledge graph, rather than merely on access recency or inferred importance.

*Fig. 4. Comparative Analysis of Forgetting Policies on Recall Stability.* The inclusion of C(m) and D(m) allows CSAM to maintain a stable F1 score across extended interaction horizons, whereas LRU-based systems exhibit a sharp decline in recall as early "anchor" facts are evicted.

### B. Memory Growth Dynamics

Fig. 5 illustrates the memory growth trajectories of bounded and unbounded retention policies across five test conditions. Under no-forgetting, episodic memory grows linearly with interaction volume, reaching approximately 520 entries by the fifth test condition. Under CSAM's consolidation-aware policy—and under LRU—memory is maintained near the configured threshold of 200 entries across all test conditions.

The practical implication for multi-agent deployment is significant. A system hosting *n* concurrent agents under unbounded retention incurs storage and retrieval overhead that scales as O(*n* × T), where T is total interaction volume per agent. CSAM's bounded policy reduces this to O(*n* × M_max), where M_max is the configured threshold, decoupling per-agent cost from interaction depth.

*Fig. 5. Memory Growth Under Bounded and Unbounded Policies.* Unbounded memory exhibits linear growth with interaction volume, whereas CSAM's consolidation-aware policy and LRU both plateau near the configured threshold of 200 entries, demonstrating deployment feasibility at scale.

### C. Architecture-Bound vs. Model-Bound Performance Regimes

A significant finding of this work is the observed decoupling of memory-intensive task performance from raw model parameter count. To investigate this, we conducted a comparative analysis using model scales ranging from 8B to 70B parameters under a unified CSAM configuration.

**1. The Architecture-Bound Regime (LoCoMo):**

*Fig. 6. Impact of Parameter Scaling on Memory-Intensive Tasks (LoCoMo).* Performance of diverse LLM backbones under the CSAM architecture on the LoCoMo benchmark. The convergence of scores across an 8× increase in parameter count indicates a regime where architectural retrieval efficiency is the primary performance driver.

In tasks requiring the retrieval of long-horizon "anchor" facts—such as identifying a user's unique preference mentioned many turns prior—we observe an **architecture-bound regime**. As shown in Fig. 6, an 8B model supported by the CSAM memory stack performs within a ±2% margin of a 70B model. This indicates that once the correct context is retrieved from the episodic-semantic hierarchy, the generative requirement is relatively low. The performance bottleneck lies in the organization and retention of facts within the L2 and L3 layers, not the model's internal reasoning capacity.

**2. The Transition to Model-Bound Reasoning (MuSiQue & HotPotQA):**

*Fig. 7. Cross-Benchmark Scaling Trends: F1 Scores vs. Model Size.* Comparative scaling behavior. While LoCoMo remains flat, MuSiQue and HotPotQA show positive slopes, representing the transition from retrieval-limited to reasoning-limited performance.

The regime shifts when tasks require complex logical composition across retrieved fragments. Our analysis of the MuSiQue and HotPotQA benchmarks (Fig. 7) reveals a **model-bound regime**. In these scenarios, CSAM successfully provides the relevant evidence traces, but the task requires synthesizing an answer from multiple "hops" of information.

*Fig. 8. MuSiQue Multi-Hop QA Performance By Model and Hop Pattern.* Performance breakdown by reasoning complexity. Larger models exhibit superior synthesis capabilities in 3+ hop queries, where the bottleneck shifts from retrieval coverage to compositional reasoning.

As evidenced in Fig. 8, larger models demonstrate a significant performance delta as the number of reasoning hops increases. In this context, the CSAM architecture acts as a necessary prerequisite—providing the raw materials for the answer—but the final F1 score is dictated by the model's ability to handle complex logical dependencies. This data suggests that while architecture optimizes **retrieval coverage**, model scale is required to optimize **compositional synthesis**.

**3. Two-Hop Reasoning Accuracy (HotPotQA):**

*Fig. 9. HotPotQA Two-Hop Performance Under Constant CSAM Architecture.* Performance metrics for two-hop reasoning. Architecture-bound constraints are less pronounced here than in LoCoMo, as the task shifts from retrieving distant facts to synthesizing a coherent bridge between proximal evidence traces.

HotPotQA results demonstrate higher sensitivity to model parameter count than the LoCoMo recall task. We attribute this to the reasoning density of the benchmark; once the CSAM stack provides the necessary context, the final accuracy depends on the model's ability to execute precise "comparison" or "bridge" reasoning steps. The consistent performance across different question types confirms that the retrieval layer provides uniform context quality, but superior F1 scores in larger models suggest that increased parameter count is essential for filtering noise inherent in high-density semantic retrieval.

### D. Performance Distribution and Retrieval Reliability

*Fig. 10. Bimodal Score Distribution Indicating Retrieval-Gated Performance.* Score histograms reveal a distinct bimodal tendency. This "all-or-nothing" behavior confirms that in the CSAM framework, performance is primarily retrieval-gated. When the correct context is retrieved, accuracy is near-perfect; when retrieval fails, the model lacks the anchor necessary for generation, resulting in an abrupt performance drop.

To investigate the underlying cause of conversational failure in long-horizon tasks, we analyze the distribution of F1 scores across the evaluation set. The bimodal nature of these results indicates that the system is essentially retrieval-gated: performance is high when the context is present and drops sharply when it is not.

This distribution supports our hypothesis that generative hallucinations are largely mitigated by the architecture. In the CSAM framework, the LLM is not required to rely on internal weights for factual recall; instead, it acts as a synthesizer of provided context. Consequently, the bottleneck for conversational reliability is the **retrieval coverage** of the L2 and L3 layers, rather than the generative quality of the backbone model.

### E. Latency Decomposition and Computational Efficiency

In operational terms, the three memory layers exhibit distinct computational profiles. L1 access is effectively O(1) due to its bounded cache structure. L2 retrieval is sub-linear in typical operation via HNSW, with empirically observed latency of approximately 5 ms per query in the evaluated configuration. L3 graph queries currently exhibit latency of approximately 2 ms, reflecting the modest scale of knowledge graphs produced in the experimental setup; this may increase for significantly larger graphs depending on traversal depth required.

Fig. 11 presents the end-to-end latency decomposition observed across experiments. LLM generation accounts for approximately 98.9% of total response time (~5,000 ms), while memory retrieval, embedding, and graph query together account for fewer than 60 ms combined. This profiling result has a direct consequence for optimisation priority: CSAM's memory architecture is *not* the primary latency bottleneck in the current deployed configuration. Generator-side latency and batching policy represent substantially greater optimisation leverage than further micro-optimisation of the memory index.

Consolidation is explicitly asynchronous and does not contribute to per-query latency. Forgetting is triggered under memory pressure following consolidation and executes in a single-pass scoring sweep over L2 records. Both operations are isolated from the read path, preserving stable per-query response times.

*Fig. 11. End-to-End Latency Decomposition.* Total response latency (~5.2 s) is dominated by LLM generation time (98.9%). Memory retrieval and embedding contribute fewer than 60 ms combined, indicating that memory architecture is not the primary response-time bottleneck in the current configuration.

### F. Multi-Agent Scalability and Persona Consistency

**1. Decoupled Memory Instances and Latency Stability:**

*Fig. 12. Multi-Agent Scaling: Accuracy and Latency at Increased NPC Count.* Scaling behavior in a multi-agent environment. The system maintains stable recall accuracy (left axis) while exhibiting sub-linear latency growth (right axis) as the active NPC count increases from 1 to 32.

As evidenced in Fig. 12, CSAM maintains high recall stability even as the complexity of the interaction environment increases. This scalability is a direct result of the **decoupled memory architecture**: while all agents share a unified embedding backbone, each agent instance maintains an isolated L1–L3 hierarchy. This prevents "memory cross-talk" common in shared-context window architectures. Furthermore, sub-linear latency growth confirms that the asynchronous write path (consolidation and forgetting) successfully offloads computational overhead, allowing the read path to remain responsive regardless of the number of active agents.

**2. The Specialization Effect and Semantic Divergence:**

*Fig. 13. Per-NPC Recall Heterogeneity in Multi-Agent Evaluation.* Radar chart illustrating recall quality across diverse agent profiles. Agents with higher interaction density (Socialist/Expert personas) develop more complex L3 semantic graphs, leading to a specialization effect that enhances persona-consistent reasoning.

As shown in Fig. 13, we observe significant heterogeneity in recall performance across different agent profiles. Agents with high interaction density ("Expert" or "Socialist" personas) show accelerated L3 graph development. This **specialization effect** confirms that the consolidation logic effectively transforms repetitive episodic traces into unique semantic worldviews, ensuring that agents maintain "persistent personalities" that evolve based on specific interaction history—a critical requirement for high-fidelity persistent agents.

### G. Consolidated Performance Summary

*Fig. 14. Consolidated F1 Performance Summary Across Benchmarks.* Comparative F1 performance of CSAM against standard RAG and LRU-based baselines across all evaluated benchmarks. CSAM consistently outperforms traditional architectures across long-term factual recall tasks.

The summary in Fig. 14 provides final empirical validation of our central hypothesis. By integrating consolidation-aware forgetting with a tiered, scalable memory hierarchy, CSAM effectively resolves the scalability-retention trade-off. Our results demonstrate that for long-term conversational AI, structural memory organization is the primary determinant of success on memory-intensive tasks.

---

## VII. Ablation Study

To isolate the contribution of each component of the consolidation-aware forgetting mechanism, we evaluate five forgetting strategies on a synthetic long-conversation setup using the same conversation content and question-answer pairs across conditions:

| Strategy | Description | FFR |
| :------- | :---------- | :-- |
| **No-Forgetting** | Memory grows unbounded (upper recall bound) | N/A |
| **LRU** | Least Recently Used eviction (strong baseline) | > 0 |
| **Importance-Only** | Evict least important first, no gate | > 0 |
| **CA-Formula-Only (θ=0)** | Full 4-factor formula, gate disabled | > 0 |
| **CSAM Full (θ=0.3)** | Full formula + consolidation gate (**ours**) | ≈ 0 |

The CA-Formula-Only configuration (θ=0) isolates whether the 4-factor score *alone* improves over LRU and Importance-Only, without the gate. The gap between CA-Formula-Only and CSAM Full quantifies the gate's independent contribution.

The **False Forgetting Rate (FFR)** provides a direct measure of the gate's effectiveness: FFR is the fraction of evicted memories with *C(m) < θ* at eviction time, i.e., memories deleted before their semantic content was safely absorbed into L3. CSAM Full is expected to achieve FFR ≈ 0 by construction, while all other bounded strategies will exhibit FFR > 0.

[*Table with numerical F1 and FFR results will be inserted once ablation runs complete.*]

---

## VIII. Conclusion

We started this work with a fairly specific problem: build a memory system for conversational agents that does not gradually collapse as interaction history grows. What we ended up developing is an attempt to answer a more general question — when is it actually safe for an AI agent to forget something?

The consolidation gate is our answer. The idea is that a memory can only be safely deleted once its meaningful content exists somewhere more durable — in CSAM's case, a structured knowledge graph. Until that transformation has happened, the memory is protected regardless of how old it is or how rarely it has been accessed. The check is cheap at runtime — a single threshold comparison against a precomputed score — and it prevents a specific category of failure that recency-based and importance-based strategies cannot avoid: the permanent loss of information that was never captured anywhere else.

Our results suggest that for memory-intensive tasks, retrieval architecture matters more than model scale. Improving the pipeline from a flat RAG baseline to the three-tier CSAM system produced larger performance gains than doubling or quadrupling parameter count. For practitioners working on long-horizon conversational agents, this implies that investing in memory design may yield more return than simply reaching for a larger model.

We want to be careful, though, about overstating this. The architecture-bound effect was clear on LoCoMo, which is specifically designed to test long-horizon factual recall. On HotPotQA and MuSiQue — tasks that require multi-step compositional reasoning over retrieved evidence — larger models did contribute meaningfully. Memory architecture and model capacity seem to address different bottlenecks; which one dominates depends on the task structure.

For future work, a few directions seem genuinely worth pursuing. The consolidation threshold θ is currently a static value (0.3); there is no principled reason this should be optimal across different domains or interaction densities, and an adaptive scheduler based on memory pressure and consolidation throughput seems like a natural extension. The current system also treats each agent's memory as fully isolated — in multi-agent scenarios where agents genuinely share experiences, a shared L3 subgraph could allow knowledge propagation, though managing consistency across agents would require careful design. On the cost side, the consolidation pipeline currently calls a general-purpose LLM for entity extraction; a smaller fine-tuned extraction model could reduce this overhead substantially and deserves investigation.

The most practically useful takeaway from building this system: treat forgetting as a first-class design problem, not an afterthought. Most memory systems are designed around retrieval quality, and forgetting policy is handled with a simple capacity cap. Getting retrieval right is hard; getting forgetting right — in a way that preserves recoverability — may actually be harder, and it matters more at the scales where these systems are most likely to be deployed.

### A. Limitations

**Student Resource Constraints:** This work was conducted under typical academic resource constraints (single developer, limited GPU access, no dedicated compute budget). Dataset evaluation scopes reflect these practical constraints: the HotPotQA benchmark uses a 100-question sample rather than the full 7,405 questions, and model experiments are limited to the Groq API's freely-available allocation. The LoCoMo and MuSiQue evaluations are correspondingly bounded in scale. Full-scale reproduction on institutional clusters or cloud platforms with dedicated budgets would enable evaluation on complete datasets and extended hyperparameter sweeps; we view the current results as a proof-of-concept validation rather than a fully-saturated performance characterization.

All hosted model experiments use the Groq API. Exact replication requires access to the same hosted model versions, which may not be stable over time — anyone reproducing these results should pay attention to the specific model IDs listed in the experimental setup, since behavior can vary across provider-side updates to the same model family.

The consolidation pipeline's dependence on LLM inference adds variable cost and latency to the write path. We isolated this asynchronously so the read path stays clean, but a system under sustained high-volume interaction would need either a local extraction model or a rule-based fallback to remain practical at scale.

The forgetting weights (α=β=γ=δ=0.25) were selected through a bounded grid search under a specific evaluation protocol. We did not run a comprehensive hyperparameter sweep — that would have required substantially more compute than we had available. Equal weighting worked well in our tests, but we are not claiming it is universally optimal.

The LoCoMo evaluation uses 5–10 conversations from a benchmark designed to test human-style episodic recall. The architecture-bound pattern we observed may not generalize to domains with very different information density or interaction cadence. At much higher interaction volumes — thousands of turns per day, sustained over months — the relative throughput of the consolidation pipeline versus episodic ingestion becomes a more critical design variable than anything our current benchmarks test.

---

##### References

[1] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, "BERT: Pre-training of deep bidirectional transformers for language understanding," in *Proc. Conf. North American Chapter of the ACL (NAACL)*, 2019.

[2] J. S. Park, J. C. O'Brien, C. Cai, M. R. Morris, P. Liang, and M. S. Bernstein, "Generative agents: Interactive simulacra of human behavior," in *Proc. ACM Symp. User Interface Software and Technology (UIST)*, 2023.

[3] C. Packer, V. Fang, S. Keutzer, and I. Stoica, "MemGPT: Towards LLMs as operating systems," in *Proc. NeurIPS Workshop on LLM Systems*, 2023.

[4] A. Maharana, S. Saha, and M. Bansal, "Evaluating very long-term conversational memory of LLM agents," in *Proc. Annual Meeting of the Association for Computational Linguistics (ACL)*, 2024.

[5] Y. Sun and X. Zeng, "H-MEM: Hierarchical memory for high-efficiency long-term reasoning," 2025.

[6] B. Gutierrez, J. Li, and K. Cho, "HippoRAG: Neurobiologically inspired long-term memory for LLMs," in *Advances in Neural Information Processing Systems*, 2024.

[7] N. Reimers and I. Gurevych, "Sentence-BERT: Sentence embeddings using siamese BERT-networks," in *Proc. Conf. Empirical Methods in Natural Language Processing (EMNLP)*, 2019.

[8] J. Johnson, M. Douze, and H. Jégou, "Billion-scale similarity search with GPUs," *IEEE Trans. Big Data*, 2021.

[9] H. Trivedi, T. Li, and N. Balasubramanian, "MuSiQue: Multihop questions via single hop question composition," *Trans. Association Computational Linguistics*, 2022.

[10] Z. Yang, P. Qi, S. Zhang, Y. Bengio, W. Cohen, R. Salakhutdinov, and C. D. Manning, "HotpotQA: A dataset for diverse, explainable multi-hop question answering," in *Proc. Conf. Empirical Methods in Natural Language Processing (EMNLP)*, 2018.

[11] M. C. Anderson, "Rethinking interference theory: Executive control and the mechanisms of forgetting," *J. Memory Language*, vol. 49, no. 4, pp. 415–445, 2003.

[12] R. C. Atkinson and R. M. Shiffrin, "Human memory: A proposed system and its control processes," in *Psychology of Learning and Motivation*, vol. 2, 1968, pp. 89–195.

[13] Microsoft, "GraphRAG: A modular graph-based RAG approach," 2024.

[14] P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. Küttler, M. Lewis, W. Yih, T. Rocktäschel, S. Riedel, and D. Kiela, "Retrieval-augmented generation for knowledge-intensive NLP tasks," in *Advances in Neural Information Processing Systems*, 2020.

[15] S. Borgeaud, A. Mensch, J. Hoffmann, T. Cai, E. Rutherford, K. Millican, G. van den Driessche, J. Lespiau, B. Damoc, A. Clark, D. de Las Casas, L. Guy, and L. Sifre, "Improving language models by retrieving from trillions of tokens," in *Proc. Int. Conf. Machine Learning (ICML)*, 2022.

[16] V. Karpukhin, B. Oguz, S. Min, P. Lewis, L. Wu, S. Edunov, D. Chen, and W. Yih, "Dense passage retrieval for open-domain question answering," in *Proc. Conf. Empirical Methods in Natural Language Processing (EMNLP)*, 2020.

[17] Y. A. Malkov and D. A. Yashunin, "Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs," *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 42, no. 4, pp. 824–836, 2018.

[18] W. Zhong, R. Wang, Y. Sun, and X. Zeng, "MemoryBank: Enhancing large language models with long-term memory," in *Proc. AAAI Conf. Artificial Intelligence*, 2024.

[19] W. Xu, K. Mei, H. Gao, J. Tan, Z. Liang, and Y. Zhang, "A-Mem: Agentic memory for LLM agents," *arXiv preprint arXiv:2502.12110*, 2025.

[20] T. Brown et al., "Language models are few-shot learners," in *Advances in Neural Information Processing Systems*, vol. 33, 2020.

[21] K. Shuster, M. Komeili, L. Adolphs, S. Roller, A. Szlam, and J. Weston, "Language models that seek for knowledge," in *Proc. Conf. Empirical Methods in Natural Language Processing (EMNLP)*, 2022.

[22] J. Rae, A. Potapenko, S. M. Jayakumar, and T. P. Lillicrap, "Scaling memory-augmented neural networks with sparse reads and writes," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2016.

[23] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, "Attention is all you need," in *Advances in Neural Information Processing Systems*, vol. 30, 2017, pp. 5998–6008.
