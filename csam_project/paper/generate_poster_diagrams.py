"""
Generate poster-specific diagrams for the DJS Sanghvi Conference Poster.
Outputs 3 high-resolution PNGs optimized for A1/A0 poster printing.

Run: python -m csam_project.paper.generate_poster_diagrams
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "poster_diagrams")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Poster Color Palette ────────────────────────────────────────────────
C = {
    'l1':         '#2563EB',  # Blue
    'l1_bg':      '#DBEAFE',
    'l2':         '#059669',  # Green
    'l2_bg':      '#D1FAE5',
    'l3':         '#7C3AED',  # Purple
    'l3_bg':      '#EDE9FE',
    'consol':     '#D97706',  # Amber
    'consol_bg':  '#FEF3C7',
    'forget':     '#DC2626',  # Red
    'forget_bg':  '#FEE2E2',
    'retrieval':  '#16A34A',  # Emerald
    'retrieval_bg': '#F0FDF4',
    'csam':       '#2563EB',
    'lru':        '#F59E0B',
    'importance': '#8B5CF6',
    'no_forget':  '#EF4444',
    'baseline':   '#9CA3AF',
    'dark':       '#1E293B',
    'text':       '#334155',
    'light_bg':   '#F8FAFC',
}

POSTER_DPI = 300  # Print quality


# ══════════════════════════════════════════════════════════════════════════
# DIAGRAM 1: Architecture Diagram (Center Column)
# ══════════════════════════════════════════════════════════════════════════
def draw_poster_architecture():
    """3-Tier CSAM Architecture — optimized for poster center column."""
    fig, ax = plt.subplots(figsize=(9, 12))
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 12)
    ax.axis('off')
    fig.patch.set_facecolor('white')

    # Title
    ax.text(4.5, 11.5, 'CSAM: 3-Tier Cognitive Memory Architecture',
            fontsize=14, fontweight='bold', ha='center', color=C['dark'])

    # ── Input ──
    inp = mpatches.FancyBboxPatch((3, 10.4), 3, 0.7, boxstyle="round,pad=0.12",
                                   facecolor='#F1F5F9', edgecolor='#64748B', linewidth=1.5)
    ax.add_patch(inp)
    ax.text(4.5, 10.75, 'User Query / Agent Interaction',
            fontsize=10, fontweight='bold', ha='center', color='#475569')

    # Arrow down
    ax.annotate('', xy=(4.5, 10.0), xytext=(4.5, 10.4),
                arrowprops=dict(arrowstyle='->', color='#64748B', lw=2))

    # ═══ L1 BOX ═══
    l1 = mpatches.FancyBboxPatch((0.5, 8.8), 4.5, 1.1, boxstyle="round,pad=0.15",
                                  facecolor=C['l1_bg'], edgecolor=C['l1'], linewidth=2.5)
    ax.add_patch(l1)
    ax.text(2.75, 9.55, 'L1: Working Memory', fontsize=12, fontweight='bold',
            ha='center', color=C['l1'])
    ax.text(2.75, 9.1, 'LRU Cache  |  20 items  |  <1 ms  |  O(1)',
            fontsize=8.5, ha='center', color='#3B82F6')

    # Arrow L1 → L2
    ax.annotate('', xy=(2.75, 7.8), xytext=(2.75, 8.8),
                arrowprops=dict(arrowstyle='->', color='#94A3B8', lw=2))
    ax.text(3.3, 8.3, 'overflow', fontsize=7.5, color='#94A3B8', style='italic')

    # ═══ L2 BOX ═══
    l2 = mpatches.FancyBboxPatch((0.5, 6.6), 4.5, 1.1, boxstyle="round,pad=0.15",
                                  facecolor=C['l2_bg'], edgecolor=C['l2'], linewidth=2.5)
    ax.add_patch(l2)
    ax.text(2.75, 7.35, 'L2: Episodic Memory', fontsize=12, fontweight='bold',
            ha='center', color=C['l2'])
    ax.text(2.75, 6.9, 'HNSW Index  |  384-dim  |  ~5 ms  |  O(log N)',
            fontsize=8.5, ha='center', color='#10B981')

    # Arrow L2 → L3
    ax.annotate('', xy=(2.75, 5.6), xytext=(2.75, 6.6),
                arrowprops=dict(arrowstyle='->', color='#94A3B8', lw=2))

    # ═══ L3 BOX ═══
    l3 = mpatches.FancyBboxPatch((0.5, 4.4), 4.5, 1.1, boxstyle="round,pad=0.15",
                                  facecolor=C['l3_bg'], edgecolor=C['l3'], linewidth=2.5)
    ax.add_patch(l3)
    ax.text(2.75, 5.15, 'L3: Semantic Knowledge Graph', fontsize=12, fontweight='bold',
            ha='center', color=C['l3'])
    ax.text(2.75, 4.7, 'NetworkX  |  Entity-Relation Triples  |  BFS Traversal',
            fontsize=8.5, ha='center', color='#8B5CF6')

    # ═══ CONSOLIDATION PIPELINE (Right side) ═══
    cp = mpatches.FancyBboxPatch((5.5, 5.8), 3, 1.8, boxstyle="round,pad=0.15",
                                  facecolor=C['consol_bg'], edgecolor=C['consol'], linewidth=2.5)
    ax.add_patch(cp)
    ax.text(7, 7.2, 'Consolidation', fontsize=11, fontweight='bold',
            ha='center', color='#92400E')
    ax.text(7, 6.75, 'Pipeline', fontsize=11, fontweight='bold',
            ha='center', color='#92400E')
    ax.text(7, 6.25, 'LLM Summarization\nEntity Extraction\nCoverage Tracking',
            fontsize=8, ha='center', color='#B45309', linespacing=1.3)

    # Arrow: L2 → Consolidation
    ax.annotate('', xy=(5.5, 7.0), xytext=(5.0, 7.15),
                arrowprops=dict(arrowstyle='->', color=C['consol'], lw=2))
    # Arrow: Consolidation → L3
    ax.annotate('', xy=(5.0, 4.95), xytext=(5.5, 5.8),
                arrowprops=dict(arrowstyle='->', color=C['l3'], lw=2,
                               connectionstyle="arc3,rad=0.25"))
    ax.text(5.8, 5.2, 'Summaries\n+ Entities', fontsize=7.5, ha='center',
            color=C['l3'], style='italic')

    # ═══ FORGETTING ENGINE (Right side, below consolidation) ═══
    fe = mpatches.FancyBboxPatch((5.5, 3.5), 3, 1.8, boxstyle="round,pad=0.15",
                                  facecolor=C['forget_bg'], edgecolor=C['forget'], linewidth=2.5)
    ax.add_patch(fe)
    ax.text(7, 5.0, 'Forgetting Engine', fontsize=11, fontweight='bold',
            ha='center', color='#991B1B')
    ax.text(7, 4.5, '(NOVEL)', fontsize=9, fontweight='bold',
            ha='center', color=C['forget'])
    ax.text(7, 3.95, '0.2·R + 0.2·(1-I)\n+ 0.3·C + 0.3·D',
            fontsize=8.5, ha='center', color='#B91C1C', family='monospace',
            linespacing=1.3)

    # Arrow: Forgetting → L2 (dashed, eviction)
    ax.annotate('', xy=(5.0, 6.8), xytext=(5.5, 4.4),
                arrowprops=dict(arrowstyle='->', color=C['forget'], lw=2,
                               linestyle='dashed', connectionstyle="arc3,rad=-0.35"))
    ax.text(4.7, 5.7, 'Safe\nEviction', fontsize=7.5, ha='center',
            color=C['forget'], fontweight='bold')

    # ═══ HYBRID RETRIEVAL (Bottom spanning) ═══
    hr = mpatches.FancyBboxPatch((0.5, 2.2), 8, 1.0, boxstyle="round,pad=0.15",
                                  facecolor=C['retrieval_bg'], edgecolor=C['retrieval'],
                                  linewidth=2.5)
    ax.add_patch(hr)
    ax.text(4.5, 2.9, 'Hybrid Retrieval: L2 (HNSW kNN) + L3 (Graph Traversal) → MMR Re-ranking',
            fontsize=10, fontweight='bold', ha='center', color='#166534')
    ax.text(4.5, 2.5, 'λ-weighted diversity  |  Configurable L2/L3 balance  |  O(log N + k²)',
            fontsize=8.5, ha='center', color='#16A34A')

    # Arrows: each layer → retrieval
    for y, src_x in [(9.4, 2.75), (7.15, 2.75), (4.95, 2.75)]:
        ax.annotate('', xy=(0.5 + 0.2, 3.2), xytext=(0.5, y - 0.1),
                    arrowprops=dict(arrowstyle='->', color=C['retrieval'],
                                   lw=1.2, alpha=0.5, connectionstyle="arc3,rad=0.3"))

    # ═══ OUTPUT ═══
    out = mpatches.FancyBboxPatch((3, 0.8), 3, 0.7, boxstyle="round,pad=0.12",
                                   facecolor='#F1F5F9', edgecolor='#64748B', linewidth=1.5)
    ax.add_patch(out)
    ax.text(4.5, 1.15, 'Context-Rich Response via LLM',
            fontsize=10, fontweight='bold', ha='center', color='#475569')
    ax.annotate('', xy=(4.5, 1.5), xytext=(4.5, 2.2),
                arrowprops=dict(arrowstyle='<-', color='#64748B', lw=2))

    # ═══ KEY METRICS FOOTER ═══
    ax.text(4.5, 0.3, 'Retrieval: <10 ms   |   Memory Cap: 200 entries   |   '
            'Consolidation Ratio: 97%   |   LLM: 98.9% of total latency',
            fontsize=7.5, ha='center', color='#94A3B8', style='italic')

    plt.tight_layout(pad=0.5)
    path = os.path.join(OUTPUT_DIR, 'poster_architecture.png')
    plt.savefig(path, dpi=POSTER_DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info("Saved %s", path)


# ══════════════════════════════════════════════════════════════════════════
# DIAGRAM 2: Results Combined (Right Column — 4-panel figure)
# ══════════════════════════════════════════════════════════════════════════
def draw_poster_results():
    """4-panel results figure optimized for poster right column."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 11))
    fig.patch.set_facecolor('white')
    fig.suptitle('Experimental Results', fontsize=16, fontweight='bold',
                 color=C['dark'], y=0.98)

    for ax_row in axes:
        for ax in ax_row:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_facecolor('#FAFBFC')

    # ── Panel A: Ablation — Forgetting Strategy Comparison (Single-hop F1) ──
    ax = axes[0, 0]
    strategies = ['LRU', 'Importance\n-Only', 'Consolidation\n-Aware (Ours)', 'No Forgetting\n(Unbounded)']
    single_f1 = [0.223, 0.271, 0.331, 0.274]
    mem_count = [76, 76, 76, 500]
    colors = [C['lru'], C['importance'], C['csam'], C['no_forget']]

    bars = ax.bar(strategies, single_f1, color=colors, width=0.65, edgecolor='white', linewidth=1.5)
    for bar, val, mem in zip(bars, single_f1, mem_count):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.text(bar.get_x() + bar.get_width() / 2, 0.02,
                f'{mem} mem', ha='center', va='bottom', fontsize=7, color='white',
                fontweight='bold')

    ax.set_ylabel('Single-hop F1 Score', fontsize=10, fontweight='bold')
    ax.set_title('(a) Forgetting Ablation — Single-hop Recall', fontsize=10,
                 fontweight='bold', pad=8)
    ax.set_ylim(0, 0.42)
    ax.tick_params(axis='x', labelsize=8)

    # Add "48% better" annotation
    ax.annotate('+48%\nvs LRU', xy=(2, 0.331), xytext=(2.7, 0.38),
                fontsize=8, fontweight='bold', color=C['csam'],
                arrowprops=dict(arrowstyle='->', color=C['csam'], lw=1.2),
                ha='center')

    # ── Panel B: Cross-Benchmark F1 ──
    ax = axes[0, 1]
    benchmarks = ['LoCoMo\n(Conv)', 'MuSiQue\n(2-4 hop)', 'HotPotQA\n(2-hop)']
    f1_8b = [0.324, 0.440, 0.654]
    f1_17b = [0.365, 0.409, 0.630]
    f1_70b = [0.329, 0.535, 0.729]

    x = np.arange(len(benchmarks))
    w = 0.22
    bars1 = ax.bar(x - w, f1_8b, w, label='Llama 8B', color='#4C72B0',
                   edgecolor='white', linewidth=0.8)
    bars2 = ax.bar(x, f1_17b, w, label='Llama 17B', color='#55A868',
                   edgecolor='white', linewidth=0.8)
    bars3 = ax.bar(x + w, f1_70b, w, label='Llama 70B', color='#C44E52',
                   edgecolor='white', linewidth=0.8)

    for bars_group in [bars1, bars2, bars3]:
        for bar in bars_group:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                    f'{h:.2f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

    ax.set_ylabel('F1 Score', fontsize=10, fontweight='bold')
    ax.set_title('(b) Cross-Benchmark F1 by Model Scale', fontsize=10,
                 fontweight='bold', pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, fontsize=8)
    ax.set_ylim(0, 0.85)
    ax.legend(fontsize=8, loc='upper left', framealpha=0.9)

    # ── Panel C: Memory Efficiency ──
    ax = axes[1, 0]
    strategies_eff = ['CSAM\n(Consolidation)', 'No Forgetting']
    memories = [200, 520]
    recall = [80, 80]

    x_pos = np.arange(len(strategies_eff))
    bars_mem = ax.bar(x_pos - 0.15, memories, 0.3, label='Memories Stored',
                      color=[C['csam'], C['no_forget']], edgecolor='white', linewidth=1.5)
    ax2 = ax.twinx()
    bars_rec = ax2.bar(x_pos + 0.15, recall, 0.3, label='Recall %',
                       color=[C['csam'], C['no_forget']], edgecolor='white',
                       linewidth=1.5, alpha=0.4, hatch='///')

    for bar, val in zip(bars_mem, memories):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 8,
                f'{val}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    for bar, val in zip(bars_rec, recall):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                 f'{val}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('Memories Stored', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Recall Accuracy (%)', fontsize=10, fontweight='bold')
    ax.set_title('(c) Memory Efficiency: Same Recall, 62% Less Storage',
                 fontsize=10, fontweight='bold', pad=8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(strategies_eff, fontsize=9)
    ax.set_ylim(0, 600)
    ax2.set_ylim(0, 105)

    # 2.6x annotation
    ax.annotate('2.6× fewer\nmemories', xy=(0, 200), xytext=(0.5, 420),
                fontsize=9, fontweight='bold', color=C['csam'],
                arrowprops=dict(arrowstyle='->', color=C['csam'], lw=1.5),
                ha='center')

    # ── Panel D: CSAM vs Baseline RAG on LoCoMo ──
    ax = axes[1, 1]
    systems = ['Baseline\nRAG', 'CSAM\n(8B)', 'CSAM\n(17B)', 'CSAM\n(70B)',
               'GPT-3.5*\n(Published)', 'GPT-4*\n(Published)']
    scores = [0.051, 0.324, 0.365, 0.329, 0.378, 0.321]
    bar_colors = [C['baseline'], C['csam'], '#2563EB', '#1D4ED8', '#6B7280', '#6B7280']
    edge = ['white'] * 6

    bars = ax.bar(systems, scores, color=bar_colors, width=0.6,
                  edgecolor=edge, linewidth=1.5)
    for bar, s in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                f'{s:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    # 6-7x annotation
    ax.annotate('6-7× over\nbaseline RAG', xy=(0, 0.051), xytext=(0.8, 0.25),
                fontsize=9, fontweight='bold', color=C['csam'],
                arrowprops=dict(arrowstyle='->', color=C['csam'], lw=1.5),
                ha='center')

    ax.set_ylabel('F1 Score', fontsize=10, fontweight='bold')
    ax.set_title('(d) LoCoMo: CSAM vs Baselines & Published Results',
                 fontsize=10, fontweight='bold', pad=8)
    ax.set_ylim(0, 0.5)
    ax.tick_params(axis='x', labelsize=7.5)
    ax.text(0.98, 0.02, '*Published scores from\nMaharana et al. (2024)',
            transform=ax.transAxes, fontsize=6.5, color='gray', ha='right',
            va='bottom', style='italic')

    plt.tight_layout(pad=1.5, rect=[0, 0, 1, 0.96])
    path = os.path.join(OUTPUT_DIR, 'poster_results_combined.png')
    plt.savefig(path, dpi=POSTER_DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info("Saved %s", path)


# ══════════════════════════════════════════════════════════════════════════
# DIAGRAM 3: Forgetting Formula Visual Breakdown
# ══════════════════════════════════════════════════════════════════════════
def draw_poster_forgetting_formula():
    """Visual breakdown of the 4-factor consolidation-aware forgetting formula."""
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 7)
    ax.axis('off')
    fig.patch.set_facecolor('white')

    # Title
    ax.text(4.5, 6.6, 'Consolidation-Aware Forgetting Formula',
            fontsize=14, fontweight='bold', ha='center', color=C['dark'])
    ax.text(4.5, 6.2, '(Novel Contribution)',
            fontsize=11, fontweight='bold', ha='center', color=C['forget'])

    # Formula bar
    formula_box = mpatches.FancyBboxPatch((0.3, 5.2), 8.4, 0.8, boxstyle="round,pad=0.15",
                                           facecolor='#F8FAFC', edgecolor=C['dark'], linewidth=2)
    ax.add_patch(formula_box)
    ax.text(4.5, 5.6, r'ForgetScore(m) = 0.2·R(m) + 0.2·(1-I(m)) + 0.3·C(m) + 0.3·D(m)',
            fontsize=12, fontweight='bold', ha='center', color=C['dark'],
            family='monospace')

    # 4 factor boxes
    factors = [
        {
            'label': 'R(m)',
            'title': 'Recency\nDecay',
            'weight': '20%',
            'desc': 'Time since\nlast access\n(age / 365)',
            'color': '#0EA5E9',
            'bg': '#E0F2FE',
            'x': 0.5,
        },
        {
            'label': '1-I(m)',
            'title': 'Inverse\nImportance',
            'weight': '20%',
            'desc': 'Low importance\n→ more likely\nto forget',
            'color': '#F59E0B',
            'bg': '#FEF3C7',
            'x': 2.5,
        },
        {
            'label': 'C(m)',
            'title': 'Consolidation\nCoverage ★',
            'weight': '30%',
            'desc': 'How well L3\nsummary captures\nthis memory',
            'color': '#059669',
            'bg': '#D1FAE5',
            'x': 4.5,
        },
        {
            'label': 'D(m)',
            'title': 'L3\nRedundancy ★',
            'weight': '30%',
            'desc': 'Similarity to\nany L3 node\n(graph overlap)',
            'color': '#7C3AED',
            'bg': '#EDE9FE',
            'x': 6.5,
        },
    ]

    for f in factors:
        x = f['x']
        # Factor box
        box = mpatches.FancyBboxPatch((x, 2.0), 1.8, 2.8, boxstyle="round,pad=0.12",
                                       facecolor=f['bg'], edgecolor=f['color'], linewidth=2)
        ax.add_patch(box)

        # Weight badge
        badge = mpatches.FancyBboxPatch((x + 0.5, 4.3), 0.8, 0.35, boxstyle="round,pad=0.08",
                                         facecolor=f['color'], edgecolor='white', linewidth=1)
        ax.add_patch(badge)
        ax.text(x + 0.9, 4.48, f['weight'], fontsize=9, fontweight='bold',
                ha='center', va='center', color='white')

        # Title
        ax.text(x + 0.9, 3.85, f['title'], fontsize=9.5, fontweight='bold',
                ha='center', va='center', color=f['color'], linespacing=1.2)

        # Label
        ax.text(x + 0.9, 3.15, f['label'], fontsize=11, fontweight='bold',
                ha='center', va='center', color=f['color'], family='monospace')

        # Description
        ax.text(x + 0.9, 2.4, f['desc'], fontsize=7.5, ha='center', va='center',
                color=f['color'], linespacing=1.2, alpha=0.85)

        # Arrow from formula to factor
        ax.annotate('', xy=(x + 0.9, 4.8), xytext=(x + 0.9, 5.2),
                    arrowprops=dict(arrowstyle='->', color=f['color'], lw=1.5))

    # Novel marker
    novel_box = mpatches.FancyBboxPatch((4.2, 1.1), 4.5, 0.65, boxstyle="round,pad=0.1",
                                         facecolor=C['forget_bg'], edgecolor=C['forget'],
                                         linewidth=2, linestyle='--')
    ax.add_patch(novel_box)
    ax.text(6.45, 1.42, '★ Novel: 60% of score from consolidation-aware factors (C + D)',
            fontsize=8.5, fontweight='bold', ha='center', color='#991B1B')

    # Protection rule
    prot_box = mpatches.FancyBboxPatch((0.3, 0.25), 8.4, 0.6, boxstyle="round,pad=0.1",
                                        facecolor='#FFF7ED', edgecolor='#EA580C', linewidth=1.5)
    ax.add_patch(prot_box)
    ax.text(4.5, 0.55,
            'Protection Rule: If C(m) < 0.3 → Memory is PROTECTED (score = 0). '
            'No memory forgotten before L3 absorption.',
            fontsize=8, fontweight='bold', ha='center', color='#C2410C')

    # Traditional vs Novel bracket labels
    ax.annotate('', xy=(0.5, 1.9), xytext=(4.3, 1.9),
                arrowprops=dict(arrowstyle='-', color='#94A3B8', lw=1))
    ax.text(2.4, 1.75, 'Traditional factors (40%)', fontsize=7.5,
            ha='center', color='#94A3B8', style='italic')
    ax.annotate('', xy=(4.5, 1.9), xytext=(8.3, 1.9),
                arrowprops=dict(arrowstyle='-', color=C['forget'], lw=1))
    ax.text(6.4, 1.75, 'Novel factors (60%)', fontsize=7.5,
            ha='center', color=C['forget'], fontweight='bold')

    plt.tight_layout(pad=0.5)
    path = os.path.join(OUTPUT_DIR, 'poster_forgetting_formula.png')
    plt.savefig(path, dpi=POSTER_DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info("Saved %s", path)


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════
def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("Generating poster diagrams to %s ...", OUTPUT_DIR)

    draw_poster_architecture()
    logger.info("[OK] poster_architecture.png")

    draw_poster_results()
    logger.info("[OK] poster_results_combined.png")

    draw_poster_forgetting_formula()
    logger.info("[OK] poster_forgetting_formula.png")

    logger.info("Done — 3 poster diagrams saved to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
