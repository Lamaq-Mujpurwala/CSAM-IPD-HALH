"""
Generate architecture and flow diagrams as PNG using matplotlib.
These replace the Mermaid diagrams for environments that don't render Mermaid.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import os

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "diagrams")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def draw_architecture_diagram():
    """CSAM 3-Tier Memory Architecture"""
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis('off')
    fig.patch.set_facecolor('white')

    # Title
    ax.text(7, 8.5, 'CSAM Architecture: 3-Tier Cognitive Memory', 
            fontsize=18, fontweight='bold', ha='center', va='center',
            color='#1a1a2e')

    # ── L1 Box ──
    l1 = mpatches.FancyBboxPatch((1, 6.8), 5, 1.2, boxstyle="round,pad=0.15",
                                  facecolor='#DBEAFE', edgecolor='#2563EB', linewidth=2)
    ax.add_patch(l1)
    ax.text(3.5, 7.6, 'L1: Working Memory Cache', fontsize=13, fontweight='bold',
            ha='center', color='#1E40AF')
    ax.text(3.5, 7.15, 'LRU Cache  •  20 items  •  <1ms access  •  TTL eviction',
            fontsize=9, ha='center', color='#3B82F6')

    # ── L2 Box ──
    l2 = mpatches.FancyBboxPatch((1, 4.8), 5, 1.2, boxstyle="round,pad=0.15",
                                  facecolor='#D1FAE5', edgecolor='#059669', linewidth=2)
    ax.add_patch(l2)
    ax.text(3.5, 5.6, 'L2: Episodic Memory (Vector Store)', fontsize=13, fontweight='bold',
            ha='center', color='#065F46')
    ax.text(3.5, 5.15, 'FAISS HNSW  •  384-dim  •  O(log N)  •  all-MiniLM-L6-v2',
            fontsize=9, ha='center', color='#10B981')

    # ── L3 Box ──
    l3 = mpatches.FancyBboxPatch((1, 2.8), 5, 1.2, boxstyle="round,pad=0.15",
                                  facecolor='#EDE9FE', edgecolor='#7C3AED', linewidth=2)
    ax.add_patch(l3)
    ax.text(3.5, 3.6, 'L3: Knowledge Graph', fontsize=13, fontweight='bold',
            ha='center', color='#5B21B6')
    ax.text(3.5, 3.15, 'NetworkX  •  Entity Nodes  •  Semantic Summaries',
            fontsize=9, ha='center', color='#8B5CF6')

    # ── Arrows between layers ──
    ax.annotate('', xy=(3.5, 6.8), xytext=(3.5, 6.0),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    ax.annotate('', xy=(3.5, 4.8), xytext=(3.5, 4.0),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

    # ── Consolidation Pipeline ──
    cp = mpatches.FancyBboxPatch((8, 4.5), 4.5, 1.5, boxstyle="round,pad=0.15",
                                  facecolor='#FEF3C7', edgecolor='#D97706', linewidth=2)
    ax.add_patch(cp)
    ax.text(10.25, 5.4, 'Consolidation Pipeline', fontsize=12, fontweight='bold',
            ha='center', color='#92400E')
    ax.text(10.25, 4.95, 'LLM-powered summarization\nEntity extraction  •  Every 20 turns',
            fontsize=9, ha='center', color='#B45309')

    # Arrow: L2 -> Consolidation
    ax.annotate('', xy=(8, 5.4), xytext=(6, 5.4),
                arrowprops=dict(arrowstyle='->', color='#D97706', lw=2))
    # Arrow: Consolidation -> L3
    ax.annotate('', xy=(6, 3.4), xytext=(8, 4.5),
                arrowprops=dict(arrowstyle='->', color='#7C3AED', lw=2, 
                               connectionstyle="arc3,rad=0.3"))
    ax.text(7.8, 3.7, 'Summaries\n& Entities', fontsize=8, ha='center', color='#7C3AED')

    # ── Forgetting Engine ──
    fe = mpatches.FancyBboxPatch((8, 2.5), 4.5, 1.4, boxstyle="round,pad=0.15",
                                  facecolor='#FEE2E2', edgecolor='#DC2626', linewidth=2)
    ax.add_patch(fe)
    ax.text(10.25, 3.4, 'Forgetting Engine', fontsize=12, fontweight='bold',
            ha='center', color='#991B1B')
    ax.text(10.25, 2.95, 'Consolidation-aware scoring\nδ = L3 redundancy  •  importance weight',
            fontsize=9, ha='center', color='#DC2626')

    # Arrow: Forgetting -> L2
    ax.annotate('', xy=(6, 5.0), xytext=(8, 3.2),
                arrowprops=dict(arrowstyle='->', color='#DC2626', lw=2, linestyle='dashed',
                               connectionstyle="arc3,rad=-0.3"))
    ax.text(7.5, 4.6, 'Safe\nDeletion', fontsize=8, ha='center', color='#DC2626')

    # ── Hybrid Retrieval ──
    hr = mpatches.FancyBboxPatch((8, 6.8), 4.5, 1.2, boxstyle="round,pad=0.15",
                                  facecolor='#F0FDF4', edgecolor='#16A34A', linewidth=2)
    ax.add_patch(hr)
    ax.text(10.25, 7.6, 'Hybrid Retrieval (MMR)', fontsize=12, fontweight='bold',
            ha='center', color='#166534')
    ax.text(10.25, 7.15, 'L1 + L2 + L3 -> Ranked Results -> LLM Response',
            fontsize=9, ha='center', color='#16A34A')

    # Arrows: Layers -> Retrieval
    for y in [7.4, 5.4, 3.4]:
        ax.annotate('', xy=(8, 7.3), xytext=(6, y),
                    arrowprops=dict(arrowstyle='->', color='#16A34A', lw=1.5, alpha=0.6,
                                   connectionstyle="arc3,rad=0.2"))

    # ── Input/Output ──
    ax.text(0.5, 7.4, '🎮\nPlayer\nMessage', fontsize=10, ha='center', va='center',
            fontweight='bold', color='#374151',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#F3F4F6', edgecolor='#9CA3AF'))
    ax.annotate('', xy=(1, 7.4), xytext=(0.9, 7.4),
                arrowprops=dict(arrowstyle='->', color='#6B7280', lw=1.5))

    ax.text(13.3, 7.4, '💬\nNPC\nResponse', fontsize=10, ha='center', va='center',
            fontweight='bold', color='#374151',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#F3F4F6', edgecolor='#9CA3AF'))
    ax.annotate('', xy=(12.8, 7.4), xytext=(12.5, 7.4),
                arrowprops=dict(arrowstyle='->', color='#6B7280', lw=1.5))

    # ── Legend ──
    ax.text(1, 1.8, 'Key Metrics:', fontsize=10, fontweight='bold', color='#374151')
    metrics = [
        ('L1 Access: <1ms', '#2563EB'),
        ('L2 Search: ~5ms', '#059669'),
        ('L3 Query: ~2ms', '#7C3AED'),
        ('LLM Gen: ~5000ms', '#DC2626'),
    ]
    for i, (txt, color) in enumerate(metrics):
        ax.text(1 + i*3.2, 1.3, txt, fontsize=9, color=color, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'diagram_architecture.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] diagram_architecture.png")


def draw_memory_flow_diagram():
    """Memory flow: Store -> Fill -> Consolidate -> Forget -> Recall"""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5)
    ax.axis('off')
    fig.patch.set_facecolor('white')

    ax.text(7, 4.6, 'CSAM Memory Flow Pipeline', fontsize=16, fontweight='bold',
            ha='center', color='#1a1a2e')

    steps = [
        ('1. STORE', 'Player shares fact\n-> L2 memory\n(importance=0.95)', '#DBEAFE', '#2563EB'),
        ('2. FILL', '50+ conversations\nhappen naturally\n(filler memories)', '#FEF3C7', '#D97706'),
        ('3. CONSOLIDATE', 'Group similar ->\nL3 summaries\n(97% ratio)', '#D1FAE5', '#059669'),
        ('4. FORGET', 'Safe deletion\nvia δ-score\n(bounded at 200)', '#FEE2E2', '#DC2626'),
        ('5. RECALL', 'Hybrid retrieval\nL1+L2+L3 -> LLM\n(80% accuracy)', '#EDE9FE', '#7C3AED'),
    ]

    for i, (title, desc, bg, fg) in enumerate(steps):
        x = 1.2 + i * 2.5
        box = mpatches.FancyBboxPatch((x, 1.2), 2, 2.8, boxstyle="round,pad=0.15",
                                       facecolor=bg, edgecolor=fg, linewidth=2)
        ax.add_patch(box)
        ax.text(x+1, 3.5, title, fontsize=11, fontweight='bold', ha='center', color=fg)
        ax.text(x+1, 2.4, desc, fontsize=8.5, ha='center', color=fg, va='center')

        if i < 4:
            ax.annotate('', xy=(x+2.2, 2.6), xytext=(x+2, 2.6),
                        arrowprops=dict(arrowstyle='->', color='#6B7280', lw=2))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'diagram_memory_flow.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] diagram_memory_flow.png")


def draw_consolidation_diagram():
    """Consolidation: L2 memories -> grouped -> summarized -> L3 nodes"""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')
    fig.patch.set_facecolor('white')

    ax.text(6, 5.6, 'Consolidation Pipeline Detail', fontsize=16, fontweight='bold',
            ha='center', color='#1a1a2e')

    # L2 memories (left)
    ax.text(1.5, 5, 'L2 Memories (Raw)', fontsize=12, fontweight='bold', ha='center', color='#059669')
    memories = [
        '"Player said: My name is Alexander"',
        '"Player asked about the weather"',
        '"Player mentioned Northern Kingdom"',
        '"Player asked for a drink"',
        '"Player said lucky number is 42"',
    ]
    for i, mem in enumerate(memories):
        y = 4.3 - i * 0.7
        box = mpatches.FancyBboxPatch((0.1, y-0.2), 2.8, 0.5, boxstyle="round,pad=0.1",
                                       facecolor='#D1FAE5', edgecolor='#059669', linewidth=1)
        ax.add_patch(box)
        ax.text(1.5, y, mem, fontsize=7, ha='center', va='center', color='#065F46')

    # Grouping step (middle)
    ax.text(5.5, 5, 'Semantic Grouping', fontsize=12, fontweight='bold', ha='center', color='#D97706')
    groups = [
        ('Group A', '"Alexander" + "Northern Kingdom"', '#FEF3C7'),
        ('Group B', '"weather" + "drink"', '#FEF3C7'),
        ('Group C', '"lucky number 42"', '#FEF3C7'),
    ]
    for i, (label, desc, bg) in enumerate(groups):
        y = 4.0 - i * 1.2
        box = mpatches.FancyBboxPatch((4.2, y-0.3), 2.6, 0.8, boxstyle="round,pad=0.1",
                                       facecolor=bg, edgecolor='#D97706', linewidth=1.5)
        ax.add_patch(box)
        ax.text(5.5, y+0.15, label, fontsize=9, fontweight='bold', ha='center', color='#92400E')
        ax.text(5.5, y-0.1, desc, fontsize=7, ha='center', color='#B45309')

    # L3 nodes (right)
    ax.text(9.5, 5, 'L3 Knowledge Nodes', fontsize=12, fontweight='bold', ha='center', color='#7C3AED')
    nodes = [
        ('Entity: Alexander', 'From Northern Kingdom'),
        ('Summary', 'Casual tavern conversation'),
        ('Entity: 42', 'Lucky number, tavern games'),
    ]
    for i, (label, desc) in enumerate(nodes):
        y = 4.0 - i * 1.2
        box = mpatches.FancyBboxPatch((8.2, y-0.3), 2.6, 0.8, boxstyle="round,pad=0.1",
                                       facecolor='#EDE9FE', edgecolor='#7C3AED', linewidth=1.5)
        ax.add_patch(box)
        ax.text(9.5, y+0.15, label, fontsize=9, fontweight='bold', ha='center', color='#5B21B6')
        ax.text(9.5, y-0.1, desc, fontsize=7, ha='center', color='#8B5CF6')

    # Arrows
    for y in [4.0, 3.3, 2.6]:
        ax.annotate('', xy=(4.1, min(y, 4.0)), xytext=(3, y),
                    arrowprops=dict(arrowstyle='->', color='#D97706', lw=1.5))
    for y in [4.0, 2.8, 1.6]:
        ax.annotate('', xy=(8.1, y), xytext=(6.9, y),
                    arrowprops=dict(arrowstyle='->', color='#7C3AED', lw=1.5))

    # LLM label
    ax.text(7.5, 4.8, 'LLM\nSummarization', fontsize=8, ha='center', color='#6B7280',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#F3F4F6', edgecolor='#D1D5DB'))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'diagram_consolidation.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] diagram_consolidation.png")


if __name__ == '__main__':
    print("Generating Mermaid-equivalent diagrams as PNG...")
    print(f"Output: {OUTPUT_DIR}\n")
    draw_architecture_diagram()
    draw_memory_flow_diagram()
    draw_consolidation_diagram()
    print(f"\n[OK] All diagrams saved to {OUTPUT_DIR}")
