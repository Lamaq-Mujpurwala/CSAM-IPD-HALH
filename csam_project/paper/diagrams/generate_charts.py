"""
CSAM Paper — Programmatic chart generation.
Produces all 5 publication figures as PDF + PNG.
Run from any directory: python generate_charts.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

OUT = os.path.dirname(os.path.abspath(__file__))

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ── Palette (colorblind-safe) ─────────────────────────────────────────────────
PAL = {
    "no_forget":   "#7f7f7f",
    "lru":         "#1f77b4",
    "importance":  "#ff7f0e",
    "ca_formula":  "#2ca02c",
    "ca_ours":     "#d62728",
    "csam":        "#1f77b4",
    "baseline":    "#aec7e8",
    "scout":       "#9467bd",
    "g70b":        "#8c564b",
    "gpt":         "#e377c2",
    "g8b":         "#17becf",
}

def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"))
    print(f"  Saved {name}.pdf / .png")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Ablation: Mean F1 and FFR per strategy
# ══════════════════════════════════════════════════════════════════════════════
def fig_ablation():
    strategies = ["No-Forgetting", "LRU", "Importance", "CA-Formula\nOnly", "CA-Ours"]
    f1_mean    = [0.4847, 0.4854, 0.4582, 0.5016, 0.5095]
    f1_std     = [0.0420, 0.0560, 0.0820, 0.0300, 0.0340]
    ffr        = [0.000,  0.128,  0.197,  0.068,  0.000 ]
    colors     = [PAL["no_forget"], PAL["lru"], PAL["importance"],
                  PAL["ca_formula"], PAL["ca_ours"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # ── Panel A: Mean F1 ──────────────────────────────────────────────────────
    x = np.arange(len(strategies))
    bars = ax1.bar(x, f1_mean, color=colors, width=0.6,
                   yerr=f1_std, capsize=4, error_kw={"linewidth": 1.2})
    ax1.set_xticks(x)
    ax1.set_xticklabels(strategies, rotation=15, ha="right")
    ax1.set_ylabel("Mean F1")
    ax1.set_ylim(0.38, 0.58)
    ax1.set_title("(a) Mean F1 Across 5 Runs")
    ax1.axhline(f1_mean[0], color=PAL["no_forget"], linestyle="--",
                linewidth=0.8, alpha=0.6)
    for bar, val in zip(bars, f1_mean):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.003,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=8)
    # Annotate CA-Ours
    ax1.annotate("Best bounded-memory\nstrategy",
                 xy=(4, 0.5095), xytext=(3.1, 0.540),
                 arrowprops=dict(arrowstyle="->", color=PAL["ca_ours"]),
                 color=PAL["ca_ours"], fontsize=8)
    ax1.text(0.02, 0.97, "84.8% memory\nreduction vs.\nNo-Forgetting",
             transform=ax1.transAxes, fontsize=8, va="top",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    # ── Panel B: FFR ──────────────────────────────────────────────────────────
    bar_colors_ffr = [PAL["no_forget"] if v == 0.0 and i == 0
                      else (PAL["ca_ours"] if i == 4 else colors[i])
                      for i, v in enumerate(ffr)]
    bars2 = ax2.bar(x, ffr, color=bar_colors_ffr, width=0.6)
    ax2.set_xticks(x)
    ax2.set_xticklabels(strategies, rotation=15, ha="right")
    ax2.set_ylabel("False Forgetting Rate (FFR)")
    ax2.set_ylim(0.0, 0.26)
    ax2.set_title("(b) False Forgetting Rate (FFR ↓ is better)")
    ax2.axhline(0.0, color="red", linestyle="--", linewidth=1.2, alpha=0.8,
                label="Target: FFR = 0")
    ax2.legend(loc="upper right")
    for bar, val in zip(bars2, ffr):
        label = "0.000\n(Safe)" if val == 0 else f"{val:.3f}"
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 val + 0.004, label, ha="center", va="bottom", fontsize=8)
    ax2.text(4, 0.01, "FFR=0\n(bounded)", ha="center", va="bottom",
             color=PAL["ca_ours"], fontsize=8, fontweight="bold")

    fig.suptitle("Figure 2: Ablation Study — Forgetting Strategy Comparison\n"
                 "(5 runs: Llama 3.1 8B × 3 seeds, Scout-17B, Llama 3.3 70B)",
                 fontsize=10, y=1.02)
    fig.tight_layout()
    save(fig, "fig2_ablation")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — HotPotQA Results (3 panels)
# ══════════════════════════════════════════════════════════════════════════════
def fig_hotpotqa():
    models      = ["8B\n(mean)", "Scout\n17B", "70B", "GPT-OSS\n120B"]
    f1          = [0.672, 0.698, 0.742, 0.641]
    f1_err      = [0.016, 0.0,   0.0,   0.0  ]
    bridge_f1   = [0.702, 0.749, 0.777, 0.612]
    comp_f1     = [0.650, 0.509, 0.609, 0.750]
    latency     = [5215,  1994,  2399,  3735 ]
    model_colors= [PAL["g8b"], PAL["scout"], PAL["g70b"], PAL["gpt"]]

    fig = plt.figure(figsize=(13, 4))
    gs = GridSpec(1, 3, figure=fig, wspace=0.35)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    # ── Panel A: F1 by model ─────────────────────────────────────────────────
    x = np.arange(len(models))
    bars = ax1.bar(x, f1, color=model_colors, width=0.6,
                   yerr=f1_err, capsize=4, error_kw={"linewidth": 1.2})
    ax1.set_xticks(x); ax1.set_xticklabels(models)
    ax1.set_ylabel("F1"); ax1.set_ylim(0.55, 0.82)
    ax1.set_title("(a) F1 by Model")
    for bar, val in zip(bars, f1):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.004,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    ax1.annotate("Best", xy=(2, 0.742), xytext=(2.5, 0.76),
                 arrowprops=dict(arrowstyle="->", color=PAL["g70b"]),
                 color=PAL["g70b"], fontsize=8)

    # ── Panel B: Bridge vs Comparison ────────────────────────────────────────
    w = 0.35
    x2 = np.arange(len(models))
    b1 = ax2.bar(x2 - w/2, bridge_f1, w, label="Bridge", color=[
        matplotlib.colors.to_rgba(c, 0.85) for c in model_colors])
    b2 = ax2.bar(x2 + w/2, comp_f1,   w, label="Comparison", color=[
        matplotlib.colors.to_rgba(c, 0.45) for c in model_colors])
    ax2.set_xticks(x2); ax2.set_xticklabels(models)
    ax2.set_ylabel("F1"); ax2.set_ylim(0.4, 0.85)
    ax2.set_title("(b) Bridge vs. Comparison F1")
    ax2.legend(loc="upper right", fontsize=8)
    for bar, val in zip(list(b1) + list(b2), bridge_f1 + comp_f1):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.005,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=7)

    # ── Panel C: Latency vs F1 scatter ───────────────────────────────────────
    for lat, score, label, col in zip(latency, f1, models, model_colors):
        ax3.scatter(lat, score, color=col, s=120, zorder=5)
        offset = (-320, 0.006) if label == "Scout\n17B" else (50, 0.004)
        ax3.annotate(label.replace("\n", " "),
                     xy=(lat, score),
                     xytext=(lat + offset[0], score + offset[1]),
                     fontsize=8, color=col)
    # Pareto frontier (lower latency, higher F1)
    pareto_pts = sorted(zip(latency, f1))
    px, py = zip(*[(1994, 0.698), (2399, 0.742)])
    ax3.plot(px, py, "k--", linewidth=0.8, alpha=0.5, label="Pareto frontier")
    ax3.set_xlabel("Avg Latency (ms)"); ax3.set_ylabel("F1")
    ax3.set_title("(c) Latency–Accuracy Tradeoff")
    ax3.legend(fontsize=8)
    circle = plt.Circle((1994, 0.698), 150, color=PAL["scout"],
                         fill=False, linewidth=2, linestyle="--")
    ax3.add_patch(circle)
    ax3.text(1994, 0.670, "Pareto-optimal", ha="center", color=PAL["scout"],
             fontsize=8, fontweight="bold")

    fig.suptitle("Figure 3: HotPotQA Multi-Hop QA Results (100 questions, hard split)",
                 fontsize=10, y=1.02)
    fig.tight_layout()
    save(fig, "fig3_hotpotqa")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — LoCoMo: CSAM vs Flat-RAG Baseline
# ══════════════════════════════════════════════════════════════════════════════
def fig_locomo():
    models    = ["Llama 3.1\n8B", "Llama 4\nScout 17B", "Llama 3.3\n70B", "GPT-OSS\n120B"]
    csam_f1   = [0.3419, 0.3384, 0.3528, 0.2269]
    base_f1   = [0.3413, 0.3378, 0.3521, 0.2262]
    # 95% CI half-width from Micro F1 data (approximated as (upper-lower)/2)
    ci_half   = [(0.3659-0.3120)/2, (0.3595-0.3055)/2,
                 (0.3755-0.3209)/2, (0.2454-0.1970)/2]

    x = np.arange(len(models))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5))

    b1 = ax.bar(x - w/2, csam_f1, w, label="CSAM (3-tier)", color=PAL["csam"],
                alpha=0.9, yerr=ci_half, capsize=4, error_kw={"linewidth": 1.2})
    b2 = ax.bar(x + w/2, base_f1, w, label="Flat-RAG Baseline", color=PAL["baseline"],
                alpha=0.9, hatch="//", edgecolor="gray")

    ax.set_xticks(x); ax.set_xticklabels(models)
    ax.set_ylabel("Macro F1")
    ax.set_ylim(0.15, 0.42)
    ax.set_title("Figure 4: LoCoMo Long-Term Conversation Benchmark\n"
                 "(821 QA pairs across 5 conversations, seed=42)", fontsize=10)
    ax.legend(loc="upper right")

    # Annotate delta for first pair
    for i, (c, b) in enumerate(zip(csam_f1, base_f1)):
        delta = c - b
        ax.text(x[i], max(c, b) + ci_half[i] + 0.005,
                f"Δ={delta:+.4f}", ha="center", fontsize=7.5,
                color="#555555")

    # GPT-OSS annotation
    ax.annotate("Instruction-format\nsensitivity (120B)",
                xy=(3, 0.2269), xytext=(2.1, 0.28),
                arrowprops=dict(arrowstyle="->", color="red"),
                color="red", fontsize=8)

    # Parity note
    ax.text(0.02, 0.05,
            "CSAM achieves parity with flat RAG (Δ ≈ +0.0006,\n"
            "not statistically significant). Architecture imposes no accuracy cost.",
            transform=ax.transAxes, fontsize=8.5, va="bottom",
            bbox=dict(boxstyle="round,pad=0.4", fc="#f0f4ff", ec="#aaaacc"))

    fig.tight_layout()
    save(fig, "fig4_locomo")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Multi-Agent Scaling
# ══════════════════════════════════════════════════════════════════════════════
def fig_scaling():
    npcs      = [1,    5,    10,   25,   50,   100  ]
    memories  = [100,  500,  1000, 2500, 5000, 10000]
    avg_lat   = [12.5, 13.0, 12.9, 12.9, 12.8, 12.4 ]
    p99_lat   = [13.5, 15.5, 15.7, 15.4, 20.0, 16.5 ]
    mem_mb    = [0.17, 0.83, 1.66, 4.14, 8.28, 16.56]
    lat_std   = [0.30, 0.40, 0.35, 0.30, 1.20, 0.50 ]   # approx std across 4 runs

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # ── Panel A: Latency vs NPC count ────────────────────────────────────────
    ax1.semilogx(npcs, avg_lat, "o-", color=PAL["csam"], linewidth=2,
                 markersize=7, label="Avg latency", zorder=5)
    ax1.fill_between(npcs,
                     [a - s for a, s in zip(avg_lat, lat_std)],
                     [a + s for a, s in zip(avg_lat, lat_std)],
                     alpha=0.15, color=PAL["csam"])
    ax1.semilogx(npcs, p99_lat, "s--", color="#ff7f0e", linewidth=1.5,
                 markersize=6, label="p99 latency")
    ax1.axhline(12.8, color="gray", linestyle=":", linewidth=1.0, alpha=0.7)
    ax1.set_xlabel("Number of Concurrent Agents (log scale)")
    ax1.set_ylabel("Query Latency (ms)")
    ax1.set_ylim(8, 28)
    ax1.set_xticks(npcs); ax1.set_xticklabels(npcs)
    ax1.set_title("(a) Query Latency vs. Agent Count")
    ax1.legend(loc="upper right")
    ax1.text(0.05, 0.85, "O(log N) — HNSW\nNearly flat scaling",
             transform=ax1.transAxes, fontsize=9,
             bbox=dict(boxstyle="round", fc="white", ec=PAL["csam"], alpha=0.8))
    # Annotate points
    for n, lat in zip(npcs, avg_lat):
        ax1.annotate(f"{lat:.1f}", (n, lat), textcoords="offset points",
                     xytext=(6, 4), fontsize=7.5)

    # ── Panel B: Memory footprint ────────────────────────────────────────────
    ax2.plot(npcs, mem_mb, "o-", color=PAL["ca_ours"], linewidth=2, markersize=7)
    # Linear fit
    coeffs = np.polyfit(npcs, mem_mb, 1)
    x_fit = np.linspace(1, 100, 200)
    ax2.plot(x_fit, np.polyval(coeffs, x_fit), "--", color="gray",
             linewidth=1.2, label=f"Linear fit (r²=1.00)\n0.165 MB / 100 memories")
    ax2.set_xlabel("Number of Concurrent Agents")
    ax2.set_ylabel("Memory Footprint (MB)")
    ax2.set_title("(b) Memory Footprint & Scaling")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.text(0.55, 0.08, "Recall = 100%\nat all scales",
             transform=ax2.transAxes, fontsize=9, color=PAL["ca_ours"],
             bbox=dict(boxstyle="round", fc="white", ec=PAL["ca_ours"], alpha=0.8))
    for n, mb in zip(npcs, mem_mb):
        ax2.annotate(f"{mb:.2f} MB", (n, mb), textcoords="offset points",
                     xytext=(5, 3), fontsize=7.5)

    fig.suptitle("Figure 5: Multi-Agent Scaling Study (100 memories/agent, no-LLM mode)",
                 fontsize=10, y=1.01)
    fig.tight_layout()
    save(fig, "fig5_scaling")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — MuSiQue Results
# ══════════════════════════════════════════════════════════════════════════════
def fig_musique():
    models     = ["8B\n(mean)", "Scout\n17B", "70B", "GPT-OSS\n120B"]
    f1_mean    = [0.388, 0.426, 0.420, 0.220]
    f1_err     = [0.031, 0.0,   0.0,   0.0  ]
    model_cols = [PAL["g8b"], PAL["scout"], PAL["g70b"], PAL["gpt"]]

    hop_types  = ["2-hop", "3-hop1", "3-hop2", "4-hop1", "4-hop3"]
    hop_8b     = [0.342,   0.744,    0.410,    0.484,    0.000   ]
    hop_scout  = [0.433,   0.546,    0.427,    0.284,    0.202   ]
    hop_70b    = [0.466,   0.499,    0.283,    0.311,    0.000   ]
    hop_gpt    = [0.304,   0.220,    0.000,    0.000,    0.000   ]

    # L3 ablation
    with_l3    = 0.3965
    without_l3 = 0.3960

    fig = plt.figure(figsize=(13, 4.5))
    gs = GridSpec(1, 3, figure=fig, wspace=0.38)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    # ── Panel A: Overall F1 ─────────────────────────────────────────────────
    x = np.arange(len(models))
    bars = ax1.bar(x, f1_mean, color=model_cols, width=0.6,
                   yerr=f1_err, capsize=4, error_kw={"linewidth": 1.2})
    ax1.set_xticks(x); ax1.set_xticklabels(models)
    ax1.set_ylabel("Mean F1"); ax1.set_ylim(0.0, 0.55)
    ax1.set_title("(a) Overall F1 (100 Questions)")
    for bar, val in zip(bars, f1_mean):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.005,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    ax1.annotate("Scout > 70B\n(retrieval-bound)", xy=(1, 0.426),
                 xytext=(1.6, 0.46),
                 arrowprops=dict(arrowstyle="->", color=PAL["scout"]),
                 color=PAL["scout"], fontsize=8)

    # ── Panel B: F1 by hop count ─────────────────────────────────────────────
    hop_x = np.arange(len(hop_types))
    w = 0.2
    ax2.bar(hop_x - 1.5*w, hop_8b,    w, label="8B",      color=PAL["g8b"])
    ax2.bar(hop_x - 0.5*w, hop_scout, w, label="Scout",   color=PAL["scout"])
    ax2.bar(hop_x + 0.5*w, hop_70b,   w, label="70B",     color=PAL["g70b"])
    ax2.bar(hop_x + 1.5*w, hop_gpt,   w, label="GPT-120B",color=PAL["gpt"])
    ax2.set_xticks(hop_x); ax2.set_xticklabels(hop_types, fontsize=8)
    ax2.set_ylabel("F1"); ax2.set_ylim(0.0, 0.85)
    ax2.set_title("(b) F1 by Hop Count")
    ax2.legend(fontsize=7.5, ncol=2)
    ax2.annotate("4-hop ceiling\nfor all models", xy=(4, 0.05),
                 xytext=(2.8, 0.35),
                 arrowprops=dict(arrowstyle="->", color="red"),
                 color="red", fontsize=8)

    # ── Panel C: L3 ablation ─────────────────────────────────────────────────
    labels_l3 = ["With L3", "Without L3"]
    vals_l3   = [with_l3, without_l3]
    bars3 = ax3.bar([0, 1], vals_l3,
                    color=[PAL["csam"], PAL["baseline"]],
                    width=0.5, edgecolor="gray")
    ax3.set_xticks([0, 1]); ax3.set_xticklabels(labels_l3)
    ax3.set_ylabel("F1 (8B, seed=42)")
    ax3.set_ylim(0.38, 0.42)
    ax3.set_title("(c) L3 Knowledge Graph Ablation\n(MuSiQue, 8B, seed=42)")
    for bar, val in zip(bars3, vals_l3):
        ax3.text(bar.get_x() + bar.get_width() / 2, val + 0.0002,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    ax3.annotate(f"Δ = +{with_l3 - without_l3:.4f}\n(< noise floor)",
                 xy=(0.5, (with_l3 + without_l3) / 2),
                 fontsize=9, ha="center", va="center",
                 bbox=dict(boxstyle="round", fc="#fff8e1", ec="#f0c040"))
    ax3.text(0.5, 0.05,
             "L3 value: consolidation\ntracking, not retrieval",
             transform=ax3.transAxes, fontsize=8, ha="center",
             bbox=dict(boxstyle="round", fc="#e8f5e9", ec="#66bb6a"))

    fig.suptitle("Figure 6: MuSiQue Multi-Hop QA Results (100 questions, seed=42)",
                 fontsize=10, y=1.01)
    fig.tight_layout()
    save(fig, "fig6_musique")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 7 — Memory Compression Visual
# ══════════════════════════════════════════════════════════════════════════════
def fig_memory_compression():
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")

    # ── Draw block stacks ─────────────────────────────────────────────────────
    def draw_stack(ax, x_center, n_blocks, color, label_top, label_bot, alpha=1.0):
        block_h = 0.03
        block_w = 0.12
        gap = 0.005
        start_y = 0.1
        shown = min(n_blocks, 20)
        for i in range(shown):
            rect = mpatches.FancyBboxPatch(
                (x_center - block_w/2, start_y + i*(block_h + gap)),
                block_w, block_h,
                boxstyle="round,pad=0.002",
                facecolor=color, edgecolor="white", alpha=alpha
            )
            ax.add_patch(rect)
        if n_blocks > shown:
            ax.text(x_center, start_y + shown*(block_h+gap) + 0.02,
                    f"... ({n_blocks} total)", ha="center", fontsize=8, color="gray")
        top_y = start_y + shown*(block_h+gap) + (0.08 if n_blocks > shown else 0.02)
        ax.text(x_center, top_y + 0.05, label_top, ha="center", fontsize=10,
                fontweight="bold")
        ax.text(x_center, start_y - 0.06, label_bot, ha="center", fontsize=9,
                color="gray")

    draw_stack(ax, 0.22, 20, "#cccccc", "No-Forgetting\n500 memories", "868 KB", alpha=0.7)
    draw_stack(ax, 0.72, 8,  "#d62728", "CA-Ours\n76 memories",       "132 KB", alpha=0.9)

    # Arrow
    ax.annotate("", xy=(0.60, 0.50), xytext=(0.40, 0.50),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", lw=2.5, color="#333333"))
    ax.text(0.50, 0.56, "84.8% reduction\n6.6× compression",
            transform=ax.transAxes, ha="center", fontsize=10, fontweight="bold",
            color="#333333")

    # F1 comparison
    ax.text(0.22, 0.04, "Mean F1 = 0.4847", transform=ax.transAxes,
            ha="center", fontsize=9, color="#555555")
    ax.text(0.72, 0.04, "Mean F1 = 0.5095 ↑", transform=ax.transAxes,
            ha="center", fontsize=9, color=PAL["ca_ours"], fontweight="bold")
    ax.text(0.50, -0.04, "CA-Ours achieves higher F1 with 84.8% fewer memories",
            transform=ax.transAxes, ha="center", fontsize=9,
            style="italic", color="#333333")

    ax.set_title("Figure 7: Memory Compression — No-Forgetting vs. CA-Ours",
                 fontsize=10, pad=15)
    ax.set_xlim(0, 1); ax.set_ylim(-0.1, 1.0)
    fig.tight_layout()
    save(fig, "fig7_memory_compression")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 8 — Question Scaling Stability
# ══════════════════════════════════════════════════════════════════════════════
def fig_question_scaling():
    q_hot  = [50,    100,   200,   500  ]
    f1_hot = [0.704, 0.681, 0.698, 0.681]
    q_mus  = [50,    100,   200   ]
    f1_mus = [0.461, 0.396, 0.396 ]

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(q_hot, f1_hot, "o-", color=PAL["csam"], linewidth=2,
            markersize=8, label="HotPotQA")
    ax.plot(q_mus, f1_mus, "s--", color=PAL["ca_ours"], linewidth=2,
            markersize=8, label="MuSiQue")
    for q, f in zip(q_hot, f1_hot):
        ax.annotate(f"{f:.3f}", (q, f), textcoords="offset points",
                    xytext=(5, 5), fontsize=8)
    for q, f in zip(q_mus, f1_mus):
        ax.annotate(f"{f:.3f}", (q, f), textcoords="offset points",
                    xytext=(5, -12), fontsize=8)
    ax.set_xlabel("Number of Questions")
    ax.set_ylabel("F1")
    ax.set_title("Figure 8: F1 Stability vs. Question Count\n(Llama 3.1 8B, seed=42)")
    ax.legend(loc="center right")
    ax.text(0.60, 0.20,
            "HotPotQA: stable 0.681–0.704\nacross 50–500 questions\n"
            "MuSiQue: converges at 100Q",
            transform=ax.transAxes, fontsize=8.5,
            bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.8))
    fig.tight_layout()
    save(fig, "fig8_question_scaling")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating CSAM paper charts...")
    fig_ablation()
    fig_hotpotqa()
    fig_locomo()
    fig_scaling()
    fig_musique()
    fig_memory_compression()
    fig_question_scaling()
    print(f"\nAll charts saved to: {OUT}")
