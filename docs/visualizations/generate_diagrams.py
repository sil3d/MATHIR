"""
MATHIR V8.4.1 — Visual Diagrams Suite
====================================

Generates 8 high-quality PNG diagrams for the master's defense.

Outputs (in this directory):
  01_architecture_main.png        — 5-tier MATHIR architecture
  02_4_memory_tiers.png          — Memory tier deep-dive
  03_retrieval_comparison.png    — Quality comparison (V7.1)
  04_latency_quality_tradeoff.png — Speed vs Quality Pareto
  05_multi_agent_stress.png      — Concurrent stores stress test
  06_multimodal_fusion.png       — Multi-modal memory (text/img/audio/video)
  07_theorem_network.png         — 6 theorems dependency graph
  08_version_timeline.png        — V1 → V8.4.1 evolution

Run:
    python visualizations/generate_diagrams.py

Author: MATHIR Research Team
Date:   2026-06-02
"""

from __future__ import annotations

import os
import sys
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Polygon
from matplotlib.lines import Line2D
import matplotlib.font_manager as fm
import networkx as nx

# =============================================================================
# Configuration
# =============================================================================

OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Brand palette (per the master's brand guidelines)
COLORS = {
    "primary":    "#1f4e79",  # Dark blue  (MATHIR core)
    "secondary":  "#d97706",  # Orange     (memory / innovation)
    "tertiary":   "#059669",  # Green      (output / success)
    "danger":     "#dc2626",  # Red        (immune / anomaly)
    "muted":      "#64748b",  # Slate      (working / neutral)
    "light":      "#f1f5f9",  # Off-white  (background fills)
    "border":     "#0f172a",  # Near-black (text / borders)
    "accent":     "#7c3aed",  # Purple     (FAISS / C)
    "gold":       "#ca8a04",  # Gold       (D / winner)
    "teal":       "#0891b2",  # Teal       (multi-modal)
}

# Tier-specific colors
TIER_COLORS = {
    "working":      "#3b82f6",  # Blue
    "episodic":     "#059669",  # Green
    "semantic":     "#d97706",  # Orange
    "procedural":   "#8b5cf6",  # Purple
    "immunological":"#dc2626",  # Red
}

# Use a clean, professional style
plt.rcParams.update({
    "figure.dpi":          150,
    "savefig.dpi":         150,
    "savefig.bbox":        "tight",
    "savefig.pad_inches":  0.3,
    "font.family":         "DejaVu Sans",
    "font.size":           11,
    "axes.titlesize":      16,
    "axes.titleweight":    "bold",
    "axes.labelsize":      12,
    "axes.labelweight":    "bold",
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "axes.edgecolor":      "#334155",
    "axes.linewidth":      1.2,
    "xtick.color":         "#334155",
    "ytick.color":         "#334155",
    "legend.frameon":      False,
    "legend.fontsize":     10,
    "figure.facecolor":    "white",
    "axes.facecolor":      "white",
})

FIG_W, FIG_H = 12, 8


# =============================================================================
# Utilities
# =============================================================================


def add_watermark(ax: plt.Axes, text: str = "MATHIR V8.4.1") -> None:
    """Add a subtle watermark in the bottom-right corner."""
    ax.text(
        0.985, 0.012, text,
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=8, color="#94a3b8",
        style="italic", alpha=0.85,
    )


def add_title_band(fig: plt.Figure, title: str, subtitle: str = "") -> None:
    """Add a top title band to the figure."""
    fig.text(0.5, 0.965, title, ha="center", va="top",
             fontsize=20, fontweight="bold", color=COLORS["primary"])
    if subtitle:
        fig.text(0.5, 0.928, subtitle, ha="center", va="top",
                 fontsize=12, color=COLORS["muted"], style="italic")


def add_footer(fig: plt.Figure, left: str = "MATHIR Research Team",
               right: str = "Generated June 2026") -> None:
    """Add a thin footer band."""
    fig.text(0.5, 0.012,
             f"{left}  •  {right}  •  Confidential — Master's Defense",
             ha="center", va="bottom",
             fontsize=8, color="#94a3b8", style="italic")


def rounded_box(ax, x, y, w, h, color, text="", text_color="white",
                fontsize=11, fontweight="bold", radius=0.018, alpha=1.0,
                edge_color=None, zorder=2):
    """Draw a rounded-corner box with centered text."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.005,rounding_size={radius}",
        linewidth=1.5,
        facecolor=color, edgecolor=edge_color or color,
        alpha=alpha, zorder=zorder,
    )
    ax.add_patch(box)
    if text:
        ax.text(x + w / 2, y + h / 2, text,
                ha="center", va="center",
                color=text_color, fontsize=fontsize,
                fontweight=fontweight, zorder=zorder + 1)
    return box


def arrow(ax, x1, y1, x2, y2, color="#475569", lw=1.6,
          style="-|>", mutation=18, connectionstyle="arc3,rad=0.0",
          zorder=1.5, alpha=0.95):
    """Draw a clean arrow between two points."""
    return FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, mutation_scale=mutation,
        color=color, lw=lw,
        connectionstyle=connectionstyle,
        zorder=zorder, alpha=alpha,
    )


# =============================================================================
# Diagram 1 — Architecture (Main)
# =============================================================================


def diagram_1_architecture_main() -> Path:
    """The 5-tier MATHIR architecture — high-level system diagram."""
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("equal")
    ax.axis("off")

    add_title_band(fig, "MATHIR V8.4.1 — Hierarchical Memory Architecture",
                   "Adaptive 5-tier memory for any LLM")

    # ---------- Column bands (3 zones) ----------
    # INPUT zone
    ax.add_patch(Rectangle((3, 12), 17, 70, facecolor="#eff6ff",
                           edgecolor=COLORS["primary"], linewidth=1.2,
                           linestyle="--", alpha=0.45, zorder=0))
    ax.text(11.5, 84, "INPUT", ha="center", va="center",
            fontsize=10, fontweight="bold", color=COLORS["primary"])

    # MATHIR zone
    ax.add_patch(Rectangle((26, 12), 48, 70, facecolor="#fff7ed",
                           edgecolor=COLORS["secondary"], linewidth=1.2,
                           linestyle="--", alpha=0.55, zorder=0))
    ax.text(50, 84, "MATHIR  (5-tier memory + KL router)",
            ha="center", va="center",
            fontsize=10, fontweight="bold", color=COLORS["secondary"])

    # OUTPUT zone
    ax.add_patch(Rectangle((80, 12), 17, 70, facecolor="#ecfdf5",
                           edgecolor=COLORS["tertiary"], linewidth=1.2,
                           linestyle="--", alpha=0.45, zorder=0))
    ax.text(88.5, 84, "OUTPUT", ha="center", va="center",
            fontsize=10, fontweight="bold", color=COLORS["tertiary"])

    # ---------- INPUT: multi-modal sources ----------
    sources = [
        ("📝 Text",   6,  68, COLORS["primary"]),
        ("🖼️ Image",  6,  55, COLORS["primary"]),
        ("🎵 Audio",  6,  42, COLORS["primary"]),
        ("🎬 Video",  6,  29, COLORS["primary"]),
    ]
    for label, x, y, c in sources:
        rounded_box(ax, x, y, 11, 7, c, label, fontsize=12)
        ax.add_patch(arrow(ax,x + 11, y + 3.5, 26, y + 3.5,
                           color=COLORS["primary"], lw=1.4, mutation=14))

    # Encoder block
    rounded_box(ax, 26, 35, 9, 32, "#e0e7ff",
                "Per-modality\nencoders\n(CLIP, Whisper,\nViT, BERT)",
                text_color="#1e3a8a", fontsize=9.5, fontweight="bold",
                edge_color="#6366f1")
    # Encoder -> tier memory
    for ty in (68, 55, 42, 29):
        ax.add_patch(arrow(ax,35, 51, 39, ty + 3.5,
                           color="#6366f1", lw=1.2, mutation=12,
                           alpha=0.7))

    # ---------- MATHIR: 4 memory tiers + Router ----------
    tiers = [
        ("⚡ WORKING\n64 slots\nlast N steps\n🔵 every step",       39, 60, TIER_COLORS["working"]),
        ("📚 EPISODIC\n1000 slots\npast experiences\n🟢 on event",   56, 60, TIER_COLORS["episodic"]),
        ("🧠 SEMANTIC\n256 prototypes\nlearned concepts\n🟠 /100 steps", 39, 42, TIER_COLORS["semantic"]),
        ("🛡️ IMMUNOLOGICAL\n100 patterns\nanomaly detection\n🔴 on event", 56, 42, TIER_COLORS["immunological"]),
    ]
    for label, x, y, c in tiers:
        rounded_box(ax, x, y, 16, 14, c, label,
                    fontsize=9, fontweight="bold", radius=0.04)

    # Router
    rounded_box(ax, 47, 17, 17, 11, COLORS["secondary"],
                "🎯 KL-CONSTRAINED\nROUTER\nprevents collapse",
                fontsize=10, text_color="white", radius=0.05)

    # Tier -> Router arrows
    for tx, ty in [(47, 60), (64, 60), (47, 42), (64, 42)]:
        ax.add_patch(arrow(ax,tx + 8, ty, 55.5, 28,
                           color=COLORS["secondary"], lw=1.3, mutation=12,
                           alpha=0.85))

    # Router -> Output
    ax.add_patch(arrow(ax,64, 22.5, 80, 22.5,
                       color=COLORS["tertiary"], lw=1.8, mutation=18))

    # ---------- OUTPUT ----------
    rounded_box(ax, 80, 48, 17, 9, COLORS["tertiary"],
                "🤖 Any LLM\nGPT-4 · Claude · Qwen",
                fontsize=10)
    rounded_box(ax, 80, 30, 17, 9, "#a7f3d0",
                "Enhanced\nembedding",
                text_color="#065f46", fontsize=10, edge_color=COLORS["tertiary"])
    rounded_box(ax, 80, 14, 17, 9, COLORS["tertiary"],
                "Action /\nResponse",
                fontsize=10)

    # Output internal arrows
    ax.add_patch(arrow(ax,88.5, 48, 88.5, 39,
                       color=COLORS["tertiary"], lw=1.4, mutation=12))
    ax.add_patch(arrow(ax,88.5, 30, 88.5, 23,
                       color=COLORS["tertiary"], lw=1.4, mutation=12))

    # Caption
    ax.text(50, 6,
            "Modality-agnostic: any data → embedding → 5-tier memory → enhanced context",
            ha="center", va="center", fontsize=10,
            color=COLORS["muted"], style="italic")
    ax.text(50, 3, "MATHIR is the hippocampus of AI",
            ha="center", va="center", fontsize=9.5,
            color=COLORS["secondary"], fontweight="bold")

    add_watermark(ax)
    add_footer(fig)
    out = OUTPUT_DIR / "01_architecture_main.png"
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    return out


# =============================================================================
# Diagram 2 — 4 Memory Tiers
# =============================================================================


def diagram_2_4_memory_tiers() -> Path:
    """4 memory tiers shown as stacked panels with slot visualizations."""
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("equal")
    ax.axis("off")

    add_title_band(fig, "5-Tier Memory System",
                   "Working · Episodic · Semantic · Procedural · Immunological")

    tiers = [
        {
            "label": "⚡  WORKING MEMORY",
            "sub":   "last N steps · circular buffer",
            "color": TIER_COLORS["working"],
            "slots": 64, "used": 64,
            "rows": 4, "cols": 16,
            "function": "Holds the most recent context. Multi-head attention reads it.",
            "update":   "Every step (O(1) push/pop)",
            "capacity": "64 slots (configurable)",
            "algo":     "Circular buffer + scaled dot-product attention",
        },
        {
            "label": "📚  EPISODIC MEMORY",
            "sub":   "past experiences · key-value",
            "color": TIER_COLORS["episodic"],
            "slots": 1000, "used": 700,
            "rows": 5, "cols": 20,
            "function": "Long-term storage of unique events with cosine retrieval.",
            "update":   "On new event (insert + LRU evict)",
            "capacity": "1000 slots (~1.5 MB raw, 117 KB compressed)",
            "algo":     "Cosine k-NN · Hybrid BM25+Dense+CE (V7.1 Approach D)",
        },
        {
            "label": "🧠  SEMANTIC MEMORY",
            "sub":   "learned concepts · online k-means",
            "color": TIER_COLORS["semantic"],
            "slots": 256, "used": 192,
            "rows": 4, "cols": 16,
            "function": "Cluster centers that represent generalized knowledge.",
            "update":   "Every 100 steps (online k-means)",
            "capacity": "256 prototypes (~80 KB)",
            "algo":     "Down/up projection · hyperbolic embeddings (V7)",
        },
        {
            "label": "🛡️  IMMUNOLOGICAL MEMORY",
            "sub":   "anomaly patterns · Mahalanobis",
            "color": TIER_COLORS["immunological"],
            "slots": 100, "used": 65,
            "rows": 2, "cols": 16,
            "function": "Recognises out-of-distribution patterns and threats.",
            "update":   "On flagged event (decay/refresh)",
            "capacity": "100 patterns (~40 KB)",
            "algo":     "Mahalanobis distance — Neyman-Pearson optimal (Theorem 4)",
        },
    ]

    # Each tier gets ~22 vertical units
    y_top = 88
    tier_h = 19
    gap = 1.5

    for i, t in enumerate(tiers):
        y0 = y_top - (i + 1) * tier_h - i * gap
        y1 = y0 + tier_h

        # Panel background
        ax.add_patch(FancyBboxPatch(
            (3, y0), 94, tier_h,
            boxstyle="round,pad=0.1,rounding_size=0.4",
            facecolor="#fafafa", edgecolor=t["color"],
            linewidth=1.5, alpha=0.7, zorder=1,
        ))

        # Left header band
        ax.add_patch(FancyBboxPatch(
            (3, y0), 28, tier_h,
            boxstyle="round,pad=0.1,rounding_size=0.4",
            facecolor=t["color"], edgecolor=t["color"],
            linewidth=0, alpha=0.95, zorder=2,
        ))
        ax.text(17, y0 + tier_h * 0.65, t["label"],
                ha="center", va="center", fontsize=14,
                fontweight="bold", color="white", zorder=3)
        ax.text(17, y0 + tier_h * 0.32, t["sub"],
                ha="center", va="center", fontsize=10,
                color="white", style="italic", alpha=0.9, zorder=3)
        ax.text(17, y0 + 1.5, t["algo"],
                ha="center", va="center", fontsize=8,
                color="white", alpha=0.85, zorder=3, wrap=True)

        # Right: metadata columns
        meta_x = 35
        ax.text(meta_x, y0 + tier_h * 0.78,
                f"📦  Capacity: {t['capacity']}",
                fontsize=10, color="#0f172a", fontweight="bold")
        ax.text(meta_x, y0 + tier_h * 0.55,
                f"⚙️  Function: {t['function']}",
                fontsize=9.5, color="#1e293b", wrap=True)
        ax.text(meta_x, y0 + tier_h * 0.32,
                f"🔄  Update: {t['update']}",
                fontsize=9.5, color="#1e293b")

        # Slot visualization (right side)
        slot_x0 = 70
        slot_y0 = y0 + 2.5
        avail_w = 25
        avail_h = tier_h - 5
        cell_w = avail_w / t["cols"]
        cell_h = avail_h / t["rows"]
        filled = int(round(t["used"] / t["slots"] * (t["rows"] * t["cols"])))
        idx = 0
        for r in range(t["rows"]):
            for c in range(t["cols"]):
                cx = slot_x0 + c * cell_w
                cy = slot_y0 + (t["rows"] - 1 - r) * cell_h
                is_filled = idx < filled
                ax.add_patch(Circle(
                    (cx + cell_w / 2, cy + cell_h / 2),
                    radius=min(cell_w, cell_h) * 0.32,
                    facecolor=t["color"] if is_filled else "#e2e8f0",
                    edgecolor=t["color"] if is_filled else "#cbd5e1",
                    linewidth=0.6, alpha=0.95 if is_filled else 0.6, zorder=3,
                ))
                idx += 1
        # Slot panel label
        ax.text(slot_x0 + avail_w / 2, y0 + tier_h * 0.92,
                f"{t['used']} / {t['slots']} slots",
                ha="center", va="center", fontsize=8.5,
                color=t["color"], fontweight="bold")

    # Vertical legend
    ax.text(50, 4, "Filled = occupied slots · Empty = available capacity",
            ha="center", va="center", fontsize=9.5,
            color=COLORS["muted"], style="italic")

    add_watermark(ax)
    add_footer(fig)
    out = OUTPUT_DIR / "02_4_memory_tiers.png"
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    return out


# =============================================================================
# Diagram 3 — Retrieval Quality Comparison
# =============================================================================


def diagram_3_retrieval_comparison() -> Path:
    """Horizontal bar chart comparing 5 retrieval systems."""
    # Data from V7.1 master benchmark (200 chunks, 50 domain queries)
    systems = [
        "V7 default (64-dim projection)",
        "B — Multi-Encoder Ensemble",
        "FAISS VectorDB (raw 384-dim)",
        "A — Raw Embedding Bypass",
        "D — Hybrid BM25 + Dense + CE  ⭐",
    ]
    quality = [19.7, 29.1, 31.6, 31.6, 45.7]
    bar_colors = [
        "#94a3b8",  # slate (legacy)
        "#a78bfa",  # lavender
        "#3b82f6",  # blue
        "#10b981",  # emerald
        COLORS["gold"],  # gold — winner
    ]

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    add_title_band(fig, "Retrieval Quality Comparison  (V7.1)",
                   "Master benchmark: 5 systems × 50 queries × 200 chunks  (White, Fluid Mechanics)")

    # Sort for visual impact — but keep label order (D on top)
    y_pos = np.arange(len(systems))[::-1]  # so D appears on top
    bars = ax.barh(y_pos, quality, color=bar_colors,
                   edgecolor="white", linewidth=1.5, height=0.7, zorder=3)

    # Value labels
    for bar, q in zip(bars, quality):
        ax.text(bar.get_width() + 0.6, bar.get_y() + bar.get_height() / 2,
                f"{q:.1f}%", va="center", ha="left",
                fontsize=12, fontweight="bold",
                color="#0f172a", zorder=4)

    # Highlight D row
    ax.add_patch(Rectangle((0, 4.55), 50, 0.9,
                           facecolor=COLORS["gold"], alpha=0.08,
                           edgecolor="none", zorder=1))

    ax.set_yticks(y_pos)
    ax.set_yticklabels(systems, fontsize=11)
    ax.set_xlim(0, 55)
    ax.set_xlabel("Retrieval Quality (% overlap with ground truth)", fontsize=12)
    ax.set_xticks(np.arange(0, 51, 10))
    ax.set_xticklabels([f"{x}%" for x in range(0, 51, 10)])
    ax.grid(axis="x", linestyle="--", alpha=0.35, zorder=0)
    ax.set_axisbelow(True)

    # Annotations
    ax.annotate("Quality winner\n(+14.1 pp vs FAISS)",
                xy=(45.7, 4.7), xytext=(38, 2.6),
                fontsize=10, color="#854d0e", fontweight="bold",
                ha="center",
                arrowprops=dict(arrowstyle="->", color="#854d0e",
                                lw=1.4, connectionstyle="arc3,rad=0.25"))

    ax.annotate("Production default\n(best speed/quality balance)",
                xy=(31.6, 3.0), xytext=(20, 0.7),
                fontsize=9.5, color="#065f46", fontweight="bold",
                ha="center",
                arrowprops=dict(arrowstyle="->", color="#065f46",
                                lw=1.2, connectionstyle="arc3,rad=-0.2"))

    # Legend
    legend_handles = [
        mpatches.Patch(facecolor=COLORS["gold"], label="D — Quality winner"),
        mpatches.Patch(facecolor="#10b981",    label="A — Online default"),
        mpatches.Patch(facecolor="#3b82f6",    label="FAISS — Edge case"),
        mpatches.Patch(facecolor="#a78bfa",    label="B — Diminishing returns"),
        mpatches.Patch(facecolor="#94a3b8",    label="V7 default — legacy"),
    ]
    ax.legend(handles=legend_handles, loc="lower right",
              ncol=1, frameon=True, framealpha=0.95,
              edgecolor="#cbd5e1", fontsize=9.5,
              bbox_to_anchor=(1.0, -0.18))

    add_watermark(ax)
    add_footer(fig, left="Quality = top-5 overlap %", right="Source: compare_all_approaches_results.json")
    out = OUTPUT_DIR / "03_retrieval_comparison.png"
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    return out


# =============================================================================
# Diagram 4 — Latency / Quality Tradeoff (Pareto)
# =============================================================================


def diagram_4_latency_quality_tradeoff() -> Path:
    """Scatter plot: X=latency (log), Y=quality; with Pareto frontier."""
    # Data: (label, latency_ms, quality_pct, color, marker_size)
    points = [
        ("FAISS raw",       0.05, 31.6, "#3b82f6",  120),
        ("V7 default",      0.66, 19.7, "#94a3b8",  100),
        ("A — Raw Emb.",    1.54, 31.6, "#10b981",  140),
        ("B — Multi-Enc.",  2.20, 29.1, "#a78bfa",  110),
        ("C — FAISS-backed",8.88, 31.6, "#7c3aed",  130),
        ("D — Hybrid BM25+CE", 494, 45.7, COLORS["gold"], 200),
    ]

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    add_title_band(fig, "Speed–Quality Tradeoff  (Pareto Frontier)",
                   "Lower-left = fast & poor ·  Upper-right = slow & rich ·  Sweet spot = upper-left")

    for label, lat, q, color, size in points:
        ax.scatter(lat, q, s=size, c=color,
                   edgecolors="white", linewidths=2, zorder=4, alpha=0.9)
        # Label placement
        offsets = {
            "FAISS raw":          (1.4, 0.2),
            "V7 default":         (0.9, -2.3),
            "A — Raw Emb.":       (1.4, 0.4),
            "B — Multi-Enc.":     (1.5, 0.3),
            "C — FAISS-backed":   (1.6, 0.5),
            "D — Hybrid BM25+CE": (0.55, 0.0),  # log-scale shift
        }
        dx, dy = offsets.get(label, (1.3, 0.3))
        is_winner = label.startswith("D")
        ax.annotate(
            label,
            xy=(lat, q), xytext=(lat * dx, q + dy),
            fontsize=10.5 if not is_winner else 12,
            fontweight="bold" if is_winner else "normal",
            color="#854d0e" if is_winner else "#0f172a",
        )

    # Pareto frontier (upper-left envelope: D is best quality, FAISS is best speed)
    pareto_lat = [0.05, 1.54, 494]
    pareto_q   = [31.6, 31.6, 45.7]
    ax.plot(pareto_lat, pareto_q, "--", color=COLORS["gold"],
            linewidth=1.6, alpha=0.55, zorder=2,
            label="Pareto frontier")

    # Region shading
    ax.axvspan(0.01, 1, ymin=0.0, ymax=0.66,
               facecolor="#10b981", alpha=0.06, zorder=0)
    ax.axvspan(1, 10, ymin=0.0, ymax=0.66,
               facecolor="#a78bfa", alpha=0.06, zorder=0)
    ax.axvspan(10, 1000, ymin=0.0, ymax=1.0,
               facecolor=COLORS["gold"], alpha=0.06, zorder=0)

    # Region labels
    ax.text(0.25, 6, "ULTRA-FAST\n(edge, <1ms)",
            ha="center", va="center", fontsize=9,
            color="#065f46", fontweight="bold", alpha=0.85)
    ax.text(3, 6, "INTERACTIVE\n(<10ms)",
            ha="center", va="center", fontsize=9,
            color="#5b21b6", fontweight="bold", alpha=0.85)
    ax.text(80, 6, "BATCH / RAG\n(high quality)",
            ha="center", va="center", fontsize=9,
            color="#854d0e", fontweight="bold", alpha=0.85)

    ax.set_xscale("log")
    ax.set_xlim(0.03, 1500)
    ax.set_ylim(15, 52)
    ax.set_xlabel("Latency (ms, log scale)  ←  faster", fontsize=12)
    ax.set_ylabel("Retrieval Quality (%)", fontsize=12)
    ax.set_yticks(np.arange(15, 51, 5))
    ax.set_yticklabels([f"{y}%" for y in range(15, 51, 5)])
    ax.grid(True, which="both", linestyle="--", alpha=0.30, zorder=0)
    ax.set_axisbelow(True)

    # Vertical reference lines
    for v, lab in [(1, "1 ms"), (10, "10 ms"), (100, "100 ms")]:
        ax.axvline(v, color="#cbd5e1", linewidth=0.8, linestyle=":", zorder=0)
        ax.text(v, 16, lab, ha="center", va="bottom",
                fontsize=8, color="#94a3b8")

    # Recommendation box
    rec_text = ("Recommendations:\n"
                "• Default → A (Raw Embedding) — 31.6% / 1.5 ms\n"
                "• Batch / RAG → D (Hybrid BM25+CE) — 45.7% / 494 ms\n"
                "• Edge / ≤50K → FAISS raw — 31.6% / 0.05 ms")
    ax.text(0.025, 0.97, rec_text, transform=ax.transAxes,
            ha="left", va="top", fontsize=9.5,
            color="#0f172a", fontweight="normal",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#fffbeb",
                      edgecolor=COLORS["gold"], linewidth=1.2, alpha=0.95),
            zorder=5)

    ax.legend(loc="lower left", frameon=True, framealpha=0.9,
              edgecolor="#cbd5e1", fontsize=10)
    add_watermark(ax)
    add_footer(fig)
    out = OUTPUT_DIR / "04_latency_quality_tradeoff.png"
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    return out


# =============================================================================
# Diagram 5 — Multi-Agent Concurrent Stores Stress
# =============================================================================


def diagram_5_multi_agent_stress() -> Path:
    """Bar chart of concurrent store stress test: 1/5/10/20 agents."""
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    add_title_band(fig, "Multi-Agent Concurrent Stores  (20 agents)",
                   "Stress test: parallel store() calls · thread-safe write paths")

    # 4 sub-scenarios with related metrics (success / throughput / conflicts)
    scenarios = ["1 agent", "5 agents", "10 agents", "20 agents"]
    success_rate   = [100, 100, 100, 100]   # %
    qps            = [1.98, 9.7, 18.4, 33.6] # QPS throughput
    store_latency  = [504, 110, 62, 38]      # ms (median)
    write_conflicts= [0, 0, 0, 0]            # 0 conflicts

    # Twin-y axis layout
    x = np.arange(len(scenarios))
    width = 0.35

    # Success rate bars (left)
    bars1 = ax.bar(x - width / 2, success_rate, width,
                   color="#10b981", alpha=0.85,
                   edgecolor="#065f46", linewidth=1.2,
                   label="Success rate (%)", zorder=3)
    for b, v in zip(bars1, success_rate):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.2,
                f"{v:.0f}%", ha="center", va="bottom",
                fontsize=12, fontweight="bold", color="#065f46")

    # QPS bars (right twin)
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + width / 2, qps, width,
                    color=COLORS["primary"], alpha=0.85,
                    edgecolor="#1e3a8a", linewidth=1.2,
                    label="Throughput (QPS)", zorder=3)
    for b, v in zip(bars2, qps):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.6,
                 f"{v:.1f}", ha="center", va="bottom",
                 fontsize=11, fontweight="bold", color="#1e3a8a")

    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=12)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Success rate (%)", fontsize=12, color="#065f46")
    ax.tick_params(axis="y", labelcolor="#065f46")
    ax2.set_ylim(0, 42)
    ax2.set_ylabel("Throughput (queries / sec)", fontsize=12, color="#1e3a8a")
    ax2.tick_params(axis="y", labelcolor="#1e3a8a")
    ax.set_xlabel("Concurrent agent count", fontsize=12)

    ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
    ax.set_axisbelow(True)

    # Big check mark banner
    ax.text(1.5, 110, "✓  100% success at every scale  —  zero write conflicts",
            ha="center", va="center", fontsize=12.5,
            color="#065f46", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#d1fae5",
                      edgecolor="#10b981", linewidth=1.5))

    # Latency mini-table
    lat_data = list(zip(scenarios, store_latency, write_conflicts))
    table_text = "Median store latency (ms):  " + "  ".join(
        f"{n}={l}ms" for n, l, _ in lat_data
    )
    ax.text(1.5, 102, table_text, ha="center", va="center",
            fontsize=10, color="#0f172a", style="italic")

    # Combined legend
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left",
              frameon=True, framealpha=0.95, edgecolor="#cbd5e1",
              fontsize=10)

    # 20x scaling callout
    ax.annotate("17× throughput\nscaling",
                xy=(3, 33.6), xytext=(2.4, 22),
                fontsize=10, color="#1e3a8a", fontweight="bold",
                ha="center",
                arrowprops=dict(arrowstyle="->", color="#1e3a8a",
                                lw=1.4, connectionstyle="arc3,rad=0.2"))

    add_watermark(ax)
    add_footer(fig, left="Concurrency: 20 parallel threads",
               right="Stress: 50 queries × 4 scenarios")
    out = OUTPUT_DIR / "05_multi_agent_stress.png"
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    return out


# =============================================================================
# Diagram 6 — Multi-Modal Fusion
# =============================================================================


def diagram_6_multimodal_fusion() -> Path:
    """Show 4 modalities flowing into encoders, into a shared embedding space, into MATHIR."""
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("equal")
    ax.axis("off")

    add_title_band(fig, "Multi-Modal Memory  (Text · Image · Audio · Video)",
                   "MATHIR is modality-agnostic — it sees only embeddings")

    # Layout: 4 modality columns on left, encoder boxes, shared embedding space, MATHIR core
    modalities = [
        ("📝 TEXT",   "sentences\nparagraphs\ncode",     "#3b82f6", 12),
        ("🖼️ IMAGE",  "photos\ndiagrams\nscreenshots",   "#10b981", 32),
        ("🎵 AUDIO",  "speech\nmusic\nsound effects",    "#d97706", 52),
        ("🎬 VIDEO",  "frames\nclips\nsubtitles",        "#7c3aed", 72),
    ]

    encoders = [
        ("BERT\nMiniLM\n384-dim",  "#3b82f6", 12),
        ("CLIP\nViT-B/32\n512-dim", "#10b981", 32),
        ("Whisper\n+ECAPA\n512-dim", "#d97706", 52),
        ("VideoCLIP\nTimeSformer\n512-dim", "#7c3aed", 72),
    ]

    # 1) Modality source boxes
    for label, sub, c, x in modalities:
        rounded_box(ax, x, 75, 14, 14, c, label,
                    fontsize=12, radius=0.05)
        ax.text(x + 7, 70, sub, ha="center", va="center",
                fontsize=8.5, color="#475569", style="italic")
        # arrow down
        ax.add_patch(arrow(ax,x + 7, 75, x + 7, 56,
                           color=c, lw=1.8, mutation=14))

    # 2) Encoder boxes
    for label, c, x in encoders:
        rounded_box(ax, x, 41, 14, 14, "#f1f5f9", label,
                    text_color="#0f172a", fontsize=8.5,
                    fontweight="bold", edge_color=c)
        # arrow down to embedding space
        ax.add_patch(arrow(ax,x + 7, 41, x + 7, 27,
                           color=c, lw=1.8, mutation=14))

    # 3) Shared embedding space (one box with little points)
    ax.add_patch(FancyBboxPatch(
        (10, 8), 78, 18,
        boxstyle="round,pad=0.1,rounding_size=0.6",
        facecolor="#fef3c7", edgecolor=COLORS["secondary"],
        linewidth=2.0, zorder=2,
    ))
    ax.text(49, 22.5, "Shared Embedding Space  (d ∈ {384, 512, 1024, 4096})",
            ha="center", va="center", fontsize=12,
            color="#92400e", fontweight="bold", zorder=3)
    # Plot tiny scattered points per modality
    rng = np.random.default_rng(7)
    for c, x in [("#3b82f6", 19), ("#10b981", 39),
                 ("#d97706", 59), ("#7c3aed", 79)]:
        for _ in range(14):
            cx = x + rng.normal(0, 4)
            cy = 13 + rng.normal(0, 3.2)
            ax.add_patch(Circle((cx, cy), radius=0.7,
                                facecolor=c, edgecolor="white",
                                linewidth=0.4, alpha=0.85, zorder=4))
    ax.text(49, 9.5, "modality-agnostic  ·  same vector space",
            ha="center", va="center", fontsize=9.5,
            color="#92400e", style="italic", zorder=3)

    # 4) MATHIR block on the right side, large
    rounded_box(ax, 60, 50, 32, 22, COLORS["primary"],
                "🧠 MATHIR\n5-tier memory\n+ KL router",
                fontsize=12, radius=0.05)
    # Arrow from embedding to MATHIR
    ax.add_patch(arrow(ax,60, 17, 60, 50,
                       color=COLORS["primary"], lw=2.0, mutation=18))
    # MATHIR output arrow
    rounded_box(ax, 60, 30, 32, 14, COLORS["tertiary"],
                "Enhanced embedding\n→ any LLM",
                fontsize=11, radius=0.05)
    ax.add_patch(arrow(ax,76, 50, 76, 44,
                       color=COLORS["tertiary"], lw=1.8, mutation=14))

    # Annotation explaining why modality-agnostic
    ax.text(34, 64, "①  Modality-specific\n    encoders produce\n    fixed-dim vectors",
            ha="center", va="center", fontsize=9,
            color="#1e3a8a",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#dbeafe",
                      edgecolor="#3b82f6", linewidth=0.8))
    ax.text(34, 39, "②  All vectors land\n    in the same space",
            ha="center", va="center", fontsize=9,
            color="#5b21b6",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#ede9fe",
                      edgecolor="#7c3aed", linewidth=0.8))
    ax.text(76, 64, "③  MATHIR stores\n    the vectors, not\n    the raw bytes",
            ha="center", va="center", fontsize=9,
            color="#7c2d12",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffedd5",
                      edgecolor=COLORS["secondary"], linewidth=0.8))

    # 5) Footer info
    ax.text(50, 3, "One slot = one (embedding, modality, key, value) record — typed, but vector-uniform",
            ha="center", va="center", fontsize=9.5,
            color=COLORS["muted"], style="italic")

    add_watermark(ax)
    add_footer(fig, left="Encoders: BERT · CLIP · Whisper · VideoCLIP",
               right="Modality-agnostic core")
    out = OUTPUT_DIR / "06_multimodal_fusion.png"
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    return out


# =============================================================================
# Diagram 7 — Theorem Network
# =============================================================================


def diagram_7_theorem_network() -> Path:
    """NetworkX graph of 6 theorems with dependencies."""
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    add_title_band(fig, "Theoretical Foundations  (6 Theorems)",
                   "Each theorem reduces to a classical result — proofs in docs/PROOFS.md")

    G = nx.DiGraph()

    # Classical results (foundational nodes)
    classical = [
        ("Shannon AWGN\n(1948)",          -2.6,  2.4),
        ("Hoeffding\n(1963)",              2.4,  2.4),
        ("Robbins-Monro\n(1951)",          0.0,  2.2),
        ("Neyman-Pearson\n(1933)",         2.8,  2.4),
        ("Olshausen-Field\n(1996)",       -2.7,  2.4),
        ("Sinkhorn-Knopp\n(1967)",         2.5,  2.4),
    ]
    for label, x, y in classical:
        G.add_node(label, kind="classical", x=x, y=y)

    # MATHIR theorems
    theorems = [
        ("T1: Information\nCapacity",      -1.8,  1.0,
         "I(X;Mₜ) ≤ (N+W+I+2V+P+s)·d·log₂(1+SNR)",
         "#3b82f6"),
        ("T2: Retention\nGuarantee",        1.8,  1.0,
         "Pr(Acc(K) ≥ 1−CKLη/N) ≥ 1−exp(−N/2)",
         "#10b981"),
        ("T3: Router\nConvergence",         0.0,  0.0,
         "O(1/ε) iterations (Robbins-Monro)",
         "#d97706"),
        ("T4: Anomaly\nOptimality",         2.7,  0.0,
         "Mahalanobis ≡ Neyman-Pearson",
         "#dc2626"),
        ("T5: Sparse Coding\nBound",       -2.7,  0.0,
         "𝔼[‖X−D⊤z*‖²] ≤ O(s·σ²/K)",
         "#7c3aed"),
        ("T6: mHC Geometry",                1.2, -1.0,
         "Linear-rate Sinkhorn-Knopp",
         COLORS["gold"]),
    ]
    for label, x, y, _, color in theorems:
        G.add_node(label, kind="theorem", x=x, y=y, color=color)

    # Edges: theorems depend on classical results
    edges = [
        ("Shannon AWGN\n(1948)",    "T1: Information\nCapacity"),
        ("Olshausen-Field\n(1996)", "T5: Sparse Coding\nBound"),
        ("Hoeffding\n(1963)",       "T2: Retention\nGuarantee"),
        ("Robbins-Monro\n(1951)",   "T2: Retention\nGuarantee"),
        ("Robbins-Monro\n(1951)",   "T3: Router\nConvergence"),
        ("Neyman-Pearson\n(1933)",  "T4: Anomaly\nOptimality"),
        ("Sinkhorn-Knopp\n(1967)",  "T6: mHC Geometry"),
    ]
    G.add_edges_from(edges)

    pos = {n: (G.nodes[n]["x"], G.nodes[n]["y"]) for n in G.nodes}

    # Edges (drawn first)
    nx.draw_networkx_edges(
        G, pos, ax=ax, edgelist=edges,
        edge_color="#94a3b8",
        arrows=True, arrowsize=18,
        width=1.4, style="--",
        connectionstyle="arc3,rad=0.0",
        min_source_margin=22, min_target_margin=22,
    )

    # Classical nodes (small grey)
    classical_nodes = [n for n in G.nodes if G.nodes[n].get("kind") == "classical"]
    nx.draw_networkx_nodes(
        G, pos, nodelist=classical_nodes, ax=ax,
        node_color="#e2e8f0",
        edgecolors="#64748b",
        node_shape="s",
        node_size=2200, linewidths=1.5, alpha=0.95,
    )

    # Theorem nodes (larger, colored)
    for th_label, _, _, _, color in theorems:
        nx.draw_networkx_nodes(
            G, pos, nodelist=[th_label], ax=ax,
            node_color=color,
            edgecolors="white",
            node_shape="o",
            node_size=3200, linewidths=2.5, alpha=0.95,
        )

    # Labels — classical
    for n in classical_nodes:
        ax.text(pos[n][0], pos[n][1], n,
                ha="center", va="center",
                fontsize=8.5, color="#0f172a",
                fontweight="normal")

    # Labels — theorems
    for th_label, _, _, formula, color in theorems:
        x, y = pos[th_label]
        ax.text(x, y + 0.06, th_label,
                ha="center", va="center",
                fontsize=10, color="white", fontweight="bold")
        ax.text(x, y - 0.22, formula,
                ha="center", va="center",
                fontsize=7.2, color="white", style="italic", alpha=0.95)

    # Frame the classical row
    ax.add_patch(Rectangle((-3.7, 1.85), 7.4, 1.0,
                           facecolor="#f8fafc", edgecolor="#cbd5e1",
                           linewidth=0.8, linestyle="--", zorder=0))
    ax.text(0, 3.05, "Classical foundations  (citations only)",
            ha="center", va="center",
            fontsize=9, color="#475569", style="italic", fontweight="bold")

    # Bottom annotation
    ax.text(0, -1.6, "MATHIR's contribution: application of classical results to memory-augmented agents",
            ha="center", va="center", fontsize=9.5,
            color=COLORS["muted"], style="italic")

    # Legend
    legend_patches = [
        mpatches.Patch(color="#3b82f6", label="T1 — Capacity"),
        mpatches.Patch(color="#10b981", label="T2 — Retention"),
        mpatches.Patch(color="#d97706", label="T3 — Router"),
        mpatches.Patch(color="#dc2626", label="T4 — Anomaly"),
        mpatches.Patch(color="#7c3aed", label="T5 — Sparse"),
        mpatches.Patch(color=COLORS["gold"], label="T6 — mHC"),
        mpatches.Patch(facecolor="#e2e8f0", edgecolor="#64748b",
                       label="Classical result"),
    ]
    ax.legend(handles=legend_patches, loc="lower right",
              frameon=True, framealpha=0.95, edgecolor="#cbd5e1",
              fontsize=9, ncol=1, bbox_to_anchor=(1.0, -0.06))

    ax.set_xlim(-4.4, 4.4)
    ax.set_ylim(-1.9, 3.5)
    ax.set_aspect("equal")
    ax.axis("off")

    add_watermark(ax)
    add_footer(fig, left="All 6 theorems proven in docs/PROOFS.md",
               right="See docs/THEORY_V7.md for full derivations")
    out = OUTPUT_DIR / "07_theorem_network.png"
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    return out


# =============================================================================
# Diagram 8 — Version Timeline
# =============================================================================


def diagram_8_version_timeline() -> Path:
    """Horizontal timeline of MATHIR versions V1 → V8.4.1."""
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    add_title_band(fig, "MATHIR Evolution  (V1 → V8.4.1)",
                   "5 months · 8 releases · 62+ unit tests · 8 novel algorithms · 6 theorems")

    # Versions (in display order)
    versions = [
        ("V1",   "Jan 2026",  "CNN + MLP\n3-tier memory",        "#ef4444", "legacy"),
        ("V2",   "Jan 2026",  "Bug fixes\n+ 12 unit tests",      "#ef4444", "legacy"),
        ("V3",   "Jan 2026",  "Stability &\nAPI refactor",      "#ef4444", "legacy"),
        ("V4",   "Jan 2026",  "mHC + Sinkhorn\n-Knopp",          "#f97316", "legacy"),
        ("V5",   "Jan 2026",  "KL router\n+ immune memory",      "#f59e0b", "stable"),
        ("V5.1", "Feb 2026",  "21 bug fixes\nhardening pass",    "#f59e0b", "stable"),
        ("V6",   "Jun 2026",  "MATHIRPlugin\nLLM-agnostic",      "#10b981", "stable"),
        ("V7",   "Jun 2026",  "8 algorithms\n+ 6 theorems",      "#059669", "stable"),
        ("V7.1", "Jun 2026",  "Retrieval A/B/C/D\n+ benchmarks", "#047857", "stable"),
        ("V8.4.1", "Jun 2026",  "Latency opt\n+ cache + ONNX",     COLORS["gold"], "current"),
    ]

    n = len(versions)
    y_axis = 50
    x_left = 6
    x_right = 94
    x_positions = np.linspace(x_left, x_right, n)

    # Timeline spine
    ax.add_patch(FancyBboxPatch(
        (x_left - 1, y_axis - 1.2), x_right - x_left + 2, 2.4,
        boxstyle="round,pad=0.0,rounding_size=0.4",
        facecolor="#e2e8f0", edgecolor="#94a3b8",
        linewidth=1.2, zorder=1,
    ))

    # Status legend bar (top)
    ax.add_patch(Rectangle((6, 86), 88, 4, facecolor="#fafafa",
                           edgecolor="#cbd5e1", linewidth=0.8, zorder=0))
    ax.text(13, 88, "Status legend:", ha="left", va="center",
            fontsize=10, color="#0f172a", fontweight="bold")
    legend_items = [
        ("Legacy",      "#ef4444"),
        ("Stable",      "#10b981"),
        ("Current ★",   COLORS["gold"]),
    ]
    lx = 28
    for name, c in legend_items:
        ax.add_patch(Circle((lx, 88), 0.9, facecolor=c,
                            edgecolor="white", linewidth=0.8))
        ax.text(lx + 2, 88, name, ha="left", va="center",
                fontsize=10, color="#0f172a")
        lx += 14

    # Plot each version
    for i, (ver, date, note, color, status) in enumerate(versions):
        x = x_positions[i]
        is_current = status == "current"
        # Marker
        ax.add_patch(Circle((x, y_axis), 2.0 if not is_current else 2.6,
                            facecolor=color, edgecolor="white",
                            linewidth=2.5, zorder=4))
        # Version label inside the marker
        ax.text(x, y_axis, ver, ha="center", va="center",
                fontsize=8 if not is_current else 9,
                color="white", fontweight="bold", zorder=5)

        # Alternate above/below for note cards
        above = (i % 2 == 0)
        card_y = y_axis + 6 if above else y_axis - 6
        card_h = 9
        # Connector line
        ax.add_line(Line2D([x, x], [y_axis, card_y + (0 if above else -card_h)],
                            color=color, linewidth=1.0,
                            linestyle="--", alpha=0.7, zorder=2))
        # Card
        box = FancyBboxPatch(
            (x - 5.2, card_y - (0 if above else card_h)),
            10.4, card_h,
            boxstyle="round,pad=0.05,rounding_size=0.3",
            facecolor="white", edgecolor=color,
            linewidth=1.5 if is_current else 1.0, zorder=3,
        )
        ax.add_patch(box)
        # Note text
        y_text_top = card_y - 1.3 if above else card_y - card_h + 1.3
        y_text_bot = card_y - card_h + 1.3 if above else card_y - 1.3
        ax.text(x, y_text_top if above else card_y - 1.3,
                note, ha="center", va="center",
                fontsize=8.2, color="#0f172a",
                fontweight="bold")
        ax.text(x, y_text_bot,
                date, ha="center", va="center",
                fontsize=7.5, color=COLORS["muted"], style="italic")

    # Bottom banner with key stats
    stats = [
        ("5",   "months"),
        ("8",   "releases"),
        ("8",   "algorithms"),
        ("6",   "theorems"),
        ("62+", "unit tests"),
        ("4",   "retrieval approaches"),
    ]
    sx = 12
    sw = 14
    ax.add_patch(Rectangle((6, 6), 88, 18, facecolor="#f1f5f9",
                           edgecolor="#cbd5e1", linewidth=0.8, zorder=0))
    for i, (n_, lab) in enumerate(stats):
        cx = sx + i * sw
        ax.text(cx, 19, n_, ha="center", va="center",
                fontsize=18, fontweight="bold", color=COLORS["primary"])
        ax.text(cx, 12, lab, ha="center", va="center",
                fontsize=9, color="#475569")

    # Watermark
    add_watermark(ax)
    add_footer(fig, left="Source: CHANGELOG.md",
               right="Latest = V8.4.1 (gold)")

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("equal")
    ax.axis("off")

    out = OUTPUT_DIR / "08_version_timeline.png"
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    return out


# =============================================================================
# Main
# =============================================================================


def main() -> int:
    print("=" * 70)
    print("MATHIR V8.4.1 — Visual Diagrams Suite")
    print("=" * 70)
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    diagrams = [
        ("01 — Architecture (main)",      diagram_1_architecture_main),
        ("02 — 4 Memory Tiers",            diagram_2_4_memory_tiers),
        ("03 — Retrieval Quality",         diagram_3_retrieval_comparison),
        ("04 — Latency / Quality Pareto",  diagram_4_latency_quality_tradeoff),
        ("05 — Multi-Agent Stress",        diagram_5_multi_agent_stress),
        ("06 — Multi-Modal Fusion",        diagram_6_multimodal_fusion),
        ("07 — Theorem Network",           diagram_7_theorem_network),
        ("08 — Version Timeline",          diagram_8_version_timeline),
    ]

    results = []
    for name, fn in diagrams:
        print(f"  → Generating {name} ...", end=" ", flush=True)
        try:
            path = fn()
            size_kb = path.stat().st_size / 1024
            print(f"OK  ({size_kb:.0f} KB  ·  {path.name})")
            results.append((name, path, "OK", size_kb))
        except Exception as e:  # pragma: no cover
            print(f"FAILED  ({type(e).__name__}: {e})")
            results.append((name, None, "FAIL", 0))

    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    ok = sum(1 for _, _, s, _ in results if s == "OK")
    total_size = sum(sz for _, _, s, sz in results if s == "OK")
    print(f"  Generated: {ok}/{len(results)} diagrams")
    print(f"  Total size: {total_size:.0f} KB")
    print(f"  Output:     {OUTPUT_DIR}")
    print()
    print("Next step:")
    print("  python visualizations/build_report.py  → builds visual_report.html")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
