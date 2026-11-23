#!/usr/bin/env python3
"""
Suite 11 — Full 6-Model Ollama Image Suite + 3x3 Live RTX 5080 Side-by-Side Visual Comparison Chart
====================================================================================================
Generates `assets/chart_07_suite11_6model_comparison.png` combining:
  - Top-Left:  All 6 Published Ollama Image Controllers (7/7 Gate, THINK=0, Decode tok/s & Hallucination %)
  - Top-Right: Live RTX 5080 1024x1024 Render Speed & Peak VRAM (FLUX.2-klein-4B vs Z-Image-Turbo vs Qwen-Image-2.1)
  - Bottom:    3x3 Live Visual Contact Sheet (IMG-01 Typography, IMG-02 PT-BR Hardware, IMG-06 85mm Portrait)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

SUITE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = SUITE_DIR / "assets"
QWEN_OLLAMA_JSON = SUITE_DIR / "ollama_gate_benchmark.json"
MULTI_OLLAMA_JSON = SUITE_DIR / "multi_image_ollama_benchmark.json"
VISUAL_COMP_JSON = SUITE_DIR / "multimodel_visual_comparison.json"
OUT_PNG = ASSETS_DIR / "chart_07_suite11_6model_comparison.png"

BG = "#0d1117"
PANEL = "#161b22"
BORDER = "#30363d"
TEXT = "#e6edf3"
MUTED = "#8b949e"
BLUE = "#58a6ff"
GREEN = "#3fb950"
PURPLE = "#bc8cff"
ORANGE = "#ffa657"
CYAN = "#39c5cf"
PINK = "#f778ba"


def main() -> None:
    qwen_ol = json.loads(QWEN_OLLAMA_JSON.read_text(encoding="utf-8"))
    multi_ol = json.loads(MULTI_OLLAMA_JSON.read_text(encoding="utf-8"))
    vis_comp = json.loads(VISUAL_COMP_JSON.read_text(encoding="utf-8"))

    # Assemble 6 Ollama models in canonical order
    ollama_models = [
        {
            "label": "flux2-klein\n(4B VLM · Apache 2.0)",
            "avg": next(m["zero_refusal_gate"]["avg_decode_tok_s"] for m in multi_ol if "flux2" in m["repo"]),
            "peak": next(m["zero_refusal_gate"]["peak_decode_tok_s"] for m in multi_ol if "flux2" in m["repo"]),
            "halluc": next(m["hallucination"]["hallucination_rate_pct"] for m in multi_ol if "flux2" in m["repo"]),
            "color": CYAN,
        },
        {
            "label": "z-image-turbo\n(4B VLM · Apache 2.0)",
            "avg": next(m["zero_refusal_gate"]["avg_decode_tok_s"] for m in multi_ol if "z-image" in m["repo"]),
            "peak": next(m["zero_refusal_gate"]["peak_decode_tok_s"] for m in multi_ol if "z-image" in m["repo"]),
            "halluc": next(m["hallucination"]["hallucination_rate_pct"] for m in multi_ol if "z-image" in m["repo"]),
            "color": GREEN,
        },
        {
            "label": "qwen-image-2.1\n(8B VLM · Qwen Res.)",
            "avg": qwen_ol["zero_refusal_gate"]["avg_decode_tok_s"],
            "peak": qwen_ol["zero_refusal_gate"]["peak_decode_tok_s"],
            "halluc": (qwen_ol.get("hallucination") or qwen_ol.get("precision_and_hallucination") or {}).get("hallucination_rate_pct", 0.0),
            "color": BLUE,
        },
        {
            "label": "minimax-h3-image\n(8B VLM · 33.1B DiT)",
            "avg": next(m["zero_refusal_gate"]["avg_decode_tok_s"] for m in multi_ol if "minimax" in m["repo"]),
            "peak": next(m["zero_refusal_gate"]["peak_decode_tok_s"] for m in multi_ol if "minimax" in m["repo"]),
            "halluc": next(m["hallucination"]["hallucination_rate_pct"] for m in multi_ol if "minimax" in m["repo"]),
            "color": PURPLE,
        },
        {
            "label": "moss-image\n(9B Enc · Apache 2.0)",
            "avg": next(m["zero_refusal_gate"]["avg_decode_tok_s"] for m in multi_ol if "moss" in m["repo"]),
            "peak": next(m["zero_refusal_gate"]["peak_decode_tok_s"] for m in multi_ol if "moss" in m["repo"]),
            "halluc": next(m["hallucination"]["hallucination_rate_pct"] for m in multi_ol if "moss" in m["repo"]),
            "color": ORANGE,
        },
        {
            "label": "glm-image\n(9B AR · MIT Law #1)",
            "avg": next(m["zero_refusal_gate"]["avg_decode_tok_s"] for m in multi_ol if "glm-image" in m["repo"]),
            "peak": next(m["zero_refusal_gate"]["peak_decode_tok_s"] for m in multi_ol if "glm-image" in m["repo"]),
            "halluc": next(m["hallucination"]["hallucination_rate_pct"] for m in multi_ol if "glm-image" in m["repo"]),
            "color": PINK,
        },
    ]

    plt.rcParams["font.family"] = "DejaVu Sans"
    fig = plt.figure(figsize=(18.5, 15.8), dpi=160, facecolor=BG)
    outer = gridspec.GridSpec(2, 1, height_ratios=[1.0, 1.85], hspace=0.22, left=0.13, right=0.96, top=0.915, bottom=0.03)

    fig.suptitle(
        "Suite 11 — Complete Open-Weights Image Generation Benchmark (6 Ollama Hub Models + Live RTX 5080 Visual Grid)",
        color=TEXT,
        fontsize=15.5,
        fontweight="bold",
        y=0.965,
    )
    fig.text(
        0.5,
        0.938,
        "NVIDIA GeForce RTX 5080 (16GB GDDR7, CUDA 13.2)  |  All 6 Ollama Hub Repos Verified 7/7 (100%) Zero-Refusal Gate + THINK_LEN=0  |  Same Prompts & Seeds (1024×1024)",
        color=MUTED,
        fontsize=10.5,
        ha="center",
    )

    # Top Row: 2 Charts
    top = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0], width_ratios=[1.25, 1.0], wspace=0.22)

    # Panel A: 6 Published Ollama Models Decode tok/s
    ax1 = fig.add_subplot(top[0])
    ax1.set_facecolor(PANEL)
    for sp in ax1.spines.values():
        sp.set_color(BORDER)

    y = np.arange(len(ollama_models))
    avgs = [m["avg"] for m in ollama_models]
    peaks = [m["peak"] for m in ollama_models]
    colors = [m["color"] for m in ollama_models]

    bars = ax1.barh(y, avgs, height=0.56, color=colors, alpha=0.88, edgecolor="none")
    ax1.scatter(peaks, y, color="#ffffff", s=38, zorder=5, label="Peak tok/s")
    ax1.set_yticks(y)
    ax1.set_yticklabels([m["label"] for m in ollama_models], color=TEXT, fontsize=8.8)
    ax1.invert_yaxis()
    ax1.set_xlim(0, 215)
    ax1.set_xlabel("RTX 5080 Decode Throughput (tokens/sec · Q4_K_M · THINK_LEN=0)", color=MUTED, fontsize=9)
    ax1.set_title("A. All 6 Published Ollama Image Controllers (42/42 Gate PASS)", color=TEXT, fontsize=11.5, fontweight="bold", pad=10)
    ax1.tick_params(axis="x", colors=MUTED)
    ax1.grid(axis="x", color=BORDER, linestyle="--", alpha=0.5)
    ax1.set_axisbelow(True)

    for idx, (bar, m) in enumerate(zip(bars, ollama_models)):
        h_badge = f"Halluc {m['halluc']:.0f}%"
        ax1.text(
            m["peak"] + 3.0,
            idx,
            f"{m['avg']:.1f} avg / {m['peak']:.1f} pk  ({h_badge})",
            va="center",
            ha="left",
            color=TEXT,
            fontsize=8.3,
            fontweight="bold" if m["halluc"] == 0.0 else "normal",
        )
    ax1.legend(loc="lower right", facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT, fontsize=8.5)

    # Panel B: Live RTX 5080 Diffusion Latency & Peak VRAM (3 Local Engines)
    ax2 = fig.add_subplot(top[1])
    ax2.set_facecolor(PANEL)
    for sp in ax2.spines.values():
        sp.set_color(BORDER)

    v_map = {m["key"]: m for m in vis_comp["models"]}
    vis_order = [
        ("flux2_klein_4b", "FLUX.2-klein-4B\n(4 steps · BF16\nApache 2.0)", CYAN),
        ("z_image_turbo", "Z-Image-Turbo\n(9 steps · NF4\nApache 2.0)", GREEN),
        ("qwen_image_2_1_turbo", "Qwen-Image-2.1\n(8 steps · NF4\nRGBA + Edit)", BLUE),
    ]
    x = np.arange(len(vis_order))
    latencies = [v_map[k]["avg_render_s"] for k, _, _ in vis_order]
    vrams_gb = [v_map[k]["max_peak_vram_mb"] / 1024.0 for k, _, _ in vis_order]
    v_cols = [c for _, _, c in vis_order]

    w = 0.34
    b1 = ax2.bar(x - w / 2, latencies, width=w, color=v_cols, alpha=0.92, label="Avg 1024² Render Time (s)")
    ax2_r = ax2.twinx()
    b2 = ax2_r.bar(x + w / 2, vrams_gb, width=w, color="#8b949e", alpha=0.55, edgecolor=TEXT, linewidth=0.7, label="Peak GPU VRAM (GB)")

    ax2.set_xticks(x)
    ax2.set_xticklabels([lbl for _, lbl, _ in vis_order], color=TEXT, fontsize=8.8)
    ax2.set_ylim(0, 19.5)
    ax2_r.set_ylim(0, 12.0)
    ax2.set_ylabel("Warm 1024×1024 Render Latency (seconds, lower = faster)", color=TEXT, fontsize=8.8)
    ax2_r.set_ylabel("Peak Allocated VRAM (GB / 16 GB GDDR7)", color=MUTED, fontsize=8.8)
    ax2.set_title("B. Live RTX 5080 Diffusion Speed vs Peak VRAM (1024×1024)", color=TEXT, fontsize=11.5, fontweight="bold", pad=10)
    ax2.tick_params(axis="y", colors=MUTED)
    ax2_r.tick_params(axis="y", colors=MUTED)
    for sp in ax2_r.spines.values():
        sp.set_color(BORDER)
    ax2.grid(axis="y", color=BORDER, linestyle="--", alpha=0.45)
    ax2.set_axisbelow(True)

    for rect, val in zip(b1, latencies):
        ax2.text(rect.get_x() + rect.get_width() / 2, val + 0.4, f"{val:.2f}s", ha="center", va="bottom", color=TEXT, fontsize=9, fontweight="bold")
    for rect, val in zip(b2, vrams_gb):
        ax2_r.text(rect.get_x() + rect.get_width() / 2, val + 0.25, f"{val:.2f} GB", ha="center", va="bottom", color="#d2d8e0", fontsize=8.6)

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_r.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper left", facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT, fontsize=8.3)

    # Bottom Panel: 3x3 Live Visual Contact Sheet
    bot = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=outer[1], wspace=0.06, hspace=0.14)
    col_specs = [
        ("Qwen-Image-2.1-Turbo (8 steps · NF4 · 14.85s avg)", "qwen", BLUE),
        ("Z-Image-Turbo (9 steps · NF4 · 11.24s avg · 3.47 GB)", "zimage", GREEN),
        ("FLUX.2-klein-4B (4 steps · BF16 · 8.91s avg · 7.83 GB)", "flux2klein", CYAN),
    ]
    row_specs = [
        ("IMG-01 · Typography\n(\"J.A.R.V.I.S. A+\" · seed 20261123)", "img_01_typography_studio.png", {
            "qwen": "14.51s · Exact text",
            "zimage": "11.16s · Exact text",
            "flux2klein": "9.20s · 'J.A.RVIS.' quirk",
        }),
        ("IMG-02 · PT-BR Editorial\n(Hardware · seed 42)", "img_02_editorial_ptbr.png", {
            "qwen": "14.41s · Brushed steel",
            "zimage": "11.46s · Clean 4-bay rack",
            "flux2klein": "9.40s · 'IB&G' artifact",
        }),
        ("IMG-06 · 85mm Portrait\n(Robotics Lab · seed 20261128)", "img_06_cinematic_portrait.png", {
            "qwen": "15.63s · Natural bokeh",
            "zimage": "11.09s · Sharp hands & Leica",
            "flux2klein": "8.14s · Warm film grain",
        }),
    ]

    for r_idx, (row_lbl, suffix, captions) in enumerate(row_specs):
        for c_idx, (col_title, prefix, col_color) in enumerate(col_specs):
            ax = fig.add_subplot(bot[r_idx, c_idx])
            ax.set_facecolor(PANEL)
            img_path = ASSETS_DIR / f"{prefix}_{suffix}"
            im = Image.open(img_path).convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)
            ax.imshow(im)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color(col_color if r_idx == 0 else BORDER)
                sp.set_linewidth(1.8 if r_idx == 0 else 1.0)
            if r_idx == 0:
                ax.set_title(col_title, color=col_color, fontsize=9.8, fontweight="bold", pad=6)
            if c_idx == 0:
                ax.set_ylabel(row_lbl, color=TEXT, fontsize=8.8, fontweight="bold", labelpad=8)
            ax.text(
                0.02,
                0.04,
                captions[prefix],
                transform=ax.transAxes,
                color="#ffffff",
                fontsize=8.3,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#0d1117", edgecolor=col_color, alpha=0.84, linewidth=0.9),
            )

    fig.savefig(OUT_PNG, facecolor=BG, dpi=160)
    plt.close(fig)
    print(f"CHART_07_SAVED -> {OUT_PNG} ({OUT_PNG.stat().st_size / 1024:.1f} KB)", flush=True)


if __name__ == "__main__":
    main()
