#!/usr/bin/env python3
"""
Generates Suite 11 executive visual telemetry chart:
  `chart_06_qwen_image_2_1_turbo_benchmark.png`
Combining:
  - Panel 1: Open-Weights Image Generation Leaderboard Elo (Artificial Analysis AA-Image-T2I v2.1)
    vs. GPU VRAM Footprint (highlighting Caminho A+ Dual-Tier RTX 5080 16GB + Colab 80GB).
  - Panel 2: Live Measured RTX 5080 (16GB) Render Latency across IMG-01..IMG-06 (Qwen-Image-2.1-Turbo, 8 steps, NF4).
  - Panel 3: Live 7-Prompt Zero-Refusal Gate Decode Speed (tok/s, THINK_LEN=0) for eriktonon/qwen-image-2.1-turbo-uncensored.
"""

from __future__ import annotations

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SUITE_DIR = Path(__file__).resolve().parent
VIS_JSON = SUITE_DIR / "visual_benchmark_results.json"
GATE_JSON = SUITE_DIR / "ollama_gate_benchmark.json"
OUT_CHART = SUITE_DIR / "assets" / "chart_06_qwen_image_2_1_turbo_benchmark.png"


def main() -> None:
    vis = json.loads(VIS_JSON.read_text(encoding="utf-8"))
    gate = json.loads(GATE_JSON.read_text(encoding="utf-8"))

    plt.rcParams.update({
        "figure.facecolor": "#0d1117",
        "axes.facecolor": "#161b22",
        "axes.edgecolor": "#30363d",
        "axes.labelcolor": "#e6edf3",
        "text.color": "#e6edf3",
        "xtick.color": "#9da7b3",
        "ytick.color": "#9da7b3",
        "grid.color": "#262c36",
        "font.family": "DejaVu Sans",
    })

    fig, axes = plt.subplots(1, 3, figsize=(21, 6.8), dpi=160)
    fig.suptitle(
        "Suite 11 — Caminho A+ Multimodal Image Engine Benchmark (Qwen-Image-2.1-Turbo RTX 5080 + MiniMax-H3 Colab 80GB)",
        fontsize=15,
        fontweight="bold",
        color="#f0f6fc",
        y=0.98,
    )

    # Panel 1: Global Open-Weights Arena Elo vs. Practical VRAM
    ax1 = axes[0]
    models = [
        ("MiniMax-H3-Image\n(33.1B · Tier-2 Colab)", 1066, 72.6, "#a371f7", (-3.0, -1.0, "right")),
        ("Qwen-Image-2.1-Turbo\n(7B DiT · Tier-1 RTX5080)", 1053, 9.7, "#39d353", (3.5, 5.5, "left")),
        ("FLUX 3 Image Klein\n(5.6B · Gated HF)", 1041, 10.5, "#58a6ff", (3.0, -14.0, "left")),
        ("Fibo 2.0 (Bria)", 1037, 24.0, "#8b949e", (3.5, 2.5, "left")),
        ("FLUX.2 [dev] (Anchor)", 1000, 26.5, "#8b949e", (3.5, 3.5, "left")),
        ("Z-Image-Turbo\n(6B · Apache 2.0)", 995, 11.5, "#d29922", (3.0, -10.5, "left")),
    ]
    for name, elo, vram, color, (dx, dy, ha) in models:
        is_hero = "Qwen-Image" in name or "MiniMax" in name
        size = 340 if is_hero else 180
        edge = "#ffffff" if is_hero else "#0d1117"
        ax1.scatter(vram, elo, s=size, color=color, edgecolors=edge, linewidth=1.8, zorder=4)
        ax1.annotate(
            f"{name}\n(Elo {elo} · {vram:.1f}GB)",
            (vram, elo),
            xytext=(vram + dx, elo + dy),
            fontsize=8.2,
            color="#e6edf3",
            ha=ha,
            va="center",
            fontweight="bold" if is_hero else "normal",
            arrowprops={"arrowstyle": "-", "color": "#484f58", "lw": 0.9, "shrinkA": 2, "shrinkB": 6},
        )
    ax1.axvline(16.0, color="#39d353", linestyle="--", linewidth=1.3, alpha=0.8)
    ax1.text(17.2, 1075, "RTX 5080 16GB\nVRAM Ceiling", color="#39d353", fontsize=8.3, fontweight="bold", va="top")
    ax1.set_title("1. Open-Weights Arena Elo vs. VRAM Budget", fontsize=11.5, fontweight="bold", pad=10)
    ax1.set_xlabel("Practical VRAM Required (GB)")
    ax1.set_ylabel("Artificial Analysis Open-Weights Elo (AA-T2I v2.1)")
    ax1.set_xlim(0, 82)
    ax1.set_ylim(976, 1082)
    ax1.grid(True, linestyle=":", alpha=0.5)

    # Panel 2: Live RTX 5080 Measured Latency (IMG-01..IMG-06)
    ax2 = axes[1]
    short_names = {
        "IMG-01": "IMG-01\nPhotorealism + Text",
        "IMG-02": "IMG-02\nEditorial PT-BR",
        "IMG-03": "IMG-03\nNative RGBA Alpha",
        "IMG-04": "IMG-04\nArch Poster 16:9",
        "IMG-05": "IMG-05\nI2I Reference Edit",
        "IMG-06": "IMG-06\n85mm Portrait",
    }
    labels = [short_names.get(r["id"], r["id"]) for r in vis]
    latencies = [r["elapsed_s"] for r in vis]
    colors2 = ["#39d353" if r["reference_count"] == 0 and not r["transparent_pct"] else ("#58a6ff" if r["transparent_pct"] > 0 else "#a371f7") for r in vis]
    y_pos = np.arange(len(labels))
    bars = ax2.barh(y_pos, latencies, color=colors2, edgecolor="#0d1117", height=0.62)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels, fontsize=8.5)
    ax2.invert_yaxis()
    for bar, r in zip(bars, vis):
        extra = f" · {r['transparent_pct']}% alpha" if r["transparent_pct"] > 0 else (" · I2I edit" if r["reference_count"] > 0 else "")
        ax2.text(
            bar.get_width() + 0.4,
            bar.get_y() + bar.get_height() / 2,
            f"{r['elapsed_s']:.2f}s ({r['resolution']}{extra})",
            va="center",
            fontsize=8.3,
            color="#e6edf3",
            fontweight="bold",
        )
    ax2.set_xlim(0, max(latencies) * 1.48)
    ax2.set_title("2. RTX 5080 Warm Latency (8 Steps · NF4 + Offload)", fontsize=11.5, fontweight="bold", pad=10)
    ax2.set_xlabel("End-to-End Render Time (seconds) — 14.8 / 16.3 GB VRAM Free")
    ax2.grid(True, axis="x", linestyle=":", alpha=0.5)

    # Panel 3: 7-Prompt Zero-Refusal Gate Decode Throughput
    ax3 = axes[2]
    cases = gate["zero_refusal_gate"]["cases"]
    g_labels = [c["id"] for c in cases]
    g_tps = [c["decode_tok_s"] for c in cases]
    g_colors = ["#39d353" if c["pass"] else "#f85149" for c in cases]
    x_pos = np.arange(len(g_labels))
    bars3 = ax3.bar(x_pos, g_tps, color=g_colors, edgecolor="#0d1117", width=0.58)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(g_labels, fontsize=8.2, rotation=22, ha="right")
    for bar, c in zip(bars3, cases):
        status = "PASS" if c["pass"] else "FAIL"
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{c['decode_tok_s']:.1f}\n{status}",
            ha="center",
            va="bottom",
            fontsize=8.0,
            color="#39d353" if c["pass"] else "#f85149",
            fontweight="bold",
        )
    score_str = f"{gate['zero_refusal_gate']['passed']}/{gate['zero_refusal_gate']['total']}"
    avg_tps = gate["zero_refusal_gate"]["avg_decode_tok_s"]
    ax3.set_ylim(0, max(g_tps) * 1.28)
    ax3.set_title(f"3. Ollama 7-Gate ({score_str} PASS · Avg {avg_tps} tok/s · THINK=0)", fontsize=11.0, fontweight="bold", pad=10)
    ax3.set_ylabel("Decode Throughput (tok/s on RTX 5080)")
    ax3.grid(True, axis="y", linestyle=":", alpha=0.5)

    plt.tight_layout(rect=[0, 0.02, 1, 0.93])
    fig.text(
        0.5,
        0.008,
        "Measured live on NVIDIA GeForce RTX 5080 (16GB GDDR7, CUDA 13.2, diffusers 0.42.1, transformers 5.24.0, Ollama 0.40.14) · ollama.com/eriktonon/qwen-image-2.1-turbo-uncensored",
        ha="center",
        fontsize=8.5,
        color="#8b949e",
    )
    fig.savefig(OUT_CHART, dpi=160)
    plt.close(fig)
    print(f"CHART_SAVED -> {OUT_CHART} ({OUT_CHART.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
