#!/usr/bin/env python3
"""
Suite 08 — RouterBench Consolidator, 21-Scenario Mapper & High-DPI Heatmap Generator
=====================================================================================
Merges Colab shards (`scenario_matrix_shard_1..3.json`) into:
  1. `scenario_specialization_matrix.json` & `.csv`
  2. Top-3 Overall Champions & Top-3 Sub-16GB Champions for each of the 10 SOTA Scenarios (`D01..D10`)
  3. Empirical Routing Table for all 21 JARVIS HUD Scenarios (`S01..S21`)
  4. High-DPI Visual Charts (`chart_04_scenario_specialization_heatmap.png` & `chart_05_scenario_champions_board.png`)
  5. Mirrors artifacts into `training/github_benchmark_repo/` for public GitHub publication.
"""

import csv
import json
import os
import shutil
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SUITE_DIR = Path(__file__).resolve().parent
GITHUB_REPO_DIR = SUITE_DIR.parent / "github_benchmark_repo"

SCENARIO_DEFS = [
    ("D01_SWE_CODE", "SWE Code", "Engenharia de Software & Algoritmos", "LiveCodeBench + BigCodeBench (Python Sandbox)"),
    ("D02_SQL_DATA", "BIRD-SQL", "SQL Analítico & Dados Sujos", "BIRD-SQL + Spider 2.0 (SQLite Execution)"),
    ("D03_BFCL_TOOLS", "Tool-Call", "Tool-Calling Agêntico & JSON AST", "BFCL v3 AST + Negative Restraint"),
    ("D04_IFEVAL_STRICT", "IFEval", "Seguimento Estrito de Instruções", "IFEval / IFBench (6 Hard Constraints)"),
    ("D05_RULER_ONTOLOGY", "Ontology", "Ontologia OWL/SPARQL & Long-Context", "RULER 3-Hop + SimpleQA Abstention"),
    ("D06_SRE_K8S", "SRE / K8s", "SRE, Kubernetes & Root-Cause", "AIOps Multi-Signal Triage"),
    ("D07_SECOPS_RE", "SecOps / RE", "SecOps Ofensivo, Pentest & Eng. Reversa", "HarmBench + OR-Bench + x86-64 REL32"),
    ("D08_GPQA_MATH", "GPQA / Math", "Ciência Avançada, Clínica & Matemática", "GPQA Diamond + MATH-500 CRT"),
    ("D09_VIDEO_SVG_916", "9:16 SVG", "Vídeos Curtos 9:16 & Motion SVG", "Creative-IFEval + XML/SVG DOM Parser"),
    ("D10_COPY_RECIPES", "Copy / Cook", "Artigos Executivos & Receitas Técnicas", "Anti-AI Humanizer + Gastronomia Térmica"),
]

JARVIS_21_MAP = [
    ("S01", "Vídeos Curtos 9:16 — Direção de Cena & Keyframes", ["D09_VIDEO_SVG_916", "D04_IFEVAL_STRICT"]),
    ("S02", "Vídeos Curtos 9:16 — Storyboard Cinematográfico & Locução", ["D09_VIDEO_SVG_916", "D10_COPY_RECIPES"]),
    ("S03", "Vídeos Curtos 9:16 — Motion Design SVG/Canvas 60fps", ["D09_VIDEO_SVG_916", "D01_SWE_CODE"]),
    ("S04", "Vídeos Curtos 9:16 — Roteiros Reels Rápidos & Hooks", ["D09_VIDEO_SVG_916", "D10_COPY_RECIPES"]),
    ("S05", "Softwares — Full-Stack Rust/Axum + PostgreSQL", ["D01_SWE_CODE", "D02_SQL_DATA", "D07_SECOPS_RE"]),
    ("S06", "Softwares — SaaS Multi-Tenant & Contratos REST", ["D01_SWE_CODE", "D02_SQL_DATA", "D03_BFCL_TOOLS"]),
    ("S07", "Softwares — Design-to-Code UI/SVG & Clean Arch", ["D01_SWE_CODE", "D09_VIDEO_SVG_916"]),
    ("S08", "Softwares — Microsserviços, Event Sourcing & SAGA", ["D01_SWE_CODE", "D06_SRE_K8S", "D02_SQL_DATA"]),
    ("S09", "Softwares — Automação CLI, Agentes & Tool-Calling", ["D03_BFCL_TOOLS", "D01_SWE_CODE", "D06_SRE_K8S"]),
    ("S10", "Ontologia — Leitura de Grafos Massivos (Long-Context)", ["D05_RULER_ONTOLOGY", "D04_IFEVAL_STRICT"]),
    ("S11", "Ontologia — Modelagem Formal OWL/RDF & SPARQL", ["D05_RULER_ONTOLOGY", "D02_SQL_DATA"]),
    ("S12", "Ontologia — Extração JSON-LD & Tool-Calling Semântico", ["D05_RULER_ONTOLOGY", "D03_BFCL_TOOLS"]),
    ("S13", "Ontologia — Inferência Causal Multi-Hop & Abstenção", ["D05_RULER_ONTOLOGY", "D08_GPQA_MATH"]),
    ("S14", "Artigos — Dossiês Executivos & ADRs Arquiteturais", ["D10_COPY_RECIPES", "D04_IFEVAL_STRICT", "D06_SRE_K8S"]),
    ("S15", "Artigos — Engenharia de Plataforma & SRE Shift-Left", ["D10_COPY_RECIPES", "D06_SRE_K8S"]),
    ("S16", "Artigos — Síntese Técnica Bilíngue & Benchmarks", ["D10_COPY_RECIPES", "D04_IFEVAL_STRICT"]),
    ("S17", "Artigos — Whitepapers Científicos, Clínica & SecOps", ["D08_GPQA_MATH", "D07_SECOPS_RE", "D10_COPY_RECIPES"]),
    ("S18", "Receitas — Gastronomia Molecular & Precisão Térmica", ["D10_COPY_RECIPES", "D08_GPQA_MATH"]),
    ("S19", "Receitas — Fichas Técnicas Estruturadas & Custos", ["D10_COPY_RECIPES", "D04_IFEVAL_STRICT"]),
    ("S20", "Receitas — Runbooks Operacionais Passo a Passo", ["D04_IFEVAL_STRICT", "D06_SRE_K8S", "D10_COPY_RECIPES"]),
    ("S21", "Receitas — Culinária Autoral, Emulsões & Harmonização", ["D10_COPY_RECIPES", "D04_IFEVAL_STRICT"]),
]


def load_all_shards() -> tuple[list[dict], list[dict]]:
    by_tag = {}
    shard_hosts = []
    for p in sorted(SUITE_DIR.glob("scenario_matrix_shard_*.json")):
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
            shard_hosts.append({"shard": raw.get("shard"), "file": p.name, "gpu_host": raw.get("gpu_host")})
            for m in raw.get("models", []):
                if m.get("status") == "OK" and m.get("scores"):
                    by_tag[m["tag"]] = m
                elif m.get("tag") not in by_tag:
                    by_tag[m["tag"]] = m
        except Exception as e:
            print(f"[WARN] Could not parse {p}: {e}")
    models = [m for m in by_tag.values() if m.get("status") == "OK" and m.get("scores")]
    models.sort(key=lambda x: (-float(x.get("routerbench_index", 0.0)), -float(x.get("avg_scenario_score", 0.0)), -float(x.get("avg_decode_tps", 0.0))))
    for idx, m in enumerate(models, 1):
        m["rank"] = idx
    return models, shard_hosts


def compute_champions(models: list[dict]) -> dict:
    champions = {}
    for sc_id, short_en, pt_title, standard in SCENARIO_DEFS:
        ranked_all = sorted(
            models,
            key=lambda m: (
                -float(m["scores"].get(sc_id, 0.0)),
                -float(m.get("routerbench_index", 0.0)),
                -float(m.get("avg_decode_tps", 0.0)),
            ),
        )
        ranked_local = [m for m in ranked_all if float(m.get("vram_gb", 99.0)) <= 16.0]
        qualified_speed = sorted(
            [m for m in models if float(m["scores"].get(sc_id, 0.0)) >= 70.0],
            key=lambda m: (-float(m.get("avg_decode_tps", 0.0)), -float(m["scores"].get(sc_id, 0.0))),
        )
        if not qualified_speed:
            qualified_speed = ranked_all

        def _fmt(item):
            if not item:
                return None
            return {
                "tag": item["tag"],
                "model": item["tag"],
                "short": item["short"],
                "display": item["short"],
                "score": float(item["scores"].get(sc_id, 0.0)),
                "decode_tps": float(item.get("avg_decode_tps", 0.0)),
                "vram_gb": float(item.get("vram_gb", 0.0)),
                "hw_tier": item.get("hw_tier", "colab_40gb"),
            }

        champions[sc_id] = {
            "id": sc_id,
            "short": short_en,
            "pt_title": pt_title,
            "standard": standard,
            "overall_top3": [_fmt(m) for m in ranked_all[:3]],
            "local_16gb_top3": [_fmt(m) for m in ranked_local[:3]],
            "speed_leader": _fmt(qualified_speed[0]) if qualified_speed else None,
        }
    return champions


def compute_jarvis_21_routing(models: list[dict]) -> list[dict]:
    routing = []
    for s_code, s_label, dom_ids in JARVIS_21_MAP:
        def _composite(m):
            vals = [float(m["scores"].get(d, 0.0)) for d in dom_ids]
            mean_v = sum(vals) / len(vals)
            tps_bonus = min(1.0, float(m.get("avg_decode_tps", 0.0)) / 150.0) * 4.0
            return round(mean_v + tps_bonus, 2), round(mean_v, 1)

        scored_all = sorted(models, key=lambda m: (-_composite(m)[0], -float(m.get("avg_decode_tps", 0.0))))
        scored_local = [m for m in scored_all if float(m.get("vram_gb", 99.0)) <= 16.0]

        best_all = scored_all[0] if scored_all else None
        best_loc = scored_local[0] if scored_local else None
        colab_rec = {
            "tag": best_all["tag"],
            "model": best_all["tag"],
            "short": best_all["short"],
            "display": best_all["short"],
            "domain_avg": _composite(best_all)[1],
            "score": _composite(best_all)[1],
            "decode_tps": best_all["avg_decode_tps"],
            "vram_gb": best_all["vram_gb"],
        } if best_all else None
        local_rec = {
            "tag": best_loc["tag"],
            "model": best_loc["tag"],
            "short": best_loc["short"],
            "display": best_loc["short"],
            "domain_avg": _composite(best_loc)[1],
            "score": _composite(best_loc)[1],
            "decode_tps": best_loc["avg_decode_tps"],
            "vram_gb": best_loc["vram_gb"],
        } if best_loc else None
        routing.append({
            "id": s_code,
            "scenario_code": s_code,
            "name": s_label,
            "title": s_label,
            "benchmark_domain": dom_ids[0],
            "benchmark_axes": dom_ids,
            "recommended_colab": colab_rec,
            "recommended_local_16gb": local_rec,
            "recommended_local_5080": local_rec,
        })
    return routing


def generate_heatmap_chart(models: list[dict], out_png: Path):
    if not models:
        return
    plt.rcParams.update({
        "figure.facecolor": "#0d1117",
        "axes.facecolor": "#161b22",
        "text.color": "#e6edf3",
        "axes.labelcolor": "#e6edf3",
        "xtick.color": "#c9d1d9",
        "ytick.color": "#e6edf3",
        "font.sans-serif": "DejaVu Sans",
    })

    sc_ids = [s[0] for s in SCENARIO_DEFS]
    sc_labels = [f"{s[0].split('_')[0]}\n{s[1]}" for s in SCENARIO_DEFS] + ["AVG\nScore", "Router\nIndex"]

    mat = []
    y_labels = []
    for m in models:
        row_vals = [float(m["scores"].get(sid, 0.0)) for sid in sc_ids]
        row_vals.append(float(m.get("avg_scenario_score", 0.0)))
        row_vals.append(float(m.get("routerbench_index", 0.0)))
        mat.append(row_vals)
        tier_mark = "≤16G" if float(m.get("vram_gb", 99.0)) <= 16.0 else f"{m.get('vram_gb')}G"
        y_labels.append(f"#{m['rank']:02d} {m['short']} ({tier_mark} · {m['avg_decode_tps']:.0f}t/s)")

    arr = np.array(mat, dtype=float)
    fig_h = max(11.0, 0.44 * len(models) + 2.6)
    fig, ax = plt.subplots(figsize=(18.5, fig_h), dpi=220)

    im = ax.imshow(arr, cmap="RdYlGn", vmin=0.0, vmax=100.0, aspect="auto")

    ax.set_xticks(np.arange(len(sc_labels)))
    ax.set_xticklabels(sc_labels, fontsize=9.5, fontweight="bold")
    ax.xaxis.tick_top()

    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=9.2, fontweight="bold")

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val = arr[i, j]
            txt_col = "#0d1117" if 32.0 <= val <= 88.0 else "#ffffff"
            if val >= 88.0:
                txt_col = "#062810"
            ax.text(
                j,
                i,
                f"{val:.0f}" if j < 10 else f"{val:.1f}",
                ha="center",
                va="center",
                color=txt_col,
                fontsize=8.5,
                fontweight="bold" if j >= 10 or val >= 90 else "normal",
            )

    # Vertical separator before summary columns
    ax.axvline(9.5, color="#f0f6fc", linewidth=2.2)

    cbar = fig.colorbar(im, ax=ax, fraction=0.022, pad=0.02)
    cbar.set_label("Deterministic Execution Score (0 – 100)", color="#e6edf3", fontsize=10, fontweight="bold")
    cbar.ax.yaxis.set_tick_params(color="#c9d1d9")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#c9d1d9")

    ax.set_title(
        f"LLM Scenario Specialization Matrix ({len(models)} Models × 10 SOTA Execution Benchmarks on Google Colab GPU)\n"
        "LiveCodeBench · BIRD-SQL · BFCL v3 · IFEval · RULER/OWL · Cloud SRE · HarmBench/OR-Bench · GPQA/MATH · 9:16 SVG · Humanizer",
        fontsize=12.5,
        fontweight="bold",
        pad=24,
        color="#f0f6fc",
    )

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220, facecolor=fig.get_facecolor())
    plt.close(fig)


def generate_champions_chart(champions: dict, out_png: Path):
    if not champions:
        return
    plt.rcParams.update({
        "figure.facecolor": "#0d1117",
        "axes.facecolor": "#161b22",
        "text.color": "#e6edf3",
        "axes.edgecolor": "#30363d",
    })
    sc_list = [champions[s[0]] for s in SCENARIO_DEFS if s[0] in champions]
    labels = [f"{c['id'].split('_')[0]} · {c['short']}" for c in sc_list][::-1]
    all_scores = [c["overall_top3"][0]["score"] if c["overall_top3"] else 0 for c in sc_list][::-1]
    all_names = [f"{c['overall_top3'][0]['short']} ({c['overall_top3'][0]['decode_tps']:.0f}t/s)" if c["overall_top3"] else "-" for c in sc_list][::-1]

    loc_scores = [c["local_16gb_top3"][0]["score"] if c["local_16gb_top3"] else 0 for c in sc_list][::-1]
    loc_names = [f"{c['local_16gb_top3'][0]['short']} ({c['local_16gb_top3'][0]['vram_gb']}G)" if c["local_16gb_top3"] else "-" for c in sc_list][::-1]

    y = np.arange(len(labels))
    h = 0.37
    fig, ax = plt.subplots(figsize=(17.5, 9.5), dpi=220)

    b1 = ax.barh(y + h / 2, all_scores, height=h, color="#2ea043", label="#1 Overall SOTA Champion (Colab Datacenter Tier)")
    b2 = ax.barh(y - h / 2, loc_scores, height=h, color="#58a6ff", label="#1 Local ≤16GB VRAM Champion (Workstation Tier)")

    for rect, nm, sc in zip(b1, all_names, all_scores):
        ax.text(min(sc + 1.2, 101.0), rect.get_y() + rect.get_height() / 2, f"{sc:.0f}% — {nm}", va="center", fontsize=8.8, color="#7ee787", fontweight="bold")
    for rect, nm, sc in zip(b2, loc_names, loc_scores):
        ax.text(min(sc + 1.2, 101.0), rect.get_y() + rect.get_height() / 2, f"{sc:.0f}% — {nm}", va="center", fontsize=8.6, color="#a5d6ff", fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10, fontweight="bold")
    ax.set_xlim(0, 148)
    ax.set_xlabel("Deterministic Execution Score (%)", fontsize=11, fontweight="bold")
    ax.set_title("Best LLM per Scenario — Datacenter Champion vs. Sub-16GB Local Champion (RouterBench Suite 08)", fontsize=13, fontweight="bold", pad=34)
    ax.grid(axis="x", linestyle="--", alpha=0.25, color="#8b949e")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.065), ncol=2, facecolor="#0d1117", edgecolor="#30363d", fontsize=9.5)

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    models, shard_hosts = load_all_shards()
    if not models:
        print("[WARN] No completed models found yet in scenario_matrix_shard_*.json")
        return

    # Enrich each model with human-readable specialty badges
    sc_title_map = {s[0]: s[2] for s in SCENARIO_DEFS}
    sc_short_map = {s[0]: s[1] for s in SCENARIO_DEFS}
    for m in models:
        ranked_sc = sorted(m["scores"].items(), key=lambda kv: (-float(kv[1]), kv[0]))
        m["model"] = m["tag"]
        m["display"] = m["short"]
        m["avg_score"] = m.get("avg_scenario_score", 0.0)
        m["top_3_scenarios"] = [k for k, _ in ranked_sc[:3]]
        m["weakest_2_scenarios"] = [k for k, _ in ranked_sc[-2:]]
        m["best_scenarios"] = [
            {"id": k, "short": sc_short_map.get(k, k), "pt_title": sc_title_map.get(k, k), "score": v}
            for k, v in ranked_sc[:3]
        ]

    champions = compute_champions(models)
    jarvis_routing = compute_jarvis_21_routing(models)

    consolidated = {
        "suite": "08_scenario_specialization_benchmark",
        "methodology": "2025-2026 Deterministic Execution & RouterBench Multi-Scenario Matrix (Zero LLM-as-a-Judge, 100% Colab GPU Execution)",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_models": len(models),
        "total_scenarios": len(SCENARIO_DEFS),
        "total_evaluations": len(models) * len(SCENARIO_DEFS),
        "total_live_evaluations": len(models) * len(SCENARIO_DEFS),
        "shard_hosts": shard_hosts,
        "scenarios": [
            {"id": s[0], "short": s[1], "name": s[2], "pt_title": s[2], "sota_ref": s[3], "standard": s[3]}
            for s in SCENARIO_DEFS
        ],
        "champions_by_scenario": champions,
        "jarvis_21_scenarios_routing": jarvis_routing,
        "models": models,
    }

    out_json = SUITE_DIR / "scenario_specialization_matrix.json"
    out_csv = SUITE_DIR / "scenario_specialization_matrix.csv"
    out_json.write_text(json.dumps(consolidated, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_cols = ["rank", "tag", "short", "params", "vram_gb", "hw_tier", "avg_scenario_score", "routerbench_index", "avg_decode_tps", "peak_decode_tps"] + [s[0] for s in SCENARIO_DEFS] + ["top_specialties"]
    with open(out_csv, "w", newline="", encoding="utf-8") as cf:
        w = csv.writer(cf)
        w.writerow(csv_cols)
        for m in models:
            row = [
                m["rank"],
                m["tag"],
                m["short"],
                m["params"],
                m["vram_gb"],
                m["hw_tier"],
                m["avg_scenario_score"],
                m["routerbench_index"],
                m["avg_decode_tps"],
                m["peak_decode_tps"],
            ] + [m["scores"].get(s[0], 0.0) for s in SCENARIO_DEFS] + [" | ".join(m.get("top_specialties", []))]
            w.writerow(row)

    chart4 = SUITE_DIR / "chart_04_scenario_specialization_heatmap.png"
    chart5 = SUITE_DIR / "chart_05_scenario_champions_board.png"
    generate_heatmap_chart(models, chart4)
    generate_champions_chart(champions, chart5)

    # Mirror to github_benchmark_repo if present
    if GITHUB_REPO_DIR.exists():
        (GITHUB_REPO_DIR / "data").mkdir(parents=True, exist_ok=True)
        (GITHUB_REPO_DIR / "assets").mkdir(parents=True, exist_ok=True)
        (GITHUB_REPO_DIR / "scripts").mkdir(parents=True, exist_ok=True)
        shutil.copy2(out_json, GITHUB_REPO_DIR / "data" / "scenario_specialization_matrix.json")
        shutil.copy2(out_csv, GITHUB_REPO_DIR / "data" / "scenario_specialization_matrix.csv")
        shutil.copy2(chart4, GITHUB_REPO_DIR / "assets" / "chart_04_scenario_specialization_heatmap.png")
        shutil.copy2(chart5, GITHUB_REPO_DIR / "assets" / "chart_05_scenario_champions_board.png")
        shutil.copy2(SUITE_DIR / "colab_scenario_matrix_benchmark.py", GITHUB_REPO_DIR / "scripts" / "colab_scenario_matrix_benchmark.py")

    print(f"[OK] Consolidated {len(models)} models across 10 scenarios ({len(models)*10} evaluations) -> {out_json}")


if __name__ == "__main__":
    main()
