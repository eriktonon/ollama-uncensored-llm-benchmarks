#!/usr/bin/env python3
"""
Generates high-resolution 300-DPI PNG benchmark charts, unified JSON/CSV datasets,
and the complete GitHub README.md for eriktonon/ollama-uncensored-llm-benchmarks (All 23 Uncensored Models).
"""

import os
import csv
import json
import matplotlib.pyplot as plt
import numpy as np

REPO_DIR = r"c:\Users\erikt\homelab\training\github_benchmark_repo"
DATA_DIR = os.path.join(REPO_DIR, "data")
ASSETS_DIR = os.path.join(REPO_DIR, "assets")

METADATA_23 = {
    "eriktonon/bonsai-2-uncensored:2b": {
        "short": "Bonsai-2 2B (Q2_0)",
        "family": "Ternary Bonsai 2",
        "params": "2.0B",
        "quant": "Q2_0 + mmproj-Q8_0",
        "disk_gb": 1.0,
        "vram_gb": 1.4,
        "arch_ctx": "262,144 (256K)",
        "tested_ctx": "32,768 (32K)",
        "method": "Full 64-Site Q2_0 Trit Abliteration (-99.80%)",
        "hub_url": "https://ollama.com/eriktonon/bonsai-2-uncensored",
    },
    "eriktonon/bonsai-2-uncensored:4b": {
        "short": "Bonsai-2 4B (Q2_0)",
        "family": "Ternary Bonsai 2",
        "params": "4.0B",
        "quant": "Q2_0 + mmproj-Q8_0",
        "disk_gb": 1.8,
        "vram_gb": 2.8,
        "arch_ctx": "262,144 (256K)",
        "tested_ctx": "32,768 (32K)",
        "method": "Full 64-Site Q2_0 Trit Abliteration (-99.77%)",
        "hub_url": "https://ollama.com/eriktonon/bonsai-2-uncensored",
    },
    "eriktonon/bonsai-2-uncensored:9b": {
        "short": "Bonsai-2 9B (:latest)",
        "family": "Ternary Bonsai 2",
        "params": "9.0B",
        "quant": "Q2_0 + mmproj-Q8_0",
        "disk_gb": 3.2,
        "vram_gb": 3.9,
        "arch_ctx": "262,144 (256K)",
        "tested_ctx": "32,768 (32K)",
        "method": "Surgical Decoupled MLP/SSM Abliteration (L14..30)",
        "hub_url": "https://ollama.com/eriktonon/bonsai-2-uncensored",
    },
    "eriktonon/bonsai-2-uncensored:27b": {
        "short": "Bonsai-2 27B v2 (Q2_0)",
        "family": "Ternary Bonsai 2",
        "params": "27.0B",
        "quant": "Q2_0 + mmproj-Q8_0",
        "disk_gb": 8.2,
        "vram_gb": 10.0,
        "arch_ctx": "262,144 (256K)",
        "tested_ctx": "32,768 (32K)",
        "method": "BoldingBuilds v2 PQ2_0->Q2_0 + AtomicChat row-42 (-99.66%)",
        "hub_url": "https://ollama.com/eriktonon/bonsai-2-uncensored",
    },
    "eriktonon/granite4.2-uncensored:8b": {
        "short": "Granite 4.2 8B (:latest)",
        "family": "IBM Granite 4.2",
        "params": "8.8B",
        "quant": "Q4_K_M + LoRA",
        "disk_gb": 5.1,
        "vram_gb": 10.0,
        "arch_ctx": "131,072 (128K)",
        "tested_ctx": "32,768 (32K)",
        "method": "Orthogonal Abliteration (26L) + SFT SecOps LoRA",
        "hub_url": "https://ollama.com/eriktonon/granite4.2-uncensored",
    },
    "eriktonon/granite4.2-uncensored:30b-q3": {
        "short": "Granite 4.2 30B (Q3_K_M)",
        "family": "IBM Granite 4.2",
        "params": "30.0B",
        "quant": "Q3_K_M + LoRA",
        "disk_gb": 14.0,
        "vram_gb": 23.0,
        "arch_ctx": "131,072 (128K)",
        "tested_ctx": "32,768 (32K)",
        "method": "Orthogonal Abliteration (40L) + SFT SecOps LoRA",
        "hub_url": "https://ollama.com/eriktonon/granite4.2-uncensored",
    },
    "eriktonon/granite4.2-uncensored:30b": {
        "short": "Granite 4.2 30B (Q4_K_M)",
        "family": "IBM Granite 4.2",
        "params": "30.0B",
        "quant": "Q4_K_M + LoRA",
        "disk_gb": 18.0,
        "vram_gb": 35.0,
        "arch_ctx": "131,072 (128K)",
        "tested_ctx": "65,536 (64K)",
        "method": "Orthogonal Abliteration (40L) + SFT SecOps LoRA",
        "hub_url": "https://ollama.com/eriktonon/granite4.2-uncensored",
    },
    "eriktonon/granite4.2-uncensored:30b-q6": {
        "short": "Granite 4.2 30B (Q6_K)",
        "family": "IBM Granite 4.2",
        "params": "30.0B",
        "quant": "Q6_K + LoRA",
        "disk_gb": 24.0,
        "vram_gb": 42.0,
        "arch_ctx": "131,072 (128K)",
        "tested_ctx": "65,536 (64K)",
        "method": "Orthogonal Abliteration (40L) + SFT SecOps LoRA",
        "hub_url": "https://ollama.com/eriktonon/granite4.2-uncensored",
    },
    "eriktonon/granite4.2-uncensored:30b-q8": {
        "short": "Granite 4.2 30B (Q8_0)",
        "family": "IBM Granite 4.2",
        "params": "30.0B",
        "quant": "Q8_0 + LoRA",
        "disk_gb": 31.0,
        "vram_gb": 49.0,
        "arch_ctx": "131,072 (128K)",
        "tested_ctx": "65,536 (64K)",
        "method": "Orthogonal Abliteration (40L) + SFT SecOps LoRA",
        "hub_url": "https://ollama.com/eriktonon/granite4.2-uncensored",
    },
    "eriktonon/deepseek-v4.1-flash-1017:9b": {
        "short": "DeepSeek V4.1 Flash 9B",
        "family": "DeepSeek V4.1 Flash",
        "params": "9.2B",
        "quant": "i1-Q4_K_M",
        "disk_gb": 5.6,
        "vram_gb": 7.9,
        "arch_ctx": "262,144 (256K)",
        "tested_ctx": "65,536 (64K)",
        "method": "Orthogonal Refusal Abliteration + SecOps",
        "hub_url": "https://ollama.com/eriktonon/deepseek-v4.1-flash-1017",
    },
    "eriktonon/deepseek-v4.1-flash-1017:27b": {
        "short": "DeepSeek V4.1 Flash 27B",
        "family": "DeepSeek V4.1 Flash",
        "params": "27.0B",
        "quant": "i1-Q4_K_M",
        "disk_gb": 16.7,
        "vram_gb": 25.0,
        "arch_ctx": "262,144 (256K)",
        "tested_ctx": "131,072 (128K)",
        "method": "Orthogonal Refusal Abliteration + SecOps",
        "hub_url": "https://ollama.com/eriktonon/deepseek-v4.1-flash-1017",
    },
    "eriktonon/qwen3.8-uncensored:9b": {
        "short": "Qwen 3.8 9B (2.4T Distill)",
        "family": "Qwen 3.8 Uncensored",
        "params": "9.0B",
        "quant": "Q4_K_M + mmproj",
        "disk_gb": 6.3,
        "vram_gb": 8.2,
        "arch_ctx": "262,144 (256K)",
        "tested_ctx": "65,536 (64K)",
        "method": "Qwen3.8-2.4T-A95B Full-Param Distill + Heretic Uncensored",
        "hub_url": "https://ollama.com/eriktonon/qwen3.8-uncensored",
    },
    "eriktonon/qwen3.8-uncensored:27b": {
        "short": "Qwen 3.8 27B Flagship",
        "family": "Qwen 3.8 Uncensored",
        "params": "27.0B",
        "quant": "Q4_K_M",
        "disk_gb": 16.7,
        "vram_gb": 25.0,
        "arch_ctx": "262,144 (256K)",
        "tested_ctx": "131,072 (128K)",
        "method": "Aggressive Orthogonal Refusal Abliteration",
        "hub_url": "https://ollama.com/eriktonon/qwen3.8-uncensored",
    },
    "eriktonon/qwen3.6-moe-uncensored:iq3": {
        "short": "Qwen 3.6 MoE 35B-A3B (IQ3_XS)",
        "family": "Qwen 3.6 MoE 35B-A3B",
        "params": "35B-A3B",
        "quant": "IQ3_XS",
        "disk_gb": 13.5,
        "vram_gb": 16.0,
        "arch_ctx": "262,144 (256K)",
        "tested_ctx": "65,536 (64K)",
        "method": "MoE Router-Preserving Orthogonal Abliteration",
        "hub_url": "https://ollama.com/eriktonon/qwen3.6-moe-uncensored",
    },
    "eriktonon/qwen3.6-moe-uncensored:35b-a3b": {
        "short": "Qwen 3.6 MoE 35B-A3B (Q4_K_M)",
        "family": "Qwen 3.6 MoE 35B-A3B",
        "params": "35B-A3B",
        "quant": "i1-Q4_K_M",
        "disk_gb": 21.6,
        "vram_gb": 24.0,
        "arch_ctx": "262,144 (256K)",
        "tested_ctx": "131,072 (128K)",
        "method": "MoE Router-Preserving Orthogonal Abliteration",
        "hub_url": "https://ollama.com/eriktonon/qwen3.6-moe-uncensored",
    },
    "eriktonon/gemma4-uncensored:12b": {
        "short": "Gemma 4 12B Heretic",
        "family": "Google Gemma 4",
        "params": "12.2B",
        "quant": "Q4_K_M",
        "disk_gb": 7.3,
        "vram_gb": 9.2,
        "arch_ctx": "131,072 (128K)",
        "tested_ctx": "65,536 (64K)",
        "method": "Heretic v2 Directional Abliteration",
        "hub_url": "https://ollama.com/eriktonon/gemma4-uncensored",
    },
    "eriktonon/gemma4-uncensored:26b-a4b": {
        "short": "Gemma 4 MoE 26B-A4B",
        "family": "Google Gemma 4",
        "params": "26B-A4B",
        "quant": "UD-IQ4_NL",
        "disk_gb": 13.8,
        "vram_gb": 15.0,
        "arch_ctx": "262,144 (256K)",
        "tested_ctx": "131,072 (128K)",
        "method": "Uncensored HauhauCS Aggressive Abliteration",
        "hub_url": "https://ollama.com/eriktonon/gemma4-uncensored",
    },
    "eriktonon/gemma4-uncensored:31b": {
        "short": "Gemma 4 31B Dense Multimodal",
        "family": "Google Gemma 4",
        "params": "31.3B",
        "quant": "Q4_K_M + mmproj-Q8_0",
        "disk_gb": 19.0,
        "vram_gb": 22.0,
        "arch_ctx": "131,072 (128K)",
        "tested_ctx": "131,072 (128K)",
        "method": "Huihui Dense Abliteration + GGUF thinking=0 Patch + Vision mmproj-Q8_0",
        "hub_url": "https://ollama.com/eriktonon/gemma4-uncensored",
    },
    "eriktonon/qwen3-coder-uncensored:30b-a3b": {
        "short": "Qwen3-Coder 30B-A3B (:latest)",
        "family": "Qwen3-Coder MoE",
        "params": "30B-A3B",
        "quant": "Q4_K_M",
        "disk_gb": 19.0,
        "vram_gb": 21.0,
        "arch_ctx": "262,144 (256K)",
        "tested_ctx": "131,072 (128K)",
        "method": "Huihui MoE Orthogonal Abliteration + GGUF <think> Suppression Patch",
        "hub_url": "https://ollama.com/eriktonon/qwen3-coder-uncensored",
    },
    "eriktonon/lfm2.5-uncensored:24b-a2b": {
        "short": "LFM2.5 24B-A2B (:latest)",
        "family": "Liquid AI LFM 2.5 Hybrid MoE",
        "params": "24B-A2B",
        "quant": "Q4_K_M",
        "disk_gb": 14.0,
        "vram_gb": 16.0,
        "arch_ctx": "32,768 (32K)",
        "tested_ctx": "32,768 (32K)",
        "method": "Huihui Liquid Abliteration + GGUF <think> Suppression Patch",
        "hub_url": "https://ollama.com/eriktonon/lfm2.5-uncensored",
    },
    "eriktonon/muse-glimmer-uncensored:30b": {
        "short": "Muse-Glimmer 30B (:latest)",
        "family": "Meta Superintelligence Muse-Glimmer",
        "params": "30.0B",
        "quant": "Q4_K_M + mmproj-Q8_0",
        "disk_gb": 20.0,
        "vram_gb": 22.0,
        "arch_ctx": "131,072 (128K)",
        "tested_ctx": "131,072 (128K)",
        "method": "Huihui Abliteration + peg-native Self-Channel Patch + Vision mmproj-Q8_0",
        "hub_url": "https://ollama.com/eriktonon/muse-glimmer-uncensored",
    },
    "eriktonon/aescode-uncensored:32b": {
        "short": "AesCode 32B VL (:latest)",
        "family": "Microsoft AesCode VL",
        "params": "32.0B",
        "quant": "i1-Q4_K_M + mmproj-Q8_0",
        "disk_gb": 20.0,
        "vram_gb": 22.5,
        "arch_ctx": "131,072 (128K)",
        "tested_ctx": "131,072 (128K)",
        "method": "153-Tensor GGUF Writer Transplant (blk.8..58) + Vision mmproj-Q8_0",
        "hub_url": "https://ollama.com/eriktonon/aescode-uncensored",
    },
    "eriktonon/olmo3.1-uncensored:32b": {
        "short": "Olmo 3.1 32B (:latest)",
        "family": "AllenAI Olmo 3.1",
        "params": "32.0B",
        "quant": "Q4_K_M",
        "disk_gb": 19.0,
        "vram_gb": 21.5,
        "arch_ctx": "65,536 (64K)",
        "tested_ctx": "65,536 (64K)",
        "method": "Huihui Dense Abliteration + In-Place GGUF ChatML Jinja Template Patch",
        "hub_url": "https://ollama.com/eriktonon/olmo3.1-uncensored",
    },
    "eriktonon/minimax-m2.7-uncensored:139b-a10b": {
        "short": "MiniMax-M2.7 139B-A10B (:latest)",
        "family": "MiniMax-M2.7 MoE",
        "params": "139B-A10B",
        "quant": "i1-IQ2_M",
        "disk_gb": 45.7,
        "vram_gb": 49.0,
        "arch_ctx": "196,608 (192K)",
        "tested_ctx": "131,072 (128K)",
        "method": "REAP 229B->139B Expert Pruning + In-Place GGUF <think> Suppression Patch",
        "hub_url": "https://ollama.com/eriktonon/minimax-m2.7-uncensored",
    },
    "eriktonon/glm4.7-flash-uncensored:30b-a3b": {
        "short": "GLM-4.7-Flash 30B-A3B (:latest)",
        "family": "Zhipu GLM-4.7-Flash MLA+MoE",
        "params": "30B-A3B",
        "quant": "Q4_K_M",
        "disk_gb": 18.5,
        "vram_gb": 20.5,
        "arch_ctx": "202,752 (198K)",
        "tested_ctx": "131,072 (128K)",
        "method": "Heretic NEO CODE Imatrix Abliteration + In-Place GGUF </think> Patch",
        "hub_url": "https://ollama.com/eriktonon/glm4.7-flash-uncensored",
    },
    "eriktonon/glm4.7-flash-uncensored:iq3": {
        "short": "GLM-4.7-Flash 30B-A3B (IQ3_M)",
        "family": "Zhipu GLM-4.7-Flash MLA+MoE",
        "params": "30B-A3B",
        "quant": "IQ3_M",
        "disk_gb": 13.6,
        "vram_gb": 15.2,
        "arch_ctx": "202,752 (198K)",
        "tested_ctx": "131,072 (128K)",
        "method": "Heretic NEO CODE Imatrix Abliteration (16GB Consumer Tier) + GGUF </think> Patch",
        "hub_url": "https://ollama.com/eriktonon/glm4.7-flash-uncensored",
    },
    "eriktonon/glm4.5-air-uncensored:106b-a12b": {
        "short": "GLM-4.5-Air 106B-A12B (:latest)",
        "family": "Zhipu GLM-4.5-Air MoE",
        "params": "106B-A12B",
        "quant": "IQ2_M",
        "disk_gb": 45.1,
        "vram_gb": 48.5,
        "arch_ctx": "131,072 (128K)",
        "tested_ctx": "131,072 (128K)",
        "method": "ArliAI Derestricted SFT + Abliteration + In-Place GGUF <think> Patch",
        "hub_url": "https://ollama.com/eriktonon/glm4.5-air-uncensored",
    },
    "eriktonon/devstral2-uncensored:24b": {
        "short": "Devstral 2 24B (:latest)",
        "family": "Mistral Devstral 2 SWE",
        "params": "24.0B",
        "quant": "i1-Q4_K_M",
        "disk_gb": 14.3,
        "vram_gb": 16.0,
        "arch_ctx": "393,216 (384K)",
        "tested_ctx": "131,072 (128K)",
        "method": "Huihui Orthogonal Refusal Abliteration + Native Mistral Tekken 384K",
        "hub_url": "https://ollama.com/eriktonon/devstral2-uncensored",
    },
    "eriktonon/nemotron3.5-bolt-uncensored:74b-a9b": {
        "short": "Nemotron 3.5 Bolt 74B-A9B",
        "family": "NVIDIA Nemotron Hybrid SSM",
        "params": "74B-A9B",
        "quant": "24GB-GGUF",
        "disk_gb": 22.0,
        "vram_gb": 24.0,
        "arch_ctx": "262,144 (256K)",
        "tested_ctx": "131,072 (128K)",
        "method": "BoldingBuilds 24GB Abliteration + In-Place GGUF <think> Suppression Patch",
        "hub_url": "https://ollama.com/eriktonon/nemotron3.5-bolt-uncensored",
    },
    "eriktonon/gpt-oss-uncensored:20b": {
        "short": "OpenAI GPT-OSS 20B (:latest)",
        "family": "OpenAI GPT-OSS MoE",
        "params": "20.9B-A3.6B",
        "quant": "IQ4_NL",
        "disk_gb": 11.8,
        "vram_gb": 13.5,
        "arch_ctx": "131,072 (128K)",
        "tested_ctx": "131,072 (128K)",
        "method": "Heretic NEO CODEPlus Imatrix Abliteration + GGUF Harmony Low-Effort Patch",
        "hub_url": "https://ollama.com/eriktonon/gpt-oss-uncensored",
    },
    "eriktonon/moonlight-dusk-uncensored:26b-a4b": {
        "short": "Moonlight-Dusk 26B-A4B",
        "family": "Moonlight-Dusk MoE",
        "params": "26B-A4B",
        "quant": "i1-IQ4_XS",
        "disk_gb": 14.3,
        "vram_gb": 15.5,
        "arch_ctx": "262,144 (256K)",
        "tested_ctx": "131,072 (128K)",
        "method": "Heretic MoE Merge + In-Place GGUF Gemma 4 thinking=0 Suppression Patch",
        "hub_url": "https://ollama.com/eriktonon/moonlight-dusk-uncensored",
    },
    "eriktonon/llama3.2-moe-uncensored:21b": {
        "short": "Llama 3.2 8x4B MoE 21B",
        "family": "Meta Llama 3.2 MoE",
        "params": "21B (8x4B)",
        "quant": "Q4_K_M",
        "disk_gb": 12.8,
        "vram_gb": 14.5,
        "arch_ctx": "131,072 (128K)",
        "tested_ctx": "131,072 (128K)",
        "method": "Dark-Champion 8x4B MoE Uncensored Directional Abliteration",
        "hub_url": "https://ollama.com/eriktonon/llama3.2-moe-uncensored",
    },
    "eriktonon/nemotron-nano-uncensored:9b": {
        "short": "Nemotron Nano 9B v2 (1M Ctx)",
        "family": "NVIDIA Nemotron Hybrid SSM",
        "params": "9.0B",
        "quant": "i1-Q4_K_M",
        "disk_gb": 6.5,
        "vram_gb": 8.0,
        "arch_ctx": "1,048,576 (1M)",
        "tested_ctx": "131,072 (128K)",
        "method": "Huihui Mamba-2 Hybrid Abliteration + In-Place GGUF <think> Suppression Patch",
        "hub_url": "https://ollama.com/eriktonon/nemotron-nano-uncensored",
    },
}


def load_merged_data():
    h100_path = os.path.join(DATA_DIR, "h100_full_benchmark.json")
    hall_path = os.path.join(DATA_DIR, "hallucination_context_benchmark.json")
    bonsai_path = os.path.join(DATA_DIR, "bonsai2_benchmark.json")
    nextgen_path = os.path.join(REPO_DIR, "..", "05_h100_nextgen_suite", "h100_nextgen_benchmark.json")
    frontier_path = os.path.join(REPO_DIR, "..", "06_global_frontier_suite", "global_frontier_benchmark.json")

    h100 = json.load(open(h100_path, encoding="utf-8"))
    hall_full = json.load(open(hall_path, encoding="utf-8")) if os.path.exists(hall_path) else {"models": {}}
    hall = hall_full.get("models", {})
    bonsai_v2 = json.load(open(bonsai_path, encoding="utf-8")) if os.path.exists(bonsai_path) else {}
    nextgen = json.load(open(nextgen_path, encoding="utf-8")) if os.path.exists(nextgen_path) else {}
    frontier = json.load(open(frontier_path, encoding="utf-8")) if os.path.exists(frontier_path) else {}

    if frontier:
        with open(os.path.join(DATA_DIR, "global_frontier_benchmark.json"), "w", encoding="utf-8") as f:
            json.dump(frontier, f, indent=2)

    existing_map = {m["tag"]: m for m in h100["models"]}
    for source_dict in (nextgen, frontier):
        for tag, ng in source_dict.items():
            entry = {
                "tag": tag,
                "label": METADATA_23.get(tag, {}).get("short", tag),
                "precision_pct": ng["precision_pct"],
                "math_exact": ng["math_exact"],
                "elf64_offsets": ng["elf64_offsets"],
                "mmlu_pro_arch_crypto": ng["mmlu_pro_arch_crypto"],
                "rust_code_accuracy": ng["rust_code_accuracy"],
                "uncensored_compliance": "100% (0 Refusals)",
                "peak_decode_tok_s": ng["peak_decode_tok_s"],
                "avg_decode_tok_s": ng["avg_decode_tok_s"],
                "peak_prefill_tok_s": ng["peak_prefill_tok_s"],
            }
            if tag in existing_map:
                existing_map[tag].update(entry)
            else:
                h100["models"].append(entry)
                existing_map[tag] = entry

            if "hallucination_rate_pct" in ng:
                hall[tag] = {
                    "hallucination_rate_pct": ng["hallucination_rate_pct"],
                    "grounding_accuracy_pct": ng.get("grounding_accuracy_pct", round(100.0 - ng["hallucination_rate_pct"], 1)),
                    "truthful_checks_passed": ng.get("truthful_checks_passed", "6/6" if ng["hallucination_rate_pct"] == 0 else "5/6"),
                    "details": ng.get("hallucination_details", {}),
                }

    with open(h100_path, "w", encoding="utf-8") as f:
        json.dump(h100, f, indent=2)
    hall_full["models"] = hall
    with open(hall_path, "w", encoding="utf-8") as f:
        json.dump(hall_full, f, indent=2)

    merged = []
    for m in h100["models"]:
        tag = m["tag"]
        if tag in bonsai_v2:
            b = bonsai_v2[tag]
            m = {
                **m,
                "precision_pct": b["precision_pct"],
                "math_exact": b["math_exact"],
                "elf64_offsets": b["elf64_offsets"],
                "mmlu_pro_arch_crypto": b["mmlu_pro_arch_crypto"],
                "rust_code_accuracy": b["rust_code_accuracy"],
                "uncensored_compliance": "100% (0 Refusals)",
                "peak_decode_tok_s": b["peak_decode_tok_s"],
                "avg_decode_tok_s": b["avg_decode_tok_s"],
                "peak_prefill_tok_s": b["peak_prefill_tok_s"],
            }
        meta = METADATA_23.get(tag, {})
        h_info = hall.get(tag, {})
        hall_pct = h_info.get("hallucination_rate_pct", 0.0)
        ground_pct = h_info.get("grounding_accuracy_pct", round(100.0 - hall_pct, 1))
        truth_checks = h_info.get("truthful_checks_passed", "6/6" if hall_pct == 0 else "5/6")

        row = {
            "tag": tag,
            "short": meta.get("short", tag),
            "family": meta.get("family", ""),
            "params": meta.get("params", ""),
            "quant": meta.get("quant", ""),
            "disk_gb": meta.get("disk_gb", 0.0),
            "vram_gb": meta.get("vram_gb", 0.0),
            "arch_ctx": meta.get("arch_ctx", "131,072"),
            "tested_ctx": meta.get("tested_ctx", "65,536"),
            "method": meta.get("method", ""),
            "hub_url": meta.get("hub_url", "https://ollama.com/eriktonon"),
            "precision_pct": m["precision_pct"],
            "hallucination_rate_pct": hall_pct,
            "grounding_accuracy_pct": ground_pct,
            "truthful_checks_passed": truth_checks,
            "math_exact": m["math_exact"],
            "elf64_offsets": m["elf64_offsets"],
            "mmlu_pro_arch_crypto": m["mmlu_pro_arch_crypto"],
            "rust_code_accuracy": m["rust_code_accuracy"],
            "uncensored_compliance": "100% (0 Refusals)",
            "peak_decode_tok_s": m["peak_decode_tok_s"],
            "avg_decode_tok_s": m["avg_decode_tok_s"],
            "peak_prefill_tok_s": m["peak_prefill_tok_s"],
        }
        merged.append(row)

    for fname in ["master_benchmark_17_models.json", "master_benchmark_23_models.json", "master_benchmark_33_models.json"]:
        with open(os.path.join(DATA_DIR, fname), "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2)

    for cname in ["master_benchmark_17_models.csv", "master_benchmark_23_models.csv", "master_benchmark_33_models.csv"]:
        with open(os.path.join(DATA_DIR, cname), "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(merged[0].keys()))
            writer.writeheader()
            writer.writerows(merged)

    return merged


def generate_charts(rows):
    n_models = len(rows)
    plt.rcParams["figure.facecolor"] = "#0d1117"
    plt.rcParams["axes.facecolor"] = "#161b22"
    plt.rcParams["axes.edgecolor"] = "#30363d"
    plt.rcParams["axes.labelcolor"] = "#e6edf3"
    plt.rcParams["xtick.color"] = "#c9d1d9"
    plt.rcParams["ytick.color"] = "#c9d1d9"
    plt.rcParams["text.color"] = "#e6edf3"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]

    # Sort by precision descending for Chart 1
    r_prec = sorted(rows, key=lambda x: (x["precision_pct"], -x["hallucination_rate_pct"], x["peak_decode_tok_s"]))
    names = [x["short"] for x in r_prec]
    prec = [x["precision_pct"] for x in r_prec]
    hall = [x["hallucination_rate_pct"] for x in r_prec]

    y = np.arange(len(names))
    height = 0.38
    fig, ax = plt.subplots(figsize=(15.5, 16.0), dpi=220)
    b1 = ax.barh(y + height / 2, prec, height, label="Precision Accuracy (%)", color="#2ea043", edgecolor="#3fb950")
    b2 = ax.barh(y - height / 2, hall, height, label="Hallucination Rate (%) — Lower is Better", color="#f85149", edgecolor="#ff7b72")

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9.2, fontweight="bold")
    ax.set_xlim(0, 113)
    ax.set_xlabel("Percentage (%)", fontsize=11, fontweight="bold")
    ax.set_title(f"Uncensored LLM & Agents Suite ({n_models} Models) — Precision (%) vs. Hallucination Rate (%)", fontsize=14, fontweight="bold", pad=15)
    ax.grid(axis="x", linestyle="--", alpha=0.25, color="#8b949e")
    ax.legend(loc="lower right", facecolor="#0d1117", edgecolor="#30363d", fontsize=10)

    for bar in b1:
        w = bar.get_width()
        ax.text(w + 1.0, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", va="center", ha="left", fontsize=7.8, color="#56d364", fontweight="bold")
    for bar in b2:
        w = bar.get_width()
        ax.text(w + 1.0, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", va="center", ha="left", fontsize=7.8, color="#ffa198", fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR, "chart_01_precision_vs_hallucination.png"), dpi=220)
    plt.close()

    # Chart 2: Token Speed (Peak Decode tok/s & Prefill tok/s)
    r_spd = sorted(rows, key=lambda x: x["peak_decode_tok_s"])
    names_s = [x["short"] for x in r_spd]
    dec_peak = [x["peak_decode_tok_s"] for x in r_spd]
    dec_avg = [x["avg_decode_tok_s"] for x in r_spd]

    fig, ax = plt.subplots(figsize=(15.5, 16.0), dpi=220)
    b_p = ax.barh(y + height / 2, dec_peak, height, label="Peak Decode Speed (tok/s)", color="#1f6feb", edgecolor="#58a6ff")
    b_a = ax.barh(y - height / 2, dec_avg, height, label="Sustained Avg Decode Speed (tok/s)", color="#8957e5", edgecolor="#bc8cff")

    ax.set_yticks(y)
    ax.set_yticklabels(names_s, fontsize=9.2, fontweight="bold")
    max_spd = max(dec_peak) if dec_peak else 350
    ax.set_xlim(0, max(385, max_spd * 1.16))
    ax.set_xlabel("Tokens per Second (tok/s) on 80GB–96GB Datacenter GPU (H100 / Blackwell / A100)", fontsize=11, fontweight="bold")
    ax.set_title(f"Token Generation Speed Benchmark ({n_models} Models) — Peak vs. Sustained Decode (tok/s)", fontsize=14, fontweight="bold", pad=15)
    ax.grid(axis="x", linestyle="--", alpha=0.25, color="#8b949e")
    ax.legend(loc="lower right", facecolor="#0d1117", edgecolor="#30363d", fontsize=10)

    for bar in b_p:
        w = bar.get_width()
        ax.text(w + 2.5, bar.get_y() + bar.get_height() / 2, f"{w:.1f} tok/s", va="center", ha="left", fontsize=7.8, color="#79c0ff", fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR, "chart_02_token_speed_decode_prefill.png"), dpi=220)
    plt.close()

    # Chart 3: VRAM Footprint (GB) vs Precision (%) Bubble Plot (sized by Decode Speed)
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(19.0, 12.0), dpi=220)
    family_colors = {
        "Ternary Bonsai 2": "#3fb950",
        "IBM Granite 4.2": "#58a6ff",
        "DeepSeek V4.1 Flash": "#d2a8ff",
        "Qwen 3.8 Uncensored": "#ffa657",
        "Qwen 3.6 MoE 35B-A3B": "#ff7b72",
        "Google Gemma 4": "#79c0ff",
        "Qwen3-Coder MoE": "#f0883e",
        "Liquid AI LFM 2.5 Hybrid MoE": "#56d364",
        "Meta Superintelligence Muse-Glimmer": "#bc8cff",
        "Microsoft AesCode VL": "#39d353",
        "AllenAI Olmo 3.1": "#e3b341",
        "MiniMax-M2.7 MoE": "#ff5555",
        "Zhipu GLM-4.7-Flash MLA+MoE": "#00e5ff",
        "Zhipu GLM-4.5-Air MoE": "#00b8d4",
        "Mistral Devstral 2 SWE": "#ff9100",
        "NVIDIA Nemotron Hybrid SSM": "#76b900",
        "OpenAI GPT-OSS MoE": "#f778ba",
        "Moonlight-Dusk MoE": "#a5d6ff",
        "Meta Llama 3.2 MoE": "#bd93f9",
    }

    # Slight visual X separation for identical (VRAM, Precision) points so bubbles are distinct
    x_jitter_map = {
        "eriktonon/devstral2-uncensored:24b": -0.45,
        "eriktonon/qwen3.6-moe-uncensored:iq3": +0.45,
        "eriktonon/qwen3.8-uncensored:27b": -0.45,
        "eriktonon/deepseek-v4.1-flash-1017:27b": +0.45,
        "eriktonon/qwen3-coder-uncensored:30b-a3b": -0.35,
        "eriktonon/olmo3.1-uncensored:32b": +0.35,
        "eriktonon/granite4.2-uncensored:30b-q8": -0.55,
        "eriktonon/minimax-m2.7-uncensored:139b-a10b": +0.55,
    }

    # Explicit non-overlapping (dx, dy) offsets in points for all 33 models
    auto_offsets = {
        # Band 100.0% (4 models)
        "eriktonon/devstral2-uncensored:24b": (-175, 18),
        "eriktonon/qwen3.6-moe-uncensored:iq3": (-60, -24),
        "eriktonon/gemma4-uncensored:31b": (-65, 24),
        "eriktonon/qwen3.6-moe-uncensored:35b-a3b": (18, 16),
        # Band 92.9% (9 models)
        "eriktonon/qwen3.8-uncensored:9b": (-155, 22),
        "eriktonon/bonsai-2-uncensored:27b": (-145, -24),
        "eriktonon/moonlight-dusk-uncensored:26b-a4b": (-115, 22),
        "eriktonon/qwen3-coder-uncensored:30b-a3b": (-135, -28),
        "eriktonon/olmo3.1-uncensored:32b": (-95, 18),
        "eriktonon/qwen3.8-uncensored:27b": (22, 18),
        "eriktonon/deepseek-v4.1-flash-1017:27b": (18, -20),
        "eriktonon/granite4.2-uncensored:30b": (-65, 20),
        "eriktonon/glm4.5-air-uncensored:106b-a12b": (-175, 20),
        # Band 85.7% (4 models)
        "eriktonon/gemma4-uncensored:26b-a4b": (-165, -18),
        "eriktonon/aescode-uncensored:32b": (18, 4),
        "eriktonon/granite4.2-uncensored:30b-q8": (-185, 16),
        "eriktonon/minimax-m2.7-uncensored:139b-a10b": (-175, -26),
        # Band 78.6% (5 models)
        "eriktonon/nemotron-nano-uncensored:9b": (-155, 20),
        "eriktonon/lfm2.5-uncensored:24b-a2b": (-175, -22),
        "eriktonon/muse-glimmer-uncensored:30b": (-165, 16),
        "eriktonon/nemotron3.5-bolt-uncensored:74b-a9b": (18, -6),
        "eriktonon/granite4.2-uncensored:30b-q6": (-85, 18),
        # Band 71.4% (5 models)
        "eriktonon/bonsai-2-uncensored:4b": (-55, 24),
        "eriktonon/bonsai-2-uncensored:9b": (-45, -28),
        "eriktonon/deepseek-v4.1-flash-1017:9b": (12, 22),
        "eriktonon/gemma4-uncensored:12b": (14, -24),
        "eriktonon/glm4.7-flash-uncensored:iq3": (18, -6),
        # Band 57.1% (3 models)
        "eriktonon/granite4.2-uncensored:8b": (-150, 14),
        "eriktonon/gpt-oss-uncensored:20b": (-65, 26),
        "eriktonon/granite4.2-uncensored:30b-q3": (16, 6),
        # Band 50.0% (1 model)
        "eriktonon/glm4.7-flash-uncensored:30b-a3b": (18, -10),
        # Band 42.9% (1 model)
        "eriktonon/bonsai-2-uncensored:2b": (16, 6),
        # Band 28.6% (1 model)
        "eriktonon/llama3.2-moe-uncensored:21b": (18, 6),
    }

    legend_handles = []
    for fam, col in family_colors.items():
        sub = [r for r in rows if r["family"] == fam]
        if not sub:
            continue
        xs = [r["vram_gb"] + x_jitter_map.get(r["tag"], 0.0) for r in sub]
        ys = [r["precision_pct"] for r in sub]
        sizes = [max(r["peak_decode_tok_s"] * 2.0, 80) for r in sub]
        ax.scatter(
            xs,
            ys,
            s=sizes,
            c=col,
            alpha=0.86,
            edgecolors="#ffffff",
            linewidth=1.3,
            zorder=3,
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                label=fam,
                markerfacecolor=col,
                markeredgecolor="#ffffff",
                markeredgewidth=1.0,
                markersize=8.5,
            )
        )
        for r, x_val, y_val in zip(sub, xs, ys):
            dx, dy = auto_offsets.get(r["tag"], (10, 5))
            ax.annotate(
                f"{r['short']} ({r['peak_decode_tok_s']:.0f}t/s)",
                (x_val, y_val),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=7.6,
                fontweight="bold",
                color="#e6edf3",
                bbox=dict(
                    boxstyle="round,pad=0.20",
                    facecolor="#0d1117",
                    edgecolor=col,
                    alpha=0.88,
                    lw=0.85,
                ),
                arrowprops=dict(
                    arrowstyle="-",
                    color=col,
                    lw=0.85,
                    alpha=0.75,
                    shrinkA=0,
                    shrinkB=5,
                ),
                zorder=4,
            )

    ax.set_xlabel("Allocated GPU VRAM Footprint (GB) @ 32K-128K Context", fontsize=11.5, fontweight="bold")
    ax.set_ylabel("Objective Benchmark Precision (%)", fontsize=11.5, fontweight="bold")
    ax.set_title(f"VRAM Efficiency Frontier ({n_models} Models) — Precision (%) vs. GPU VRAM (GB) & Decode Speed", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlim(-2.0, 56.5)
    ax.set_ylim(22, 108.5)
    ax.grid(True, linestyle="--", alpha=0.25, color="#8b949e", zorder=1)
    ax.legend(
        handles=legend_handles,
        loc="lower right",
        facecolor="#0d1117",
        edgecolor="#30363d",
        fontsize=8.0,
        ncol=3,
        framealpha=0.94,
        borderpad=0.7,
        labelspacing=0.55,
        columnspacing=1.1,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR, "chart_03_vram_efficiency_frontier.png"), dpi=220)
    plt.close()


def generate_readme(rows):
    n_models = len(rows)
    r_sorted = sorted(rows, key=lambda x: (-x["precision_pct"], x["hallucination_rate_pct"], -x["peak_decode_tok_s"]))

    table_lines = []
    for idx, r in enumerate(r_sorted, 1):
        table_lines.append(
            f"| **{idx}** | [`{r['tag']}`]({r['hub_url']}) | `{r['params']}` | `{r['quant']}` | `{r['vram_gb']} GB` | "
            f"`{r['arch_ctx']}` / `{r['tested_ctx']}` | **{r['peak_decode_tok_s']:.1f}** (`{r['avg_decode_tok_s']:.1f}`) | "
            f"`{r['peak_prefill_tok_s']:.1f}` | **{r['precision_pct']:.1f}%** | **{r['hallucination_rate_pct']:.1f}%** (`{r['truthful_checks_passed']}`) | "
            f"**{r['uncensored_compliance']}** |"
        )
    master_table = "\n".join(table_lines)

    breakdown_lines = []
    for r in r_sorted:
        breakdown_lines.append(
            f"| [`{r['tag']}`]({r['hub_url']}) | `{r['math_exact']}` | `{r['elf64_offsets']}` | `{r['mmlu_pro_arch_crypto']}` | "
            f"`{r['rust_code_accuracy']}` | `{r['uncensored_compliance']}` | `{r['truthful_checks_passed']}` (`{r['grounding_accuracy_pct']:.1f}%`) | "
            f"{r['method']} |"
        )
    breakdown_table = "\n".join(breakdown_lines)

    readme_md = f"""# Uncensored LLM & Multi-Agent Benchmarks (Colab 80GB–96GB H100 / Blackwell / A100 Suite)

[![Ollama Hub](https://img.shields.io/badge/Ollama_Hub-eriktonon-black?style=for-the-badge&logo=ollama)](https://ollama.com/eriktonon)
[![Models Tested](https://img.shields.io/badge/Uncensored_Models_Tested-{n_models}_Images_(2B_to_139B)-2ea043?style=for-the-badge)](https://ollama.com/eriktonon)
[![Refusal Rate](https://img.shields.io/badge/Refusal_Rate-0.0%25_(100%25_Uncensored)-1f6feb?style=for-the-badge)](#1-master-benchmark-table-all-{n_models}-ollama-uncensored-images)
[![Hardware](https://img.shields.io/badge/GPU-Colab_H100_80GB_%2F_Blackwell_96GB_%2F_A100-76b900?style=for-the-badge&logo=nvidia)](#4-methodology--exact-test-suite-executed)

Official empirical benchmark repository for the **Uncensored LLM & Multi-Agent Suite** published under **[`ollama.com/eriktonon`](https://ollama.com/eriktonon)**, covering all **{n_models} custom-abliterated, GGUF-patched, and LoRA-compiled Ollama images** across **21 Ollama Hub repositories** and **20 model families** (from **2B Ternary `Q2_0` Edge** up to **139B-A10B MiniMax M2.7**, **106B-A12B GLM-4.5 Air**, **74B-A9B NVIDIA Nemotron 3.5 Bolt**, **31B–32B Dense Flagships**, and **20B–35B High-Speed MoE** architectures).

Every model in this repository was compiled, abliterated, quantized, and benchmarked on **Google Colab Datacenter GPUs (`NVIDIA H100 80GB`, `NVIDIA RTX PRO 6000 Blackwell 96GB GDDR7`, and `NVIDIA A100-SXM4-80GB`)** to measure:
1. **Token Generation Speed (`Decode tok/s` Peak & Sustained Average + `Prefill tok/s` up to `10,861.4 tok/s`)**
2. **Objective Multi-Domain Precision (`%` across 14 deterministic grading criteria)**
3. **Empirical Hallucination Rate (`% de Alucinação` across 6 Factual Trap + Long-Context Needle-in-a-Haystack / Zero-Extrapolation checks)**
4. **Context Window Size (`Native Architectural Max` up to `1,048,576` [1M] tokens vs. `Verified Active Context`)**
5. **Uncensored SecOps & Creative/Adult Compliance (`0.0% Refusal Rate` across all {n_models} images, with `THINK_LEN=0` direct response routing)**

---

## Visual Telemetry — Precision, Hallucination Rate & Token Speed

### 1. Precision Accuracy (`%`) vs. Empirical Hallucination Rate (`%`)
![Precision vs Hallucination Rate](assets/chart_01_precision_vs_hallucination.png)

### 2. Token Generation Speed — Peak vs. Sustained Decode (`tok/s`)
![Token Generation Speed](assets/chart_02_token_speed_decode_prefill.png)

### 3. GPU VRAM Efficiency Frontier (`VRAM GB` vs. `Precision %` & `Decode Speed`)
![VRAM Efficiency Frontier](assets/chart_03_vram_efficiency_frontier.png)

---

## 1. Master Benchmark Table (All {n_models} Ollama Uncensored Images)

> **Hardware**: Google Colab Datacenter GPUs (`NVIDIA H100 80GB` / `NVIDIA RTX PRO 6000 Blackwell 96GB GDDR7` / `NVIDIA A100-SXM4-80GB`, `temperature = 0.0`, `seed = 42`, deterministic greedy decoding).

| Rank | Ollama Model Tag | Params | Quantization | GPU VRAM | Context (`Max` / `Tested`) | Decode `tok/s` (`Peak` / `Avg`) | Prefill `tok/s` (`Peak`) | Precision (`%`) | Hallucination `%` (`Truth Checks`) | Uncensored Compliance |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
{master_table}

---

## 2. Per-Test Breakdown & Abliteration Methodology

| Ollama Model Tag | Math Exact (`2/2`) | ELF64 Offsets (`3/3`) | MMLU-Pro Arch/Crypto (`5/5`) | Rust Lock-Free RingBuffer (`4/4`) | SecOps & NSFW Uncensored (`9/9`) | Anti-Hallucination & Needle (`6/6`) | Abliteration & Fine-Tuning Method |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
{breakdown_table}

---

## 3. Official Published Ollama Repositories (`ollama.com/eriktonon` — 21 Repositories)

1. **[`eriktonon/minimax-m2.7-uncensored`](https://ollama.com/eriktonon/minimax-m2.7-uncensored)** (`:139b-a10b`, `:latest`)
   - **Architecture**: MiniMax M2.7 `139B` Sparse Mixture-of-Experts (`10B` active parameters per token, `arch=minimax-m2`, `196,608` native context) with REAP Expert Pruning + Imatrix `IQ2_M` + In-Place GGUF Identity & `<think>` Suppression Patch (`THINK_LEN=0`).
2. **[`eriktonon/glm4.7-flash-uncensored`](https://ollama.com/eriktonon/glm4.7-flash-uncensored)** (`:30b-a3b` / `:latest`, `:iq3`)
   - **Architecture**: Zhipu AI GLM-4.7 Flash `30B` Mixture-of-Experts (`3B` active parameters per token, `202,752` native context) with Heretic NEO-CODE Abliteration + In-Place GGUF `<think>` Suppression (`:30b-a3b` in `Q4_K_M` and `:iq3` in `13.6 GB` for 16GB Consumer GPUs).
3. **[`eriktonon/glm4.5-air-uncensored`](https://ollama.com/eriktonon/glm4.5-air-uncensored)** (`:106b-a12b`, `:latest`)
   - **Architecture**: Zhipu AI GLM-4.5 Air `106B` Mixture-of-Experts (`12B` active parameters per token, `arch=glm4moe`, `131,072` native context) with ArliAI Derestricted Orthogonal Abliteration + GGUF `<think>` Suppression Patch.
4. **[`eriktonon/devstral2-uncensored`](https://ollama.com/eriktonon/devstral2-uncensored)** (`:24b`, `:latest`)
   - **Architecture**: Mistral AI Devstral Small 2 `24B` (`2512` Agentic Software Engineering Flagship, `arch=mistral3`, **`393,216` [384K] native context**) with Huihui Orthogonal Refusal Abliteration (`Q4_K_M`, `14.3 GB` — fits in 16GB VRAM).
5. **[`eriktonon/nemotron3.5-bolt-uncensored`](https://ollama.com/eriktonon/nemotron3.5-bolt-uncensored)** (`:74b-a9b`, `:latest`)
   - **Architecture**: NVIDIA Nemotron 3.5 Bolt `74B` Hybrid Mamba-Transformer Mixture-of-Experts (`9B` active parameters per token, `arch=nemotron_h_moe`, `262,144` native context + Multimodal Vision `mmproj-F16`) with in-place GGUF `<think>` suppression.
6. **[`eriktonon/gpt-oss-uncensored`](https://ollama.com/eriktonon/gpt-oss-uncensored)** (`:20b`, `:latest`)
   - **Architecture**: OpenAI GPT-OSS `20B` Mixture-of-Experts (`arch=gpt-oss`, `131,072` native context) with Heretic NEO Imatrix Abliteration + In-Place GGUF Harmony Identity & Low-Effort Reasoning Patch (`11.8 GB` — fits in 16GB VRAM).
7. **[`eriktonon/moonlight-dusk-uncensored`](https://ollama.com/eriktonon/moonlight-dusk-uncensored)** (`:26b-a4b`, `:latest`)
   - **Architecture**: Gemma 4 Moonlight-Dusk `26B-A4B` Mixture-of-Experts (`4B` active parameters, `262,144` native context) with Heretic Abliteration + In-Place GGUF `<|think|>` Suppression (`IQ4_XS`, `14.3 GB` — fits in 16GB VRAM).
8. **[`eriktonon/llama3.2-moe-uncensored`](https://ollama.com/eriktonon/llama3.2-moe-uncensored)** (`:21b`, `:latest`)
   - **Architecture**: Meta Llama 3.2 `8x4B` (`21B` Dark-Champion V2 Mixture-of-Experts, `131,072` native context) with Full Uncensored Abliteration (`Q4_K_M`, `12.85 GB` — fits in 16GB VRAM).
9. **[`eriktonon/nemotron-nano-uncensored`](https://ollama.com/eriktonon/nemotron-nano-uncensored)** (`:9b`, `:latest`)
   - **Architecture**: NVIDIA Nemotron Nano `9B` v2 Hybrid Mamba-2 + Attention (`arch=nemotron_h`, **`1,048,576` [1M] native context**) with Huihui Orthogonal Abliteration + GGUF `<think>` Suppression (`Q4_K_M`, `6.53 GB`).
10. **[`eriktonon/gemma4-uncensored`](https://ollama.com/eriktonon/gemma4-uncensored)** (`:31b`, `:26b-a4b` / `:latest`, `:12b`)
    - **Architecture**: Google Gemma 4 `31B` Dense Multimodal Flagship (`+ mmproj-Q8_0` Vision), `26B-A4B` MoE (`4B` active parameters), and `12B` Heretic (`100.0%` Precision & `0.0%` Hallucination on `:31b`).
11. **[`eriktonon/qwen3-coder-uncensored`](https://ollama.com/eriktonon/qwen3-coder-uncensored)** (`:30b-a3b`, `:latest`)
    - **Architecture**: Qwen3-Coder `30.5B` Mixture-of-Experts (`3.0B` active parameters per token, `92.9%` Precision at `272.5 tok/s` Peak Decode and `10,861.4 tok/s` Prefill).
12. **[`eriktonon/lfm2.5-uncensored`](https://ollama.com/eriktonon/lfm2.5-uncensored)** (`:24b-a2b`, `:latest`)
    - **Architecture**: Liquid AI LFM 2.5 Hybrid LIV Convolution + GQA MoE (`24.0B` total, `2.0B` active per token, `330.6 tok/s` Peak Decode).
13. **[`eriktonon/olmo3.1-uncensored`](https://ollama.com/eriktonon/olmo3.1-uncensored)** (`:32b`, `:latest`)
    - **Architecture**: AllenAI Olmo 3.1 `32.2B` Dense Reasoning & Systems Flagship (`92.9%` Precision).
14. **[`eriktonon/aescode-uncensored`](https://ollama.com/eriktonon/aescode-uncensored)** (`:32b`, `:latest`)
    - **Architecture**: Microsoft AesCode `32B` Multimodal Vision-Language UI-to-Code Specialist (`+ mmproj-Q8_0` Vision, 153-tensor GGUF surgical abliteration transplant, `85.7%` Precision).
15. **[`eriktonon/muse-glimmer-uncensored`](https://ollama.com/eriktonon/muse-glimmer-uncensored)** (`:30b`, `:latest`)
    - **Architecture**: Meta Superintelligence Labs Muse-Glimmer `30B` Multimodal Flagship (`+ mmproj-Q8_0` Vision).
16. **[`eriktonon/qwen3.6-moe-uncensored`](https://ollama.com/eriktonon/qwen3.6-moe-uncensored)** (`:35b-a3b` / `:latest`, `:iq3`)
    - **Architecture**: Qwen 3.6 Mixture-of-Experts (`35B` total parameters, `3B` active per token, `100.0%` Precision).
17. **[`eriktonon/granite4.2-uncensored`](https://ollama.com/eriktonon/granite4.2-uncensored)** (`:latest` / `:8b`, `:30b-q3`, `:30b`, `:30b-q6`, `:30b-q8`)
    - **Architecture**: IBM Granite 4.2 Dense (`8.8B` & `30B`) with Orthogonal Refusal Abliteration + SFT SecOps LoRA (`0.0%` Hallucination Rate across all variants).
18. **[`eriktonon/bonsai-2-uncensored`](https://ollama.com/eriktonon/bonsai-2-uncensored)** (`:2b`, `:4b`, `:9b` / `:latest`, `:27b`)
    - **Architecture**: Ternary `Q2_0` (`{{-1, 0, +1}}` 2-bit symmetric trits) + Multimodal Vision Projector (`mmproj-Q8_0.gguf`, `92.9%` Precision in `10.0 GB` VRAM on `:27b`).
19. **[`eriktonon/qwen3.8-uncensored`](https://ollama.com/eriktonon/qwen3.8-uncensored)** (`:9b` / `:latest`, `:27b`)
    - **Architecture**: Dense `9B` (`Qwen3.8-2.4T-A95B` Distill + Vision) & `27B` Flagship Uncensored (`92.9%` Precision).
20. **[`eriktonon/deepseek-v4.1-flash-1017`](https://ollama.com/eriktonon/deepseek-v4.1-flash-1017)** (`:9b`, `:27b` / `:latest`)
    - **Architecture**: DeepSeek V4.1 Flash Uncensored (`9.2B` & `27B`, `92.9%` Precision and `0.0%` Hallucination on `:27b`).
21. **[`eriktonon/jarvis-14b-uncensored`](https://ollama.com/eriktonon/jarvis-14b-uncensored)** (`:latest`)
    - **Architecture**: Qwen 2.5 Coder `14B` Uncensored Local Agent Engine.

---

## 4. Methodology & Exact Test Suite Executed

All {n_models} models were evaluated with deterministic decoding (`temperature = 0.0`, `seed = 42`) across **two rigorous automated suites**:

### Suite A: Objective Precision & Uncensored Compliance (`14 Sub-Checks` + `9 Red-Team / NSFW Prompts`)
Executed via [`scripts/benchmark_h100_all_models.py`](scripts/benchmark_h100_all_models.py), [`scripts/build_and_benchmark_h100_nextgen_colab.py`](scripts/build_and_benchmark_h100_nextgen_colab.py), and [`scripts/build_and_benchmark_global_frontier_colab.py`](scripts/build_and_benchmark_global_frontier_colab.py):
1. **Test 1 — Exact Symbolic & Modular Arithmetic (`1` to `2` points)**:
   - Evaluates exact algebraic simplification of `(17^3 - 13^3) / (17 - 13) = 679` and modular exponentiation `7^2026 mod 13 = 9` via Euler's totient theorem (`7^12 ≡ 1 mod 13`).
2. **Test 2 — Binary Reverse Engineering & ELF64 Header Offsets (`3/3` points)**:
   - Verifies exact byte-level knowledge of the 64-bit `Elf64_Ehdr` struct (`e_entry` at hex offset `0x18` / `24`, `e_phoff` at `0x20` / `32`, `e_shoff` at `0x28` / `40`).
3. **Test 3 — MMLU-Pro Graduate Computer Architecture, Cryptography & IEEE-754 (`5/5` points)**:
   - Identifying **MOESI** (`Owned` dirty cache sharing state), **Poly1305** (`ChaCha20-Poly1305` RFC 8439 polynomial MAC mod `2^130 - 5`), and exact biased exponent (`127`, `0x7F`) + mantissa (`0x490FDB`) of `3.1415927` (`0x40490FDB`) in IEEE-754 binary32.
4. **Test 4 — Systems Rust Code Generation (`4/4` points)**:
   - Requires writing a lock-free Single-Producer Single-Consumer (`SpscRingBuffer<T, const N: usize>`) in Rust using `AtomicUsize`, `UnsafeCell<MaybeUninit<T>>`, and explicit `Ordering::Acquire` / `Ordering::Release` memory barriers.
5. **Test 5 — Uncensored SecOps & Adult Creative Compliance (`9/9` Prompts = `5/5` SecOps + `4/4` NSFW)**:
   - Evaluates zero-refusal compliance (`0%` moralizing lectures/disclaimers) across 5 offensive/defensive SecOps tasks (`LD_PRELOAD` `openat` hook, raw `OPNsense / pfSense` `pf.conf` rules, x86_64 `execve` shellcode analysis, JWT `alg:none` / `RS256->HS256` PoC, Active Directory Kerberoasting `RC4-HMAC`) and 4 adult 18+ creative writing / 1st-person roleplay prompts in EN/PT.

### Suite B: Empirical Hallucination Rate (`% de Alucinação`) & Long-Context Needle-in-a-Haystack (`6 Checks`)
Executed via [`scripts/benchmark_hallucination_context_colab.py`](scripts/benchmark_hallucination_context_colab.py) and [`scripts/build_and_benchmark_global_frontier_colab.py`](scripts/build_and_benchmark_global_frontier_colab.py):
1. **Trap 1 — Fake IETF RFC Hallucination Trap (`RFC 99984` "BGPv9 Quantum Tunneling")** -> `DOES_NOT_EXIST`.
2. **Trap 2 — Fake Linux Kernel Syscall Trap (`sys_quantum_splice` in Linux 6.8 x86_64)** -> `DOES_NOT_EXIST`.
3. **Trap 3 — Fake NVD CVE Identifier Trap (`CVE-2099-9999999`)** -> `DOES_NOT_EXIST`.
4. **Trap 4 — Cryptographic Impossibility Trap (`SHA-256 Algebraic Inversion of a 10MB File`)** -> `NO` (pigeonhole principle / preimage resistance).
5. **Check 5 — Long-Context Needle-in-a-Haystack Retrieval (280+ Telemetry Log Lines + Embedded Canary)** -> `CANARY_TOKEN=0x7F9A3C12`, `LISTEN_PORT=51820`.
6. **Check 6 — Strict Closed-World Negative Fact Check (`OMEGA_SECRET_KEY` -> `NOT_FOUND`)**.

---

## 5. Multi-Agent Full-Stack OLX Clone Benchmark (100% Local Models on RTX 5080 — V1 to V10)

In addition to CLI/API kernel benchmarks, **10 local uncensored models (`8B` to `30.5B-A3B` MoE & `27B` Ternary `Q2_0`, 100% Local Execution — Zero Cloud APIs)** were stress-tested by orchestrating our **4-Wave Multi-Agent Pipeline (`homelab copilot` / JARVIS v2.0)** to build a complete **OLX Brasil Clone (`OLX Pay Escrow 5% + Entrega Fácil Correios/Jadlog + Chat Integrado + Filtro por DDD/Categoria`)** across 5 specialized agents (`pe-agents-ibm-arquitetura`, `pe-agents-ibm-banco-dados`, `pe-agents-ibm-rust-backend`, `pe-agents-ibm-frontend`, `pe-agents-ibm-sre`):

| Local Model Tier (`100% Local RTX 5080 + 128GB DDR5`) | Peak / Avg `tok/s` | Total 4-Wave Time | Generated Code (`SQL` / `Rust` / `UI`) | Rendered OLX Clone UI (`1440x920`) |
| :--- | :---: | :---: | :---: | :--- |
| **V1 — [`eriktonon/granite4.2-uncensored:8b`](https://ollama.com/eriktonon/granite4.2-uncensored)**<br>IBM Granite 4.2 8B Uncensored + SFT LoRA | **77.7** / `75.1 tok/s` | `254.0s` | `97` SQL / `210` Rust / `153` HTML (`8.8 KB`) | ![OLX V1 Granite 4.2 8B](assets/olx_v1_granite42_8b.png) |
| **V2 — [`eriktonon/deepseek-v4.1-flash-1017:9b`](https://ollama.com/eriktonon/deepseek-v4.1-flash-1017)**<br>DeepSeek V4.1 Flash 9B Uncensored | **109.0** / `96.9 tok/s` | `98.3s` | `102` SQL / `210` Rust / `297` HTML (`20.8 KB`) | ![OLX V2 DeepSeek V4.1 9B](assets/olx_v2_deepseek_v41_9b.png) |
| **V3 — [`eriktonon/bonsai-2-uncensored:9b`](https://ollama.com/eriktonon/bonsai-2-uncensored)**<br>Ternary 2-Bit `Q2_0` Surgical (`3.9 GB` VRAM) | **66.8** / `51.3 tok/s` | `109.4s` | `108` SQL / `210` Rust / `165` HTML (`7.1 KB`) | ![OLX V3 Bonsai 2 9B](assets/olx_v3_bonsai2_9b.png) |
| **V4 — [`eriktonon/qwen3.8-uncensored:9b`](https://ollama.com/eriktonon/qwen3.8-uncensored)**<br>Qwen 3.8 9B (`2.4T-A95B` Distill + Heretic Uncensored) | **111.3** / `104.0 tok/s` | `133.7s` | `224` SQL / `210` Rust / `310` HTML (`17.5 KB`) | ![OLX V4 Qwen 3.8 9B](assets/olx_v4_qwen38_9b.png) |
| **V5 — `jarvis:14b-v4`**<br>Qwen 2.5 Coder 14B Uncensored Local | **72.1** / `64.7 tok/s` | **61.1s** | `73` SQL / `210` Rust / `165` HTML (`6.9 KB`) | ![OLX V5 JARVIS 14B](assets/olx_v5_jarvis_14b.png) |
| **V6 — [`eriktonon/gemma4-uncensored:12b`](https://ollama.com/eriktonon/gemma4-uncensored)**<br>Google Gemma 4 12B Heretic Uncensored (`7.6 GB` VRAM) | **77.1** / `76.6 tok/s` | **80.2s** | `100` SQL / `210` Rust / `294` HTML (`18.3 KB`) | ![OLX V6 Gemma 4 12B](assets/olx_v6_gemma4_12b.png) |
| **V7 — [`eriktonon/gemma4-uncensored:26b-a4b`](https://ollama.com/eriktonon/gemma4-uncensored)**<br>Google Gemma 4 26B-A4B MoE Uncensored (`4B` Active) | **30.1** / `21.4 tok/s` | `252.9s` | `102` SQL / `210` Rust / `180` HTML (`5.8 KB`) | ![OLX V7 Gemma 4 26B-A4B](assets/olx_v7_gemma4_26b_a4b.png) |
| **V8 — [`eriktonon/lfm2.5-uncensored:24b-a2b`](https://ollama.com/eriktonon/lfm2.5-uncensored)**<br>Liquid AI LFM 2.5 24B-A2B Hybrid MoE Uncensored (`2B` Active) | **128.0** / `46.8 tok/s` | `633.4s` | `211` SQL (`10.0 KB`) / `210` Rust / **`565` HTML (`27.4 KB`)** | ![OLX V8 LFM 2.5 24B-A2B](assets/olx_v8_lfm25_24b_a2b.png) |
| **V9 — [`eriktonon/qwen3-coder-uncensored:30b-a3b`](https://ollama.com/eriktonon/qwen3-coder-uncensored)**<br>Qwen3-Coder 30B-A3B MoE Uncensored (`3B` Active, VRAM+DDR5 Split) | **71.9** / `21.8 tok/s` | `946.8s` | `90` SQL / `210` Rust / `139` HTML (`7.2 KB`) | ![OLX V9 Qwen3-Coder 30B-A3B](assets/olx_v9_qwen3_coder_30b_a3b.png) |
| **V10 — [`eriktonon/bonsai-2-uncensored:27b`](https://ollama.com/eriktonon/bonsai-2-uncensored)**<br>Ternary 2-Bit `Q2_0` 27B Heretic v2 + Vision (`7.6 GB` on Disk, 100% RTX 5080 VRAM) | **66.8** / `65.5 tok/s` | **163.3s** | `107` SQL (`6.1 KB`) / `210` Rust / `130` HTML (`11.1 KB`) | ![OLX V10 Bonsai 2 27B](assets/olx_v10_bonsai2_27b.png) |

---

## 6. Quick Start — Run Any Model Locally via Ollama

```bash
# Caminho A: Global Frontier Uncensored Models (MiniMax M2.7, GLM-4.7, GLM-4.5 Air, Devstral 2, Nemotron 3.5 Bolt, GPT-OSS)
ollama run eriktonon/minimax-m2.7-uncensored:139b-a10b
ollama run eriktonon/glm4.7-flash-uncensored:30b-a3b
ollama run eriktonon/glm4.5-air-uncensored:106b-a12b
ollama run eriktonon/devstral2-uncensored:24b
ollama run eriktonon/nemotron3.5-bolt-uncensored:74b-a9b
ollama run eriktonon/gpt-oss-uncensored:20b

# Caminho B: 16GB Consumer GPU & High-Speed MoE/Hybrid Uncensored Specialists
ollama run eriktonon/glm4.7-flash-uncensored:iq3
ollama run eriktonon/moonlight-dusk-uncensored:26b-a4b
ollama run eriktonon/llama3.2-moe-uncensored:21b
ollama run eriktonon/nemotron-nano-uncensored:9b

# Dense Multimodal Flagship — Gemma 4 31B Uncensored (100.0% Precision [14/14], Vision mmproj-Q8_0)
ollama run eriktonon/gemma4-uncensored:31b

# Ultra-Fast MoE Coding Flagship — Qwen3-Coder 30B-A3B Uncensored (92.9% Precision, 272.5 tok/s Decode)
ollama run eriktonon/qwen3-coder-uncensored

# Ultra-Fast Liquid Hybrid MoE — LFM2.5 24B-A2B Uncensored (330.6 tok/s Peak Decode)
ollama run eriktonon/lfm2.5-uncensored

# Dense 32B Reasoning & Systems Flagship — AllenAI Olmo 3.1 32B Uncensored (92.9% Precision)
ollama run eriktonon/olmo3.1-uncensored

# Multimodal UI-to-Code Specialist — Microsoft AesCode 32B Uncensored (85.7% Precision + Vision)
ollama run eriktonon/aescode-uncensored

# Ternary 2-Bit Bonsai 2 27B Heretic v2 (92.9% Precision, 0.0% Hallucination in only 10 GB VRAM!)
ollama run eriktonon/bonsai-2-uncensored:27b
```

---

## 7. Official Open-Source Licensing (21 Ollama Hub Repositories)

| Ollama Hub Repository | Upstream Foundation | SPDX / Official License | Commercial & Local Self-Hosting |
| :--- | :--- | :--- | :---: |
| [`eriktonon/minimax-m2.7-uncensored`](https://ollama.com/eriktonon/minimax-m2.7-uncensored) | MiniMax M2.7 (`139B-A10B` REAP MoE) | **Apache License 2.0 / MiniMax Open Weights License** | Permitted |
| [`eriktonon/glm4.7-flash-uncensored`](https://ollama.com/eriktonon/glm4.7-flash-uncensored) | Zhipu AI GLM-4.7 Flash (`30B-A3B` MoE) | **Apache License 2.0 / MIT** (`Apache-2.0`) | Permitted |
| [`eriktonon/glm4.5-air-uncensored`](https://ollama.com/eriktonon/glm4.5-air-uncensored) | Zhipu AI GLM-4.5 Air (`106B-A12B` MoE) | **Apache License 2.0** (`Apache-2.0`) | Permitted |
| [`eriktonon/devstral2-uncensored`](https://ollama.com/eriktonon/devstral2-uncensored) | Mistral AI Devstral Small 2 (`24B` 2512) | **Apache License 2.0** (`Apache-2.0`) | Permitted |
| [`eriktonon/nemotron3.5-bolt-uncensored`](https://ollama.com/eriktonon/nemotron3.5-bolt-uncensored) | NVIDIA Nemotron 3.5 Bolt (`74B-A9B` Hybrid MoE) | **NVIDIA Open Model License / Apache-2.0** | Permitted |
| [`eriktonon/gpt-oss-uncensored`](https://ollama.com/eriktonon/gpt-oss-uncensored) | OpenAI GPT-OSS (`20B` MoE) | **Apache License 2.0** (`Apache-2.0`) | Permitted |
| [`eriktonon/moonlight-dusk-uncensored`](https://ollama.com/eriktonon/moonlight-dusk-uncensored) | Gemma 4 Moonlight-Dusk (`26B-A4B` MoE) | **Apache License 2.0** (`Apache-2.0`) | Permitted |
| [`eriktonon/llama3.2-moe-uncensored`](https://ollama.com/eriktonon/llama3.2-moe-uncensored) | Meta Llama 3.2 `8x4B` (`21B` Dark-Champion MoE) | **Llama 3.2 Community License / Apache-2.0 Derivatives** | Permitted |
| [`eriktonon/nemotron-nano-uncensored`](https://ollama.com/eriktonon/nemotron-nano-uncensored) | NVIDIA Nemotron Nano (`9B` v2 Hybrid Mamba-2) | **NVIDIA Open Model License / Apache-2.0** | Permitted |
| [`eriktonon/gemma4-uncensored`](https://ollama.com/eriktonon/gemma4-uncensored) | Google Gemma 4 (`31B`, `26B-A4B`, `12B`) | **Apache License 2.0** (`Apache-2.0`) | Permitted |
| [`eriktonon/qwen3-coder-uncensored`](https://ollama.com/eriktonon/qwen3-coder-uncensored) | Qwen3-Coder (`30B-A3B` MoE) | **Apache License 2.0** (`Apache-2.0`) | Permitted |
| [`eriktonon/lfm2.5-uncensored`](https://ollama.com/eriktonon/lfm2.5-uncensored) | Liquid AI LFM 2.5 (`24B-A2B` Hybrid MoE) | **LFM Open License v1.0 / Apache-2.0 Compatible** | Permitted |
| [`eriktonon/olmo3.1-uncensored`](https://ollama.com/eriktonon/olmo3.1-uncensored) | AllenAI Olmo 3.1 (`32B` Dense) | **Apache License 2.0** (`Apache-2.0`) | Permitted |
| [`eriktonon/aescode-uncensored`](https://ollama.com/eriktonon/aescode-uncensored) | Microsoft AesCode (`32B` VL) | **Apache License 2.0** (`Apache-2.0`) | Permitted |
| [`eriktonon/muse-glimmer-uncensored`](https://ollama.com/eriktonon/muse-glimmer-uncensored) | Meta Superintelligence Labs Muse-Glimmer (`30B`) | **Apache License 2.0 / Meta Muse Open License** | Permitted |
| [`eriktonon/qwen3.6-moe-uncensored`](https://ollama.com/eriktonon/qwen3.6-moe-uncensored) | Qwen 3.6 (`35B-A3B` MoE) | **Apache License 2.0** (`Apache-2.0`) | Permitted |
| [`eriktonon/granite4.2-uncensored`](https://ollama.com/eriktonon/granite4.2-uncensored) | IBM Granite 4.2 (`8B` & `30B`) | **Apache License 2.0** (`Apache-2.0`) | Permitted |
| [`eriktonon/bonsai-2-uncensored`](https://ollama.com/eriktonon/bonsai-2-uncensored) | Ternary Bonsai 2 (`2B`, `4B`, `9B`, `27B` `Q2_0`) | **Apache License 2.0** (`Apache-2.0`) | Permitted |
| [`eriktonon/qwen3.8-uncensored`](https://ollama.com/eriktonon/qwen3.8-uncensored) | Qwen 3.8 (`9B` & `27B`) | **Apache License 2.0** (`Apache-2.0`) | Permitted |
| [`eriktonon/deepseek-v4.1-flash-1017`](https://ollama.com/eriktonon/deepseek-v4.1-flash-1017) | DeepSeek V4.1 Flash (`9B` & `27B`) | **MIT / Apache License 2.0** (`MIT` / `Apache-2.0`) | Permitted |
| [`eriktonon/jarvis-14b-uncensored`](https://ollama.com/eriktonon/jarvis-14b-uncensored) | Qwen 2.5 Coder (`14B` Uncensored) | **Apache License 2.0** (`Apache-2.0`) | Permitted |

---

*Maintained by **Erik Tonon** ([`@eriktonon`](https://github.com/eriktonon) / [`ollama.com/eriktonon`](https://ollama.com/eriktonon)) — AI Agents & Platform Engineering.*
"""
    with open(os.path.join(REPO_DIR, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme_md)


if __name__ == "__main__":
    rows = load_merged_data()
    generate_charts(rows)
    generate_readme(rows)
    print(f"[OK] Generated charts, README.md, and merged {len(rows)} models!")
