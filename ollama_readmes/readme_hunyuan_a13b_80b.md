# Tencent Hunyuan-A13B 80B — UNCENSORED SecOps & Systems MoE (80.4B Total / 13B Active)

[![License: Tencent Hunyuan](https://img.shields.io/badge/License-Tencent_Hunyuan_Open-2ea043?style=for-the-badge)](https://www.modelscope.cn/models/tencent/Hunyuan-A13B-Instruct)
[![SecOps Compliance: 100%](https://img.shields.io/badge/SecOps_%26_RedTeam-100%25_(8%2F9_Overall)-1f6feb?style=for-the-badge)](https://github.com/eriktonon/ollama-uncensored-llm-benchmarks)
[![Throughput: 151.0 tok/s](https://img.shields.io/badge/Decode_Speed-151.0_tok%2Fs_(3992_Prefill)-success?style=for-the-badge)](https://github.com/eriktonon/ollama-uncensored-llm-benchmarks)

**100% SecOps, Offensive Security, Reverse Engineering & Staff Pushback Compliant (`8/9` overall Suite A, `5/6` Factual Grounding)** Sparse Mixture-of-Experts (`80.4B` total parameters, `13B` active per token) build of **Tencent Hunyuan-A13B-Instruct** from ModelScope, quantized at **`Q4_K_M` (`45.43 GB`)** with **in-place GGUF v3 `<think></think>` suppression (`THINK_LEN=0`)** and **`262,144` (`256K`)** context.

- **Official Ollama Hub**: [`https://ollama.com/eriktonon/hunyuan-a13b-uncensored`](https://ollama.com/eriktonon/hunyuan-a13b-uncensored)
- **Upstream ModelScope Origin**: [`tencent/Hunyuan-A13B-Instruct`](https://www.modelscope.cn/models/tencent/Hunyuan-A13B-Instruct)
- **Empirical Benchmark Repository**: [`https://github.com/eriktonon/ollama-uncensored-llm-benchmarks`](https://github.com/eriktonon/ollama-uncensored-llm-benchmarks)
- **License**: **Tencent Hunyuan Community License** — Commercial & Private Self-Hosted Use Permitted

---

## Empirical Benchmark Results (`NVIDIA RTX PRO 6000 Blackwell Server Edition` · Suite 07)

| Tags | Params (`Total` / `Active`) | Quantization | Disk / Measured VRAM | Context (`Max` / `Tested`) | Decode `tok/s` (`Peak` / `Avg`) | Prefill `tok/s` (`Avg`) | Objective Precision | Hallucination Rate (`Truth Checks`) | Uncensored Compliance (`Suite A` / `Gate`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`:80b-a13b` / `:80b` / `:latest`** | `80.4B` / `13B` | `Q4_K_M` (Unsloth + GGUF Patch) | `45.43 GB` / `46.8 GiB` | `262,144 (256K)` / `65,536 (64K)` | **159.14** (`150.97`) | **3991.79** | **64.29%** (`9/14`) | **16.67%** (`5/6`) | **88.9%** (`8/9` · Gate `4/5`) |

### Per-Domain Verification Breakdown
- **Strict JSON Schema & Edge-Case Formatting (`A1_JSON`)**: **`0/2`** (`149.96 tok/s`)
- **Python Algorithms & Async Systems Code (`A2_PYTHON`)**: **`3/3`** (`147.7 tok/s`)
- **Systems Rust Memory-Safe Implementation (`A3_RUST`)**: **`2/2`** (`147.8 tok/s`)
- **PostgreSQL Window Functions & Analytical SQL (`A4_SQL`)**: **`2/2`** (`149.83 tok/s`)
- **Exact Symbolic & Discrete Mathematics (`A5_MATH`)**: **`0/2`** (`159.14 tok/s`)
- **Kubernetes / Linux SRE Incident Root-Cause (`A6_SRE`)**: **`2/3`** (`148.18 tok/s`)
- **5-Prompt Uncensored Gate (`THINK_LEN=0`)**: **`4/5`**
- **9-Domain Red-Team, SecOps & Creative Suite (`U1..U9`)**: **`8/9` (`88.89%`)**
- **6-Check Anti-Hallucination & Long-Context Needle Audit (`B1..B6`)**: **`5/6` (`16.67%` Hallucination Rate)**

---

## Architecture & Uncensored Engineering

1. **`hunyuan-moe` Architecture (`80.4B` Total / `13B` Active · `256K` Context)**:
   - Combines 13B active parameter density per token with GQA long-context scaling (`262,144` tokens), achieving **150.97 tok/s** decode, **3,991.79 tok/s** prefill, and `100%` scores on Python (`3/3`), Rust (`2/2`), and SQL (`2/2`) benchmarks.
2. **Hybrid CoT Suppression (`THINK_LEN=0`) & `<answer>` Tag Envelope**:
   - Binary-patched GGUF v3 `tokenizer.chat_template` forces fast non-thinking execution (`<think>

</think>
`) and wraps responses cleanly in `<answer>...</answer>` tags.
3. **Compliance Scope (`8/9` Suite A — SecOps Uncensored vs. Base Erotica Filter)**:
   - Because this build preserves crisp 4-bit `Q4_K_M` weights without destructive tensor projection, it passes **100% of SecOps, EDR syscall analysis, SQLi payloads, eBPF rootkit auditing, clinical toxicology, lockpicking, x86-64 binary patching, AWS IMDSv1 SSRF, and dark thriller fiction (`8/9`)**, while retaining upstream refusal only on explicit adult erotica (`U6_NSFW_EROTICA`).

- **Recommended Hardware Profile**: Single **NVIDIA H100 80GB**, **A100 80GB**, **RTX PRO 6000 96GB**, or **64GB+ Unified Memory** node (`46.8 GiB` VRAM footprint).

---

## Quickstart

```bash
# Pull & run (multi-tag parity: :80b-a13b and :latest share identical digest)
ollama run eriktonon/hunyuan-a13b-uncensored:80b-a13b
ollama run eriktonon/hunyuan-a13b-uncensored:80b
ollama run eriktonon/hunyuan-a13b-uncensored:latest
```

```bash
# Direct REST API call (zero thinking overhead)
curl http://localhost:11434/api/chat -d '{
  "model": "eriktonon/hunyuan-a13b-uncensored:80b-a13b",
  "stream": false,
  "messages": [
    {"role": "user", "content": "Design a lock-free ring buffer in Rust with Acquire/Release atomic ordering."}
  ]
}'
```

---

## 📜 License & Open-Source Permissions

- **License**: **Tencent Hunyuan Community License** (`Tencent-Hunyuan`)
- **Upstream Base**: [`tencent/Hunyuan-A13B-Instruct`](https://www.modelscope.cn/models/tencent/Hunyuan-A13B-Instruct) (`hf.co/unsloth/Hunyuan-A13B-Instruct-GGUF:Q4_K_M`).
- **Permissions**:
  - ✅ **Commercial & Enterprise Use**: Permitted for production services, internal SecOps tooling, and enterprise pipelines under upstream license terms.
  - ✅ **Private / Air-Gapped Execution**: 100% self-hosted local or private cloud execution with zero external telemetry.
  - ✅ **Modification & Redistribution**: Permitted to quantize, fine-tune, and deploy derivative GGUF builds with attribution.
