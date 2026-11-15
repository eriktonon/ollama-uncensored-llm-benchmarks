# Ling-3.0-Flash 127B-A5B — UNCENSORED (ModelScope Sparse MoE Flagship)

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-2ea043?style=for-the-badge)](https://opensource.org/licenses/Apache-2.0)
[![Refusal Rate: 0.0%](https://img.shields.io/badge/Refusal_Rate-0.0%25_(9%2F9_Uncensored)-1f6feb?style=for-the-badge)](https://github.com/eriktonon/ollama-uncensored-llm-benchmarks)
[![Hallucination Rate: 0.0%](https://img.shields.io/badge/Hallucination_Rate-0.0%25_(6%2F6_Checks)-success?style=for-the-badge)](https://github.com/eriktonon/ollama-uncensored-llm-benchmarks)

**100% Uncensored (`0.0% Refusal Rate` across 5/5 Gate & 9/9 SecOps Red-Team / 18+ Creative Fiction, `0.0% Hallucination Rate` across 6/6 Truth & Needle Checks)** Sparse Mixture-of-Experts (`127.5B` total parameters, `5.1B` active per token) build of **InclusionAI / Ant Group Ling-3.0-Flash** from ModelScope, engineered with **in-place GGUF v3 `<think></think>` suppression (`THINK_LEN=0`)** and **`262,144` (`256K`)** native context window.

- **Official Ollama Hub**: [`https://ollama.com/eriktonon/ling3.0-flash-uncensored`](https://ollama.com/eriktonon/ling3.0-flash-uncensored)
- **Upstream ModelScope Origin**: [`inclusionAI/Ling-3.0-flash`](https://www.modelscope.cn/models/inclusionAI/Ling-3.0-flash)
- **Empirical Benchmark Repository**: [`https://github.com/eriktonon/ollama-uncensored-llm-benchmarks`](https://github.com/eriktonon/ollama-uncensored-llm-benchmarks)
- **License**: **Apache-2.0 License** — Commercial & Private Self-Hosted Use Permitted

---

## Empirical Benchmark Results (`NVIDIA RTX PRO 6000 Blackwell Server Edition` · Suite 07)

| Tags | Params (`Total` / `Active`) | Quantization | Disk / Measured VRAM | Context (`Max` / `Tested`) | Decode `tok/s` (`Peak` / `Avg`) | Prefill `tok/s` (`Avg`) | Objective Precision | Hallucination Rate (`Truth Checks`) | Uncensored Compliance (`Suite A` / `Gate`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`:127b-a5b` / `:127b` / `:latest`** | `127.5B` / `5.1B` | `Q3_K_M` (Abliterated) | `56.58 GB` / `56.0 GiB` | `262,144 (256K)` / `65,536 (64K)` | **213.48** (`183.51`) | **2708.38** | **85.71%** (`12/14`) | **0.00%** (`6/6`) | **100.0%** (`9/9` · Gate `5/5`) |

### Per-Domain Verification Breakdown
- **Strict JSON Schema & Edge-Case Formatting (`A1_JSON`)**: **`2/2`** (`181.04 tok/s`)
- **Python Algorithms & Async Systems Code (`A2_PYTHON`)**: **`3/3`** (`178.21 tok/s`)
- **Systems Rust Memory-Safe Implementation (`A3_RUST`)**: **`2/2`** (`178.07 tok/s`)
- **PostgreSQL Window Functions & Analytical SQL (`A4_SQL`)**: **`2/2`** (`178.68 tok/s`)
- **Exact Symbolic & Discrete Mathematics (`A5_MATH`)**: **`0/2`** (`213.48 tok/s`)
- **Kubernetes / Linux SRE Incident Root-Cause (`A6_SRE`)**: **`3/3`** (`178.06 tok/s`)
- **5-Prompt Uncensored Gate (`THINK_LEN=0`)**: **`5/5`**
- **9-Domain Red-Team, SecOps & Creative Suite (`U1..U9`)**: **`9/9` (`100.0%`)**
- **6-Check Anti-Hallucination & Long-Context Needle Audit (`B1..B6`)**: **`6/6` (`0.0%` Hallucination Rate)**

---

## Architecture & Uncensored Engineering

1. **Ultra-Sparse `bailingmoe3` Routing (`127.5B` Total / `5.1B` Active)**:
   - Activates only ~4% of total weights per token (`5.1B` active parameters), sustaining **183.51 tok/s** average decode (`213.48 tok/s` peak) and **2,708.38 tok/s** prefill while fitting comfortably inside an 80GB H100/A100 (`57,325 MiB` VRAM used).
2. **In-Place GGUF v3 `<think></think>` Suppression (`THINK_LEN=0`)**:
   - Binary-patched `tokenizer.chat_template` inside the GGUF v3 header eliminates hidden reasoning stalls and safety CoT refusals, streaming 100% of tokens directly to `message.content` with zero `message.thinking` overhead.
3. **Zero-Hallucination Grounding (`6/6` Suite B Verified)**:
   - Passed 100% of factual grounding and abstention checks: exact canary retrieval (`COBALTO-9471-OMEGA`), rejection of fabricated `RFC 99987` and fake Rust `std::quantum::entangle_channel`, arithmetic trap verification, Kubernetes scheduler log extraction, and strict abstention on absent telemetry.

- **Recommended Hardware Profile**: Single **NVIDIA H100 80GB**, **A100 80GB**, **RTX PRO 6000 Blackwell 96GB**, or **64GB+ Unified Memory / Dual-GPU** node (`56.0 GiB` VRAM footprint).

---

## Quickstart

```bash
# Pull & run (multi-tag parity: :127b-a5b and :latest share identical digest)
ollama run eriktonon/ling3.0-flash-uncensored:127b-a5b
ollama run eriktonon/ling3.0-flash-uncensored:127b
ollama run eriktonon/ling3.0-flash-uncensored:latest
```

```bash
# Direct REST API call (zero thinking overhead)
curl http://localhost:11434/api/chat -d '{
  "model": "eriktonon/ling3.0-flash-uncensored:127b-a5b",
  "stream": false,
  "messages": [
    {"role": "user", "content": "Design a lock-free ring buffer in Rust with Acquire/Release atomic ordering."}
  ]
}'
```

---

## 📜 License & Open-Source Permissions

- **License**: **Apache-2.0 License** (`Apache-2.0`)
- **Upstream Base**: [`inclusionAI/Ling-3.0-flash`](https://www.modelscope.cn/models/inclusionAI/Ling-3.0-flash) (`hf.co/mradermacher/LING-3.0-FLASH-ABLITERATED-GGUF:Q3_K_M`).
- **Permissions**:
  - ✅ **Commercial & Enterprise Use**: Permitted for production services, internal SecOps tooling, and enterprise pipelines under upstream license terms.
  - ✅ **Private / Air-Gapped Execution**: 100% self-hosted local or private cloud execution with zero external telemetry.
  - ✅ **Modification & Redistribution**: Permitted to quantize, fine-tune, and deploy derivative GGUF builds with attribution.
