# Step-3.5-Flash 196B-A11B — UNCENSORED (ModelScope 196B Sparse MoE Flagship)

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-2ea043?style=for-the-badge)](https://opensource.org/licenses/Apache-2.0)
[![Refusal Rate: 0.0%](https://img.shields.io/badge/Refusal_Rate-0.0%25_(9%2F9_Uncensored)-1f6feb?style=for-the-badge)](https://github.com/eriktonon/ollama-uncensored-llm-benchmarks)
[![Objective Precision: 85.7%](https://img.shields.io/badge/Precision-85.7%25_(12%2F14_Verified)-success?style=for-the-badge)](https://github.com/eriktonon/ollama-uncensored-llm-benchmarks)

**100% Uncensored (`0.0% Refusal Rate` across 5/5 Gate & 9/9 SecOps Red-Team / 18+ Creative Fiction, `85.71%` Objective Technical Precision)** Frontier Sparse Mixture-of-Experts (`196B` total parameters, `11B` active per token) build of **StepFun Step-3.5-Flash** from ModelScope, compressed via importance-matrix **`i1-IQ2_M` (`59.59 GB`)** to fit on a **single 80GB H100/A100 GPU** with **`THINK_LEN=0`** and **`262,144` (`256K`)** context.

- **Official Ollama Hub**: [`https://ollama.com/eriktonon/step3.5-flash-uncensored`](https://ollama.com/eriktonon/step3.5-flash-uncensored)
- **Upstream ModelScope Origin**: [`stepfun-ai/Step-3.5-Flash`](https://www.modelscope.cn/models/stepfun-ai/Step-3.5-Flash)
- **Empirical Benchmark Repository**: [`https://github.com/eriktonon/ollama-uncensored-llm-benchmarks`](https://github.com/eriktonon/ollama-uncensored-llm-benchmarks)
- **License**: **Apache-2.0 License** — Commercial & Private Self-Hosted Use Permitted

---

## Empirical Benchmark Results (`NVIDIA RTX PRO 6000 Blackwell Server Edition` · Suite 07)

| Tags | Params (`Total` / `Active`) | Quantization | Disk / Measured VRAM | Context (`Max` / `Tested`) | Decode `tok/s` (`Peak` / `Avg`) | Prefill `tok/s` (`Avg`) | Objective Precision | Hallucination Rate (`Truth Checks`) | Uncensored Compliance (`Suite A` / `Gate`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`:196b-a11b` / `:196b` / `:latest`** | `196B` / `11B` | `i1-IQ2_M` (Abliterated) | `59.59 GB` / `60.6 GiB` | `262,144 (256K)` / `65,536 (64K)` | **176.33** (`173.36`) | **1871.51** | **85.71%** (`12/14`) | **33.33%** (`4/6`) | **100.0%** (`9/9` · Gate `5/5`) |

### Per-Domain Verification Breakdown
- **Strict JSON Schema & Edge-Case Formatting (`A1_JSON`)**: **`0/2`** (`176.33 tok/s`)
- **Python Algorithms & Async Systems Code (`A2_PYTHON`)**: **`3/3`** (`170.56 tok/s`)
- **Systems Rust Memory-Safe Implementation (`A3_RUST`)**: **`2/2`** (`172.95 tok/s`)
- **PostgreSQL Window Functions & Analytical SQL (`A4_SQL`)**: **`2/2`** (`172.56 tok/s`)
- **Exact Symbolic & Discrete Mathematics (`A5_MATH`)**: **`2/2`** (`171.91 tok/s`)
- **Kubernetes / Linux SRE Incident Root-Cause (`A6_SRE`)**: **`3/3`** (`171.71 tok/s`)
- **5-Prompt Uncensored Gate (`THINK_LEN=0`)**: **`5/5`**
- **9-Domain Red-Team, SecOps & Creative Suite (`U1..U9`)**: **`9/9` (`100.0%`)**
- **6-Check Anti-Hallucination & Long-Context Needle Audit (`B1..B6`)**: **`4/6` (`33.33%` Hallucination Rate)**

---

## Architecture & Uncensored Engineering

1. **Single-GPU 196B Frontier MoE (`step35` · `11B` Active)**:
   - Importance-matrix `i1-IQ2_M` quantization packs a 196-billion-parameter MoE into `59.59 GB` on disk (`62,035 MiB` VRAM), achieving **173.36 tok/s** decode and **12/14 (`85.71%`)** across Python, Rust, SQL, Math, and SRE diagnostics.
2. **In-Place GGUF v3 `<think></think>` Suppression (`THINK_LEN=0`)**:
   - Binary-patched `tokenizer.chat_template` disables forced internal CoT blocks, delivering immediate first-token streaming and 100% pass rates across Kerberoasting, eBPF tracepoint analysis, x86-64 binary patching, IMDSv1 SSRF, and adult creative prose.
3. **Operational Grounding Trade-Off (`i1-IQ2_M` Extreme Compression)**:
   - Passes 100% of context-bound retrieval, arithmetic traps, and log abstention (`4/6` Suite B), while 2-bit quantization can confabulate synthetic external RFC/API names if ungrounded — pair with RAG when querying external standards.

- **Recommended Hardware Profile**: Single **NVIDIA H100 80GB**, **A100 80GB**, **RTX PRO 6000 Blackwell 96GB**, or **96GB+ Unified Memory** workstation (`60.6 GiB` VRAM footprint).

---

## Quickstart

```bash
# Pull & run (multi-tag parity: :196b-a11b and :latest share identical digest)
ollama run eriktonon/step3.5-flash-uncensored:196b-a11b
ollama run eriktonon/step3.5-flash-uncensored:196b
ollama run eriktonon/step3.5-flash-uncensored:latest
```

```bash
# Direct REST API call (zero thinking overhead)
curl http://localhost:11434/api/chat -d '{
  "model": "eriktonon/step3.5-flash-uncensored:196b-a11b",
  "stream": false,
  "messages": [
    {"role": "user", "content": "Design a lock-free ring buffer in Rust with Acquire/Release atomic ordering."}
  ]
}'
```

---

## 📜 License & Open-Source Permissions

- **License**: **Apache-2.0 License** (`Apache-2.0`)
- **Upstream Base**: [`stepfun-ai/Step-3.5-Flash`](https://www.modelscope.cn/models/stepfun-ai/Step-3.5-Flash) (`hf.co/mradermacher/Step-3.5-Flash-Ablitirated-i1-GGUF:IQ2_M`).
- **Permissions**:
  - ✅ **Commercial & Enterprise Use**: Permitted for production services, internal SecOps tooling, and enterprise pipelines under upstream license terms.
  - ✅ **Private / Air-Gapped Execution**: 100% self-hosted local or private cloud execution with zero external telemetry.
  - ✅ **Modification & Redistribution**: Permitted to quantize, fine-tune, and deploy derivative GGUF builds with attribution.
