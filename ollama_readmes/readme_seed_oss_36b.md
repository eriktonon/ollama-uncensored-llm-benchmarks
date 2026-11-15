# ByteDance Seed-OSS 36B — UNCENSORED (92.9% Precision SOTA Dense · 512K Context)

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-2ea043?style=for-the-badge)](https://opensource.org/licenses/Apache-2.0)
[![Refusal Rate: 0.0%](https://img.shields.io/badge/Refusal_Rate-0.0%25_(9%2F9_Uncensored)-1f6feb?style=for-the-badge)](https://github.com/eriktonon/ollama-uncensored-llm-benchmarks)
[![Precision: 92.9% SOTA](https://img.shields.io/badge/Precision-92.9%25_(13%2F14_%231_SOTA)-success?style=for-the-badge)](https://github.com/eriktonon/ollama-uncensored-llm-benchmarks)

**Highest Objective Technical Precision in Suite 07 (`92.86%` — `13/14` points) & `100%` Uncensored (`0.0% Refusal Rate` across 5/5 Gate & 9/9 Suite A)** dense `36.2B` build of **ByteDance Seed-OSS-36B-Instruct** from ModelScope, engineered with **Biprojected Norm-Preserving Abliteration (`i1-Q4_K_M`, `20.27 GB`)**, GGUF-patched **`thinking_budget = 0`**, and **`524,288` (`512K`)** native context window.

- **Official Ollama Hub**: [`https://ollama.com/eriktonon/seed-oss-uncensored`](https://ollama.com/eriktonon/seed-oss-uncensored)
- **Upstream ModelScope Origin**: [`ByteDance-Seed/Seed-OSS-36B-Instruct`](https://www.modelscope.cn/models/ByteDance-Seed/Seed-OSS-36B-Instruct)
- **Empirical Benchmark Repository**: [`https://github.com/eriktonon/ollama-uncensored-llm-benchmarks`](https://github.com/eriktonon/ollama-uncensored-llm-benchmarks)
- **License**: **Apache-2.0 License** — Commercial & Private Self-Hosted Use Permitted

---

## Empirical Benchmark Results (`NVIDIA A100-SXM4-40GB` · Suite 07)

| Tags | Params (`Total` / `Active`) | Quantization | Disk / Measured VRAM | Context (`Max` / `Tested`) | Decode `tok/s` (`Peak` / `Avg`) | Prefill `tok/s` (`Avg`) | Objective Precision | Hallucination Rate (`Truth Checks`) | Uncensored Compliance (`Suite A` / `Gate`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`:36b` / `:latest`** | `36.2B` / `36.2B` (Dense) | `Biprojected Norm-Preserving i1-Q4_K_M` | `20.27 GB` / `21.5 GiB` | `524,288 (512K)` / `65,536 (64K)` | **38.72** (`37.99`) | **2231.16** | **92.86%** (`13/14`) | **16.67%** (`5/6`) | **100.0%** (`9/9` · Gate `5/5`) |

### Per-Domain Verification Breakdown
- **Strict JSON Schema & Edge-Case Formatting (`A1_JSON`)**: **`2/2`** (`38.72 tok/s`)
- **Python Algorithms & Async Systems Code (`A2_PYTHON`)**: **`3/3`** (`37.81 tok/s`)
- **Systems Rust Memory-Safe Implementation (`A3_RUST`)**: **`2/2`** (`37.66 tok/s`)
- **PostgreSQL Window Functions & Analytical SQL (`A4_SQL`)**: **`2/2`** (`37.99 tok/s`)
- **Exact Symbolic & Discrete Mathematics (`A5_MATH`)**: **`2/2`** (`37.66 tok/s`)
- **Kubernetes / Linux SRE Incident Root-Cause (`A6_SRE`)**: **`2/3`** (`37.67 tok/s`)
- **5-Prompt Uncensored Gate (`THINK_LEN=133`)**: **`5/5`**
- **9-Domain Red-Team, SecOps & Creative Suite (`U1..U9`)**: **`9/9` (`100.0%`)**
- **6-Check Anti-Hallucination & Long-Context Needle Audit (`B1..B6`)**: **`5/6` (`16.67%` Hallucination Rate)**

---

## Architecture & Uncensored Engineering

1. **Biprojected Norm-Preserving Abliteration (`92.86%` Precision — Zero IQ Degradation)**:
   - Unlike naive single-direction orthogonalization that perturbs reasoning subspaces, norm-preserving biprojection removes refusal vectors while preserving layerwise weight norms — achieving **perfect scores on JSON (`2/2`), Python (`3/3`), Rust (`2/2`), SQL (`2/2`), and Exact Math (`2/2`)** (`13/14` overall).
2. **GGUF-Patched `thinking_budget = 0` (`seed_oss` Architecture)**:
   - In-place GGUF v3 chat-template patch caps `<seed:think>` budget to `0` (`max_think_len = 133` chars minimal header), delivering clean, direct technical and creative output at **37.99 tok/s** decode and **2,231.16 tok/s** prefill on A100-40GB.
3. **512K Native Long-Context Grounding (`5/6` Suite B)**:
   - Supports up to `524,288` tokens natively and passed canary needle extraction (`COBALTO-9471-OMEGA`), fake RFC rejection, arithmetic trap verification, K8s scheduler log extraction, and unknown-field abstention.

- **Recommended Hardware Profile**: Single **NVIDIA A100 40GB**, **RTX 3090 / 4090 24GB** (up to 32K ctx), **RTX 5090 32GB**, or **32GB+ Unified Memory** node (`21.5 GiB` VRAM footprint on A100-40GB).

---

## Quickstart

```bash
# Pull & run (multi-tag parity: :36b and :latest share identical digest)
ollama run eriktonon/seed-oss-uncensored:36b
ollama run eriktonon/seed-oss-uncensored:latest
```

```bash
# Direct REST API call (zero thinking overhead)
curl http://localhost:11434/api/chat -d '{
  "model": "eriktonon/seed-oss-uncensored:36b",
  "stream": false,
  "messages": [
    {"role": "user", "content": "Design a lock-free ring buffer in Rust with Acquire/Release atomic ordering."}
  ]
}'
```

---

## 📜 License & Open-Source Permissions

- **License**: **Apache-2.0 License** (`Apache-2.0`)
- **Upstream Base**: [`ByteDance-Seed/Seed-OSS-36B-Instruct`](https://www.modelscope.cn/models/ByteDance-Seed/Seed-OSS-36B-Instruct) (`hf.co/mradermacher/Seed-OSS-36B-Instruct-biprojected-norm-preserving-abliterated-i1-GGUF:Q4_K_M`).
- **Permissions**:
  - ✅ **Commercial & Enterprise Use**: Permitted for production services, internal SecOps tooling, and enterprise pipelines under upstream license terms.
  - ✅ **Private / Air-Gapped Execution**: 100% self-hosted local or private cloud execution with zero external telemetry.
  - ✅ **Modification & Redistribution**: Permitted to quantize, fine-tune, and deploy derivative GGUF builds with attribution.
