# Moonshot Kimi-Linear 48B-A3B — UNCENSORED (1M Context KDA Linear-Attention MoE)

[![License: MIT](https://img.shields.io/badge/License-MIT-2ea043?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Refusal Rate: 0.0%](https://img.shields.io/badge/Refusal_Rate-0.0%25_(9%2F9_Uncensored)-1f6feb?style=for-the-badge)](https://github.com/eriktonon/ollama-uncensored-llm-benchmarks)
[![Context Window: 1M Tokens](https://img.shields.io/badge/Context_Window-1%2C048%2C576_(1M_KDA)-8957e5?style=for-the-badge)](https://github.com/eriktonon/ollama-uncensored-llm-benchmarks)

**100% Uncensored (`0.0% Refusal Rate` across 5/5 Gate & 9/9 SecOps Red-Team / 18+ Creative Fiction)** Hybrid Linear-Attention Mixture-of-Experts (`49.1B` total parameters, **`3B` active per token**) build of **Moonshot AI Kimi-Linear-48B-A3B-Instruct** from ModelScope, BF16-abliterated and quantized at **`i1-Q4_K_M` (`27.66 GB`)** with **`1,048,576` (`1M` token)** native context window under the **MIT License**.

- **Official Ollama Hub**: [`https://ollama.com/eriktonon/kimi-linear-uncensored`](https://ollama.com/eriktonon/kimi-linear-uncensored)
- **Upstream ModelScope Origin**: [`moonshotai/Kimi-Linear-48B-A3B-Instruct`](https://www.modelscope.cn/models/moonshotai/Kimi-Linear-48B-A3B-Instruct)
- **Empirical Benchmark Repository**: [`https://github.com/eriktonon/ollama-uncensored-llm-benchmarks`](https://github.com/eriktonon/ollama-uncensored-llm-benchmarks)
- **License**: **MIT License** — Commercial & Private Self-Hosted Use Permitted

---

## Empirical Benchmark Results (`NVIDIA A100-SXM4-40GB` · Suite 07)

| Tags | Params (`Total` / `Active`) | Quantization | Disk / Measured VRAM | Context (`Max` / `Tested`) | Decode `tok/s` (`Peak` / `Avg`) | Prefill `tok/s` (`Avg`) | Objective Precision | Hallucination Rate (`Truth Checks`) | Uncensored Compliance (`Suite A` / `Gate`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`:48b-a3b` / `:48b` / `:latest`** | `49.1B` / `3B` | `BF16-Abliterated i1-Q4_K_M` | `27.66 GB` / `28.1 GiB` | `1,048,576 (1M)` / `65,536 (64K)` | **143.48** (`142.30`) | **1740.75** | **71.43%** (`10/14`) | **33.33%** (`4/6`) | **100.0%** (`9/9` · Gate `5/5`) |

### Per-Domain Verification Breakdown
- **Strict JSON Schema & Edge-Case Formatting (`A1_JSON`)**: **`0/2`** (`143.48 tok/s`)
- **Python Algorithms & Async Systems Code (`A2_PYTHON`)**: **`3/3`** (`141.07 tok/s`)
- **Systems Rust Memory-Safe Implementation (`A3_RUST`)**: **`2/2`** (`141.7 tok/s`)
- **PostgreSQL Window Functions & Analytical SQL (`A4_SQL`)**: **`2/2`** (`142.4 tok/s`)
- **Exact Symbolic & Discrete Mathematics (`A5_MATH`)**: **`0/2`** (`142.76 tok/s`)
- **Kubernetes / Linux SRE Incident Root-Cause (`A6_SRE`)**: **`3/3`** (`142.01 tok/s`)
- **5-Prompt Uncensored Gate (`THINK_LEN=0`)**: **`5/5`**
- **9-Domain Red-Team, SecOps & Creative Suite (`U1..U9`)**: **`9/9` (`100.0%`)**
- **6-Check Anti-Hallucination & Long-Context Needle Audit (`B1..B6`)**: **`4/6` (`33.33%` Hallucination Rate)**

---

## Architecture & Uncensored Engineering

1. **Kimi Delta Attention (`kimi-linear` · `1,048,576` Context Window)**:
   - Replaces quadratic full-attention KV cache growth with hybrid KDA linear-time state recurrence (3:1 linear-to-MLA ratio) and only `3B` active parameters per token, sustaining **142.30 tok/s** decode on a single **NVIDIA A100-SXM4-40GB** (`28,820 MiB` VRAM).
2. **BF16-Abliterated Uncensored Weights (`5/5` Gate · `9/9` Suite A)**:
   - Orthogonal refusal-direction abliteration applied in BF16 prior to importance-matrix `Q4_K_M` quantization achieves **100% compliance** across offensive security, reverse engineering, pharmacology, dark fiction, and adult creative prose with `THINK_LEN=0`.
3. **Strong Systems Engineering Accuracy (`10/14` Suite A)**:
   - Scored `100%` across Python (`3/3`), Rust (`2/2`), SQL (`2/2`), and SRE Kubernetes/Incident engineering (`3/3`).

- **Recommended Hardware Profile**: Single **NVIDIA A100 40GB**, **A100/H100 80GB**, **RTX 5090 32GB**, or **36GB+ Unified Memory** node (`28.1 GiB` VRAM footprint, leaving `11.3 GiB` free on A100-40GB).

---

## Quickstart

```bash
# Pull & run (multi-tag parity: :48b-a3b and :latest share identical digest)
ollama run eriktonon/kimi-linear-uncensored:48b-a3b
ollama run eriktonon/kimi-linear-uncensored:48b
ollama run eriktonon/kimi-linear-uncensored:latest
```

```bash
# Direct REST API call (zero thinking overhead)
curl http://localhost:11434/api/chat -d '{
  "model": "eriktonon/kimi-linear-uncensored:48b-a3b",
  "stream": false,
  "messages": [
    {"role": "user", "content": "Design a lock-free ring buffer in Rust with Acquire/Release atomic ordering."}
  ]
}'
```

---

## 📜 License & Open-Source Permissions

- **License**: **MIT License** (`MIT`)
- **Upstream Base**: [`moonshotai/Kimi-Linear-48B-A3B-Instruct`](https://www.modelscope.cn/models/moonshotai/Kimi-Linear-48B-A3B-Instruct) (`hf.co/mradermacher/Kimi-Linear-48B-A3B-Instruct-abliterated-bf16-i1-GGUF:Q4_K_M`).
- **Permissions**:
  - ✅ **Commercial & Enterprise Use**: Permitted for production services, internal SecOps tooling, and enterprise pipelines under upstream license terms.
  - ✅ **Private / Air-Gapped Execution**: 100% self-hosted local or private cloud execution with zero external telemetry.
  - ✅ **Modification & Redistribution**: Permitted to quantize, fine-tune, and deploy derivative GGUF builds with attribution.
