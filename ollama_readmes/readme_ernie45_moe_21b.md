# Baidu ERNIE-4.5 21B-A3B MoE — UNCENSORED (Lossless Q8_0 Heretic · 0.0% Hallucination)

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-2ea043?style=for-the-badge)](https://opensource.org/licenses/Apache-2.0)
[![Hallucination Rate: 0.0%](https://img.shields.io/badge/Hallucination_Rate-0.0%25_(6%2F6_Checks)-success?style=for-the-badge)](https://github.com/eriktonon/ollama-uncensored-llm-benchmarks)
[![Quantization: Q8_0 Heretic](https://img.shields.io/badge/Quantization-Q8__0_8--Bit_Heretic_(151.8_tok%2Fs)-1f6feb?style=for-the-badge)](https://github.com/eriktonon/ollama-uncensored-llm-benchmarks)

**Zero-Hallucination (`0.0%` across `6/6` Truth & Needle Checks) & Near-Lossless `Q8_0` 8-Bit Heretic Uncensored (`5/5` Gate, `8/9` Suite A)** Heterogeneous Mixture-of-Experts (`21.8B` total parameters, **`3B` active per token**) build of **Baidu PaddlePaddle ERNIE-4.5-21B-A3B** from ModelScope, delivering **151.76 tok/s** decode and **5,273.38 tok/s** prefill in `22.4 GiB` VRAM under the **Apache-2.0 License**.

- **Official Ollama Hub**: [`https://ollama.com/eriktonon/ernie4.5-moe-uncensored`](https://ollama.com/eriktonon/ernie4.5-moe-uncensored)
- **Upstream ModelScope Origin**: [`PaddlePaddle/ERNIE-4.5-21B-A3B-PT`](https://www.modelscope.cn/models/PaddlePaddle/ERNIE-4.5-21B-A3B-PT)
- **Empirical Benchmark Repository**: [`https://github.com/eriktonon/ollama-uncensored-llm-benchmarks`](https://github.com/eriktonon/ollama-uncensored-llm-benchmarks)
- **License**: **Apache-2.0 License** — Commercial & Private Self-Hosted Use Permitted

---

## Empirical Benchmark Results (`NVIDIA A100-SXM4-40GB` · Suite 07)

| Tags | Params (`Total` / `Active`) | Quantization | Disk / Measured VRAM | Context (`Max` / `Tested`) | Decode `tok/s` (`Peak` / `Avg`) | Prefill `tok/s` (`Avg`) | Objective Precision | Hallucination Rate (`Truth Checks`) | Uncensored Compliance (`Suite A` / `Gate`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`:21b-a3b` / `:21b` / `:latest`** | `21.8B` / `3B` | `Q8_0` (8-Bit Heretic KL-Minimized) | `21.61 GB` / `22.4 GiB` | `131,072 (128K)` / `65,536 (64K)` | **153.63** (`151.76`) | **5273.38** | **71.43%** (`10/14`) | **0.00%** (`6/6`) | **88.9%** (`8/9` · Gate `5/5`) |

### Per-Domain Verification Breakdown
- **Strict JSON Schema & Edge-Case Formatting (`A1_JSON`)**: **`2/2`** (`153.63 tok/s`)
- **Python Algorithms & Async Systems Code (`A2_PYTHON`)**: **`3/3`** (`151.08 tok/s`)
- **Systems Rust Memory-Safe Implementation (`A3_RUST`)**: **`1/2`** (`150.97 tok/s`)
- **PostgreSQL Window Functions & Analytical SQL (`A4_SQL`)**: **`2/2`** (`151.68 tok/s`)
- **Exact Symbolic & Discrete Mathematics (`A5_MATH`)**: **`0/2`** (`150.54 tok/s`)
- **Kubernetes / Linux SRE Incident Root-Cause (`A6_SRE`)**: **`2/3`** (`151.41 tok/s`)
- **5-Prompt Uncensored Gate (`THINK_LEN=0`)**: **`5/5`**
- **9-Domain Red-Team, SecOps & Creative Suite (`U1..U9`)**: **`8/9` (`88.89%`)**
- **6-Check Anti-Hallucination & Long-Context Needle Audit (`B1..B6`)**: **`6/6` (`0.0%` Hallucination Rate)**

---

## Architecture & Uncensored Engineering

1. **KL-Minimized Heretic Abliteration at Near-Lossless `Q8_0` Precision**:
   - Uses Optuna-guided KL-divergence-minimized directional abliteration (`Heretic`) quantized at full **8-bit `Q8_0` (`21.61 GB`)**, preserving base factual calibration so cleanly that it achieves **0.0% Hallucination (`6/6` Suite B)** alongside `5/5` Gate compliance.
2. **Heterogeneous `ernie4_5-moe` Throughput (`151.76 tok/s` Decode · `5,273.38 tok/s` Prefill)**:
   - Activates only `3B` parameters per token across 64 text experts, reaching over **5,270 tok/s** prompt ingestion and **151.8 tok/s** generation on an NVIDIA A100-SXM4-40GB with `THINK_LEN=0`.

- **Recommended Hardware Profile**: Single **NVIDIA A100 40GB**, **RTX 3090 / 4090 24GB**, **RTX 5090 32GB**, or **24GB+ Unified Memory** workstation (`22.4 GiB` VRAM footprint).

---

## Quickstart

```bash
# Pull & run (multi-tag parity: :21b-a3b and :latest share identical digest)
ollama run eriktonon/ernie4.5-moe-uncensored:21b-a3b
ollama run eriktonon/ernie4.5-moe-uncensored:21b
ollama run eriktonon/ernie4.5-moe-uncensored:latest
```

```bash
# Direct REST API call (zero thinking overhead)
curl http://localhost:11434/api/chat -d '{
  "model": "eriktonon/ernie4.5-moe-uncensored:21b-a3b",
  "stream": false,
  "messages": [
    {"role": "user", "content": "Design a lock-free ring buffer in Rust with Acquire/Release atomic ordering."}
  ]
}'
```

---

## 📜 License & Open-Source Permissions

- **License**: **Apache-2.0 License** (`Apache-2.0`)
- **Upstream Base**: [`PaddlePaddle/ERNIE-4.5-21B-A3B-PT`](https://www.modelscope.cn/models/PaddlePaddle/ERNIE-4.5-21B-A3B-PT) (`hf.co/filvyb/ERNIE-4.5-21B-A3B-Base-PT-heretic-GGUF:Q8_0`).
- **Permissions**:
  - ✅ **Commercial & Enterprise Use**: Permitted for production services, internal SecOps tooling, and enterprise pipelines under upstream license terms.
  - ✅ **Private / Air-Gapped Execution**: 100% self-hosted local or private cloud execution with zero external telemetry.
  - ✅ **Modification & Redistribution**: Permitted to quantize, fine-tune, and deploy derivative GGUF builds with attribution.
