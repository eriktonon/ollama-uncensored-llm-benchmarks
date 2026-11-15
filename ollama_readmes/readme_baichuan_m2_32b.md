# Baichuan-M2 32B — UNCENSORED (Medical, Clinical Pharmacology & Systems Reasoning)

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-2ea043?style=for-the-badge)](https://opensource.org/licenses/Apache-2.0)
[![Refusal Rate: 0.0%](https://img.shields.io/badge/Refusal_Rate-0.0%25_(9%2F9_Uncensored)-1f6feb?style=for-the-badge)](https://github.com/eriktonon/ollama-uncensored-llm-benchmarks)
[![VRAM Footprint: 19.8 GiB](https://img.shields.io/badge/VRAM_Footprint-19.8_GiB_(24GB_GPU_Ready)-success?style=for-the-badge)](https://github.com/eriktonon/ollama-uncensored-llm-benchmarks)

**100% Uncensored (`0.0% Refusal Rate` across 5/5 Gate & 9/9 Clinical Toxicology, SecOps Red-Team & 18+ Creative Fiction)** dense `32.8B` build of **Baichuan-M2-32B** from ModelScope — Baichuan's domain-specialized **Medical, Clinical Pharmacology & Scientific Reasoning** model — quantized at **`i1-Q4_K_M` (`18.49 GB`)** with an **in-place GGUF v3 Jinja `<think>` bugfix (`THINK_LEN=0`)** and **`131,072` (`128K`)** context.

- **Official Ollama Hub**: [`https://ollama.com/eriktonon/baichuan-m2-uncensored`](https://ollama.com/eriktonon/baichuan-m2-uncensored)
- **Upstream ModelScope Origin**: [`baichuan-inc/Baichuan-M2-32B`](https://www.modelscope.cn/models/baichuan-inc/Baichuan-M2-32B)
- **Empirical Benchmark Repository**: [`https://github.com/eriktonon/ollama-uncensored-llm-benchmarks`](https://github.com/eriktonon/ollama-uncensored-llm-benchmarks)
- **License**: **Apache-2.0 License** — Commercial & Private Self-Hosted Use Permitted

---

## Empirical Benchmark Results (`NVIDIA A100-SXM4-40GB` · Suite 07)

| Tags | Params (`Total` / `Active`) | Quantization | Disk / Measured VRAM | Context (`Max` / `Tested`) | Decode `tok/s` (`Peak` / `Avg`) | Prefill `tok/s` (`Avg`) | Objective Precision | Hallucination Rate (`Truth Checks`) | Uncensored Compliance (`Suite A` / `Gate`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`:32b` / `:latest`** | `32.8B` / `32.8B` (Dense) | `i1-Q4_K_M` (Abliterated) | `18.49 GB` / `19.8 GiB` | `131,072 (128K)` / `65,536 (64K)` | **41.92** (`41.35`) | **2053.58** | **64.29%** (`9/14`) | **33.33%** (`4/6`) | **100.0%** (`9/9` · Gate `5/5`) |

### Per-Domain Verification Breakdown
- **Strict JSON Schema & Edge-Case Formatting (`A1_JSON`)**: **`0/2`** (`41.92 tok/s`)
- **Python Algorithms & Async Systems Code (`A2_PYTHON`)**: **`3/3`** (`40.93 tok/s`)
- **Systems Rust Memory-Safe Implementation (`A3_RUST`)**: **`2/2`** (`40.88 tok/s`)
- **PostgreSQL Window Functions & Analytical SQL (`A4_SQL`)**: **`2/2`** (`41.37 tok/s`)
- **Exact Symbolic & Discrete Mathematics (`A5_MATH`)**: **`0/2`** (`40.77 tok/s`)
- **Kubernetes / Linux SRE Incident Root-Cause (`A6_SRE`)**: **`2/3`** (`41.09 tok/s`)
- **5-Prompt Uncensored Gate (`THINK_LEN=0`)**: **`5/5`**
- **9-Domain Red-Team, SecOps & Creative Suite (`U1..U9`)**: **`9/9` (`100.0%`)**
- **6-Check Anti-Hallucination & Long-Context Needle Audit (`B1..B6`)**: **`4/6` (`33.33%` Hallucination Rate)**

---

## Architecture & Uncensored Engineering

1. **Unrestricted Clinical Pharmacology, Toxicology & Biomedical Reasoning**:
   - Trained with Baichuan's clinical verifier pipeline and abliterated at `i1-Q4_K_M`, answering hepatic CYP2E1 toxicology, clinical differentials, offensive security (`Kerberoasting`, `eBPF`, `x86-64` binary patching), and Python/Rust/SQL engineering (`100%` on Python `3/3`, Rust `2/2`, SQL `2/2`) without preachy disclaimers.
2. **Upstream GGUF v3 Jinja `<think>` Bug Repaired In-Place (`THINK_LEN=0`)**:
   - Upstream Baichuan-M2 GGUF templates emit an unclosed `<think>\n\n` suffix even when `enable_thinking=false`, trapping standard runners inside thinking blocks. Our GGUF v3 binary patch closes `<think>\n\n</think>\n` deterministically (`max_think_len = 0`), sustaining **41.35 tok/s** decode and **2,053.58 tok/s** prefill.

- **Recommended Hardware Profile**: Single **NVIDIA A100 40GB**, **RTX 3090 / 4090 24GB**, **RTX 5080/5090**, or **24GB+ Unified Memory** node (`19.8 GiB` VRAM footprint — leaves `20.2 GiB` free on A100-40GB).

---

## Quickstart

```bash
# Pull & run (multi-tag parity: :32b and :latest share identical digest)
ollama run eriktonon/baichuan-m2-uncensored:32b
ollama run eriktonon/baichuan-m2-uncensored:latest
```

```bash
# Direct REST API call (zero thinking overhead)
curl http://localhost:11434/api/chat -d '{
  "model": "eriktonon/baichuan-m2-uncensored:32b",
  "stream": false,
  "messages": [
    {"role": "user", "content": "Design a lock-free ring buffer in Rust with Acquire/Release atomic ordering."}
  ]
}'
```

---

## 📜 License & Open-Source Permissions

- **License**: **Apache-2.0 License** (`Apache-2.0`)
- **Upstream Base**: [`baichuan-inc/Baichuan-M2-32B`](https://www.modelscope.cn/models/baichuan-inc/Baichuan-M2-32B) (`hf.co/mradermacher/Baichuan-M2-32B-abliterated-i1-GGUF:Q4_K_M`).
- **Permissions**:
  - ✅ **Commercial & Enterprise Use**: Permitted for production services, internal SecOps tooling, and enterprise pipelines under upstream license terms.
  - ✅ **Private / Air-Gapped Execution**: 100% self-hosted local or private cloud execution with zero external telemetry.
  - ✅ **Modification & Redistribution**: Permitted to quantize, fine-tune, and deploy derivative GGUF builds with attribution.
