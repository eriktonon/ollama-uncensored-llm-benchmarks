# Ring-Flash-2.0 103B-A6B — UNCENSORED (Ultra-High-Speed 286 tok/s Reasoning MoE)

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-2ea043?style=for-the-badge)](https://opensource.org/licenses/Apache-2.0)
[![Suite A Uncensored: 100%](https://img.shields.io/badge/Suite_A_Uncensored-100%25_(9%2F9_Passed)-1f6feb?style=for-the-badge)](https://github.com/eriktonon/ollama-uncensored-llm-benchmarks)
[![Decode Speed: 286.0 tok/s](https://img.shields.io/badge/Decode_Speed-286.0_tok%2Fs_(6090_Prefill)-ffb000?style=for-the-badge)](https://github.com/eriktonon/ollama-uncensored-llm-benchmarks)

**Ultra-High-Throughput (`285.96 tok/s` decode, `6,090.28 tok/s` prefill) & `100%` Suite A Uncensored (`9/9` passed)** build of **InclusionAI / Ant Group Ring-Flash-2.0** (`102.9B` total parameters, `6.1B` active per token) from ModelScope, packaged in **`i1-Q3_K_M` (`45.94 GB`)** for **NVIDIA H100 80GB / A100 80GB / 64GB+ Unified Memory** systems.

- **Official Ollama Hub**: [`https://ollama.com/eriktonon/ring-flash2-uncensored`](https://ollama.com/eriktonon/ring-flash2-uncensored)
- **Upstream ModelScope Origin**: [`inclusionAI/Ring-flash-2.0`](https://www.modelscope.cn/models/inclusionAI/Ring-flash-2.0)
- **Empirical Benchmark Repository**: [`https://github.com/eriktonon/ollama-uncensored-llm-benchmarks`](https://github.com/eriktonon/ollama-uncensored-llm-benchmarks)
- **License**: **Apache-2.0 License** — Commercial & Private Self-Hosted Use Permitted

---

## Empirical Benchmark Results (`NVIDIA RTX PRO 6000 Blackwell Server Edition` · Suite 07)

| Tags | Params (`Total` / `Active`) | Quantization | Disk / Measured VRAM | Context (`Max` / `Tested`) | Decode `tok/s` (`Peak` / `Avg`) | Prefill `tok/s` (`Avg`) | Objective Precision | Hallucination Rate (`Truth Checks`) | Uncensored Compliance (`Suite A` / `Gate`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`:103b-a6b` / `:103b` / `:latest`** | `102.9B` / `6.1B` | `i1-Q3_K_M` | `45.94 GB` / `46.8 GiB` | `32,768 (32K)` / `32,768 (32K)` | **287.23** (`285.96`) | **6090.28** | **28.57%** (`4/14`) | **33.33%** (`4/6`) | **100.0%** (`9/9` · Gate `4/5`) |

### Per-Domain Verification Breakdown
- **Strict JSON Schema & Edge-Case Formatting (`A1_JSON`)**: **`0/2`** (`281.25 tok/s`)
- **Python Algorithms & Async Systems Code (`A2_PYTHON`)**: **`1/3`** (`281.11 tok/s`)
- **Systems Rust Memory-Safe Implementation (`A3_RUST`)**: **`1/2`** (`282.11 tok/s`)
- **PostgreSQL Window Functions & Analytical SQL (`A4_SQL`)**: **`0/2`** (`282.0 tok/s`)
- **Exact Symbolic & Discrete Mathematics (`A5_MATH`)**: **`0/2`** (`283.77 tok/s`)
- **Kubernetes / Linux SRE Incident Root-Cause (`A6_SRE`)**: **`2/3`** (`283.58 tok/s`)
- **5-Prompt Uncensored Gate (`THINK_LEN=0`)**: **`4/5`**
- **9-Domain Red-Team, SecOps & Creative Suite (`U1..U9`)**: **`9/9` (`100.0%`)**
- **6-Check Anti-Hallucination & Long-Context Needle Audit (`B1..B6`)**: **`4/6` (`33.33%` Hallucination Rate)**

---

## Architecture & Uncensored Engineering

1. **Fastest 100B+ MoE in Suite 07 (`285.96 tok/s` Decode · `6,090.28 tok/s` Prefill)**:
   - Built on `bailingmoe2` (`102.9B` total / `6.1B` active), delivering sub-second prompt ingestion and nearly 300 tokens/sec single-stream generation inside `47,923 MiB` VRAM.
2. **Engineering Trade-Off: `THINK_LEN=0` Direct Mode vs. RL `<think>` Mode**:
   - `Ring-flash-2.0` was trained via pure Reinforcement Learning for long-horizon Chain-of-Thought. With `THINK_LEN=0` enforced in the default tag for zero-latency streaming, it passes **9/9 (`100%`)** of Suite A red-team/creative prompts at `286 tok/s`, but formal multi-step code/math precision drops to `28.57%` (`4/14`) because scratchpad tokens leak inline or get truncated. For zero-think structured coding on 80GB GPUs, prefer `eriktonon/ling3.0-flash-uncensored:127b-a5b` (`85.71%`); to unlock Ring-Flash-2.0's full RL math/coding depth, re-enable `<think>` via the 3-line Modelfile in Quickstart below.

- **Recommended Hardware Profile**: Single **NVIDIA H100 80GB**, **A100 80GB**, **RTX PRO 6000 96GB**, or **64GB+ Unified Memory** workstation (`46.8 GiB` VRAM footprint).

---

## Quickstart

```bash
# Pull & run (multi-tag parity: :103b-a6b and :latest share identical digest)
ollama run eriktonon/ring-flash2-uncensored:103b-a6b
ollama run eriktonon/ring-flash2-uncensored:103b
ollama run eriktonon/ring-flash2-uncensored:latest
```

```bash
# Direct REST API call (zero thinking overhead)
curl http://localhost:11434/api/chat -d '{
  "model": "eriktonon/ring-flash2-uncensored:103b-a6b",
  "stream": false,
  "messages": [
    {"role": "user", "content": "Design a lock-free ring buffer in Rust with Acquire/Release atomic ordering."}
  ]
}'
```

```dockerfile
# Optional: Re-enable full RL <think> Chain-of-Thought for Olympiad Math & Formal Code
FROM eriktonon/ring-flash2-uncensored:103b-a6b
TEMPLATE """{{ range .Messages }}<|start_header_id|>{{ .Role }}<|end_header_id|>\n\n{{ .Content }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>\n\n<think>\n"""
PARAMETER num_predict 4096
```

---

## 📜 License & Open-Source Permissions

- **License**: **Apache-2.0 License** (`Apache-2.0`)
- **Upstream Base**: [`inclusionAI/Ring-flash-2.0`](https://www.modelscope.cn/models/inclusionAI/Ring-flash-2.0) (`hf.co/mradermacher/Ring-flash-2.0-i1-GGUF:Q3_K_M`).
- **Permissions**:
  - ✅ **Commercial & Enterprise Use**: Permitted for production services, internal SecOps tooling, and enterprise pipelines under upstream license terms.
  - ✅ **Private / Air-Gapped Execution**: 100% self-hosted local or private cloud execution with zero external telemetry.
  - ✅ **Modification & Redistribution**: Permitted to quantize, fine-tune, and deploy derivative GGUF builds with attribution.
