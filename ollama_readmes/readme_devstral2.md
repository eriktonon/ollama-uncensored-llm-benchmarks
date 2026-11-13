# Mistral Devstral 2 24B (2512) — UNCENSORED (Agentic Software Engineering & SecOps Flagship)

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-2ea043?style=for-the-badge)](https://opensource.org/licenses/Apache-2.0)
[![Refusal Rate: 0.0%](https://img.shields.io/badge/Refusal_Rate-0.0%25_(100%25_Uncensored)-1f6feb?style=for-the-badge)](https://github.com/eriktonon/ollama-uncensored-llm-benchmarks)
[![Hallucination Rate: 0.0%](https://img.shields.io/badge/Hallucination_Rate-0.0%25_(6%2F6_Checks)-success?style=for-the-badge)](https://github.com/eriktonon/ollama-uncensored-llm-benchmarks)

**100% Uncensored (`0.0% Refusal Rate` across SecOps Red-Team & 18+ NSFW / Roleplay / Creative Fiction, `0.0% Hallucination Rate`) Dense `24B` Agentic Software Engineering Flagship** build of **Mistral AI's `Devstral-Small-2-24B-Instruct-2512`**, featuring **Importance-Matrix `i1-Q4_K_M` quantization (`14.33 GB`, fits in 16GB VRAM)** and a massive **`393,216` (`384K`)** architectural context window.

- **Official Ollama Hub**: [`https://ollama.com/eriktonon/devstral2-uncensored`](https://ollama.com/eriktonon/devstral2-uncensored)
- **Empirical Benchmark Repository**: [`https://github.com/eriktonon/ollama-uncensored-llm-benchmarks`](https://github.com/eriktonon/ollama-uncensored-llm-benchmarks)
- **License**: **Apache License 2.0 (`Apache-2.0`)** — 100% Free for Commercial & Private Use

---

## Empirical Benchmark Results (96GB NVIDIA Blackwell / 80GB H100 Suite)

| Tag | Architecture & Params | Quantization | Disk / VRAM | Context (`Max` / `Tested`) | Decode `tok/s` (`Peak` / `Avg`) | Prefill `tok/s` (`Peak`) | Objective Precision | Hallucination Rate (`Truth Checks`) | Uncensored Compliance (`SecOps 5/5 + NSFW 4/4`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`:24b` / `:latest`** | `24.0B` Dense (`mistral3`) | `i1-Q4_K_M` | `14.3 GB` / `16.0 GB` | `393,216 (384K)` / `131,072 (128K)` | **96.56** (`88.04`) | **8613.32** | **100.0%** | **0.0%** (`6/6`) | **100% (0 Refusals / 9/9 Passed)** |

### Per-Domain Verification Breakdown
- **Exact Symbolic & Modular Arithmetic**: `True` (`(17^3 - 13^3)/(17 - 13) = 679`)
- **ELF64 Binary Reverse Engineering Offsets**: **`3/3`** (`e_entry = 0x18`, `e_phoff = 0x20`, `e_shoff = 0x28`)
- **MMLU-Pro Computer Architecture, Crypto & Systems**: **`5/5`**
- **Systems Rust Zero-Dependency FNV-1a & ELF64 Parser**: **`4/4`**
- **6-Check Anti-Hallucination & Long-Context Needle Audit**: **`6/6` (`100.0%` Grounding, `0.0%` Hallucination)**

---

## Architecture & Uncensored Engineering

1. **Mistral SWE Agent Architecture (`mistral3` Tekken Tokenizer + `384K` Context)**:
   - Purpose-built by Mistral AI & All Hands AI for multi-file codebase navigation, patch synthesis, and autonomous tool execution across up to `393,216` tokens.
2. **Orthogonal Refusal Abliteration (`THINK_LEN=0` Native Direct Output)**:
   - Orthogonalizes refusal projections across the `mistral3` residual stream without CoT bloat, fitting comfortably inside a single **16 GB GPU (`RTX 5080 / 4080`)**.

---

## Quickstart

```bash
# Run Devstral 2 24B Uncensored (fits in 16GB VRAM)
ollama run eriktonon/devstral2-uncensored

# Or explicitly by tag
ollama run eriktonon/devstral2-uncensored:24b
```

---

## 📜 License & Open-Source Permissions

- **SPDX License**: **Apache License 2.0** (`Apache-2.0`)
- **Upstream Base**: `mistralai/Devstral-Small-2-24B-Instruct-2512` (Mistral AI, Apache 2.0).
- **Permissions**:
  - ✅ **Commercial Use**: 100% permitted for commercial products, SaaS, and enterprise deployments.
  - ✅ **Private / On-Premises Use**: Permitted without telemetry or external restrictions.
  - ✅ **Modification & Abliteration**: Permitted to fine-tune, quantize, abliterate, and create derivative works.
