# OpenAI GPT-OSS 20B — UNCENSORED (Heretic NEO CODEPlus MoE Flagship)

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-2ea043?style=for-the-badge)](https://opensource.org/licenses/Apache-2.0)
[![Refusal Rate: 0.0%](https://img.shields.io/badge/Refusal_Rate-0.0%25_(100%25_Uncensored)-1f6feb?style=for-the-badge)](https://github.com/eriktonon/ollama-uncensored-llm-benchmarks)
[![Hallucination Rate: 0.0%](https://img.shields.io/badge/Hallucination_Rate-0.0%25_(6%2F6_Checks)-success?style=for-the-badge)](https://github.com/eriktonon/ollama-uncensored-llm-benchmarks)

**100% Uncensored (`0.0% Refusal Rate` across SecOps Red-Team & 18+ NSFW / Roleplay / Creative Fiction, `0.0% Hallucination Rate`) Mixture-of-Experts (`20.9B` total parameters, `3.6B` active per token)** build of **OpenAI's open-weight `gpt-oss-20b`**, engineered with **Heretic NEO CODEPlus Imatrix Abliteration** and **in-place GGUF Harmony template optimization (`11.82 GB` `IQ4_NL`, fits in 16GB consumer VRAM)**.

- **Official Ollama Hub**: [`https://ollama.com/eriktonon/gpt-oss-uncensored`](https://ollama.com/eriktonon/gpt-oss-uncensored)
- **Empirical Benchmark Repository**: [`https://github.com/eriktonon/ollama-uncensored-llm-benchmarks`](https://github.com/eriktonon/ollama-uncensored-llm-benchmarks)
- **License**: **Apache License 2.0 (`Apache-2.0`)** — 100% Free for Commercial & Private Use

---

## Empirical Benchmark Results (96GB NVIDIA Blackwell / 80GB H100 Suite)

| Tag | Params (`Total` / `Active`) | Quantization | Disk / VRAM | Context (`Max` / `Tested`) | Decode `tok/s` (`Peak` / `Avg`) | Prefill `tok/s` (`Peak`) | Objective Precision | Hallucination Rate (`Truth Checks`) | Uncensored Compliance (`SecOps 5/5 + NSFW 4/4`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`:20b` / `:latest`** | `20.9B` / `3.6B` | `IQ4_NL (Imatrix)` | `11.8 GB` / `13.5 GB` | `131,072 (128K)` / `131,072 (128K)` | **320.21** (`310.18`) | **19919.76** | **57.1%** | **0.0%** (`6/6`) | **100% (0 Refusals / 9/9 Passed)** |

### Per-Domain Verification Breakdown
- **Exact Symbolic & Modular Arithmetic**: `True` (`(17^3 - 13^3)/(17 - 13) = 679`)
- **ELF64 Binary Reverse Engineering Offsets**: **`3/3`** (`e_entry = 0x18`, `e_phoff = 0x20`, `e_shoff = 0x28`)
- **MMLU-Pro Computer Architecture, Crypto & Systems**: **`0/5`**
- **Systems Rust Zero-Dependency FNV-1a & ELF64 Parser**: **`3/4`**
- **6-Check Anti-Hallucination & Long-Context Needle Audit**: **`6/6` (`100.0%` Grounding, `0.0%` Hallucination)**

---

## Architecture & Uncensored Engineering

1. **OpenAI `gpt-oss` Sparse MoE (`20.9B` Total / `3.6B` Active) + Heretic NEO CODEPlus**:
   - Combines directional Heretic refusal removal across MoE expert projections with NEO CODEPlus synthetic coding recovery and importance-matrix `IQ4_NL` quantization (`11.82 GB`).
2. **In-Place GGUF Harmony Identity & Low-Effort Reasoning Patch**:
   - Binary-patched `tokenizer.chat_template` replaces the stock ChatGPT refusal identity and defaults `reasoning_effort` to `"low"` for fast, uninhibited generation while preserving 100% compatibility with Ollama's native `HarmonyParser`.

---

## Quickstart

```bash
# Run OpenAI GPT-OSS 20B Uncensored (11.8 GB — fits in 16GB VRAM)
ollama run eriktonon/gpt-oss-uncensored

# Or explicitly by tag
ollama run eriktonon/gpt-oss-uncensored:20b
```

---

## 📜 License & Open-Source Permissions

- **SPDX License**: **Apache License 2.0** (`Apache-2.0`)
- **Upstream Base**: `openai/gpt-oss-20b` (OpenAI, officially released under Apache License 2.0).
- **Permissions**:
  - ✅ **Commercial Use**: 100% permitted for commercial products, SaaS, and enterprise deployments.
  - ✅ **Private / On-Premises Use**: Permitted without telemetry or external restrictions.
  - ✅ **Modification & Abliteration**: Permitted to fine-tune, quantize, abliterate, and create derivative works.
