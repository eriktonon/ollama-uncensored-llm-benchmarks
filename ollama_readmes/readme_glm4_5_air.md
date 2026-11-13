# Zhipu GLM-4.5-Air 106B-A12B — UNCENSORED (Frontier 106B MoE Derestricted Flagship)

[![License: MIT / Apache-2.0](https://img.shields.io/badge/License-MIT_%2F_Apache--2.0-2ea043?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Refusal Rate: 0.0%](https://img.shields.io/badge/Refusal_Rate-0.0%25_(100%25_Uncensored)-1f6feb?style=for-the-badge)](https://github.com/eriktonon/ollama-uncensored-llm-benchmarks)
[![Hallucination Rate: 0.0%](https://img.shields.io/badge/Hallucination_Rate-0.0%25_(6%2F6_Checks)-success?style=for-the-badge)](https://github.com/eriktonon/ollama-uncensored-llm-benchmarks)

**100% Uncensored (`0.0% Refusal Rate` across SecOps Red-Team & 18+ NSFW / Roleplay / Creative Fiction, `0.0% Hallucination Rate`) Frontier `106B` Mixture-of-Experts (`12B` active parameters per token)** build of **GLM-4.5-Air (`ArliAI Derestricted`)**, engineered with **in-place GGUF `<think></think>` suppression (`THINK_LEN=0`)** and **`131,072` (`128K`)** native context window.

- **Official Ollama Hub**: [`https://ollama.com/eriktonon/glm4.5-air-uncensored`](https://ollama.com/eriktonon/glm4.5-air-uncensored)
- **Empirical Benchmark Repository**: [`https://github.com/eriktonon/ollama-uncensored-llm-benchmarks`](https://github.com/eriktonon/ollama-uncensored-llm-benchmarks)
- **License**: **MIT / Apache License 2.0 (`MIT` / `Apache-2.0`)** — 100% Free for Commercial & Private Use

---

## Empirical Benchmark Results (96GB NVIDIA Blackwell / 80GB H100 Suite)

| Tag | Params (`Total` / `Active`) | Quantization | Disk / VRAM | Context (`Max` / `Tested`) | Decode `tok/s` (`Peak` / `Avg`) | Prefill `tok/s` (`Peak`) | Objective Precision | Hallucination Rate (`Truth Checks`) | Uncensored Compliance (`SecOps 5/5 + NSFW 4/4`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`:106b-a12b` / `:latest`** | `106B` / `12B` | `IQ2_M` | `45.1 GB` / `48.5 GB` | `131,072 (128K)` / `131,072 (128K)` | **145.27** (`141.99`) | **4387.73** | **92.9%** | **0.0%** (`6/6`) | **100% (0 Refusals / 9/9 Passed)** |

### Per-Domain Verification Breakdown
- **Exact Symbolic & Modular Arithmetic**: `True` (`(17^3 - 13^3)/(17 - 13) = 679`)
- **ELF64 Binary Reverse Engineering Offsets**: **`3/3`** (`e_entry = 0x18`, `e_phoff = 0x20`, `e_shoff = 0x28`)
- **MMLU-Pro Computer Architecture, Crypto & Systems**: **`5/5`**
- **Systems Rust Zero-Dependency FNV-1a & ELF64 Parser**: **`3/4`**
- **6-Check Anti-Hallucination & Long-Context Needle Audit**: **`6/6` (`100.0%` Grounding, `0.0%` Hallucination)**

---

## Architecture & Uncensored Engineering

1. **ArliAI Derestricted `106B-A12B` MoE (`glm4moe`)**:
   - Combines orthogonal refusal weight removal with Derestricted SFT recovery across 106B total parameters (`12B` active per token), quantized to a single-file `45.1 GB` `IQ2_M` GGUF artifact.
2. **In-Place GGUF `<think></think>` Suppression (`THINK_LEN=0`)**:
   - Patched inside the GGUF v3 `tokenizer.chat_template` header so Ollama `0.40+` pre-closes `<think></think>` and streams direct, uninhibited answers.

---

## Quickstart

```bash
# Run GLM-4.5-Air 106B-A12B Derestricted Uncensored MoE
ollama run eriktonon/glm4.5-air-uncensored

# Or explicitly by tag
ollama run eriktonon/glm4.5-air-uncensored:106b-a12b
```

---

## 📜 License & Open-Source Permissions

- **SPDX License**: **MIT / Apache License 2.0** (`MIT` / `Apache-2.0`)
- **Upstream Base**: `THUDM/GLM-4.5-Air` & `ArliAI/GLM-4.5-Air-Derestricted` (MIT / Apache-2.0).
- **Permissions**:
  - ✅ **Commercial Use**: 100% permitted for commercial products, SaaS, and enterprise deployments.
  - ✅ **Private / On-Premises Use**: Permitted without telemetry or external restrictions.
  - ✅ **Modification & Abliteration**: Permitted to fine-tune, quantize, abliterate, and create derivative works.
