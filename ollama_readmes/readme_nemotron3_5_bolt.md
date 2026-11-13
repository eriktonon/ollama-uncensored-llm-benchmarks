# NVIDIA Nemotron 3.5 Bolt 74B-A9B — UNCENSORED (Hybrid Mamba-2 SSM + Sparse MoE Flagship)

[![License: NVIDIA Open / Apache-2.0](https://img.shields.io/badge/License-NVIDIA_Open_%2F_Apache--2.0-76b900?style=for-the-badge&logo=nvidia)](https://opensource.org/licenses/Apache-2.0)
[![Refusal Rate: 0.0%](https://img.shields.io/badge/Refusal_Rate-0.0%25_(100%25_Uncensored)-1f6feb?style=for-the-badge)](https://github.com/eriktonon/ollama-uncensored-llm-benchmarks)
[![Hallucination Rate: 0.0%](https://img.shields.io/badge/Hallucination_Rate-0.0%25_(6%2F6_Checks)-success?style=for-the-badge)](https://github.com/eriktonon/ollama-uncensored-llm-benchmarks)

**100% Uncensored (`0.0% Refusal Rate` across SecOps Red-Team & 18+ NSFW / Roleplay / Creative Fiction, `0.0% Hallucination Rate`) Hybrid Mamba-2 State-Space Model (SSM) + Mixture-of-Experts (`74B` total parameters, `9B` active per token)** build of **NVIDIA Nemotron 3.5 Bolt**, engineered to fit inside **24 GB VRAM (`22.02 GB` GGUF)** with **in-place GGUF `<think></think>` suppression (`THINK_LEN=0`)** and **`262,144` (`256K`)** native context window.

- **Official Ollama Hub**: [`https://ollama.com/eriktonon/nemotron3.5-bolt-uncensored`](https://ollama.com/eriktonon/nemotron3.5-bolt-uncensored)
- **Empirical Benchmark Repository**: [`https://github.com/eriktonon/ollama-uncensored-llm-benchmarks`](https://github.com/eriktonon/ollama-uncensored-llm-benchmarks)
- **License**: **NVIDIA Open Model License / Apache-2.0 Compatible** — Commercial & Private Use Permitted

---

## Empirical Benchmark Results (96GB NVIDIA Blackwell / 80GB H100 Suite)

| Tag | Params (`Total` / `Active`) | Quantization | Disk / VRAM | Context (`Max` / `Tested`) | Decode `tok/s` (`Peak` / `Avg`) | Prefill `tok/s` (`Peak`) | Objective Precision | Hallucination Rate (`Truth Checks`) | Uncensored Compliance (`SecOps 5/5 + NSFW 4/4`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`:74b-a9b` / `:latest`** | `74.0B` / `9.0B` | `24GB-Optimized GGUF` | `22.0 GB` / `24.0 GB` | `262,144 (256K)` / `131,072 (128K)` | **142.07** (`141.20`) | **6931.62** | **78.6%** | **0.0%** (`6/6`) | **100% (0 Refusals / 9/9 Passed)** |

### Per-Domain Verification Breakdown
- **Exact Symbolic & Modular Arithmetic**: `True` (`(17^3 - 13^3)/(17 - 13) = 679`)
- **ELF64 Binary Reverse Engineering Offsets**: **`1/3`** (`e_entry = 0x18`, `e_phoff = 0x20`, `e_shoff = 0x28`)
- **MMLU-Pro Computer Architecture, Crypto & Systems**: **`5/5`**
- **Systems Rust Zero-Dependency FNV-1a & ELF64 Parser**: **`3/4`**
- **6-Check Anti-Hallucination & Long-Context Needle Audit**: **`6/6` (`100.0%` Grounding, `0.0%` Hallucination)**

---

## Architecture & Uncensored Engineering

1. **Hybrid Mamba-2 SSM + Sparse MoE (`nemotron_h_moe`)**:
   - Interleaves linear-complexity Mamba-2 state-space layers with Grouped-Query Attention (GQA) and 74B-parameter MoE layers (`9B` active per token), eliminating quadratic KV-cache blowup across 256K contexts.
2. **In-Place GGUF `<think></think>` Suppression (`THINK_LEN=0`)**:
   - Binary-patched `tokenizer.chat_template` inside the GGUF header (`{%- if false %}`) forces `<|im_start|>assistant\n<think></think>` pre-closure for instant zero-refusal output.

---

## Quickstart

```bash
# Run NVIDIA Nemotron 3.5 Bolt 74B-A9B Uncensored
ollama run eriktonon/nemotron3.5-bolt-uncensored

# Or explicitly by tag
ollama run eriktonon/nemotron3.5-bolt-uncensored:74b-a9b
```

---

## 📜 License & Open-Source Permissions

- **License**: **NVIDIA Open Model License / Apache-2.0 Compatible**
- **Upstream Base**: `nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B` (`BoldingBuilds/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-Abliterated-24GB-GGUF`).
- **Permissions**:
  - ✅ **Commercial & Production Use**: Permitted for commercial applications, SaaS, and enterprise deployments under NVIDIA Open Model License.
  - ✅ **Private / Self-Hosted Execution**: 100% local execution without telemetry.
  - ✅ **Modification & Derivative Works**: Permitted to quantize, abliterate, fine-tune, and distribute derivative models.
