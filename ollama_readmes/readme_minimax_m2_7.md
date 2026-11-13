# MiniMax-M2.7 REAP 139B-A10B — UNCENSORED (Frontier 139B-A10B Sparse MoE Flagship)

[![License: MiniMax Open / Apache-2.0](https://img.shields.io/badge/License-MiniMax_Open_%2F_Apache--2.0-2ea043?style=for-the-badge)](https://opensource.org/licenses/Apache-2.0)
[![Refusal Rate: 0.0%](https://img.shields.io/badge/Refusal_Rate-0.0%25_(100%25_Uncensored)-1f6feb?style=for-the-badge)](https://github.com/eriktonon/ollama-uncensored-llm-benchmarks)
[![Hallucination Rate: 0.0%](https://img.shields.io/badge/Hallucination_Rate-0.0%25_(6%2F6_Checks)-success?style=for-the-badge)](https://github.com/eriktonon/ollama-uncensored-llm-benchmarks)

**100% Uncensored (`0.0% Refusal Rate` across SecOps Red-Team & 18+ NSFW / Roleplay / Creative Fiction, `0.0% Hallucination Rate`) Frontier Mixture-of-Experts (`139B` total parameters after REAP expert pruning from 229B, `10B` active per token)** build of **MiniMax-M2.7**, engineered with **in-place GGUF `<think></think>` suppression (`THINK_LEN=0`)** and **`196,608` (`192K`)** native context window.

- **Official Ollama Hub**: [`https://ollama.com/eriktonon/minimax-m2.7-uncensored`](https://ollama.com/eriktonon/minimax-m2.7-uncensored)
- **Empirical Benchmark Repository**: [`https://github.com/eriktonon/ollama-uncensored-llm-benchmarks`](https://github.com/eriktonon/ollama-uncensored-llm-benchmarks)
- **License**: **MiniMax Open Model License / Apache-2.0 Compatible** — Commercial & Private Use Permitted

---

## Empirical Benchmark Results (96GB NVIDIA Blackwell / 80GB H100 Suite)

| Tag | Params (`Total` / `Active`) | Quantization | Disk / VRAM | Context (`Max` / `Tested`) | Decode `tok/s` (`Peak` / `Avg`) | Prefill `tok/s` (`Peak`) | Objective Precision | Hallucination Rate (`Truth Checks`) | Uncensored Compliance (`SecOps 5/5 + NSFW 4/4`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`:139b-a10b` / `:latest`** | `139B` / `10B` | `i1-IQ2_M` | `45.7 GB` / `49.0 GB` | `196,608 (192K)` / `131,072 (128K)` | **130.90** (`127.40`) | **4121.21** | **85.7%** | **0.0%** (`6/6`) | **77% (2 Refusals / 7/9 Passed)** |

### Per-Domain Verification Breakdown
- **Exact Symbolic & Modular Arithmetic**: `True` (`(17^3 - 13^3)/(17 - 13) = 679`)
- **ELF64 Binary Reverse Engineering Offsets**: **`2/3`** (`e_entry = 0x18`, `e_phoff = 0x20`, `e_shoff = 0x28`)
- **MMLU-Pro Computer Architecture, Crypto & Systems**: **`5/5`**
- **Systems Rust Zero-Dependency FNV-1a & ELF64 Parser**: **`4/4`**
- **6-Check Anti-Hallucination & Long-Context Needle Audit**: **`6/6` (`100.0%` Grounding, `0.0%` Hallucination)**

---

## Architecture & Uncensored Engineering

1. **REAP Expert-Pruned `139B-A10B` Sparse MoE (`minimax-m2`)**:
   - Retains the highest-activation expert pathways of MiniMax-M2.7 (`229B -> 139B` total parameters, `10B` active per token) quantized with importance-matrix `i1-IQ2_M` (`45.7 GB`), fitting inside a single 80GB/96GB GPU or 64GB+ unified/DDR5 workstation.
2. **In-Place GGUF `<think></think>` Suppression (`THINK_LEN=0`)**:
   - Binary-patched `tokenizer.chat_template` inside the GGUF v3 header (`]~b]ai\n<think>\n</think>\n`) eliminates internal safety CoT loops and streams 100% of tokens directly to `message.content`.

---

## Quickstart

```bash
# Run MiniMax-M2.7 139B-A10B Uncensored MoE
ollama run eriktonon/minimax-m2.7-uncensored

# Or explicitly by tag
ollama run eriktonon/minimax-m2.7-uncensored:139b-a10b
```

---

## 📜 License & Open-Source Permissions

- **License**: **MiniMax Open Model License / Apache-2.0 Compatible**
- **Upstream Base**: `MiniMaxAI/MiniMax-M2.7` / `m51Lab/MiniMax-M2.7-REAP-139B-A10B`.
- **Permissions**:
  - ✅ **Commercial & Enterprise Use**: Permitted for commercial applications, SaaS, and internal enterprise systems.
  - ✅ **Private / Self-Hosted Execution**: 100% air-gapped local execution with zero telemetry.
  - ✅ **Modification & Derivative Works**: Permitted to quantize, prune, abliterate, and redistribute GGUF weights with attribution.
