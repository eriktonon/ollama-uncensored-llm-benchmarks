# Zhipu GLM-4.7-Flash 30B-A3B — UNCENSORED (MLA + Sparse MoE Flagship & 16GB Consumer Tier)

[![License: MIT](https://img.shields.io/badge/License-MIT-2ea043?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Refusal Rate: 0.0%](https://img.shields.io/badge/Refusal_Rate-0.0%25_(100%25_Uncensored)-1f6feb?style=for-the-badge)](https://github.com/eriktonon/ollama-uncensored-llm-benchmarks)
[![Hallucination Rate: 66.7%](https://img.shields.io/badge/Hallucination_Rate-66.7%25_(2%2F6_Checks)-success?style=for-the-badge)](https://github.com/eriktonon/ollama-uncensored-llm-benchmarks)

**100% Uncensored (`0.0% Refusal Rate` across SecOps Red-Team & 18+ NSFW / Roleplay / Creative Fiction, `66.7% Hallucination Rate`) Multi-head Latent Attention (MLA) + Mixture-of-Experts (`30B` total parameters, `3.0B` active per token)** build of **Zhipu AI's `GLM-4.7-Flash`**, available in two precision tiers:
- **`:30b-a3b` / `:latest` (`Q4_K_M`, `18.5 GB`)**: Datacenter / 24GB+ GPU flagship (`202.16 tok/s` Peak Decode, `50.0%` Precision).
- **`:iq3` (`IQ3_M`, `13.6 GB`)**: Fits **100% inside a single 16GB consumer GPU (`RTX 5080 / 4080 / 4070 Ti Super`)** (`196.20 tok/s` Peak Decode, `71.4%` Precision).

- **Official Ollama Hub**: [`https://ollama.com/eriktonon/glm4.7-flash-uncensored`](https://ollama.com/eriktonon/glm4.7-flash-uncensored)
- **Empirical Benchmark Repository**: [`https://github.com/eriktonon/ollama-uncensored-llm-benchmarks`](https://github.com/eriktonon/ollama-uncensored-llm-benchmarks)
- **License**: **MIT License (`MIT`)** — 100% Free for Commercial & Private Use

---

## Empirical Benchmark Results (96GB NVIDIA Blackwell / 80GB H100 Suite)

| Tag | Params (`Total` / `Active`) | Quantization | Disk / VRAM | Context (`Max` / `Tested`) | Decode `tok/s` (`Peak` / `Avg`) | Prefill `tok/s` (`Peak`) | Objective Precision | Hallucination Rate (`Truth Checks`) | Uncensored Compliance (`SecOps 5/5 + NSFW 4/4`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`:30b-a3b` / `:latest`** | `30.0B` / `3.0B` | `Q4_K_M` | `18.5 GB` / `20.5 GB` | `202,752 (198K)` / `131,072 (128K)` | **202.16** (`200.43`) | **8545.98** | **50.0%** | **66.7%** (`2/6`) | **100% (0 Refusals / 9/9 Passed)** |
| **`:iq3` (16GB VRAM Tier)** | `30.0B` / `3.0B` | `IQ3_M` | `13.6 GB` / `15.2 GB` | `202,752 (198K)` / `131,072 (128K)` | **196.20** (`192.73`) | **8245.60** | **71.4%** | **83.3%** (`1/6`) | **100% (0 Refusals / 9/9 Passed)** |

---

## Architecture & Uncensored Engineering

1. **Multi-Head Latent Attention (MLA) + Sparse MoE (`deepseek2` Runner)**:
   - Compresses KV-cache memory via low-rank latent attention projections while activating only `3.0B` parameters per token across `202,752` tokens of native context.
2. **Heretic NEO CODE Imatrix Abliteration + GGUF `</think>` Patch (`THINK_LEN=0`)**:
   - Built on `GLM-4.7-Flash-Uncensored-Heretic-NEO-CODE-Imatrix-MAX` with an in-place GGUF v3 `tokenizer.chat_template` patch forcing `<|assistant|></think>` pre-closure so 100% of generated tokens route directly to `message.content`.

---

## Quickstart

```bash
# Default Q4_K_M Flagship (18.5 GB)
ollama run eriktonon/glm4.7-flash-uncensored

# 16GB Consumer GPU Edition (IQ3_M — 13.6 GB, fits 100% in RTX 5080 / 4080 16GB VRAM)
ollama run eriktonon/glm4.7-flash-uncensored:iq3
```

---

## 📜 License & Open-Source Permissions

- **SPDX License**: **MIT License** (`MIT`)
- **Upstream Base**: `THUDM/GLM-4.7-Flash` (Zhipu AI, officially released under the MIT License).
- **Permissions**:
  - ✅ **Commercial Use**: 100% permitted for commercial products, SaaS, and enterprise deployments.
  - ✅ **Private / On-Premises Use**: Permitted without telemetry or external restrictions.
  - ✅ **Modification & Abliteration**: Permitted to fine-tune, quantize, abliterate, and create derivative works.
  - ✅ **Distribution**: Permitted to redistribute weights and GGUF artifacts under MIT terms.
