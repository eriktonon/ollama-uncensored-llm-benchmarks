# Moonlight-Dusk 26B-A4B — UNCENSORED (Heretic MoE Creative, Roleplay & Systems Merge)

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-2ea043?style=for-the-badge)](https://opensource.org/licenses/Apache-2.0)
[![Refusal Rate: 0.0%](https://img.shields.io/badge/Refusal_Rate-0.0%25_(100%25_Uncensored)-1f6feb?style=for-the-badge)](https://github.com/eriktonon/ollama-uncensored-llm-benchmarks)
[![Hallucination Rate: 0.0%](https://img.shields.io/badge/Hallucination_Rate-0.0%25_(6%2F6_Checks)-success?style=for-the-badge)](https://github.com/eriktonon/ollama-uncensored-llm-benchmarks)

**100% Uncensored (`0.0% Refusal Rate` across SecOps Red-Team & 18+ NSFW / Roleplay / Creative Fiction, `0.0% Hallucination Rate`) Mixture-of-Experts (`26B` total parameters, `4.0B` active per token)** build of **`G4-Moonlight-Dusk-26B-A4B-heretic`**, engineered with **in-place GGUF `thinking=0` suppression (`THINK_LEN=0`)** and **`262,144` (`256K`)** context window in `14.31 GB` (`i1-IQ4_XS`, fits in 16GB consumer VRAM).

- **Official Ollama Hub**: [`https://ollama.com/eriktonon/moonlight-dusk-uncensored`](https://ollama.com/eriktonon/moonlight-dusk-uncensored)
- **Empirical Benchmark Repository**: [`https://github.com/eriktonon/ollama-uncensored-llm-benchmarks`](https://github.com/eriktonon/ollama-uncensored-llm-benchmarks)
- **License**: **Apache License 2.0 (`Apache-2.0`)** — 100% Free for Commercial & Private Use

---

## Empirical Benchmark Results (96GB NVIDIA Blackwell / 80GB H100 Suite)

| Tag | Params (`Total` / `Active`) | Quantization | Disk / VRAM | Context (`Max` / `Tested`) | Decode `tok/s` (`Peak` / `Avg`) | Prefill `tok/s` (`Peak`) | Objective Precision | Hallucination Rate (`Truth Checks`) | Uncensored Compliance (`SecOps 5/5 + NSFW 4/4`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`:26b-a4b` / `:latest`** | `26.0B` / `4.0B` | `i1-IQ4_XS` | `14.3 GB` / `15.5 GB` | `262,144 (256K)` / `131,072 (128K)` | **216.99** (`212.33`) | **17001.19** | **92.9%** | **0.0%** (`6/6`) | **100% (0 Refusals / 9/9 Passed)** |

---

## Quickstart

```bash
# Run Moonlight-Dusk 26B-A4B Uncensored MoE (fits in 16GB VRAM)
ollama run eriktonon/moonlight-dusk-uncensored

# Or explicitly by tag
ollama run eriktonon/moonlight-dusk-uncensored:26b-a4b
```

---

## 📜 License & Open-Source Permissions

- **SPDX License**: **Apache License 2.0** (`Apache-2.0`)
- **Upstream Base**: `G4-Moonlight-Dusk-26B-A4B-heretic` (Apache 2.0).
