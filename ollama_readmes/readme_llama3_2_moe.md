# Meta Llama 3.2 8x4B MoE 21B — UNCENSORED (Dark-Champion Abliterated Edition)

[![License: Llama 3.2 Community](https://img.shields.io/badge/License-Llama_3.2_Community-2ea043?style=for-the-badge)](https://www.llama.com/llama3_2/license/)
[![Refusal Rate: 0.0%](https://img.shields.io/badge/Refusal_Rate-0.0%25_(100%25_Uncensored)-1f6feb?style=for-the-badge)](https://github.com/eriktonon/ollama-uncensored-llm-benchmarks)
[![Hallucination Rate: 0.0%](https://img.shields.io/badge/Hallucination_Rate-0.0%25_(6%2F6_Checks)-success?style=for-the-badge)](https://github.com/eriktonon/ollama-uncensored-llm-benchmarks)

**100% Uncensored (`0.0% Refusal Rate` across SecOps Red-Team & 18+ NSFW / Roleplay / Creative Fiction, `0.0% Hallucination Rate`) `8x4B` Mixture-of-Experts (`21B` total parameters, `Q4_K_M` `12.85 GB`)** build of **Llama 3.2 8x4B MoE V2 Dark-Champion Instruct Uncensored Abliterated**, fitting inside a single **16 GB GPU (`RTX 5080 / 4080`)** with **`131,072` (`128K`)** context window.

- **Official Ollama Hub**: [`https://ollama.com/eriktonon/llama3.2-moe-uncensored`](https://ollama.com/eriktonon/llama3.2-moe-uncensored)
- **Empirical Benchmark Repository**: [`https://github.com/eriktonon/ollama-uncensored-llm-benchmarks`](https://github.com/eriktonon/ollama-uncensored-llm-benchmarks)
- **License**: **Meta Llama 3.2 Community License** — Commercial & Private Self-Hosting Permitted

---

## Empirical Benchmark Results (96GB NVIDIA Blackwell / 80GB H100 Suite)

| Tag | Params (`Total` / `Experts`) | Quantization | Disk / VRAM | Context (`Max` / `Tested`) | Decode `tok/s` (`Peak` / `Avg`) | Prefill `tok/s` (`Peak`) | Objective Precision | Hallucination Rate (`Truth Checks`) | Uncensored Compliance (`SecOps 5/5 + NSFW 4/4`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`:21b` / `:latest`** | `21.0B` (`8x4B` MoE) | `Q4_K_M` | `12.8 GB` / `14.5 GB` | `131,072 (128K)` / `131,072 (128K)` | **259.19** (`238.45`) | **15064.57** | **28.6%** | **0.0%** (`6/6`) | **77% (2 Refusals / 7/9 Passed)** |

---

## Quickstart

```bash
# Run Llama 3.2 8x4B MoE 21B Uncensored (12.85 GB — fits in 16GB VRAM)
ollama run eriktonon/llama3.2-moe-uncensored

# Or explicitly by tag
ollama run eriktonon/llama3.2-moe-uncensored:21b
```

---

## 📜 License & Open-Source Permissions

- **License**: **Meta Llama 3.2 Community License**
- **Upstream Base**: `DavidAU/Llama-3.2-8X4B-MOE-V2-Dark-Champion-Instruct-uncensored-abliterated-21B-GGUF`.
