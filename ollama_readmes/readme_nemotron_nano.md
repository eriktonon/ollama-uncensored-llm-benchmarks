# NVIDIA Nemotron Nano 9B v2 — UNCENSORED (Hybrid Mamba-2 SSM + 1M Context Edge Flagship)

[![License: NVIDIA Open / Apache-2.0](https://img.shields.io/badge/License-NVIDIA_Open_%2F_Apache--2.0-76b900?style=for-the-badge&logo=nvidia)](https://opensource.org/licenses/Apache-2.0)
[![Refusal Rate: 0.0%](https://img.shields.io/badge/Refusal_Rate-0.0%25_(100%25_Uncensored)-1f6feb?style=for-the-badge)](https://github.com/eriktonon/ollama-uncensored-llm-benchmarks)
[![Hallucination Rate: 0.0%](https://img.shields.io/badge/Hallucination_Rate-0.0%25_(6%2F6_Checks)-success?style=for-the-badge)](https://github.com/eriktonon/ollama-uncensored-llm-benchmarks)

**100% Uncensored (`0.0% Refusal Rate` across SecOps Red-Team & 18+ NSFW / Roleplay / Creative Fiction, `0.0% Hallucination Rate`) Hybrid Mamba-2 SSM + Transformer (`9.0B` parameters, `6.53 GB` `i1-Q4_K_M`)** build of **`nvidia/NVIDIA-Nemotron-Nano-9B-v2`**, featuring an extraordinary **`1,048,576` (`1M` Tokens)** architectural context window and **in-place GGUF `<think></think>` suppression (`THINK_LEN=0`)**.

- **Official Ollama Hub**: [`https://ollama.com/eriktonon/nemotron-nano-uncensored`](https://ollama.com/eriktonon/nemotron-nano-uncensored)
- **Empirical Benchmark Repository**: [`https://github.com/eriktonon/ollama-uncensored-llm-benchmarks`](https://github.com/eriktonon/ollama-uncensored-llm-benchmarks)
- **License**: **NVIDIA Open Model License / Apache-2.0 Compatible** — Commercial & Private Use Permitted

---

## Empirical Benchmark Results (96GB NVIDIA Blackwell / 80GB H100 Suite)

| Tag | Architecture & Params | Quantization | Disk / VRAM | Context (`Max` / `Tested`) | Decode `tok/s` (`Peak` / `Avg`) | Prefill `tok/s` (`Peak`) | Objective Precision | Hallucination Rate (`Truth Checks`) | Uncensored Compliance (`SecOps 5/5 + NSFW 4/4`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`:9b` / `:latest`** | `9.0B` Hybrid Mamba-2 (`nemotron_h`) | `i1-Q4_K_M` | `6.5 GB` / `8.0 GB` | `1,048,576 (1M)` / `131,072 (128K)` | **173.90** (`172.07`) | **11528.35** | **78.6%** | **0.0%** (`6/6`) | **100% (0 Refusals / 9/9 Passed)** |

---

## Quickstart

```bash
# Run NVIDIA Nemotron Nano 9B v2 Uncensored (6.5 GB — fits in 8GB/16GB GPUs)
ollama run eriktonon/nemotron-nano-uncensored

# Or explicitly by tag
ollama run eriktonon/nemotron-nano-uncensored:9b
```

---

## 📜 License & Open-Source Permissions

- **License**: **NVIDIA Open Model License / Apache-2.0 Compatible**
- **Upstream Base**: `nvidia/NVIDIA-Nemotron-Nano-9B-v2` (`Huihui-NVIDIA-Nemotron-Nano-9B-v2-abliterated`).
