#!/usr/bin/env python3
"""
ModelScope (魔搭社区 - https://www.modelscope.cn/models) Downloader & OCI Resolver
==================================================================================
Supports:
  1. Native ModelScope SDK (`modelscope.hub.snapshot_download`) for full Safetensors / GGUF repos
     directly from https://www.modelscope.cn/models/<org>/<repo>
  2. ModelScope Ollama Registry (`ollama pull modelscope.cn/<org>/<repo>:<quant>`) for Asia-Pacific nodes
  3. High-throughput OCI GGUF Mirror (`ollama pull hf.co/<repo>:<quant>`) for US/EU Google Colab H100/A100 VMs
"""
import argparse
import json
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

MODELSCOPE_SUITE_CATALOG = [
    {
        "id": 1,
        "key": "ling3_flash_127b",
        "tier": "H100-80GB / A100-80GB",
        "modelscope_id": "inclusionAI/Ling-3.0-flash",
        "modelscope_url": "https://www.modelscope.cn/models/inclusionAI/Ling-3.0-flash",
        "oci_gguf_ref": "hf.co/mradermacher/LING-3.0-FLASH-ABLITERATED-GGUF:Q3_K_M",
        "ollama_repo": "eriktonon/ling3.0-flash-uncensored",
        "tags": ["127b-a5b", "127b", "latest"],
        "arch": "bailingmoe3",
        "params": "127.5B MoE (5.1B active)",
        "vram_gb": 56.58,
    },
    {
        "id": 2,
        "key": "step35_flash_196b",
        "tier": "H100-80GB / A100-80GB",
        "modelscope_id": "stepfun-ai/Step-3.5-Flash",
        "modelscope_url": "https://www.modelscope.cn/models/stepfun-ai/Step-3.5-Flash",
        "oci_gguf_ref": "hf.co/mradermacher/Step-3.5-Flash-Ablitirated-i1-GGUF:IQ2_M",
        "ollama_repo": "eriktonon/step3.5-flash-uncensored",
        "tags": ["196b-a11b", "196b", "latest"],
        "arch": "step35",
        "params": "196B MoE (11B active)",
        "vram_gb": 59.59,
    },
    {
        "id": 3,
        "key": "ring_flash2_103b",
        "tier": "H100-80GB / A100-80GB",
        "modelscope_id": "inclusionAI/Ring-flash-2.0",
        "modelscope_url": "https://www.modelscope.cn/models/inclusionAI/Ring-flash-2.0",
        "oci_gguf_ref": "hf.co/mradermacher/Ring-flash-2.0-i1-GGUF:Q3_K_M",
        "ollama_repo": "eriktonon/ring-flash2-uncensored",
        "tags": ["103b-a6b", "103b", "latest"],
        "arch": "bailingmoe2",
        "params": "102.9B MoE (6.1B active)",
        "vram_gb": 45.94,
    },
    {
        "id": 4,
        "key": "hunyuan_a13b_80b",
        "tier": "H100-80GB / A100-80GB",
        "modelscope_id": "tencent/Hunyuan-A13B-Instruct",
        "modelscope_url": "https://www.modelscope.cn/models/tencent/Hunyuan-A13B-Instruct",
        "oci_gguf_ref": "hf.co/unsloth/Hunyuan-A13B-Instruct-GGUF:Q4_K_M",
        "ollama_repo": "eriktonon/hunyuan-a13b-uncensored",
        "tags": ["80b-a13b", "80b", "latest"],
        "arch": "hunyuan-moe",
        "params": "80.4B MoE (13B active)",
        "vram_gb": 45.43,
    },
    {
        "id": 5,
        "key": "kimi_linear_48b",
        "tier": "A100-40GB / A100-80GB / H100",
        "modelscope_id": "moonshotai/Kimi-Linear-48B-A3B-Instruct",
        "modelscope_url": "https://www.modelscope.cn/models/moonshotai/Kimi-Linear-48B-A3B-Instruct",
        "oci_gguf_ref": "hf.co/mradermacher/Kimi-Linear-48B-A3B-Instruct-abliterated-bf16-i1-GGUF:Q4_K_M",
        "ollama_repo": "eriktonon/kimi-linear-uncensored",
        "tags": ["48b-a3b", "48b", "latest"],
        "arch": "kimi-linear",
        "params": "49.1B Linear-MoE (3B active, 1M ctx)",
        "vram_gb": 27.66,
    },
    {
        "id": 6,
        "key": "seed_oss_36b",
        "tier": "A100-40GB / A100-80GB / H100",
        "modelscope_id": "ByteDance-Seed/Seed-OSS-36B-Instruct",
        "modelscope_url": "https://www.modelscope.cn/models/ByteDance-Seed/Seed-OSS-36B-Instruct",
        "oci_gguf_ref": "hf.co/mradermacher/Seed-OSS-36B-Instruct-biprojected-norm-preserving-abliterated-i1-GGUF:Q4_K_M",
        "oci_ablation": "biprojected-norm-preserving-abliterated",
        "ollama_repo": "eriktonon/seed-oss-uncensored",
        "tags": ["36b", "latest"],
        "arch": "seed_oss",
        "params": "36.2B Dense (512k ctx)",
        "vram_gb": 20.27,
    },
    {
        "id": 7,
        "key": "baichuan_m2_32b",
        "tier": "A100-40GB / A100-80GB / H100",
        "modelscope_id": "baichuan-inc/Baichuan-M2-32B",
        "modelscope_url": "https://www.modelscope.cn/models/baichuan-inc/Baichuan-M2-32B",
        "oci_gguf_ref": "hf.co/mradermacher/Baichuan-M2-32B-abliterated-i1-GGUF:Q4_K_M",
        "ollama_repo": "eriktonon/baichuan-m2-uncensored",
        "tags": ["32b", "latest"],
        "arch": "qwen2",
        "params": "32.8B Dense Medical/Reasoning SOTA (131k ctx)",
        "vram_gb": 18.49,
    },
    {
        "id": 8,
        "key": "ernie45_moe_21b",
        "tier": "A100-40GB / A100-80GB / H100",
        "modelscope_id": "PaddlePaddle/ERNIE-4.5-21B-A3B-PT",
        "modelscope_url": "https://www.modelscope.cn/models/PaddlePaddle/ERNIE-4.5-21B-A3B-PT",
        "oci_gguf_ref": "hf.co/filvyb/ERNIE-4.5-21B-A3B-Base-PT-heretic-GGUF:Q8_0",
        "ollama_repo": "eriktonon/ernie4.5-moe-uncensored",
        "tags": ["21b-a3b", "21b", "latest"],
        "arch": "ernie4_5-moe",
        "params": "21.8B MoE (3B active, Q8_0 8-bit Heretic)",
        "vram_gb": 21.61,
    },
]


def check_modelscope_meta(modelscope_id: str) -> dict:
    """Query ModelScope Hub metadata or fallback to catalog metadata."""
    org, name = modelscope_id.split("/", 1)
    api_url = f"https://www.modelscope.cn/api/v1/models/{org}/{name}"
    req = urllib.request.Request(api_url, headers={"User-Agent": "Mozilla/5.0 (PE-AGENTS-IBM Foundry)"})
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            payload = json.loads(resp.read().decode("utf-8", "replace"))
            data = payload.get("Data") or {}
            return {
                "reachable": True,
                "modelscope_id": modelscope_id,
                "downloads": data.get("Downloads"),
                "stars": data.get("Stars"),
                "license": data.get("License"),
            }
    except Exception as exc:
        return {"reachable": False, "modelscope_id": modelscope_id, "note": str(exc)}


def download_via_modelscope_sdk(modelscope_id: str, cache_dir: str, allow_patterns: str = "*.json") -> str:
    """Download model artifacts directly from www.modelscope.cn via official SDK."""
    try:
        from modelscope.hub.snapshot_download import snapshot_download
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "modelscope"], check=True)
        from modelscope.hub.snapshot_download import snapshot_download

    patterns = [p.strip() for p in allow_patterns.split(",") if p.strip()]
    local_dir = snapshot_download(model_id=modelscope_id, cache_dir=cache_dir, allow_patterns=patterns)
    return str(local_dir)


def main():
    ap = argparse.ArgumentParser(description="ModelScope H100/A100 Suite Downloader & Catalog Inspector")
    ap.add_argument("--list", action="store_true", help="List all 8 homologated ModelScope ecosystems")
    ap.add_argument("--probe", action="store_true", help="Probe www.modelscope.cn API metadata for all 8 models")
    ap.add_argument("--download-config", type=str, default="", help="Download config/tokenizer JSONs from ModelScope for <org>/<repo>")
    ap.add_argument("--cache-dir", type=str, default="./ms_cache")
    args = ap.parse_args()

    if args.list or (not args.probe and not args.download_config):
        for item in MODELSCOPE_SUITE_CATALOG:
            print(
                f"[{item['id']}] {item['modelscope_id']:42s} | {item['tier']:28s} | "
                f"{item['vram_gb']:5.2f} GB | {item['ollama_repo']}:{item['tags'][0]}"
            )

    if args.probe:
        for item in MODELSCOPE_SUITE_CATALOG:
            meta = check_modelscope_meta(item["modelscope_id"])
            print(json.dumps(meta, ensure_ascii=False))

    if args.download_config:
        Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
        out = download_via_modelscope_sdk(args.download_config, args.cache_dir, "*.json,README.md")
        print(f"Downloaded ModelScope config bundle to: {out}")


if __name__ == "__main__":
    main()
