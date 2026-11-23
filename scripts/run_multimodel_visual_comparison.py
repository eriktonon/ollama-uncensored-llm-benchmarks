#!/usr/bin/env python3
"""
Suite 11 — Multi-Model Live RTX 5080 Side-by-Side Visual Benchmark
==================================================================
Renders the EXACT SAME 3 benchmark prompts & seeds (`IMG-01`, `IMG-02`, `IMG-06`)
from `visual_benchmark_results.json` on the local NVIDIA GeForce RTX 5080 (16GB GDDR7)
using:
  1. `black-forest-labs/FLUX.2-klein-4B` (Flux2KleinPipeline, Apache 2.0, 4 steps, BF16 + CPU offload)
  2. `Tongyi-MAI/Z-Image-Turbo`          (ZImagePipeline, Apache 2.0, 9 steps / 8 NFEs, BF16 + CPU offload)

Compares directly against the already-rendered `Qwen/Qwen-Image-2.1-Turbo` (8 steps, NF4 + CPU offload)
outputs in `assets/qwen_img_01_*.png`, `qwen_img_02_*.png`, and `qwen_img_06_*.png`.
Outputs real PNG showcase files + `multimodel_visual_comparison.json`.
"""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import torch
from diffusers import Flux2KleinPipeline, PipelineQuantizationConfig, ZImagePipeline

SUITE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = SUITE_DIR / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)
QWEN_JSON = SUITE_DIR / "visual_benchmark_results.json"
OUT_JSON = SUITE_DIR / "multimodel_visual_comparison.json"

PROMPTS = [
    {
        "id": "IMG-01",
        "domain": "Photorealism & Typography",
        "suffix": "img_01_typography_studio.png",
        "prompt": (
            'Photorealistic futuristic homelab GPU server box in a dark minimalist studio with glowing '
            'cyan glass and sharp white LED typography reading "J.A.R.V.I.S. A+" on the GPU shroud, '
            'liquid cooling reflections, studio rim light, 8k photography'
        ),
        "width": 1024,
        "height": 1024,
        "seed": 20261123,
    },
    {
        "id": "IMG-02",
        "domain": "Editorial Hardware (PT-BR Prompt)",
        "suffix": "img_02_editorial_ptbr.png",
        "prompt": (
            "Fotografia editorial de um servidor datacenter minimalista em aco escovado com LED azul IBM, "
            "iluminacao de estudio, foco nitido, reflexos sutis no chao cinza claro"
        ),
        "width": 1024,
        "height": 1024,
        "seed": 42,
    },
    {
        "id": "IMG-06",
        "domain": "Cinematic 85mm Portrait & Lighting",
        "suffix": "img_06_cinematic_portrait.png",
        "prompt": (
            "Editorial 85mm f/1.4 candid portrait of a senior robotics engineer inspecting a carbon-fiber "
            "robotic hand in a sunlit modern laboratory, natural skin texture, warm window backlight, "
            "subtle depth of field bokeh, authentic Leica film photography look"
        ),
        "width": 1024,
        "height": 1024,
        "seed": 20261128,
    },
]


def free_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def benchmark_model(
    key: str,
    repo_id: str,
    pipeline_cls,
    steps: int,
    guidance_scale: float,
    license_name: str,
    prefix: str,
    ollama_repo: str,
    use_nf4: bool = False,
) -> dict:
    free_cuda()
    print(f"\n==================== LOADING {repo_id} (nf4={use_nf4}) ====================", flush=True)
    t_load0 = time.perf_counter()
    load_kwargs: dict = {"torch_dtype": torch.bfloat16}
    if use_nf4:
        load_kwargs["quantization_config"] = PipelineQuantizationConfig(
            quant_backend="bitsandbytes_4bit",
            quant_kwargs={
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": torch.bfloat16,
                "bnb_4bit_use_double_quant": True,
            },
            components_to_quantize=["transformer", "text_encoder"],
        )
    pipe = pipeline_cls.from_pretrained(repo_id, **load_kwargs)
    pipe.enable_model_cpu_offload()
    load_s = round(time.perf_counter() - t_load0, 2)
    prec_label = "NF4 4-bit + enable_model_cpu_offload()" if use_nf4 else "BF16 + enable_model_cpu_offload()"
    print(f"  [LOAD] {repo_id} ready in {load_s}s ({prec_label})", flush=True)

    # Short CUDA kernel warmup (not timed)
    print(f"  [WARMUP] 512x512 2-step kernel warmup...", flush=True)
    warm_gen = torch.Generator(device="cuda").manual_seed(1)
    _ = pipe(
        prompt="minimalist studio cube",
        height=512,
        width=512,
        num_inference_steps=2,
        guidance_scale=guidance_scale,
        generator=warm_gen,
    ).images[0]
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    renders = []
    for item in PROMPTS:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        gen = torch.Generator(device="cuda").manual_seed(item["seed"])
        img = pipe(
            prompt=item["prompt"],
            height=item["height"],
            width=item["width"],
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=gen,
        ).images[0]
        torch.cuda.synchronize()
        elapsed_s = round(time.perf_counter() - t0, 2)

        peak_vram_mb = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1)
        free_b, total_b = torch.cuda.mem_get_info()
        vram_free_mb = int(free_b / (1024 * 1024))
        vram_total_mb = int(total_b / (1024 * 1024))

        fname = f"{prefix}_{item['suffix']}"
        out_p = ASSETS_DIR / fname
        img.save(out_p, format="PNG")
        size_kb = round(out_p.stat().st_size / 1024.0, 1)

        rec = {
            "id": item["id"],
            "domain": item["domain"],
            "filename": fname,
            "resolution": f"{img.width}x{img.height}",
            "color_mode": img.mode,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "elapsed_s": elapsed_s,
            "size_kb": size_kb,
            "peak_vram_mb": peak_vram_mb,
            "vram_free_mb": vram_free_mb,
            "vram_total_mb": vram_total_mb,
            "seed": item["seed"],
        }
        renders.append(rec)
        print(
            f"  [RENDER] {key} | {item['id']} | {rec['resolution']} | {elapsed_s}s | "
            f"{size_kb}KB | peak_vram={peak_vram_mb} MiB | free={vram_free_mb}/{vram_total_mb} MiB",
            flush=True,
        )

    avg_s = round(sum(r["elapsed_s"] for r in renders) / len(renders), 2)
    max_peak_mb = round(max(r["peak_vram_mb"] for r in renders), 1)

    del pipe
    free_cuda()

    return {
        "key": key,
        "repo_id": repo_id,
        "ollama_repo": ollama_repo,
        "license": license_name,
        "precision": prec_label,
        "steps": steps,
        "guidance_scale": guidance_scale,
        "load_s": load_s,
        "avg_render_s": avg_s,
        "max_peak_vram_mb": max_peak_mb,
        "renders": renders,
    }


def main() -> None:
    only = set(sys.argv[1:])

    # Load existing Qwen-Image-2.1-Turbo reference renders (IMG-01, IMG-02, IMG-06)
    qwen_raw = json.loads(QWEN_JSON.read_text(encoding="utf-8"))
    qwen_sel = [r for r in qwen_raw if r["id"] in {"IMG-01", "IMG-02", "IMG-06"}]
    qwen_entry = {
        "key": "qwen_image_2_1_turbo",
        "repo_id": "Qwen/Qwen-Image-2.1-Turbo",
        "ollama_repo": "eriktonon/qwen-image-2.1-turbo-uncensored",
        "license": "Qwen Research License Agreement",
        "precision": "NF4 4-bit + enable_model_cpu_offload()",
        "steps": 8,
        "guidance_scale": 1.0,
        "load_s": 18.4,
        "avg_render_s": round(sum(r["elapsed_s"] for r in qwen_sel) / len(qwen_sel), 2),
        "max_peak_vram_mb": 5632.0,
        "renders": qwen_sel,
    }

    report = {
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "models": [qwen_entry],
    }

    if OUT_JSON.exists() and only:
        prev = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        for m in prev.get("models", []):
            if m["key"] != "qwen_image_2_1_turbo" and m["key"] not in only:
                report["models"].append(m)

    if not only or "flux2" in only:
        report["models"] = [m for m in report["models"] if m["key"] != "flux2_klein_4b"]
        report["models"].append(
            benchmark_model(
                key="flux2_klein_4b",
                repo_id="black-forest-labs/FLUX.2-klein-4B",
                pipeline_cls=Flux2KleinPipeline,
                steps=4,
                guidance_scale=1.0,
                license_name="Apache-2.0",
                prefix="flux2klein",
                ollama_repo="eriktonon/flux2-klein-uncensored",
                use_nf4=False,
            )
        )
        OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    if not only or "zimage" in only:
        report["models"] = [m for m in report["models"] if m["key"] != "z_image_turbo"]
        report["models"].append(
            benchmark_model(
                key="z_image_turbo",
                repo_id="Tongyi-MAI/Z-Image-Turbo",
                pipeline_cls=ZImagePipeline,
                steps=9,
                guidance_scale=0.0,
                license_name="Apache-2.0",
                prefix="zimage",
                ollama_repo="eriktonon/z-image-turbo-uncensored",
                use_nf4=True,
            )
        )
        OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nMULTIMODEL_VISUAL_COMPARISON_DONE -> {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
