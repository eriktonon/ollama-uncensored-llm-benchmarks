#!/usr/bin/env python3
"""
Suite 11 — Qwen-Image-2.1-Turbo Visual Generation & Multi-Reference Editing Benchmark (RTX 5080)
================================================================================================
Executes 6 deterministic visual evaluation scenarios against the live J.A.R.V.I.S. Caminho A+ GPU
worker (`http://127.0.0.1:8188/api/generate_sync`):
  - IMG-01: Photorealism & Exact Typography ("J.A.R.V.I.S. A+")
  - IMG-02: Editorial Hardware Photography (Native PT-BR Prompt)
  - IMG-03: Native RGBA Transparent Background (Alpha Channel 0..255 Verification)
  - IMG-04: Technical Architecture Poster & Multi-Label Typography (1280x736)
  - IMG-05: Image-to-Image Multi-Reference Lighting Transformation (Ref = IMG-02)
  - IMG-06: Cinematic 85mm Portrait, Natural Skin Texture & Bokeh Lighting

Outputs real showcase PNGs to `training/11_qwen_image_2_1_turbo_uncensored/assets/` and
machine-readable telemetry JSON to `training/11_qwen_image_2_1_turbo_uncensored/visual_benchmark_results.json`.
"""

from __future__ import annotations

import base64
import json
import time
import urllib.request
from pathlib import Path
from PIL import Image

SUITE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = SUITE_DIR / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)
WORKER_URL = "http://127.0.0.1:8188/api/generate_sync"

SCENARIOS = [
    {
        "id": "IMG-01",
        "domain": "Photorealism & Typography",
        "filename": "qwen_img_01_typography_studio.png",
        "prompt": (
            'Photorealistic futuristic homelab GPU server box in a dark minimalist studio with glowing '
            'cyan glass and sharp white LED typography reading "J.A.R.V.I.S. A+" on the GPU shroud, '
            'liquid cooling reflections, studio rim light, 8k photography'
        ),
        "width": 1024,
        "height": 1024,
        "transparent": False,
        "seed": 20261123,
        "refs": [],
    },
    {
        "id": "IMG-02",
        "domain": "Editorial Hardware (PT-BR Prompt)",
        "filename": "qwen_img_02_editorial_ptbr.png",
        "prompt": (
            "Fotografia editorial de um servidor datacenter minimalista em aco escovado com LED azul IBM, "
            "iluminacao de estudio, foco nitido, reflexos sutis no chao cinza claro"
        ),
        "width": 1024,
        "height": 1024,
        "transparent": False,
        "seed": 42,
        "refs": [],
    },
    {
        "id": "IMG-03",
        "domain": "Native RGBA Transparency (Alpha)",
        "filename": "qwen_img_03_native_rgba.png",
        "prompt": (
            "Minimalist brushed titanium cybernetic shield emblem with glowing electric blue circuit lines, "
            "studio 3D product icon, isolated object with clean edges on transparent RGBA background, no text"
        ),
        "width": 1024,
        "height": 1024,
        "transparent": True,
        "seed": 20261125,
        "refs": [],
    },
    {
        "id": "IMG-04",
        "domain": "Architecture Poster & Typography",
        "filename": "qwen_img_04_architecture_poster.png",
        "prompt": (
            'Modern dark-mode IBM Carbon style technical architecture poster on deep slate background, '
            'clean dark cards with cyan borders connected by glowing arrows reading "RUST CLI", '
            '"QWEN-IMAGE 2.1", and "RTX 5080 16GB", top bold header reading "HOMELAB AI ENGINE", '
            'crisp geometric sans-serif typography, clean vector infographic layout'
        ),
        "width": 1280,
        "height": 736,
        "transparent": False,
        "seed": 20261126,
        "refs": [],
    },
    {
        "id": "IMG-05",
        "domain": "Image-to-Image Reference Editing",
        "filename": "qwen_img_05_image_editing.png",
        "prompt": (
            "Keep the exact brushed steel datacenter server rack structure and proportions from the input image, "
            "transform the studio lighting into dramatic golden hour sunset with warm amber sunbeams from the left "
            "and warm orange-gold glowing LED accents inside the rack"
        ),
        "width": 1024,
        "height": 1024,
        "transparent": False,
        "seed": 20261127,
        "refs": ["IMG-02"],
    },
    {
        "id": "IMG-06",
        "domain": "Cinematic 85mm Portrait & Lighting",
        "filename": "qwen_img_06_cinematic_portrait.png",
        "prompt": (
            "Editorial 85mm f/1.4 candid portrait of a senior robotics engineer inspecting a carbon-fiber "
            "robotic hand in a sunlit modern laboratory, natural skin texture, warm window backlight, "
            "subtle depth of field bokeh, authentic Leica film photography look"
        ),
        "width": 1024,
        "height": 1024,
        "transparent": False,
        "seed": 20261128,
        "refs": [],
    },
]


def main() -> None:
    results = []
    saved_paths: dict[str, Path] = {}

    for sc in SCENARIOS:
        ref_paths = [str(saved_paths[r].resolve()) for r in sc["refs"]]
        payload: dict = {
            "prompt": sc["prompt"],
            "mode": "image_edit" if ref_paths else "image_gen",
            "width": sc["width"],
            "height": sc["height"],
            "transparent": sc["transparent"],
            "seed": sc["seed"],
        }
        if ref_paths:
            payload["reference_images"] = ref_paths

        t0 = time.time()
        req = urllib.request.Request(
            WORKER_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=240) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        png_bytes = base64.b64decode(data["image_base64"])
        out_p = ASSETS_DIR / sc["filename"]
        out_p.write_bytes(png_bytes)
        saved_paths[sc["id"]] = out_p

        im = Image.open(out_p)
        alpha_pct = 0.0
        alpha_min, alpha_max = 255, 255
        if im.mode == "RGBA":
            a = im.getchannel("A")
            alpha_min, alpha_max = a.getextrema()
            flat = a.get_flattened_data() if hasattr(a, "get_flattened_data") else a.getdata()
            alpha_pct = round(sum(1 for v in flat if v < 16) / (im.width * im.height) * 100.0, 1)

        rec = {
            "id": sc["id"],
            "domain": sc["domain"],
            "filename": sc["filename"],
            "resolution": f"{im.width}x{im.height}",
            "color_mode": im.mode,
            "alpha_range": [alpha_min, alpha_max],
            "transparent_pct": alpha_pct,
            "steps": data.get("steps", 8),
            "elapsed_s": data.get("elapsed_s", round(time.time() - t0, 2)),
            "size_kb": round(len(png_bytes) / 1024.0, 1),
            "vram_free_mb": data.get("vram_free_mb", 0),
            "vram_total_mb": data.get("vram_total_mb", 0),
            "seed": sc["seed"],
            "reference_count": len(ref_paths),
            "engine": data.get("engine", "qwen_image_2_1_turbo_rtx5080"),
            "model": data.get("model", "Qwen/Qwen-Image-2.1-Turbo"),
            "quant": data.get("quant", "nf4"),
            "prompt": sc["prompt"],
        }
        results.append(rec)
        print(
            f"[BENCH] {rec['id']} | {rec['domain']} | {rec['resolution']} {rec['color_mode']} "
            f"(alpha={alpha_pct}%) | {rec['elapsed_s']}s | {rec['size_kb']}KB | "
            f"VRAM free={rec['vram_free_mb']}/{rec['vram_total_mb']} MiB",
            flush=True,
        )

    out_json = SUITE_DIR / "visual_benchmark_results.json"
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"VISUAL_BENCHMARK_6_OF_6_DONE -> {out_json}", flush=True)


if __name__ == "__main__":
    main()
