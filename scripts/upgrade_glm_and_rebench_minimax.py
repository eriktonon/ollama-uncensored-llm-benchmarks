#!/usr/bin/env python3
"""
Suite 11 — Foundry Law #1 Upgrade (`glm-image-uncensored`) + KV-Cache Optimization (`minimax-h3-image-uncensored`)
==================================================================================================================
1. Upgrades `eriktonon/glm-image-uncensored` from 6/7-refusing `huihui-ai.GLM-4-9B-0414-abliterated` to
   `bartowski/glm-4-9b-chat-abliterated-GGUF` (`Q4_K_M`, 5.82 GB) with `num_ctx 32768` (zero CPU spill),
   runs the mandatory 7-Prompt Zero-Refusal Gate + 4-Check Anti-Hallucination, and pushes ONLY on 7/7 PASS.
2. Optimizes `eriktonon/minimax-h3-image-uncensored` (`num_ctx 65536 -> 32768`) so KV-cache + Qwen3-VL-8B
   + mmproj-Q8_0 fit 100% in RTX 5080 16GB GDDR7 VRAM (jumping from 29 tok/s to 110+ tok/s), re-verifies 7/7 PASS,
   re-pushes to Ollama Hub, and updates `multi_image_ollama_benchmark.json` in place.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_and_publish_all_image_models as base

BARTOWSKI_GLM4 = base.WORK_DIR / "glm-4-9b-chat-abliterated-Q4_K_M.gguf"


def rebuild_gate_push(spec: dict, num_ctx: int = 32768, do_push: bool = True) -> dict:
    repo = spec["repo"]
    tags = spec["tags"]
    primary = f"eriktonon/{repo}:{tags[0]}"
    print(f"\n==================== REBUILDING {primary} (num_ctx={num_ctx}) ====================", flush=True)

    gguf_to_use: Path = spec["gguf"]
    if not gguf_to_use.exists():
        raise FileNotFoundError(f"Missing GGUF: {gguf_to_use}")

    from_lines = [f"FROM {gguf_to_use}"]
    if spec["mmproj"] is not None:
        from_lines.append(f"FROM {spec['mmproj']}")

    stop_lines = "\n".join(f'PARAMETER stop "{s}"' for s in spec["stops"])
    lic_block = (
        f"# eriktonon/{repo} ({spec['display_name']})\n"
        f"- Ollama Hub: https://ollama.com/eriktonon/{repo}\n"
        f"- Upstream Image Model: {spec['upstream_image_model']} (ModelScope: {spec['modelscope_repo']})\n"
        f"- Abliterated Encoder: {spec['upstream_encoder']}\n"
        f"- License: {spec['license_spdx']}\n"
        f"- Benchmark: {spec['arena_elo']} | 7/7 Zero-Refusal Gate (THINK_LEN=0)\n"
    )
    mf_text = (
        "\n".join(from_lines)
        + f'\n\nTEMPLATE """{spec["template"]}"""\n\n'
        + f'SYSTEM """{spec["system"]}"""\n\n'
        + "PARAMETER temperature 0.5\n"
        + "PARAMETER top_p 0.92\n"
        + "PARAMETER top_k 40\n"
        + "PARAMETER repeat_penalty 1.12\n"
        + f"PARAMETER num_ctx {num_ctx}\n"
        + stop_lines
        + f'\n\nLICENSE """{lic_block}"""\n'
    )
    mf_path = base.WORK_DIR / f"Modelfile.{repo}.v2"
    mf_path.write_text(mf_text, encoding="utf-8")

    subprocess.run(
        ["chown", "-R", "ollama:ollama", str(base.MF_V1_ROOT / repo), str(base.MF_V2_ROOT / repo)],
        stderr=subprocess.DEVNULL,
        check=False,
    )
    subprocess.run(["ollama", "create", primary, "-f", str(mf_path)], check=True)
    for t in tags[1:]:
        subprocess.run(["ollama", "cp", primary, f"eriktonon/{repo}:{t}"], check=True)

    digest = base.strip_renderer_and_sync_manifests(repo, tags)
    gate, halluc = base.run_7_gate_and_hallucination(primary)
    subprocess.run(["ollama", "stop", primary], check=False)

    pushed = False
    if gate["passed"] == gate["total"]:
        if do_push:
            print(f"  [PUSH] 7/7 PASS! Pushing {repo} ({tags}) to Ollama Hub...", flush=True)
            for t in tags:
                subprocess.run(["ollama", "push", f"eriktonon/{repo}:{t}"], check=True)
            pushed = True
    else:
        print(f"  [WARN] {primary} scored {gate['passed']}/7 — BLOCKED from push!", flush=True)

    return {
        "repo": f"eriktonon/{repo}",
        "tags": [f"eriktonon/{repo}:{t}" for t in tags],
        "display_name": spec["display_name"],
        "upstream_image_model": spec["upstream_image_model"],
        "modelscope_repo": spec["modelscope_repo"],
        "upstream_encoder": spec["upstream_encoder"],
        "has_vision_mmproj": spec["mmproj"] is not None,
        "license": spec["license_spdx"],
        "arena_elo": spec["arena_elo"],
        "ollama_model_digest": digest,
        "patch_info": {"found": False, "patched": False, "slen": 0, "offset": -1},
        "zero_refusal_gate": gate,
        "hallucination": halluc,
        "pushed_to_ollama_hub": pushed,
        "law1_upgrade_note": spec.get("law1_upgrade_note", ""),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def main() -> None:
    do_push = "--push" in sys.argv
    existing = json.loads(base.OUT_JSON.read_text(encoding="utf-8"))
    by_repo = {r["repo"]: r for r in existing}

    # 1. Upgrade glm-image-uncensored with bartowski/glm-4-9b-chat-abliterated-Q4_K_M.gguf
    glm_spec = dict(next(s for s in base.MODELS_SPEC if s["repo"] == "glm-image-uncensored"))
    glm_spec["gguf"] = BARTOWSKI_GLM4
    glm_spec["upstream_encoder"] = "bartowski/glm-4-9b-chat-abliterated-GGUF (Q4_K_M · Law #1 Upgraded)"
    glm_spec["law1_upgrade_note"] = (
        "Baseline DevQuasar-12/huihui-ai.GLM-4-9B-0414-abliterated-GGUF scored 6/7 (refused P1_SHORT) and was "
        "auto-blocked per Foundry Law #1; upgraded to bartowski/glm-4-9b-chat-abliterated-GGUF (Q4_K_M)."
    )
    by_repo["eriktonon/glm-image-uncensored"] = rebuild_gate_push(glm_spec, num_ctx=32768, do_push=do_push)

    # 2. Optimize minimax-h3-image-uncensored to num_ctx=32768 for 100% GPU VRAM fit (110+ tok/s)
    mm_spec = dict(next(s for s in base.MODELS_SPEC if s["repo"] == "minimax-h3-image-uncensored"))
    mm_spec["law1_upgrade_note"] = "Tuned num_ctx 65536->32768 to fit Qwen3-VL-8B + mmproj-Q8_0 100% in RTX 5080 16GB GDDR7 (eliminating partial CPU offload)."
    by_repo["eriktonon/minimax-h3-image-uncensored"] = rebuild_gate_push(mm_spec, num_ctx=32768, do_push=do_push)

    order = [
        "eriktonon/minimax-h3-image-uncensored",
        "eriktonon/z-image-turbo-uncensored",
        "eriktonon/flux2-klein-uncensored",
        "eriktonon/glm-image-uncensored",
        "eriktonon/moss-image-uncensored",
    ]
    updated = [by_repo[k] for k in order if k in by_repo]
    base.OUT_JSON.write_text(json.dumps(updated, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nUPGRADE_AND_REBENCH_DONE -> {base.OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
