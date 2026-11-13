#!/usr/bin/env python3
"""
Pushes pure :XXb parameter tag aliases to registry.ollama.ai for all MoE / quantized
repositories under eriktonon/* so Ollama Hub renders the blue parameter badge pills
(139b, 106b, 74b, 35b, 30b, 26b, 24b, 22b) on https://ollama.com/eriktonon.
Uses ext4 sparse blob placeholders so ollama push verifies existing server-side layers
via HEAD without re-downloading or re-uploading multi-GB GGUF blobs.
"""
import json
import os
import subprocess
import urllib.request

ALIASES = [
    ("eriktonon/minimax-m2.7-uncensored", "139b-a10b", "139b"),
    ("eriktonon/glm4.5-air-uncensored", "106b-a12b", "106b"),
    ("eriktonon/nemotron3.5-bolt-uncensored", "74b-a9b", "74b"),
    ("eriktonon/qwen3.6-moe-uncensored", "35b-a3b", "35b"),
    ("eriktonon/qwen3-coder-uncensored", "30b-a3b", "30b"),
    ("eriktonon/glm4.7-flash-uncensored", "30b-a3b", "30b"),
    ("eriktonon/moonlight-dusk-uncensored", "26b-a4b", "26b"),
    ("eriktonon/gemma4-uncensored", "26b-a4b", "26b"),
    ("eriktonon/lfm2.5-uncensored", "24b-a2b", "24b"),
    ("eriktonon/ltx-2.5-uncensored", "22b-int8", "22b"),
]

OLLAMA_MODELS_DIRS = [
    "/usr/share/ollama/.ollama/models",
    os.path.expanduser("~/.ollama/models"),
]


def main():
    for repo, src_tag, dst_tag in ALIASES:
        print(f"\n=== Aliasing {repo}:{src_tag} -> {repo}:{dst_tag} ===", flush=True)
        url = f"https://registry.ollama.ai/v2/{repo}/manifests/{src_tag}"
        req = urllib.request.Request(
            url,
            headers={"Accept": "application/vnd.docker.distribution.manifest.v2+json"},
        )
        raw = urllib.request.urlopen(req, timeout=30).read()
        manifest = json.loads(raw.decode("utf-8"))

        created_files = []
        try:
            for base_dir in OLLAMA_MODELS_DIRS:
                if not os.path.exists(os.path.dirname(base_dir)):
                    continue
                manifest_dir = os.path.join(base_dir, "manifests", "registry.ollama.ai", repo)
                blob_dir = os.path.join(base_dir, "blobs")
                os.makedirs(manifest_dir, exist_ok=True)
                os.makedirs(blob_dir, exist_ok=True)

                for item in [manifest["config"]] + manifest["layers"]:
                    digest = item["digest"].replace(":", "-")
                    bpath = os.path.join(blob_dir, digest)
                    if not os.path.exists(bpath):
                        with open(bpath, "wb") as f:
                            f.truncate(item["size"])
                        created_files.append(bpath)

                mpath = os.path.join(manifest_dir, dst_tag)
                with open(mpath, "wb") as f:
                    f.write(raw)
                created_files.append(mpath)

            subprocess.run(["ollama", "push", f"{repo}:{dst_tag}"], check=True)
            print(f"[OK] Pushed {repo}:{dst_tag}", flush=True)
        finally:
            for p in created_files:
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass


if __name__ == "__main__":
    main()
