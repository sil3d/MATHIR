#!/usr/bin/env python3
"""
Quick download: fetches Q4_0 quantized files for all enabled models in config.json.
NO HARDCODED PATHS. Reads models from config.json.
"""
import os
import sys
import json
import time
import urllib.request
import re
from pathlib import Path
from datetime import datetime

HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "config.json"


def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


def resolve_path(p):
    pp = Path(p)
    if pp.is_absolute():
        return pp
    return (HERE / pp).resolve()


def list_hf_files(repo_id):
    url = f"https://huggingface.co/api/models/{repo_id}/tree/main"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "MATHIR-vision"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except Exception as e:
        return []


def download_with_progress(url, dest):
    print(f"  Downloading: {dest.name}")
    start = time.time()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "MATHIR-vision"})
        with urllib.request.urlopen(req, timeout=600) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 1024 * 1024
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = (downloaded / total) * 100
                        mb = downloaded / 1024 / 1024
                        total_mb = total / 1024 / 1024
                        print(f"\r    {mb:.0f}/{total_mb:.0f} MB ({pct:.0f}%)", end="", flush=True)
            duration = time.time() - start
            size_mb = downloaded / 1024 / 1024
            print(f"\n    OK ({size_mb:.0f} MB in {duration:.1f}s)")
            return True
    except Exception as e:
        print(f"\n    FAIL: {e}")
        return False


def find_q4_or_smallest(files, prefer_q4=True):
    """Find Q4_0 file, or smallest if none found."""
    gguf_files = [f for f in files if f["path"].endswith(".gguf") and "mmproj" not in f["path"].lower()
                  and "tokenizer" not in f["path"].lower() and "vocoder" not in f["path"].lower()]

    if not gguf_files:
        return None

    if prefer_q4:
        q4 = [f for f in gguf_files if "Q4_0" in f["path"]]
        if q4:
            return sorted(q4, key=lambda f: f.get("size", 0))[0]

    return sorted(gguf_files, key=lambda f: f.get("size", 0))[0]


def find_matching_component(files, base_filename, suffix):
    """Find mmproj/tokenizer/vocoder that matches the main model."""
    base = base_filename.replace(".gguf", "")
    # Try to find matching component
    pattern = re.compile(re.escape(base) + r".*" + re.escape(suffix), re.IGNORECASE)
    matches = [f for f in files if pattern.search(f["path"])]
    if matches:
        # Prefer Q4_0 or Q8_0 for memory efficiency
        q4 = [f for f in matches if "Q4_0" in f["path"] or "Q8_0" in f["path"]]
        if q4:
            return sorted(q4, key=lambda f: f.get("size", 0))[0]
        return sorted(matches, key=lambda f: f.get("size", 0))[0]
    return None


def main():
    print("=" * 70)
    print("Q4_0 AUTO-DOWNLOAD (smallest, fits 8GB VRAM)")
    print("=" * 70)

    config = load_config()
    models = config.get("models", {})

    enabled = {k: v for k, v in models.items() if v.get("enabled", True)}
    if not enabled:
        print("No enabled models. Add some with:")
        print('  python model_manager.py hf-add --hf LiquidAI/LFM2.5-VL-1.6B-GGUF')
        return 1

    total_files = 0
    total_size = 0
    success = 0

    for model_name, m in enabled.items():
        if not m.get("huggingface_url"):
            continue

        print(f"\n[{model_name}]")
        url = m["huggingface_url"]
        if "huggingface.co/" not in url:
            continue
        repo_id = url.split("huggingface.co/")[-1].rstrip("/")
        print(f"  Repo: {repo_id}")

        # List files
        files = list_hf_files(repo_id)
        if not files:
            print(f"  ERROR: could not list files")
            continue

        # Find main model
        main = find_q4_or_smallest(files)
        if not main:
            print(f"  No GGUF main file found")
            continue

        model_path = resolve_path(m["path"])
        if not model_path.exists():
            if not download_with_progress(
                f"https://huggingface.co/{repo_id}/resolve/main/{main['path']}",
                model_path
            ):
                continue
            success += 1
            total_files += 1
            total_size += main.get("size", 0)
        else:
            print(f"  [SKIP] {model_path.name} (exists)")

        # Optional components
        for component in ["mmproj", "tokenizer", "vocoder"]:
            comp_file = find_matching_component(files, main["path"], component)
            if not comp_file:
                continue
            comp_path = resolve_path(m.get(component, f"models/placeholder/{comp_file['path']}"))
            if not comp_path.exists():
                print(f"  [{component}] downloading {comp_file['path']}")
                if download_with_progress(
                    f"https://huggingface.co/{repo_id}/resolve/main/{comp_file['path']}",
                    comp_path
                ):
                    total_files += 1
                    total_size += comp_file.get("size", 0)
            else:
                print(f"  [{component}] {comp_path.name} exists")

    print(f"\n{'='*70}")
    print(f"DONE: {total_files} files, {total_size/1024/1024/1024:.2f} GB total")
    print("=" * 70)
    print(f"\nNext: python model_manager.py validate")
    print(f"      python vision_test.py")


if __name__ == "__main__":
    main()