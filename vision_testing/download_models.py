#!/usr/bin/env python3
"""
Download vision/audio models from HuggingFace.
NO HARDCODED PATHS. Uses models from config.json.

Usage:
  python download_models.py                    # Download all enabled models
  python download_models.py --model LFM2.5-VL-1.6B-Q4_0  # Specific model
  python download_models.py --model "AnyModel"   # Any model in config
"""
import os
import sys
import json
import time
import urllib.request
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

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


def list_hf_files(repo_id: str) -> List[Dict]:
    """List files in a HuggingFace repo via API."""
    url = f"https://huggingface.co/api/models/{repo_id}/tree/main"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "MATHIR-vision"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except Exception as e:
        return []


def download_file(url: str, dest: Path) -> bool:
    """Download a file with progress."""
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
                    if total > 0 and downloaded % (10 * chunk_size) == 0:
                        pct = (downloaded / total) * 100
                        print(f"\r    {downloaded/1024/1024:.0f}/{total/1024/1024:.0f} MB ({pct:.0f}%)", end="", flush=True)
            if total > 0:
                print(f"\r    {downloaded/1024/1024:.0f} MB downloaded", flush=True)
            return True
    except Exception as e:
        print(f"    FAIL: {e}")
        return False


def download_model_files(model_name: str, model_config: dict, models_dir: Path) -> bool:
    """Download a single model's required files."""
    if not model_config.get("huggingface_url"):
        print(f"  [SKIP] {model_name}: no HuggingFace URL")
        return True

    # Extract repo from URL
    url = model_config["huggingface_url"]
    if "huggingface.co/" not in url:
        print(f"  [SKIP] {model_name}: not a HuggingFace URL")
        return True
    repo_id = url.split("huggingface.co/")[-1].rstrip("/")

    print(f"\n[{model_name}]")
    print(f"  HuggingFace: {url}")
    print(f"  Repo: {repo_id}")

    # List available files
    files = list_hf_files(repo_id)
    if not files:
        print(f"  ERROR: could not list files")
        return False

    # Files to download (based on config)
    files_to_download = []
    for key in ["path", "mmproj", "tokenizer", "vocoder"]:
        v = model_config.get(key)
        if v:
            # v is relative to HERE/models_dir or just relative path
            # The convention is: <repo_basename>/<filename>
            # E.g., "models/LFM2.5-VL-1.6B-GGUF/LFM2.5-VL-1.6B-Q4_0.gguf"
            # We need just the filename from the repo
            filename = Path(v).name
            files_to_download.append((key, filename))

    print(f"  Files to download: {len(files_to_download)}")
    success = True
    for key, filename in files_to_download:
        # Determine destination - same dir as model_config["path"]
        model_path = resolve_path(model_config["path"])
        dest = model_path.parent / filename

        if dest.exists():
            size = dest.stat().st_size / 1024 / 1024
            print(f"  [SKIP] {key} = {filename} (exists, {size:.0f} MB)")
            continue

        # Find file in HF listing
        hf_file = next((f for f in files if f["path"] == filename), None)
        if not hf_file:
            print(f"  [WARN] {filename} not in HF repo, trying closest match...")
            # Try to find by basename
            candidates = [f for f in files if Path(f["path"]).name == filename]
            if candidates:
                hf_file = candidates[0]
            else:
                print(f"  [FAIL] {filename} not found")
                success = False
                continue

        dl_url = f"https://huggingface.co/{repo_id}/resolve/main/{hf_file['path']}"
        size_mb = hf_file.get("size", 0) / 1024 / 1024
        print(f"  Downloading {key} = {filename} ({size_mb:.0f} MB)...")
        if not download_file(dl_url, dest):
            success = False

    return success


def main():
    parser = argparse.ArgumentParser(description="Download vision/audio models from HuggingFace")
    parser.add_argument("--model", help="Specific model name to download (default: all enabled)")
    parser.add_argument("--list", action="store_true", help="List available models without downloading")
    args = parser.parse_args()

    config = load_config()
    models_dir = resolve_path(config["models_dir"])
    models = config.get("models", {})

    print("=" * 70)
    print("VISION/AUDIO MODEL DOWNLOADER")
    print("=" * 70)
    print(f"Models dir: {models_dir}")
    print(f"Config: {CONFIG_PATH}")

    if args.list:
        print(f"\nConfigured models:")
        for name, m in models.items():
            enabled = "ON" if m.get("enabled", True) else "OFF"
            print(f"  [{enabled}] {name}: {m.get('display_name', name)}")
            print(f"          {m.get('huggingface_url', '(no URL)')}")
        return 0

    # Filter
    to_download = {}
    if args.model:
        if args.model not in models:
            print(f"ERROR: Model '{args.model}' not in config.")
            print(f"Available: {list(models.keys())}")
            return 1
        to_download[args.model] = models[args.model]
    else:
        to_download = {k: v for k, v in models.items() if v.get("enabled", True)}

    if not to_download:
        print("No enabled models. Add one with:")
        print('  python model_manager.py hf-add --hf LiquidAI/LFM2.5-VL-1.6B-GGUF')
        return 0

    print(f"\nDownloading {len(to_download)} model(s):")
    total_success = 0
    for name, m in to_download.items():
        if download_model_files(name, m, models_dir):
            total_success += 1

    print(f"\n{'='*70}")
    print(f"Downloaded {total_success}/{len(to_download)} models")
    print("=" * 70)

    return 0 if total_success == len(to_download) else 1


if __name__ == "__main__":
    sys.exit(main())