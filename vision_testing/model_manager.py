#!/usr/bin/env python3
"""
MATHIR Vision Model Manager
============================

User-friendly tool to:
- List installed models from config.json
- Add new GGUF models (any model, any provider)
- Remove models
- Enable/disable models
- Validate model files

NO HARDCODED PATHS. All from config.json (relative to script).

Usage:
  python model_manager.py list
  python model_manager.py add --name "MyModel" --type vision --path models/MyModel/file.gguf
  python model_manager.py add --name "QwenVL" --type vision --hf LiquidAI/LFM2.5-VL-1.6B-GGUF
  python model_manager.py remove --name "MyModel"
  python model_manager.py enable --name "MyModel"
  python model_manager.py validate
"""
import os
import sys
import json
import argparse
import urllib.request
from pathlib import Path
from datetime import datetime

HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "config.json"


def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


def save_config(config):
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)


def resolve_path(p):
    pp = Path(p)
    if pp.is_absolute():
        return pp
    return (HERE / pp).resolve()


def cmd_list(args):
    """List all models in config."""
    config = load_config()
    models = config.get("models", {})

    print("=" * 70)
    print("MODELS (from config.json)")
    print("=" * 70)

    if not models:
        print("\nNo models configured. Add one with:")
        print('  python model_manager.py add --name "MyModel" --type vision --path models/MyModel/file.gguf')
        return

    for name, m in models.items():
        status = "ON " if m.get("enabled", True) else "OFF"
        path = resolve_path(m["path"]) if m.get("path") else None
        file_exists = path.exists() if path else False
        ready = "READY" if file_exists else "MISSING"

        supports = []
        if m.get("supports_vision"): supports.append("vision")
        if m.get("supports_audio"): supports.append("audio")
        if not supports:
            supports.append(m.get("type", "?"))

        print(f"\n  [{ready}] [{status}] {name}")
        print(f"          display: {m.get('display_name', name)}")
        print(f"          type: {m.get('type')}, supports: {', '.join(supports)}")
        if m.get("description"):
            print(f"          desc: {m['description']}")
        if path:
            exists = "EXISTS" if file_exists else "NOT FOUND"
            print(f"          path: {m['path']} ({exists})")
        if m.get("huggingface_url"):
            print(f"          hf: {m['huggingface_url']}")
        if m.get("mmproj"):
            mp = resolve_path(m["mmproj"])
            print(f"          mmproj: {m['mmproj']} ({'EXISTS' if mp.exists() else 'NOT FOUND'})")
        if m.get("vocoder"):
            vp = resolve_path(m["vocoder"])
            print(f"          vocoder: {m['vocoder']} ({'EXISTS' if vp.exists() else 'NOT FOUND'})")


def cmd_add(args):
    """Add a new model to config.json."""
    config = load_config()

    if "models" not in config:
        config["models"] = {}

    if args.name in config["models"]:
        print(f"Model '{args.name}' already exists. Use 'update' or 'remove' first.")
        return 1

    # Build model entry
    entry = {
        "enabled": True,
        "type": args.type,
        "display_name": args.display_name or args.name,
        "description": args.description or "",
        "path": args.path,
        "size_mb": args.size_mb or 0,
        "vram_mb": args.vram_mb or 0,
        "context_length": args.context_length or 4096,
        "supports_vision": args.type in ["vision", "vision-language"] or args.supports_vision,
        "supports_audio": args.type == "audio" or args.supports_audio,
    }

    if args.mmproj:
        entry["mmproj"] = args.mmproj
    if args.tokenizer:
        entry["tokenizer"] = args.tokenizer
    if args.vocoder:
        entry["vocoder"] = args.vocoder
    if args.hf:
        entry["huggingface_url"] = args.hf

    config["models"][args.name] = entry
    save_config(config)
    print(f"OK: Added model '{args.name}' to config.json")
    print(f"  Type: {args.type}")
    print(f"  Path: {args.path}")
    print(f"  Path resolves to: {resolve_path(args.path)}")
    print(f"  File exists: {resolve_path(args.path).exists()}")

    return 0


def cmd_remove(args):
    """Remove a model from config.json."""
    config = load_config()
    if args.name not in config.get("models", {}):
        print(f"Model '{args.name}' not found.")
        return 1
    del config["models"][args.name]
    save_config(config)
    print(f"OK: Removed model '{args.name}' from config.json")
    return 0


def cmd_enable(args):
    """Enable a model in config.json."""
    config = load_config()
    if args.name not in config.get("models", {}):
        print(f"Model '{args.name}' not found.")
        return 1
    config["models"][args.name]["enabled"] = True
    save_config(config)
    print(f"OK: Enabled model '{args.name}'")
    return 0


def cmd_disable(args):
    """Disable a model in config.json."""
    config = load_config()
    if args.name not in config.get("models", {}):
        print(f"Model '{args.name}' not found.")
        return 1
    config["models"][args.name]["enabled"] = False
    save_config(config)
    print(f"OK: Disabled model '{args.name}'")
    return 0


def cmd_validate(args):
    """Validate all models have their files."""
    config = load_config()
    models = config.get("models", {})

    print("=" * 70)
    print("VALIDATION")
    print("=" * 70)

    all_ok = True
    for name, m in models.items():
        if not m.get("enabled", True):
            print(f"\n  [SKIP] {name} (disabled)")
            continue
        print(f"\n  {name}:")
        for key in ["path", "mmproj", "tokenizer", "vocoder"]:
            v = m.get(key)
            if not v:
                continue
            p = resolve_path(v)
            status = "OK" if p.exists() else "MISSING"
            size = f"{p.stat().st_size/1024/1024:.0f} MB" if p.exists() else "?"
            print(f"    {key}: {status} ({size}) - {p}")
            if not p.exists():
                all_ok = False

    print()
    if all_ok:
        print("All enabled models have their files. Ready to use.")
    else:
        print("Some files are missing. Download or check paths.")


def cmd_hf_add(args):
    """Add a model by fetching file list from HuggingFace."""
    if not args.hf:
        print("--hf required (e.g., LiquidAI/LFM2.5-VL-1.6B-GGUF)")
        return 1

    # Fetch file list
    url = f"https://huggingface.co/api/models/{args.hf}/tree/main"
    print(f"Fetching file list from {url}...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "MATHIR-vision"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        print(f"ERROR: {e}")
        return 1

    gguf_files = [f for f in data if f["path"].endswith(".gguf")]

    print(f"Found {len(gguf_files)} GGUF files:")
    for f in gguf_files[:20]:
        size_mb = f.get("size", 0) / 1024 / 1024
        print(f"  {f['path']:<50} {size_mb:.0f} MB")

    # Auto-detect main model and mmproj
    main_files = [f for f in gguf_files if "mmproj" not in f["path"].lower() and "tokenizer" not in f["path"].lower() and "vocoder" not in f["path"].lower()]
    mmproj_files = [f for f in gguf_files if "mmproj" in f["path"].lower()]
    tokenizer_files = [f for f in gguf_files if "tokenizer" in f["path"].lower()]
    vocoder_files = [f for f in gguf_files if "vocoder" in f["path"].lower()]

    if not main_files:
        print("No main model file detected.")
        return 1

    # Use smallest Q4_0 by default
    main_files.sort(key=lambda f: f.get("size", 0))
    selected = main_files[0]
    print(f"\nAuto-selected main model: {selected['path']} ({selected.get('size', 0)/1024/1024:.0f} MB)")

    # Build model entry
    model_id = args.name or args.hf.split("/")[-1]
    model_dir = args.hf.split("/")[-1]
    if model_dir.endswith("-GGUF"):
        model_dir = model_dir[:-5]

    entry = {
        "enabled": True,
        "type": args.type or "vision-language",
        "display_name": args.display_name or model_id,
        "description": f"From {args.hf}",
        "path": f"models/{model_dir}/{selected['path']}",
        "size_mb": int(selected.get("size", 0) / 1024 / 1024),
        "vram_mb": int(selected.get("size", 0) / 1024 / 1024 * 2),
        "context_length": 4096,
        "supports_vision": "VL" in model_id or "Vision" in model_id or "vision" in (args.type or ""),
        "supports_audio": "Audio" in model_id or "audio" in (args.type or ""),
        "huggingface_url": f"https://huggingface.co/{args.hf}",
    }

    # Add mmproj if found
    if mmproj_files:
        # Pick smallest mmproj
        mmproj_files.sort(key=lambda f: f.get("size", 0))
        mp = mmproj_files[0]
        entry["mmproj"] = f"models/{model_dir}/{mp['path']}"
        print(f"Auto-selected mmproj: {mp['path']} ({mp.get('size', 0)/1024/1024:.0f} MB)")

    if tokenizer_files:
        tf = tokenizer_files[0]
        entry["tokenizer"] = f"models/{model_dir}/{tf['path']}"
        print(f"Auto-selected tokenizer: {tf['path']}")

    if vocoder_files:
        vf = vocoder_files[0]
        entry["vocoder"] = f"models/{model_dir}/{vf['path']}"
        print(f"Auto-selected vocoder: {vf['path']}")

    # Save
    config = load_config()
    if "models" not in config:
        config["models"] = {}
    config["models"][model_id] = entry
    save_config(config)

    print(f"\nOK: Added model '{model_id}' to config.json")
    print(f"\nNext: Download the model files to models/{model_dir}/")
    print(f"  python download_q4.py   (or manual)")

    return 0


def main():
    parser = argparse.ArgumentParser(description="MATHIR vision model manager")
    subparsers = parser.add_subparsers(dest="command", help="Command")

    # list
    p_list = subparsers.add_parser("list", help="List all models")
    p_list.set_defaults(func=cmd_list)

    # add
    p_add = subparsers.add_parser("add", help="Add a model manually")
    p_add.add_argument("--name", required=True, help="Model name (key in config)")
    p_add.add_argument("--type", required=True,
                       choices=["vision", "vision-language", "audio", "text-only"],
                       help="Model type")
    p_add.add_argument("--path", required=True, help="Path to GGUF file (relative to vision_testing/)")
    p_add.add_argument("--display-name", help="Display name")
    p_add.add_argument("--description", help="Description")
    p_add.add_argument("--size-mb", type=int, help="Model file size in MB")
    p_add.add_argument("--vram-mb", type=int, help="VRAM usage in MB")
    p_add.add_argument("--context-length", type=int, default=4096)
    p_add.add_argument("--mmproj", help="Multimodal projector path")
    p_add.add_argument("--tokenizer", help="Audio tokenizer path")
    p_add.add_argument("--vocoder", help="Audio vocoder path")
    p_add.add_argument("--supports-vision", action="store_true")
    p_add.add_argument("--supports-audio", action="store_true")
    p_add.add_argument("--hf", help="HuggingFace URL (informational)")
    p_add.set_defaults(func=cmd_add)

    # hf-add
    p_hf = subparsers.add_parser("hf-add", help="Add a model from HuggingFace (auto-detect)")
    p_hf.add_argument("--hf", required=True, help="HuggingFace repo (e.g., LiquidAI/LFM2.5-VL-1.6B-GGUF)")
    p_hf.add_argument("--name", help="Override model name")
    p_hf.add_argument("--type", help="Override type")
    p_hf.add_argument("--display-name", help="Display name")
    p_hf.set_defaults(func=cmd_hf_add)

    # remove
    p_remove = subparsers.add_parser("remove", help="Remove a model")
    p_remove.add_argument("--name", required=True, help="Model name to remove")
    p_remove.set_defaults(func=cmd_remove)

    # enable
    p_enable = subparsers.add_parser("enable", help="Enable a model")
    p_enable.add_argument("--name", required=True)
    p_enable.set_defaults(func=cmd_enable)

    # disable
    p_disable = subparsers.add_parser("disable", help="Disable a model")
    p_disable.add_argument("--name", required=True)
    p_disable.set_defaults(func=cmd_disable)

    # validate
    p_validate = subparsers.add_parser("validate", help="Validate all model files exist")
    p_validate.set_defaults(func=cmd_validate)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 0

    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())