#!/usr/bin/env python3
"""
Setup source files: copy reference Rust/JS source from any source directory.
NO HARDCODED PATHS. User provides source via CLI argument.

Usage:
  python setup_sources.py /path/to/secret_project/Mycerise_V2_Taur
  SECRET_PROJECT=/path/to/secret/project python setup_sources.py

v8.4.0 MIGRATION: Removed LlamaSetupModal.jsx + wizardModels_llamacpp.json
+ convert_lfm2_to_gguf.py entries (llama.cpp is gone, replaced by OpenRouter).
"""
import os
import sys
import shutil
import argparse
from pathlib import Path
from datetime import datetime

HERE = Path(__file__).resolve().parent


# Source file patterns: (source_relative_path, dest_dir_in_this_package)
# Source paths are RELATIVE to the secret project root
SOURCE_FILES = [
    {
        "source_rel": "src-tauri/src/services/turbo_loader.rs",
        "dest_dir": "memory",
        "dest_name": "turbo_loader_reference.rs",
        "description": "Rust turboquant loader",
    },
    {
        "source_rel": "src/components/OpenRouterSetupModal.jsx",
        "dest_dir": "interface",
        "dest_name": "OpenRouterSetupModal_reference.jsx",
        "description": "Tauri UI for OpenRouter setup (replaces LlamaSetupModal in v8.4.0)",
    },
    {
        "source_rel": "src/config/wizardModels_openrouter.json",
        "dest_dir": "interface",
        "dest_name": "wizardModels_openrouter_reference.json",
        "description": "Model definitions JSON (replaces wizardModels_llamacpp.json in v8.4.0)",
    },
]


def copy_sources(source_root: Path, here: Path, only_files: list = None):
    """Copy source files from source_root to the destination directory."""
    copied = 0
    failed = []
    skipped = 0

    if not source_root.exists():
        print(f"ERROR: Source directory not found: {source_root}")
        return {"copied": 0, "skipped": 0, "failed": SOURCE_FILES}

    for entry in SOURCE_FILES:
        if only_files and entry["source_rel"] not in only_files:
            continue

        src = source_root / entry["source_rel"]
        if not src.exists():
            failed.append((entry["source_rel"], "not found"))
            print(f"  [MISS] {entry['source_rel']} (not in source)")
            continue

        dest_dir = here / entry["dest_dir"]
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / entry["dest_name"]

        if dest.exists() and dest.stat().st_size == src.stat().st_size:
            skipped += 1
            print(f"  [SKIP] {entry['source_rel']} (already exists)")
        else:
            shutil.copy2(src, dest)
            copied += 1
            size = src.stat().st_size
            print(f"  [OK]   {entry['source_rel']} -> {entry['dest_dir']}/{entry['dest_name']} ({size} bytes)")

    return {"copied": copied, "skipped": skipped, "failed": failed}


def main():
    parser = argparse.ArgumentParser(
        description="Setup reference source files from any secret project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Provide source root as CLI argument
  python setup_sources.py /path/to/Mycerise_V2_Taur

  # Use environment variable
  SECRET_PROJECT=/path/to/secret python setup_sources.py

  # Specify exact source root
  python setup_sources.py --source /path/to/project_root
        """,
    )
    parser.add_argument("source", nargs="?", help="Path to source project root (e.g., Mycerise_V2_Taur)")
    parser.add_argument("--source", dest="source2", help="Alternative way to specify source")
    parser.add_argument("--only", help="Only copy specific file (by source_rel)")

    args = parser.parse_args()

    # Determine source root
    source = args.source or args.source2 or os.environ.get("SECRET_PROJECT")
    if not source:
        print("No source specified. Use:")
        print("  python setup_sources.py <path>")
        print("  SECRET_PROJECT=<path> python setup_sources.py")
        sys.exit(1)

    source_root = Path(source).resolve()
    here = Path(__file__).resolve().parent

    print("=" * 70)
    print("SETUP REFERENCE SOURCES")
    print("=" * 70)
    print(f"Source: {source_root}")
    print(f"Target: {here}")

    result = copy_sources(source_root, here, [args.only] if args.only else None)

    print()
    print(f"Copied: {result['copied']}, Skipped: {result['skipped']}, Failed: {len(result['failed'])}")
    if result["failed"]:
        print("\nFailed files:")
        for f, reason in result["failed"]:
            print(f"  {f}: {reason}")

    return 0 if not result["failed"] else 1


if __name__ == "__main__":
    sys.exit(main())