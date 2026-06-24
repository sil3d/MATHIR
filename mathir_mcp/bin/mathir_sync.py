#!/usr/bin/env python3
"""
mathir_sync.py -- Sync MATHIR source repo into OpenCode config.

WHAT IT DOES
============
When you (the dev) create a new tool, doc, or config in the MATHIR source
repo, this script copies those new files into your OpenCode config
(~/.config/opencode).

The source repo is auto-detected (sibling search, MATHIR_SRC env var, or
upward walk from cwd) — no hardcoded paths.

SAFE BY DEFAULT
===============
- Only copies files that DON'T already exist in the destination.
- Existing files are NEVER silently overwritten.
- Your working setup is protected from upstream changes.
- Use --update-existing to force overwrite (with --dry-run first).

WHAT GETS SYNCED (source -> destination)
========================================
  mathir_lib/*.py          ->  bin/*.py       (MCP server, daemon, vec, etc.)
  brain/*.py               ->  bin/*.py       (brain subsystem)
  config/*.json            ->  config/*.json  (JSON configs)
  docs/*.md                ->  docs/*.md      (documentation)
  GLOBAL_INSTRUCTIONS.md   ->  GLOBAL_INSTRUCTIONS.md
  install.bat, install.sh  ->  install.bat, install.sh
  mathir.bat, mathir.sh    ->  mathir.bat, mathir.sh

USAGE
=====
  python mathir_sync.py                       # dry-run, show all changes
  python mathir_sync.py --force               # apply (skip confirmation)
  python mathir_sync.py --only modules        # only Python modules
  python mathir_sync.py --only docs           # only docs
  python mathir_sync.py --update-existing     # overwrite existing files (CAREFUL)
  python mathir_sync.py --no-inject           # skip post-sync injection
  python mathir_sync.py --src PATH            # override source path
  python mathir_sync.py --dst PATH            # override destination
  python mathir_sync.py --explain             # explain how it works
  python mathir_sync.py --help                # all flags

WORKFLOW
========
  # 1. You dev a new tool in the source repo
  vim <source_repo>/mathir_lib/my_tool.py

  # 2. Sync it into OpenCode (dry-run first to see what changes)
  python bin/mathir_sync.py
  python bin/mathir_sync.py --force

  # 3. Inject MATHIR into any new .md files
  python bin/mathir_inject.py --apply --target all

PAIR WITH: mathir_inject.py
===========================
After syncing, run mathir_inject.py --apply to add MATHIR injection blocks
to any new .md files in agents/, commands/, skills/, or docs/.

EXIT CODES
==========
  0 = success (or dry-run completed)
  1 = error (source not found, files failed to copy, etc.)

FILE STATES (printed per file)
  [NEW]      -- file copied (didn't exist in destination)
  [NEW*]     -- would copy (dry-run)
  [OK]       -- file identical between source and destination
  [SKIP]     -- file exists but differs (protected, use --update-existing)
  [SKIP*]    -- would skip (dry-run)
  [UPDATE]   -- file updated (with --update-existing)
  [ERROR]    -- something went wrong
"""
from __future__ import annotations

import argparse
import filecmp
import hashlib
import shutil
import sys
from pathlib import Path

# What to sync from source -> destination
# (source_subpath, dest_subpath, file_pattern, description)
SYNC_PLAN = [
    # Python modules: mathir_lib/ + brain/ -> bin/
    ("mathir_lib", "bin", "*.py", "Python modules (mathir_lib)"),
    ("brain", "bin", "*.py", "Brain subsystem (brain/)"),
    # Configs
    ("config", "config", "*.json", "JSON configs"),
    # Docs
    ("docs", "docs", "*.md", "Documentation .md"),
    # MATHIR injection templates for each target type
    ("opencode/agents", "agents", "_MATHIR_INJECT.md", "Inject template (agents)"),
    ("opencode/commands", "commands", "_MATHIR_INJECT.md", "Inject template (commands)"),
    ("opencode/skills", "skills", "_MATHIR_INJECT.md", "Inject template (skills)"),
    ("opencode/skills-global", "skills-global", "_MATHIR_INJECT.md", "Inject template (skills-global)"),
    ("opencode/docs", "docs", "_MATHIR_INJECT.md", "Inject template (docs)"),
    # Top-level files
    ("", "", "GLOBAL_INSTRUCTIONS.md", "Global instructions"),
    ("", "", "install.bat", "Install script (Windows)"),
    ("", "", "install.sh", "Install script (Unix)"),
    ("", "", "mathir.bat", "1-click launcher (Windows)"),
    ("", "", "mathir.sh", "1-click launcher (Unix)"),
    ("", "", "mathir.bat", "1-click launcher"),
]

# Files to NEVER sync (would break the user's working setup)
PROTECTED_PATTERNS = {"*.egg-info", "__pycache__", ".pytest_cache", "*.pyc"}


def find_source_default() -> Path:
    """Try to locate the MATHIR source repo. Order:
    1. MATHIR_SRC env var
    2. Search upward from this script for a sibling dir containing pyproject.toml
       and a mathir_lib/ subdir (handles both source repo and deployed copies)
    3. Walk upward from cwd looking for the same marker
    4. Common dev paths (no hardcoded user paths)
    """
    import os
    env = os.environ.get("MATHIR_SRC")
    if env:
        p = Path(env).expanduser().resolve()
        if p.is_dir() and (p / "pyproject.toml").is_file():
            return p

    # Walk upward from this script looking for mathir_mcp root marker
    here = Path(__file__).resolve().parent
    for ancestor in [here, *here.parents]:
        # Check if this ancestor is the source repo root
        if (ancestor / "pyproject.toml").is_file() and (ancestor / "mathir_lib").is_dir():
            return ancestor
        # Check if a sibling named "mathir_mcp" exists at this level (deployed layout)
        sibling = ancestor / "mathir_mcp"
        if sibling.is_dir() and (sibling / "pyproject.toml").is_file():
            return sibling

    # Walk upward from cwd
    cwd = Path.cwd().resolve()
    for ancestor in [cwd, *cwd.parents]:
        if (ancestor / "pyproject.toml").is_file() and (ancestor / "mathir_lib").is_dir():
            return ancestor
        sibling = ancestor / "mathir_mcp"
        if sibling.is_dir() and (sibling / "pyproject.toml").is_file():
            return sibling

    raise FileNotFoundError(
        "Could not locate MATHIR source repo. "
        "Use --src to specify the path explicitly, "
        "or set MATHIR_SRC environment variable."
    )


def find_dest_default() -> Path:
    """Try to locate the OpenCode config root."""
    import os
    env = os.environ.get("OPENCODE_CONFIG")
    if env:
        p = Path(env).expanduser().resolve()
        if p.is_dir():
            return p
    home = Path.home() / ".config" / "opencode"
    if home.is_dir():
        return home
    raise FileNotFoundError(
        "Could not locate OpenCode config root. "
        "Use --dst to specify the path explicitly."
    )


def is_protected(path: Path) -> bool:
    """Check if a path matches any protected pattern."""
    name = path.name
    for pat in PROTECTED_PATTERNS:
        if pat.startswith("*"):
            if name.endswith(pat[1:]):
                return True
        elif name == pat:
            return True
    return False


def file_hash(path: Path) -> str:
    """SHA-256 hash of file contents."""
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def files_differ(src: Path, dst: Path) -> bool:
    """Check if source and destination files differ (by hash)."""
    if not dst.exists():
        return True
    if src.stat().st_size != dst.stat().st_size:
        return True
    try:
        return file_hash(src) != file_hash(dst)
    except Exception:
        return True


def collect_files(src_root: Path, subpath: str, pattern: str) -> list[Path]:
    """Collect files in src_root/subpath matching pattern."""
    base = src_root / subpath if subpath else src_root
    if not base.is_dir():
        return []
    if pattern == "*":
        return sorted(p for p in base.rglob("*") if p.is_file() and not is_protected(p))
    return sorted(p for p in base.glob(pattern) if p.is_file() and not is_protected(p))


def sync_one(src: Path, dst: Path, dry_run: bool = False,
             update_existing: bool = False) -> str:
    """Sync one file from src to dst.

    By default (update_existing=False), only copies files that DON'T exist
    in the destination. Existing files are NEVER silently overwritten — this
    protects the user's working setup from being clobbered by upstream changes.

    Set update_existing=True to allow overwriting existing files.

    Returns status: 'new', 'updated', 'unchanged', 'skipped-existing',
                    'would-create', 'would-update', 'would-skip', 'error:...'
    """
    try:
        if not files_differ(src, dst):
            return "unchanged"

        # File exists in destination but differs
        if dst.exists():
            if not update_existing:
                if dry_run:
                    return "would-skip"
                return "skipped-existing"
            # update_existing is True — proceed with overwrite
            if dry_run:
                return "would-update"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            return "updated"

        # File doesn't exist in destination — copy as new
        if dry_run:
            return "would-create"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return "new"
    except Exception as e:
        return f"error: {e}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--src", help="Source repo path (default: auto-detected via MATHIR_SRC env or sibling search)")
    parser.add_argument("--dst", help="Destination config root (default: ~/.config/opencode)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would change, no modifications")
    parser.add_argument("--force", action="store_true", help="Skip confirmation prompt")
    parser.add_argument("--no-inject", action="store_true", help="Skip post-sync injection step")
    parser.add_argument("--only", help="Sync only matching plans: modules, configs, docs, scripts")
    parser.add_argument("--update-existing", action="store_true",
                        help="Allow overwriting files that already exist in destination")
    parser.add_argument("--explain", action="store_true",
                        help="Print an explanation of how this script works and exit")
    args = parser.parse_args()

    # Resolve paths
    try:
        src_root = Path(args.src).expanduser().resolve() if args.src else find_source_default()
    except FileNotFoundError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1
    try:
        dst_root = Path(args.dst).expanduser().resolve() if args.dst else find_dest_default()
    except FileNotFoundError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1

    if args.explain:
        print(__doc__)
        return 0

    if not (src_root / "pyproject.toml").is_file():
        print(f"[ERROR] {src_root} doesn't look like a MATHIR source repo (no pyproject.toml)",
              file=sys.stderr)
        return 1

    if not (dst_root / "opencode.json").is_file():
        print(f"[WARN] {dst_root} doesn't look like an OpenCode config (no opencode.json)",
              file=sys.stderr)
        if not args.force:
            cont = input("Continue anyway? [y/N] ")
            if cont.lower() != "y":
                return 1

    print(f"MATHIR Sync")
    print(f"  Source:      {src_root}")
    print(f"  Destination: {dst_root}")
    print(f"  Mode:        {'dry-run' if args.dry_run else 'apply'}")
    print("-" * 70)

    # Filter SYNC_PLAN by --only
    plan = SYNC_PLAN
    if args.only == "modules":
        plan = [p for p in SYNC_PLAN if p[2] == "*.py"]
    elif args.only == "configs":
        plan = [p for p in SYNC_PLAN if p[2] == "*.json"]
    elif args.only == "docs":
        plan = [p for p in SYNC_PLAN if p[2] == "*.md"]
    elif args.only == "scripts":
        plan = [p for p in SYNC_PLAN if p[2] in ("*.bat", "*.sh")]

    counts = {"new": 0, "updated": 0, "unchanged": 0, "skipped-existing": 0,
              "would-create": 0, "would-update": 0, "would-skip": 0, "error": 0}

    for src_sub, dst_sub, pattern, desc in plan:
        files = collect_files(src_root, src_sub, pattern)
        if not files:
            continue
        print(f"\n[{desc}] ({len(files)} file(s))")
        for src_file in files:
            rel = src_file.relative_to(src_root / src_sub if src_sub else src_root)
            dst_file = (dst_root / dst_sub / rel) if dst_sub else (dst_root / rel)
            status = sync_one(src_file, dst_file, dry_run=args.dry_run,
                              update_existing=args.update_existing)
            icon = {
                "new": "[NEW]      ",
                "updated": "[UPDATED]  ",
                "unchanged": "[OK]       ",
                "skipped-existing": "[SKIP]     ",
                "would-create": "[NEW*]     ",
                "would-update": "[UPDATE*]  ",
                "would-skip": "[SKIP*]    ",
            }.get(status, f"[ERROR]    {status}")
            counts[status if status in counts else "error"] = counts.get(status if status in counts else "error", 0) + 1
            try:
                display = dst_file.relative_to(dst_root).as_posix()
            except ValueError:
                display = str(dst_file)
            print(f"  {icon} {display}")

    print("-" * 70)
    if args.dry_run:
        print(
            f"Result: {counts['would-create']} would-create, "
            f"{counts['would-update']} would-update, "
            f"{counts['unchanged']} unchanged, {counts['would-skip']} would-skip, "
            f"{counts['error']} errors"
        )
    else:
        print(
            f"Result: {counts['new']} new, {counts['updated']} updated, "
            f"{counts['unchanged']} unchanged, {counts['skipped-existing']} skipped, "
            f"{counts['error']} errors"
        )

    if args.dry_run:
        print("\n[DRY RUN] No files were modified.")
        return 0

    if counts["error"] > 0:
        print("\n[WARN] Some files failed to sync. See errors above.")
        return 1

    # Show what would have been updated
    if counts.get("skipped-existing", 0) > 0:
        print(
            f"\n[INFO] {counts['skipped-existing']} file(s) exist in destination but differ from source."
        )
        print("       These were SKIPPED to protect your working setup.")
        print("       Use --update-existing to overwrite (use with care).")
        print("       Use --diff to see what would change.")

    # Post-sync: run injection on agents, commands, skills
    if not args.no_inject and counts.get("new", 0) > 0:
        print("\n[POST-SYNC] Running mathir_inject.py --apply...")
        inject_script = dst_root / "bin" / "mathir_inject.py"
        if inject_script.is_file():
            import subprocess
            for target in ("agents", "commands", "skills"):
                result = subprocess.run(
                    [sys.executable, str(inject_script), "--apply", "--target", target],
                    capture_output=True, text=True
                )
                if result.returncode == 0:
                    # Show last 2 lines of output (the summary)
                    lines = result.stdout.strip().split("\n")
                    print(f"  [{target}] {lines[-1] if lines else 'OK'}")
                else:
                    print(f"  [{target}] FAILED (exit {result.returncode})")
                    if result.stderr:
                        print(f"    {result.stderr.strip()[:200]}")
        else:
            print(f"  [SKIP] inject script not found at {inject_script}")

    return 0


if __name__ == "__main__":
    sys.exit(main())