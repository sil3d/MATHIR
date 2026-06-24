#!/usr/bin/env python3
"""
mathir_inject.py -- Dynamic MATHIR injection for any tool/agent/command/skill.

WHAT IT DOES
============
Every agent, command, skill, and doc in your OpenCode config needs a "MATHIR
injection block" at the top -- this tells the AI it has persistent memory via
MCP tools. Without this block, the AI doesn't know it can call memory_recall,
memory_save, etc.

This script AUTOMATES that injection. Instead of manually copy-pasting a
180-line block into every new file, you run:

    python mathir_inject.py --apply

It reads the template from `<target_dir>/_MATHIR_INJECT.md` and injects it
into every file that needs it. Idempotent -- running it twice is safe.

TEMPLATES (one per target type, each in its own directory)
==========================================================
  agents/_MATHIR_INJECT.md         -- full block, for agent system prompts
  commands/_MATHIR_INJECT.md       -- short block, for slash commands
  skills/_MATHIR_INJECT.md         -- minimal block, for skills
  skills-global/_MATHIR_INJECT.md  -- same as skills/
  docs/_MATHIR_INJECT.md           -- reference, for documentation
  _MATHIR_INJECT.md                -- fallback (root)

TARGETS
=======
  agents        -- agent .md files in agents/*.md (32 files)
  commands      -- slash command .md files in commands/*.md (9 files)
  skills        -- skill SKILL.md files in skills/*/SKILL.md (68 files)
  skills-global -- global skills in skills-global/*/SKILL.md (88 files)
  docs          -- .md files in docs/*.md
  all           -- everything above in one pass

USAGE
=====
  python mathir_inject.py                       # check agents (default)
  python mathir_inject.py --apply               # inject into agents
  python mathir_inject.py --target all --apply  # inject into everything
  python mathir_inject.py --target commands     # check commands
  python mathir_inject.py --apply --file PATH   # inject into one file
  python mathir_inject.py --list                # show all targets/templates
  python mathir_inject.py --explain             # explain how it works
  python mathir_inject.py --help                # show all flags

COMMON WORKFLOWS
================
  # You just created a new agent:
  python mathir_inject.py --apply --file agents/foo.md

  # You edited the injection template:
  python mathir_inject.py --apply --target all

  # You want to see which files need updating:
  python mathir_inject.py --check --target all

  # You want to see what's available:
  python mathir_inject.py --list

PAIR WITH: mathir_sync.py
=========================
  Use mathir_sync.py to import NEW files from the source repo, then run
  mathir_inject.py --apply to inject MATHIR into the imported .md files.

SEE ALSO
========
  bin/mathir_sync.py    -- import new files from source repo
  agents/_MATHIR_INJECT.md  -- template that gets injected
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

# Anchor patterns — see _find_inject_start() below.
INJECT_START_PATTERNS = [
    "# MATHIR MEMORY — v",
    "# MATHIR MEMORY - v",
    "# MATHIR MEMORY v",
    "# MATHIR MEMORY",
]
INJECT_END = "# === END MATHIR INJECTION ==="

# The standard end-marker that appears in agent templates (last meaningful line)
STANDARD_END_MARKER = "**Model:** paraphrase-multilingual-MiniLM-L12-v2"

# Files to skip in any target (README, system files, templates themselves)
SKIP_NAMES = {"README.md", "_MATHIR_INJECT.md", "_MATHIR_INJECT_AGENTS.md",
              "_MATHIR_INJECT_COMMANDS.md", "_MATHIR_INJECT_SKILLS.md",
              "_MATHIR_INJECT_DOCS.md"}

# Targets: (name, subdir_under_config_root, file_glob)
TARGETS = {
    "agents": ("agents", "*.md"),
    "commands": ("commands", "*.md"),
    "skills": ("skills", "*/SKILL.md"),       # OpenCode skills layout
    "skills-global": ("skills-global", "*/SKILL.md"),
    "docs": ("docs", "*.md"),
}


def find_config_root() -> Path:
    """Locate the OpenCode config root. Order:
    1. OPENCODE_CONFIG env var
    2. ~/.config/opencode
    3. ../  (if running from bin/)"""
    env = os.environ.get("OPENCODE_CONFIG")
    if env:
        p = Path(env).expanduser().resolve()
        if p.is_dir():
            return p
    home = Path.home() / ".config" / "opencode"
    if home.is_dir():
        return home
    here = Path(__file__).resolve().parent
    if (here.parent / "opencode.json").is_file():
        return here.parent
    raise FileNotFoundError(
        "Could not locate OpenCode config root. Set OPENCODE_CONFIG env var."
    )


def _find_inject_start(text: str) -> int:
    """Find the index of the actual MATHIR block start in `text`. Skips any
    template meta-comments that come before the actual block (lines starting
    with '# MATHIR MEMORY' that are followed by 'Source of truth', 'This file
    is auto-injected', etc., are treated as meta and ignored)."""
    for pat in INJECT_START_PATTERNS:
        idx = text.find(pat)
        if idx != -1:
            # Verify this is the real block (not a meta-comment)
            after = text[idx:idx + 300]
            # The real block continues with a recognizable pattern like
            # "## 🧠 YOUR ACTIVE MEMORY" or "## MATHIR v" or similar.
            # A meta-comment typically just describes the file's role.
            meta_keywords = ["Source of truth", "auto-injected",
                             "DO NOT EDIT", "then run:"]
            if any(kw.lower() in after.lower() for kw in meta_keywords):
                # Look for the next occurrence after this position
                next_idx = text.find(pat, idx + 1)
                if next_idx != -1:
                    return next_idx
            return idx
    return -1


def _find_block_end(text: str, start_idx: int) -> int:
    """Find the end of the MATHIR block starting at start_idx. Returns the
    position just after the block's last line."""
    # 1. Explicit end marker
    end_idx = text.find(INJECT_END, start_idx)
    if end_idx != -1:
        nl = text.find("\n", end_idx)
        return nl + 1 if nl != -1 else len(text)

    # 2. Standard agent end-marker
    end_idx = text.find(STANDARD_END_MARKER, start_idx)
    if end_idx != -1:
        nl = text.find("\n", end_idx)
        return nl + 1 if nl != -1 else len(text)

    # 3. Heuristic: heading that starts a new section ("## AGENT", "## ROLE", etc.)
    lines = text[start_idx:].split("\n")
    cut = len(lines)
    for i, line in enumerate(lines):
        if i > 5:
            stripped = line.strip()
            if (
                stripped.startswith("## AGENT")
                or stripped.startswith("## ROLE")
                or stripped.startswith("## YOU ARE")
                or stripped.startswith("## BEHAVIOR")
                or stripped.startswith("## YOUR ")
                or stripped.startswith("## IDENTITY")
                or stripped.startswith("## DESCRIPTION")
                or stripped.startswith("## INSTRUCTIONS")
            ):
                cut = i
                break
    return start_idx + sum(len(l) + 1 for l in lines[:cut])


def _normalize(text: str) -> str:
    """Normalize text for comparison: strip whitespace, collapse blank lines."""
    lines = [l.rstrip() for l in text.strip().split("\n")]
    out = []
    prev_blank = False
    for line in lines:
        blank = not line.strip()
        if blank and prev_blank:
            continue
        out.append(line)
        prev_blank = blank
    return "\n".join(out).strip()


def _extract_existing_block(body: str) -> str | None:
    """Extract the existing injection block from the body, if present."""
    start_idx = _find_inject_start(body)
    if start_idx == -1:
        return None
    end_idx = _find_block_end(body, start_idx)
    return _normalize(body[start_idx:end_idx])


def _content_equivalent(old: str, new: str) -> bool:
    """Check if existing block is content-equivalent to new template."""
    return _normalize(old) == _normalize(new)


def extract_yaml_frontmatter(content: str) -> tuple[str, str]:
    """Split YAML frontmatter (between --- markers) from body."""
    if not content.startswith("---"):
        return "", content
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n?", content, re.DOTALL)
    if not m:
        return "", content
    return content[: m.end()], content[m.end():]


def load_template(target: str, config_root: Path) -> str:
    """Load the injection template for a given target. Looks for:
    1. <target_subdir>/_MATHIR_INJECT.md
    2. _MATHIR_INJECT.md (root or generic fallback)
    3. Agents template (for backwards compat)"""
    target_dir = TARGETS.get(target, (target, "*.md"))[0]
    candidates = [
        config_root / target_dir / "_MATHIR_INJECT.md",
        config_root / "_MATHIR_INJECT.md",
        config_root / "agents" / "_MATHIR_INJECT.md",  # backwards compat
    ]
    for path in candidates:
        if path.is_file():
            full = path.read_text(encoding="utf-8")
            # Extract just the actual block, skip meta-headers
            start_idx = _find_inject_start(full)
            if start_idx == -1:
                continue
            end_idx = _find_block_end(full, start_idx)
            return full[start_idx:end_idx].strip()

    # No template found — return empty (caller should warn)
    return ""


def inject_block(content: str, template: str) -> str:
    """Insert/replace the injection block in content. Returns new content."""
    if not template.strip():
        return content  # No template to inject

    frontmatter, body = extract_yaml_frontmatter(content)

    existing = _extract_existing_block(body)
    if existing is not None and _content_equivalent(existing, template):
        return content  # Already up-to-date

    # Strip previous block from body
    start_idx = _find_inject_start(body)
    if start_idx != -1:
        end_idx = _find_block_end(body, start_idx)
        body = body[:start_idx] + body[end_idx:]

    body = body.lstrip("\n")

    if frontmatter:
        return frontmatter + "\n" + template.strip() + "\n\n" + body
    return template.strip() + "\n\n" + body


def process_file(file_path: Path, template: str, check_only: bool = False) -> str:
    """Process a single file. Returns status."""
    content = file_path.read_text(encoding="utf-8")
    new_content = inject_block(content, template)
    if new_content == content:
        return "ok"
    if not check_only:
        file_path.write_text(new_content, encoding="utf-8")
        return "updated" if _extract_existing_block(content) else "injected"
    return "needs-update" if _extract_existing_block(content) else "missing"


def discover_targets(config_root: Path, target: str) -> list[Path]:
    """Discover files to process for the given target."""
    if target == "all":
        files = []
        for tgt in ("agents", "commands", "docs"):
            files.extend(discover_targets(config_root, tgt))
        # Skills: both skills/ and skills-global/
        files.extend(discover_targets(config_root, "skills"))
        files.extend(discover_targets(config_root, "skills-global"))
        return sorted(set(files))

    if target not in TARGETS:
        raise ValueError(f"Unknown target: {target}. Use one of: {', '.join(TARGETS)} or 'all'")

    subdir, glob_pattern = TARGETS[target]
    base = config_root / subdir
    if not base.is_dir():
        return []
    files = [p for p in base.glob(glob_pattern) if p.name not in SKIP_NAMES]
    return sorted(files)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true", help="Check only, no modifications")
    parser.add_argument("--apply", action="store_true", help="Apply (default)")
    parser.add_argument("--target", default="agents",
                        help="Target type: agents | commands | skills | skills-global | docs | all")
    parser.add_argument("--agent", help="Process a single agent (agents/AGENT.md)")
    parser.add_argument("--file", help="Process a single explicit file path")
    parser.add_argument("--list", action="store_true", help="List targets and templates")
    parser.add_argument("--config-root", help="Override config root path")
    parser.add_argument("--explain", action="store_true",
                        help="Print an explanation of how this script works and exit")
    args = parser.parse_args()

    check_only = args.check and not args.apply
    if args.apply:
        check_only = False

    try:
        config_root = Path(args.config_root).expanduser().resolve() if args.config_root else find_config_root()
    except FileNotFoundError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1

    if args.list:
        print(f"MATHIR Inject — targets at {config_root}")
        print("-" * 60)
        for tgt, (subdir, glob_pat) in TARGETS.items():
            base = config_root / subdir
            template = load_template(tgt, config_root)
            n_files = len(list(base.glob(glob_pat))) if base.is_dir() else 0
            has_template = "yes" if template else "NO"
            print(f"  {tgt:<15} {subdir:<20} files={n_files:<4} template={has_template}")
        return 0

    if args.explain:
        print(__doc__)
        print("\nEXIT CODES")
        print("  0  = success, no changes needed or all applied")
        print("  1  = error (template missing, file read/write failed, etc.)")
        print("  2  = --check mode: some files need update (exit so scripts can react)")
        print()
        print("FILE STATES (printed per file)")
        print("  [OK]        — injection is up-to-date")
        print("  [INJECTED]  — new file, block was added")
        print("  [UPDATED]   — block was outdated and replaced")
        print("  [STALE]     — block exists but doesn't match template (--check only)")
        print("  [MISSING]   — no injection block found (--check only)")
        print("  [ERROR]     — something went wrong")
        return 0

    if args.file:
        targets = [Path(args.file).resolve()]
        # Infer target from path
        try:
            rel = targets[0].relative_to(config_root)
            target_name = rel.parts[0] if rel.parts else "agents"
        except ValueError:
            target_name = "agents"
    elif args.agent:
        targets = [config_root / "agents" / args.agent]
        target_name = "agents"
    else:
        targets = discover_targets(config_root, args.target)
        target_name = args.target

    if not targets:
        print(f"[INFO] No files found for target '{target_name}' in {config_root}", file=sys.stderr)
        return 0

    template = load_template(target_name, config_root)
    if not template:
        print(f"[WARN] No template found for target '{target_name}'.", file=sys.stderr)
        print(f"       Create: {config_root}/{TARGETS.get(target_name, (target_name,))[0]}/_MATHIR_INJECT.md", file=sys.stderr)
        return 1

    print(f"MATHIR Inject — {len(targets)} file(s) — target={target_name}")
    print(f"Mode: {'check' if check_only else 'apply'}")
    print("-" * 60)

    counts = {"ok": 0, "injected": 0, "updated": 0, "needs-update": 0, "missing": 0, "error": 0}
    for fp in targets:
        if not fp.is_file():
            print(f"  [SKIP] {fp.name} (not a file)")
            continue
        try:
            status = process_file(fp, template, check_only=check_only)
        except Exception as e:
            print(f"  [ERROR] {fp.name}: {e}")
            counts["error"] += 1
            continue
        counts[status] = counts.get(status, 0) + 1
        icon = {
            "ok": "[OK]      ",
            "injected": "[INJECTED]",
            "updated": "[UPDATED] ",
            "needs-update": "[STALE]   ",
            "missing": "[MISSING] ",
        }.get(status, "[?]       ")
        try:
            display = fp.relative_to(config_root).as_posix()
        except ValueError:
            display = str(fp)
        print(f"  {icon} {display}")

    print("-" * 60)
    print(
        f"Result: {counts['ok']} ok, {counts['injected']} injected, "
        f"{counts['updated']} updated, {counts['error']} errors"
    )
    if check_only and (counts.get("needs-update", 0) + counts.get("missing", 0)) > 0:
        return 2
    return 0 if counts["error"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())