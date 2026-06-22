#!/usr/bin/env python3
"""
MATHIR Proactive Setup — Auto-inject MATHIR into any AI tool
=============================================================
Detects which AI tool you use, finds its config files, and injects
MATHIR instructions + MCP server registration.

Works with: OpenCode, Claude Code, Cursor, MiMo, Kilo Code, and more.

Usage:
    python setup_proactive.py              # Scan + inject everywhere
    python setup_proactive.py --dry-run    # Show what would change
    python setup_proactive.py --tool opencode  # Force a specific tool
"""
import os
import sys
import json
import argparse
import platform
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ─── MATHIR instructions (tool-agnostic) ─────────────────────────────────────

MATHIR_RECALL_INSTRUCTIONS = """
## MATHIR Memory — Active

Before responding to ANY user message, recall relevant context:

```bash
python -c "from mathir_mcp.mathir_lib.mathir_client import call; print(call('memory_recall', {'query': '<USER_MESSAGE_SUMMARY>', 'k': 3}))"
```

After completing a task, save what you learned:

```bash
python -c "from mathir_mcp.mathir_lib.mathir_client import call; print(call('memory_save', {'content': '<WHAT_YOU_LEARNED>', 'agent': '<AGENT_NAME>', 'block_type': 'episodic', 'label': '<task-label>', 'priority': 7}))"
```

Rules:
- NEVER start work without recalling first
- ALWAYS save after learning something new
- NEVER say "let me check my memory" — just do it
"""

# ─── Tool detection ───────────────────────────────────────────────────────────

def detect_tools() -> Dict[str, Dict]:
    """Detect which AI tools are installed and their config locations."""
    home = Path.home()
    detected = {}

    # OpenCode
    oc_config = home / ".config" / "opencode" / "opencode.json"
    oc_agents = home / ".config" / "opencode" / "AGENTS.md"
    oc_global = home / ".config" / "opencode" / "GLOBAL_INSTRUCTIONS.md"
    if oc_config.exists() or oc_agents.exists():
        detected["opencode"] = {
            "config_file": oc_config if oc_config.exists() else None,
            "prompt_files": [f for f in [oc_agents, oc_global] if f.exists()],
            "config_dir": home / ".config" / "opencode",
            "mcp_key": "mcp",
            "command_format": "list",
        }

    # Claude Code
    claude_config = home / ".config" / "claude" / "claude_desktop_config.json"
    claude_code_config = home / ".claude" / "settings.json"
    if claude_config.exists() or claude_code_config.exists():
        detected["claude"] = {
            "config_file": claude_config if claude_config.exists() else claude_code_config,
            "prompt_files": [],
            "config_dir": claude_config.parent if claude_config.exists() else claude_code_config.parent,
            "mcp_key": "mcpServers",
            "command_format": "string",
        }

    # Cursor
    cursor_config = home / ".cursor" / "mcp.json"
    if cursor_config.exists():
        detected["cursor"] = {
            "config_file": cursor_config,
            "prompt_files": [],
            "config_dir": cursor_config.parent,
            "mcp_key": "mcpServers",
            "command_format": "string",
        }

    # MiMo Code
    mimo_config = home / ".mimo" / "config.json"
    if mimo_config.exists():
        detected["mimo"] = {
            "config_file": mimo_config,
            "prompt_files": [],
            "config_dir": mimo_config.parent,
            "mcp_key": "mcp",
            "command_format": "list",
        }

    # Kilo Code
    kilo_config = home / ".kilo" / "config.json"
    if kilo_config.exists():
        detected["kilo"] = {
            "config_file": kilo_config,
            "prompt_files": [],
            "config_dir": kilo_config.parent,
            "mcp_key": "mcpServers",
            "command_format": "list",
        }

    return detected


# ─── Injection logic ──────────────────────────────────────────────────────────

def has_mathir(content: str) -> bool:
    """Check if file already has MATHIR instructions."""
    markers = ["MATHIR", "mathir_client", "mathir_mcp", "memory_save", "memory_recall"]
    return any(m in content for m in markers)


def inject_into_md(filepath: Path) -> bool:
    """Inject MATHIR recall instructions into a markdown file."""
    content = filepath.read_text(encoding="utf-8")
    if has_mathir(content):
        return False

    lines = content.split("\n")
    insert_idx = 0

    # Insert after first "---" or first heading
    for i, line in enumerate(lines):
        if line.strip() == "---" and i > 0:
            insert_idx = i + 1
            break
        if line.startswith("# ") and i == 0:
            insert_idx = 1
            break

    lines.insert(insert_idx, MATHIR_RECALL_INSTRUCTIONS)
    filepath.write_text("\n".join(lines), encoding="utf-8")
    return True


def setup_mcp_config(tool_name: str, tool_info: Dict) -> bool:
    """Add MATHIR MCP server to tool config."""
    config_file = tool_info.get("config_file")
    if not config_file or not config_file.exists():
        return False

    try:
        config = json.loads(config_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return False

    mcp_key = tool_info["mcp_key"]
    mcp_section = config.get(mcp_key, {})

    if "mathir" in mcp_section:
        print(f"  [OK] {tool_name}: MATHIR already configured")
        return False

    # Build MATHIR config based on tool format
    if tool_info["command_format"] == "list":
        mathir_config = {
            "type": "local",
            "command": ["python", "-m", "mathir_mcp"],
            "environment": {
                "MATHIR_EMBEDDING_DIM": "384",
                "MATHIR_PORT": "7338",
            },
            "enabled": True,
        }
    else:
        mathir_config = {
            "type": "local",
            "command": "python -m mathir_mcp",
            "environment": {
                "MATHIR_EMBEDDING_DIM": "384",
                "MATHIR_PORT": "7338",
            },
            "enabled": True,
        }

    mcp_section["mathir"] = mathir_config
    config[mcp_key] = mcp_section
    config_file.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  [ADD] {tool_name}: MATHIR MCP server added to {config_file.name}")
    return True


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MATHIR Proactive Setup — works with any AI tool")
    parser.add_argument("--dry-run", action="store_true", help="Show what would change")
    parser.add_argument("--tool", type=str, help="Force specific tool (opencode, claude, cursor, mimo, kilo)")
    args = parser.parse_args()

    print("=" * 60)
    print("MATHIR Proactive Setup")
    print("=" * 60)
    print()

    # Detect tools
    if args.tool:
        detected = detect_tools()
        if args.tool not in detected:
            print(f"Tool '{args.tool}' not detected. Detected: {list(detected.keys())}")
            sys.exit(1)
        detected = {args.tool: detected[args.tool]}
    else:
        detected = detect_tools()

    if not detected:
        print("No AI tools detected.")
        print()
        print("Supported tools: opencode, claude, cursor, mimo, kilo")
        print()
        print("To force a specific tool:")
        print("  python setup_proactive.py --tool opencode")
        sys.exit(1)

    print(f"Detected tools: {', '.join(detected.keys())}")
    print()

    changes = 0

    # Process each tool
    for tool_name, tool_info in detected.items():
        print(f"--- {tool_name.upper()} ---")

        # Inject into prompt files
        for md_file in tool_info.get("prompt_files", []):
            if args.dry_run:
                content = md_file.read_text(encoding="utf-8")
                if has_mathir(content):
                    print(f"  [OK] {md_file.name} — MATHIR already present")
                else:
                    print(f"  [WOULD INJECT] {md_file.name}")
                    changes += 1
            else:
                if inject_into_md(md_file):
                    print(f"  [INJECTED] {md_file.name}")
                    changes += 1
                else:
                    print(f"  [OK] {md_file.name} — MATHIR already present")

        # Setup MCP config
        if args.dry_run:
            config_file = tool_info.get("config_file")
            if config_file and config_file.exists():
                try:
                    config = json.loads(config_file.read_text(encoding="utf-8"))
                    mcp_key = tool_info["mcp_key"]
                    if "mathir" in config.get(mcp_key, {}):
                        print(f"  [OK] {config_file.name} — MATHIR MCP already configured")
                    else:
                        print(f"  [WOULD ADD] MATHIR MCP to {config_file.name}")
                        changes += 1
                except Exception:
                    print(f"  [?] {config_file.name} — cannot parse")
            else:
                print(f"  [?] No config file found for {tool_name}")
        else:
            if setup_mcp_config(tool_name, tool_info):
                changes += 1

        print()

    # Summary
    print("=" * 60)
    if changes > 0:
        print(f"Done: {changes} file(s) modified")
        print()
        print("RESTART your AI tool for changes to take effect.")
        print()
        print("To start the MATHIR brain stack (proactive injection):")
        print("  python -m mathir_mcp  # starts daemon on port 7338")
    else:
        print("Everything is already configured!")
    print("=" * 60)


if __name__ == "__main__":
    main()
