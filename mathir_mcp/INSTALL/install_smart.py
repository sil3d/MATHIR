#!/usr/bin/env python3
"""
MATHIR Smart Installer — Auto-detect ALL coding agents, inject config + system prompt.
Works on Windows, Mac, Linux. Supports 40+ coding agents.
"""

import os
import sys
import json
import platform
import shutil
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Fix Windows encoding
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# ── Colors ──────────────────────────────────────────────────
class C:
    CYAN    = "\033[96m"
    GREEN   = "\033[92m"
    RED     = "\033[91m"
    YELLOW  = "\033[93m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    RESET   = "\033[0m"

def banner():
    print(f"""
{C.CYAN}╔══════════════════════════════════════════════════════════╗
║       MATHIR Smart Installer — Auto-Detect & Inject      ║
║       5-tier cognitive memory for 40+ coding agents      ║
╚══════════════════════════════════════════════════════════╝{C.RESET}
""")

# ── Helpers ─────────────────────────────────────────────────
def _home():
    return Path.home()

def _appdata():
    return Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))

def _localappdata():
    return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))

# ── Agent Registry (40+ agents) ─────────────────────────────
# config_paths: per-platform paths to MCP config file
# config_key: JSON key where MCP servers are stored
# supports_instructions: whether we can inject system prompt
# instructions_path: per-platform path to instructions file
# install_dirs: fallback detection (program installed but no config yet)
AGENTS = [
    # ════════════════════════════════════════════════════════
    # TIER 1: MCP + System Prompt injection
    # ════════════════════════════════════════════════════════
    {
        "name": "OpenCode",
        "config_paths": {
            "windows": str(_home() / ".config" / "opencode" / "opencode.json"),
            "linux": str(_home() / ".config" / "opencode" / "opencode.json"),
            "darwin": str(_home() / ".config" / "opencode" / "opencode.json"),
        },
        "config_key": "mcp",
        "supports_instructions": True,
        "instructions_path": {
            "windows": str(_home() / ".config" / "opencode" / "GLOBAL_INSTRUCTIONS.md"),
            "linux": str(_home() / ".config" / "opencode" / "GLOBAL_INSTRUCTIONS.md"),
            "darwin": str(_home() / ".config" / "opencode" / "GLOBAL_INSTRUCTIONS.md"),
        },
        # Plugin system: deploys mathir-auto-inject.ts for true auto-injection
        # of relevant memories into the system prompt at session start.
        "plugins_subdir": "plugins",
        "plugin_variant": "opencode",
    },
    {
        "name": "MiMo Code",
        "config_paths": {
            "windows": str(_home() / ".config" / "mimocode" / "mimocode.json"),
            "linux": str(_home() / ".config" / "mimocode" / "mimocode.json"),
            "darwin": str(_home() / ".config" / "mimocode" / "mimocode.json"),
        },
        "config_key": "mcpServers",
        "supports_instructions": True,
        "instructions_path": {
            "windows": str(_home() / ".config" / "mimocode" / "GLOBAL_INSTRUCTIONS.md"),
            "linux": str(_home() / ".config" / "mimocode" / "GLOBAL_INSTRUCTIONS.md"),
            "darwin": str(_home() / ".config" / "mimocode" / "GLOBAL_INSTRUCTIONS.md"),
        },
        # Plugin system: fork of opencode, uses @mimo-ai/plugin import.
        "plugins_subdir": "plugins",
        "plugin_variant": "mimocode",
    },
    {
        "name": "Claude Code",
        "config_paths": {
            "windows": str(_home() / ".claude.json"),
            "linux": str(_home() / ".claude.json"),
            "darwin": str(_home() / ".claude.json"),
        },
        "config_key": "mcpServers",
        "supports_instructions": True,
        "instructions_path": {
            "windows": str(_home() / ".claude" / "CLAUDE.md"),
            "linux": str(_home() / ".claude" / "CLAUDE.md"),
            "darwin": str(_home() / ".claude" / "CLAUDE.md"),
        },
    },
    {
        "name": "Claude Desktop",
        "config_paths": {
            "windows": str(_appdata() / "Claude" / "claude_desktop_config.json"),
            "linux": str(_home() / ".config" / "Claude" / "claude_desktop_config.json"),
            "darwin": str(Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"),
        },
        "config_key": "mcpServers",
        "supports_instructions": False,
    },
    {
        "name": "Cursor",
        "config_paths": {
            "windows": str(_home() / ".cursor" / "mcp.json"),
            "linux": str(_home() / ".cursor" / "mcp.json"),
            "darwin": str(_home() / ".cursor" / "mcp.json"),
        },
        "config_key": "mcpServers",
        "supports_instructions": True,
        "instructions_path": {
            "windows": str(_home() / ".cursor" / "rules" / "mathir.md"),
            "linux": str(_home() / ".cursor" / "rules" / "mathir.md"),
            "darwin": str(_home() / ".cursor" / "rules" / "mathir.md"),
        },
        "install_dirs": {
            "windows": str(_localappdata() / "Programs" / "Cursor"),
            "linux": "/usr/bin/cursor",
            "darwin": "/Applications/Cursor.app",
        },
    },
    {
        "name": "Cline",
        "config_paths": {
            "windows": str(_appdata() / "Code" / "User" / "globalStorage" / "saoudrizwan.claude-dev" / "settings" / "cline_mcp_settings.json"),
            "linux": str(_home() / ".config" / "Code" / "User" / "globalStorage" / "saoudrizwan.claude-dev" / "settings" / "cline_mcp_settings.json"),
            "darwin": str(Path.home() / "Library" / "Application Support" / "Code" / "User" / "globalStorage" / "saoudrizwan.claude-dev" / "settings" / "cline_mcp_settings.json"),
        },
        "config_key": "mcpServers",
        "supports_instructions": True,
        "instructions_path": {
            "windows": str(Path.home() / "Documents" / "Cline" / "Rules" / "mathir.md"),
            "linux": str(_home() / "Documents" / "Cline" / "Rules" / "mathir.md"),
            "darwin": str(Path.home() / "Documents" / "Cline" / "Rules" / "mathir.md"),
        },
    },
    {
        "name": "Roo Code",
        "config_paths": {
            "windows": str(_home() / ".roo" / "mcp.json"),
            "linux": str(_home() / ".roo" / "mcp.json"),
            "darwin": str(_home() / ".roo" / "mcp.json"),
        },
        "config_key": "mcpServers",
        "supports_instructions": True,
        "instructions_path": {
            "windows": str(_home() / ".roo" / "rules" / "mathir.md"),
            "linux": str(_home() / ".roo" / "rules" / "mathir.md"),
            "darwin": str(_home() / ".roo" / "rules" / "mathir.md"),
        },
    },
    {
        "name": "Continue.dev",
        "config_paths": {
            "windows": str(_home() / ".continue" / "config.yaml"),
            "linux": str(_home() / ".continue" / "config.yaml"),
            "darwin": str(_home() / ".continue" / "config.yaml"),
        },
        "config_key": "mcpServers",
        "supports_instructions": True,
        "instructions_path": {
            "windows": str(_home() / ".continue" / "rules" / "mathir.md"),
            "linux": str(_home() / ".continue" / "rules" / "mathir.md"),
            "darwin": str(_home() / ".continue" / "rules" / "mathir.md"),
        },
    },
    {
        "name": "Hermes Agent",
        "config_paths": {
            "windows": str(_home() / ".hermes" / "config.yaml"),
            "linux": str(_home() / ".hermes" / "config.yaml"),
            "darwin": str(_home() / ".hermes" / "config.yaml"),
        },
        "config_key": "mcp_servers",
        "supports_instructions": True,
        "instructions_path": {
            "windows": str(_home() / ".hermes" / "SOUL.md"),
            "linux": str(_home() / ".hermes" / "SOUL.md"),
            "darwin": str(_home() / ".hermes" / "SOUL.md"),
        },
    },
    {
        "name": "pi (pi.dev)",
        "config_paths": {
            "windows": str(_home() / ".pi" / "agent" / "mcp.json"),
            "linux": str(_home() / ".pi" / "agent" / "mcp.json"),
            "darwin": str(_home() / ".pi" / "agent" / "mcp.json"),
        },
        "config_key": "mcpServers",
        "supports_instructions": False,
    },
    {
        "name": "OpenHands",
        "config_paths": {
            "windows": str(_home() / ".openhands" / "mcp.json"),
            "linux": str(_home() / ".openhands" / "mcp.json"),
            "darwin": str(_home() / ".openhands" / "mcp.json"),
        },
        "config_key": "mcpServers",
        "supports_instructions": False,
    },
    {
        "name": "Agent Zero",
        "config_paths": {
            "windows": str(_home() / ".agentzero" / "mcp.json"),
            "linux": str(_home() / ".agentzero" / "mcp.json"),
            "darwin": str(_home() / ".agentzero" / "mcp.json"),
        },
        "config_key": "mcpServers",
        "supports_instructions": False,
    },
    {
        "name": "Slate Agent",
        "config_paths": {
            "windows": str(_home() / ".config" / "slate" / "slate.json"),
            "linux": str(_home() / ".config" / "slate" / "slate.json"),
            "darwin": str(_home() / ".config" / "slate" / "slate.json"),
        },
        "config_key": "mcpServers",
        "supports_instructions": True,
        "instructions_path": {
            "windows": str(_home() / ".config" / "slate" / "AGENTS.md"),
            "linux": str(_home() / ".config" / "slate" / "AGENTS.md"),
            "darwin": str(_home() / ".config" / "slate" / "AGENTS.md"),
        },
    },
    # ════════════════════════════════════════════════════════
    # TIER 2: MCP only (no instructions injection)
    # ════════════════════════════════════════════════════════
    {
        "name": "Kilo Code",
        "config_paths": {
            "windows": str(_home() / ".config" / "kilo" / "kilo.json"),
            "linux": str(_home() / ".config" / "kilo" / "kilo.json"),
            "darwin": str(_home() / ".config" / "kilo" / "kilo.json"),
        },
        "config_key": "mcp",
        "supports_instructions": False,
        # Kilo VS Code extension uses .kilocode/mcp.json
        "vscode_config": {
            "config_key": "mcpServers",
            "project_file": ".kilocode/mcp.json",
        },
    },
    {
        "name": "Windsurf",
        "config_paths": {
            "windows": str(_home() / ".codeium" / "windsurf" / "mcp_config.json"),
            "linux": str(_home() / ".codeium" / "windsurf" / "mcp_config.json"),
            "darwin": str(_home() / ".codeium" / "windsurf" / "mcp_config.json"),
        },
        "config_key": "mcpServers",
        "supports_instructions": False,
    },
    {
        "name": "Gemini CLI",
        "config_paths": {
            "windows": str(_home() / ".gemini" / "settings.json"),
            "linux": str(_home() / ".gemini" / "settings.json"),
            "darwin": str(_home() / ".gemini" / "settings.json"),
        },
        "config_key": "mcpServers",
        "supports_instructions": False,
    },
    {
        "name": "Zcode",
        "config_paths": {
            "windows": str(_home() / ".zcode" / "v2" / "config.json"),
            "linux": str(_home() / ".zcode" / "v2" / "config.json"),
            "darwin": str(_home() / ".zcode" / "v2" / "config.json"),
        },
        "config_key": "mcpServers",
        "supports_instructions": True,
        "instructions_path": {
            "windows": str(_home() / ".zcode" / "AGENTS.md"),
            "linux": str(_home() / ".zcode" / "AGENTS.md"),
            "darwin": str(_home() / ".zcode" / "AGENTS.md"),
        },
    },
    {
        "name": "OpenClaw",
        "config_paths": {
            "windows": str(_home() / ".openclaw" / "openclaw.json"),
            "linux": str(_home() / ".openclaw" / "openclaw.json"),
            "darwin": str(_home() / ".openclaw" / "openclaw.json"),
        },
        "config_key": "mcp.servers",
        "supports_instructions": False,
    },
    {
        "name": "Kiro",
        "config_paths": {
            "windows": str(_home() / ".kiro" / "settings" / "mcp.json"),
            "linux": str(_home() / ".kiro" / "settings" / "mcp.json"),
            "darwin": str(_home() / ".kiro" / "settings" / "mcp.json"),
        },
        "config_key": "mcpServers",
        "supports_instructions": False,
    },
    {
        "name": "Qwen Code",
        "config_paths": {
            "windows": str(_home() / ".qwen" / "settings.json"),
            "linux": str(_home() / ".qwen" / "settings.json"),
            "darwin": str(_home() / ".qwen" / "settings.json"),
        },
        "config_key": "mcpServers",
        "supports_instructions": False,
    },
    {
        "name": "Crush",
        "config_paths": {
            "windows": str(_home() / ".config" / "crush" / "crush.json"),
            "linux": str(_home() / ".config" / "crush" / "crush.json"),
            "darwin": str(_home() / ".config" / "crush" / "crush.json"),
        },
        "config_key": "mcp",
        "supports_instructions": False,
    },
    {
        "name": "Warp",
        "config_paths": {
            "windows": str(_home() / ".warp" / ".mcp.json"),
            "linux": str(_home() / ".warp" / ".mcp.json"),
            "darwin": str(_home() / ".warp" / ".mcp.json"),
        },
        "config_key": "mcpServers",
        "supports_instructions": False,
    },
    {
        "name": "Trae",
        "config_paths": {
            "windows": str(_appdata() / "Trae" / "User" / "mcp.json"),
            "linux": str(_home() / ".trae" / "User" / "mcp.json"),
            "darwin": str(Path.home() / "Library" / "Application Support" / "Trae" / "User" / "mcp.json"),
        },
        "config_key": "mcpServers",
        "supports_instructions": False,
    },
    {
        "name": "Factory",
        "config_paths": {
            "windows": str(_home() / ".factory" / "mcp.json"),
            "linux": str(_home() / ".factory" / "mcp.json"),
            "darwin": str(_home() / ".factory" / "mcp.json"),
        },
        "config_key": "mcpServers",
        "supports_instructions": False,
    },
    {
        "name": "Goose",
        "config_paths": {
            "windows": str(_home() / ".config" / "goose" / "config.yaml"),
            "linux": str(_home() / ".config" / "goose" / "config.yaml"),
            "darwin": str(_home() / ".config" / "goose" / "config.yaml"),
        },
        "config_key": "extensions",
        "supports_instructions": True,
        "instructions_path": {
            "windows": str(_home() / ".config" / "goose" / "prompts" / "mathir.md"),
            "linux": str(_home() / ".config" / "goose" / "prompts" / "mathir.md"),
            "darwin": str(_home() / ".config" / "goose" / "prompts" / "mathir.md"),
        },
    },
    {
        "name": "Amp",
        "config_paths": {
            "windows": str(_home() / ".amp" / "settings.json"),
            "linux": str(_home() / ".amp" / "settings.json"),
            "darwin": str(_home() / ".amp" / "settings.json"),
        },
        "config_key": "mcpServers",
        "supports_instructions": True,
        "instructions_path": {
            "windows": str(_home() / ".amp" / "rules" / "mathir.md"),
            "linux": str(_home() / ".amp" / "rules" / "mathir.md"),
            "darwin": str(_home() / ".amp" / "rules" / "mathir.md"),
        },
    },
    {
        "name": "Zed",
        "config_paths": {
            "windows": str(_appdata() / "Zed" / "settings.json"),
            "linux": str(_home() / ".config" / "zed" / "settings.json"),
            "darwin": str(Path.home() / "Library" / "Application Support" / "Zed" / "settings.json"),
        },
        "config_key": "context_servers",
        "supports_instructions": True,
        "instructions_path": {
            "windows": str(_appdata() / "Zed" / "settings.json"),
            "linux": str(_home() / ".config" / "zed" / "settings.json"),
            "darwin": str(Path.home() / "Library" / "Application Support" / "Zed" / "settings.json"),
        },
    },
    {
        "name": "Augment Code",
        "config_paths": {
            "windows": str(_home() / ".augment" / "settings.json"),
            "linux": str(_home() / ".augment" / "settings.json"),
            "darwin": str(_home() / ".augment" / "settings.json"),
        },
        "config_key": "mcpServers",
        "supports_instructions": True,
        "instructions_path": {
            "windows": str(_home() / ".augment" / "settings.json"),
            "linux": str(_home() / ".augment" / "settings.json"),
            "darwin": str(_home() / ".augment" / "settings.json"),
        },
    },
    {
        "name": "Antigravity",
        "config_paths": {
            "windows": str(_home() / ".gemini" / "config" / "mcp_config.json"),
            "linux": str(_home() / ".gemini" / "config" / "mcp_config.json"),
            "darwin": str(_home() / ".gemini" / "config" / "mcp_config.json"),
        },
        "config_key": "mcpServers",
        "supports_instructions": False,
    },
    {
        "name": "GitHub Copilot",
        "config_paths": {
            "windows": str(_home() / ".vscode" / "mcp.json"),
            "linux": str(_home() / ".vscode" / "mcp.json"),
            "darwin": str(_home() / ".vscode" / "mcp.json"),
        },
        "config_key": "servers",
        "supports_instructions": True,
        "instructions_path": {
            "windows": str(_home() / ".github" / "copilot-instructions.md"),
            "linux": str(_home() / ".github" / "copilot-instructions.md"),
            "darwin": str(_home() / ".github" / "copilot-instructions.md"),
        },
    },
    {
        "name": "Aider",
        "config_paths": {
            "windows": str(_home() / ".aider.conf.yml"),
            "linux": str(_home() / ".aider.conf.yml"),
            "darwin": str(_home() / ".aider.conf.yml"),
        },
        "config_key": "mcpServers",
        "supports_instructions": False,
    },
    {
        "name": "Codex (OpenAI)",
        "config_paths": {
            "windows": str(_home() / ".codex" / "config.toml"),
            "linux": str(_home() / ".codex" / "config.toml"),
            "darwin": str(_home() / ".codex" / "config.toml"),
        },
        "config_key": "mcp_servers",
        "supports_instructions": True,
        "instructions_path": {
            "windows": str(_home() / ".codex" / "config.toml"),
            "linux": str(_home() / ".codex" / "config.toml"),
            "darwin": str(_home() / ".codex" / "config.toml"),
        },
    },
    # ════════════════════════════════════════════════════════
    # TIER 3: Emerging agents (2026)
    # ════════════════════════════════════════════════════════
    {
        "name": "Codebuff",
        "config_paths": {
            "windows": str(_home() / ".codebuff" / "config.json"),
            "linux": str(_home() / ".codebuff" / "config.json"),
            "darwin": str(_home() / ".codebuff" / "config.json"),
        },
        "config_key": "mcpServers",
        "supports_instructions": False,
    },
    {
        "name": "Letaido",
        "config_paths": {
            "windows": str(_home() / ".letaido" / "config.json"),
            "linux": str(_home() / ".letaido" / "config.json"),
            "darwin": str(_home() / ".letaido" / "config.json"),
        },
        "config_key": "mcpServers",
        "supports_instructions": False,
    },
    {
        "name": "CodeAF",
        "config_paths": {
            "windows": str(_home() / ".codeaf" / "config.json"),
            "linux": str(_home() / ".codeaf" / "config.json"),
            "darwin": str(_home() / ".codeaf" / "config.json"),
        },
        "config_key": "mcpServers",
        "supports_instructions": False,
    },
    {
        "name": "JONI",
        "config_paths": {
            "windows": str(_home() / ".joni" / "config.json"),
            "linux": str(_home() / ".joni" / "config.json"),
            "darwin": str(_home() / ".joni" / "config.json"),
        },
        "config_key": "mcpServers",
        "supports_instructions": False,
    },
    {
        "name": "Minara",
        "config_paths": {
            "windows": str(_home() / ".minara" / "config.json"),
            "linux": str(_home() / ".minara" / "config.json"),
            "darwin": str(_home() / ".minara" / "config.json"),
        },
        "config_key": "mcpServers",
        "supports_instructions": False,
    },
    {
        "name": "V12X",
        "config_paths": {
            "windows": str(_home() / ".v12x" / "config.json"),
            "linux": str(_home() / ".v12x" / "config.json"),
            "darwin": str(_home() / ".v12x" / "config.json"),
        },
        "config_key": "mcpServers",
        "supports_instructions": False,
    },
    {
        "name": "Anakot Agent",
        "config_paths": {
            "windows": str(_home() / ".anakot" / "config.json"),
            "linux": str(_home() / ".anakot" / "config.json"),
            "darwin": str(_home() / ".anakot" / "config.json"),
        },
        "config_key": "mcpServers",
        "supports_instructions": False,
    },
    {
        "name": "PostQode",
        "config_paths": {
            "windows": str(_home() / ".postqode" / "config.json"),
            "linux": str(_home() / ".postqode" / "config.json"),
            "darwin": str(_home() / ".postqode" / "config.json"),
        },
        "config_key": "mcpServers",
        "supports_instructions": False,
    },
    {
        "name": "starchild",
        "config_paths": {
            "windows": str(_home() / ".starchild" / "config.json"),
            "linux": str(_home() / ".starchild" / "config.json"),
            "darwin": str(_home() / ".starchild" / "config.json"),
        },
        "config_key": "mcpServers",
        "supports_instructions": False,
    },
    {
        "name": "Flo Agent",
        "config_paths": {
            "windows": str(_home() / ".flo" / "config.json"),
            "linux": str(_home() / ".flo" / "config.json"),
            "darwin": str(_home() / ".flo" / "config.json"),
        },
        "config_key": "mcpServers",
        "supports_instructions": False,
    },
    {
        "name": "camelAI",
        "config_paths": {
            "windows": str(_home() / ".camelai" / "config.json"),
            "linux": str(_home() / ".camelai" / "config.json"),
            "darwin": str(_home() / ".camelai" / "config.json"),
        },
        "config_key": "mcpServers",
        "supports_instructions": False,
    },
    {
        "name": "Kern Agent",
        "config_paths": {
            "windows": str(_home() / ".kern" / "config.json"),
            "linux": str(_home() / ".kern" / "config.json"),
            "darwin": str(_home() / ".kern" / "config.json"),
        },
        "config_key": "mcpServers",
        "supports_instructions": False,
    },
    {
        "name": "AGNT",
        "config_paths": {
            "windows": str(_home() / ".agnt" / "config.json"),
            "linux": str(_home() / ".agnt" / "config.json"),
            "darwin": str(_home() / ".agnt" / "config.json"),
        },
        "config_key": "mcpServers",
        "supports_instructions": False,
    },
    {
        "name": "Verdent",
        "config_paths": {
            "windows": str(_home() / ".verdent" / "config.json"),
            "linux": str(_home() / ".verdent" / "config.json"),
            "darwin": str(_home() / ".verdent" / "config.json"),
        },
        "config_key": "mcpServers",
        "supports_instructions": False,
    },
    {
        "name": "Ante",
        "config_paths": {
            "windows": str(_home() / ".ante" / "config.json"),
            "linux": str(_home() / ".ante" / "config.json"),
            "darwin": str(_home() / ".ante" / "config.json"),
        },
        "config_key": "mcpServers",
        "supports_instructions": False,
    },
    {
        "name": "GitLawb",
        "config_paths": {
            "windows": str(_home() / ".gitlawb" / "config.json"),
            "linux": str(_home() / ".gitlawb" / "config.json"),
            "darwin": str(_home() / ".gitlawb" / "config.json"),
        },
        "config_key": "mcpServers",
        "supports_instructions": False,
    },
    {
        "name": "Favur",
        "config_paths": {
            "windows": str(_home() / ".favur" / "config.json"),
            "linux": str(_home() / ".favur" / "config.json"),
            "darwin": str(_home() / ".favur" / "config.json"),
        },
        "config_key": "mcpServers",
        "supports_instructions": False,
    },
    {
        "name": "OpenSquilla",
        "config_paths": {
            "windows": str(_home() / ".opensquilla" / "config.json"),
            "linux": str(_home() / ".opensquilla" / "config.json"),
            "darwin": str(_home() / ".opensquilla" / "config.json"),
        },
        "config_key": "mcpServers",
        "supports_instructions": False,
    },
]


# ── Core Functions ──────────────────────────────────────────
def detect_platform() -> str:
    s = platform.system().lower()
    return {"windows": "windows", "linux": "linux", "darwin": "darwin"}.get(s, "linux")


def _deploy_auto_inject_plugin(agent: Dict, agent_config_dir: Path) -> Tuple[bool, str]:
    """Deploy the mathir-auto-inject plugin for agents that support it.

    Plugin-aware hosts (opencode, mimocode) load .ts plugins from their
    plugins/ subdirectory. The plugin hooks session.started +
    experimental.chat.system.transform to inject relevant memories into
    the system prompt — true auto-injection that doesn't rely on the agent
    remembering to call memory_recall.

    Returns (success, message).
    """
    plugins_subdir = agent.get("plugins_subdir")
    plugin_variant = agent.get("plugin_variant")
    if not plugins_subdir or not plugin_variant:
        return True, "no plugin support for this agent (directive-only)"

    # Locate the plugin template: mathir_mcp/{variant}_templates/plugins/mathir-auto-inject.ts
    source_dir = Path(__file__).resolve().parent.parent  # mathir_mcp/
    plugin_src = source_dir / f"{plugin_variant}_templates" / "plugins" / "mathir-auto-inject.ts"
    if not plugin_src.is_file():
        return False, f"plugin template not found: {plugin_src}"

    plugins_dst_dir = agent_config_dir / plugins_subdir
    try:
        plugins_dst_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        return False, f"mkdir {plugins_dst_dir} failed: {e}"

    plugin_dst = plugins_dst_dir / "mathir-auto-inject.ts"
    try:
        import shutil
        shutil.copy2(plugin_src, plugin_dst)
    except OSError as e:
        return False, f"copy plugin failed: {e}"

    return True, f"plugin deployed → {plugin_dst}"


def _copy_mathir_to_agent(agent_config_dir: Path) -> Path:
    """Copy the entire mathir_mcp package into the agent's tools directory.

    This makes MATHIR standalone — no dependency on source folder location.
    Returns the path to the copied package.
    """
    import shutil
    
    source_dir = Path(__file__).resolve().parent.parent  # mathir_mcp parent = MATHIR repo
    mathir_source = source_dir / "mathir_mcp"
    target_dir = agent_config_dir / "tools" / "mathir_mcp"
    
    # Skip if already installed and same version
    if target_dir.exists():
        version_file = target_dir / "VERSION"
        source_version = mathir_source / "VERSION"
        if version_file.exists() and source_version.exists():
            if version_file.read_text().strip() == source_version.read_text().strip():
                return target_dir
    
    # Copy entire package
    if target_dir.exists():
        shutil.rmtree(target_dir)
    
    shutil.copytree(mathir_source, target_dir, ignore=shutil.ignore_patterns(
        '__pycache__', '*.pyc', '*.pyo', '.git', '*.egg-info', 'dev', 'tests'
    ))
    
    # Write VERSION file
    version_file = target_dir / "VERSION"
    version_file.write_text("8.4.0")
    
    print(f"  {C.GREEN}Copied mathir_mcp to: {target_dir}{C.RESET}")
    return target_dir


def _resolve_server_path(agent_config_dir: Path = None) -> Path:
    """Resolve the server path at INSTALL TIME, not hardcoded.

    Priority:
      1. MATHIR_SERVER_PATH env var (user override)
      2. Copied package in agent's tools dir (standalone)
      3. mathir_lib/mathir_mcp_server.py relative to this script
      4. mathir_mcp_server.py in PATH (pip-installed)
    """
    # 1. Env var override (for non-standard installs)
    env_path = os.environ.get("MATHIR_SERVER_PATH")
    if env_path:
        p = Path(env_path).resolve()
        if p.exists():
            return p
        print(f"{C.YELLOW}  WARNING: MATHIR_SERVER_PATH={env_path} — file not found, falling back{C.RESET}")

    # 2. Copied package in agent's tools dir (standalone install)
    if agent_config_dir:
        copied = agent_config_dir / "tools" / "mathir_mcp" / "mathir_lib" / "mathir_mcp_server.py"
        if copied.exists():
            return copied

    # 3. Relative to this script (standard install)
    mathir_dir = Path(__file__).resolve().parent
    server_path = mathir_dir / "mathir_lib" / "mathir_mcp_server.py"
    if server_path.exists():
        return server_path

    # 4. pip-installed module (python -m mathir_mcp.mathir_lib.mathir_mcp_server)
    # Return a sentinel — caller will use -m invocation
    return None


def get_mathir_server_cmd(agent_config_dir: Path = None) -> List[str]:
    """Return the command to launch the MATHIR MCP server.
    
    Uses copied standalone path if available, else falls back to source.
    """
    server_path = _resolve_server_path(agent_config_dir)
    if server_path is not None:
        return ["python", str(server_path)]
    # Fallback: use -m (requires pip install -e .)
    return ["python", "-m", "mathir_mcp.mathir_lib.mathir_mcp_server"]


def get_mathir_server_entry(agent_config_dir: Path = None) -> Dict:
    cmd = get_mathir_server_cmd(agent_config_dir)
    return {
        "type": "local",
        "command": cmd,
        "environment": {"MATHIR_EMBEDDING_DIM": "384"},
        "enabled": True,
    }


def detect_agents() -> List[Dict]:
    platform_name = detect_platform()
    detected = []
    for agent in AGENTS:
        config_path_str = agent["config_paths"].get(platform_name)
        if not config_path_str:
            continue
        config_path = Path(config_path_str)
        found = config_path.exists()
        # Fallback: check install_dirs
        if not found and agent.get("install_dirs"):
            install_dirs = agent["install_dirs"]
            install_dir = install_dirs.get(platform_name) if isinstance(install_dirs, dict) else install_dirs
            if install_dir and Path(install_dir).exists():
                found = True
        # Resolve instructions path
        instr_path = None
        if agent.get("supports_instructions"):
            instr_raw = agent.get("instructions_path", {})
            instr_path = instr_raw.get(platform_name) if isinstance(instr_raw, dict) else instr_raw
        detected.append({
            **agent,
            "config_path": config_path,
            "config_path_str": config_path_str,
            "instructions_path_resolved": instr_path,
            "installed": found,
        })
    return detected


def read_config(path: Path) -> Dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def write_config(path: Path, data: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def inject_mcp_config(agent: Dict) -> Tuple[bool, str]:
    config_path = agent["config_path"]
    config_key = agent["config_key"]
    if not config_key:
        return False, "No config key defined"
    
    # Step 1: Copy mathir_mcp into agent's tools directory (standalone)
    config_dir = Path(config_path).parent
    try:
        copied_dir = _copy_mathir_to_agent(config_dir)
    except Exception as e:
        return False, f"Failed to copy MATHIR: {e}"

    # Step 1b: Deploy auto-inject plugin (for plugin-aware hosts only)
    try:
        plugin_ok, plugin_msg = _deploy_auto_inject_plugin(agent, config_dir)
        print(f"  {C.CYAN}Plugin: {plugin_msg}{C.RESET}")
    except Exception as e:
        print(f"  {C.YELLOW}Plugin deploy skipped: {e}{C.RESET}")
    
    # Step 2: Get command pointing to copied standalone package
    config = read_config(config_path)
    entry = get_mathir_server_entry(config_dir)
    
    # Handle different formats
    if config_key in ("mcpServers", "mcp"):
        if config_key not in config:
            config[config_key] = {}
        # Clean old entries
        old_keys = [k for k in config[config_key] if k == "mathir" and config[config_key][k].get("command", [""])[-1] != entry["command"][-1]]
        for k in old_keys:
            del config[config_key][k]
        config[config_key]["mathir"] = entry
    elif config_key == "mcp.servers":
        if "mcp" not in config:
            config["mcp"] = {}
        if "servers" not in config["mcp"]:
            config["mcp"]["servers"] = {}
        config["mcp"]["servers"]["mathir"] = entry
    elif config_key == "context_servers":
        if config_key not in config:
            config[config_key] = {}
        config[config_key]["mathir"] = {
            "command": entry["command"],
            "settings": entry.get("environment", {}),
        }
    elif config_key == "extensions":
        return False, "YAML format not supported yet"
    elif config_key == "mcp_servers":
        return False, "TOML format not supported yet"
    else:
        if config_key not in config:
            config[config_key] = {}
        config[config_key]["mathir"] = entry
    write_config(config_path, config)
    # Also create .kilocode/mcp.json for VS Code extension (only if in a project)
    if agent.get("vscode_config"):
        vscode_key = agent["vscode_config"]["config_key"]
        project_file = agent["vscode_config"]["project_file"]
        in_project = any((Path.cwd() / p).exists() for p in [".git", "package.json", "pyproject.toml", "Cargo.toml"])
        if in_project:
            vscode_path = Path.cwd() / project_file
            vscode_config = read_config(vscode_path)
            if vscode_key not in vscode_config:
                vscode_config[vscode_key] = {}
            vscode_config[vscode_key]["mathir"] = entry
            write_config(vscode_path, vscode_config)
            return True, f"Injected into {config_path} + {vscode_path}"
    return True, f"Injected into {config_path}"


def inject_instructions(agent: Dict) -> Tuple[bool, str]:
    if not agent.get("supports_instructions"):
        return False, "No instructions support"
    instructions_path_str = agent.get("instructions_path_resolved")
    if not instructions_path_str:
        return False, "No instructions path defined"
    instructions_path = Path(instructions_path_str)
    mathir_dir = Path(__file__).parent
    mathir_instructions = mathir_dir / "GLOBAL_INSTRUCTIONS.md"
    if not mathir_instructions.exists():
        return False, f"MATHIR GLOBAL_INSTRUCTIONS.md not found"
    with open(mathir_instructions, "r", encoding="utf-8") as f:
        content = f.read()
    # Check if already injected (look for stable MATHIR signature, not version)
    if instructions_path.exists():
        with open(instructions_path, "r", encoding="utf-8") as f:
            existing = f.read()
        if "MATHIR Memory-Augmented Tensor Hybrid" in existing:
            return True, f"Already injected"
    instructions_path.parent.mkdir(parents=True, exist_ok=True)
    if instructions_path.exists():
        with open(instructions_path, "r", encoding="utf-8") as f:
            existing = f.read()
        with open(instructions_path, "w", encoding="utf-8") as f:
            f.write(existing + "\n\n" + content)
    else:
        with open(instructions_path, "w", encoding="utf-8") as f:
            f.write(content)
    return True, f"Injected into {instructions_path}"


def show_menu(detected: List[Dict]) -> List[Dict]:
    print(f"{C.BOLD}Detected coding agents:{C.RESET}\n")
    installed = [a for a in detected if a["installed"]]
    not_installed = [a for a in detected if not a["installed"]]
    if installed:
        print(f"{C.GREEN}  INSTALLED ({len(installed)}):{C.RESET}")
        for i, agent in enumerate(installed, 1):
            instr = f" + instructions" if agent["supports_instructions"] else ""
            print(f"    {C.CYAN}[{i}]{C.RESET} {agent['name']}{C.DIM}{instr}{C.RESET}")
        print()
    if not_installed:
        print(f"{C.DIM}  NOT FOUND ({len(not_installed)}):{C.RESET}")
        for agent in not_installed:
            print(f"    {C.DIM}  {agent['name']}{C.RESET}")
        print()
    print(f"{C.BOLD}Options:{C.RESET}")
    print(f"    {C.CYAN}[A]{C.RESET} Configure ALL installed agents")
    print(f"    {C.CYAN}[1-N]{C.RESET} Select specific agents (comma-separated)")
    print(f"    {C.CYAN}[Q]{C.RESET} Quit\n")
    choice = input(f"{C.YELLOW}  Your choice: {C.RESET}").strip().upper()
    if choice == "Q":
        print(f"\n{C.DIM}Cancelled.{C.RESET}")
        sys.exit(0)
    if choice == "A":
        return installed
    selected = []
    for num in choice.split(","):
        num = num.strip()
        if num.isdigit():
            idx = int(num) - 1
            if 0 <= idx < len(installed):
                selected.append(installed[idx])
    return selected


# ── Auto-Start Setup (Windows / macOS / Linux) ───────────────
def _mathir_bin_dir() -> Path:
    """Resolve the MATHIR bin/ directory from this script's location.

    Works for both source layout (mathir_mcp/INSTALL/install_smart.py -> ../bin)
    and deployed layout (~/.config/opencode/bin/INSTALL/install_smart.py -> ../bin).
    """
    return Path(__file__).resolve().parent.parent / "bin"


def _setup_autostart_windows(bin_dir: Path, dry_run: bool = False) -> Tuple[bool, str]:
    """Set up MATHIR daemon auto-start on Windows.

    Strategy:
      1. No-admin (preferred): copy auto_start.bat to the user's Startup folder.
         Uses the existing auto_start.bat (which uses `start "" /B` to detach).
      2. Optionally also drop a hidden VBS wrapper for zero-flash startup.
      3. If admin (best-effort): register a Task Scheduler entry via
         auto_start_helpers.ps1.
    Returns (success, message).
    """
    userprofile = Path(os.environ.get("USERPROFILE", str(Path.home())))
    startup = userprofile / "AppData" / "Roaming" / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"

    bat_src = bin_dir / "auto_start.bat"
    vbs_src = bin_dir / "auto_start_vbs.vbs"
    helpers_ps1 = bin_dir / "auto_start_helpers.ps1"

    if not bat_src.exists():
        return False, f"Missing source: {bat_src}"

    if dry_run:
        return True, (
            f"Would copy auto_start.bat -> {startup}\\mathir_daemon_startup.bat "
            f"+ optional VBS wrapper (no admin) "
            f"+ optional Task Scheduler entry via {helpers_ps1.name}"
        )

    try:
        startup.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        return False, f"Cannot create Startup folder ({startup}): {e}"

    # 1) Copy auto_start.bat to Startup folder (always works, no admin)
    target_bat = startup / "mathir_daemon_startup.bat"
    try:
        shutil.copy2(bat_src, target_bat)
    except Exception as e:
        return False, f"Failed to copy auto_start.bat -> Startup: {e}"

    placed = [f"copied .bat -> {target_bat}"]

    # 2) Optional: hidden VBS wrapper so the console never flashes
    if vbs_src.exists():
        try:
            import re
            target_vbs = startup / "mathir_daemon_startup.vbs"
            vbs_text = vbs_src.read_text(encoding="utf-8")
            # Rewrite the Const BAT_PATH = "..." line to point at the .bat we
            # just placed in the Startup folder. Use a callable for the
            # replacement so Windows backslashes in target_bat are NOT
            # interpreted as regex backreferences by re.sub.
            new_line = 'Const BAT_PATH = "{}"'.format(str(target_bat))
            vbs_text = re.sub(
                r'Const BAT_PATH\s*=\s*"[^"]+"',
                lambda _m: new_line,
                vbs_text,
                count=1,
            )
            target_vbs.write_text(vbs_text, encoding="utf-8")
            placed.append(f"copied .vbs -> {target_vbs}")
        except Exception as e:
            placed.append(f"VBS wrapper skipped ({e})")

    # 3) Optional: Task Scheduler via helpers (only if elevated). Best-effort.
    if helpers_ps1.exists():
        try:
            probe = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "(New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)"],
                capture_output=True, text=True, timeout=10,
            )
            if "True" in probe.stdout:
                # Try to register a Task Scheduler entry (DELAYED, so it doesn't
                # block login if python is slow to spin up).
                task_cmd = (
                    f"$a = New-ScheduledTaskAction -Execute 'powershell.exe' "
                    f"-Argument '-NoProfile -ExecutionPolicy Bypass -File \"{helpers_ps1}\"'; "
                    f"$t = New-ScheduledTaskTrigger -AtLogOn; "
                    f"$p = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive; "
                    f"Register-ScheduledTask -TaskName 'MATHIR Daemon' "
                    f"-Action $a -Trigger $t -Principal $p -Force | Out-Null; "
                    f"Write-Output 'TASK_REGISTERED'"
                )
                reg = subprocess.run(
                    ["powershell", "-NoProfile", "-Command", task_cmd],
                    capture_output=True, text=True, timeout=30,
                )
                if "TASK_REGISTERED" in (reg.stdout or ""):
                    placed.append("registered Task Scheduler entry: 'MATHIR Daemon'")
                else:
                    placed.append(f"Task Scheduler skipped (register output: {(reg.stdout or '').strip()[:120]})")
            else:
                placed.append("Task Scheduler skipped (not admin — Startup folder launcher is active)")
        except Exception as e:
            placed.append(f"Task Scheduler probe failed ({e})")

    return True, "; ".join(placed)


def _setup_autostart_macos(bin_dir: Path, dry_run: bool = False) -> Tuple[bool, str]:
    """Set up MATHIR daemon auto-start on macOS via launchd.

    Renders com.mathir.daemon.plist with the actual user's $HOME path (since
    plist strings are literal and `~` is not expanded), copies it into
    ~/Library/LaunchAgents/, and loads it with `launchctl load -w`.
    """
    plist_src = bin_dir / "com.mathir.daemon.plist"
    if not plist_src.exists():
        return False, f"Missing source: {plist_src}"

    home = Path.home()
    launch_agents = home / "Library" / "LaunchAgents"
    target = launch_agents / "com.mathir.daemon.plist"

    if dry_run:
        return True, (
            f"Would render {plist_src.name} with home={home}, "
            f"copy -> {target}, then `launchctl load -w {target}`"
        )

    try:
        launch_agents.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        return False, f"Cannot create {launch_agents}: {e}"

    # Render: substitute /Users/USERNAME placeholder with actual $HOME.
    python_path = "/usr/bin/python3"
    # Try to discover a venv python first (more user-friendly)
    venv_python = home / ".config" / "opencode" / "bin" / ".venv" / "bin" / "python3"
    if venv_python.exists():
        python_path = str(venv_python)

    rendered = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
        '<plist version="1.0">\n'
        '<dict>\n'
        '    <key>Label</key>\n'
        '    <string>com.mathir.daemon</string>\n'
        '    <key>ProgramArguments</key>\n'
        '    <array>\n'
        f'        <string>{python_path}</string>\n'
        f'        <string>{home}/.config/opencode/bin/mathir_daemon.py</string>\n'
        '    </array>\n'
        '    <key>RunAtLoad</key>\n'
        '    <true/>\n'
        '    <key>KeepAlive</key>\n'
        '    <true/>\n'
        '    <key>StandardOutPath</key>\n'
        f'    <string>{home}/.config/opencode/bin/mathir_daemon.log</string>\n'
        '    <key>StandardErrorPath</key>\n'
        f'    <string>{home}/.config/opencode/bin/mathir_daemon.err.log</string>\n'
        '    <key>WorkingDirectory</key>\n'
        f'    <string>{home}/.config/opencode/bin</string>\n'
        '</dict>\n'
        '</plist>\n'
    )

    try:
        target.write_text(rendered, encoding="utf-8")
    except Exception as e:
        return False, f"Failed to write {target}: {e}"

    # Unload old version if present, then load new one
    try:
        subprocess.run(
            ["launchctl", "unload", str(target)],
            capture_output=True, timeout=10,
        )
    except Exception:
        pass  # not loaded yet is fine

    try:
        load = subprocess.run(
            ["launchctl", "load", "-w", str(target)],
            capture_output=True, text=True, timeout=15,
        )
    except FileNotFoundError:
        return False, "launchctl not found on PATH (are you on macOS?)"
    except Exception as e:
        return False, f"launchctl load failed: {e}"

    if load.returncode != 0:
        return False, f"launchctl load returned {load.returncode}: {(load.stderr or load.stdout).strip()}"

    return True, f"wrote {target}; `launchctl load -w` succeeded (RunAtLoad + KeepAlive enabled)"


def _setup_autostart_linux(bin_dir: Path, dry_run: bool = False) -> Tuple[bool, str]:
    """Set up MATHIR daemon auto-start on Linux via systemd user services.

    Copies mathir-daemon.service to ~/.config/systemd/user/, runs daemon-reload,
    then enables + starts the service. Warns about `loginctl enable-linger`
    on headless servers (without it, user services stop when the user logs out).
    """
    service_src = bin_dir / "mathir-daemon.service"
    if not service_src.exists():
        return False, f"Missing source: {service_src}"

    home = Path.home()
    user_systemd = home / ".config" / "systemd" / "user"
    target = user_systemd / "mathir-daemon.service"

    if dry_run:
        return True, (
            f"Would copy {service_src.name} -> {target}, "
            f"run `systemctl --user daemon-reload && enable --now mathir-daemon`. "
            f"Note: headless servers need `loginctl enable-linger $USER`."
        )

    # Detect if systemd is available
    systemctl = shutil.which("systemctl")
    if not systemctl:
        return False, "systemctl not found on PATH (systemd not available on this Linux)"

    try:
        user_systemd.mkdir(parents=True, exist_ok=True)
        shutil.copy2(service_src, target)
    except Exception as e:
        return False, f"Failed to install service file: {e}"

    notes = []

    try:
        subprocess.run([systemctl, "--user", "daemon-reload"],
                       capture_output=True, timeout=15, check=True)
    except subprocess.CalledProcessError as e:
        return False, f"systemctl --user daemon-reload failed: {e}"

    try:
        subprocess.run([systemctl, "--user", "enable", "mathir-daemon.service"],
                       capture_output=True, timeout=15, check=True)
    except subprocess.CalledProcessError as e:
        return False, f"systemctl --user enable mathir-daemon.service failed: {e}"

    try:
        subprocess.run([systemctl, "--user", "start", "mathir-daemon.service"],
                       capture_output=True, timeout=15, check=True)
    except subprocess.CalledProcessError as e:
        notes.append(f"enable OK, but start failed: {e}")

    # Linger warning for headless / server use
    linger_path = Path("/var/lib/systemd/linger") / os.environ.get("USER", "")
    if not linger_path.exists():
        notes.append(
            "Heads-up: for headless servers, run "
            f"`loginctl enable-linger {os.environ.get('USER', '$USER')}` "
            "so the daemon survives logout."
        )

    msg = f"installed {target}; enabled + started via systemctl --user"
    if notes:
        msg += " (" + "; ".join(notes) + ")"
    return True, msg


def _setup_autostart(dry_run: bool = False) -> Tuple[bool, str]:
    """Top-level auto-start dispatcher. Detects OS and runs the right setup.

    Never raises — failures return (False, reason) so the install can continue.
    """
    plat = detect_platform()
    bin_dir = _mathir_bin_dir()

    if not bin_dir.exists():
        return False, f"MATHIR bin/ directory not found at {bin_dir}"

    if dry_run:
        if plat == "windows":
            return _setup_autostart_windows(bin_dir, dry_run=True)
        if plat == "darwin":
            return _setup_autostart_macos(bin_dir, dry_run=True)
        if plat == "linux":
            return _setup_autostart_linux(bin_dir, dry_run=True)
        return False, f"Unknown platform '{plat}' — no auto-start configured"

    print(f"  {C.CYAN}Setting up auto-start for {plat}...{C.RESET}")

    try:
        if plat == "windows":
            ok, msg = _setup_autostart_windows(bin_dir)
        elif plat == "darwin":
            ok, msg = _setup_autostart_macos(bin_dir)
        elif plat == "linux":
            ok, msg = _setup_autostart_linux(bin_dir)
        else:
            return False, f"Unsupported platform: {plat}"
    except Exception as e:
        return False, f"Auto-start setup raised: {type(e).__name__}: {e}"

    if ok:
        print(f"    {C.GREEN}\u2713{C.RESET} Auto-start: {msg}")
    else:
        print(f"    {C.YELLOW}!{C.RESET} Auto-start: {msg}")
    return ok, msg


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="install_smart.py",
        description="MATHIR Smart Installer — detect coding agents, inject config, "
                    "and (optionally) set up daemon auto-start.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without modifying anything (auto-start step is described).",
    )
    p.add_argument(
        "--no-autostart",
        action="store_true",
        help="Skip the auto-start setup step (agent config injection only).",
    )
    p.add_argument(
        "--autostart-only",
        action="store_true",
        help="Only set up auto-start for the MATHIR daemon — skip agent detection/injection.",
    )
    return p.parse_args(argv)


def main():
    args = _parse_args()

    # ── --autostart-only short-circuit ──
    if args.autostart_only:
        banner()
        print(f"{C.CYAN}[autostart-only]{C.RESET} Setting up auto-start (no agent changes).\n")
        ok, msg = _setup_autostart(dry_run=args.dry_run)
        if ok:
            print(f"\n{C.GREEN}\u2713 Auto-start ready.{C.RESET}")
            print(f"  {msg}\n")
        else:
            print(f"\n{C.YELLOW}! Auto-start setup failed.{C.RESET}")
            print(f"  {msg}\n")
        return

    banner()
    # ── Verify server path resolves correctly ──
    server_path = _resolve_server_path()
    cmd = get_mathir_server_cmd()
    if server_path is not None:
        print(f"{C.GREEN}  Server found:{C.RESET} {server_path}")
    else:
        print(f"{C.YELLOW}  Server not found locally — using -m fallback (requires pip install -e .){C.RESET}")
    print(f"{C.DIM}  Command: {' '.join(cmd)}{C.RESET}\n")

    if args.dry_run:
        print(f"{C.CYAN}[1/3]{C.RESET} Detecting coding agents (dry-run)...\n")
    else:
        print(f"{C.CYAN}[1/3]{C.RESET} Detecting coding agents...\n")
    detected = detect_agents()
    if not detected:
        print(f"{C.RED}No coding agents found.{C.RESET}")
        print(f"""
{C.BOLD}── No Agent? Install OpenCode (free models) ──{C.RESET}

  EN: Install OpenCode from https://opencode.ai/ — many free models available.
      Then give this MATHIR folder to OpenCode and it will configure itself.

  FR: Installez OpenCode depuis https://opencode.ai/ — beaucoup de modeles gratuits.
      Donnez ce dossier MATHIR a OpenCode et il se configurera tout seul.

  ES: Instale OpenCode desde https://opencode.ai/ — hay muchos modelos gratuitos.
      De esta carpeta MATHIR a OpenCode y se configurara solo.

  ZH: install OpenCode https://opencode.ai/ — henduo mofei moxing.
      GEI OpenCode zhege MATHIR danglu jiu hui zidong peizhi.

{C.BOLD}── Troubleshooting ──{C.RESET}

  EN: If installer fails, give the entire ~/.config/MATHIR/ folder to your coding agent.
      It will read docs/AGENT.md and configure MATHIR automatically.

  FR: Si l'installeur echoue, donnez tout le dossier ~/.config/MATHIR/ a votre agent de code.
      Il lira docs/AGENT.md et configurera MATHIR automatiquement.

  ES: Si el instalador falla, dea toda la carpeta ~/.config/MATHIR/ a su agente de codigo.
      Leera docs/AGENT.md y configurara MATHIR automaticamente.

  ZH: Ruguio anzhuan shibai, ba ~/.config/MATHIR/ zhengge danglu gei nide coding agent.
      Hui du docs/AGENT.md zidong peizhi MATHIR.
""")
        # Even with no agent, autostart is still useful
        if not args.no_autostart:
            print(f"\n{C.CYAN}[2/3]{C.RESET} Auto-start setup...\n")
            _setup_autostart(dry_run=args.dry_run)
        return

    if args.dry_run:
        selected = installed_only(detected)
        if not selected:
            print(f"{C.YELLOW}No installed agents detected in dry-run.{C.RESET}")
            if not args.no_autostart:
                print(f"\n{C.CYAN}[2/3]{C.RESET} Auto-start setup (dry-run)...\n")
                ok, msg = _setup_autostart(dry_run=True)
                print(f"    {C.GREEN}\u2713{C.RESET} {msg}" if ok else f"    {C.YELLOW}!{C.RESET} {msg}")
            return
        print(f"{C.CYAN}[2/3]{C.RESET} Dry-run: would inject into {len(selected)} agent(s):\n")
        for agent in selected:
            print(f"    {C.CYAN}\u00b7{C.RESET} {agent['name']}")
        print()
        if not args.no_autostart:
            print(f"{C.CYAN}[3/3]{C.RESET} Auto-start setup (dry-run)...\n")
            ok, msg = _setup_autostart(dry_run=True)
            print(f"    {C.GREEN}\u2713{C.RESET} {msg}" if ok else f"    {C.YELLOW}!{C.RESET} {msg}")
        print(f"\n{C.DIM}  (dry-run: no files were modified){C.RESET}\n")
        return

    selected = show_menu(detected)
    if not selected:
        print(f"\n{C.YELLOW}No agents selected.{C.RESET}")
        return
    print(f"\n{C.CYAN}[2/3]{C.RESET} Injecting MATHIR config + instructions...\n")
    needs_manual_instructions = []
    for agent in selected:
        print(f"  {C.BOLD}{agent['name']}{C.RESET}")
        ok, msg = inject_mcp_config(agent)
        if ok:
            print(f"    {C.GREEN}\u2713{C.RESET} MCP config: {msg}")
        else:
            print(f"    {C.YELLOW}!{C.RESET} MCP config: {msg}")
        ok, msg = inject_instructions(agent)
        if ok:
            print(f"    {C.GREEN}\u2713{C.RESET} Instructions: {msg}")
        else:
            print(f"    {C.YELLOW}!{C.RESET} Instructions: {msg}")
            if not agent.get("supports_instructions"):
                needs_manual_instructions.append(agent["name"])
        print()

    if args.no_autostart:
        print(f"{C.CYAN}[3/3]{C.RESET} Done!\n")
    else:
        print(f"{C.CYAN}[3/3]{C.RESET} Setting up MATHIR daemon auto-start...\n")
        _setup_autostart(dry_run=False)
        print()

    print(f"{C.GREEN}=== MATHIR configured for {len(selected)} agent(s) ==={C.RESET}\n")
    print(f"  Restart your agent(s) to use MATHIR.\n")
    print(f"  Each project gets its own database at .mathir/mathir.db\n")
    if needs_manual_instructions:
        print(f"{C.YELLOW}── ACTION REQUIRED for: {', '.join(needs_manual_instructions)} ──{C.RESET}")
        print(f"{C.DIM}  These agents don't auto-load instructions.")
        print(f"  You MUST manually add ~/.config/MATHIR/GLOBAL_INSTRUCTIONS.md")
        print(f"  to your agent's instructions file. See docs/AGENT.md for details.{C.RESET}\n")
    print(f"{C.BOLD}── No Agent? Install OpenCode (free models) ──{C.RESET}")
    print(f"{C.DIM}  EN: Install OpenCode from https://opencode.ai/ — many free models available.")
    print(f"      Give this MATHIR folder to OpenCode and it configures itself.")
    print(f"  FR: Installez OpenCode depuis https://opencode.ai/ — beaucoup de modeles gratuits.")
    print(f"      Donnez ce dossier MATHIR a OpenCode et il se configurera tout seul.")
    print(f"  ES: Instale OpenCode desde https://opencode.ai/ — hay muchos modelos gratuitos.")
    print(f"      De esta carpeta MATHIR a OpenCode y se configurara solo.")
    print(f"  ZH: Install OpenCode https://opencode.ai/ — henduo mofei moxing.")
    print(f"      GEI OpenCode zhege MATHIR danglu jiu hui zidong peizhi.{C.RESET}\n")
    print(f"{C.BOLD}── Troubleshooting ──{C.RESET}")
    print(f"{C.DIM}  EN: If problems occur, give ~/.config/MATHIR/ to your agent. It reads docs/AGENT.md.")
    print(f"  FR: Si problemes, donnez ~/.config/MATHIR/ a votre agent. Il lit docs/AGENT.md.")
    print(f"  ES: Si hay problemas, dea toda la carpeta ~/.config/MATHIR/ a su agente. Lee docs/AGENT.md.")
    print(f"  ZH: Ruguio wenti, ba ~/.config/MATHIR/ gei nide agent. Hui du docs/AGENT.md.{C.RESET}\n")


def installed_only(detected: List[Dict]) -> List[Dict]:
    """Helper for --dry-run: list the agents that would be auto-configured."""
    return [a for a in detected if a["installed"]]


if __name__ == "__main__":
    main()
