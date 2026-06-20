#!/usr/bin/env python3
"""
MATHIR Smart Installer — Auto-detect ALL coding agents, inject config + system prompt.
Works on Windows, Mac, Linux. Supports 40+ coding agents.
"""

import os
import sys
import json
import platform
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
║       4-tier cognitive memory for 40+ coding agents      ║
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


def get_mathir_server_cmd() -> List[str]:
    mathir_dir = Path(__file__).parent
    server_path = mathir_dir / "mathir_lib" / "mathir_mcp_server.py"
    return ["python", str(server_path)]


def get_mathir_server_entry() -> Dict:
    return {
        "type": "local",
        "command": get_mathir_server_cmd(),
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
    config = read_config(config_path)
    entry = get_mathir_server_entry()
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
        # OpenClaw nested format
        if "mcp" not in config:
            config["mcp"] = {}
        if "servers" not in config["mcp"]:
            config["mcp"]["servers"] = {}
        config["mcp"]["servers"]["mathir"] = entry
    elif config_key == "context_servers":
        # Zed format
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
        # Only create if we're in a real project (has .git, package.json, etc.)
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
    # Check if already injected
    if instructions_path.exists():
        with open(instructions_path, "r", encoding="utf-8") as f:
            existing = f.read()
        if "MATHIR" in existing and "4-tier cognitive memory" in existing:
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


def main():
    banner()
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
    print(f"{C.CYAN}[3/3]{C.RESET} Done!\n")
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
    print(f"  ES: Si hay problemas, dea ~/.config/MATHIR/ a su agente. Lee docs/AGENT.md.")
    print(f"  ZH: Ruguio wenti, ba ~/.config/MATHIR/ gei nide agent. Hui du docs/AGENT.md.{C.RESET}\n")


if __name__ == "__main__":
    main()
