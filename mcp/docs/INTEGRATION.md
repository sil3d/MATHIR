# Integration Guide

**Universal: install once at `~/.config/MATHIR/`, works with 50 agents.**

---

## Quick Start (All Platforms)

```bash
# 1. Clone
git clone https://github.com/sil3d/MATHIR.git /tmp/MATHIR
cp -r /tmp/MATHIR/mcp ~/.config/MATHIR

# 2. Install deps
cd ~/.config/MATHIR
pip install -r mathir_lib/requirements.txt

# 3. Run smart installer
python install_smart.py
# Select your agent(s) from the menu
```

---

## Per-Agent Integration

### OpenCode

**Config:** `~/.config/opencode/opencode.json`

```json
{
  "mcp": {
    "mathir": {
      "type": "local",
      "command": ["python", "~/.config/MATHIR/mathir_lib/mathir_mcp_server.py"],
      "environment": { "MATHIR_EMBEDDING_DIM": "384" },
      "enabled": true
    }
  }
}
```

**Instructions:** Copy `GLOBAL_INSTRUCTIONS.md` into `~/.config/opencode/GLOBAL_INSTRUCTIONS.md`

---

### MiMo

**Config:** `~/.config/mimocode/mimocode.json`

```json
{
  "mcpServers": {
    "mathir": {
      "command": "python",
      "args": ["~/.config/MATHIR/mathir_lib/mathir_mcp_server.py"],
      "env": { "MATHIR_EMBEDDING_DIM": "384" }
    }
  }
}
```

**Instructions:** Copy `GLOBAL_INSTRUCTIONS.md` into `~/.config/mimocode/MEMORY.md`

---

### Claude Code

**Config:** `~/.claude.json`

```json
{
  "mcpServers": {
    "mathir": {
      "command": "python",
      "args": ["~/.config/MATHIR/mathir_lib/mathir_mcp_server.py"],
      "env": { "MATHIR_EMBEDDING_DIM": "384" }
    }
  }
}
```

**Instructions:** Copy `GLOBAL_INSTRUCTIONS.md` into `~/.claude/CLAUDE.md`

---

### Claude Desktop

**Config:** `~/Library/Application Support/Claude/claude_desktop_config.json` (Mac)
or `%APPDATA%\Claude\claude_desktop_config.json` (Windows)

```json
{
  "mcpServers": {
    "mathir": {
      "command": "python",
      "args": ["~/.config/MATHIR/mathir_lib/mathir_mcp_server.py"],
      "env": { "MATHIR_EMBEDDING_DIM": "384" }
    }
  }
}
```

---

### Cursor

**Config:** `~/.cursor/mcp.json`

```json
{
  "mcpServers": {
    "mathir": {
      "command": "python",
      "args": ["~/.config/MATHIR/mathir_lib/mathir_mcp_server.py"],
      "env": { "MATHIR_EMBEDDING_DIM": "384" }
    }
  }
}
```

**Instructions:** Copy `GLOBAL_INSTRUCTIONS.md` into `~/.cursor/rules/mathir.md`

---

### Kilo Code

**CLI config:** `~/.config/kilo/kilo.json` (uses `"mcp"` key)
**VS Code extension:** `.kilocode/mcp.json` (uses `"mcpServers"` key)

```json
{
  "mcpServers": {
    "mathir": {
      "command": "python",
      "args": ["~/.config/MATHIR/mathir_lib/mathir_mcp_server.py"],
      "env": { "MATHIR_EMBEDDING_DIM": "384" }
    }
  }
}
```

---

### Cline

**Config:** `AppData\Roaming\Code\User\globalStorage\saoudrizwan.claude-dev\settings\cline_mcp_settings.json`

```json
{
  "mcpServers": {
    "mathir": {
      "command": "python",
      "args": ["~/.config/MATHIR/mathir_lib/mathir_mcp_server.py"],
      "env": { "MATHIR_EMBEDDING_DIM": "384" }
    }
  }
}
```

---

### Windsurf

**Config:** `~/.codeium/windsurf/mcp_config.json`

```json
{
  "mcpServers": {
    "mathir": {
      "command": "python",
      "args": ["~/.config/MATHIR/mathir_lib/mathir_mcp_server.py"],
      "env": { "MATHIR_EMBEDDING_DIM": "384" }
    }
  }
}
```

---

### Gemini CLI

**Config:** `~/.gemini/settings.json`

```json
{
  "mcpServers": {
    "mathir": {
      "command": "python",
      "args": ["~/.config/MATHIR/mathir_lib/mathir_mcp_server.py"],
      "env": { "MATHIR_EMBEDDING_DIM": "384" }
    }
  }
}
```

---

### Zcode

**Config:** `~/.zcode/v2/config.json`

```json
{
  "mcpServers": {
    "mathir": {
      "command": "python",
      "args": ["~/.config/MATHIR/mathir_lib/mathir_mcp_server.py"],
      "env": { "MATHIR_EMBEDDING_DIM": "384" }
    }
  }
}
```

---

## Manual Integration (Any Agent)

If your agent isn't listed:

1. Find its MCP config file
2. Add the appropriate JSON block:
   - `"mcpServers"` key: Claude Code, Cursor, Cline, Windsurf, Gemini, Zcode, etc.
   - `"mcp"` key: OpenCode, Kilo Code CLI
3. Use `command: ["python", "~/.config/MATHIR/mathir_lib/mathir_mcp_server.py"]`
4. Add `"env": {"MATHIR_EMBEDDING_DIM": "384"}`
5. Copy `GLOBAL_INSTRUCTIONS.md` into your agent's instructions

**If you have no agent, install OpenCode:** https://opencode.ai/

---

## Troubleshooting Integration

| Problem | Fix |
|---------|-----|
| MCP server not showing | Check config key: some agents use `mcp`, others `mcpServers` |
| "python not found" | Use full path: `["C:\Python312\python.exe", "..."]` |
| Wrong script path | Use `~/.config/MATHIR/mathir_lib/mathir_mcp_server.py` |
| Agent ignores memory | Ensure `GLOBAL_INSTRUCTIONS.md` is in instructions |

---

## Multilingual Help (EN/FR/ES/ZH)

**EN:** Give `~/.config/MATHIR/` to your coding agent. It reads `docs/AGENT.md` and configures automatically.

**FR:** Donnez `~/.config/MATHIR/` a votre agent de code. Il lit `docs/AGENT.md` et se configure automatiquement.

**ES:** Dea `~/.config/MATHIR/` a su agente de codigo. Leera `docs/AGENT.md` y se configurara solo.

**ZH:** GEI nide agent `~/.config/MATHIR/`, hui du `docs/AGENT.md` zidong peizhi.
