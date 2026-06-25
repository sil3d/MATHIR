#!/usr/bin/env bash
# Lit data/model-assignments.json et INSÈRE ou MET À JOUR model: dans chaque agent .md
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENCODE_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_DIR="$OPENCODE_DIR"
DATA_DIR="$OPENCODE_DIR/data"

python3 - "$CONFIG_DIR" "$DATA_DIR" <<'PYEOF'
import json, re, sys
from pathlib import Path

config_dir = Path(sys.argv[1])
data_dir   = Path(sys.argv[2])

cfg = json.loads((data_dir / "model-assignments.json").read_text())
agents_dir = config_dir / "agents"

changed = 0
print("")
print("  OpenCode · Apply Models")
print("  ──────────────────────────────────────────────")
print("")

for name, model in cfg.get("agents", {}).items():
    md = agents_dir / f"{name}.md"
    if not md.exists():
        print(f"  [skip]  @{name}")
        continue
    raw = md.read_text(encoding="utf-8")
    if re.search(r"(?m)^model: ", raw):
        new = re.sub(r"(?m)^model: [^\r\n]+", f"model: {model}", raw)
    else:
        new = re.sub(r"(?s)(^---\s*\n)", rf"\1model: {model}\n", raw, count=1)
    if new != raw:
        md.write_text(new, encoding="utf-8")
        print(f"  [ok]    @{name:<24} → {model.split('/')[-1]}")
        changed += 1
    else:
        print(f"  [=]     @{name:<24}   {model.split('/')[-1]}")

print("")
print("  ──────────────────────────────────────────────")
print(f"  {changed} fichier(s) mis à jour")
print("")
PYEOF
