#!/bin/bash
# MATHIR Root Installer (Linux/Mac)
# Convenience proxy — runs the smart installer from mathir_mcp/bin/

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALLER="$SCRIPT_DIR/mathir_mcp/bin/install.sh"

if [ ! -f "$INSTALLER" ]; then
    echo "❌ Error: MATHIR installer not found at $INSTALLER"
    echo "   Make sure you cloned the full repo (with subdirs)."
    exit 1
fi

echo "→ Delegating to mathir_mcp/bin/install.sh"
echo ""
exec "$INSTALLER" "$@"