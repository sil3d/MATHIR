#!/usr/bin/env bash
# Lance le serveur OpenCode Config sur port 7337 si pas déjà actif.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENCODE_DIR="$(dirname "$SCRIPT_DIR")"
PORT=7337
SERVER_SCRIPT="$OPENCODE_DIR/lib/config-server.py"

# Déjà actif ?
if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then
	echo "  [OK] Config server déjà actif sur port $PORT"
	exit 0
fi

# Vérification Python
if ! command -v python3 >/dev/null 2>&1; then
	echo "  [ERROR] python3 introuvable dans PATH"
	exit 1
fi
if [ ! -f "$SERVER_SCRIPT" ]; then
	echo "  [ERROR] config-server.py introuvable : $SERVER_SCRIPT"
	exit 1
fi

# Démarrage en arrière-plan
nohup python3 "$SERVER_SCRIPT" >/dev/null 2>&1 &
sleep 1.5

if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then
	echo "  [OK] Config server démarré sur http://localhost:$PORT"
else
	echo "  [ERROR] Le serveur n'a pas répondu. Lance manuellement :"
	echo "         python3 \"$SERVER_SCRIPT\""
	exit 1
fi
