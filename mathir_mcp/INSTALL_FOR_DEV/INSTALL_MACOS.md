# MATHIR Install — macOS (launchd)

**Audience:** developers running OpenCode on macOS 10.15+ (Intel or Apple Silicon).
**Time:** ~5 minutes.
**Result:** `mathir_server.py` running on `127.0.0.1:7338`, auto-started at login via a **LaunchAgent** (`~/Library/LaunchAgents/`), and registered as an MCP server in `opencode.json`.

> **Why LaunchAgent, not LaunchDaemon?**
> LaunchAgents run as the logged-in user, have your `$HOME`, your `$PATH`, and your keychain. LaunchDaemons run as root and need to bind to ports < 1024. We want 7338 (unprivileged) and per-user data, so LaunchAgent it is. They start at login and respawn automatically if they die.

---

## 0. Prerequisites

```bash
# Python 3.10+ — every macOS since 10.15 ships 3.8+; 3.11+ is recommended.
# If missing or too old, install via Homebrew:
brew install python@3.12
# or via the official installer: https://python.org/downloads/macos/

# Verify
python3 --version
# expected: Python 3.10.x or newer

# curl — for verification
command -v curl
```

Apple Silicon (M1/M2/M3) note: Homebrew's `python3` lives at `/opt/homebrew/bin/python3`. The plist uses `/usr/bin/python3` (Apple's system Python). If you install via Homebrew, see step 6 for the swap.

---

## 1. Choose the install path

```bash
# Resolved by OpenCode at runtime — same string works on macOS, Linux, WSL.
CONFIG=~/.config/opencode
BIN=$CONFIG/bin
DATA=$CONFIG/data
CONFIGDIR=$CONFIG/config

mkdir -p "$BIN" "$DATA" "$CONFIGDIR"
```

---

## 2. Copy the MATHIR Python package

From the cloned repo:

```bash
REPO=/path/to/mathir_mcp            # adjust to your checkout
cp -r "$REPO/mathir_lib" "$BIN/"
ls "$BIN/mathir_lib/mathir_server.py"   # sanity check
```

---

## 3. Install Python dependencies

```bash
python3 -m pip install --user --upgrade pip
python3 -m pip install --user -r "$BIN/mathir_lib/requirements.txt"
```

`--user` puts deps in `~/Library/Python/3.x/lib/python/site-packages/`. No `sudo` needed.

On Apple Silicon with Homebrew's Python, you may need to point pip at the right interpreter:

```bash
/opt/homebrew/bin/python3 -m pip install --user -r "$BIN/mathir_lib/requirements.txt"
```

Then use `/opt/homebrew/bin/python3` everywhere `python3` appears in the plist (see step 6).

---

## 4. Smoke test — start the daemon in the foreground

```bash
python3 "$BIN/mathir_lib/mathir_server.py"
```

You should see:

```
2026-06-23 12:00:00 [MATHIR-DAEMON] INFO Starting MATHIR daemon...
2026-06-23 12:00:03 [MATHIR-DAEMON] INFO Loaded embedder: paraphrase-multilingual-MiniLM-L12-v2 (dim=384)
2026-06-23 12:00:04 [MATHIR-DAEMON] INFO Listening on 127.0.0.1:7338
```

Leave it running and open a second terminal.

**Verify with curl** (TCP connect — the daemon doesn't speak HTTP, only a tiny JSON-RPC protocol):

```bash
# 1. Port is bound
lsof -nP -iTCP:7338 -sTCP:LISTEN
# expected: a line with python3 in the COMMAND column

# 2. TCP handshake works
curl -v telnet://127.0.0.1:7338 --max-time 2 2>&1 | grep -E "(Connected|Connection refused)"
# expected: Connected to 127.0.0.1 (127.0.0.1) port 7338

# 3. JSON-RPC ping (proves the protocol is live, not just the socket)
printf '{"jsonrpc":"2.0","id":1,"method":"ping","params":{}}' | nc -G1 127.0.0.1 7338
# expected: {"jsonrpc":"2.0","id":1,"result":{"ok":true}}
```

Stop the foreground daemon with `Ctrl+C` before continuing.

---

## 5. Install the LaunchAgent

The plist ships in the repo at `bin/com.mathir.daemon.plist`. It contains a placeholder `/Users/USERNAME/.config/opencode/bin/mathir_server.py` — we need to substitute the real path.

```bash
PLIST_SRC=/path/to/mathir_mcp/bin/com.mathir.daemon.plist
PLIST_DST=~/Library/LaunchAgents/com.mathir.daemon.plist

# Substitute the real home directory and the real python3 path
REAL_HOME="$HOME"
REAL_PY="$(command -v python3 || echo /usr/bin/python3)"
DAEMON_PATH="$REAL_HOME/.config/opencode/bin/mathir_lib/mathir_server.py"

# Sanity check — fail loudly if the daemon isn't there yet
test -f "$DAEMON_PATH" || { echo "[FAIL] Daemon not found at $DAEMON_PATH"; exit 1; }

# Render the plist with sed and install
sed -e "s|/Users/USERNAME|$REAL_HOME|g" \
    -e "s|<string>/usr/bin/python3</string>|<string>$REAL_PY</string>|" \
    "$PLIST_SRC" > "$PLIST_DST"

# Validate the plist
plutil -lint "$PLIST_DST"
# expected: OK

# Load it (starts the daemon right now and registers it for next login)
launchctl load -w "$PLIST_DST"
```

> **Why edit the plist at install time?** The plist format doesn't support `~` or `$HOME` — strings are literal. So we render once with `sed` and write the result. This is the macOS-idiomatic pattern (e.g. Homebrew does the same with its cask plists).

**Verify the agent is loaded:**

```bash
launchctl list | grep mathir
# expected: <pid>  0  com.mathir.daemon

# Live logs (last 50 lines, follow with -f to stream)
log show --predicate 'process == "python3"' --last 1m --style compact
# Or just tail the system log filtered to the daemon:
log stream --predicate 'process == "python3"' --style compact
```

**Verify the port with curl:**

```bash
curl -v telnet://127.0.0.1:7338 --max-time 2 2>&1 | grep -E "(Connected|Connection refused)"
# expected: Connected to 127.0.0.1 (127.0.0.1) port 7338
```

**LaunchAgent management cheat sheet:**

```bash
launchctl list | grep mathir                         # status
launchctl kickstart -k gui/$(id -u)/com.mathir.daemon # restart
launchctl unload ~/Library/LaunchAgents/com.mathir.daemon.plist  # stop
launchctl load   -w ~/Library/LaunchAgents/com.mathir.daemon.plist  # start + persist
rm ~/Library/LaunchAgents/com.mathir.daemon.plist  # uninstall
```

---

## 6. Apple Silicon / Homebrew Python variant

If `python3 --version` reports the Homebrew build (path starts with `/opt/homebrew/`) but the plist still references `/usr/bin/python3`, the agent will fail at launch with `bad interpreter` or import errors.

Fix by re-running step 5 with `REAL_PY` set to the Homebrew path:

```bash
REAL_PY=/opt/homebrew/bin/python3
sed -e "s|/Users/USERNAME|$HOME|g" \
    -e "s|<string>/usr/bin/python3</string>|<string>$REAL_PY</string>|" \
    /path/to/mathir_mcp/bin/com.mathir.daemon.plist \
    > ~/Library/LaunchAgents/com.mathir.daemon.plist
launchctl unload ~/Library/LaunchAgents/com.mathir.daemon.plist 2>/dev/null
launchctl load   -w ~/Library/LaunchAgents/com.mathir.daemon.plist
```

Or use a pyenv/shim — any `python3` on `launchd`'s `PATH` will work; just make sure the path is absolute and resolvable when the user is *not* logged in interactively (launchd has a very minimal env).

**Debug `PATH` issues** by adding this to the plist temporarily:

```xml
<key>EnvironmentVariables</key>
<dict>
  <key>PATH</key>
  <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string>
</dict>
```

---

## 7. Register the MCP server in `opencode.json`

The daemon is running; now teach OpenCode how to talk to it. Edit `~/.config/opencode/opencode.json` and add (or merge) this under the top-level `"mcp"` key:

```jsonc
"mcp": {
  // ...other MCP servers...

  "mathir": {
    "type": "local",
    "command": [
      "python3",
      "~/.config/opencode/bin/mathir_lib/mathir_mcp_server.py"
    ],
    "environment": {
      "MATHIR_EMBEDDING_DIM": "384",
      "MATHIR_PORT": "7338",
      "MATHIR_CONFIG": "~/.config/opencode/config/mathir.json",
      "PYTHONPATH": "~/.config/opencode/bin/mathir_lib"
    },
    "enabled": true
  }
}
```

**Validate the JSON:**

```bash
python3 -c "import json; json.load(open('$HOME/.config/opencode/opencode.json'))" \
  && echo "[OK] opencode.json is valid JSON" \
  || echo "[FAIL] opencode.json has a syntax error"
```

Restart OpenCode so it picks up the new MCP server.

---

## 8. End-to-end verification

```bash
# 1. Daemon socket
curl -s telnet://127.0.0.1:7338 --max-time 2 </dev/null
# expected: "Connected to 127.0.0.1 (127.0.0.1) port 7338"

# 2. JSON-RPC round-trip
printf '{"jsonrpc":"2.0","id":1,"method":"ping","params":{}}' | nc -G1 127.0.0.1 7338
# expected: {"jsonrpc":"2.0","id":1,"result":...}

# 3. LaunchAgent is loaded
launchctl print gui/$(id -u)/com.mathir.daemon 2>/dev/null | head -20
# expected: state = running

# 4. From OpenCode: ask the agent "Recall what you know about my last session."
#    If MCP is wired correctly, you'll see a memory_recall call in the trace.
#    If not, check OpenCode's console for "MCP server mathir failed to start".
```

---

## 9. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `launchctl list` shows nothing for `mathir` | Plist not loaded | Re-run `launchctl load -w ~/Library/LaunchAgents/com.mathir.daemon.plist` |
| `Bad interpreter` or `No such file or directory` in `log show` | `/usr/bin/python3` doesn't exist (newer macOS stripped it) or is too old | Re-render plist with the correct path: see step 6. |
| `ModuleNotFoundError: sentence_transformers` | Daemon can't see `--user` site-packages | Either add the site-packages to `PYTHONPATH` in the plist's `EnvironmentVariables`, or install into a venv and reference the venv's python in the plist. |
| Daemon binds port 7338, then dies | `KeepAlive=true` is respawning a crashing process | `log show --predicate 'process == "python3"' --last 5m` to find the crash reason. |
| Port 7338 already in use | Another process | `lsof -nP -iTCP:7338 -sTCP:LISTEN`, kill it, or set `EnvironmentVariables.MATHIR_PORT=7339` in the plist and `opencode.json`. |
| Embedding model download is slow / fails | Hugging Face blocked, or first run | Pre-download: `huggingface-cli download sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| OpenCode says "MCP server mathir failed to start" | `~` not expanding in the MCP command, or path wrong | Use full absolute path in `command` for debugging, then revert to `~` once it works. |
| macOS firewall prompt keeps appearing | launchd's `python3` is not signed | Either allow in System Settings → Network → Firewall, or sign the binary. For local dev, the simplest fix is `sudo /usr/libexec/ApplicationFirewall/socketfilterfw --add /usr/bin/python3`. |
| Want to nuke and reinstall | — | `launchctl unload ~/Library/LaunchAgents/com.mathir.daemon.plist` → `rm ~/Library/LaunchAgents/com.mathir.daemon.plist` → `rm -rf ~/.config/opencode/bin/mathir_lib` → re-do steps 2-5. |

---

## 10. Unattended / scripted install (CI, provisioning, `dotfiles`)

```bash
#!/usr/bin/env bash
# install_mathir_macos.sh — run as the target user, no sudo needed.
set -euo pipefail
REPO="${MATHIR_REPO:-/opt/mathir_mcp}"
DEST="$HOME/.config/opencode/bin"
PLIST_SRC="$REPO/bin/com.mathir.daemon.plist"
PLIST_DST="$HOME/Library/LaunchAgents/com.mathir.daemon.plist"

# 1. Copy code
mkdir -p "$DEST"
cp -r "$REPO/mathir_lib" "$DEST/"

# 2. Deps
python3 -m pip install --user --quiet -r "$DEST/mathir_lib/requirements.txt"

# 3. LaunchAgent
mkdir -p "$(dirname "$PLIST_DST")"
REAL_HOME="$HOME"
REAL_PY="$(command -v python3 || echo /usr/bin/python3)"
sed -e "s|/Users/USERNAME|$REAL_HOME|g" \
    -e "s|<string>/usr/bin/python3</string>|<string>$REAL_PY</string>|" \
    "$PLIST_SRC" > "$PLIST_DST"
plutil -lint "$PLIST_DST"

# 4. Load
launchctl unload "$PLIST_DST" 2>/dev/null || true
launchctl load   -w "$PLIST_DST"

# 5. Smoke test
sleep 8
if ! lsof -nP -iTCP:7338 -sTCP:LISTEN >/dev/null 2>&1; then
  echo "[FAIL] MATHIR daemon did not bind port 7338" >&2
  log show --predicate 'process == "python3"' --last 1m >&2 || true
  exit 1
fi
echo "[OK] MATHIR daemon running on 127.0.0.1:7338"
```

---

**Next:** see [INSTALL_WINDOWS.md](INSTALL_WINDOWS.md) or [INSTALL_LINUX.md](INSTALL_LINUX.md) for other platforms. The OpenCode MCP config (step 7) is identical across all three.
