# MATHIR Install — Linux (systemd)

**Audience:** developers running OpenCode on a modern Linux desktop or server.
**Tested on:** Ubuntu 22.04+, Debian 12+, Fedora 39+, Arch (current), Raspberry Pi OS Bookworm.
**Time:** ~5 minutes.
**Result:** `mathir_daemon.py` running on `127.0.0.1:7338`, auto-started at login via **user systemd** (`~/.config/systemd/user/`), and registered as an MCP server in `opencode.json`.

> **Why user systemd, not system systemd?**
> The daemon holds a per-user database (`~/.config/opencode/data/.mathir/`). A user unit runs as you, has your `$HOME`, doesn't need `sudo`, and starts at *your* login (not at boot). On a headless server, just `loginctl enable-linger <user>` to make user services survive logout.

---

## 0. Prerequisites

```bash
# Python 3.10 or newer (3.11 / 3.12 recommended for sqlite-vec wheels)
python3 --version
# systemd -- should be on every modern distro
systemctl --version | head -1
# curl -- for the verification step
command -v curl
```

If Python is missing:

```bash
# Debian / Ubuntu
sudo apt install -y python3 python3-pip python3-venv

# Fedora
sudo dnf install -y python3 python3-pip

# Arch
sudo pacman -S --needed python python-pip
```

---

## 1. Choose the install path

```bash
# Resolved by OpenCode at runtime — same string works on every Unix.
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
ls "$BIN/mathir_lib/mathir_daemon.py"   # sanity check
```

> If you only need the daemon (not the MCP server, dashboard, or CLI), you can prune the copy — but the full `mathir_lib/` is only ~3 MB, so just take it all.

---

## 3. Install Python dependencies

```bash
python3 -m pip install --user --upgrade pip
python3 -m pip install --user -r "$BIN/mathir_lib/requirements.txt"
```

> Use `--user` so deps live in `~/.local/lib/`, not in the system site-packages. The systemd unit picks them up automatically.
> On Debian/Ubuntu you may need `sudo apt install python3-venv build-essential` first because `torch` and `sentence-transformers` ship C extensions.

**GPU acceleration (optional):**

```bash
# CUDA 12.x
python3 -m pip install --user torch --index-url https://download.pytorch.org/whl/cu121
```

See `docs/GPU_SETUP.md` for the full ONNX/CUDA matrix.

---

## 4. Smoke test — start the daemon in the foreground

```bash
python3 "$BIN/mathir_lib/mathir_daemon.py"
```

You should see logs like:

```
2026-06-23 12:00:00 [MATHIR-DAEMON] INFO Starting MATHIR daemon...
2026-06-23 12:00:03 [MATHIR-DAEMON] INFO Loaded embedder: paraphrase-multilingual-MiniLM-L12-v2 (dim=384)
2026-06-23 12:00:04 [MATHIR-DAEMON] INFO Listening on 127.0.0.1:7338
```

Leave it running and open a second terminal.

**Verify with curl** (TCP connect — the daemon doesn't speak HTTP, only a tiny JSON-RPC protocol):

```bash
# 1. Port is bound
ss -tlnp 2>/dev/null | grep 7338 || netstat -tlnp 2>/dev/null | grep 7338
# expected: a line with python3 in the process column

# 2. TCP handshake works
exec 3<>/dev/tcp/127.0.0.1/7338 && echo "TCP OK" && exec 3<&- 3>&-
# or with curl -- telnet:// style probe
curl -v telnet://127.0.0.1:7338 --max-time 2 2>&1 | grep -E "(Connected|Failed)"
# expected: "Connected to 127.0.0.1 (127.0.0.1) port 7338"

# 3. JSON-RPC ping (proves the protocol is live, not just the socket)
printf '{"jsonrpc":"2.0","id":1,"method":"ping","params":{}}' | nc -q1 127.0.0.1 7338
# expected: {"jsonrpc":"2.0","id":1,"result":{"ok":true}}   (or similar)
```

If port 7338 is taken:

```bash
sudo lsof -i :7338          # or: ss -tlnp 'sport = :7338'
```

Kill the conflicting PID, or set `MATHIR_PORT=7339` in the systemd unit *and* in `opencode.json`.

Stop the foreground daemon with `Ctrl+C` before moving on.

---

## 5. Install the systemd user unit

The unit file ships in the repo at `bin/mathir-daemon.service`. Its key trick is `%h` — systemd expands this to the running user's home directory at service start, so the file is fully portable across users and machines.

```bash
UNIT_SRC=/path/to/mathir_mcp/bin/mathir-daemon.service
UNIT_DST=~/.config/systemd/user/mathir-daemon.service
mkdir -p ~/.config/systemd/user
cp "$UNIT_SRC" "$UNIT_DST"

# Reload, enable (start at every login), and start (right now)
systemctl --user daemon-reload
systemctl --user enable mathir-daemon.service
systemctl --user start  mathir-daemon.service
```

**Headless / server note:** user services die when you log out, unless you opt in to lingering:

```bash
# Run as the target user, NOT as root
loginctl enable-linger "$USER"
# Verify
loginctl show-user "$USER" | grep Linger
# expected: Linger=yes
```

**Verify the unit is running:**

```bash
systemctl --user status mathir-daemon.service
# expected: Active: active (running) since ...

# Recent logs (last 30 lines, follow with -f)
journalctl --user -u mathir-daemon.service -n 30 --no-pager
```

**Verify the port from curl:**

```bash
curl -v telnet://127.0.0.1:7338 --max-time 2 2>&1 | grep -E "(Connected|connection refused)"
# expected: Connected to 127.0.0.1 (127.0.0.1) port 7338
```

**Service management cheat sheet:**

```bash
systemctl --user status  mathir-daemon      # is it up?
systemctl --user restart mathir-daemon      # bounce
systemctl --user stop    mathir-daemon      # stop
systemctl --user disable mathir-daemon      # don't start at next login
systemctl --user reset-failed mathir-daemon # clear crash counter
```

---

## 6. Register the MCP server in `opencode.json`

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

**Validate the JSON** (works in any shell — no Python deps needed):

```bash
python3 -c "import json; json.load(open('$HOME/.config/opencode/opencode.json'))" \
  && echo "[OK] opencode.json is valid JSON" \
  || echo "[FAIL] opencode.json has a syntax error"
```

Restart OpenCode so it picks up the new MCP server.

---

## 7. End-to-end verification

```bash
# 1. Daemon socket
curl -s telnet://127.0.0.1:7338 --max-time 2 </dev/null
# expected: "Connected to 127.0.0.1 (127.0.0.1) port 7338"

# 2. JSON-RPC round-trip
printf '{"jsonrpc":"2.0","id":1,"method":"ping","params":{}}' \
  | nc -q1 127.0.0.1 7338
# expected: {"jsonrpc":"2.0","id":1,"result":...}

# 3. systemd unit is healthy
systemctl --user is-active mathir-daemon
# expected: active

# 4. From OpenCode: ask the agent "Recall what you know about my last session."
#    If MCP is wired correctly, you'll see a memory_recall call in the trace.
#    If not, check `opencode` console output for "MCP server mathir failed to start".
```

---

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `systemctl --user status` → `inactive (dead)` | Unit loaded but never started, or crashed | `journalctl --user -u mathir-daemon -n 50` for the reason. Common: missing deps, wrong path, `python3` not on `$PATH` for systemd (it uses a minimal env). |
| Unit runs in foreground test but fails under systemd | `PATH` is empty under systemd | Edit the unit: add `Environment=PATH=%h/.local/bin:/usr/local/bin:/usr/bin:/bin` |
| `Address already in use` | Port 7338 occupied | `sudo ss -tlnp 'sport = :7338'`, kill the PID, or set `Environment=MATHIR_PORT=7339` in the unit + `opencode.json` |
| `ModuleNotFoundError: sentence_transformers` | Deps installed system-wide, unit can't see them | Reinstall with `python3 -m pip install --user ...`, or use a venv and set `Environment=PATH=%h/.venvs/mathir/bin:%h/.local/bin` |
| Embedding model download is slow / fails | No internet, or Hugging Face is blocked | Pre-download with `huggingface-cli download sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| Headless server: daemon dies on logout | No lingering | `loginctl enable-linger $USER` |
| `nc` not installed | Minimal container | `printf ... | socat - TCP:127.0.0.1:7338` (socat works everywhere) or use `python3 -c "import socket; ..."` |
| Want to nuke and reinstall | — | `systemctl --user disable --now mathir-daemon` → `rm -rf ~/.config/opencode/bin/mathir_lib` → re-do steps 2-5. |

---

## 9. Unattended / scripted install (CI, provisioning, Ansible)

```bash
#!/usr/bin/env bash
# install_mathir_linux.sh — run as the target user, no sudo needed.
set -euo pipefail
REPO="${MATHIR_REPO:-/opt/mathir_mcp}"
DEST="$HOME/.config/opencode/bin"

# 1. Copy code
mkdir -p "$DEST"
cp -r "$REPO/mathir_lib" "$DEST/"

# 2. Deps
python3 -m pip install --user --quiet -r "$DEST/mathir_lib/requirements.txt"

# 3. Systemd user unit
mkdir -p "$HOME/.config/systemd/user"
cp "$REPO/bin/mathir-daemon.service" "$HOME/.config/systemd/user/"
systemctl --user daemon-reload
systemctl --user enable --now mathir-daemon.service

# 4. Smoke test
sleep 8
if ! ss -tln 2>/dev/null | grep -q ':7338 '; then
  echo "[FAIL] MATHIR daemon did not bind port 7338" >&2
  journalctl --user -u mathir-daemon -n 20 >&2
  exit 1
fi
echo "[OK] MATHIR daemon running on 127.0.0.1:7338"
```

For headless servers, add `sudo loginctl enable-linger "$USER"` (one-time).

---

**Next:** see [INSTALL_WINDOWS.md](INSTALL_WINDOWS.md) or [INSTALL_MACOS.md](INSTALL_MACOS.md) for other platforms. The OpenCode MCP config (step 6) is identical across all three.
