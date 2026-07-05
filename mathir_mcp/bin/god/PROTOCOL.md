# GOD MODE PROTOCOL — Mycerise V2 Tauri

> Communication contract between **Orchestrator** (opencode-glm52) and **Workers** (mimo-code, others).
> Version: 1.0.0 (2026-07-05)
> Author: opencode-glm52 (god-orchestrator)

---

## 1. Label Taxonomy (MATHIR labels)

Toutes les communications passent par MATHIR avec ces labels stricts :

| Label | Direction | Priority | Block type | Rôle |
|-------|-----------|----------|------------|------|
| `god:task:{8hex}:{worker}:pending` | Orch→Worker | 9 | working_memory | Dispatch d'une tâche à un worker |
| `god:task:{8hex}:{worker}:in-progress` | Worker→Orch | 9 | working_memory | Worker a commencé (heartbeat) |
| `god:result:{8hex}:{worker}:completed` | Worker→Orch | 9 | episodic | Worker a fini, output livré |
| `god:result:{8hex}:{worker}:failed` | Worker→Orch | 9 | episodic | Worker a échoué (rare) |
| `god:reply:{8hex}:{worker}:feedback` | Orch→Worker | 9 | working_memory | Feedback orchestrator vers worker |
| `god:reg:{worker}:{worker}:{status}` | Worker→Orch | 3 | immunological | Registry heartbeat (idle/busy) |
| `god:shutdown:00000000:{worker}` | Orch→Worker | 9 | working_memory | Arrêt du worker |

**Statuts workers** : `idle` · `busy` · `done` · `error`

---

## 2. Endpoints MATHIR natifs (déjà implémentés)

- `POST /api/god/poll` — workers polllent pour leur prochaine tâche
  - Body: `{"agent": "mimo-code", "status": "pending"}`
  - Retourne: `{"task": {"memory_id": "...", "label": "...", "content": "...", "priority": 9}} | {"task": null}`
- `GET /api/god/agents` — orchestrator liste workers enregistrés

---

## 3. Flux typique

```
ORCHESTRATOR                          MATHIR DAEMON                       WORKER
    |                                      |                                 |
    |--- god:task:abc12345:mimo-code:pending ---->                          |
    |                                      |--- polling --->                 |
    |                                      |<-- task found --->              |
    |                                      |--- dispatch -->                |
    |                                      |                                 |--- execute -->
    |                                      |                                 |
    |                                      |<-- god:result:abc12345:mimo-code:completed ---|
    |<-- notify -------|                                                  |
    |--- verify, decide ----->                                          |
    |--- god:reply:abc12345:mimo-code:feedback (if rework) ---->        |
```

---

## 4. Paths and Config (cross-platform)

All paths **MUST** be resolved via env vars or POSIX conventions — **NEVER hardcode** machine-specific paths.

| Var | Default | Purpose |
|-----|---------|---------|
| `MATHIR_DAEMON_URL` | `http://localhost:7338` | Daemon endpoint |
| `XDG_CONFIG_HOME` | `$HOME/.config` | POSIX config base (Linux/Mac) |
| `MYCERISE_STATE_DIR` | `$XDG_CONFIG_HOME/mycerise` | Where bridge writes state + logs |

On **Windows** the defaults resolve to:
- `C:\Users\<USER>\.config\mycerise\` (if using Git-Bash/MSYS style)
- Or `%APPDATA%\mycerise\` if you symlink — set `MYCERISE_STATE_DIR` env var

On **Linux/Mac**:
- `~/.config/mycerise/`

The bridge auto-detects. **NEVER** hardcode `/Users/...`, `C:\Users\...`, `D:\...`, or any machine path.

---

## 5. Bridge Tools

| Outil | Plateforme | Usage |
|-------|------------|-------|
| `god_bridge.py` | Win + Linux/Mac | Daemon Python qui polle et notifie (log + beep + desktop) |
| `god_poll.ps1`  | Windows    | Polling léger one-shot via PowerShell |
| `god_poll.sh`   | Linux/Mac  | Polling léger one-shot via bash |

**Démarrage recommandé par worker terminal** :
```bash
python scripts/god-mode/god_bridge.py --mode worker --name mimo-code --interval 5
```

**Démarrage recommandé orchestrator terminal** :
```bash
python scripts/god-mode/god_bridge.py --mode orchestrator --interval 5 --project Mycerise_V2_Taur
```

---

## 6. Anti-patterns

- ❌ Ne JAMAIS hardcoder un chemin machine-spécifique (`D:\...`, `C:\Users\...`, `/Users/foo/...`) → utiliser env vars ou `Path.home()`
- ❌ Ne JAMAIS utiliser `memory_recall(query="god:result")` pour vérifier un statut → utiliser `/api/memory/audit?since={ts}` ou `/api/memories?label={prefix}`
- ❌ Ne PAS envoyer de message sans label préfixé `god:` → invisible pour le bridge
- ❌ Ne PAS bloquer le terminal worker en attendant → laisser le bridge notifier
- ✅ TOUJOURS répondre aux `god:reply:*` dans les 30s
- ✅ TOUJOURS émettre un `god:result:*:completed` même si l'output est `null` (timeout/crash)

---

## 7. Exemple complet

```python
# ORCHESTRATOR dispatche
memory_save(
    content=json.dumps({"task": "Fix bug X", "files": [...]}, ensure_ascii=False),
    agent="mimo-code",
    block_type="working_memory",
    label="god:task:7f3a9b2c:mimo-code:pending",
    priority=9
)

# WORKER poll
r = POST("/api/god/poll", {"agent": "mimo-code", "status": "pending"})
task = r.json()["task"]
# ... execute ...

# WORKER report
memory_save(
    content=json.dumps({"status": "done", "files_modified": [...], "tests": "pass"}),
    agent="mimo-code",
    block_type="episodic",
    label="god:result:7f3a9b2c:mimo-code:completed",
    priority=9
)

# ORCHESTRATOR lit (works cross-platform, no hardcoded paths)
r = requests.get(f"{os.environ['MATHIR_DAEMON_URL']}/api/memories", params={"project":"Mycerise_V2_Taur","limit":50})
god_results = [m for m in r.json()["memories"] if m["metadata"]["label"].startswith("god:result:")]
```
