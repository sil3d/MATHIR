#!/usr/bin/env python3
"""
MATHIR Neural Memory Dashboard — Portable Standalone Server
Reads from MATHIR SQLite DB and serves JSON API + HTML dashboard.
No hardcoded paths — auto-detects DB locations or uses env vars.

Usage:
    python dashboard_server.py              # auto-detect DB, port 7420
    MATHIR_DB=./my.db python dashboard_server.py  # explicit DB
    MATHIR_STATS_PORT=8080 python dashboard_server.py  # custom port
"""

import json
import os
import sqlite3
import sys
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs

# ---------------------------------------------------------------------------
# Configuration — all overridable via env vars, zero hardcoded paths
# ---------------------------------------------------------------------------

PORT = int(os.environ.get("MATHIR_STATS_PORT", "7420"))
DB_PATH_OVERRIDE = os.environ.get("MATHIR_DB", "")
CONFIG_PATH_OVERRIDE = os.environ.get("MATHIR_CONFIG", "")
HTML_PATH = Path(__file__).parent / "dashboard.html"


# ---------------------------------------------------------------------------
# DB auto-detection
# ---------------------------------------------------------------------------

def _candidate_dirs() -> list[Path]:
    """Ordered list of directories to scan for .mathir/mathir.db."""
    home = Path.home()
    cwd = Path.cwd()
    candidates = [
        cwd,
        cwd.parent,
        home,
    ]
    for subdir in ("Documents", "Desktop", "Projects", "dev", "Code"):
        d = home / subdir
        if d.exists():
            candidates.append(d)
    return candidates


def _find_mathir_dbs() -> list[dict]:
    """Walk candidate directories and discover .mathir/mathir.db files."""
    seen: set[str] = set()
    results: list[dict] = []

    for parent in _candidate_dirs():
        if not parent.exists():
            continue
        try:
            for item in parent.iterdir():
                if not item.is_dir():
                    continue
                db = item / ".mathir" / "mathir.db"
                if db.exists() and str(db) not in seen:
                    seen.add(str(db))
                    results.append({
                        "name": item.name,
                        "path": str(db),
                        "project_dir": str(item),
                        "size_bytes": db.stat().st_size,
                    })
        except PermissionError:
            continue

    return results


def _resolve_db(project: str | None = None) -> tuple[sqlite3.Connection | None, str | None]:
    """
    Return (connection, db_path_str) or (None, None) if nothing found.
    If *project* is given, try to match by name first.
    """
    # 1. Explicit env var takes priority
    if DB_PATH_OVERRIDE:
        p = Path(DB_PATH_OVERRIDE)
        if p.exists():
            conn = sqlite3.connect(str(p))
            conn.row_factory = sqlite3.Row
            return conn, str(p)
        print(f"WARNING: MATHIR_DB={DB_PATH_OVERRIDE} does not exist", file=sys.stderr)

    # 2. Scan for databases
    dbs = _find_mathir_dbs()

    if project:
        for db in dbs:
            if db["name"] == project:
                conn = sqlite3.connect(db["path"])
                conn.row_factory = sqlite3.Row
                return conn, db["path"]

    # 3. Return first found
    if dbs:
        conn = sqlite3.connect(dbs[0]["path"])
        conn.row_factory = sqlite3.Row
        return conn, dbs[0]["path"]

    return None, None


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    """Load config from .mathir/config.json or config/mathir.json."""
    if CONFIG_PATH_OVERRIDE:
        p = Path(CONFIG_PATH_OVERRIDE)
        if p.exists():
            return json.loads(p.read_text())

    cwd = Path.cwd()
    for rel in (".mathir/config.json", "config/mathir.json"):
        p = cwd / rel
        if p.exists():
            return json.loads(p.read_text())

    # Walk up looking for .mathir/config.json
    for parent in [cwd, *cwd.parents]:
        p = parent / ".mathir" / "config.json"
        if p.exists():
            return json.loads(p.read_text())

    return {}


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _count_by_tier(conn: sqlite3.Connection) -> dict:
    tiers = {"working": 0, "episodic": 0, "semantic": 0, "procedural": 0, "unknown": 0}
    agents: dict[str, int] = {}
    total = 0
    try:
        rows = conn.execute("SELECT metadata FROM memories WHERE metadata IS NOT NULL").fetchall()
    except Exception:
        return tiers, agents, 0

    for row in rows:
        try:
            meta = json.loads(row["metadata"])
            bt = meta.get("block_type", "unknown")
            tier = "working" if bt == "working_memory" else bt
            if tier not in tiers:
                tier = "unknown"
            tiers[tier] += 1
            agent = meta.get("agent", "unknown")
            agents[agent] = agents.get(agent, 0) + 1
            total += 1
        except Exception:
            tiers["unknown"] += 1
            total += 1

    return tiers, agents, total


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

def api_overview(project: str | None = None) -> dict:
    conn, db_path = _resolve_db(project)
    if conn is None:
        return {"error": "No database found", "project": project}

    config = _load_config().get("memory", {})
    tiers, agents, total = _count_by_tier(conn)
    db_size = Path(db_path).stat().st_size if db_path else 0
    conn.close()

    return {
        "total_memories": total,
        "tiers": tiers,
        "agents": agents,
        "db_size_bytes": db_size,
        "db_size_mb": round(db_size / (1024 * 1024), 2),
        "db_path": db_path,
        "config": {
            "embedding_dim": config.get("embedding_dim", 384),
            "working_capacity": config.get("working_capacity", 64),
            "episodic_capacity": config.get("episodic_capacity", 1000),
            "semantic_prototypes": config.get("semantic_prototypes", 256),
            "immunological_capacity": config.get("immunological_capacity", 100),
            "kl_coefficient": config.get("kl_coefficient", 0.01),
            "anomaly_threshold": config.get("anomaly_threshold", 2.0),
            "decay_rate": config.get("decay_rate", 0.95),
        },
        "project": project,
        "timestamp": datetime.now().isoformat(),
    }


def api_memories(limit=100, offset=0, agent=None, block_type=None, project=None):
    conn, _ = _resolve_db(project)
    if conn is None:
        return {"error": "No database found", "project": project}

    query = "SELECT memory_id, modality_text, metadata, tier, timestamp, provider, model FROM memories WHERE 1=1"
    params: list = []
    if agent:
        query += " AND json_extract(metadata, '$.agent') = ?"
        params.append(agent)
    if block_type:
        query += " AND json_extract(metadata, '$.block_type') = ?"
        params.append(block_type)

    count_query = "SELECT COUNT(*) as cnt FROM memories WHERE 1=1"
    count_params: list = list(params)
    if agent:
        count_query += " AND json_extract(metadata, '$.agent') = ?"
    if block_type:
        count_query += " AND json_extract(metadata, '$.block_type') = ?"

    query += " ORDER BY rowid DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    try:
        rows = conn.execute(query, params).fetchall()
        total = conn.execute(count_query, count_params).fetchone()["cnt"]
    except Exception:
        rows, total = [], 0

    memories = []
    for row in rows:
        r = dict(row)
        try:
            r["metadata"] = json.loads(r["metadata"]) if r["metadata"] else {}
        except Exception:
            r["metadata"] = {}
        memories.append(r)

    conn.close()
    return {"memories": memories, "total": total, "limit": limit, "offset": offset, "project": project}


def api_tier_details(project: str | None = None) -> dict:
    conn, _ = _resolve_db(project)
    if conn is None:
        return {"error": "No database found", "project": project}

    try:
        rows = conn.execute("SELECT metadata FROM memories WHERE metadata IS NOT NULL").fetchall()
    except Exception:
        rows = []

    tiers: dict = {}
    for row in rows:
        try:
            meta = json.loads(row["metadata"])
            bt = meta.get("block_type", "unknown")
            tier = "working" if bt == "working_memory" else bt
            if tier not in tiers:
                tiers[tier] = {"count": 0, "agents": {}, "labels": []}
            tiers[tier]["count"] += 1
            agent = meta.get("agent", "unknown")
            tiers[tier]["agents"][agent] = tiers[tier]["agents"].get(agent, 0) + 1
            label = meta.get("label", "")
            if label:
                tiers[tier]["labels"].append(label)
        except Exception:
            pass

    config = _load_config().get("memory", {})
    capacities = {
        "working": config.get("working_capacity", 64),
        "episodic": config.get("episodic_capacity", 1000),
        "semantic": config.get("semantic_prototypes", 256),
        "procedural": config.get("semantic_prototypes", 256),
    }

    result = {}
    for tier, data in tiers.items():
        cap = capacities.get(tier, 0)
        result[tier] = {
            **data,
            "capacity": cap,
            "usage_pct": round(data["count"] / cap * 100, 1) if cap > 0 else 0,
        }

    conn.close()
    return result


def api_router_weights(project: str | None = None) -> dict:
    tiers = api_tier_details(project)
    if "error" in tiers:
        return {"error": tiers["error"], "project": project}

    total = sum(t["count"] for t in tiers.values()) or 1
    weights = {tier: round(data["count"] / total, 3) for tier, data in tiers.items()}
    config = _load_config()

    return {
        "weights": weights,
        "total_memories": total,
        "router_type": config.get("router", {}).get("type", "kl_constrained"),
        "kl_coefficient": config.get("router", {}).get("kl_coefficient", 0.01),
        "project": project,
    }


def api_agents(project: str | None = None) -> dict:
    conn, _ = _resolve_db(project)
    if conn is None:
        return {"error": "No database found", "project": project}

    try:
        rows = conn.execute("SELECT metadata FROM memories WHERE metadata IS NOT NULL").fetchall()
    except Exception:
        rows = []

    agents: dict = {}
    for row in rows:
        try:
            meta = json.loads(row["metadata"])
            agent = meta.get("agent", "unknown")
            bt = meta.get("block_type", "unknown")
            tier = "working" if bt == "working_memory" else bt
            if agent not in agents:
                agents[agent] = {"total": 0, "tiers": {}}
            agents[agent]["total"] += 1
            agents[agent]["tiers"][tier] = agents[agent]["tiers"].get(tier, 0) + 1
        except Exception:
            pass

    conn.close()
    return {"agents": agents, "project": project}


def api_timeline(project: str | None = None) -> dict:
    conn, _ = _resolve_db(project)
    if conn is None:
        return {"error": "No database found", "project": project}

    try:
        rows = conn.execute(
            "SELECT metadata, timestamp FROM memories WHERE timestamp IS NOT NULL ORDER BY timestamp"
        ).fetchall()
    except Exception:
        rows = []

    buckets: dict = {}
    for row in rows:
        try:
            ts = row["timestamp"]
            if ts:
                dt = datetime.fromtimestamp(ts)
                key = dt.strftime("%Y-%m-%d %H:00")
                meta = json.loads(row["metadata"]) if row["metadata"] else {}
                tier = meta.get("block_type", "unknown")
                if key not in buckets:
                    buckets[key] = {"total": 0, "tiers": {}}
                buckets[key]["total"] += 1
                buckets[key]["tiers"][tier] = buckets[key]["tiers"].get(tier, 0) + 1
        except Exception:
            pass

    timeline = [{"time": k, **v} for k, v in sorted(buckets.items())]
    conn.close()
    return {"timeline": timeline, "project": project}


def api_list_projects() -> list[dict]:
    """Return all discovered MATHIR databases."""
    dbs = _find_mathir_dbs()
    projects = []
    for db in dbs:
        projects.append({
            "name": db["name"],
            "path": db["path"],
            "project_dir": db["project_dir"],
            "size_bytes": db["size_bytes"],
        })
    return projects


def api_delete_memory(memory_id: str, reason: str = "user requested", project: str | None = None):
    conn, _ = _resolve_db(project)
    if conn is None:
        return {"error": "No database found", "project": project}

    try:
        cursor = conn.execute("DELETE FROM memories WHERE memory_id = ?", (memory_id,))
        deleted = cursor.rowcount > 0
        conn.commit()
    except Exception:
        deleted = False
    conn.close()
    return {"memory_id": memory_id, "deleted": deleted, "reason": reason, "project": project}


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class DashboardHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)
        project = params.get("project", [None])[0]

        if path in ("/", "/dashboard"):
            self._serve_html()
        elif path == "/api/projects":
            self._json_response({"projects": api_list_projects()})
        elif path == "/api/overview":
            self._json_response(api_overview(project))
        elif path == "/api/memories":
            limit = int(params.get("limit", [100])[0])
            offset = int(params.get("offset", [0])[0])
            agent = params.get("agent", [None])[0]
            block_type = params.get("block_type", [None])[0]
            self._json_response(api_memories(limit, offset, agent, block_type, project))
        elif path == "/api/tiers":
            self._json_response(api_tier_details(project))
        elif path == "/api/router":
            self._json_response(api_router_weights(project))
        elif path == "/api/agents":
            self._json_response(api_agents(project))
        elif path == "/api/timeline":
            self._json_response(api_timeline(project))
        else:
            self.send_error(404)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)
        project = params.get("project", [None])[0]

        if path == "/api/memory/delete":
            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length)) if content_length else {}
            memory_id = body.get("memory_id", "")
            reason = body.get("reason", "user requested")
            self._json_response(api_delete_memory(memory_id, reason, project))
        else:
            self.send_error(404)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _serve_html(self):
        if not HTML_PATH.exists():
            self.send_error(404, "dashboard.html not found alongside server")
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(HTML_PATH.read_bytes())

    def _json_response(self, data):
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2, default=str).encode())

    def log_message(self, format, *args):
        pass  # Silence request logs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  MATHIR Neural Memory Dashboard")
    print("=" * 60)
    print()

    # Show DB detection info
    if DB_PATH_OVERRIDE:
        print(f"  DB (env):   {DB_PATH_OVERRIDE}")
    else:
        print("  DB:         auto-detecting...")

    dbs = _find_mathir_dbs()
    if dbs:
        print(f"  Found {len(dbs)} database(s):")
        for db in dbs:
            size = db["size_bytes"]
            size_str = f"{size / (1024*1024):.2f} MB" if size > 1048576 else f"{size / 1024:.1f} KB"
            print(f"    - {db['name']}: {size_str} @ {db['path']}")
    else:
        print("  WARNING: No MATHIR databases found")
        print("  Dashboard will start but show empty data")
        print("  Set MATHIR_DB env var to point to a .mathir/mathir.db file")

    # Config
    config = _load_config()
    if config:
        print(f"  Config:     loaded")
    else:
        print(f"  Config:     not found (using defaults)")

    print(f"  Port:       {PORT}")
    print(f"  URL:        http://127.0.0.1:{PORT}")
    print()

    server = HTTPServer(("0.0.0.0", PORT), DashboardHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()


if __name__ == "__main__":
    main()
