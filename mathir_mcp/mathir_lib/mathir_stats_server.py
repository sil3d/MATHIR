#!/usr/bin/env python3
"""
MATHIR Stats Dashboard — Backend API
Reads from MATHIR SQLite DB and serves JSON + HTML dashboard.
Supports per-project databases.
"""

import json
import os
import sqlite3
import sys
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs

# Per-project database support
PROJECTS_DIR = Path(os.environ.get(
    "MATHIR_PROJECTS_DIR",
    os.path.expanduser("~/.config/opencode/data/projects")
))
LEGACY_DB_PATH = Path(os.environ.get(
    "MATHIR_DB",
    os.path.expanduser("~/.config/opencode/data/mathir.db")
))
CONFIG_PATH = Path(os.environ.get(
    "MATHIR_CONFIG",
    os.path.expanduser("~/.config/opencode/config/mathir.json")
))
# Central registry - tracks all projects that have ever used MATHIR
REGISTRY_PATH = Path(os.environ.get(
    "MATHIR_REGISTRY",
    os.path.expanduser("~/.config/opencode/data/mathir_registry.json")
))
HTML_PATH = Path(__file__).parent / "mathir_dashboard.html"
PORT = int(os.environ.get("MATHIR_STATS_PORT", "7420"))


def get_project_db_path(project_name: str = None) -> Path:
    """Get database path for a specific project from the central registry."""
    if project_name is None or project_name == "legacy":
        # Return legacy DB if it exists
        if LEGACY_DB_PATH.exists():
            return LEGACY_DB_PATH
        # Try to find any project with .mathir/mathir.db
        return None
    
    # Check central registry first
    if REGISTRY_PATH.exists():
        try:
            with open(REGISTRY_PATH) as f:
                registry = json.load(f)
            if project_name in registry.get("projects", {}):
                info = registry["projects"][project_name]
                db_path = Path(info.get("db_path", ""))
                if db_path.exists():
                    return db_path
        except Exception:
            pass
    
    # Fallback: scan common directories
    home = Path.home()
    common_dirs = [
        home / "Documents",
        home / "Desktop",
        home / "Projects",
        home / "dev",
        home / "Code",
    ]
    
    for parent_dir in common_dirs:
        if parent_dir.exists():
            for project_dir in parent_dir.iterdir():
                if project_dir.is_dir() and project_dir.name == project_name:
                    db_path = project_dir / ".mathir" / "mathir.db"
                    if db_path.exists():
                        return db_path
    
    return None


def list_projects() -> list:
    """List all available projects from the central registry."""
    projects = []
    seen = set()
    
    # Check legacy DB
    if LEGACY_DB_PATH.exists():
        projects.append({
            "name": "legacy",
            "path": str(LEGACY_DB_PATH),
            "size_bytes": LEGACY_DB_PATH.stat().st_size,
            "cwd": str(LEGACY_DB_PATH.parent)
        })
        seen.add(str(LEGACY_DB_PATH))
    
    # Load from central registry
    if REGISTRY_PATH.exists():
        try:
            with open(REGISTRY_PATH) as f:
                registry = json.load(f)
            for name, info in registry.get("projects", {}).items():
                db_path = Path(info.get("db_path", ""))
                if db_path.exists() and str(db_path) not in seen:
                    seen.add(str(db_path))
                    projects.append({
                        "name": name,
                        "path": str(db_path),
                        "size_bytes": db_path.stat().st_size,
                        "cwd": info.get("cwd", str(db_path.parent)),
                        "last_used": info.get("last_used", ""),
                        "imported": info.get("imported", False)
                    })
        except Exception:
            pass
    
    return projects


def get_db(project_name: str = None):
    """Get database connection for a project."""
    db_path = get_project_db_path(project_name)
    if db_path is None:
        # Return None if no database found
        return None
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def load_config():
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return json.load(f)
    return {}


def api_overview(project_name: str = None):
    """Global overview stats for a project."""
    conn = get_db(project_name)
    if conn is None:
        return {"error": "No database found", "project": project_name}
    
    config = load_config()

    # Count memories by tier (via metadata block_type)
    try:
        rows = conn.execute(
            "SELECT metadata FROM memories WHERE metadata IS NOT NULL"
        ).fetchall()
    except Exception:
        rows = []

    tiers = {"working": 0, "episodic": 0, "semantic": 0, "procedural": 0, "immunological": 0, "unknown": 0}
    agents = {}
    total = 0

    for row in rows:
        try:
            meta = json.loads(row["metadata"])
            bt = meta.get("block_type", "unknown")
            agent = meta.get("agent", "unknown")
            tier = "working" if bt == "working_memory" else bt
            if tier not in tiers:
                tier = "unknown"
            tiers[tier] += 1
            agents[agent] = agents.get(agent, 0) + 1
            total += 1
        except Exception:
            tiers["unknown"] += 1
            total += 1

    # DB size
    db_path = get_project_db_path(project_name)
    db_size = db_path.stat().st_size if db_path and db_path.exists() else 0

    # Memory config
    mem_config = config.get("memory", {})

    conn.close()

    return {
        "total_memories": total,
        "tiers": tiers,
        "agents": agents,
        "db_size_bytes": db_size,
        "db_size_mb": round(db_size / (1024 * 1024), 2),
        "config": {
            "embedding_dim": mem_config.get("embedding_dim", 1024),
            "working_capacity": mem_config.get("working_capacity", 64),
            "episodic_capacity": mem_config.get("episodic_capacity", 1000),
            "semantic_prototypes": mem_config.get("semantic_prototypes", 256),
            "immunological_capacity": mem_config.get("immunological_capacity", 100),
            "kl_coefficient": mem_config.get("kl_coefficient", 0.01),
            "anomaly_threshold": mem_config.get("anomaly_threshold", 2.0),
            "decay_rate": mem_config.get("decay_rate", 0.95),
        },
        "project": project_name,
        "timestamp": datetime.now().isoformat()
    }


def api_memories(limit=100, offset=0, agent=None, block_type=None, project_name=None):
    """List all memories with metadata."""
    conn = get_db(project_name)
    if conn is None:
        return {"error": "No database found", "project": project_name}

    query = "SELECT memory_id, modality_text, metadata, tier, timestamp, provider, model FROM memories WHERE 1=1"
    params = []

    if agent:
        query += " AND json_extract(metadata, '$.agent') = ?"
        params.append(agent)
    if block_type:
        query += " AND json_extract(metadata, '$.block_type') = ?"
        params.append(block_type)

    query += " ORDER BY rowid DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    try:
        rows = conn.execute(query, params).fetchall()
    except Exception:
        rows = []

    # Total count
    count_query = "SELECT COUNT(*) as cnt FROM memories WHERE 1=1"
    count_params = []
    if agent:
        count_query += " AND json_extract(metadata, '$.agent') = ?"
        count_params.append(agent)
    if block_type:
        count_query += " AND json_extract(metadata, '$.block_type') = ?"
        count_params.append(block_type)

    try:
        total = conn.execute(count_query, count_params).fetchone()["cnt"]
    except Exception:
        total = 0

    memories = []
    for row in rows:
        row = dict(row)
        try:
            row["metadata"] = json.loads(row["metadata"]) if row["metadata"] else {}
        except Exception:
            row["metadata"] = {}
        memories.append(row)

    conn.close()

    return {
        "memories": memories,
        "total": total,
        "limit": limit,
        "offset": offset,
        "project": project_name
    }


def api_tier_details(project_name: str = None):
    """Detailed breakdown per tier."""
    conn = get_db(project_name)
    if conn is None:
        return {"error": "No database found", "project": project_name}

    try:
        rows = conn.execute(
            "SELECT metadata FROM memories WHERE metadata IS NOT NULL"
        ).fetchall()
    except Exception:
        rows = []

    tiers = {}
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

    config = load_config().get("memory", {})
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


def api_router_weights(project_name: str = None):
    """Simulated router weights based on tier distribution."""
    tiers = api_tier_details(project_name)
    total = sum(t["count"] for t in tiers.values()) or 1

    weights = {}
    for tier, data in tiers.items():
        weights[tier] = round(data["count"] / total, 3)

    return {
        "weights": weights,
        "total_memories": total,
        "router_type": load_config().get("router", {}).get("type", "kl_constrained"),
        "kl_coefficient": load_config().get("router", {}).get("kl_coefficient", 0.01),
        "project": project_name
    }


def api_agents(project_name: str = None):
    """Per-agent breakdown with tier distribution."""
    conn = get_db(project_name)
    if conn is None:
        return {"error": "No database found", "project": project_name}
    
    try:
        rows = conn.execute(
            "SELECT metadata FROM memories WHERE metadata IS NOT NULL"
        ).fetchall()
    except Exception:
        rows = []

    agents = {}
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
    return {"agents": agents, "project": project_name}


def api_timeline(project_name: str = None):
    """Memory creation timeline (grouped by hour)."""
    conn = get_db(project_name)
    if conn is None:
        return {"error": "No database found", "project": project_name}
    
    try:
        rows = conn.execute(
            "SELECT metadata, timestamp FROM memories WHERE timestamp IS NOT NULL ORDER BY timestamp"
        ).fetchall()
    except Exception:
        rows = []

    # Group by hour
    buckets = {}
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
    return {"timeline": timeline, "project": project_name}


def api_delete_memory(memory_id, reason="user requested", project_name=None):
    """Delete a memory by ID."""
    conn = get_db(project_name)
    if conn is None:
        return {"error": "No database found", "project": project_name}
    
    try:
        cursor = conn.execute(
            "DELETE FROM memories WHERE memory_id = ?", (memory_id,)
        )
        deleted = cursor.rowcount > 0
        conn.commit()
    except Exception as e:
        deleted = False
    conn.close()
    return {"memory_id": memory_id, "deleted": deleted, "reason": reason, "project": project_name}


def validate_mathir_db(db_path: str) -> bool:
    """Validate that a file is a valid MATHIR database."""
    try:
        import sqlite3
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check for required tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        
        # MATHIR uses 'memories' and 'memory_embeddings' tables
        required_tables = {"memories", "memory_embeddings"}
        if not required_tables.issubset(tables):
            conn.close()
            return False
        
        # Check memories table has required columns
        cursor.execute("PRAGMA table_info(memories)")
        columns = {row[1] for row in cursor.fetchall()}
        
        required_columns = {"memory_id", "modality_text", "timestamp", "tier"}
        if not required_columns.issubset(columns):
            conn.close()
            return False
        
        # Check memory_embeddings table
        cursor.execute("PRAGMA table_info(memory_embeddings)")
        emb_columns = {row[1] for row in cursor.fetchall()}
        
        required_emb_columns = {"memory_id", "embedding"}
        if not required_emb_columns.issubset(emb_columns):
            conn.close()
            return False
        
        # Try to read a few rows to ensure data integrity
        cursor.execute("SELECT COUNT(*) FROM memories")
        count = cursor.fetchone()[0]
        
        conn.close()
        return True
    except Exception:
        return False


def api_import_db(project_name: str, db_data_b64: str, db_path: str = "") -> dict:
    """Import a MATHIR database from base64-encoded data or file path."""
    import base64
    import tempfile
    
    if not project_name:
        return {"error": "project_name is required"}
    
    # Handle file path import (local file)
    if db_path and not db_data_b64:
        if not os.path.exists(db_path):
            return {"error": f"File not found: {db_path}"}
        if not db_path.endswith('.db'):
            return {"error": "File must have .db extension"}
        
        # Validate
        if not validate_mathir_db(db_path):
            return {"error": "Not a valid MATHIR database. Must have 'blocks' and 'embeddings' tables with correct schema."}
        
        # Read file data
        with open(db_path, 'rb') as f:
            db_bytes = f.read()
        file_size = len(db_bytes)
        
    # Handle base64 data import (from upload)
    elif db_data_b64:
        try:
            db_bytes = base64.b64decode(db_data_b64)
            file_size = len(db_bytes)
        except Exception:
            return {"error": "Invalid base64 data"}
        
        # Write to temp file for validation
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            tmp.write(db_bytes)
            tmp_path = tmp.name
        
        # Validate it's a real MATHIR database
        if not validate_mathir_db(tmp_path):
            os.unlink(tmp_path)
            return {"error": "Not a valid MATHIR database. Must have 'blocks' and 'embeddings' tables with correct schema."}
        
        # Clean up temp file
        os.unlink(tmp_path)
    else:
        return {"error": "Either db_data (base64) or db_path is required"}
    
    try:
        # Save to projects directory
        import hashlib
        project_hash = hashlib.md5(project_name.encode()).hexdigest()[:8]
        safe_name = f"{project_name}_{project_hash}"
        
        project_dir = PROJECTS_DIR / safe_name
        project_dir.mkdir(parents=True, exist_ok=True)
        
        final_db_path = project_dir / "mathir.db"
        with open(final_db_path, 'wb') as f:
            f.write(db_bytes)
        
        # Register in central registry
        registry = {"projects": {}}
        if REGISTRY_PATH.exists():
            try:
                with open(REGISTRY_PATH) as f:
                    registry = json.load(f)
            except:
                pass
        
        registry["projects"][project_name] = {
            "db_path": str(final_db_path),
            "cwd": str(project_dir),
            "last_used": datetime.now().isoformat(),
            "name": project_name,
            "imported": True
        }
        
        REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(REGISTRY_PATH, "w") as f:
            json.dump(registry, f, indent=2, default=str)
        
        return {
            "success": True,
            "project_name": project_name,
            "db_path": str(final_db_path),
            "size_bytes": file_size
        }
        
    except Exception as e:
        return {"error": f"Import failed: {str(e)}"}


class DashboardHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)
        project = params.get("project", [None])[0]

        if path == "/" or path == "/dashboard":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PATH.read_bytes())

        elif path == "/api/projects":
            self._json_response({"projects": list_projects()})

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
        elif path == "/api/import":
            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length)) if content_length else {}
            project_name = body.get("project_name", "")
            db_data_b64 = body.get("db_data", "")
            db_path = body.get("db_path", "")
            self._json_response(api_import_db(project_name, db_data_b64, db_path))
        else:
            self.send_error(404)

    def _json_response(self, data):
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2, default=str).encode())

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def log_message(self, format, *args):
        pass  # Silence logs


def main():
    # Check if any database exists
    projects = list_projects()
    if not projects:
        print(f"WARNING: No MATHIR databases found")
        print(f"  Legacy DB: {LEGACY_DB_PATH}")
        print(f"  Projects dir: {PROJECTS_DIR}")
        print(f"  Dashboard will start but show empty data")
    else:
        print(f"MATHIR Stats Dashboard")
        print(f"  Found {len(projects)} project(s):")
        for p in projects:
            print(f"    - {p['name']}: {p['size_bytes']} bytes")
    
    print(f"  Config: {CONFIG_PATH}")
    print(f"  URL:    http://127.0.0.1:{PORT}")
    print()

    server = HTTPServer(("127.0.0.1", PORT), DashboardHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()


if __name__ == "__main__":
    main()
