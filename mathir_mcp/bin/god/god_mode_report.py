#!/usr/bin/env python3
"""GOD MODE REPORT — deterministic, model-independent visibility into what
god-mode workers actually did/said.

Why this exists: right now the only way to see what a worker produced is
for the ORCHESTRATING LLM to remember to search memory and relay it to the
human -- reported live, 2026-07-21: the user asked 3 workers a question and
never saw any of their answers, because that step depended entirely on the
orchestrator's judgment/memory, not on anything deterministic. If a
different, less careful model is orchestrating, this silently breaks. This
script has no LLM in the loop at all -- it's a plain HTTP+SQL report the
human can run themselves, anytime, regardless of which model (if any) is
currently orchestrating.

Usage:
    python god_mode_report.py --cwd D:/path/to/project
    python god_mode_report.py --cwd D:/path/to/project --task perfdisc1
    python god_mode_report.py --cwd D:/path/to/project --since 2026-07-21T00:00:00

NOTE: reads the local SQLite DB (<cwd>/.mathir/mathir.db) directly instead
of going through /api/memories. That HTTP route resolves the project via
_get_project_db() (the dashboard/legacy project-registry lookup), not the
_resolve_db(project, cwd) pattern every other route uses (see guardrail
"guardrail-resolve-db-pattern") -- verified live, 2026-07-21: it returned
"No database found" for a project that unquestionably has an active local
DB, because it wasn't registered under that exact key. Reading the DB file
directly sidesteps that route's bug entirely and needs no daemon running.
"""

import argparse
import json
import sqlite3
from datetime import datetime
from pathlib import Path


def fetch_memories(cwd: str, limit: int) -> list[dict]:
    db_path = Path(cwd) / ".mathir" / "mathir.db"
    if not db_path.exists():
        print(f"ERROR: no MATHIR database found at {db_path}")
        return []
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    columns = {col[1] for col in conn.execute("PRAGMA table_info(memories)").fetchall()}
    text_col = "content" if "content" in columns else "modality_text"
    ts_col = "created_at" if "created_at" in columns else "timestamp"
    rows = conn.execute(
        f"SELECT memory_id, {text_col} AS content, metadata, tier, label, {ts_col} AS created_at "
        f"FROM memories ORDER BY rowid DESC LIMIT ?",
        (limit,),
    ).fetchall()
    memories = []
    for row in rows:
        d = dict(row)
        try:
            d["metadata"] = json.loads(d["metadata"]) if d["metadata"] else {}
        except (json.JSONDecodeError, TypeError):
            d["metadata"] = {}
        # The real `label` column is the source of truth -- /api/god/ack
        # and /api/god/poll both read/write it directly. metadata.label is
        # only a snapshot taken at memory_save time and goes stale the
        # moment /api/god/ack flips the column (verified live, 2026-07-21:
        # ack correctly moved the column to ":running" while metadata.label
        # stayed frozen at ":pending", making every in-flight task look
        # stuck even though the queue itself was working fine). Fall back
        # to metadata.label only for rows saved before the `label` column
        # existed.
        d["label"] = row["label"] or d["metadata"].get("label", "")
        memories.append(d)
    conn.close()
    return memories


def parse_god_label(label: str) -> dict | None:
    parts = label.split(":")
    if len(parts) != 5 or parts[0] != "god":
        return None
    return {"kind": parts[1], "task_id": parts[2], "target": parts[3], "status": parts[4]}


def pretty_content(raw: str) -> str:
    try:
        obj = json.loads(raw)
        return json.dumps(obj, indent=2, ensure_ascii=False)
    except (json.JSONDecodeError, TypeError):
        return raw


def main() -> int:
    ap = argparse.ArgumentParser(description="Human-readable report of MATHIR god-mode activity")
    ap.add_argument("--cwd", required=True, help="Project working directory (reads <cwd>/.mathir/mathir.db)")
    ap.add_argument("--limit", type=int, default=300, help="How many recent memories to scan (default 300)")
    ap.add_argument("--task", default=None, help="Only show this task_id (the 8-char hex in god:*:<task_id>:*)")
    ap.add_argument("--since", default=None, help="Only show entries created after this ISO timestamp")
    args = ap.parse_args()

    project = Path(args.cwd).name
    memories = fetch_memories(args.cwd, args.limit)

    since_dt = None
    if args.since:
        try:
            since_dt = datetime.fromisoformat(args.since)
        except ValueError:
            print(f"WARNING: could not parse --since '{args.since}', ignoring")

    # Keyed by task_id -> {"assignments": {target: latest_record}, "results": [record, ...]}.
    # A single task_id can legitimately be dispatched to MULTIPLE targets at
    # once (e.g. the same directive fanned out to 3 different workers, each
    # with their own god:task:<id>:<target>:<status> label) -- collapsing to
    # one "latest" record per task_id silently hid the other targets'
    # assignments. Fixed live, 2026-07-21, same session that exposed it.
    by_task: dict[str, dict] = {}
    for mem in memories:
        meta = mem.get("metadata") or {}
        label = mem.get("label", "")
        parsed = parse_god_label(label)
        if not parsed:
            continue
        if parsed["kind"] not in ("task", "result"):
            continue
        if args.task and parsed["task_id"] != args.task:
            continue

        created_at = mem.get("created_at") or mem.get("timestamp") or ""
        if since_dt and created_at:
            try:
                ts = created_at[:-1] if created_at.endswith("Z") else created_at
                if datetime.fromisoformat(ts) < since_dt:
                    continue
            except ValueError:
                pass

        entry = by_task.setdefault(parsed["task_id"], {"assignments": {}, "results": []})
        record = {
            "label": label,
            "target": parsed["target"],
            "status": parsed["status"],
            "content": mem.get("content", ""),
            "agent": meta.get("agent", "unknown"),
            "created_at": created_at,
        }
        if parsed["kind"] == "task":
            existing = entry["assignments"].get(parsed["target"])
            if existing is None or created_at >= existing["created_at"]:
                entry["assignments"][parsed["target"]] = record
        else:
            entry["results"].append(record)

    if not by_task:
        print(f"No god-mode task/result activity found for project '{project}' in the last {args.limit} memories.")
        return 0

    def task_sort_key(kv):
        _, entry = kv
        times = [a["created_at"] for a in entry["assignments"].values()]
        return min(times) if times else ""

    print(f"=== GOD MODE REPORT — project '{project}' — {len(by_task)} task(s) ===\n")
    for task_id, entry in sorted(by_task.items(), key=task_sort_key):
        print(f"--- Task {task_id} ---")
        results_by_agent: dict[str, list[dict]] = {}
        for r in entry["results"]:
            results_by_agent.setdefault(r["agent"], []).append(r)

        if not entry["assignments"]:
            print("  (no task record found -- only result(s) exist)")
            for agent, results in results_by_agent.items():
                for r in sorted(results, key=lambda x: x["created_at"]):
                    print(f"\n  >>> Response from '{agent}' ({r['created_at']}):")
                    for line in pretty_content(r["content"]).splitlines():
                        print(f"      {line}")
            print()
            continue

        for target, task in sorted(entry["assignments"].items(), key=lambda kv: kv[1]["created_at"]):
            desc = pretty_content(task["content"])
            try:
                desc_obj = json.loads(task["content"])
                desc = desc_obj.get("description", desc)
            except (json.JSONDecodeError, TypeError):
                pass
            print(f"  Assigned to : {target}")
            print(f"  Status      : {task['status']}")
            print(f"  Dispatched  : {task['created_at']}")
            print(f"  Description : {desc[:500]}")

            matching = results_by_agent.pop(target, [])
            if not matching:
                print("  Result      : NONE YET -- worker has not reported back.")
            else:
                for r in sorted(matching, key=lambda x: x["created_at"]):
                    print(f"\n  >>> Response from '{target}' ({r['created_at']}):")
                    for line in pretty_content(r["content"]).splitlines():
                        print(f"      {line}")
            print()

        # Any results left over didn't match a known target (e.g. the
        # worker self-identified under a different name than it was
        # dispatched to) -- show them anyway rather than silently dropping.
        for agent, leftover in results_by_agent.items():
            for r in sorted(leftover, key=lambda x: x["created_at"]):
                print(f"  >>> Unmatched response from '{agent}' ({r['created_at']}):")
                for line in pretty_content(r["content"]).splitlines():
                    print(f"      {line}")
            print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
