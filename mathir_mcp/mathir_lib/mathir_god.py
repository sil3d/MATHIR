"""MATHIR God Orchestrator — cross-process multi-agent coordination.

Uses MATHIR shared memory as a message queue with structured labels:
    god:{type}:{id}:{target}:{status}

Classes:
    GodProtocol   — label encoding/decoding
    TaskGraph     — DAG of tasks with dependencies
    WorkerRegistry — track registered workers
    WorktreeManager — git worktree lifecycle
"""

import json
import subprocess
import uuid
from pathlib import Path


GOD_PREFIX = "god:"


class GodProtocol:
    """Label encoding/decoding and message helpers."""

    @staticmethod
    def make_label(msg_type: str, task_id: str, target: str, status: str) -> str:
        return f"god:{msg_type}:{task_id}:{target}:{status}"

    @staticmethod
    def parse_label(label: str) -> dict | None:
        if not label or not label.startswith(GOD_PREFIX):
            return None
        parts = label.split(":")
        if len(parts) != 5:
            return None
        return {
            "type": parts[1],
            "id": parts[2],
            "target": parts[3],
            "status": parts[4],
        }

    @staticmethod
    def generate_task_id() -> str:
        return uuid.uuid4().hex[:8]


class TaskGraph:
    """DAG of tasks with dependencies."""

    def __init__(self, directive: str, directive_id: str = None):
        self.directive = directive
        self.directive_id = directive_id or GodProtocol.generate_task_id()
        self._tasks: dict[str, dict] = {}

    def add_task(
        self,
        task_id: str,
        description: str,
        agent: str,
        capabilities_required: list[str] = None,
        depends_on: list[str] = None,
    ) -> None:
        self._tasks[task_id] = {
            "task_id": task_id,
            "description": description,
            "agent": agent,
            "capabilities_required": capabilities_required or [],
            "depends_on": depends_on or [],
            "status": "queued",
            "worktree_branch": f"god/{task_id}",
        }

    def set_status(self, task_id: str, status: str) -> None:
        if task_id not in self._tasks:
            raise KeyError(f"Unknown task: {task_id}")
        self._tasks[task_id]["status"] = status

    def get_ready_tasks(self) -> list[dict]:
        ready = []
        done_statuses = {"completed", "verified"}
        for task in self._tasks.values():
            if task["status"] != "queued":
                continue
            deps_met = all(
                self._tasks[d]["status"] in done_statuses
                for d in task["depends_on"]
                if d in self._tasks
            )
            if deps_met:
                ready.append(task)
        return ready

    def get_all_tasks(self) -> dict[str, dict]:
        return dict(self._tasks)

    def is_all_done(self) -> bool:
        if not self._tasks:
            return False
        return all(
            t["status"] in ("completed", "verified")
            for t in self._tasks.values()
        )

    def to_json(self) -> str:
        return json.dumps({
            "directive_id": self.directive_id,
            "directive": self.directive,
            "tasks": self._tasks,
        })

    @classmethod
    def from_json(cls, data: str) -> "TaskGraph":
        d = json.loads(data)
        g = cls(d["directive"], d.get("directive_id"))
        g._tasks = d["tasks"]
        return g


class WorkerRegistry:
    """Track registered workers (in-memory, populated from daemon responses)."""

    def __init__(self):
        self._workers: dict[str, dict] = {}

    def register(self, name: str, capabilities: list[str]) -> None:
        self._workers[name] = {
            "name": name,
            "capabilities": capabilities,
            "status": "idle",
        }

    def unregister(self, name: str) -> None:
        self._workers.pop(name, None)

    def set_status(self, name: str, status: str) -> None:
        if name in self._workers:
            self._workers[name]["status"] = status

    def list_idle(self) -> list[dict]:
        return [w for w in self._workers.values() if w["status"] == "idle"]

    def list_all(self) -> list[dict]:
        return list(self._workers.values())

    def find_by_capability(self, cap: str) -> list[str]:
        return [
            w["name"]
            for w in self._workers.values()
            if cap in w["capabilities"]
        ]

    @classmethod
    def from_daemon_response(cls, agents: list[dict]) -> "WorkerRegistry":
        r = cls()
        for a in agents:
            r._workers[a["name"]] = {
                "name": a["name"],
                "capabilities": a.get("capabilities", []),
                "status": a.get("status", "idle"),
            }
        return r


class WorktreeManager:
    """Git worktree lifecycle for task isolation."""

    def __init__(self, base_dir: Path = None):
        self.base_dir = base_dir or Path.cwd()
        self.worktree_dir = self.base_dir / ".worktrees"

    def create(self, task_id: str) -> tuple[bool, str, Path | None]:
        wt_path = self.worktree_dir / f"god-{task_id}"
        branch = f"god/{task_id}"
        result = subprocess.run(
            ["git", "worktree", "add", str(wt_path), "-b", branch],
            capture_output=True,
            text=True,
            cwd=str(self.base_dir),
        )
        if result.returncode == 0:
            return True, f"Created worktree at {wt_path}", wt_path
        return False, result.stderr.strip(), None

    def merge(self, task_id: str) -> tuple[bool, str]:
        branch = f"god/{task_id}"
        result = subprocess.run(
            ["git", "merge", branch, "--no-edit"],
            capture_output=True,
            text=True,
            cwd=str(self.base_dir),
        )
        if result.returncode == 0:
            return True, result.stdout.strip()
        return False, result.stderr.strip() or result.stdout.strip()

    def cleanup(self, task_id: str) -> None:
        wt_path = self.worktree_dir / f"god-{task_id}"
        subprocess.run(
            ["git", "worktree", "remove", str(wt_path), "--force"],
            capture_output=True,
            text=True,
            cwd=str(self.base_dir),
        )
        branch = f"god/{task_id}"
        subprocess.run(
            ["git", "branch", "-D", branch],
            capture_output=True,
            text=True,
            cwd=str(self.base_dir),
        )

    def list_active(self) -> list[str]:
        result = subprocess.run(
            ["git", "worktree", "list"],
            capture_output=True,
            text=True,
            cwd=str(self.base_dir),
        )
        task_ids = []
        for line in result.stdout.strip().split("\n"):
            if "god-" in line:
                for part in line.split():
                    if "god-" in part:
                        tid = part.split("god-")[-1].rstrip("]").rstrip("/")
                        task_ids.append(tid)
                        break
        return task_ids
