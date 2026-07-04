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
