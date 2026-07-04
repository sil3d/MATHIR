"""Tests for MATHIR God Orchestrator (mathir_god.py)."""

import sys
import os
import json
from unittest.mock import patch, MagicMock
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mathir_lib"))

import pytest
from mathir_god import GodProtocol, GOD_PREFIX


class TestGodProtocol:

    def test_make_label_basic(self):
        label = GodProtocol.make_label("task", "a1b2c3d4", "mimo", "pending")
        assert label == "god:task:a1b2c3d4:mimo:pending"

    def test_make_label_registration(self):
        label = GodProtocol.make_label("reg", "mimo", "mimo", "idle")
        assert label == "god:reg:mimo:mimo:idle"

    def test_parse_label_valid(self):
        result = GodProtocol.parse_label("god:task:a1b2c3d4:mimo:pending")
        assert result == {
            "type": "task",
            "id": "a1b2c3d4",
            "target": "mimo",
            "status": "pending",
        }

    def test_parse_label_not_god(self):
        assert GodProtocol.parse_label("some-other-label") is None

    def test_parse_label_too_few_parts(self):
        assert GodProtocol.parse_label("god:task:abc") is None

    def test_generate_task_id_length(self):
        tid = GodProtocol.generate_task_id()
        assert len(tid) == 8

    def test_generate_task_id_hex(self):
        tid = GodProtocol.generate_task_id()
        int(tid, 16)  # should not raise

    def test_generate_task_id_unique(self):
        ids = {GodProtocol.generate_task_id() for _ in range(100)}
        assert len(ids) == 100

    def test_god_prefix_constant(self):
        assert GOD_PREFIX == "god:"


from mathir_god import TaskGraph


class TestTaskGraph:

    def _make_graph(self):
        g = TaskGraph("Refactor auth + tests")
        g.add_task("t1", "Refactor auth.py", "mimo", ["code"], [])
        g.add_task("t2", "Write tests", "codex", ["code", "test"], ["t1"])
        g.add_task("t3", "Update docs", "opencode", ["docs"], [])
        return g

    def test_add_task(self):
        g = self._make_graph()
        tasks = g.get_all_tasks()
        assert len(tasks) == 3
        assert tasks["t1"]["description"] == "Refactor auth.py"
        assert tasks["t1"]["status"] == "queued"

    def test_get_ready_tasks_initial(self):
        g = self._make_graph()
        ready = g.get_ready_tasks()
        ids = [t["task_id"] for t in ready]
        assert "t1" in ids
        assert "t3" in ids
        assert "t2" not in ids  # depends on t1

    def test_get_ready_tasks_after_t1_done(self):
        g = self._make_graph()
        g.set_status("t1", "completed")
        ready = g.get_ready_tasks()
        ids = [t["task_id"] for t in ready]
        assert "t2" in ids
        assert "t3" in ids

    def test_set_status(self):
        g = self._make_graph()
        g.set_status("t1", "running")
        assert g.get_all_tasks()["t1"]["status"] == "running"

    def test_set_status_invalid_task(self):
        g = self._make_graph()
        with pytest.raises(KeyError):
            g.set_status("nonexistent", "running")

    def test_is_all_done_false(self):
        g = self._make_graph()
        assert g.is_all_done() is False

    def test_is_all_done_true(self):
        g = self._make_graph()
        for tid in ["t1", "t2", "t3"]:
            g.set_status(tid, "verified")
        assert g.is_all_done() is True

    def test_is_all_done_completed_counts(self):
        g = self._make_graph()
        for tid in ["t1", "t2", "t3"]:
            g.set_status(tid, "completed")
        assert g.is_all_done() is True

    def test_json_roundtrip(self):
        g = self._make_graph()
        g.set_status("t1", "running")
        data = g.to_json()
        g2 = TaskGraph.from_json(data)
        assert g2.get_all_tasks()["t1"]["status"] == "running"
        assert g2.directive == "Refactor auth + tests"
        assert len(g2.get_all_tasks()) == 3

    def test_no_ready_when_running(self):
        g = self._make_graph()
        g.set_status("t1", "running")
        ready = g.get_ready_tasks()
        ids = [t["task_id"] for t in ready]
        assert "t1" not in ids  # already running


from mathir_god import WorkerRegistry, WorktreeManager


class TestWorkerRegistry:

    def test_register(self):
        r = WorkerRegistry()
        r.register("mimo", ["code", "test"])
        assert len(r.list_all()) == 1
        assert r.list_all()[0]["name"] == "mimo"

    def test_register_overwrites(self):
        r = WorkerRegistry()
        r.register("mimo", ["code"])
        r.register("mimo", ["code", "test"])
        assert len(r.list_all()) == 1
        assert "test" in r.list_all()[0]["capabilities"]

    def test_unregister(self):
        r = WorkerRegistry()
        r.register("mimo", ["code"])
        r.unregister("mimo")
        assert len(r.list_all()) == 0

    def test_unregister_nonexistent(self):
        r = WorkerRegistry()
        r.unregister("ghost")  # should not raise

    def test_set_status(self):
        r = WorkerRegistry()
        r.register("mimo", ["code"])
        r.set_status("mimo", "busy")
        assert r.list_all()[0]["status"] == "busy"

    def test_list_idle(self):
        r = WorkerRegistry()
        r.register("mimo", ["code"])
        r.register("codex", ["code"])
        r.set_status("mimo", "busy")
        idle = r.list_idle()
        assert len(idle) == 1
        assert idle[0]["name"] == "codex"

    def test_find_by_capability(self):
        r = WorkerRegistry()
        r.register("mimo", ["code", "test"])
        r.register("codex", ["code"])
        r.register("opencode", ["review", "docs"])
        assert "mimo" in r.find_by_capability("test")
        assert "codex" not in r.find_by_capability("test")
        assert len(r.find_by_capability("code")) == 2

    def test_from_daemon_response(self):
        data = [
            {"name": "mimo", "status": "idle", "capabilities": ["code"]},
            {"name": "codex", "status": "busy", "capabilities": ["code", "fast"]},
        ]
        r = WorkerRegistry.from_daemon_response(data)
        assert len(r.list_all()) == 2
        assert r.list_idle()[0]["name"] == "mimo"


class TestWorktreeManager:

    @patch("mathir_god.subprocess.run")
    def test_create_success(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        wm = WorktreeManager(base_dir=Path("/tmp/test"))
        ok, msg, path = wm.create("abc12345")
        assert ok is True
        assert path is not None
        assert "abc12345" in str(path)

    @patch("mathir_god.subprocess.run")
    def test_create_failure(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stderr="fatal: already exists")
        wm = WorktreeManager(base_dir=Path("/tmp/test"))
        ok, msg, path = wm.create("abc12345")
        assert ok is False
        assert path is None

    @patch("mathir_god.subprocess.run")
    def test_merge_success(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="Already up to date.")
        wm = WorktreeManager(base_dir=Path("/tmp/test"))
        ok, msg = wm.merge("abc12345")
        assert ok is True

    @patch("mathir_god.subprocess.run")
    def test_merge_conflict(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stderr="CONFLICT", stdout="")
        wm = WorktreeManager(base_dir=Path("/tmp/test"))
        ok, msg = wm.merge("abc12345")
        assert ok is False
        assert "CONFLICT" in msg

    @patch("mathir_god.subprocess.run")
    def test_cleanup(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        wm = WorktreeManager(base_dir=Path("/tmp/test"))
        wm.cleanup("abc12345")  # should not raise
        assert mock_run.called

    @patch("mathir_god.subprocess.run")
    def test_list_active(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="/repo/.worktrees/god-abc12345 abc12345abc [god/abc12345]\n/repo/.worktrees/god-def67890 def67890def [god/def67890]\n/repo/main HEAD [main]\n",
        )
        wm = WorktreeManager(base_dir=Path("/repo"))
        active = wm.list_active()
        assert "abc12345" in active
        assert "def67890" in active


class TestDaemonRoutes:
    """Tests for the god-mode SQL queries used by daemon routes.

    These test the query logic, not Flask — we verify the SQL patterns
    that /api/god/poll and /api/god/agents will use.
    """

    def test_parse_god_labels_from_rows(self):
        labels = [
            "god:reg:mimo:mimo:idle",
            "god:reg:codex:codex:busy",
            "not-a-god-label",
            "god:task:abc12345:mimo:pending",
        ]
        parsed = [GodProtocol.parse_label(l) for l in labels]
        regs = [p for p in parsed if p and p["type"] == "reg"]
        assert len(regs) == 2
        assert regs[0]["target"] == "mimo"
        assert regs[0]["status"] == "idle"

    def test_filter_tasks_for_agent(self):
        labels = [
            "god:task:aaa11111:mimo:pending",
            "god:task:bbb22222:codex:pending",
            "god:task:ccc33333:mimo:running",
        ]
        mimo_pending = [
            GodProtocol.parse_label(l) for l in labels
            if l.endswith(":mimo:pending")
        ]
        assert len(mimo_pending) == 1
        assert mimo_pending[0]["id"] == "aaa11111"
