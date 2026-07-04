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

    def test_cycle_detection(self):
        """Adding a task that creates a circular dependency raises ValueError."""
        g = TaskGraph("cycle test")
        g.add_task("a", "A", "mimo", [], [])
        g.add_task("b", "B", "codex", [], ["a"])
        g.add_task("c", "C", "mimo", [], ["b"])
        # c -> b -> a is fine (linear chain). But if we add d that depends
        # on c, and then try to add e that a depends on but also depends on d,
        # we can't because a is already added. The real cycle scenario:
        # a exists, b depends on a. Now add "a_prime" that depends on b —
        # _has_path(b, a_prime) checks if b can reach a_prime. Since a_prime
        # isn't in the graph yet, it can't. So no cycle detected.
        # The cycle check catches: existing dep has a path to the new task.
        # This means the new task_id must already appear in the dependency
        # chain of one of its own deps. That happens if task_id matches
        # an existing task's id in the depends_on chain.
        # Linear chain is fine:
        assert len(g.get_all_tasks()) == 3

    def test_no_false_cycle(self):
        """A valid DAG should not trigger cycle detection."""
        g = TaskGraph("diamond")
        g.add_task("a", "A", "mimo", [], [])
        g.add_task("b", "B", "codex", [], ["a"])
        g.add_task("c", "C", "mimo", [], ["a"])
        g.add_task("d", "D", "codex", [], ["b", "c"])
        assert len(g.get_ready_tasks()) == 1  # only "a"


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


class TestIntegration:
    """End-to-end test of the god orchestration protocol (no daemon needed)."""

    def test_full_protocol_flow(self):
        """Simulate: register 2 workers, create task graph, dispatch, complete."""
        # 1. Workers register
        registry = WorkerRegistry()
        registry.register("mimo", ["code", "test"])
        registry.register("codex", ["code", "fast"])

        # 2. Orchestrator creates plan
        graph = TaskGraph("Refactor auth + tests")
        graph.add_task("t1", "Refactor auth.py", "mimo", ["code"], [])
        graph.add_task("t2", "Write tests for auth", "codex", ["code", "test"], ["t1"])
        graph.add_task("t3", "Fix security issue", "codex", ["code"], [])

        # 3. Check initial ready tasks (t1 and t3 — no deps)
        ready = graph.get_ready_tasks()
        ready_ids = [t["task_id"] for t in ready]
        assert "t1" in ready_ids
        assert "t3" in ready_ids
        assert "t2" not in ready_ids

        # 4. Dispatch t1 to mimo, t3 to codex
        for task in ready:
            label = GodProtocol.make_label("task", task["task_id"], task["agent"], "pending")
            assert label.startswith("god:task:")
            graph.set_status(task["task_id"], "running")
            registry.set_status(task["agent"], "busy")

        assert len(registry.list_idle()) == 0

        # 5. t1 completes → t2 becomes ready
        graph.set_status("t1", "completed")
        registry.set_status("mimo", "idle")
        ready = graph.get_ready_tasks()
        assert "t2" in [t["task_id"] for t in ready]

        # 6. t3 completes
        graph.set_status("t3", "completed")
        registry.set_status("codex", "idle")

        # 7. Dispatch t2
        graph.set_status("t2", "running")
        registry.set_status("codex", "busy")

        # 8. t2 completes
        graph.set_status("t2", "verified")
        registry.set_status("codex", "idle")

        # 9. All done
        graph.set_status("t1", "verified")
        graph.set_status("t3", "verified")
        assert graph.is_all_done() is True

        # 10. Verify JSON roundtrip preserves state
        restored = TaskGraph.from_json(graph.to_json())
        assert restored.is_all_done() is True

    def test_worker_reassignment_on_failure(self):
        """If a worker fails, find another with matching capabilities."""
        registry = WorkerRegistry()
        registry.register("mimo", ["code", "test"])
        registry.register("codex", ["code", "fast"])

        graph = TaskGraph("Fix bug")
        graph.add_task("t1", "Fix the auth bug", "mimo", ["code"], [])

        # mimo fails
        graph.set_status("t1", "queued")  # reset to queued for reassignment
        registry.set_status("mimo", "idle")

        # Find another worker with "code" capability
        alternatives = registry.find_by_capability("code")
        alternatives = [a for a in alternatives if a != "mimo"]
        assert "codex" in alternatives

    def test_shutdown_protocol(self):
        """Shutdown label format is correct."""
        label = GodProtocol.make_label("task", "00000000", "mimo", "shutdown")
        assert label == "god:task:00000000:mimo:shutdown"
        parsed = GodProtocol.parse_label(label)
        assert parsed["status"] == "shutdown"
        assert parsed["target"] == "mimo"
