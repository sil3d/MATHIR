"""Tests for MATHIR God Orchestrator (mathir_god.py)."""

import sys
import os
import json

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
