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
