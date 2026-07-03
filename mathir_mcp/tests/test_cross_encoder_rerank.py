"""Tests for CrossEncoderReranker in mathir_search.py."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_MCP_ROOT = _HERE.parent
if str(_MCP_ROOT) not in sys.path:
    sys.path.insert(0, str(_MCP_ROOT))
if str(_MCP_ROOT / "mathir_lib") not in sys.path:
    sys.path.insert(0, str(_MCP_ROOT / "mathir_lib"))

from mathir_lib.mathir_search import CrossEncoderReranker


@pytest.fixture(scope="module")
def reranker():
    return CrossEncoderReranker()


def test_rerank_empty(reranker):
    assert reranker.rerank("test query", []) == []


def test_rerank_reorders(reranker):
    candidates = [
        {"text": "The capital of France is Paris.", "memory_id": "a"},
        {"text": "Python is a programming language.", "memory_id": "b"},
        {"text": "What is the capital city of France?", "memory_id": "c"},
    ]
    results = reranker.rerank("capital of France", candidates)
    assert len(results) == 3
    assert all("rerank_score" in r for r in results)
    assert results[0]["memory_id"] in ("a", "c")


def test_rerank_top_k(reranker):
    candidates = [
        {"text": f"Document {i}", "memory_id": f"doc_{i}"}
        for i in range(10)
    ]
    results = reranker.rerank("test", candidates, top_k=3)
    assert len(results) == 3


def test_rerank_preserves_fields(reranker):
    candidates = [
        {"text": "hello world", "memory_id": "x", "agent": "test", "tier": "semantic"},
    ]
    results = reranker.rerank("hello", candidates)
    assert results[0]["agent"] == "test"
    assert results[0]["tier"] == "semantic"
    assert results[0]["memory_id"] == "x"
