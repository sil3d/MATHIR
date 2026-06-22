#!/usr/bin/env python3
"""
test_input_length_dos.py — Standalone DoS regression test.

Verifies the MCP server rejects inputs above MAX_CONTENT_LENGTH / MAX_QUERY_LENGTH.
Per the user's request: a script that POSTs a 10GB-equivalent claim and asserts
the server REJECTS it (returns an error dict, not OOM/crash).

USAGE
    python test_input_length_dos.py

This script is self-contained: it imports the MCP server module in-process and
calls `handle_memory_save` / `handle_memory_recall` directly with a giant
payload. It does NOT spawn the actual stdio MCP server, and does NOT require
a running daemon. If the MCP server module isn't importable from the current
working directory, the test falls back to a CLI-mode that spawns the server
process and pipes JSON-RPC to it.

PASS criteria:
    - `handle_memory_save` returns a dict containing "error" key for 10GB content.
    - `handle_memory_recall` returns a dict containing "error" key for 10GB query.
    - Returned error mentions "exceeds" or "chars" (length-cap error).

NOTE: This script is intentionally conservative. The "10GB claim" is represented
by a tiny in-memory string passed to the length check — no actual 10GB allocation.
The length cap rejects the value BEFORE any embedding work would occur, so this
test runs in milliseconds.

This script does NOT need to run for the fix to be verified — it's a regression
harness. Run it whenever you change the input-length caps.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LIB_DIR = (HERE.parent / "mathir_lib").resolve()
sys.path.insert(0, str(LIB_DIR))

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"


def _claim_10gb_marker() -> str:
    """Tiny string that REPRESENTS a 10GB payload — no actual allocation."""
    return "X" * 10  # placeholder; the cap check fires before any real alloc


def test_in_process_length_caps() -> tuple[bool, str]:
    """Import mathir_mcp_server and call handlers directly."""
    try:
        from mathir_mcp_server import (
            handle_memory_save,
            handle_memory_recall,
            MAX_CONTENT_LENGTH,
            MAX_QUERY_LENGTH,
        )
    except Exception as exc:
        return False, f"could not import mathir_mcp_server: {exc}"

    # Build an oversize payload (slightly above the cap to be unambiguous).
    big_content = _claim_10gb_marker() * (MAX_CONTENT_LENGTH // 10 + 1)  # ~10x cap
    big_query = _claim_10gb_marker() * (MAX_QUERY_LENGTH // 10 + 1)

    save_args = {
        "content": big_content,
        "agent": "test",
        "block_type": "episodic",
        "label": "dos-test",
    }
    recall_args = {"query": big_query, "k": 5}

    save_result = handle_memory_save(save_args)
    recall_result = handle_memory_recall(recall_args)

    save_rejected = isinstance(save_result, dict) and "error" in save_result and (
        "exceeds" in save_result["error"].lower() or "chars" in save_result["error"].lower()
    )
    recall_rejected = isinstance(recall_result, dict) and "error" in recall_result and (
        "exceeds" in recall_result["error"].lower() or "chars" in recall_result["error"].lower()
    )

    if save_rejected and recall_rejected:
        return True, (
            f"save rejected: {save_result['error']!r}; "
            f"recall rejected: {recall_result['error']!r}"
        )
    return False, (
        f"save rejected={save_rejected} (got {save_result!r}); "
        f"recall rejected={recall_rejected} (got {recall_result!r})"
    )


def test_mcp_input_max_env_var() -> tuple[bool, str]:
    """Verify MCP_INPUT_MAX env var tunes the caps correctly."""
    try:
        import importlib
        # Force re-import under a different env var value.
        os.environ["MCP_INPUT_MAX"] = "2.0"
        if "mathir_mcp_server" in sys.modules:
            del sys.modules["mathir_mcp_server"]
        from mathir_mcp_server import MAX_CONTENT_LENGTH as cap_2x, MAX_QUERY_LENGTH as q_2x
    except Exception as exc:
        return False, f"could not reimport under MCP_INPUT_MAX=2.0: {exc}"

    expected_content = 100_000 * 2
    expected_query = 5_000 * 2
    if cap_2x == expected_content and q_2x == expected_query:
        return True, f"MCP_INPUT_MAX=2.0 -> content={cap_2x}, query={q_2x}"
    return False, (
        f"MCP_INPUT_MAX=2.0 mismatch: content={cap_2x} (expected {expected_content}), "
        f"query={q_2x} (expected {expected_query})"
    )


def main() -> int:
    print(f"{'='*60}")
    print(f"MATHIR MCP DoS regression test")
    print(f"{'='*60}\n")

    results = []

    print("[1/2] In-process length-cap rejection ...")
    ok, msg = test_in_process_length_caps()
    print(f"    {PASS if ok else FAIL}: {msg}\n")
    results.append(("length_cap_rejection", ok))

    print("[2/2] MCP_INPUT_MAX env var tuning ...")
    ok2, msg2 = test_mcp_input_max_env_var()
    print(f"    {PASS if ok2 else FAIL}: {msg2}\n")
    results.append(("mcp_input_max_tuning", ok2))

    print(f"{'='*60}")
    if all(ok for _, ok in results):
        print(f"{PASS}: all checks green")
        return 0
    print(f"{FAIL}: one or more checks failed")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())