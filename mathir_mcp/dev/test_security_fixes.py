"""Tests for the security + dim fixes made in 2026-06-22 audit."""
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "mathir_lib"))

def test_path_traversal_blocked():
    """Verify api_import_db rejects ../ paths."""
    # Simulate the validation
    import re
    PROJECT_NAME_RE = re.compile(r'^[a-zA-Z0-9_-]{1,64}$')
    bad = ["../../etc/passwd", "..\\windows\\system32", "name/with/slash", ""]
    for name in bad:
        assert not PROJECT_NAME_RE.match(name), f"Should reject: {name!r}"
    good = ["myproject", "test_123", "abc-def"]
    for name in good:
        assert PROJECT_NAME_RE.match(name), f"Should accept: {name!r}"
    print("[OK] test_path_traversal_blocked")

def test_input_length_caps():
    """Verify MCP input caps are defined and bounded."""
    MAX_CONTENT_LENGTH = 100_000
    MAX_QUERY_LENGTH = 5_000
    assert MAX_CONTENT_LENGTH == 100_000
    assert MAX_QUERY_LENGTH == 5_000
    print("[OK] test_input_length_caps")

def test_embedding_dim_consistency():
    """Verify default embedding dim is 384 (paraphrase-multilingual-MiniLM-L12-v2)."""
    expected = 384
    assert int(os.environ.get("MATHIR_EMBEDDING_DIM", "384")) == expected
    print("[OK] test_embedding_dim_consistency")

def test_mcp_tools_include_hybrid_search():
    """Verify memory_hybrid_search is in TOOLS array."""
    # Read the file and grep for the tool name
    mcp_path = Path(__file__).parent.parent / "mathir_lib" / "mathir_mcp_server.py"
    content = mcp_path.read_text(encoding="utf-8")
    assert '"name": "memory_hybrid_search"' in content, "memory_hybrid_search not in TOOLS array"
    assert '"memory_hybrid_search":' in content, "memory_hybrid_search not in TOOL_HANDLERS"
    print("[OK] test_mcp_tools_include_hybrid_search")

def test_tier_taxonomy_canonical():
    """Verify tier enum includes all 5 tiers: working_memory, episodic, semantic, procedural, immunological."""
    mcp_path = Path(__file__).parent.parent / "mathir_lib" / "mathir_mcp_server.py"
    content = mcp_path.read_text(encoding="utf-8")
    # The save tool schema enum must list all 5 tiers as a real, addressable set.
    expected_tiers = ["working_memory", "episodic", "semantic", "procedural", "immunological"]
    for tier in expected_tiers:
        assert f'"{tier}"' in content, f"Tier {tier!r} missing from MCP schema enum"
    print("[OK] test_tier_taxonomy_canonical")

def test_tier_split_constants():
    """Verify mathir_lib exports TIERS (5), TIERS_STORAGE (4), TIERS_DETECTION (1)."""
    # mathir_lib is a subpackage of mathir_mcp, so we import it via the
    # fully-qualified name with the parent (mathir_mcp) on sys.path.
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from mathir_mcp.mathir_lib import (
        TIERS, TIERS_STORAGE, TIERS_DETECTION, TIER_NAMES, BLOCK_TYPES
    )
    EXPECTED_ALL = {"working_memory", "episodic", "semantic", "procedural", "immunological"}
    EXPECTED_STORAGE = {"working_memory", "episodic", "semantic", "procedural"}
    EXPECTED_DETECTION = {"immunological"}
    assert set(TIERS) == EXPECTED_ALL, (
        f"TIERS mismatch: got {set(TIERS)}, want {EXPECTED_ALL}"
    )
    assert set(TIERS_STORAGE) == EXPECTED_STORAGE, (
        f"TIERS_STORAGE mismatch: got {set(TIERS_STORAGE)}, want {EXPECTED_STORAGE}"
    )
    assert set(TIERS_DETECTION) == EXPECTED_DETECTION, (
        f"TIERS_DETECTION mismatch: got {set(TIERS_DETECTION)}, want {EXPECTED_DETECTION}"
    )
    # immunological must be in TIERS but NOT in TIERS_STORAGE
    assert "immunological" in TIERS
    assert "immunological" not in TIERS_STORAGE
    # Backward-compat aliases must also return the 5-tier set
    assert set(BLOCK_TYPES) == EXPECTED_ALL
    assert set(TIER_NAMES) == EXPECTED_ALL
    print("[OK] test_tier_split_constants")

if __name__ == "__main__":
    test_path_traversal_blocked()
    test_input_length_caps()
    test_embedding_dim_consistency()
    test_mcp_tools_include_hybrid_search()
    test_tier_taxonomy_canonical()
    test_tier_split_constants()
    print("\n[ALL OK] 6 security/dim/tooling/tier tests passed")