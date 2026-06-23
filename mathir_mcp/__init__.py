"""MATHIR Memory MCP Server - portable package.

This is the package marker for ``mathir-mcp`` (the distribution name in pyproject.toml).
It exposes the runtime version and the underlying mathir_lib/ sub-package.

Canonical entry points:
    python -m mathir_mcp           # Daemon (TCP, port 7338)
    python -m mathir_mcp --selftest
    python -m mathir_mcp --list-tools
    python -m mathir_mcp --version
    python -m mathir_mcp.mathir_lib.mathir_daemon
    python -m mathir_mcp.mathir_lib.mathir_mcp_server

Console scripts (after ``pip install -e .``):
    mathir-daemon, mathir-mcp, mathir-client, mathir-watchdog
"""
__version__ = "8.4.0"

__all__ = ["__version__"]