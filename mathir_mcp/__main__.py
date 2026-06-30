"""Allow ``python -m mathir_mcp`` to start the MATHIR daemon.

This is a convenience entrypoint so that after installing the package with
``pip install -e .`` (or via the wheel), users can launch the daemon with
``python -m mathir_mcp`` from any working directory.

Special flags:
  --selftest      Run a 9-step validation (Python, deps, embedder, DB, tools, daemon, e2e)
  --list-tools    List all 23 MCP tools exposed by the server
  --version       Print version and exit
"""
import sys
from pathlib import Path

# Ensure the real mathir_lib/ (inside this package) is importable, not the
# legacy top-level mathir_lib/ shim that may exist in the source tree.
_PKG_ROOT = Path(__file__).parent.resolve()
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))


def _import(module_name):
    """Import module from mathir_mcp.mathir_lib or mathir_lib (fallback)."""
    try:
        import importlib
        return importlib.import_module(f"mathir_mcp.mathir_lib.{module_name}")
    except ImportError:
        import importlib
        return importlib.import_module(f"mathir_lib.{module_name}")


def _cmd_selftest():
    """Run a 9-step validation of the install. Exits 0 on success, 1 on fail."""
    import sys as _sys
    import tempfile
    import time
    import traceback
    from pathlib import Path as _P

    def _robust_cleanup(db_path: "_P", attempts: int = 5, delay: float = 0.2) -> bool:
        """Unlink db file + WAL/SHM sidecars with retries for Windows file locks.

        On Windows, an sqlite-vec connection close + unlink can race with the
        OS releasing the file handle, leaving the next open() to fail with
        WinError 32 (ERROR_SHARING_VIOLATION). Retry the unlink a few times;
        if it still fails, rename the stale file to a backup so the next
        VecMemory() opens a clean path.
        """
        suffixes = ("", "-wal", "-shm", "-journal")
        for i in range(attempts):
            failed = False
            for sfx in suffixes:
                p = _P(str(db_path) + sfx)
                if not p.exists():
                    continue
                try:
                    p.unlink()
                except (OSError, PermissionError):
                    failed = True
                    break
            if not failed:
                return True
            time.sleep(delay)
        # All retries failed — rename the stale file aside so the test can
        # still proceed with a fresh path.
        try:
            backup = _P(str(db_path) + f".stale.{int(time.time())}")
            db_path.rename(backup)
            for sfx in ("-wal", "-shm", "-journal"):
                p = _P(str(db_path) + sfx)
                if p.exists():
                    try:
                        p.rename(_P(str(backup) + sfx))
                    except OSError:
                        pass
            return True
        except OSError:
            return False

    results = []
    def _check(name, fn):
        try:
            msg = fn() or ""
            results.append((True, name, msg))
            print(f"  [OK]   {name}{(' — ' + msg) if msg else ''}")
        except Exception as e:
            results.append((False, name, str(e)[:200]))
            print(f"  [FAIL] {name}: {str(e)[:200]}")

    print("=" * 60)
    print("MATHIR self-test (v8.5.0)")
    print("=" * 60)
    print()

    def _python():
        v = _sys.version_info
        if v < (3, 10):
            raise RuntimeError(f"Python {v.major}.{v.minor} < 3.10")
        return f"{v.major}.{v.minor}.{v.micro}"
    _check("Python >= 3.10", _python)

    def _torch():
        import torch
        cuda = torch.cuda.is_available()
        return f"CUDA={cuda}, version={torch.__version__}"
    _check("PyTorch installed", _torch)

    def _st():
        import sentence_transformers
        return f"v{sentence_transformers.__version__}"
    _check("sentence-transformers installed", _st)

    def _sqlite_vec():
        import sqlite_vec
        return "ok"
    _check("sqlite-vec installed", _sqlite_vec)

    def _db_init():
        mathir_vec = _import("mathir_vec")
        VecMemory = mathir_vec.VecMemory
        # Use a stable path in cwd to avoid Windows file lock issues
        # with temp dir cleanup racing against sqlite-vec handle release
        test_dir = _P(".mathir_selftest")
        test_dir.mkdir(exist_ok=True)
        db = test_dir / "selftest.db"
        # Robust cleanup: unlink with retries, rename if still locked
        _robust_cleanup(db)
        m = VecMemory(db, 384)  # VecMemory needs a Path, not a string
        n = m.count()
        m.close()
        # Best-effort cleanup
        _robust_cleanup(db)
        try:
            test_dir.rmdir()
        except Exception:
            pass
        return f"count={n}"
    _check("DB initialization works", _db_init)

    def _embedder():
        mathir_daemon = _import("mathir_daemon")
        get_embedder = mathir_daemon.get_embedder
        e = get_embedder()
        v = e.encode("test")
        if len(v) not in (384,):
            raise RuntimeError(f"unexpected dim: {len(v)}")
        return f"dim={len(v)}"
    _check("Embedder loads (384d)", _embedder)

    def _tools():
        mathir_mcp_server = _import("mathir_mcp_server")
        TOOLS = mathir_mcp_server.TOOLS
        n = len(TOOLS)
        if n != 23:
            raise RuntimeError(f"expected 23 tools, got {n}")
        return f"{n} tools registered"
    _check("All 23 tools registered", _tools)

    def _daemon():
        import os, urllib.request, json as _json
        port = int(os.environ.get("MATHIR_PORT", "7338"))
        try:
            url = f"http://127.0.0.1:{port}/api/ping"
            with urllib.request.urlopen(url, timeout=2) as r:
                data = _json.loads(r.read().decode("utf-8", "ignore"))
            if data.get("pong"):
                return f"port={port}"
            raise RuntimeError(f"no pong in response: {str(data)[:100]}")
        except Exception as e:
            raise RuntimeError(f"daemon not reachable on port {port}: {e}")
    print("  [..]   Daemon reachable on port 7338 (may not be running, this is OK)")
    try:
        result = _daemon()
        print(f"  [OK]   Daemon reachable — {result}")
        results.append((True, "Daemon reachable", result))
    except Exception as e:
        msg = str(e)
        # Daemon-not-running is non-fatal — user can start it later
        if "not reachable" in msg or "Connection refused" in msg:
            print(f"  [WARN] Daemon not running on port (start with: python -m mathir_mcp): {msg[:100]}")
            results.append((True, "Daemon (not running — start with python -m mathir_mcp)", ""))
        else:
            print(f"  [FAIL] Daemon reachable: {msg[:200]}")
            results.append((False, "Daemon reachable", msg[:200]))

    def _e2e():
        mathir_vec = _import("mathir_vec")
        mathir_daemon = _import("mathir_daemon")
        VecMemory = mathir_vec.VecMemory
        get_embedder = mathir_daemon.get_embedder
        _embedding_to_numpy = mathir_daemon._embedding_to_numpy
        # Stable path to avoid Windows file lock race in temp dir cleanup
        test_dir = _P(".mathir_selftest")
        test_dir.mkdir(exist_ok=True)
        db = test_dir / "e2e.db"
        # Robust cleanup: unlink with retries, rename if still locked.
        # Without this, a previous interrupted run can leave e2e.db in a
        # sharing-violation state (WinError 32) and break this test.
        _robust_cleanup(db)
        m = VecMemory(db, 384)  # VecMemory needs a Path, not a string
        e = get_embedder()
        emb = _embedding_to_numpy(e.encode("hello world test"))
        m.store("mem_e2e", emb, {"agent": "selftest", "block_type": "episodic",
                                   "label": "e2e", "priority": 5, "content": "hello world test"})
        results = m.search(query_embedding=emb, k=1)
        m.close()
        _robust_cleanup(db)
        try:
            test_dir.rmdir()
        except Exception:
            pass
        if not results or results[0]["memory_id"] != "mem_e2e":
            raise RuntimeError("e2e recall failed")
        return "save+recall verified"
    _check("End-to-end: save+recall", _e2e)

    print()
    n_ok = sum(1 for r in results if r[0])
    n_fail = sum(1 for r in results if not r[0])
    print("=" * 60)
    print(f"Result: {n_ok} passed, {n_fail} failed")
    print("=" * 60)
    return 0 if n_fail == 0 else 1


def _cmd_list_tools():
    """Print the list of all MCP tools exposed by the server."""
    mathir_mcp_server = _import("mathir_mcp_server")
    TOOLS = mathir_mcp_server.TOOLS
    # Use ASCII-only output for Windows console compatibility
    print(f"MATHIR exposes {len(TOOLS)} MCP tools (v8.5.0):")
    print()
    for i, t in enumerate(TOOLS, 1):
        name = t.get("name", "?")
        desc = t.get("description", "").strip().split("\n")[0]  # first line
        # Sanitize to ASCII for Windows console
        desc = desc.encode("ascii", "replace").decode("ascii")
        if len(desc) > 80:
            desc = desc[:77] + "..."
        print(f"  {i:>2}. {name:<25} {desc}")
    return 0


def _cmd_version():
    """Print version and exit."""
    try:
        # Try to read from pyproject.toml
        from pathlib import Path
        import re
        pyproject = Path(__file__).parent / "pyproject.toml"
        if pyproject.is_file():
            m = re.search(r'version\s*=\s*"([^"]+)"', pyproject.read_text())
            if m:
                print(f"MATHIR v{m.group(1)}")
                return 0
    except Exception:
        pass
    print("MATHIR v8.5.0")
    return 0


def main():
    # Parse args (simple: first non-flag arg = subcommand)
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "--selftest":
            return _cmd_selftest()
        elif arg == "--list-tools":
            return _cmd_list_tools()
        elif arg in ("--version", "-V"):
            return _cmd_version()
        elif arg in ("--help", "-h"):
            print(__doc__)
            return 0
        elif arg == "--mcp":
            # Start the MCP server (stdio JSON-RPC) — used by OpenCode agent
            mathir_mcp_server = _import("mathir_mcp_server")
            return mathir_mcp_server.main()
        elif arg == "update" or arg == "check" or arg == "rollback":
            # Self-updater subcommands. Delegated to mathir_updater.main()
            # which has its own argparse with subcommands.
            updater = _import("mathir_updater")
            # Re-prepend the subcommand name since mathir_updater.main expects
            # it as its first positional arg.
            return updater.main([arg] + sys.argv[2:])

    # No special flag -> start the unified HTTP server on MATHIR_PORT.
    # As of v8.5.0 the TCP daemon (mathir_daemon.py) is retired in favour of
    # mathir_server.py (Flask + Waitress). All MCP clients speak HTTP.
    mathir_server = _import("mathir_server")
    return mathir_server.main()


if __name__ == "__main__":
    sys.exit(main())
