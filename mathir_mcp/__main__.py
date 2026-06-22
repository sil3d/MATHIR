"""Allow ``python -m mathir_mcp`` to start the MATHIR daemon.

This is a convenience entrypoint so that after installing the package with
``pip install -e .`` (or via the wheel), users can launch the daemon with
``python -m mathir_mcp`` from any working directory.
"""
import sys
from pathlib import Path

# Ensure the real mathir_lib/ (inside this package) is importable, not the
# legacy top-level mathir_lib/ shim that may exist in the source tree.
_PKG_ROOT = Path(__file__).parent.resolve()
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

# Now we can import the real mathir_lib package.
from mathir_lib.mathir_daemon import main

if __name__ == "__main__":
    sys.exit(main())