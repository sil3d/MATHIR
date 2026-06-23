# `_deprecated/legacy_root/` — Root-level duplicates removed in v8.4.0

Four files were sitting at the repo root (`D:\SECRET_PROJECT\MATHIR\mathir_*.py`) and
confusing import resolution. They were moved here in v8.4.0.

## Files

| File | Original size | Status |
|---|---|---|
| `mathir_search.py` | 19,830 B | **Bit-identical** to `mathir_mcp/mathir_lib/mathir_search.py` (same SHA256). Pure duplicate. |
| `mathir_vec.py` | 7,199 B (15/06/2026) | **Stale 8 days** vs canonical `mathir_mcp/mathir_lib/mathir_vec.py` (77,599 B, 23/06/2026). 10× smaller — was a v7 stub. |
| `mathir_vec_optimized.py` | 9,470 B (15/06/2026) | No importers found anywhere. Optimizations were folded into `VecMemory` directly. |
| `mathir_gpu_vec.py` | 11,105 B (15/06/2026) | No importers found. GPU path is auto-detected via `torch.cuda.is_available()` in `mathir_daemon`. |

## Migration pattern (canonical example)

If you have a script that did:

```python
sys.path.insert(0, r"D:\SECRET_PROJECT\MATHIR")
from mathir_vec import VecMemory
```

Replace with the portable form:

```python
from pathlib import Path
_PKG_ROOT = Path(__file__).resolve().parent.parent.parent / "mathir_mcp"
_LIB = _PKG_ROOT / "mathir_lib"
sys.path.insert(0, str(_PKG_ROOT))
sys.path.insert(0, str(_LIB))
from mathir_vec import VecMemory
```

`benchmarks/03_vector_search_benchmarks/benchmark_beir.py` is the canonical migration example.

## Why kept here (not deleted)

User-mandated decision (2026-06-23): the deprecated files are part of the project's
narrative. `git log --follow <file>` still shows history. They can be safely deleted
in a future release if the maintenance cost outweighs the storytelling value.