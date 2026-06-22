# mathir-raspberry-jetson

**MATHIR Memory for Raspberry Pi / NVIDIA Jetson — CPU-only portable subset.**

This is a thin deployment wrapper around the portable
[`mathir-mcp`](../mcp/) package. It does NOT ship a parallel copy of the
codebase; it just sets CPU/ONNX-friendly defaults and installs the right
extras for resource-constrained boards.

## Install

```bash
# From the project root, one-time setup
./setup.sh
```

This will:
1. Create a Python venv at `./.venv/`
2. Install the portable `mathir-mcp` package from `../mcp/` (editable mode)
3. Install ARM-friendly extras (`onnxruntime`, `numpy`)

## Start

```bash
./start.sh
```

Forces the following defaults (overridable via env vars):

| Var | Default | Why |
|---|---|---|
| `MATHIR_EMBEDDING_MODEL` | `Xenova/all-MiniLM-L6-v2-onnx` | ONNX quantised, no PyTorch |
| `MATHIR_EMBEDDING_DIM` | `384` | Matches the ONNX model output |
| `MATHIR_DEVICE` | `cpu` | No CUDA on Pi/Jetson by default |
| `MATHIR_USE_ONNX` | `1` | Bypass sentence-transformers, use onnxruntime |
| `MATHIR_PORT` | `7338` | Same as desktop |

Then it `exec`s `python3 -m mathir_mcp` (which starts the daemon).

## Verify

```bash
python -c "import mathir_lib; print(mathir_lib.__version__)"   # should print 8.3.0
python -c "import raspberry_jetson; print(raspberry_jetson.__version__)"  # 8.3.0
```

## Why this package exists (vs running `mcp/` directly)

The `mcp/` package works on Pi/Jetson out of the box, but its defaults
target desktop GPUs (CUDA, larger model, larger memory footprint). This
package pre-configures CPU/ONNX so a Pi 4 / Jetson Nano doesn't OOM.

## Data location

Override with `MATHIR_DATA_DIR`. Default on Linux: `/var/lib/mathir` (if
writable) or `~/.local/share/mathir`. On Windows (for testing):
`~/.local/share/mathir`.

## Files in this package

| File | Purpose |
|---|---|
| `__init__.py` | Sets Jetson env defaults before daemon imports embedder |
| `__main__.py` | `python -m raspberry_jetson` — prints config + version |
| `pyproject.toml` | `pip install -e .` metadata, depends on `mathir-mcp` |
| `setup.sh` | venv + `pip install -e ../mcp` + ARM extras |
| `start.sh` | Sets env, `exec python3 -m mathir_mcp` |
| `requirements.txt` | Pi/Jetson-specific pip extras |

## What is NOT here (vs old structure)

This package used to be a 41-file carbon copy of `mcp/`. As of V8.3.0
(2026-06-22 audit cleanup), all duplicate code was removed. To update
behavior, edit `../mcp/` — this package just re-exports it.
