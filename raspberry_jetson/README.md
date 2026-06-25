# mathir-raspberry-jetson

**MATHIR Memory for Raspberry Pi / NVIDIA Jetson — CPU-only portable subset (v8.5.0).**

This is a thin deployment wrapper around the portable
[`mathir-mcp`](../mathir_mcp/) package. It does NOT ship a parallel copy of the
codebase; it just sets CPU-friendly defaults and installs the right
extras for resource-constrained boards.

## Install

```bash
# From the raspberry_jetson/ directory, one-time setup
./setup.sh
```

This will:
1. Create a Python venv at `./.venv/`
2. Install the portable `mathir-mcp` package from `../mathir_mcp/` (editable mode)
3. Install ARM-friendly extras (`onnxruntime`, `numpy`)

## Start

```bash
./start.sh
```

Forces the following defaults (overridable via env vars):

| Var | Default | Why |
|---|---|---|
| `MATHIR_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Smallest MiniLM, ~50 MB RAM |
| `MATHIR_EMBEDDING_DIM` | `384` | Matches the MiniLM-L6 output |
| `MATHIR_DEVICE` | `cpu` | No CUDA on Pi/Jetson by default |
| `MATHIR_PORT` | `7338` | Same as desktop |

Then it `exec`s `python3 -m mathir_mcp` (which starts the daemon).

## Verify

```bash
python3 -m mathir_mcp --selftest            # 9/9 should pass
python3 -m mathir_mcp --version             # MATHIR v8.5.0
python3 -c "from mathir_mcp.mathir_lib import __version__; print(__version__)"
```

## Run benchmarks to validate on Pi

The lifecycle bench scripts live in `benchmarks/04_lifecycle_bench/` and
have been tested on RTX 4060 + GPU. To reproduce the **+52.3% recall@5**
claim on your Pi (CPU-only):

```bash
# 1) Start the daemon in another terminal (./start.sh)
# 2) Run the micro bench (no LLM, just memory operations — fast on Pi)
python3 benchmarks/04_lifecycle_bench/micro_bench.py \
    --count 500 --duplicates 100 --out pi_micro_bench.json

# 3) Optionally run the AI cognitive bench if you have an LLM endpoint
#    reachable from the Pi (e.g., Ollama on your desktop):
python3 benchmarks/04_lifecycle_bench/ai_cognitive_bench.py \
    --experiences 50 --questions 10 --duration 30 \
    --out pi_ai_bench.json
```

Render the HTML report:
```bash
python3 benchmarks/04_lifecycle_bench/render_report.py pi_micro_bench.json
```

## Why this package exists (vs running `mathir_mcp/` directly)

The `mathir_mcp/` package works on Pi/Jetson out of the box, but its defaults
target desktop GPUs (CUDA, larger model, larger memory footprint). This
package pre-configures CPU + the smallest MiniLM so a Pi 4 / Jetson Nano doesn't OOM.

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
| `setup.sh` | venv + `pip install -e ../mathir_mcp` + ARM extras |
| `start.sh` | Sets env, `exec python3 -m mathir_mcp` |
| `requirements.txt` | Pi/Jetson-specific pip extras |

## What is NOT here (vs old structure)

This package used to be a 41-file carbon copy of `mcp/`. As of V8.3.0
(2026-06-22 audit cleanup), all duplicate code was removed. As of v8.5.0
(2026-06-23), `__init__.py` was updated to use the nested
`from mathir_mcp.mathir_lib import __version__` form (top-level
`mathir_lib` is no longer importable after the package restructure).
To update behavior, edit `../mathir_mcp/mathir_lib/` — this package just re-exports it.
