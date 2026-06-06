# MATHIR — Agent Guide

Quick reference for AI agents working on this codebase.

## Project Overview

MATHIR (Memory-Augmented Tensor Hybrid with Intelligent Routing) is a plug-and-play memory layer for LLMs. It maintains 4 tiers of memory that learn online.

## Architecture

```
mathir_lib/          # Full library (8 algorithms, 6 theorems, torch required)
mathir_dropin/       # Drop-in memory (copy to your project)
  ├── memory.py      # Full MATHIRMemory (torch)
  ├── simple.py      # SimpleMemory (FTS5, no torch) ← USE THIS
  └── store.py       # SQLite storage layer
vision_testing/      # Vision/audio testing UI (Flask + OpenCV)
  ├── ui_server.py   # Flask backend (17 API routes)
  ├── ui/            # Web UI (HTML/CSS/JS)
  └── config.json    # Model configuration
```

## Key Files

| File | Purpose |
|------|---------|
| `mathir_dropin/simple.py` | SimpleMemory — zero-dep memory (FTS5) |
| `mathir_dropin/memory.py` | MATHIRMemory — full 4-tier (torch) |
| `mathir_lib/plugin_v7.py` | V7 plugin with 8 algorithms |
| `vision_testing/ui_server.py` | Flask backend (17 routes) |
| `vision_testing/ui/static/app.js` | Frontend JS |
| `vision_testing/config.json` | Model configuration |
| `vision_testing/system_context.json` | System prompt (~126 tokens) |

## Quick Commands

```bash
# Start vision testing UI
cd vision_testing && python start_ui.py

# Run tests
pytest tests/test_v7_memory.py -v
pytest mathir_dropin/tests/ -v

# Run benchmarks
python benchmarks/compare_all_approaches.py --chunks 200 --queries 50

# SimpleMemory usage
python examples/simple_memory_demo.py
```

## API Routes (Vision Testing)

| Route | Method | Description |
|-------|--------|-------------|
| `/api/system/context` | GET | System context + models |
| `/api/system/info` | GET | System info |
| `/api/models` | GET | List models |
| `/api/models/switch` | POST | Switch model |
| `/api/models/toggle` | POST | Enable/disable model |
| `/api/models/add-from-hf` | POST | Add from HuggingFace |
| `/api/chat` | POST | Send message `{message, image?, audio?}` |
| `/api/camera/start` | POST | Start camera |
| `/api/camera/stop` | POST | Stop camera |
| `/api/camera/frame` | GET | Get frame (JPEG) |
| `/api/camera/stream` | GET | MJPEG stream |
| `/api/camera/ask` | POST | Ask about scene |
| `/api/memory/recall` | POST | Search memory |
| `/api/memory/stats` | GET | Memory stats |
| `/api/accuracy/tests` | GET | List tests |
| `/api/accuracy/results` | GET | Get results |
| `/api/accuracy/test` | POST | Run battery |

## Memory System

- **SimpleMemory** (FTS5): Zero deps, text-only, SQLite
- **MATHIRMemory** (torch): Full 4-tier, embeddings
- **MATHIRPluginV7**: 8 algorithms, 6 theorems

Memory is stored in `memory/vision_test.db` (SQLite). Every chat interaction is stored and recalled for future context.

## Configuration

- `config.json` — Models, paths (no hardcoded)
- `ui_config.json` — Port, camera, audio, theme
- `system_context.json` — System prompt (~126 tokens)

## Version

Current: **v7.7.1**
- SimpleMemory (FTS5, no torch)
- UI overhaul (SVG, chat history, backend camera)
- 17 API routes
- 31/31 memory audit checks
