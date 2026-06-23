# MATHIR Playground

A web playground to chat with vision/audio LLMs (via OpenRouter) and exercise the
MATHIR memory backend (via the running daemon). **NO HARDCODED PATHS** — everything
is configurable via JSON files.

**v8.4.0**: Migrated from local `llama.cpp` binaries (~1GB) to **OpenRouter** cloud API. No more `bin/` or `models/` directories — all inference goes through `https://openrouter.ai/api/v1`.

## Quick Start

```bash
cd vision_testing
pip install -r requirements.txt

# Set your OpenRouter key (get one at https://openrouter.ai/keys):
echo 'export OPENROUTER_API_KEY=sk-or-v1-...' >> ~/.bashrc
source ~/.bashrc

python start_ui.py
# Opens at http://127.0.0.1:5000
```

## Architecture

```
vision_testing/
├── config.json                # OpenRouter config + models (id-based, no paths)
├── ui_config.json             # UI settings (port, camera, audio)
├── system_context.json        # System prompt for the model
│
├── ui/                        # Web UI (SVG icons, dark theme)
│   ├── index.html             # Single-page app (6 views)
│   └── static/
│       ├── style.css          # Dark theme, 8px grid
│       └── app.js             # Chat, Camera, Models, Memory, Accuracy, Settings
│
├── ui_server.py               # Flask backend (17 API routes)
├── start_ui.py                # Launcher (installs deps, runs server)
├── vision_test.py             # Core: VisionTester, OpenRouterClient, ModelManager
├── mathir_dropin.py           # MATHIR memory (SQLite FTS5)
│
└── memory/                    # SQLite memory database
```

No `bin/`, no `models/` — all inference is cloud-based.

## MATHIR Memory System

MATHIR provides persistent memory across sessions using SQLite FTS5:

- **Store**: Every chat interaction is stored with metadata (model, timestamp, query type)
- **Recall**: Before each response, relevant memories are retrieved via FTS5 full-text search
- **Get Last**: The last 3 memories are always included for context
- **Cross-model**: All models share the same `memory/vision_test.db`

### How it works

1. User sends message → `universal_recall()` finds relevant memories
2. Last 3 memories are appended for recent context
3. Memories are injected into the system prompt
4. Model responds with awareness of past observations
5. Response is stored in memory for future recall

### Files

- `mathir_dropin.py` — MATHIRMemory class (FTS5, no external deps)
- `memory/vision_test.db` — SQLite database (auto-created)

## UI Features (6 views)

### 1. Chat
- Real-time chat with active model
- **Talk button** (push-to-talk) for voice input
- Attach images for vision models
- Chat history persisted in localStorage
- Camera preview strip (when camera is active)

### 2. Camera
- Live webcam feed via backend OpenCV
- **Describe** — model describes what it sees
- **Ask** — type a question about the scene
- **Talk** — voice input while camera is active
- **Snapshot** — save current frame

### 3. Models
- List all configured OpenRouter models
- Switch active model at runtime
- Enable/disable models
- Free models highlighted

### 4. Memory
- Query MATHIR memory (FTS5 search)
- View stored interactions
- Memory stats

### 5. Accuracy
- Run accuracy test battery
- Compare models side-by-side
- Per-model detail view
- Test catalog

### 6. Settings
- Camera device, resolution, FPS
- Audio settings (max record, push-to-talk key)
- UI theme
- System info
- Keyboard shortcuts

## Configuration

### `config.json` — OpenRouter + models

```json
{
  "openrouter": {
    "api_key": "",          // or set OPENROUTER_API_KEY env var
    "api_base": "https://openrouter.ai/api/v1",
    "timeout_seconds": 120,
    "max_retries": 2
  },
  "models": {
    "google/gemini-2.0-flash-exp:free": {
      "enabled": true,
      "type": "vision-language",
      "id": "google/gemini-2.0-flash-exp:free",
      "display_name": "Gemini 2.0 Flash (free)",
      "supports_vision": true
    }
  }
}
```

### `ui_config.json` — UI settings

```json
{
  "server": { "host": "127.0.0.1", "port": 5000 },
  "camera": { "device_id": 0, "width": 1280, "height": 720, "fps": 30 },
  "audio": { "push_to_talk_key": "Space", "max_record_seconds": 30 }
}
```

### `system_context.json` — Model behavior

Compact system prompt (~126 tokens). Defines identity, behavior rules, and context template.

## API Routes (17)

| Route | Method | Description |
|-------|--------|-------------|
| `/api/system/context` | GET | System context + available models |
| `/api/system/info` | GET | System info (platform, paths) |
| `/api/models` | GET | List all models |
| `/api/models/switch` | POST | Switch active model |
| `/api/models/toggle` | POST | Enable/disable model |
| `/api/models/add-from-or` | POST | Add model from OpenRouter ID |
| `/api/chat` | POST | Send chat message (with optional image/audio) |
| `/api/camera/start` | POST | Start backend camera |
| `/api/camera/stop` | POST | Stop backend camera |
| `/api/camera/frame` | GET | Get current frame (JPEG) |
| `/api/camera/stream` | GET | MJPEG stream |
| `/api/camera/ask` | POST | Ask about camera scene |
| `/api/memory/recall` | POST | Search MATHIR memory |
| `/api/memory/stats` | GET | Memory statistics |
| `/api/accuracy/tests` | GET | List accuracy tests |
| `/api/accuracy/results` | GET | Get accuracy results |
| `/api/accuracy/test` | POST | Run accuracy battery |

## Adding Models

**Via UI**: Models → Add from OpenRouter → paste model ID (e.g., `openai/gpt-4o-mini`)

**Via config.json**: Edit directly (see above). Find model IDs at https://openrouter.ai/models

**Free models filter**: https://openrouter.ai/models?max_price=0

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `1`–`6` | Switch view (Chat, Camera, Models, Memory, Accuracy, Settings) |
| `Enter` | Send chat message |
| `Shift+Enter` | Newline in chat |
| `Space` | Hold to record audio |
| `Ctrl+K` | Focus chat input |
| `Esc` | Close modal / clear focus |

## Hardware

- **No local GPU required** — all inference is cloud-based via OpenRouter
- Internet connection required for chat/describe/ask endpoints
- Webcam + microphone required for Camera view

## What was removed in v8.4.0

| Removed | Replacement |
|---|---|
| `bin/` (~1GB llama.cpp + CUDA DLLs) | OpenRouter cloud API |
| `models/` (5 GGUF model dirs) | OpenRouter model IDs in `config.json` |
| `convert_lfm2_to_gguf.py` | n/a (no GGUF conversion needed) |
| `download_models.py` | n/a (no local downloads) |
| `download_q4.py` | n/a (no local quantisation) |
| `setup_binaries.py` | n/a (no binaries to setup) |
| `interface/LlamaSetupModal_reference.jsx` | `interface/OpenRouterSetupModal_reference.jsx` |
| `interface/wizardModels_llamacpp_reference.json` | `interface/wizardModels_openrouter_reference.json` |
| `LlamaServer` class in `vision_test.py` | `OpenRouterClient` class |
| `config.json:llama_server` section | `config.json:openrouter` section |

## License

MATHIR: MIT | OpenRouter: see https://openrouter.ai | OpenCV: Apache 2.0 | Flask: BSD-3