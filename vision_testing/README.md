# MATHIR Vision Testing

Test environment for vision and audio models with MATHIR memory integration. **NO HARDCODED PATHS** — everything is configurable via JSON files.

## Quick Start

```bash
cd vision_testing
pip install -r requirements.txt
python start_ui.py
# Opens at http://127.0.0.1:5000
```

## Architecture

```
vision_testing/
├── config.json                # Models + paths
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
├── vision_test.py             # Core: VisionTester, LlamaServer, ModelManager
├── mathir_dropin.py           # MATHIR memory (SQLite FTS5)
│
├── models/                    # GGUF models
├── bin/                       # llama.cpp binaries
└── memory/                    # SQLite memory database
```

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
- List all configured models
- Switch active model at runtime
- **Add from HuggingFace** — paste any HF GGUF URL
- Enable/disable models

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

### `config.json` — Models and paths

```json
{
  "models": {
    "MyModel": {
      "enabled": true,
      "type": "vision-language",
      "path": "models/MyModel/model.gguf",
      "mmproj": "models/MyModel/mmproj.gguf",
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
| `/api/models/add-from-hf` | POST | Add model from HuggingFace |
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

**Via UI**: Models → Add from HuggingFace → paste URL

**Via CLI**:
```bash
python model_manager.py hf-add --hf Qwen/Qwen2-VL-7B-Instruct-GGUF
python download_q4.py
```

**Via config.json**: Edit directly (see above)

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

- **Tested on**: NVIDIA RTX 4060 Laptop (8.5 GB VRAM)
- LFM2.5-VL-1.6B-Q4_0: ~2.4 GB VRAM
- LFM2.5-Audio-1.5B-Q4_0: ~2.2 GB VRAM
- gemma-4-E2B: ~4.4 GB VRAM

## License

LFM2.5 models: [LFM2.5 License](https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B-GGUF)
llama.cpp: MIT | MATHIR: MIT | OpenCV: Apache 2.0 | Flask: BSD-3
