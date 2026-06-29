# MATHIR Playground

A web playground to chat with vision/audio LLMs (via OpenRouter) and exercise the MATHIR memory backend (via the running daemon). **NO HARDCODED PATHS** — everything is configurable via JSON files.

**v8.5.1** (2026-06-29) — Current version. Uses OpenRouter cloud API for inference and the new MCP daemon (23 tools) for memory.

---

## ⚡ Quick Start

```bash
cd vision_testing
pip install -r requirements.txt

# Set your OpenRouter key (get one at https://openrouter.ai/keys):
echo 'export OPENROUTER_API_KEY=sk-or-v1-...' >> ~/.bashrc
source ~/.bashrc

python start_ui.py
# Opens at http://127.0.0.1:5000
```

Or use the `.env` file (auto-loaded by `env_config.py`):

```bash
cp .env.example .env
# Edit .env, add your OPENROUTER_API_KEY
python start_ui.py
```

---

## 📁 File Structure

```
vision_testing/
├── config.json              # OpenRouter + models config
├── ui_config.json           # UI settings (port, camera, audio)
├── system_context.json      # System prompt for the model
├── .env / .env.example      # API keys (auto-loaded)
│
├── ui/                      # Web UI (dark theme, 8px grid, SVG icons)
│   ├── index.html           # Single-page app (6 views)
│   ├── memory_dashboard.html
│   ├── playground.html
│   └── static/
│       ├── style.css
│       └── app.js
│
├── ui_server.py             # Flask backend (17 API routes)
├── start_ui.py              # Launcher (installs deps, runs server)
├── vision_test.py           # Core: VisionTester, OpenRouterClient, ModelManager
├── mathir_daemon_client.py  # MATHIR MCP daemon bridge (23 tools)
│
├── env_config.py            # .env loader
├── model_manager.py         # Model list/select/enable
├── accuracy_tests.py        # Accuracy test battery
├── synthetic_images.py      # Test image generator
├── setup_sources.py         # Tauri UI setup sources
│
└── memory/                  # SQLite memory database (auto-created)
    └── vision_test.db
```

---

## 🖥️ UI Features (6 views)

| # | View | What it does |
|---|---|---|
| 1 | **Chat** | Real-time chat, push-to-talk audio, image attachments, history in localStorage |
| 2 | **Camera** | Live webcam (OpenCV backend), describe/ask/snapshot, hold-to-talk |
| 3 | **Models** | List/switch active model, enable/disable, free models highlighted |
| 4 | **Memory** | Query MATHIR memory (FTS5), view interactions, stats |
| 5 | **Accuracy** | Run test batteries, compare models side-by-side, nDCG@10/MRR/latency |
| 6 | **Settings** | Camera/audio/theme, system info, keyboard shortcuts |

### Keyboard Shortcuts

| Key | Action |
|---|---|
| `1`–`6` | Switch view (Chat, Camera, Models, Memory, Accuracy, Settings) |
| `Enter` | Send chat |
| `Shift+Enter` | Newline |
| `Space` (hold) | Record audio |
| `Ctrl+K` | Focus chat input |
| `Esc` | Close modal / clear focus |

---

## 🔌 MATHIR Memory Integration

Two paths to memory:

### A) Local SimpleMemory (always works)
- `mathir_daemon_client.py` wraps the SQLite FTS5 memory in `memory/`
- Every chat message stored with metadata (model, timestamp, query type)
- Before each response, relevant memories retrieved via FTS5 search
- Last 3 memories always appended for context

### B) MATHIR MCP Daemon (recommended)
- Connects to running daemon at `http://127.0.0.1:7338` (default)
- Full 23-tool MCP surface: `memory_save`, `memory_recall`, `memory_by_path`, etc.
- If daemon is down, falls back to local SimpleMemory gracefully

---

## 📡 API Routes (19)

| Route | Method | Description |
|---|---|---|
| `/api/system/context` | GET | System context + models |
| `/api/system/info` | GET | System info (platform, paths) |
| `/api/models` | GET | List all models + provider status |
| `/api/models/switch` | POST | Switch active model |
| `/api/models/toggle` | POST | Enable/disable model |
| `/api/models/add-from-or` | POST | Add model from OpenRouter ID |
| `/api/chat` | POST | Chat (with optional image/audio) |
| `/api/camera/start` | POST | Start camera |
| `/api/camera/stop` | POST | Stop camera |
| `/api/camera/frame` | GET | Current frame (JPEG) |
| `/api/camera/stream` | GET | MJPEG stream |
| `/api/camera/ask` | POST | Ask about scene |
| `/api/memory/recall` | POST | Search memory |
| `/api/memory/stats` | GET | Memory stats |
| `/api/accuracy/tests` | GET | List accuracy tests |
| `/api/accuracy/results` | GET | Get accuracy results |
| `/api/accuracy/test` | POST | Run accuracy battery |
| `/api/local/status` | GET | Check local backends (llama.cpp + Ollama) |
| `/api/local/chat` | POST | Chat via local llama.cpp (no network) |

---

## ⚙️ Configuration

### Providers (4 backends)

| Provider | Backend | Key required | Privacy | Use case |
|---|---|---|---|---|
| **OpenRouter** | Cloud API | Yes (`OPENROUTER_API_KEY`) | Data sent to cloud | Best free models (26+) |
| **OpenCode Zen** | Cloud API | Yes (`OPENCODE_ZEN_API_KEY`) | Data sent to cloud | Curated free models |
| **Ollama** | Local server | No | Fully local | Self-hosted models |
| **llama.cpp** | Local GGUF | No | Fully local | Offline / edge / privacy |

### `config.json` — All providers + models

```json
{
  "openrouter": { "api_key": "", "api_base": "https://openrouter.ai/api/v1" },
  "opencode_zen": { "api_key": "", "api_base": "https://opencode.ai/zen/v1" },
  "ollama": { "api_base": "http://localhost:11434" },
  "llama_local": { "models": {} },
  "models": {
    "gemma4-free": { "provider": "openrouter", "enabled": true, "supports_vision": true },
    "gemma2-9b-ollama": { "provider": "ollama", "enabled": false },
    "my-gguf": { "provider": "llama_local", "path": "models/my.gguf", "enabled": false }
  }
}
```

### Adding local models

**Ollama** (recommended for local): `ollama pull gemma2:9b`, then add to `config.json:models` with `"provider": "ollama"`.

**GGUF (llama.cpp)**: Set `path` to your `.gguf` file, optional `mmproj` for vision. Enable with `"provider": "llama_local"`.

### `ui_config.json` — UI settings

```json
{
  "server": { "host": "127.0.0.1", "port": 5000 },
  "camera": { "device_id": 0, "width": 1280, "height": 720, "fps": 30 },
  "audio": { "push_to_talk_key": "Space", "max_record_seconds": 30 }
}
```

### `.env` — API keys (auto-loaded)

```
OPENROUTER_API_KEY=sk-or-v1-PUT-YOUR-KEY-HERE
```

---

## 🛠️ Adding Models

- **Via UI**: Models → Add from OpenRouter → paste model ID (e.g., `openai/gpt-4o-mini`)
- **Via config.json**: Edit directly. Find IDs at https://openrouter.ai/models
- **Ollama**: `ollama pull <model>`, then add entry with `"provider": "ollama"`
- **GGUF**: Set `path` to `.gguf` file, optional `mmproj` for vision models
- **Free models filter**: https://openrouter.ai/models?max_price=0

---

## 🖥️ Hardware Requirements

- **Cloud (OpenRouter/Zen)**: No GPU, internet required
- **Ollama**: 4-8GB RAM per model, CPU or GPU
- **llama.cpp GGUF**: Varies by quantization — 4GB for Q4, 8GB+ for Q8
- Webcam + microphone for Camera view (optional)

---

## 🗑️ What was removed in v8.5.0 (kept for historical reference)

| Removed | Replacement |
|---|---|
| `bin/` (~1GB llama.cpp + CUDA DLLs) | OpenRouter cloud API |
| `models/` (5 GGUF model dirs) | OpenRouter model IDs in `config.json` |
| `convert_lfm2_to_gguf.py` | n/a (no GGUF conversion) |
| `download_models.py` | n/a (no local downloads) |
| `download_q4.py` | n/a (no local quantisation) |
| `setup_binaries.py` | n/a (no binaries to setup) |
| `interface/LlamaSetupModal_reference.jsx` | `interface/OpenRouterSetupModal_reference.jsx` |
| `interface/wizardModels_llamacpp_reference.json` | `interface/wizardModels_openrouter_reference.json` |
| `LlamaServer` class in `vision_test.py` | `OpenRouterClient` class |
| `config.json:llama_server` section | `config.json:openrouter` section |

---

## 📜 License

MATHIR: MIT | OpenRouter: see https://openrouter.ai | OpenCV: Apache 2.0 | Flask: BSD-3
