#!/usr/bin/env python3
"""
MATHIR Playground — Backend Server (v8.5.1)
============================================

Adds:
- System context injection (system_context.json -> system message on every chat)
- Learning data endpoint (for graphs)
- Tests endpoint (loads test_results.json)
- Multi-modal awareness (image + audio + text in same request)
- NO hardcoded questions - all questions come from the user

NO HARDCODED PATHS. All from config.json.
"""
import os
import sys
import json
import time
import base64
import threading
import subprocess
import shutil
import sqlite3
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

from flask import Flask, request, jsonify, Response, send_from_directory
from flask_cors import CORS

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # Add MATHIR root

# SimpleMemory — FTS5-only, no torch, no daemon needed
from mathir_dropin.simple import SimpleMemory

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# Global memory (SimpleMemory — FTS5-only, no daemon)
# ============================================================

MEMORY_DB = str(HERE / "memory" / "vision_memory.db")
Path(MEMORY_DB).parent.mkdir(parents=True, exist_ok=True)
memory = SimpleMemory(db_path=MEMORY_DB)


# ============================================================
# Config loading
# ============================================================

def load_config(name="config.json"):
    path = HERE / name
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def load_ui_config():
    return load_config("ui_config.json")


def save_ui_config(cfg):
    with open(HERE / "ui_config.json", "w") as f:
        json.dump(cfg, f, indent=2)


def resolve_path(p, base=HERE):
    pp = Path(p)
    if pp.is_absolute():
        return pp
    return (base / pp).resolve()


# Import vision_test components
sys.path.insert(0, str(HERE))
from vision_test import VisionTester, ModelManager, OpenRouterClient, VirtualRoom, load_config as vt_load_config

# Local llama.cpp backend (optional, requires llama-cpp-python)
try:
    from local_llama_cpp import LlamaCppBackend, OllamaClient, get_llama_cpp_available, get_ollama_available
    LLAMA_CPP_AVAILABLE = get_llama_cpp_available()
    OLLAMA_AVAILABLE = get_ollama_available()
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    OLLAMA_AVAILABLE = False

# Accuracy test framework (optional - server boots even if import fails
# because accuracy is a feature, not a hard dependency for the rest of
# the UI). We log the import failure so the operator knows.
_accuracy_framework_error: Optional[str] = None
try:
    from accuracy_tests import AccuracyTestFramework, ALL_TEST_TYPES
except Exception as _acc_exc:  # pragma: no cover
    AccuracyTestFramework = None  # type: ignore
    ALL_TEST_TYPES = ()
    _accuracy_framework_error = str(_acc_exc)
    print(f"[ui_server] WARN: accuracy_tests unavailable: {_acc_exc}")


# ============================================================
# App + state
# ============================================================

app = Flask(__name__, static_folder=str(HERE / "ui" / "static"), static_url_path="/static")
CORS(app)

# Learning data: tracked in-memory, surfaces in Learning view
learning_data = {
    "memory_timeline": [],          # list of {timestamp, count}
    "response_times": {},           # {model: [durations in seconds]}
    "storage_timeline": [],         # list of {timestamp, size_bytes}
    "query_types": {                # counters
        "text": 0, "image": 0, "audio": 0, "multimodal": 0,
    },
    "modality_usage": {             # counters
        "vision": 0, "audio": 0, "multimodal": 0, "text_only": 0,
    },
    "recalls": {"total": 0, "nonempty": 0},  # accuracy proxy
    "model_modality": {},           # {model: {vision, audio, multimodal, text}}
    "started_at": datetime.now().isoformat(),
    "last_event": None,
}
learning_lock = threading.Lock()

# Cached system context (re-read on file change)
_system_context_cache = {"mtime": 0, "data": None}


def get_system_context():
    """Load system_context.json with simple file mtime cache."""
    path = HERE / "system_context.json"
    if not path.exists():
        return {}
    mtime = path.stat().st_mtime
    if _system_context_cache["data"] is None or mtime != _system_context_cache["mtime"]:
        try:
            with open(path) as f:
                _system_context_cache["data"] = json.load(f)
                _system_context_cache["mtime"] = mtime
        except Exception:
            return {}
    return _system_context_cache["data"] or {}


def build_system_message() -> str:
    """Render a minimal system prompt from system_context.json."""
    ctx = get_system_context()
    if not ctx:
        return ""

    # Build capabilities string from active model
    active = state.get("active_model")
    models = state["models"].list_models(enabled_only=False)
    active_caps: List[str] = []
    for name, m in models.items():
        if name == active:
            if m.get("supports_vision"):
                active_caps.append("vision")
            if m.get("supports_audio"):
                active_caps.append("audio")
            active_caps.append("text")

    # Render template
    template = ctx.get("current_context_template", "")
    try:
        context_line = template.format(
            active_model=active or "none",
            model_capabilities=", ".join(active_caps) if active_caps else "text",
            date=datetime.now().strftime("%Y-%m-%d"),
            platform=sys.platform,
            memory_db_path=MEMORY_DB,
        )
    except Exception:
        context_line = ""

    # Assemble: identity + rules + context
    parts: List[str] = []
    if ctx.get("identity"):
        parts.append(ctx["identity"])

    rules = ctx.get("behavior_rules", [])
    if rules:
        parts.append("\n".join(f"- {r}" for r in rules))

    if context_line.strip():
        parts.append(context_line)

    return "\n\n".join(p for p in parts if p)


def classify_query(message: str, has_image: bool, has_audio: bool) -> str:
    """Classify a chat query into a single bucket (TASK 4).

    Priority order (first match wins):
      1. has_image              -> "with_image"
      2. has_audio              -> "with_audio"
      3. text contains "count"  -> "count_objects"
         OR text contains "how many"
      4. text length < 20 chars -> "short"
      5. else                   -> "basic"

    Returns one of: "with_image", "with_audio", "count_objects",
                    "short", "basic".
    """
    if has_image:
        return "with_image"
    if has_audio:
        return "with_audio"
    text = (message or "").lower()
    if "count" in text or "how many" in text:
        return "count_objects"
    if len((message or "").strip()) < 20:
        return "short"
    return "basic"


def record_event(event_type: str, **kwargs):
    """Record a learning event (thread-safe)."""
    with learning_lock:
        learning_data["last_event"] = {
            "type": event_type,
            "timestamp": datetime.now().isoformat(),
            **kwargs,
        }
        if event_type == "chat":
            model = kwargs.get("model", "unknown")
            duration = kwargs.get("duration", 0.0)
            learning_data["response_times"].setdefault(model, []).append(duration)
            # Cap to last 100
            if len(learning_data["response_times"][model]) > 100:
                learning_data["response_times"][model] = learning_data["response_times"][model][-100:]

            modality = kwargs.get("modality", "text_only")
            learning_data["modality_usage"][modality] = learning_data["modality_usage"].get(modality, 0) + 1

            qtypes = kwargs.get("query_types", [])
            for qt in qtypes:
                learning_data["query_types"][qt] = learning_data["query_types"].get(qt, 0) + 1

            # Memory count snapshot via SimpleMemory
            try:
                stats = memory.get_stats()
                count = stats.get("total_memories", 0)
                learning_data["memory_timeline"].append({
                    "timestamp": datetime.now().isoformat(),
                    "count": count,
                })
                if len(learning_data["memory_timeline"]) > 500:
                    learning_data["memory_timeline"] = learning_data["memory_timeline"][-500:]
            except Exception:
                pass

        elif event_type == "recall":
            learning_data["recalls"]["total"] += 1
            if kwargs.get("nonempty"):
                learning_data["recalls"]["nonempty"] += 1


# Global state
state = {
    "config": vt_load_config(),
    "ui_config": load_ui_config(),
    "models": ModelManager(vt_load_config()),
    "active_model": None,
    "active_tester": None,
    "active_server": None,
    "memory_db": None,
    "camera_thread": None,
    "camera_running": False,
    "latest_frame": None,
    "frame_lock": threading.Lock(),
}


def get_active_tester():
    """Get or create the active VisionTester."""
    cfg = state["config"]
    active = state.get("active_model")
    if not active:
        enabled = state["models"].list_models(enabled_only=True)
        if not enabled:
            raise ValueError("No enabled models in config.json")
        active = list(enabled.keys())[0]
        state["active_model"] = active

    if state.get("active_tester") is None or state["active_tester"].model_name != active:
        if state.get("active_server"):
            state["active_server"].stop()
        tester = VisionTester(active, port=state["ui_config"]["server"]["port"] + 1)
        tester.setup_memory()
        tester.start_server()
        state["active_tester"] = tester
        state["active_server"] = tester.server

    return state["active_tester"]


# ============================================================
# Static UI routes
# ============================================================

@app.route("/")
def index():
    return send_from_directory(str(HERE / "ui"), "index.html")


@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(str(HERE / "ui"), filename)


@app.route("/dashboard")
def memory_dashboard():
    """Serve the Memory Dashboard HTML."""
    return send_from_directory(str(HERE / "ui"), "memory_dashboard.html")


# ============================================================
# System context endpoint
# ============================================================

@app.route("/api/system/context", methods=["GET"])
def get_system_context_endpoint():
    """Return the current system_context.json plus model info.

    Response shape:
      - context:        rendered system message (already injected on /api/chat)
      - raw:            the raw system_context.json content
      - active_model:   the model currently active
      - available_models: list of {name, type, capabilities, active, enabled}
      - model_info:     the inline system context block listing active model,
                        capabilities, available models, platform, date, db path
      - platform:       sys.platform
      - date:           ISO timestamp at request time
    """
    raw_ctx = get_system_context()

    # Build the available-models list with capability info
    available_models: List[Dict[str, Any]] = []
    active_caps: List[str] = []
    for name, m in state["models"].list_models(enabled_only=False).items():
        caps: List[str] = ["text"]
        if m.get("supports_vision"):
            caps.append("vision")
        if m.get("supports_audio"):
            caps.append("audio")
        if len(caps) > 1:
            caps.append("multimodal")
        caps = list(dict.fromkeys(caps))  # dedupe, preserve order
        available_models.append({
            "name": name,
            "type": m.get("type", "text-only"),
            "capabilities": caps,
            "supports_vision": bool(m.get("supports_vision")),
            "supports_audio": bool(m.get("supports_audio")),
            "enabled": bool(m.get("enabled", True)),
            "active": name == state.get("active_model"),
        })
        if name == state.get("active_model"):
            active_caps = caps

    # Inline model-info block (mirrors the block injected on /api/chat)
    try:
        db_path_str = MEMORY_DB
    except Exception:
        db_path_str = ""
    model_info = {
        "active_model": state.get("active_model") or "none",
        "model_capabilities": active_caps or ["text"],
        "available_models": available_models,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "platform": sys.platform,
        "memory_db_path": db_path_str,
    }

    return jsonify({
        "context": build_system_message(),
        "raw": raw_ctx,
        "active_model": state.get("active_model"),
        "available_models": available_models,
        "model_info": model_info,
        "platform": sys.platform,
        "date": datetime.now().isoformat(),
    })


# ============================================================
# Models API
# ============================================================

@app.route("/api/models", methods=["GET"])
def list_models():
    """List all configured models (OpenRouter, Zen, Ollama, local GGUF)."""
    models = state["models"].list_models(enabled_only=False)
    result = []
    for name, m in models.items():
        provider = m.get("provider", "openrouter")
        paths = state["models"].get_model_paths(name)
        modalities: List[str] = ["text"]
        if m.get("supports_vision"):
            modalities.append("vision")
        if m.get("supports_audio"):
            modalities.append("audio")
        if len(modalities) > 1:
            modalities.append("multimodal")
        result.append({
            "name": name,
            "display_name": m.get("display_name", name),
            "description": m.get("description", ""),
            "type": m.get("type"),
            "provider": provider,
            "enabled": m.get("enabled", True),
            "active": name == state.get("active_model"),
            "supports_vision": m.get("supports_vision", False),
            "supports_audio": m.get("supports_audio", False),
            "modalities": modalities,
            "path": str(paths["path"]) if paths and paths.get("path") else None,
            "path_exists": paths["path"].exists() if paths and paths.get("path") else False,
            "size_mb": m.get("size_mb", 0),
            "vram_mb": m.get("vram_mb", 0),
            "context_length": m.get("context_length", 4096),
            "huggingface_url": m.get("huggingface_url"),
        })

    # Provider availability summary
    providers = {
        "openrouter": {"available": bool(state["config"].get("openrouter", {}).get("api_key")),
                       "models": sum(1 for m in result if m["provider"] == "openrouter")},
        "opencode_zen": {"available": bool(state["config"].get("opencode_zen", {}).get("api_key")),
                          "models": sum(1 for m in result if m["provider"] == "opencode_zen")},
        "ollama": {"available": OLLAMA_AVAILABLE,
                    "models": sum(1 for m in result if m["provider"] == "ollama")},
        "llama_local": {"available": LLAMA_CPP_AVAILABLE,
                         "models": sum(1 for m in result if m["provider"] == "llama_local")},
    }

    return jsonify({
        "models": result,
        "active": state.get("active_model"),
        "providers": providers,
    })


@app.route("/api/models/switch", methods=["POST"])
def switch_model():
    data = request.json
    name = data.get("name")
    if not name:
        return jsonify({"error": "name required"}), 400

    if name not in state["models"].list_models(enabled_only=False):
        return jsonify({"error": f"Model '{name}' not found"}), 404

    if state.get("active_server"):
        state["active_server"].stop()
    state["active_server"] = None
    state["active_tester"] = None
    state["active_model"] = name

    try:
        get_active_tester()
        return jsonify({"status": "switched", "active": name})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/models/validate", methods=["GET"])
def validate_models():
    models = state["models"].list_models(enabled_only=False)
    result = {}
    for name in models:
        v = state["models"].validate_model(name)
        result[name] = v
    return jsonify(result)


@app.route("/api/local/status", methods=["GET"])
def local_status():
    """Check if local backends are available."""
    ollama_status = OLLAMA_AVAILABLE
    llama_status = LLAMA_CPP_AVAILABLE
    # Try to get Ollama model list if available
    ollama_models = []
    if ollama_status:
        try:
            import urllib.request
            cfg = state["config"].get("ollama", {})
            url = cfg.get("api_base", "http://localhost:11434")
            req = urllib.request.Request(f"{url}/api/tags", headers={"User-Agent": "MATHIR-UI"})
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read())
                ollama_models = [m.get("name", "") for m in data.get("models", [])]
        except Exception:
            pass
    return jsonify({
        "ollama_available": ollama_status,
        "llama_cpp_available": llama_status,
        "ollama_models": ollama_models,
    })


@app.route("/api/local/chat", methods=["POST"])
def local_chat():
    """Send a chat to a local GGUF model (no network, no API key)."""
    if not LLAMA_CPP_AVAILABLE:
        return jsonify({"error": "llama-cpp-python not installed"}), 503

    data = request.json
    model_name = data.get("model")
    messages = data.get("messages", [])
    max_tokens = int(data.get("max_tokens", 512))
    temperature = float(data.get("temperature", 0.7))

    if not model_name or not messages:
        return jsonify({"error": "model and messages required"}), 400

    models_cfg = state["config"].get("models", {})
    model_info = models_cfg.get(model_name)
    if not model_info:
        return jsonify({"error": f"Model '{model_name}' not found in config.json"}), 404

    try:
        model_paths = state["models"].get_model_paths(model_name)
        backend = LlamaCppBackend(model_paths)
        result = backend.chat(messages, max_tokens=max_tokens, temperature=temperature)
        return jsonify({"response": result, "model": model_name, "backend": "llama-cpp"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/models/add-from-hf", methods=["POST"])
def add_model_from_hf():
    data = request.json
    hf_url = data.get("hf_url")
    name = data.get("name")

    if not hf_url:
        return jsonify({"error": "hf_url required"}), 400

    if "huggingface.co/" not in hf_url:
        return jsonify({"error": "Not a HuggingFace URL"}), 400
    repo_id = hf_url.split("huggingface.co/")[-1].rstrip("/")

    try:
        url = f"https://huggingface.co/api/models/{repo_id}/tree/main"
        req = urllib.request.Request(url, headers={"User-Agent": "MATHIR-UI"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            files = json.loads(resp.read())
    except Exception as e:
        return jsonify({"error": f"Could not list HF files: {e}"}), 400

    gguf_files = [f for f in files if f["path"].endswith(".gguf") and "mmproj" not in f["path"].lower()
                  and "tokenizer" not in f["path"].lower() and "vocoder" not in f["path"].lower()]
    mmproj_files = [f for f in files if "mmproj" in f["path"].lower()]

    if not gguf_files:
        return jsonify({"error": "No GGUF files found in repo"}), 400

    main = sorted(gguf_files, key=lambda f: f.get("size", 0))[0]
    mmproj = sorted(mmproj_files, key=lambda f: f.get("size", 0))[0] if mmproj_files else None

    is_vision = "VL" in repo_id or "vision" in repo_id.lower() or "Vision" in repo_id
    is_audio = "audio" in repo_id.lower() or "Audio" in repo_id
    is_multimodal = is_vision and is_audio
    mtype = "multimodal" if is_multimodal else "vision-language" if is_vision else "audio" if is_audio else "text-only"

    model_id = name or repo_id.split("/")[-1]
    if model_id.endswith("-GGUF"):
        model_id = model_id[:-5]
    model_dir = model_id

    entry = {
        "enabled": True,
        "type": mtype,
        "display_name": model_id,
        "description": f"From {repo_id}",
        "path": f"models/{model_dir}/{main['path']}",
        "size_mb": int(main.get("size", 0) / 1024 / 1024),
        "vram_mb": int(main.get("size", 0) / 1024 / 1024 * 2),
        "context_length": 4096,
        "supports_vision": is_vision,
        "supports_audio": is_audio,
        "huggingface_url": f"https://huggingface.co/{repo_id}",
    }

    if mmproj:
        entry["mmproj"] = f"models/{model_dir}/{mmproj['path']}"

    state["models"].add_model(model_id, entry)
    state["config"] = vt_load_config()
    state["models"] = ModelManager(state["config"])

    return jsonify({
        "status": "added",
        "model": model_id,
        "modalities": entry.get("supports_vision", False) * ["vision"] +
                      entry.get("supports_audio", False) * ["audio"] + ["text"],
        "files_to_download": [{"path": main["path"], "size_mb": entry["size_mb"]}] +
                              ([{"path": mmproj["path"], "size_mb": int(mmproj["size"]/1024/1024)}] if mmproj else []),
    })


@app.route("/api/models/download", methods=["POST"])
def download_model_files():
    data = request.json
    name = data.get("name")
    if not name:
        return jsonify({"error": "name required"}), 400

    m = state["models"].get_model(name)
    if not m:
        return jsonify({"error": f"Model '{name}' not found"}), 404

    if not m.get("huggingface_url"):
        return jsonify({"error": "Model has no HuggingFace URL"}), 400

    repo_id = m["huggingface_url"].split("huggingface.co/")[-1].rstrip("/")
    files_to_dl = []
    for key in ["path", "mmproj", "tokenizer", "vocoder"]:
        v = m.get(key)
        if not v:
            continue
        filename = Path(v).name
        dest = resolve_path(v)
        if dest.exists():
            continue
        files_to_dl.append((key, filename, dest))

    if not files_to_dl:
        return jsonify({"status": "already_downloaded", "name": name})

    try:
        url = f"https://huggingface.co/api/models/{repo_id}/tree/main"
        req = urllib.request.Request(url, headers={"User-Agent": "MATHIR-UI"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            hf_files = json.loads(resp.read())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    downloaded = []
    for key, filename, dest in files_to_dl:
        hf_file = next((f for f in hf_files if f["path"] == filename), None)
        if not hf_file:
            continue
        dl_url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
        try:
            req = urllib.request.Request(dl_url, headers={"User-Agent": "MATHIR-UI"})
            with urllib.request.urlopen(req, timeout=600) as resp:
                dest.parent.mkdir(parents=True, exist_ok=True)
                with open(dest, "wb") as f:
                    shutil.copyfileobj(resp, f)
            downloaded.append({"key": key, "path": filename, "size_mb": int(hf_file.get("size", 0) / 1024 / 1024)})
        except Exception as e:
            return jsonify({"error": f"Failed to download {filename}: {e}"}), 500

    return jsonify({"status": "downloaded", "name": name, "files": downloaded})


@app.route("/api/models/toggle", methods=["POST"])
def toggle_model():
    data = request.json
    name = data.get("name")
    enabled = data.get("enabled", True)
    if not name:
        return jsonify({"error": "name required"}), 400
    state["models"].toggle_model(name, enabled)
    return jsonify({"status": "toggled", "name": name, "enabled": enabled})


@app.route("/api/models/remove", methods=["POST"])
def remove_model():
    data = request.json
    name = data.get("name")
    if not name:
        return jsonify({"error": "name required"}), 400
    if name == state.get("active_model"):
        return jsonify({"error": "Cannot remove active model"}), 400
    state["models"].remove_model(name)
    return jsonify({"status": "removed", "name": name})


# ============================================================
# Chat API - with system context injection
# ============================================================

@app.route("/api/chat", methods=["POST"])
def chat():
    """Send a chat message to the active model with system context injection."""
    data = request.json
    message = data.get("message")
    image_b64 = data.get("image")  # Optional base64 image
    audio_b64 = data.get("audio")  # Optional base64 audio
    system_override = data.get("system")  # Optional override of system prompt

    if not message and not image_b64 and not audio_b64:
        return jsonify({"error": "message, image, or audio required"}), 400

    if not message:
        message = ""  # Allow empty text if image/audio provided

    try:
        tester = get_active_tester()
    except Exception as e:
        return jsonify({"error": f"No model loaded: {e}"}), 500

    system_msg = system_override or build_system_message()

    # Determine modality + query-type classification (TASK 4)
    has_image = bool(image_b64)
    has_audio = bool(audio_b64)
    if has_image and has_audio:
        modality = "multimodal"
    elif has_image:
        modality = "vision"
    elif has_audio:
        modality = "audio"
    else:
        modality = "text_only"

    # New single-bucket classifier (TASK 4). One of:
    #   "with_image", "with_audio", "count_objects", "short", "basic"
    query_type = classify_query(message, has_image, has_audio)

    # Build messages
    start = time.time()
    try:
        # RECALL relevant memories from MATHIR daemon
        memory_context = ""
        if message:
            try:
                recalled = memory.recall(message, k=5)
                last_memories = memory.get_last(n=3)
                # Merge and deduplicate
                seen_ids = set()
                all_memories = []
                for r in recalled + last_memories:
                    mid = r.get("memory_id", id(r))
                    if mid not in seen_ids:
                        seen_ids.add(mid)
                        all_memories.append(r)
                if all_memories:
                    mem_lines = []
                    for r in all_memories[:5]:
                        txt = r.get("text", "") or r.get("metadata", {}).get("text", "")
                        if txt:
                            mem_lines.append(f"- {txt}")
                    if mem_lines:
                        memory_context = "\n\n## RELEVANT MEMORIES (from shared memory.db — use these to compare with current image):\n" + "\n".join(mem_lines)
            except Exception as e:
                print(f"  [WARN] Memory recall failed: {e}")

        # Prepend memory context to system message
        full_system = system_msg + memory_context if memory_context else system_msg

        if has_image:
            # Save image to temp file
            import tempfile
            img_data = base64.b64decode(image_b64.split(",", 1)[-1] if "," in image_b64 else image_b64)
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                f.write(img_data)
                tmp_path = f.name
            try:
                # Build messages with system + user (with image)
                with open(tmp_path, "rb") as f:
                    image_data = base64.b64encode(f.read()).decode()
                messages = []
                if full_system:
                    messages.append({"role": "system", "content": full_system})
                user_content = []
                if message:
                    user_content.append({"type": "text", "text": message})
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                })
                messages.append({"role": "user", "content": user_content if user_content else ""})
                response = tester.server.chat(messages)
            finally:
                os.unlink(tmp_path)
        else:
            messages = []
            if full_system:
                messages.append({"role": "system", "content": full_system})
            messages.append({"role": "user", "content": message or "(no text - see attached)"})
            response = tester.server.chat(messages)

        duration = time.time() - start

        # Store in MATHIR memory — full conversation, not truncated
        try:
            # Skip trivial messages
            skip_words = {"hi", "hello", "ok", "thanks", "thank you", "bye", "yes", "no", "hey"}
            is_trivial = (message or "").strip().lower() in skip_words
            is_short = len((message or "").strip()) < 5
            
            if not is_trivial and not is_short:
                # Build memory text: full conversation
                mem_text = f"User: {message or '[image/audio]'}\nAssistant: {str(response)}"
                
                # Add file/image info if present
                if has_image:
                    mem_text = f"[IMAGE ATTACHED] {mem_text}"
                if has_audio:
                    mem_text = f"[AUDIO ATTACHED] {mem_text}"
                
                memory.store(text=mem_text, metadata={
                    "query_type": query_type,
                    "modality": modality,
                    "has_image": has_image,
                    "has_audio": has_audio,
                    "model": tester.model_name,
                    "duration": round(duration, 3),
                    "agent": "vision_ui",
                    "label": "vision_chat",
                })
        except Exception as e:
            print(f"  [WARN] Memory store failed: {e}")

        # Record learning event (back-compat: keep query_types as a list)
        record_event("chat", model=tester.model_name, duration=duration,
                     modality=modality, query_types=[query_type])

        return jsonify({
            "response": response,
            "model": tester.model_name,
            "timestamp": datetime.now().isoformat(),
            "duration": round(duration, 3),
            "modality": modality,
            "query_type": query_type,
            "query_types": [query_type],
        })
    except Exception as e:
        return jsonify({"error": str(e), "model": state.get("active_model")}), 500


# ============================================================
# Camera API
# ============================================================

@app.route("/api/camera/start", methods=["POST"])
def start_camera():
    if state["camera_running"]:
        return jsonify({"status": "already_running"})

    try:
        import cv2
    except ImportError:
        return jsonify({"error": "OpenCV not installed. Run: pip install opencv-python"}), 500

    cam_cfg = state["ui_config"]["camera"]

    def capture_loop():
        cap = cv2.VideoCapture(cam_cfg["device_id"])
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg["width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg["height"])
        cap.set(cv2.CAP_PROP_FPS, cam_cfg["fps"])

        while state["camera_running"]:
            ret, frame = cap.read()
            if not ret:
                break
            _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, cam_cfg["frame_quality"]])
            with state["frame_lock"]:
                state["latest_frame"] = jpeg.tobytes()
            time.sleep(1 / cam_cfg["fps"])

        cap.release()

    state["camera_running"] = True
    state["camera_thread"] = threading.Thread(target=capture_loop, daemon=True)
    state["camera_thread"].start()
    return jsonify({"status": "started"})


@app.route("/api/camera/stop", methods=["POST"])
def stop_camera():
    state["camera_running"] = False
    if state["camera_thread"]:
        state["camera_thread"].join(timeout=5)
        state["camera_thread"] = None
    return jsonify({"status": "stopped"})


@app.route("/api/camera/frame", methods=["GET"])
def get_frame():
    with state["frame_lock"]:
        if state["latest_frame"] is None:
            return Response(status=204)
        return Response(state["latest_frame"], mimetype="image/jpeg")


@app.route("/api/camera/stream")
def camera_stream():
    def generate():
        while True:
            with state["frame_lock"]:
                if state["latest_frame"] is None:
                    time.sleep(0.1)
                    continue
                frame = state["latest_frame"]
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            time.sleep(0.033)
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/camera/ask", methods=["POST"])
def ask_about_camera():
    """Capture current frame and ask model about it. Uses USER-PROVIDED question (no hardcoded fallback)."""
    data = request.json or {}
    question = (data.get("question") or "").strip()

    if not question:
        return jsonify({"error": "question required (type your own question - no default)"}), 400

    with state["frame_lock"]:
        if state["latest_frame"] is None:
            return jsonify({"error": "No camera frame available"}), 400
        frame_b64 = base64.b64encode(state["latest_frame"]).decode()

    try:
        tester = get_active_tester()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        f.write(base64.b64decode(frame_b64))
        tmp_path = f.name

    system_msg = build_system_message()
    start = time.time()
    try:
        # Build messages with system + user (with image)
        with open(tmp_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode()
        messages = []
        if system_msg:
            messages.append({"role": "system", "content": system_msg})
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
            ],
        })
        response = tester.server.chat(messages)
    finally:
        os.unlink(tmp_path)

    duration = time.time() - start

    try:
        memory.store(
            text=f"Q(cam): {question} | A: {str(response)[:200]}",
            metadata={
                "query_type": classify_query(question, has_image=True, has_audio=False),
                "modality": "vision",
                "has_image": True,
                "has_audio": False,
                "agent": "vision_ui",
                "label": "camera_ask",
            },
        )
    except Exception:
        pass

    cam_qtype = classify_query(question, has_image=True, has_audio=False)
    record_event("chat", model=tester.model_name, duration=duration,
                 modality="vision", query_types=[cam_qtype])

    return jsonify({
        "response": response,
        "question": question,
        "model": tester.model_name,
        "timestamp": datetime.now().isoformat(),
        "duration": round(duration, 3),
        "modality": "vision",
        "query_type": cam_qtype,
    })


@app.route("/api/camera/count", methods=["POST"])
def count_objects():
    """Count objects in the current frame. Uses USER-PROVIDED object name (no hardcoded fallback)."""
    data = request.json or {}
    obj = (data.get("object") or "").strip()
    question = (data.get("question") or "").strip()

    if not obj:
        return jsonify({"error": "object name required (type what you want to count)"}), 400

    # Build question from user input - no hardcoded phrasings
    if question:
        full_q = question
    else:
        full_q = f"Count: {obj}"

    with state["frame_lock"]:
        if state["latest_frame"] is None:
            return jsonify({"error": "No camera frame available"}), 400
        frame_b64 = base64.b64encode(state["latest_frame"]).decode()

    try:
        tester = get_active_tester()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        f.write(base64.b64decode(frame_b64))
        tmp_path = f.name

    system_msg = build_system_message()
    start = time.time()
    try:
        with open(tmp_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode()
        messages = []
        if system_msg:
            messages.append({"role": "system", "content": system_msg})
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": full_q},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
            ],
        })
        response = tester.server.chat(messages)
    finally:
        os.unlink(tmp_path)

    duration = time.time() - start

    try:
        memory.store(
            text=f"Q(cam count {obj}): {full_q} | A: {str(response)[:200]}",
            metadata={
                "query_type": "count_objects",
                "modality": "vision",
                "has_image": True,
                "has_audio": False,
                "count_object": obj,
                "agent": "vision_ui",
                "label": "camera_count",
            },
        )
    except Exception:
        pass

    record_event("chat", model=tester.model_name, duration=duration,
                 modality="vision", query_types=["count_objects"])

    return jsonify({
        "response": response,
        "object": obj,
        "question": full_q,
        "model": tester.model_name,
        "timestamp": datetime.now().isoformat(),
        "duration": round(duration, 3),
        "modality": "vision",
        "query_type": "count_objects",
    })


# ============================================================
# Memory API
# ============================================================

@app.route("/api/memory/recall", methods=["POST"])
def memory_recall():
    data = request.json or {}
    query = data.get("query", "")
    k = data.get("k", 5)
    try:
        results = memory.recall(query, k=k)
        return jsonify({"results": results, "total": len(results)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/memory/stats", methods=["GET"])
def memory_stats():
    try:
        stats = memory.get_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/memory/delete", methods=["POST"])
def memory_delete():
    """Delete a memory by ID or clear all memories."""
    data = request.json or {}
    memory_id = data.get("memory_id")
    clear_all = data.get("clear_all", False)
    if clear_all:
        try:
            memory.__init__(db_path=MEMORY_DB)
            return jsonify({"cleared": True})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    if memory_id:
        try:
            conn = sqlite3.connect(MEMORY_DB)
            cursor = conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            deleted = cursor.rowcount > 0
            conn.commit()
            conn.close()
            return jsonify({"memory_id": memory_id, "deleted": deleted})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify({"error": "provide memory_id or clear_all"}), 400


# ============================================================
# MATHIR Daemon Bridge — proxy to the running mathir_mcp daemon
# (separate from the local SimpleMemory above; the daemon has the
# full 23-tool MCP surface including promote/decay/consolidate/link,
# plus the v8.5.1 advanced tools: memory_by_path, memory_recall_quality,
# memory_incoming_links, block_type='auto' auto-classify)
# ============================================================
try:
    import mathir_daemon_client as daemon_client
    DAEMON_BRIDGE_AVAILABLE = True
except ImportError:
    DAEMON_BRIDGE_AVAILABLE = False


@app.route("/api/daemon/status", methods=["GET"])
def daemon_status():
    """Check if the MATHIR daemon (mathir_mcp) is reachable."""
    if not DAEMON_BRIDGE_AVAILABLE:
        return jsonify({"available": False, "error": "mathir_daemon_client module not importable"})
    return jsonify({
        "available": daemon_client.is_daemon_available(),
        "host": daemon_client.get_mathir_daemon_host(),
        "port": daemon_client.get_mathir_daemon_port(),
    })


@app.route("/api/daemon/ping", methods=["GET"])
def daemon_ping():
    """Ping the daemon and return its status."""
    if not DAEMON_BRIDGE_AVAILABLE:
        return jsonify({"error": "mathir_daemon_client module not importable"}), 503
    result = daemon_client.daemon_ping()
    if "error" in result:
        return jsonify(result), 503
    return jsonify(result)


@app.route("/api/daemon/memory/save", methods=["POST"])
def daemon_memory_save():
    """Save a chat message to MATHIR daemon memory (full MCP backend)."""
    if not DAEMON_BRIDGE_AVAILABLE:
        return jsonify({"error": "mathir_daemon_client module not importable"}), 503
    data = request.json or {}
    content = data.get("content", "").strip()
    if not content:
        return jsonify({"error": "content is required"}), 400
    result = daemon_client.memory_save(
        content=content,
        agent=data.get("agent", "vision_ui"),
        block_type=data.get("block_type", "episodic"),
        label=data.get("label", "chat"),
        priority=int(data.get("priority", 5)),
        project=data.get("project", "vision_testing"),
    )
    if "error" in result:
        return jsonify(result), 503
    return jsonify(result)


@app.route("/api/daemon/memory/recall", methods=["POST"])
def daemon_memory_recall():
    """Recall memories from the MATHIR daemon by similarity."""
    if not DAEMON_BRIDGE_AVAILABLE:
        return jsonify({"error": "mathir_daemon_client module not importable"}), 503
    data = request.json or {}
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "query is required"}), 400
    result = daemon_client.memory_recall(
        query=query,
        k=int(data.get("k", 5)),
        project=data.get("project", "vision_testing"),
    )
    if "error" in result:
        return jsonify(result), 503
    return jsonify(result)


@app.route("/api/daemon/memory/stats", methods=["GET"])
def daemon_memory_stats():
    """Get MATHIR daemon memory statistics."""
    if not DAEMON_BRIDGE_AVAILABLE:
        return jsonify({"error": "mathir_daemon_client module not importable"}), 503
    result = daemon_client.memory_stats(project=request.args.get("project", "vision_testing"))
    if "error" in result:
        return jsonify(result), 503
    return jsonify(result)


@app.route("/api/daemon/memory/promote", methods=["POST"])
def daemon_memory_promote():
    """Promote a daemon memory to the next tier."""
    if not DAEMON_BRIDGE_AVAILABLE:
        return jsonify({"error": "mathir_daemon_client module not importable"}), 503
    data = request.json or {}
    memory_id = data.get("memory_id")
    if not memory_id:
        return jsonify({"error": "memory_id is required"}), 400
    result = daemon_client.memory_promote(
        memory_id=memory_id,
        force=bool(data.get("force", False)),
    )
    if "error" in result:
        return jsonify(result), 503
    return jsonify(result)


@app.route("/api/daemon/memory/decay", methods=["POST"])
def daemon_memory_decay():
    """Apply Ebbinghaus decay to old daemon memories."""
    if not DAEMON_BRIDGE_AVAILABLE:
        return jsonify({"error": "mathir_daemon_client module not importable"}), 503
    data = request.json or {}
    result = daemon_client.memory_decay(
        threshold_days=int(data.get("threshold_days", 30)),
    )
    if "error" in result:
        return jsonify(result), 503
    return jsonify(result)


# ============================================================
# Dashboard API — Memory Dashboard (mirrors MCP dashboard)
# ============================================================

def _get_db_conn():
    """Get a SQLite connection to the SimpleMemory DB."""
    conn = sqlite3.connect(MEMORY_DB)
    conn.row_factory = sqlite3.Row
    return conn


@app.route("/api/dashboard/overview", methods=["GET"])
def dashboard_overview():
    """Overview: total memories, metadata summary, DB size."""
    try:
        conn = _get_db_conn()
        total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        
        # Count by provider
        providers = {}
        for row in conn.execute("SELECT provider, COUNT(*) as cnt FROM memories GROUP BY provider"):
            providers[row["provider"]] = row["cnt"]
        
        # Count by model
        models = {}
        for row in conn.execute("SELECT model, COUNT(*) as cnt FROM memories GROUP BY model"):
            models[row["model"]] = row["cnt"]
        
        # Count by metadata keys
        metadata_keys = {}
        for row in conn.execute("SELECT metadata FROM memories WHERE metadata IS NOT NULL"):
            try:
                meta = json.loads(row["metadata"])
                for k in meta.keys():
                    metadata_keys[k] = metadata_keys.get(k, 0) + 1
            except Exception:
                pass
        
        # DB size
        db_size = Path(MEMORY_DB).stat().st_size if Path(MEMORY_DB).exists() else 0
        
        conn.close()
        return jsonify({
            "total_memories": total,
            "providers": providers,
            "models": models,
            "metadata_keys": metadata_keys,
            "db_size_bytes": db_size,
            "db_size_mb": round(db_size / (1024 * 1024), 2),
            "db_path": MEMORY_DB,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/dashboard/memories", methods=["GET"])
def dashboard_memories():
    """List memories with pagination and optional search."""
    try:
        limit = int(request.args.get("limit", 100))
        offset = int(request.args.get("offset", 0))
        search = request.args.get("search", "")
        
        conn = _get_db_conn()
        
        if search:
            # Use FTS5 search
            try:
                rows = conn.execute("""
                    SELECT m.id, m.text, m.metadata, m.provider, m.model, m.created_at, rank
                    FROM memories_fts fts
                    JOIN memories m ON m.id = fts.rowid
                    WHERE memories_fts MATCH ?
                    ORDER BY rank
                    LIMIT ? OFFSET ?
                """, (search, limit, offset)).fetchall()
                total = conn.execute("""
                    SELECT COUNT(*) FROM memories_fts WHERE memories_fts MATCH ?
                """, (search,)).fetchone()[0]
            except Exception:
                rows = conn.execute("""
                    SELECT id, text, metadata, provider, model, created_at, 0 as rank
                    FROM memories WHERE text LIKE ?
                    ORDER BY id DESC LIMIT ? OFFSET ?
                """, (f"%{search}%", limit, offset)).fetchall()
                total = conn.execute(
                    "SELECT COUNT(*) FROM memories WHERE text LIKE ?",
                    (f"%{search}%",)
                ).fetchone()[0]
        else:
            rows = conn.execute(
                "SELECT id, text, metadata, provider, model, created_at, 0 as rank FROM memories ORDER BY id DESC LIMIT ? OFFSET ?",
                (limit, offset)
            ).fetchall()
            total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        
        memories = []
        for row in rows:
            meta = {}
            try:
                meta = json.loads(row["metadata"]) if row["metadata"] else {}
            except Exception:
                pass
            memories.append({
                "id": row["id"],
                "text": row["text"][:200] if row["text"] else "",
                "metadata": meta,
                "provider": row["provider"],
                "model": row["model"],
                "created_at": row["created_at"],
                "score": abs(row["rank"]) if row["rank"] else 0.0,
            })
        
        conn.close()
        return jsonify({
            "memories": memories,
            "total": total,
            "limit": limit,
            "offset": offset,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/dashboard/timeline", methods=["GET"])
def dashboard_timeline():
    """Timeline: memories bucketed by hour/day."""
    try:
        conn = _get_db_conn()
        rows = conn.execute(
            "SELECT created_at FROM memories WHERE created_at IS NOT NULL ORDER BY created_at"
        ).fetchall()
        conn.close()
        
        buckets = {}
        for row in rows:
            try:
                ts = row["created_at"]
                if ts:
                    # Bucket by hour
                    key = ts[:13] + ":00"  # "2026-06-19T14:00"
                    buckets[key] = buckets.get(key, 0) + 1
            except Exception:
                pass
        
        timeline = [{"time": k, "count": v} for k, v in sorted(buckets.items())]
        return jsonify({"timeline": timeline})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/dashboard/providers", methods=["GET"])
def dashboard_providers():
    """Provider breakdown."""
    try:
        conn = _get_db_conn()
        providers = {}
        for row in conn.execute("SELECT provider, COUNT(*) as cnt FROM memories GROUP BY provider"):
            providers[row["provider"]] = row["cnt"]
        conn.close()
        return jsonify({"providers": providers})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/dashboard/search", methods=["POST"])
def dashboard_search():
    """Search memories using FTS5."""
    try:
        data = request.json or {}
        query = data.get("query", "")
        k = data.get("k", 10)
        
        if not query:
            return jsonify({"results": [], "total": 0})
        
        results = memory.recall(query, k=k)
        return jsonify({"results": results, "total": len(results)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/dashboard/config", methods=["GET"])
def dashboard_config():
    """Memory configuration."""
    try:
        return jsonify({
            "backend": "SimpleMemory (FTS5)",
            "embedding_dim": 0,
            "db_path": MEMORY_DB,
            "db_exists": Path(MEMORY_DB).exists(),
            "db_size_bytes": Path(MEMORY_DB).stat().st_size if Path(MEMORY_DB).exists() else 0,
            "features": {
                "fts5_search": True,
                "vector_search": False,
                "tier_system": False,
                "agent_tracking": True,
                "metadata": True,
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# Learning API (for graphs)
# ============================================================

@app.route("/api/learning/data", methods=["GET"])
def learning_data_endpoint():
    """Return aggregated learning data for the Learning view graphs."""
    with learning_lock:
        # Compute aggregates
        rt_aggregates = {}
        for model, durations in learning_data["response_times"].items():
            if durations:
                rt_aggregates[model] = {
                    "avg": round(sum(durations) / len(durations), 3),
                    "min": round(min(durations), 3),
                    "max": round(max(durations), 3),
                    "count": len(durations),
                    "recent": [round(d, 3) for d in durations[-20:]],
                }

        # Recall accuracy
        total = learning_data["recalls"]["total"]
        nonempty = learning_data["recalls"]["nonempty"]
        accuracy = (nonempty / total * 100) if total > 0 else 0

        return jsonify({
            "memory_timeline": learning_data["memory_timeline"][-100:],
            "storage_timeline": learning_data["storage_timeline"][-100:],
            "response_times": rt_aggregates,
            "query_types": learning_data["query_types"],
            "modality_usage": learning_data["modality_usage"],
            "recall_stats": {
                "total": total,
                "nonempty": nonempty,
                "accuracy": round(accuracy, 1),
            },
            "started_at": learning_data["started_at"],
            "last_event": learning_data["last_event"],
        })


# ============================================================
# Learning stats - structured view for graphs (TASK 2)
# ============================================================

def _memory_growth_from_mathir() -> List[Dict[str, Any]]:
    """Get memory growth data from SimpleMemory.

    Returns a list of {date, count} for cumulative memory growth.
    """
    try:
        stats = memory.get_stats()
        total = stats.get("total_memories", 0)
        if total > 0:
            return [{"date": datetime.now().strftime("%Y-%m-%d"), "count": total}]
    except Exception:
        pass
    # Fallback: return in-memory timeline
    with learning_lock:
        return list(learning_data.get("memory_timeline", []))



def _model_performance_from_inmem() -> List[Dict[str, Any]]:
    """Aggregate per-model response time stats from in-memory learning_data."""
    out: List[Dict[str, Any]] = []
    with learning_lock:
        for model, durations in learning_data["response_times"].items():
            if not durations:
                continue
            avg = sum(durations) / len(durations)
            out.append({
                "model": model,
                "avg_latency_ms": round(avg * 1000.0, 2),  # seconds -> ms
                "calls": len(durations),
                "min_ms": round(min(durations) * 1000.0, 2),
                "max_ms": round(max(durations) * 1000.0, 2),
            })
    # Sort by call count desc — busiest models first
    out.sort(key=lambda r: r["calls"], reverse=True)
    return out


def _query_types_from_inmem() -> Dict[str, int]:
    """Return query-type histogram from in-memory counters."""
    with learning_lock:
        # Return a copy to avoid races
        return dict(learning_data.get("query_types", {}))


def _recall_accuracy_from_inmem() -> Dict[str, Any]:
    """Compute recall accuracy = nonempty / total * 100."""
    with learning_lock:
        total = int(learning_data["recalls"].get("total", 0))
        nonempty = int(learning_data["recalls"].get("nonempty", 0))
    accuracy = (nonempty / total * 100.0) if total > 0 else 0.0
    return {
        "successful": nonempty,
        "total": total,
        "accuracy": round(accuracy, 1),
    }


@app.route("/api/learning/stats", methods=["GET"])
def learning_stats():
    """Return MATHIR learning stats for the Learning view.

    Schema (per spec):
      - memory_growth: list of {date, count} (cumulative)
      - storage_size: int (bytes)
      - model_performance: list of {model, avg_latency_ms, calls}
      - query_types: {type: count}
      - recall_accuracy: {successful, total, accuracy}
    """
    memory_growth = _memory_growth_from_mathir()
    model_performance = _model_performance_from_inmem()
    query_types = _query_types_from_inmem()
    recall_accuracy = _recall_accuracy_from_inmem()

    # Get storage size from SimpleMemory DB file
    storage_size = 0
    try:
        db_file = Path(MEMORY_DB)
        if db_file.exists():
            storage_size = db_file.stat().st_size
    except Exception:
        pass

    return jsonify({
        "memory_growth": memory_growth,
        "storage_size": storage_size,
        "model_performance": model_performance,
        "query_types": query_types,
        "recall_accuracy": recall_accuracy,
        "generated_at": datetime.now().isoformat(),
    })


# ============================================================
# Tests API
# ============================================================

@app.route("/api/tests", methods=["GET"])
def tests_endpoint():
    """Return test results for the Tests view."""
    return _load_test_results()


@app.route("/api/test_results", methods=["GET"])
def test_results_endpoint():
    """Return the full test_results.json content as JSON.

    Same payload as /api/tests - aliased for clarity in the Tests view
    (the orchestrator and tests view call /api/test_results).
    """
    return _load_test_results()


def _load_test_results():
    """Shared loader for /api/tests and /api/test_results."""
    path = HERE / "test_results.json"
    if not path.exists():
        return jsonify({"error": "test_results.json not found", "tests": [], "categories": []}), 404
    try:
        with open(path) as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e), "tests": [], "categories": []}), 500


# ============================================================
# Accuracy Test API
# ============================================================
# These endpoints drive the Accuracy view in the UI. They cover the full
# loop: list available tests, run a battery against the active (or named)
# model, fetch per-model results, and compare across models.
#
# We don't auto-import the framework at module load because that drags in
# PIL + sentence-transformers which we don't need for every request. The
# lazy import is wrapped in a guard so a single broken import doesn't
# kill the whole server.

# Cache of long-lived test-image summaries (cheap to build, expensive to
# recompute) so the GET /api/accuracy/tests endpoint stays sub-millisecond
_accuracy_tests_cache: Dict[str, Any] = {"data": None, "built_at": 0.0}
_ACCURACY_CACHE_TTL = 30.0  # seconds


def _accuracy_framework_or_error() -> Tuple[Optional[Any], Optional[Response]]:
    """Return the framework class or a Flask error response.

    Use as: `fw_cls, err = _accuracy_framework_or_error(); if err: return err`
    Centralizes the 'is this thing even importable' check.
    """
    if AccuracyTestFramework is None:
        return None, jsonify({
            "error": "AccuracyTestFramework unavailable",
            "detail": _accuracy_framework_error or "import failed",
        }), 500
    return AccuracyTestFramework, None


def _read_test_results_json() -> Dict[str, Any]:
    """Read the on-disk test_results.json with a safe default."""
    path = HERE / "test_results.json"
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def _ensure_test_images_built() -> None:
    """Regenerate the synthetic test images on disk if they're missing.

    The test framework's `generate_test_images()` returns PIL objects in
    memory, so for the UI we also need actual PNG files (so the Accuracy
    view can display thumbnails and the user can see what was tested).
    """
    images_dir = HERE / "accuracy_benchmarks" / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    # If at least one PNG + JSON pair exists we assume the battery is built
    has_any = any(images_dir.glob("*.png")) and any(images_dir.glob("*.json"))
    if has_any:
        return
    try:
        from synthetic_images import generate_test_images, save_test_image
        for scene in generate_test_images():
            png_dest = images_dir / f"{scene['name']}.png"
            meta_dest = images_dir / f"{scene['name']}.json"
            save_test_image(scene["image"], png_dest)
            meta = {k: v for k, v in scene.items() if k != "image"}
            with open(meta_dest, "w") as f:
                json.dump(meta, f, indent=2)
    except Exception as e:  # pragma: no cover
        print(f"[ui_server] WARN: could not build test images: {e}")


def _list_test_images() -> List[Dict[str, Any]]:
    """Read every synthetic test image's metadata from disk."""
    _ensure_test_images_built()
    out: List[Dict[str, Any]] = []
    images_dir = HERE / "accuracy_benchmarks" / "images"
    if not images_dir.exists():
        return out
    # Stable order: sort by filename
    for meta_path in sorted(images_dir.glob("*.json")):
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except Exception:
            continue
        png_path = meta_path.with_suffix(".png")
        # Build a relative URL the frontend can use as <img src=...>
        rel = png_path.relative_to(HERE / "accuracy_benchmarks").as_posix()
        out.append({
            "name": meta.get("name", meta_path.stem),
            "category": meta.get("category", "unknown"),
            "description": meta.get("description", ""),
            "ground_truth": meta.get("ground_truth", {}),
            "image_url": f"/static-accuracy/{rel}",
        })
    return out


@app.route("/api/accuracy/tests", methods=["GET"])
def accuracy_tests_list():
    """List every available test image with category + ground truth.

    Used by the Accuracy view to render the test catalog and by the run
    endpoint to validate user-supplied test names.
    """
    now = time.time()
    if (_accuracy_tests_cache["data"] is None
            or now - _accuracy_tests_cache["built_at"] > _ACCURACY_CACHE_TTL):
        images = _list_test_images()
        # Group by category for the UI's filter sidebar
        categories: Dict[str, int] = {}
        for img in images:
            cat = img.get("category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1
        _accuracy_tests_cache["data"] = {
            "images": images,
            "categories": categories,
            "test_types": list(ALL_TEST_TYPES),
            "total": len(images),
        }
        _accuracy_tests_cache["built_at"] = now
    return jsonify(_accuracy_tests_cache["data"])


@app.route("/static-accuracy/<path:relpath>", methods=["GET"])
def accuracy_image_serve(relpath):
    """Serve a synthetic test image from accuracy_benchmarks/images/.

    Sandboxed to the accuracy_benchmarks directory - no path traversal.
    """
    target = (HERE / "accuracy_benchmarks" / relpath).resolve()
    if not str(target).startswith(str((HERE / "accuracy_benchmarks").resolve())):
        return jsonify({"error": "forbidden"}), 403
    if not target.exists() or not target.is_file():
        return jsonify({"error": "not found", "path": relpath}), 404
    # send_from_directory needs a directory + filename, so split
    return send_from_directory(str(target.parent), target.name)


@app.route("/api/accuracy/test", methods=["POST"])
def accuracy_run_test():
    """Run the full accuracy battery against the active (or specified) model.

    Request body (all optional):
        {
            "model":      "<name>",     # defaults to active model
            "tests":      ["..."],      # optional list of test image names; runs all if omitted
            "save":       true,         # persist results to test_results.json (default: true)
        }

    NOTE: this endpoint can take 1-10 minutes to run because it makes many
    sequential inference calls. The UI should show a spinner and let the
    user navigate away. We use a thread to keep the request non-blocking.
    """
    fw_cls, err = _accuracy_framework_or_error()
    if err:
        return err

    data = request.json or {}
    model_name = (data.get("model") or "").strip() or state.get("active_model")
    if not model_name:
        return jsonify({"error": "No model specified and no active model"}), 400

    # Validate the model exists
    if model_name not in state["models"].list_models(enabled_only=False):
        return jsonify({"error": f"Model '{model_name}' not found in config"}), 404

    save = bool(data.get("save", True))
    only_tests = data.get("tests")  # optional filter

    # Run in a background thread - the UI can poll /api/accuracy/results
    # to see the file appear once it lands
    import threading as _thr

    def _runner():
        try:
            fw = fw_cls(model_name, port=state["ui_config"]["server"]["port"] + 1)
            try:
                fw.start()
                # If the user supplied a test filter, monkey-patch the
                # available-test list so we only run those
                if only_tests:
                    only_set = set(only_tests)
                    original = fw.generate_test_images
                    def _filtered():
                        return [s for s in original() if s["name"] in only_set]
                    fw.generate_test_images = _filtered
                results = fw.run_full_battery()
                if save:
                    fw.save_results(results)
                # also stash in state for /api/accuracy/results to return
                state.setdefault("_accuracy_last_runs", {})[model_name] = results
            finally:
                fw.stop()
        except Exception as e:
            print(f"[ui_server] accuracy_run_test error: {e}")
            import traceback
            traceback.print_exc()

    t = _thr.Thread(target=_runner, daemon=True)
    t.start()

    return jsonify({
        "status": "started",
        "model": model_name,
        "tests": only_tests or "all",
        "save": save,
        "message": "Battery running in background. Poll /api/accuracy/results to see completion.",
    }), 202  # 202 Accepted - the request is being processed


@app.route("/api/accuracy/results", methods=["GET"])
def accuracy_results_all():
    """Return every model's accuracy results from test_results.json.

    Response shape:
        {
            "models": {
                "<model>": {
                    "last_run", "overall_score", "tests_passed", "tests_run",
                    "by_type", "by_image", "per_test": [...]
                },
                ...
            },
            "test_battery": {...}  # static catalog of test images
        }
    """
    data = _read_test_results_json()
    acc = data.get("accuracy_tests", {}) or {}
    # Strip the meta keys (start with _) from the model map
    models_block: Dict[str, Any] = {}
    for k, v in acc.items():
        if k.startswith("_"):
            continue
        models_block[k] = v
    return jsonify({
        "models": models_block,
        "test_battery": acc.get("_test_battery", {}),
        "model_count": len(models_block),
    })


@app.route("/api/accuracy/results/<model_name>", methods=["GET"])
def accuracy_results_for_model(model_name: str):
    """Return one model's accuracy results.

    404 if the model has never been run through the framework.
    """
    data = _read_test_results_json()
    acc = data.get("accuracy_tests", {}) or {}
    block = acc.get(model_name)
    if not block:
        return jsonify({"error": f"No accuracy results for '{model_name}'"}), 404
    return jsonify({"model": model_name, "results": block})


@app.route("/api/accuracy/compare", methods=["POST"])
def accuracy_compare():
    """Compare accuracy results across models.

    Request body (optional):
        {"models": ["name1", "name2", ...]}

    If `models` is omitted, compares all models that have results.
    Response shape:
        {
            "models": [list of model names],
            "by_type": {
                "<test_type>": {
                    "<model>": avg_score,
                    ...
                },
                ...
            },
            "by_image": {
                "<image_name>": {
                    "<model>": avg_score,
                    ...
                },
                ...
            },
            "overall": {
                "<model>": overall_score,
                ...
            }
        }
    """
    data = _read_test_results_json()
    acc = data.get("accuracy_tests", {}) or {}
    full = {k: v for k, v in acc.items() if not k.startswith("_")}

    requested = (request.json or {}).get("models")
    if requested:
        models = [m for m in requested if m in full]
    else:
        models = sorted(full.keys())

    if not models:
        return jsonify({"error": "No models with accuracy results to compare"}), 404

    # Build the comparison tables
    by_type: Dict[str, Dict[str, float]] = {}
    by_image: Dict[str, Dict[str, float]] = {}
    overall: Dict[str, float] = {}
    for m in models:
        block = full[m]
        overall[m] = float(block.get("overall_score", 0.0))
        for tt, stats in (block.get("by_type") or {}).items():
            by_type.setdefault(tt, {})[m] = float(stats.get("avg_score", 0.0))
        for img, stats in (block.get("by_image") or {}).items():
            by_image.setdefault(img, {})[m] = float(stats.get("avg_score", 0.0))

    return jsonify({
        "models": models,
        "by_type": by_type,
        "by_image": by_image,
        "overall": overall,
    })


# ============================================================
# Settings API
# ============================================================

@app.route("/api/settings", methods=["GET"])
def get_settings():
    """Return merged settings: UI config + OpenRouter + MATHIR daemon + .env defaults.

    Sensitive values (api_key) are returned as-is. The UI should mask them when displaying.
    """
    cfg = dict(state["ui_config"])
    # Merge OpenRouter config from config.json (api_key, api_base, timeout, max_retries)
    try:
        vt_cfg = vt_load_config()
        cfg["openrouter"] = {
            "api_key": vt_cfg.get("openrouter", {}).get("api_key", "") or "",
            "api_base": vt_cfg.get("openrouter", {}).get("api_base", "https://openrouter.ai/api/v1"),
            "timeout_seconds": vt_cfg.get("openrouter", {}).get("timeout_seconds", 120),
            "max_retries": vt_cfg.get("openrouter", {}).get("max_retries", 2),
        }
        cfg["models"] = vt_cfg.get("models", {})
    except Exception:
        cfg["openrouter"] = {"api_key": "", "api_base": "https://openrouter.ai/api/v1"}
    # MATHIR daemon config (from env_config)
    try:
        from env_config import get_mathir_daemon_host, get_mathir_daemon_port
        cfg["mathir_daemon"] = {
            "host": get_mathir_daemon_host(),
            "port": get_mathir_daemon_port(),
        }
    except ImportError:
        cfg["mathir_daemon"] = {"host": "127.0.0.1", "port": 7338}
    return jsonify(cfg)


@app.route("/api/settings", methods=["POST"])
def update_settings():
    """Update settings — writes to ui_config.json (UI-only) AND config.json (openrouter)."""
    cfg = request.json or {}
    # Handle reset to defaults
    if cfg.get("reset"):
        try:
            # Reload the original ui_config from disk (factory defaults)
            default_path = HERE / "ui_config.json"
            if default_path.exists():
                with open(default_path) as f:
                    state["ui_config"] = json.load(f)
                save_ui_config(state["ui_config"])
                return jsonify({"status": "reset_to_defaults"})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # OpenRouter → config.json (persistent)
    if "openrouter" in cfg:
        try:
            vt_cfg = vt_load_config()
            or_cfg = vt_cfg.setdefault("openrouter", {})
            for k in ("api_key", "api_base", "timeout_seconds", "max_retries"):
                if k in cfg["openrouter"]:
                    or_cfg[k] = cfg["openrouter"][k]
            with open(CONFIG_PATH, "w") as f:
                json.dump(vt_cfg, f, indent=2)
            # Reload config in vision_test module so subsequent calls see new key
            import importlib
            import vision_test
            importlib.reload(vision_test)
        except Exception as e:
            return jsonify({"error": f"Failed to write config.json: {e}"}), 500

    # MATHIR daemon → env vars (runtime, not persistent)
    if "mathir_daemon" in cfg:
        os.environ["MATHIR_DAEMON_HOST"] = str(cfg["mathir_daemon"].get("host", "127.0.0.1"))
        os.environ["MATHIR_DAEMON_PORT"] = str(cfg["mathir_daemon"].get("port", 7338))

    # UI / camera / audio → ui_config.json
    for top_level in ("ui", "camera", "audio", "server"):
        if top_level in cfg:
            state["ui_config"][top_level] = cfg[top_level]
    save_ui_config(state["ui_config"])
    return jsonify({"status": "updated"})


@app.route("/api/openrouter/test", methods=["POST"])
def test_openrouter():
    """Test the OpenRouter API key by sending a minimal request."""
    try:
        from vision_test import OpenRouterClient
        # Use the default model for the test
        cfg = vt_load_config()
        models = cfg.get("models", {})
        enabled = [k for k, m in models.items() if m.get("enabled", True)]
        if not enabled:
            return jsonify({"error": "No enabled models in config.json"}), 400
        test_model_id = enabled[0]
        model_paths = models[test_model_id]
        client = OpenRouterClient(model_paths)
        if not client.api_key:
            return jsonify({"error": "API key not set"}), 400
        result = client.chat(
            messages=[{"role": "user", "content": "Reply with the single word: pong"}],
            max_tokens=10,
            temperature=0.0,
        )
        if isinstance(result, str) and result.startswith("ERROR"):
            return jsonify({"error": result}), 502
        return jsonify({"status": "ok", "model": test_model_id, "sample": result[:200]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/system/info", methods=["GET"])
def system_info():
    info = {
        "platform": sys.platform,
        "python": sys.version,
        "active_model": state.get("active_model"),
        "camera_running": state.get("camera_running", False),
        "config_path": str(HERE / "config.json"),
        "ui_config_path": str(HERE / "ui_config.json"),
        "system_context_path": str(HERE / "system_context.json"),
        "test_results_path": str(HERE / "test_results.json"),
        "memory_db_path": MEMORY_DB,
    }
    return jsonify(info)


# ============================================================
# Main
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", help="Override server host")
    parser.add_argument("--port", type=int, help="Override server port")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    args = parser.parse_args()

    host = args.host or state["ui_config"]["server"]["host"]
    port = args.port or state["ui_config"]["server"]["port"]
    debug = args.debug or state["ui_config"]["server"]["debug"]

    print("=" * 60)
    print("MATHIR Playground Server (v8.5.1)")
    print("=" * 60)
    print(f"URL: http://{host}:{port}")
    print(f"Active model: {state.get('active_model') or 'none'}")
    print(f"Camera: {state['ui_config']['camera']['enabled']}")
    print(f"System context: {(HERE / 'system_context.json').exists()}")
    print(f"Test results: {(HERE / 'test_results.json').exists()}")
    print(f"Local llama.cpp: {'YES (llama-cpp-python installed)' if LLAMA_CPP_AVAILABLE else 'NO (optional: pip install llama-cpp-python)'}")
    print()

    try:
        import flask
        print(f"Flask: {flask.__version__}")
    except ImportError:
        print("WARNING: flask not installed. Run: pip install flask flask-cors")
        return 1

    try:
        import cv2
        print(f"OpenCV: {cv2.__version__}")
    except ImportError:
        print("WARNING: opencv-python not installed. Run: pip install opencv-python")
        print("         Camera features will not work.")

    print()
    # Auto-load first enabled model in background
    threading.Thread(target=_preload_model, daemon=True).start()
    app.run(host=host, port=port, debug=debug, threaded=True)


def _preload_model():
    """Auto-load the first enabled model at startup (background thread)."""
    import time
    time.sleep(2)  # wait for Flask to start
    try:
        enabled = state["models"].list_models(enabled_only=True)
        if enabled:
            first = list(enabled.keys())[0]
            state["active_model"] = first
            print(f"[PRELOAD] Auto-loading model: {first}")
            tester = VisionTester(first, port=state["ui_config"]["server"]["port"] + 1)
            tester.setup_memory()
            tester.start_server()
            state["active_tester"] = tester
            state["active_server"] = tester.server
            print(f"[PRELOAD] Model {first} loaded successfully")
    except Exception as e:
        print(f"[PRELOAD] Failed to auto-load model: {e}")


if __name__ == "__main__":
    sys.exit(main() or 0)
