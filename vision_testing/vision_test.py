#!/usr/bin/env python3
"""
MATHIR Playground — Vision Testing Interface
============================================

A playground to chat with vision/audio LLMs and exercise the MATHIR memory
backend. NO HARDCODED PATHS. All paths come from config.json (relative to
this file). User can add/remove models by editing config.json — no code
changes needed.

Tests MATHIR memory by:
- Setting up a virtual room with objects
- Asking models what's in the room
- "Moving" objects
- Asking what was done
- Storing interactions in MATHIR memory
- Testing recall across models

v8.4.0 MIGRATION: Switched from local llama.cpp binaries to OpenRouter cloud API.
All model invocations now go through the OpenRouterClient class below.
The config.json `llama_server` section has been replaced with `openrouter`.
"""
import os
import sys
import json
import time
import base64
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any

# All paths RELATIVE to this file - works from any clone location
HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "config.json"

# Load .env early so OpenRouter keys are available before any client instantiation
from env_config import load_env, get_openrouter_api_key  # noqa: E402
load_env()

# MATHIR drop-in — direct import, no daemon needed
MATHIR_ROOT = HERE.parent
sys.path.insert(0, str(MATHIR_ROOT))
from mathir_dropin.simple import SimpleMemory


def load_config():
    """Load config from config.json (relative to this file)."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"config.json not found at {CONFIG_PATH}. "
            f"Please create it or copy from a template."
        )
    with open(CONFIG_PATH) as f:
        return json.load(f)


def save_config(config):
    """Save config back to disk."""
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)


def resolve_path(relative_or_absolute):
    """Resolve a path: if absolute, use as-is; if relative, resolve against HERE."""
    p = Path(relative_or_absolute)
    if p.is_absolute():
        return p
    return (HERE / p).resolve()


class ModelManager:
    """
    Manages vision/audio models from config.json.
    User adds models by editing config.json - no code changes needed.
    """

    def __init__(self, config: dict):
        self.config = config
        self._cache = None

    def list_models(self, enabled_only: bool = True) -> Dict[str, dict]:
        """List all models from config."""
        models = self.config.get("models", {})
        if enabled_only:
            return {k: v for k, v in models.items() if v.get("enabled", True)}
        return models

    def get_model(self, name: str) -> Optional[dict]:
        """Get a specific model by name (key in config)."""
        return self.config.get("models", {}).get(name)

    def add_model(self, name: str, config_entry: dict):
        """Add or update a model in config.json."""
        if "models" not in self.config:
            self.config["models"] = {}
        self.config["models"][name] = config_entry
        save_config(self.config)

    def remove_model(self, name: str) -> bool:
        """Remove a model from config.json."""
        if name in self.config.get("models", {}):
            del self.config["models"][name]
            save_config(self.config)
            return True
        return False

    def toggle_model(self, name: str, enabled: bool):
        """Enable or disable a model without removing it."""
        if name in self.config.get("models", {}):
            self.config["models"][name]["enabled"] = enabled
            save_config(self.config)

    def get_model_paths(self, name: str) -> Optional[dict]:
        """Get resolved absolute paths for a model's files."""
        m = self.get_model(name)
        if not m:
            return None
        result = {
            "name": name,
            "type": m.get("type"),
            "display_name": m.get("display_name", name),
            "description": m.get("description", ""),
            "path": resolve_path(m["path"]) if m.get("path") else None,
            "supports_vision": m.get("supports_vision", False),
            "supports_audio": m.get("supports_audio", False),
            "context_length": m.get("context_length", 4096),
            "vram_mb": m.get("vram_mb", 0),
        }
        # Optional components
        for key in ["mmproj", "tokenizer", "vocoder"]:
            if m.get(key):
                result[key] = resolve_path(m[key])
        return result

    def validate_model(self, name: str) -> Dict[str, bool]:
        """Check if all files for a model exist."""
        paths = self.get_model_paths(name)
        if not paths:
            return {"valid": False, "reason": "Model not found"}
        result = {"valid": True, "files": {}}
        for key in ["path", "mmproj", "tokenizer", "vocoder"]:
            p = paths.get(key)
            if p:
                result["files"][key] = p.exists()
                if not p.exists():
                    result["valid"] = False
        return result


class OpenRouterClient:
    """OpenRouter cloud API client for vision/language models.

    Replaces the old LlamaServer class (v8.4.0 migration). No more local
    GGUF binaries — all inference goes through https://openrouter.ai/api/v1.

    Models are configured in config.json under the `openrouter` section.
    Each model entry specifies its `id` (e.g. "google/gemini-2.0-flash-exp:free")
    and what features it supports (vision, audio, grounding, etc.).
    """

    def __init__(self, model_paths: dict):
        self.model_paths = model_paths
        self.model_id = model_paths["id"]
        self.display_name = model_paths.get("display_name", self.model_id)
        self._config = load_config()
        # Determine provider: explicit "provider" field, else default to openrouter
        provider = model_paths.get("provider", "openrouter").lower()
        self.provider = provider
        if provider == "opencode_zen":
            zen_cfg = self._config.get("opencode_zen", {})
            from env_config import get_opencode_zen_api_key
            self.api_key = zen_cfg.get("api_key") or get_opencode_zen_api_key()
            self.api_base = zen_cfg.get("api_base", "https://opencode.ai/zen/v1")
            self.timeout = zen_cfg.get("timeout_seconds", 120)
            self.max_retries = zen_cfg.get("max_retries", 2)
            if not self.api_key:
                print(f"  [WARN] OpenCode Zen API key not set for {self.display_name}.")
                print(f"         Set in config.json:opencode_zen.api_key, or .env, or OPENCODE_ZEN_API_KEY env var.")
        else:  # default: openrouter
            or_cfg = self._config.get("openrouter", {})
            # Priority: config.json > .env > env var (env var already loaded by env_config)
            self.api_key = or_cfg.get("api_key") or get_openrouter_api_key()
            self.api_base = or_cfg.get("api_base", "https://openrouter.ai/api/v1")
            self.timeout = or_cfg.get("timeout_seconds", 120)
            self.max_retries = or_cfg.get("max_retries", 2)
            if not self.api_key:
                print(f"  [WARN] OpenRouter API key not set for {self.display_name}.")
                print(f"         Set in config.json:openrouter.api_key, or .env, or OPENROUTER_API_KEY env var.")

    def chat(self, messages, max_tokens=512, temperature=0.7):
        """Send a chat completion request to OpenRouter or OpenCode Zen."""
        if not self.api_key:
            return f"ERROR: {self.provider} API key not configured (see config.json:{self.provider}.api_key)"
        # Convert messages to OpenAI Responses format if needed (for OpenCode Zen)
        data = self._build_request_body(messages, max_tokens, temperature)
        for attempt in range(self.max_retries + 1):
            try:
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                    # OpenCode Zen REQUIRES a non-empty User-Agent (else 403)
                    "User-Agent": "opencode-cli/1.0",
                }
                # Provider-specific attribution headers + endpoint
                if self.provider == "openrouter":
                    headers["HTTP-Referer"] = "https://github.com/SECRET_PROJECT/MATHIR"
                    headers["X-Title"] = "MATHIR Playground"
                    endpoint = f"{self.api_base}/chat/completions"
                elif self.provider == "opencode_zen":
                    # OpenCode Zen uses the OpenAI Responses API at /v1/responses
                    endpoint = f"{self.api_base}/responses"
                else:
                    endpoint = f"{self.api_base}/chat/completions"
                req = urllib.request.Request(
                    endpoint,
                    data=data,
                    headers=headers,
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    result = json.loads(resp.read())
                    # OpenAI Responses API has a different response shape
                    if "output" in result and isinstance(result["output"], list):
                        # New OpenAI Responses format
                        for item in result["output"]:
                            if item.get("type") == "message":
                                content = item.get("content", [])
                                if isinstance(content, list):
                                    for c in content:
                                        if c.get("type") == "output_text":
                                            return c.get("text", "")
                                return str(content)
                        return str(result["output"])
                    # Classic OpenAI chat.completions format
                    return result["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as e:
                body = e.read().decode(errors="replace")[:300]
                if e.code == 429 and attempt < self.max_retries:
                    # Rate-limited — wait then retry with exponential backoff
                    wait = 2 ** attempt
                    print(f"  [RETRY] 429 rate-limited, waiting {wait}s (attempt {attempt+1}/{self.max_retries})")
                    time.sleep(wait)
                    continue
                return f"ERROR: HTTP {e.code} - {body}"
            except Exception as e:
                if attempt < self.max_retries:
                    wait = 2 ** attempt
                    print(f"  [RETRY] {type(e).__name__}, waiting {wait}s")
                    time.sleep(wait)
                    continue
                return f"ERROR: {type(e).__name__}: {e}"
        return "ERROR: exhausted retries"

    def _build_request_body(self, messages, max_tokens, temperature):
        """Build the request body in the right format for the provider."""
        if self.provider == "opencode_zen":
            # OpenAI Responses API format: input is a list of role/content items
            input_items = []
            for m in messages:
                role = m.get("role", "user")
                content = m.get("content", "")
                if isinstance(content, str):
                    input_items.append({
                        "role": role,
                        "content": [{"type": "input_text", "text": content}],
                    })
                elif isinstance(content, list):
                    # Already in multimodal format
                    input_items.append({"role": role, "content": content})
            return json.dumps({
                "model": self.model_id,
                "input": input_items,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }).encode()
        # Default: OpenAI chat.completions format
        return json.dumps({
            "model": self.model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }).encode()

    def chat_with_image(self, text, image_path, max_tokens=512):
        """Chat with an image attachment (works with vision-capable models)."""
        if not self.model_paths.get("supports_vision"):
            return f"ERROR: {self.display_name} does not support vision"
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode()
        # Detect image MIME type from extension
        ext = Path(image_path).suffix.lower()
        mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".png": "image/png", ".webp": "image/webp",
                ".gif": "image/gif"}.get(ext, "image/jpeg")
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_data}"}},
            ],
        }]
        return self.chat(messages, max_tokens=max_tokens)


class VirtualRoom:
    """Virtual room with objects for testing vision models + MATHIR memory."""

    def __init__(self):
        self.objects = {}
        self.interactions = []

    def add_object(self, name, position, properties=None):
        self.objects[name] = {
            "position": position,
            "properties": properties or {},
            "added_at": datetime.now().isoformat(),
            "moved_from": None,
        }
        self.interactions.append(("add", name, position))
        print(f"  [ADD] {name} at {position}")

    def move_object(self, name, new_position):
        if name in self.objects:
            self.objects[name]["moved_from"] = self.objects[name]["position"]
            self.objects[name]["position"] = new_position
            self.interactions.append(("move", name, new_position))
            print(f"  [MOVE] {name} -> {new_position}")
        else:
            print(f"  [WARN] {name} not in room")

    def remove_object(self, name):
        if name in self.objects:
            old_pos = self.objects[name]["position"]
            del self.objects[name]
            self.interactions.append(("remove", name, old_pos))
            print(f"  [REMOVE] {name}")

    def get_state_description(self):
        lines = ["Current room state:"]
        for name, info in self.objects.items():
            lines.append(f"  - {name} at {info['position']}")
        return "\n".join(lines)

    def get_changes_since(self, idx):
        return self.interactions[idx:]

    def describe_changes(self, since_idx):
        changes = self.get_changes_since(since_idx)
        descriptions = []
        for action, name, info in changes:
            if action == "add":
                descriptions.append(f"Added {name} at {info}")
            elif action == "move":
                obj = self.objects.get(name, {})
                moved_from = obj.get("moved_from", "unknown")
                descriptions.append(f"Moved {name} from {moved_from} to {info}")
            elif action == "remove":
                descriptions.append(f"Removed {name} (was at {info})")
        return "; ".join(descriptions) if descriptions else "No changes"


class VisionTester:
    """Main test orchestrator with MATHIR memory integration."""

    def __init__(self, model_name: str, port: int = 8080,
                 config: Optional[dict] = None):
        if config is None:
            config = load_config()
        self.model_name = model_name
        self.config = config
        self.models = ModelManager(config)
        self.model_paths = self.models.get_model_paths(model_name)
        if not self.model_paths:
            available = list(self.models.list_models().keys())
            raise ValueError(
                f"Model '{model_name}' not found in config.json. "
                f"Available: {available}. "
                f"Edit config.json to add it."
            )
        self.port = port
        self.server = None
        self.room = VirtualRoom()
        self.memory = None

    def setup_memory(self, db_path=None):
        """Initialize SimpleMemory with a local SQLite DB."""
        if db_path is None:
            db_path = str(HERE / "memory.db")
        self.memory = SimpleMemory(db_path=db_path)
        stats = self.memory.get_stats()
        print(f"  Memory DB: {db_path}")
        print(f"  Stored memories: {stats['total_memories']}")

    def start_server(self):
        """No-op in v8.4.0 (OpenRouter is cloud-based, no local server to start)."""
        self.server = OpenRouterClient(self.model_paths)

    def stop_server(self):
        # No-op: cloud API has no local process to stop
        self.server = None

    def switch_model(self, new_model_name: str):
        """Switch to a different model (stops current server, starts new one)."""
        if self.server:
            self.server.stop()
        self.model_name = new_model_name
        self.model_paths = self.models.get_model_paths(new_model_name)
        if not self.model_paths:
            raise ValueError(f"Model '{new_model_name}' not in config")
        self.start_server()

    def store_memory(self, text, metadata=None):
        """Store interaction in local SimpleMemory DB."""
        if self.memory is None:
            print("  [WARN] Memory not initialized — call setup_memory() first")
            return
        try:
            meta = {
                "model": self.model_name,
                "timestamp": datetime.now().isoformat(),
            }
            if metadata:
                meta.update(metadata)
            content = f"[model={self.model_name}] {text}"
            self.memory.store(text=content, metadata=meta)
        except Exception as e:
            print(f"  [WARN] Memory save failed: {e}")

    def ask_model(self, question, image_path=None):
        """Ask the model a question and return response."""
        if not self.server:
            return None
        if image_path:
            response = self.server.chat_with_image(question, image_path)
        else:
            response = self.server.chat([{"role": "user", "content": question}])
        self.store_memory(f"Q: {question} | A: {response[:200]}")
        return response


# Make torch import lazy
def __getattr__(name):
    if name == "torch":
        import torch
        return torch
    raise AttributeError(f"module 'vision_test' has no attribute '{name}'")


def main():
    print("=" * 70)
    print("MATHIR VISION TESTING INTERFACE")
    print("=" * 70)

    # Load config
    try:
        config = load_config()
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print(f"\nCreate {CONFIG_PATH} first.")
        return

    memory_db = resolve_path(config["memory_db"])

    print(f"\nConfig: {CONFIG_PATH}")
    print(f"Memory DB: {memory_db}")

    # Check OpenRouter config
    or_cfg = config.get("openrouter", {})
    api_key = or_cfg.get("api_key") or os.environ.get("OPENROUTER_API_KEY", "")
    api_base = or_cfg.get("api_base", "https://openrouter.ai/api/v1")
    print(f"\n[OR] OpenRouter config:")
    print(f"  API base: {api_base}")
    print(f"  API key:  {'SET (' + api_key[:8] + '...)' if api_key else 'MISSING (set openrouter.api_key in config.json or OPENROUTER_API_KEY env)'}")

    # Check SimpleMemory DB
    print(f"\n[MEMORY] Checking local DB...")
    if memory_db.exists():
        mem = SimpleMemory(db_path=str(memory_db))
        stats = mem.get_stats()
        print(f"  DB: OK | {stats['total_memories']} memories stored")
        last = mem.get_last(n=3)
        if last:
            print(f"  Recent memories:")
            for r in last:
                print(f"    - [{r.get('created_at', '?')}] {r.get('text', '')[:80]}")
    else:
        print(f"  DB: NEW (will be created at {memory_db})")

    # Show available models
    print(f"\n[1] Available Models (from config.json):")
    models = ModelManager(config)
    for name, m in models.list_models(enabled_only=False).items():
        status = "ON " if m.get("enabled", True) else "OFF"
        supports = []
        if m.get("supports_vision"): supports.append("vision")
        if m.get("supports_audio"): supports.append("audio")
        if m.get("supports_grounding"): supports.append("grounding")
        print(f"  [{status}] {name} (id: {m.get('id', '?')})")
        print(f"          type: {m.get('type')}, supports: {supports}")
        print(f"          display: {m.get('display_name', name)}")

    # OpenRouter connectivity quick check (cheap HEAD ping)
    print(f"\n[2] OpenRouter connectivity:")
    print(f"  Models above will be called via cloud — no local binaries needed in v8.4.0.")

    # Quick start
    print(f"\n[3] Usage (no hardcoded paths, all from config.json):")
    print(f"""
import sys
sys.path.insert(0, 'mathir_repo_root')
from vision_testing.vision_test import VisionTester, ModelManager, load_config

# Load config (relative to this file)
config = load_config()
models = ModelManager(config)

# List available
for name in models.list_models().keys():
    print(f"Available: {{name}}")

# Test with a model
tester = VisionTester("LFM2.5-VL-1.6B-Q4_0")
tester.setup_memory()
tester.start_server()
tester.room.add_object("lamp", "desk")
response = tester.ask_model("What's in the room?")

# Switch model at runtime (stops current, starts new)
tester.switch_model("LFM2.5-Audio-1.5B-Q4_0")

# Add a new model (edit config.json programmatically)
models.add_model("MyModel", {{
    "type": "vision-language",
    "path": "models/MyModel/model.gguf",
    "mmproj": "models/MyModel/mmproj.gguf",
    "enabled": True,
    "supports_vision": True,
}})
""")
    print("=" * 70)


if __name__ == "__main__":
    main()
