#!/usr/bin/env python3
"""
MATHIR Vision Testing Interface
==============================

NO HARDCODED PATHS. All paths come from config.json (relative to this file).
User can add/remove models by editing config.json - no code changes needed.

Tests MATHIR memory by:
- Setting up a virtual room with objects
- Asking models what's in the room
- "Moving" objects
- Asking what was done
- Storing interactions in MATHIR memory
- Testing recall across models
"""
import os
import sys
import json
import time
import shutil
import subprocess
import urllib.request
import urllib.error
import base64
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any

# All paths RELATIVE to this file - works from any clone location
HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "config.json"


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


class LlamaServer:
    """Manages a llama-server.exe instance for a specific model."""

    def __init__(self, model_paths: dict, port: int = 8080,
                 ctx_size: int = 4096, n_gpu_layers: int = 20,
                 log_level: str = "info"):
        self.model_paths = model_paths
        self.model_name = model_paths["name"]
        self.display_name = model_paths.get("display_name", self.model_name)
        self.port = port
        self.ctx_size = ctx_size
        self.n_gpu_layers = n_gpu_layers
        self.log_level = log_level
        self.process = None
        self.server_url = f"http://127.0.0.1:{port}"
        self._config = load_config()
        self.bin_dir = resolve_path(self._config["bin_dir"])
        self.executable_name = self._config["llama_server"].get("executable", "llama-server.exe")
        self.executable = self._find_executable()

    def _find_executable(self) -> Path:
        """Find the right llama-server executable for this OS."""
        cfg = self._config["llama_server"]
        if sys.platform == "win32":
            return self.bin_dir / cfg.get("executable", "llama-server.exe")
        elif sys.platform == "darwin":
            return self.bin_dir / cfg.get("macos_executable", "llama-server")
        else:
            return self.bin_dir / cfg.get("linux_executable", "llama-server")

    def start(self):
        """Start the llama-server process."""
        if self.process and self.process.poll() is None:
            print(f"  Server already running for {self.display_name}")
            return

        if not self.executable.exists():
            raise FileNotFoundError(
                f"llama-server not found at {self.executable}. "
                f"Run setup_binaries.py first or update config.json:bin_dir"
            )

        model_path = self.model_paths.get("path")
        if not model_path or not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}. "
                f"Check config.json:models:{self.model_name}:path"
            )

        cmd = [
            str(self.executable),
            "-m", str(model_path),
            "--port", str(self.port),
            "-c", str(self.ctx_size),
            "-ngl", str(self.n_gpu_layers),
        ]
        if self.model_paths.get("mmproj"):
            cmd.extend(["--mmproj", str(self.model_paths["mmproj"])])
        if self.model_paths.get("vocoder"):
            cmd.extend(["--vocoder", str(self.model_paths["vocoder"])])

        print(f"  Starting llama-server for {self.display_name} on port {self.port}...")
        print(f"  Model: {model_path.name}")
        if self.model_paths.get("mmproj"):
            print(f"  mmproj: {self.model_paths['mmproj'].name}")

        # Set PATH so DLLs are found on Windows
        env = os.environ.copy()
        env["PATH"] = str(self.bin_dir) + os.pathsep + env.get("PATH", "")

        # Use CREATE_NO_WINDOW on Windows
        creationflags = 0
        if hasattr(subprocess, "CREATE_NO_WINDOW") and sys.platform == "win32":
            creationflags = subprocess.CREATE_NO_WINDOW

        self.process = subprocess.Popen(
            cmd,
            cwd=str(self.bin_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            creationflags=creationflags,
        )

        # Wait for server to be ready
        for i in range(60):
            time.sleep(1)
            try:
                req = urllib.request.Request(f"{self.server_url}/health")
                with urllib.request.urlopen(req, timeout=2) as resp:
                    if resp.status == 200:
                        print(f"  Server ready on port {self.port}")
                        return
            except Exception:
                if self.process.poll() is not None:
                    stderr = self.process.stderr.read().decode(errors='replace')[:500]
                    raise RuntimeError(f"Server died: {stderr}")
        raise RuntimeError("Server failed to start in 60s")

    def stop(self):
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
            print(f"  Server {self.display_name} stopped")

    def chat(self, messages, max_tokens=512, temperature=0.7):
        """Send a chat completion request."""
        data = json.dumps({
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }).encode()
        req = urllib.request.Request(
            f"{self.server_url}/v1/chat/completions",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read())
                return result["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors='replace')[:200]
            return f"ERROR: HTTP {e.code} - {body}"

    def chat_with_image(self, text, image_path, max_tokens=512):
        """Chat with an image attachment."""
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode()
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
            ]
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
        self._st_model = None

    def setup_memory(self):
        """Set up MATHIR memory (uses relative path from config). Preserves existing DB."""
        from mathir_dropin import MATHIRMemory
        db_path = resolve_path(self.config["memory_db"])
        # DO NOT delete existing DB — preserve memories across restarts
        self.memory = MATHIRMemory(embedding_dim=384, db_path=str(db_path))
        print(f"  MATHIR memory initialized: {db_path}")

    def start_server(self):
        """Start llama-server for the current model."""
        self.server = LlamaServer(self.model_paths, port=self.port)
        self.server.start()

    def stop_server(self):
        if self.server:
            self.server.stop()

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
        """Store interaction in MATHIR memory using FTS5 (no embeddings needed)."""
        if not self.memory:
            return
        try:
            meta = {
                "text": text,
                "model": self.model_name,
                "timestamp": datetime.now().isoformat(),
            }
            if metadata:
                meta.update(metadata)
            self.memory.store(
                metadata=meta,
                provider="fts5",
                model="text"
            )
        except Exception as e:
            print(f"  [WARN] Memory store failed: {e}")

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

    bin_dir = resolve_path(config["bin_dir"])
    models_dir = resolve_path(config["models_dir"])
    memory_db = resolve_path(config["memory_db"])

    print(f"\nConfig: {CONFIG_PATH}")
    print(f"Bin dir: {bin_dir}")
    print(f"Models dir: {models_dir}")
    print(f"Memory DB: {memory_db}")

    # Show available models
    print(f"\n[1] Available Models (from config.json):")
    models = ModelManager(config)
    for name, m in models.list_models(enabled_only=False).items():
        status = "ON " if m.get("enabled", True) else "OFF"
        paths = models.get_model_paths(name)
        file_exists = paths["path"].exists() if paths and paths.get("path") else False
        ready = "READY" if file_exists else "MISSING"
        supports = []
        if m.get("supports_vision"): supports.append("vision")
        if m.get("supports_audio"): supports.append("audio")
        print(f"  [{ready}] [{status}] {name}")
        print(f"          type: {m.get('type')}, supports: {supports}")
        print(f"          display: {m.get('display_name', name)}")
        if file_exists:
            print(f"          path: {paths['path'].name} ({paths['path'].stat().st_size/1024/1024:.0f} MB)")

    # Check binaries
    print(f"\n[2] Binaries:")
    if bin_dir.exists():
        files = list(bin_dir.glob("*"))
        print(f"  {bin_dir}: {len(files)} files")
    else:
        print(f"  MISSING: {bin_dir}")
        print(f"  Run: python vision_testing/setup_binaries.py")

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