# MATHIR MCP — Universal Installation

**5-tier cognitive memory for 50 AI coding agents. Install once, use everywhere.**

> **v8.6.0** — 23 MCP tools, INT8 quantization (4x compression), cross-encoder reranking, 22 algorithms. See [CHANGELOG.md](CHANGELOG.md).

---

## ⚡ Quick Start

```bash
# 1. Install (one time)
git clone https://github.com/sil3d/MATHIR.git
cd MATHIR/mathir_mcp
pip install -e .

# 2. Start the daemon
mathir-server &   # listens on 127.0.0.1:7338

# 3. Add to your agent's MCP config — done. 23 tools available.
```

For cold-boot auto-start: `python mathir_mcp/INSTALL_FOR_DEV/install_smart.py --autostart-only`

Platform-specific guides: [INSTALL_FOR_AGENT/INSTALL_WINDOWS.md](INSTALL_FOR_AGENT/INSTALL_WINDOWS.md) · [INSTALL_FOR_AGENT/INSTALL_LINUX.md](INSTALL_FOR_AGENT/INSTALL_LINUX.md) · [INSTALL_FOR_AGENT/INSTALL_MACOS.md](INSTALL_FOR_AGENT/INSTALL_MACOS.md)

---

## 🔌 MCP Tools (23)

| Category | Tools |
|---|---|
| **Auto-injection** | `memory_session_start`, `memory_context` |
| **Basic** | `memory_save`, `memory_recall`, `memory_smart_search`, `memory_hybrid_search`, `memory_delete`, `memory_stats`, `memory_audit`, `memory_export`, `memory_sessions`, `memory_dashboard` |
| **Lifecycle** | `memory_promote`, `memory_auto_promote`, `memory_decay`, `memory_consolidate`, `memory_link`, `memory_get_links`, `memory_build_links` |
| **Advanced (v8.5.1)** | `memory_by_path`, `memory_recall_quality`, `memory_incoming_links` |
| **Health** | `mathir_health` |

Full signatures: see [`mathir_lib/mathir_mcp_server.py`](mathir_lib/mathir_mcp_server.py).

### Key Algorithms (v8.6.0)

| Algorithm | Purpose |
|---|---|
| **INT8 scalar quantization** | 4x embedding compression, zero recall loss |
| **Cross-encoder reranking** | `ms-marco-MiniLM-L-6-v2` second-pass scoring (+20pp) |
| **Hybrid search** | Vector cosine + BM25 + RRF fusion |
| **Ebbinghaus decay** | Time-based forgetting (5%/30d floor) |
| **Mahalanobis anomaly** | Immunological tier (threshold=25.0) |
| **Spreading activation** | Collins & Loftus link-graph traversal |

Full list (22 algorithms): see [benchmarks/06_results/current/README.md](../benchmarks/06_results/current/README.md).

---

## 📚 Documentation Index

| Doc | Purpose |
|---|---|
| **[INSTALL_FOR_AGENT/AGENT.md](INSTALL_FOR_AGENT/AGENT.md)** | Per-agent config (50+ agents) & troubleshooting |
| **[docs/DAEMON.md](docs/DAEMON.md)** | Daemon HTTP/JSON-RPC protocol + security |
| **[docs/DASHBOARD_GUIDE.md](docs/DASHBOARD_GUIDE.md)** | Stats dashboard setup |
| **[docs/GPU_SETUP.md](docs/GPU_SETUP.md)** | GPU/ONNX acceleration |
| **[docs/DIMENSIONS.md](docs/DIMENSIONS.md)** | Embedding model selection |
| **[CHANGELOG.md](../CHANGELOG.md)** | Version history |

---

## 🆘 If Installer Fails

Give the `~/.config/MATHIR/` folder to your coding agent. It reads `INSTALL_FOR_AGENT/AGENT.md` and configures MATHIR automatically.

---

## 🔒 Security

- DoS protection via per-field length caps (see [docs/DAEMON.md](docs/DAEMON.md))
- Run daemon behind a firewall; don't expose port 7338 publicly
- The **immunological tier** is a research prototype, not a certified security layer

---

## ⚠️ Moving the Folder

The installer writes an **absolute path**. Re-run `install.bat` / `install.sh` after moving.

---

## 🌐 Multilingual

- **FR** : Donnez `~/.config/MATHIR/` à votre agent.
- **ES** : Dea `~/.config/MATHIR/` a su agente.
- **ZH** : 把 `~/.config/MATHIR/` 给你的 agent。
