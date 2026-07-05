# MATHIR MCP — Universal Installation

**6-tier cognitive memory for 50 AI coding agents. Install once, use everywhere.**

> **v8.9.0** — **God Mode** (multi-agent orchestration), 27 MCP tools, 3-layer auto-cache, INT8 quantization, cross-encoder reranking. See [CHANGELOG.md](CHANGELOG.md).

---

## ⚡ Quick Start

```bash
# 1. Install (one time)
git clone https://github.com/sil3d/MATHIR.git
cd MATHIR/mathir_mcp
pip install -e .

# 2. Start the daemon
mathir-server &   # listens on 127.0.0.1:7338

# 3. Add to your agent's MCP config — done. 27 tools available.
```

For cold-boot auto-start: `python mathir_mcp/INSTALL_FOR_DEV/install_smart.py --autostart-only`

Platform-specific guides: [INSTALL_FOR_AGENT/INSTALL_WINDOWS.md](INSTALL_FOR_AGENT/INSTALL_WINDOWS.md) · [INSTALL_FOR_AGENT/INSTALL_LINUX.md](INSTALL_FOR_AGENT/INSTALL_LINUX.md) · [INSTALL_FOR_AGENT/INSTALL_MACOS.md](INSTALL_FOR_AGENT/INSTALL_MACOS.md)

---

## 🔌 MCP Tools (27)

| Category | Tools |
|---|---|
| **Auto-injection** | `memory_session_start`, `memory_context` |
| **Basic** | `memory_save`, `memory_recall`, `memory_smart_search`, `memory_hybrid_search`, `memory_delete`, `memory_stats`, `memory_audit`, `memory_export`, `memory_sessions`, `memory_dashboard` |
| **Lifecycle** | `memory_promote`, `memory_auto_promote`, `memory_decay`, `memory_consolidate`, `memory_link`, `memory_get_links`, `memory_build_links` |
| **Advanced (v8.5.1)** | `memory_by_path`, `memory_recall_quality`, `memory_incoming_links` |
| **Guardrail (v8.9.0)** | `memory_list_guardrails` |
| **Immunological** | `memory_audit_immunological` |
| **Health** | `mathir_health` |
| **God Mode (v8.8.0)** | `mathir_mathir_god_agent`, `mathir_mathir_god_orchestre` |

Full signatures: see [`mathir_lib/mathir_mcp_server.py`](mathir_lib/mathir_mcp_server.py).

### Key Algorithms (v8.7.0)

| Algorithm | Purpose |
|---|---|
| **3-layer auto-cache** | L1 embedding LRU + L2 recall TTL + L3 session pre-warm |
| **INT8 scalar quantization** | 4x embedding compression, zero recall loss |
| **Cross-encoder reranking** | `ms-marco-MiniLM-L-6-v2` second-pass scoring (+20pp) |
| **Hybrid search** | Vector cosine + BM25 + RRF fusion |
| **Ebbinghaus decay** | Time-based forgetting (5%/30d floor) |
| **Mahalanobis anomaly** | Immunological tier (threshold=25.0) |
| **Spreading activation** | Collins & Loftus link-graph traversal |

Full list (22 algorithms): see [benchmarks/06_results/current/README.md](../benchmarks/06_results/current/README.md).

### Auto-Cache Performance (v8.7.0)

3-layer transparent cache — zero config, shared across all agents (Claude Code, MiMo, OpenCode).

| Scenario | Latency | Speedup |
|---|---|---|
| Cold query (L1 miss + L2 miss) | ~37ms | baseline |
| Warm embedding (L1 hit + L2 miss) | ~7ms | **5x** |
| Full cache hit (L1 hit + L2 hit) | ~2ms | **18x** |

| Layer | What it caches | Size | Expiry | Invalidation |
|---|---|---|---|---|
| **L1 Embedding** | `encode()` vectors | 1024 LRU | Never (deterministic) | LRU eviction |
| **L2 Recall** | Search results | 256 entries | 60s TTL | On write (save/delete/promote/consolidate) |
| **L3 Session** | Hot memories/project | top-20 | 5 min TTL | On write (per-project) |

Monitor: `GET /api/cache/stats` returns hits, misses, hit ratio per layer.

**Design references**: L1 = pure-function memoization; L2 = HTTP cache-control with write-invalidation; L3 = working-set model (Denning, 1968 — "The Working Set Model for Program Behavior", *Communications of the ACM*).

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
