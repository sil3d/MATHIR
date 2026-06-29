# MATHIR SEO Tags

This document lists all the search/SEO tags that should be applied to improve MATHIR's discoverability.

---

## GitHub Repository Topics (20 recommended)

These are tags that appear on the GitHub repo page. Set via:
- Web UI: https://github.com/sil3d/MATHIR (click ⚙️ next to "About")
- CLI: `gh repo edit sil3d/MATHIR --add-topic TOPIC` (requires `gh` authenticated)

**Topics to add (in priority order):**

| Topic | Category | Search intent |
|---|---|---|
| `llm-memory` | Primary | "memory for LLM" |
| `memory-augmented` | Primary | "memory augmented" |
| `cognitive-memory` | Primary | "cognitive memory AI" |
| `mcp` | Primary | "model context protocol" |
| `model-context-protocol` | Primary | "model context protocol" |
| `ai-agent` | Primary | "AI agent memory" |
| `rag` | Primary | "retrieval augmented generation" |
| `knowledge-graph` | Primary | "knowledge graph AI" |
| `open-source` | Primary | "open source LLM memory" |
| `mit-license` | Primary | "MIT licensed AI" |
| `sqlite` | Technical | "SQLite AI" |
| `local-ai` | Technical | "local AI memory" |
| `edge-ai` | Technical | "edge AI" |
| `jetson` | Hardware | "Jetson AI" |
| `raspberry-pi` | Hardware | "Raspberry Pi AI" |
| `neuroscience` | Conceptual | "neuroscience AI" |
| `ebbinghaus` | Conceptual | "Ebbinghaus forgetting curve" |
| `prompt-injection-detection` | Security | "prompt injection detection" |
| `anomaly-detection` | Security | "anomaly detection AI" |
| `vector-database` | Alternative | "vector database alternative" |

---

## Repository Description (155 chars max)

```
Cognitive memory layer for LLMs. 5 brain-inspired tiers (working, episodic, semantic, procedural, immunological), Ebbinghaus forgetting, MCP, MIT, SQLite, edge-ready.
```

---

## Homepage URL

```
https://github.com/sil3d/MATHIR/
```

---

## Topics to AVOID

| Topic | Why avoid |
|---|---|
| `machine-learning` | Too generic, lost in noise |
| `python` | Already implied by repo language |
| `ai` | Too broad |
| `chatgpt` | Vendor-specific (we're vendor-neutral) |
| `transformer` | Implies model, not memory |
| `vector-search` | We do more than vector search |

---

## Search Terms We Want to Rank For

When someone searches on Google or GitHub for:

1. **"memory layer for LLM"** → should find MATHIR
2. **"memory augmented generation"** → should find MATHIR
3. **"cognitive memory AI"** → should find MATHIR
4. **"MCP memory tool"** → should find MATHIR
5. **"Ebbinghaus AI memory"** → should find MATHIR
6. **"local AI memory"** → should find MATHIR
7. **"open source MemGPT alternative"** → should find MATHIR
8. **"AI agent long term memory"** → should find MATHIR
9. **"memory consolidation AI"** → should find MATHIR
10. **"Jetson AI memory"** → should find MATHIR

---

## Meta Tags (already added in `index.html`)

The `index.html` at repo root contains:
- Title tag with keywords
- Description (160 chars, SEO-optimized)
- Keywords meta tag (30+ keywords)
- Open Graph tags (Facebook)
- Twitter Card tags
- JSON-LD structured data (SoftwareApplication schema)
- Canonical URL
- Theme color

---

## How to Verify SEO

After deploying, check:
1. **Google Search Console**: submit sitemap at `https://github.com/sil3d/MATHIR/sitemap.xml` (need to create)
2. **GitHub topics**: visible at https://github.com/sil3d/MATHIR#topics
3. **Rich snippet test**: https://search.google.com/test/rich-results
4. **Open Graph preview**: https://www.opengraph.xyz/

---

*Last updated: 2026-06-29 — MATHIR v8.5.1*