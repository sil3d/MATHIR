# MATHIR Stress Test

Outil de validation de l'architecture MATHIR v8.4.1+. Teste le vrai système sous charge avec des données synthétiques réalistes.

## Lancement

```bash
pip install flask flask-socketio psutil sentence-transformers
python stress_test/server.py
```

→ Ouvre http://localhost:5000

## Ce que le stress test valide

**Ce N'est PAS un benchmark synthétique.** Le stress test utilise la vraie architecture MATHIR :

### Composants MATHIR testés

| Composant | Methode | Description |
|-----------|---------|-------------|
| **5-Tier Memory** | `MATHIRMemory.perceive()` | Working (buffer circulaire) + Episodic (cosine) + Semantic (k-means) + Procedural (rules) + Immunological (anomaly) |
| **KL Router** | `_KLRouter.forward()` | Blend 4-way avec perte KL pour éviter le collapse |
| **Input Projection** | `input_proj()` | 384d → 384d internal |
| **Anomaly Detection** | `immune.anomaly_score()` | Score L2 distance to nearest normal |
| **Vector Recall** | `MATHIRMemory.recall()` | Cosine similarity sur vrai embeddings |
| **Universal Recall** | `MATHIRMemory.universal_recall()` | Hybrid: text + embedding + cross-lingual |

### Ce qui est mesuré

- **Router Weights** : part de chaque tier dans le blend final [Working, Episodic, Semantic, Immune]
- **Anomaly Score** : distance L2 au nearest normal (immune tier)
- **Recall Latency** : latence cosine similarity (ms)
- **Hybrid Latency** : latence universal_recall (ms)
- **CPU / GPU** : metrics système standards

## Modes de test

| Mode | Backend |
|------|---------|
| `direct` | Appelle `MATHIRMemory` directement (default) |
| `daemon` | JSON-RPC sur port 7338 (MATHIR daemon) |
| `mcp` | Outils MCP (via daemon) |

```bash
python stress_test/server.py
# puis dans l'UI: Configuration → Mode = direct/daemon/mcp
```

## Architecture

```
stress_test/
├── server.py           ← Flask + WebSocket, perceive() + recall()
├── metrics.py          ← Router weights, anomaly score, latences
├── generator.py        ← Données synthétiques depuis data/*.json (250k phrases)
├── data/               ← {technical,personal,structured,trivial,anomaly}.json
├── static/
│   ├── stress.html     ← Dashboard (router weights, anomaly gauge, health bar)
│   └── changelog.html
└── stress_memory.db    ← SQLite (créé au démarrage)
```

## Données synthétiques

250k phrases réparties sur 5 catégories (les mêmes que dans `mathir_dropin` utilise) :
- **Technical** (30%) — questions mémoire, routing, APIs MATHIR
- **Personal** (25%) — conversations réalistes
- **Structured** (20%) — données formatées
- **Trivial** (25%) — "ok", "merci", "compris"
- **Anomalies** (10%) — gibberish, injection SQL, XSS, binaire

Les anomalies servent à tester le **immune tier** de MATHIR.

## API Endpoints

| Méthode | Route | Description |
|---------|-------|-------------|
| POST | `/api/start` | Démarre le test |
| POST | `/api/stop` | Arrête |
| POST | `/api/pause` | Pause / reprend |
| POST | `/api/config` | Met à jour config |
| GET | `/api/metrics` | Snapshot (router_weights_avg, anomaly_score_avg, latences) |
| GET | `/api/status` | Status + GPU |
| GET | `/api/download/csv` | Export CSV |
| GET | `/api/download/html` | Export HTML |

## Dépendances

```
flask
flask-socketio
psutil
sentence-transformers  # embeddings réels (paraphrase-multilingual-MiniLM-L12-v2)
torch                  # GPU + MATHIR
```

## Différences avec les anciens stress tests

| Ancien | Nouveau |
|--------|---------|
| `torch.randn` fake embeddings | SentenceTransformer real embeddings |
| `store()` direct sans router | `perceive()` avec KL 4-way routing |
| Tier counters (counts) | Router weights (blend distribution) |
| Pas d'anomaly score | `immune.anomaly_score()` mesuré |
| FTS5 text search | `recall()` + `universal_recall()` vector |

## Plateformes

- Windows / Linux / macOS
- GPU: auto-detect CUDA (RTX 4060, etc.)
- CPU fallback si pas de GPU