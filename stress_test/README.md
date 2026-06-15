# MATHIR Stress Test

Plateforme de test mémoire en temps réel pour MATHIR. Teste les 4 tiers mémoire sous charge continue.

## Lancement

```bash
# Depuis la racine du projet
pip install flask flask-socketio psutil
python stress_test/server.py
```

→ Ouvre http://localhost:5000

## Fonctionnalités

### Monitoring live
- **4 graphiques temps réel** : RAM, DB Size, Recall Latency, Conversations
- **6 métriques clés** : RAM, GPU, DB, Conversations, Recall, Errors
- **System Health Bar** — 3-color bar (green/blue/red) scoring CPU, GPU, Recall, Errors, DB Write. Thresholds server-driven via WebSocket `health_config` event.
- **Logs en direct** avec niveaux (info/warn/error)

### Contrôle
- **Start / Pause / Stop** — boutons en header
- **Configuration en temps réel** — batch size, intervalle, taux d'anomalies
- **Tiers sélectifs** — activer/désactiver Working, Episodic, Semantic, Immune, KL Router
- **Clean slate** — `start()` deletes old DB + WAL/SHM before reinit
- **Start after Stop** — Recreates ThreadPoolExecutor automatically

### Métriques corrigées
- **CPU** — Utilise `cpu_times()` delta (process-wide, thread-independent) au lieu de `psutil.cpu_percent(interval=None)` qui retournait toujours 0%
- **GPU** — Utilise `torch.cuda.memory_allocated()` (MATHIR-only tensors) au lieu de `nvidia-smi memory.used` (global, tous processus)

### Export
- **CSV** — toutes les métriques horodatées
- **HTML** — rapport visuel avec graphiques statiques

### Architecture
- **Changelog** — Documentation complète de l'architecture à `/changelog` avec diffs code avant/après, benchmarks, références fichiers

## Architecture

```
stress_test/
├── server.py           ← Flask + WebSocket (port 5000), executor recreation, clean slate
├── metrics.py          ← CPU via cpu_times() delta, GPU via torch.cuda.memory_allocated()
├── generator.py        ← Générateur synthétique (126 phrases)
├── static/
│   ├── stress.html     ← UI dark theme (Chart.js + Socket.IO), health bar, server-driven thresholds
│   └── changelog.html  ← Architecture documentation page
├── reports/            ← CSV/HTML exportés
└── stress_memory.db    ← SQLite (créé automatiquement, supprimé au restart)
```

## API Endpoints

| Méthode | Route | Description |
|---------|-------|-------------|
| GET | `/` | UI principale |
| POST | `/api/start` | Démarre le stress test |
| POST | `/api/stop` | Arrête le stress test |
| POST | `/api/pause` | Pause / reprend |
| POST | `/api/config` | Met à jour la config |
| GET | `/api/metrics` | Snapshot actuel (REST) |
| GET | `/api/status` | État du test |
| GET | `/api/download/csv` | Télécharge le CSV |
| GET | `/api/download/html` | Télécharge le rapport HTML |
| WS | `health_config` | Envoie les seuils de health au client |

## Données synthétiques

Le générateur produit 5 types de messages :
- **Technical** (30%) — questions sur MATHIR, mémoire, routing
- **Personal** (25%) — conversations réalistes avec noms
- **Structured** (20%) — données formatées (commandes, tickets, logs)
- **Trivial** (25%) — "ok", "merci", "compris"
- **Anomalies** (10%) — gibberish, injection SQL, XSS, binaire

## Dépendances

```
flask
flask-socketio
psutil
```

Optionnel : `torch` (pour GPU VRAM monitoring)

## Plateformes supportées

- Windows (psutil natif)
- Linux (psutil natif, nvidia-smi pour GPU)
- macOS (psutil natif)
