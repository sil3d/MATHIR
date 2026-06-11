"""
server.py — Serveur Flask + WebSocket du MATHIR Stress Test
Lance avec: python stress_test/server.py
Ouvre: http://localhost:5000
"""

import os
import sys
import json
import time
import threading
from flask import Flask, render_template, request, send_file, jsonify
from flask_socketio import SocketIO, emit

# Add parent dir to path for MATHIR imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from stress_test.metrics import MetricsCollector
from stress_test.generator import ConversationGenerator

# ============================================================
# Flask app
# ============================================================

app = Flask(__name__, static_folder="static", template_folder="static")
app.config["SECRET_KEY"] = "mathir-stress-test"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")


# ============================================================
# Stress Test Engine
# ============================================================

class StressTest:
    """Orchestre le stress test en arrière-plan"""

    def __init__(self):
        self.running = False
        self.paused = False
        self.config = {
            "batch_size": 50,
            "interval_seconds": 5,
            "anomaly_rate": 0.10,
            "tiers_enabled": {
                "working": True,
                "episodic": True,
                "semantic": True,
                "immune": True,
                "kl_router": True,
            },
        }
        self.memory = None
        self.metrics = None
        self.generator = ConversationGenerator()
        self.conversations_sent = 0
        self._thread = None
        self._db_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "stress_memory.db"
        )

    def start(self, config=None):
        """Démarre le stress test"""
        if self.running:
            return {"status": "already_running"}

        if config:
            self.config.update(config)

        # Init MATHIR 4-tier memory
        self.memory = None
        self._embedding_dim = 384
        try:
            import torch
            from mathir_dropin import MATHIRMemory
            self.memory = MATHIRMemory(
                embedding_dim=self._embedding_dim,
                db_path=self._db_path,
            )
            self._log("info", f"MATHIR 4-tier memory initialized (dim={self._embedding_dim})")
        except ImportError:
            self._log("warn", "mathir_dropin not available, using raw sqlite3 fallback")
        except Exception as e:
            self._log("warn", f"MATHIR init failed: {e}, using raw sqlite3 fallback")

        # Init metrics
        self.metrics = MetricsCollector(self._db_path)

        # Init generator
        self.generator = ConversationGenerator(
            anomaly_rate=self.config.get("anomaly_rate", 0.10)
        )

        self.running = True
        self.paused = False
        self.conversations_sent = 0

        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

        self._log("info", f"Stress test started — batch={self.config['batch_size']}, interval={self.config['interval_seconds']}s")
        socketio.emit("status_update", {"status": "running"})
        return {"status": "running"}

    def stop(self):
        """Arrête le stress test"""
        self.running = False
        self.paused = False
        if self._thread:
            self._thread.join(timeout=5)
        self._log("warn", f"Stress test stopped — {self.conversations_sent} conversations injected")
        socketio.emit("status_update", {"status": "stopped"})
        return {"status": "stopped"}

    def pause(self):
        """Pause / reprend"""
        if not self.running:
            return {"status": "stopped"}
        self.paused = not self.paused
        status = "paused" if self.paused else "running"
        self._log("info", f"Stress test {status}")
        socketio.emit("status_update", {"status": status})
        return {"status": status}

    def update_config(self, config):
        """Met à jour la config en temps réel"""
        self.config.update(config)
        # Regenerate with new anomaly rate
        self.generator = ConversationGenerator(
            anomaly_rate=self.config.get("anomaly_rate", 0.10)
        )
        self._log("info", f"Config updated: batch={self.config['batch_size']}, interval={self.config['interval_seconds']}s, anomaly={self.config['anomaly_rate']:.0%}")
        return {"status": "updated"}

    def _run_loop(self):
        """Boucle principale du stress test (thread séparé)"""
        while self.running:
            if self.paused:
                time.sleep(0.5)
                continue

            try:
                # 1. Generate batch
                batch = self.generator.generate_batch(self.config["batch_size"])

                # 2. Store each conversation
                for conv in batch:
                    self._store_conversation(conv)

                # 2b. Recall benchmark (every 50 conversations)
                if self.conversations_sent > 0 and self.conversations_sent % 50 == 0:
                    self._benchmark_recall()

                # 3. Collect metrics
                snapshot = self.metrics.collect()

                # 4. Push to WebSocket
                history_data = []
                for s in self.metrics.history[-120:]:
                    history_data.append({
                        "x": s.timestamp,
                        "y": s.ram_mb,
                        "ram_mb": s.ram_mb,
                        "db_size_mb": s.db_size_mb,
                        "recall_latency_ms": s.recall_latency_ms,
                        "conversations": s.conversations_total,
                    })

                socketio.emit("metrics_update", {
                    "ram_mb": snapshot.ram_mb,
                    "gpu_mb": snapshot.gpu_mb,
                    "db_size_mb": snapshot.db_size_mb,
                    "conversations": self.conversations_sent,
                    "tokens": snapshot.tokens_total,
                    "recall_latency_ms": snapshot.recall_latency_ms,
                    "errors": snapshot.errors,
                    "cpu_percent": snapshot.cpu_percent,
                    "history": history_data,
                })

                # 5. Log every 10 batches
                if self.conversations_sent % (self.config["batch_size"] * 10) == 0:
                    rates = self.metrics.get_growth_rates()
                    self._log("info",
                        f"Convos: {self.conversations_sent:,} | "
                        f"RAM: {snapshot.ram_mb:.0f}MB ({rates['ram_mb_per_hour']:+.0f}MB/h) | "
                        f"DB: {snapshot.db_size_mb:.1f}MB ({rates['db_mb_per_hour']:+.1f}MB/h) | "
                        f"Recall: {snapshot.recall_latency_ms:.0f}ms"
                    )

            except Exception as e:
                self.metrics.increment_errors()
                self._log("error", f"Error in run loop: {e}")

            # Sleep
            time.sleep(self.config["interval_seconds"])

    def _store_conversation(self, conv):
        """Stocke une conversation dans MATHIR avec routing par tier"""
        msg = conv["user_message"]
        tiers = self.config.get("tiers_enabled", {})

        if self.memory is not None:
            try:
                import torch
                # Créer un embedding à partir du texte (hash → vecteur déterministe)
                # Pour un stress test, on utilise un embedding déterministe basé sur le hash du texte
                text_hash = hash(msg) % (2**31)
                torch.manual_seed(text_hash)
                emb = torch.randn(1, self._embedding_dim)

                # Déterminer le tier selon le type de conversation
                if conv.get("is_anomaly"):
                    tier = "immune"
                elif conv["type"] == "trivial":
                    tier = "working"
                elif conv["type"] == "technical":
                    tier = "semantic"
                else:
                    tier = "episodic"

                # Stocker dans le tier actif correspondant
                if tiers.get(tier, False):
                    self.memory.store(
                        embedding=emb,
                        metadata={"text": msg, "type": conv["type"], "is_anomaly": conv.get("is_anomaly", False)},
                        tier=tier,
                    )

            except Exception as e:
                self.metrics.increment_errors()
        else:
            # Fallback: raw sqlite3
            self._raw_store(msg)

        # Update counters
        self.conversations_sent += 1
        self.metrics.increment_conversations(1)
        self.metrics.increment_tokens(conv.get("token_count", 0))

    def _raw_store(self, message):
        """Fallback: stockage brut en SQLite"""
        import sqlite3
        try:
            conn = sqlite3.connect(self._db_path)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    message TEXT,
                    timestamp REAL,
                    token_count INTEGER
                )
            """)
            cursor.execute(
                "INSERT INTO conversations VALUES (?, ?, ?, ?)",
                (str(time.time()), message, time.time(), len(message) // 4)
            )
            conn.commit()
            conn.close()
        except Exception:
            pass

    def _benchmark_recall(self):
        """Benchmark du recall MATHIR — test de latence FTS5"""
        if self.memory is None:
            return

        try:
            # Requêtes de test variées
            queries = ["Python", "bug", "réunion", "MATHIR", "memory"]
            start = time.perf_counter()
            for q in queries:
                self.memory.recall_text(query_text=q, k=3)
            elapsed_ms = (time.perf_counter() - start) * 1000 / len(queries)

            # Mettre à jour la métrique de latence
            if self.metrics.history:
                self.metrics.history[-1].recall_latency_ms = round(elapsed_ms, 2)
        except Exception:
            pass

    def export_csv(self):
        """Exporte les métriques en CSV"""
        filepath = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "reports", "metrics.csv"
        )
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.metrics.to_csv(filepath)
        return filepath

    def export_html(self):
        """Génère un rapport HTML statique"""
        filepath = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "reports", "report.html"
        )
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        rates = self.metrics.get_growth_rates()
        snapshots = self.metrics.history

        # Build chart data
        ram_data = json.dumps([{"x": i, "y": s.ram_mb} for i, s in enumerate(snapshots[-200:])])
        db_data = json.dumps([{"x": i, "y": s.db_size_mb} for i, s in enumerate(snapshots[-200:])])
        recall_data = json.dumps([{"x": i, "y": s.recall_latency_ms} for i, s in enumerate(snapshots[-200:])])

        html = f"""<!DOCTYPE html>
<html><head><title>MATHIR Stress Test Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<style>
body {{ font-family: -apple-system, sans-serif; background: #0a0a0f; color: #e0e0e0; padding: 40px; }}
h1 {{ color: #6366f1; }}
.grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin: 20px 0; }}
.card {{ background: #12121a; border: 1px solid #2a2a3a; border-radius: 8px; padding: 16px; }}
.card h3 {{ font-size: 12px; color: #888; text-transform: uppercase; margin-bottom: 8px; }}
.card .value {{ font-size: 28px; font-weight: 700; }}
.chart {{ background: #12121a; border: 1px solid #2a2a3a; border-radius: 8px; padding: 16px; margin: 16px 0; }}
canvas {{ height: 200px !important; }}
</style></head><body>
<h1>MATHIR Stress Test Report</h1>
<p>Generated: {time.strftime("%Y-%m-%d %H:%M:%S")} | Duration: {self.metrics.get_uptime_seconds():.0f}s | Conversations: {self.conversations_sent:,}</p>
<div class="grid">
  <div class="card"><h3>Peak RAM</h3><div class="value">{max((s.ram_mb for s in snapshots), default=0):.0f} MB</div></div>
  <div class="card"><h3>Final DB Size</h3><div class="value">{snapshots[-1].db_size_mb if snapshots else 0:.1f} MB</div></div>
  <div class="card"><h3>Avg Recall</h3><div class="value">{sum(s.recall_latency_ms for s in snapshots)/len(snapshots) if snapshots else 0:.0f} ms</div></div>
  <div class="card"><h3>RAM Growth</h3><div class="value">{rates['ram_mb_per_hour']:+.1f} MB/h</div></div>
  <div class="card"><h3>DB Growth</h3><div class="value">{rates['db_mb_per_hour']:+.1f} MB/h</div></div>
  <div class="card"><h3>Errors</h3><div class="value">{sum(s.errors for s in snapshots)}</div></div>
</div>
<div class="chart"><h3>RAM Usage</h3><canvas id="c1"></canvas></div>
<div class="chart"><h3>DB Size</h3><canvas id="c2"></canvas></div>
<div class="chart"><h3>Recall Latency</h3><canvas id="c3"></canvas></div>
<script>
const opts = {{responsive:true, animation:false, plugins:{{legend:{{display:false}}}}, scales:{{x:{{display:false}},y:{{beginAtZero:true,grid:{{color:'#1e1e2e'}},ticks:{{color:'#888'}}}}}}}};
new Chart(document.getElementById('c1'),{{type:'line',data:{{labels:{ram_data}.map((_,i)=>i),datasets:[{{data:{ram_data}.map(d=>d.y),borderColor:'#6366f1',fill:true,backgroundColor:'#6366f120'}}]}},options:opts}});
new Chart(document.getElementById('c2'),{{type:'line',data:{{labels:{db_data}.map((_,i)=>i),datasets:[{{data:{db_data}.map(d=>d.y),borderColor:'#22c55e',fill:true,backgroundColor:'#22c55e20'}}]}},options:opts}});
new Chart(document.getElementById('c3'),{{type:'line',data:{{labels:{recall_data}.map((_,i)=>i),datasets:[{{data:{recall_data}.map(d=>d.y),borderColor:'#eab308',fill:true,backgroundColor:'#eab30820'}}]}},options:opts}});
</script></body></html>"""

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html)
        return filepath

    def _log(self, level, message):
        """Envoie un log au navigateur"""
        socketio.emit("log", {"level": level, "message": message})


# ============================================================
# Instance globale
# ============================================================

stress = StressTest()


# ============================================================
# Routes Flask
# ============================================================

@app.route("/")
def index():
    return render_template("stress.html")


@app.route("/api/start", methods=["POST"])
def api_start():
    config = request.json if request.is_json else None
    return jsonify(stress.start(config))


@app.route("/api/stop", methods=["POST"])
def api_stop():
    return jsonify(stress.stop())


@app.route("/api/pause", methods=["POST"])
def api_pause():
    return jsonify(stress.pause())


@app.route("/api/config", methods=["POST"])
def api_config():
    if request.is_json:
        return jsonify(stress.update_config(request.json))
    return jsonify({"error": "no config"}), 400


@app.route("/api/metrics")
def api_metrics():
    """Snapshot actuel (REST fallback)"""
    if stress.metrics:
        snapshot = stress.metrics.collect()
        return jsonify({
            "ram_mb": snapshot.ram_mb,
            "gpu_mb": snapshot.gpu_mb,
            "db_size_mb": snapshot.db_size_mb,
            "conversations": stress.conversations_sent,
            "tokens": snapshot.tokens_total,
            "recall_latency_ms": snapshot.recall_latency_ms,
            "errors": snapshot.errors,
        })
    return jsonify({"error": "not started"}), 400


@app.route("/api/status")
def api_status():
    return jsonify({
        "status": "running" if stress.running else ("paused" if stress.paused else "stopped"),
        "conversations": stress.conversations_sent,
        "uptime": stress.metrics.get_uptime_seconds() if stress.metrics else 0,
    })


@app.route("/api/download/csv")
def api_download_csv():
    if not stress.metrics or not stress.metrics.history:
        return jsonify({"error": "no data"}), 400
    filepath = stress.export_csv()
    return send_file(filepath, as_attachment=True, download_name="mathir_stress_metrics.csv")


@app.route("/api/download/html")
def api_download_html():
    if not stress.metrics or not stress.metrics.history:
        return jsonify({"error": "no data"}), 400
    filepath = stress.export_html()
    return send_file(filepath, as_attachment=True, download_name="mathir_stress_report.html")


# ============================================================
# WebSocket events
# ============================================================

@socketio.on("connect")
def on_connect():
    print("[WS] Client connected")
    if stress.running and stress.metrics:
        emit("status_update", {"status": "running" if not stress.paused else "paused"})


@socketio.on("disconnect")
def on_disconnect():
    print("[WS] Client disconnected")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    print("=" * 50)
    print("  MATHIR Stress Test")
    print("  http://localhost:5000")
    print("=" * 50)
    socketio.run(app, host="0.0.0.0", port=5000, debug=True, allow_unsafe_werkzeug=True)
