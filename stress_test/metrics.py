"""
metrics.py — Collecte de métriques en temps réel pour le Stress Test MATHIR
Fonctionne sur Windows et Linux via psutil
"""

import time
import os
import psutil
import sqlite3
from dataclasses import dataclass, field
from typing import List, Optional, Dict


@dataclass
class MetricsSnapshot:
    """Un instantané des métriques système"""
    timestamp: float
    ram_mb: float
    gpu_mb: float
    db_size_mb: float
    conversations_total: int
    tokens_total: int
    recall_latency_ms: float
    errors: int
    cpu_percent: float = 0.0
    ram_percent: float = 0.0
    peak_ram_mb: float = 0.0
    throughput_conv_per_sec: float = 0.0
    db_write_latency_ms: float = 0.0
    uptime_seconds: float = 0.0
    gpu_util_percent: float = 0.0
    tier_working: int = 0
    tier_episodic: int = 0
    tier_semantic: int = 0
    tier_immune: int = 0


class MetricsCollector:
    """Collecte les métriques en arrière-plan sans bloquer le serveur"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.history: List[MetricsSnapshot] = []
        self.start_time = time.time()
        self.error_count = 0
        self.total_conversations = 0
        self.total_tokens = 0
        self.peak_ram_mb = 0.0
        self.tier_counts: Dict[str, int] = {
            "working": 0, "episodic": 0, "semantic": 0, "immune": 0
        }
        self._last_write_time = 0.0
        self._write_latencies: List[float] = []
        self._gpu_util = 0.0
        # Manual CPU tracking via cpu_times() delta — process-wide, thread-independent,
        # works correctly regardless of which thread calls it (metrics vs storage).
        # psutil.Process().cpu_percent() is unreliable when called from a sleeping thread.
        self._process = psutil.Process()
        self._last_cpu_times = None
        self._last_cpu_mono = None

    def collect(self) -> MetricsSnapshot:
        """Prend un snapshot des métriques actuelles"""
        process = self._process
        mem_info = process.memory_info()
        ram_mb = round(mem_info.rss / 1024 / 1024, 2)

        # Track peak RAM
        if ram_mb > self.peak_ram_mb:
            self.peak_ram_mb = ram_mb

        # Compute throughput
        uptime = time.time() - self.start_time
        throughput = self.total_conversations / uptime if uptime > 0 else 0.0

        # Average DB write latency
        avg_write_ms = 0.0
        if self._write_latencies:
            avg_write_ms = round(sum(self._write_latencies) / len(self._write_latencies), 3)
            # Keep last 100 only
            if len(self._write_latencies) > 100:
                self._write_latencies = self._write_latencies[-100:]

        gpu_mb, gpu_util = self._get_gpu_usage()

        # CPU percent: manual cpu_times() delta.
        # Measures PROCESS-wide CPU time (all threads), not just calling thread.
        # Robust across Flask reloader forks and thread pool workers.
        cpu = self._get_cpu_percent()

        snapshot = MetricsSnapshot(
            timestamp=time.time(),
            ram_mb=ram_mb,
            gpu_mb=gpu_mb,
            db_size_mb=self._get_db_size(),
            conversations_total=self.total_conversations,
            tokens_total=self.total_tokens,
            recall_latency_ms=self._benchmark_recall(),
            errors=self.error_count,
            cpu_percent=cpu,
            ram_percent=process.memory_percent(),
            peak_ram_mb=round(self.peak_ram_mb, 2),
            throughput_conv_per_sec=round(throughput, 2),
            db_write_latency_ms=avg_write_ms,
            uptime_seconds=round(uptime, 1),
            gpu_util_percent=gpu_util,
            tier_working=self.tier_counts["working"],
            tier_episodic=self.tier_counts["episodic"],
            tier_semantic=self.tier_counts["semantic"],
            tier_immune=self.tier_counts["immune"],
        )

        self.history.append(snapshot)
        return snapshot

    def record_write_latency(self, seconds: float):
        """Enregistre la latence d'une opération d'écriture DB"""
        self._write_latencies.append(seconds * 1000)  # convert to ms

    def increment_tier(self, tier: str):
        """Incrémenter le compteur d'un tier"""
        if tier in self.tier_counts:
            self.tier_counts[tier] += 1

    def _get_gpu_usage(self) -> tuple:
        """GPU VRAM (MB) and utilization (%) for THIS process only.

        VRAM: torch.cuda.memory_allocated() = only MATHIR's tensors (not global).
        Util: nvidia-smi = system-wide GPU util (no per-process API exists).

        Returns (vram_mb, util_percent).
        """
        # --- VRAM: MATHIR-specific via torch.cuda ---
        vram_mb = 0.0
        try:
            import torch
            if torch.cuda.is_available():
                vram_mb = round(torch.cuda.memory_allocated() / 1024 / 1024, 2)
        except (ImportError, RuntimeError):
            pass

        # --- GPU Utilization: nvidia-smi (system-wide, only option) ---
        util_pct = self._gpu_util  # keep last known value as default
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=3
            )
            if result.returncode == 0:
                util_pct = float(result.stdout.strip().split("\n")[0])
                self._gpu_util = util_pct
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, IndexError):
            pass

        return (vram_mb, util_pct)

    def _get_cpu_percent(self) -> float:
        """Process CPU % via cpu_times() delta — process-wide, thread-independent.

        Uses process.cpu_times().user + .system (monotonically increasing) divided
        by wall-clock elapsed. Works correctly regardless of which thread calls it,
        and survives Flask reloader forks because psutil.Process() is re-acquired
        at __init__ time (child process gets its own instance).
        """
        try:
            ct = self._process.cpu_times()
            cpu_time = ct.user + ct.system
            now = time.monotonic()

            if self._last_cpu_times is not None and self._last_cpu_mono is not None:
                dt_wall = now - self._last_cpu_mono
                dt_cpu = cpu_time - self._last_cpu_times
                if dt_wall > 0.01:  # avoid division by tiny wall time
                    percent = (dt_cpu / dt_wall) * 100.0
                    return min(round(percent, 1), 100.0)

            # First call or too-fast call: store baseline
            self._last_cpu_times = cpu_time
            self._last_cpu_mono = now
            return 0.0
        except Exception:
            return 0.0

    def _get_db_size(self) -> float:
        """Taille du fichier SQLite en MB"""
        if os.path.exists(self.db_path):
            return round(os.path.getsize(self.db_path) / 1024 / 1024, 2)
        return 0.0

    def _benchmark_recall(self) -> float:
        """Test de latence recall"""
        if not os.path.exists(self.db_path):
            return 0.0

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]

            if not tables:
                conn.close()
                return 0.0

            start = time.perf_counter()
            for _ in range(3):
                for table in tables:
                    cursor.execute(f"SELECT COUNT(*) FROM [{table}]")
                    cursor.fetchone()
            elapsed_ms = (time.perf_counter() - start) * 1000 / 3

            conn.close()
            return round(elapsed_ms, 2)

        except Exception:
            return 0.0

    def increment_conversations(self, count: int = 1):
        self.total_conversations += count

    def increment_tokens(self, count: int):
        self.total_tokens += count

    def increment_errors(self, count: int = 1):
        self.error_count += count

    def get_uptime_seconds(self) -> float:
        return round(time.time() - self.start_time, 1)

    def get_growth_rates(self) -> dict:
        if len(self.history) < 2:
            return {"ram_mb_per_hour": 0, "db_mb_per_hour": 0}

        first = self.history[0]
        last = self.history[-1]
        elapsed_hours = (last.timestamp - first.timestamp) / 3600

        if elapsed_hours < 0.001:
            return {"ram_mb_per_hour": 0, "db_mb_per_hour": 0}

        return {
            "ram_mb_per_hour": round(
                (last.ram_mb - first.ram_mb) / elapsed_hours, 2
            ),
            "db_mb_per_hour": round(
                (last.db_size_mb - first.db_size_mb) / elapsed_hours, 2
            ),
        }

    def to_csv(self, filepath: str):
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("timestamp,datetime,ram_mb,ram_percent,cpu_percent,gpu_mb,gpu_util_percent,"
                    "db_size_mb,conversations,tokens,recall_latency_ms,errors,"
                    "peak_ram_mb,throughput_conv_per_sec,db_write_latency_ms,"
                    "uptime_seconds,tier_working,tier_episodic,tier_semantic,tier_immune,"
                    "ram_mb_per_hour,db_mb_per_hour\n")

            rates = self.get_growth_rates()

            for s in self.history:
                dt = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(s.timestamp))
                f.write(f"{s.timestamp},{dt},{s.ram_mb},{s.ram_percent},"
                        f"{s.cpu_percent},{s.gpu_mb},{s.gpu_util_percent},"
                        f"{s.db_size_mb},"
                        f"{s.conversations_total},{s.tokens_total},"
                        f"{s.recall_latency_ms},{s.errors},"
                        f"{s.peak_ram_mb},{s.throughput_conv_per_sec},"
                        f"{s.db_write_latency_ms},{s.uptime_seconds},"
                        f"{s.tier_working},{s.tier_episodic},{s.tier_semantic},{s.tier_immune},"
                        f"{rates['ram_mb_per_hour']},{rates['db_mb_per_hour']}\n")

        return filepath
