"""
metrics.py — Collecte de métriques en temps réel pour le Stress Test MATHIR
Fonctionne sur Windows et Linux via psutil
"""

import time
import os
import psutil
import sqlite3
from dataclasses import dataclass, field
from typing import List, Optional


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


class MetricsCollector:
    """Collecte les métriques en arrière-plan sans bloquer le serveur"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.history: List[MetricsSnapshot] = []
        self.start_time = time.time()
        self.error_count = 0
        self.total_conversations = 0
        self.total_tokens = 0
    
    def collect(self) -> MetricsSnapshot:
        """Prend un snapshot des métriques actuelles (~10ms, non-bloquant)"""
        process = psutil.Process()
        mem_info = process.memory_info()
        
        snapshot = MetricsSnapshot(
            timestamp=time.time(),
            ram_mb=round(mem_info.rss / 1024 / 1024, 2),
            gpu_mb=self._get_gpu_usage(),
            db_size_mb=self._get_db_size(),
            conversations_total=self.total_conversations,
            tokens_total=self.total_tokens,
            recall_latency_ms=self._benchmark_recall(),
            errors=self.error_count,
            cpu_percent=process.cpu_percent(interval=None),
            ram_percent=process.memory_percent(),
        )
        
        self.history.append(snapshot)
        return snapshot
    
    def _get_gpu_usage(self) -> float:
        """GPU VRAM si CUDA disponible, 0 sinon. Supporte NVIDIA via nvidia-smi sur Linux."""
        # Try PyTorch CUDA first
        try:
            import torch
            if torch.cuda.is_available():
                return round(torch.cuda.memory_allocated() / 1024 / 1024, 2)
        except (ImportError, RuntimeError):
            pass
        
        # Fallback: nvidia-smi (Linux + Windows)
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=3
            )
            if result.returncode == 0:
                return float(result.stdout.strip().split("\n")[0])
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
            pass
        
        return 0.0
    
    def _get_db_size(self) -> float:
        """Taille du fichier SQLite en MB"""
        if os.path.exists(self.db_path):
            return round(os.path.getsize(self.db_path) / 1024 / 1024, 2)
        return 0.0
    
    def _benchmark_recall(self) -> float:
        """Test de latence recall — temps moyen de 3 requêtes simples"""
        if not os.path.exists(self.db_path):
            return 0.0
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get table names
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            if not tables:
                conn.close()
                return 0.0
            
            # Benchmark: 3 SELECT queries
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
        """Incrémenter le compteur de conversations"""
        self.total_conversations += count
    
    def increment_tokens(self, count: int):
        """Incrémenter le compteur de tokens"""
        self.total_tokens += count
    
    def increment_errors(self, count: int = 1):
        """Incrémenter le compteur d'erreurs"""
        self.error_count += count
    
    def get_uptime_seconds(self) -> float:
        """Durée depuis le démarrage"""
        return round(time.time() - self.start_time, 1)
    
    def get_growth_rates(self) -> dict:
        """Calcule les taux de croissance (par heure) depuis le début"""
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
        """Exporte tout l'historique en CSV"""
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        
        with open(filepath, "w", encoding="utf-8") as f:
            # Header
            f.write("timestamp,datetime,ram_mb,ram_percent,cpu_percent,gpu_mb,"
                    "db_size_mb,conversations,tokens,recall_latency_ms,errors,"
                    "ram_mb_per_hour,db_mb_per_hour\n")
            
            rates = self.get_growth_rates()
            
            for s in self.history:
                dt = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(s.timestamp))
                f.write(f"{s.timestamp},{dt},{s.ram_mb},{s.ram_percent},"
                        f"{s.cpu_percent},{s.gpu_mb},{s.db_size_mb},"
                        f"{s.conversations_total},{s.tokens_total},"
                        f"{s.recall_latency_ms},{s.errors},"
                        f"{rates['ram_mb_per_hour']},{rates['db_mb_per_hour']}\n")
        
        return filepath
