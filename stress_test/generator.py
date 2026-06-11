"""
generator.py — Générateur de conversations synthétiques pour Stress Test MATHIR
Charge les données depuis des fichiers JSON (10K entrées par catégorie)
"""

import json
import os
import random
import time
import uuid
from typing import List, Dict

# Chemin vers les données JSON
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


class ConversationGenerator:
    """Génère des conversations synthétiques réalistes à partir de JSON"""

    def __init__(self, anomaly_rate: float = 0.10, seed: int = None):
        """
        Args:
            anomaly_rate: Proportion d'anomalies (0.0 - 1.0)
            seed: Seed aléatoire pour reproductibilité
        """
        self.anomaly_rate = anomaly_rate
        if seed is not None:
            random.seed(seed)

        # Charger les données JSON
        self._data = {}
        self._load_data()

    def _load_data(self):
        """Charge tous les fichiers JSON depuis data/"""
        categories = ["technical", "personal", "structured", "trivial", "anomaly"]

        for cat in categories:
            filepath = os.path.join(DATA_DIR, f"{cat}.json")
            if os.path.exists(filepath):
                with open(filepath, "r", encoding="utf-8") as f:
                    self._data[cat] = json.load(f)
            else:
                # Fallback: liste minimale si le fichier manque
                self._data[cat] = [{"id": 0, "type": cat, "text": f"fallback_{cat}", "category": "fallback"}]

    def generate_batch(self, count: int) -> List[Dict]:
        """Génère un batch de conversations"""
        return [self._generate_one() for _ in range(count)]

    def _generate_one(self) -> Dict:
        """Génère une conversation aléatoire"""

        # Anomalie ?
        if random.random() < self.anomaly_rate:
            entry = random.choice(self._data["anomaly"])
            return {
                "id": str(uuid.uuid4()),
                "type": "anomaly",
                "user_message": entry["text"],
                "category": entry.get("category", "unknown"),
                "timestamp": time.time(),
                "token_count": self._estimate_tokens(entry["text"]),
                "is_anomaly": True,
            }

        # Choisir type par pondération
        types = ["technical", "personal", "structured", "trivial"]
        weights = [0.30, 0.25, 0.20, 0.25]
        conv_type = random.choices(types, weights=weights)[0]

        entry = random.choice(self._data[conv_type])

        return {
            "id": str(uuid.uuid4()),
            "type": conv_type,
            "user_message": entry["text"],
            "category": entry.get("category", "unknown"),
            "timestamp": time.time(),
            "token_count": self._estimate_tokens(entry["text"]),
            "is_anomaly": False,
        }

    def _estimate_tokens(self, text: str) -> int:
        """Estimation ~4 chars/token"""
        return max(1, len(text) // 4)

    def get_stats(self) -> Dict:
        """Retourne les stats des données chargées"""
        return {cat: len(data) for cat, data in self._data.items()}
