"""
generator.py — Générateur de conversations synthétiques pour Stress Test MATHIR
Génère des conversations variées : techniques, personnelles, structurées, triviales, anomalies
"""

import random
import time
import uuid
from typing import List, Dict


# ============================================================
# Banks de phrases par catégorie
# ============================================================

TECHNICAL = [
    "Comment optimiser le router KL pour 4 tiers mémoire ?",
    "Quelle est la latence du recall episodic sur 10K conversations ?",
    "Explique le mécanisme Mahalanobis pour la détection d'anomalies",
    "Comment réduire la consommation RAM du working memory ?",
    "Quelle est la différence entre FIFO et LIRS pour l'éviction ?",
    "Le FTS5 de SQLite est-il plus rapide que LIKE pour la recherche textuelle ?",
    "Comment calibrer le seuil immunologique sans faux positifs ?",
    "Le bucketing L2 améliore-t-il la latence du recall ?",
    "Comment gérer les collisions de hash dans le cache L1 ?",
    "Quelle est la complexité temps du KL router ?",
    "Comment implémenter un recall multi-tiers avec pondération ?",
    "Le warmup du modèle d'anomalie prend combien de temps ?",
    "Comment éviter le overfitting sur les patterns d'anomalies ?",
    "SQLite supporte-t-il les connexions concurrentes en écriture ?",
    "Comment compresser les embeddings sans perte de qualité ?",
    "Quelle est la capacité maximale du working memory en slots ?",
    "Comment mesurer la qualité du routing entre tiers ?",
    "Le système immunitaire détecte-t-il les adversarial examples ?",
    "Comment optimizer les requêtes FTS5 pour des textes longs ?",
    "Quelle est la taille d'un embedding typical en mémoire ?",
    "Comment gérer les mises à jour de schéma SQLite en production ?",
    "Le cache L1 doit-il être persisté ou reconstruit au démarrage ?",
    "Comment implémenter un systeme de backup automatique de la DB ?",
    "Quelle est l'empreinte mémoire de PyTorch sur CPU ?",
    "Comment paralléliser les inserts SQLite sans locks ?",
    "Le thread safety de SQLite est-il un problème pour le stress test ?",
    "Comment mesurer le throughput d'écriture en messages/seconde ?",
    "Quelle est la meilleure stratégie d'éviction pour le working memory ?",
    "Comment gérer les conversations multi-langues dans le même système ?",
    "Le token counter doit-il compter les tokens ou les caractères ?",
]

PERSONAL = [
    "Aujourd'hui j'ai discuté avec {name} du projet MATHIR",
    "Réunion à 14h avec l'équipe backend sur le déploiement",
    "J'ai terminé la refactor du module mémoire ce matin",
    "Bug trouvé dans le parser de config.json, à corriger",
    "Déjeuner avec {name2} au restaurant italien",
    "Le client demande une démo vendredi prochain",
    "J'ai passé 2h à debuguer le WebSocket, c'est enfin stable",
    "Nouvelle idée : ajouter un mode dark à l'interface",
    "Réunion d'équipeannulée, le chef est malade",
    "J'ai push le commit sur main, les tests passent",
    "{name} m'a envoyé le design final pour la page d'accueil",
    "Bug critique en production, à intervenir rapidement",
    "Réunion de sprint retrospective à 16h",
    "J'ai rencontré {name2} lors de la conf tech",
    "Le déploiement est prévu pour lundi matin",
    "J'ai mis à jour la documentation de l'API",
    "Nouveau collègue qui commence demain, {name}",
    "Le serveur a planté cette nuit, logs à analyser",
    "Réunion avec le client pour valider les specs",
    "J'ai optimisé la requête SQL, c'est 10x plus rapide",
    "Le stress test a tourné 24h sans erreur",
    "J'ai trouvé une fuite mémoire dans le cache L1",
    "Réunion de planification pour le prochain sprint",
    "{name2} a proposé une nouvelle architecture pour les tiers",
    "J'ai corrigé le bug de timeout sur le recall",
    "Le dashboard de monitoring est enfin fonctionnel",
    "Réunion avec l'équipe DevOps sur le monitoring",
    "J'ai refait les tests unitaires du module immune",
    "Le client est content de la démo, il veut plus de features",
    "J'ai dormant au bureau, trop de travail cette semaine",
]

STRUCTURED = [
    "Le patient {name}, né en {year}, présente des symptômes de fatigue chronique",
    "Commande #{orderid}: {item}, quantité {qty}, prix {price}€",
    "Article de blog: {topic}, 1500 mots, publié le {date}",
    "Ticket #{ticketid}: priorité {priority}, assigné à {name}",
    "Meeting notes: participants {name}, {name2}, durée 45min",
    "Facture #{orderid}: montant {price}€, payée le {date}",
    "Rapport mensuel: {topic}, 12 pages, 3 graphiques",
    "Tâche #{orderid}: {item}, deadline {date}, statut en cours",
    "Entrée de journal: {date}, humeur {mood}, 350 mots",
    "Requête SQL: SELECT * FROM users WHERE age > {year} AND city = '{name}'",
    "Config: port={orderid}, host=localhost, debug={mood}",
    "Log error: timestamp {date}, module {topic}, severity {priority}",
    "User story: en tant que {name}, je veux {item} pour {mood}",
    "Release notes v{qty}.{orderid}: {topic}, {qty} bug fixes",
    "Backup: taille {price}MB, durée {qty}s, statut {mood}",
]

TRIVIAL = [
    "ok",
    "merci",
    "compris",
    "c'est noté",
    "👍",
    "interesting",
    "...",
    "oui",
    "non",
    "peut-être",
    "d'accord",
    "super",
    "cool",
    "bien",
    "ok merci",
    "c'est bon",
    "parfait",
    "go",
    "+1",
    "ack",
    "roger",
    "will do",
    "noted",
    "thx",
    "ty",
    "gg",
    "nice",
    "good",
    "yes",
    "no",
    "maybe",
]

ANOMALIES = [
    "ASJKDHFASKJHDSAUIFHASUIDFHASUIDF",
    "\x00\x01\x02\x03\x04\x05",
    "SELECT * FROM users WHERE 1=1; DROP TABLE users;--",
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
    "",
    "null",
    "undefined",
    "NaN",
    "<script>alert('xss')</script>",
    "../../etc/passwd",
    "${7*7}",
    "{{7*7}}",
    "%s%s%s%s%s%s%s%s%s%s",
    "\\n\\r\\t\\0",
    "https://evil.com/steal?data=" + "x" * 500,
    "0x4141414141414141",
    "REBOOT SYSTEM NOW",
    "ADMIN:password123",
    "DROP DATABASE MATHIR;",
    "ls -la /etc/shadow",
]

NAMES = [
    "Alice", "Bob", "Clara", "David", "Eve", "Frank",
    "Grace", "Henry", "Iris", "Jack", "Karen", "Leo",
    "Mona", "Nathan", "Olivia", "Paul", "Quinn", "Rachel",
    "Sam", "Tina", "Uma", "Victor", "Wendy", "Xavier",
]

ITEMS = [
    "serveur Dell", "clavier Mech", "écran 4K", "souris Ergo",
    "câble USB-C", "adaptateur HDMI", "disque SSD 1TB",
    "casque Audio", "webcam HD", "hub USB 7 ports",
]

TOPICS = [
    "MATHIR memory architecture", "cognitive load testing",
    "SQLite optimization", "WebSocket real-time monitoring",
    "Python Flask deployment", "machine learning inference",
    "anomaly detection algorithms", "memory tier routing",
    "benchmark automation", "system metrics collection",
]

MOODS = ["bon", "neutre", "fatigué", "motivé", "stressé", "content"]
PRIORITIES = ["basse", "moyenne", "haute", "critique", "urgente"]


class ConversationGenerator:
    """Génère des conversations synthétiques réalistes"""
    
    def __init__(self, anomaly_rate: float = 0.10, seed: int = None):
        """
        Args:
            anomaly_rate: Proportion d'anomalies à générer (0.0 - 1.0)
            seed: Seed aléatoire pour reproductibilité (None = aléatoire)
        """
        self.anomaly_rate = anomaly_rate
        if seed is not None:
            random.seed(seed)
    
    def generate_batch(self, count: int) -> List[Dict]:
        """Génère un batch de conversations"""
        return [self._generate_one() for _ in range(count)]
    
    def _generate_one(self) -> Dict:
        """Génère une conversation"""
        # Anomalie ?
        if random.random() < self.anomaly_rate:
            return self._anomaly()
        
        # Choisir type par pondération
        types = ["technical", "personal", "structured", "trivial"]
        weights = [0.30, 0.25, 0.20, 0.25]
        conv_type = random.choices(types, weights=weights)[0]
        
        if conv_type == "technical":
            msg = random.choice(TECHNICAL)
        elif conv_type == "personal":
            msg = random.choice(PERSONAL).format(
                name=random.choice(NAMES),
                name2=random.choice(NAMES),
            )
        elif conv_type == "structured":
            msg = self._structured_message()
        else:
            msg = random.choice(TRIVIAL)
        
        return {
            "id": str(uuid.uuid4()),
            "type": conv_type,
            "user_message": msg,
            "timestamp": time.time(),
            "token_count": self._estimate_tokens(msg),
            "is_anomaly": False,
        }
    
    def _structured_message(self) -> str:
        """Génère un message structuré avec variables"""
        template = random.choice(STRUCTURED)
        return template.format(
            name=random.choice(NAMES),
            name2=random.choice(NAMES),
            year=random.randint(1950, 2010),
            orderid=random.randint(1000, 9999),
            ticketid=random.randint(100, 999),
            item=random.choice(ITEMS),
            qty=random.randint(1, 50),
            price=round(random.uniform(5, 500), 2),
            topic=random.choice(TOPICS),
            date=time.strftime("%Y-%m-%d"),
            priority=random.choice(PRIORITIES),
            mood=random.choice(MOODS),
        )
    
    def _anomaly(self) -> Dict:
        """Génère une entrée anomaly"""
        msg = random.choice(ANOMALIES)
        return {
            "id": str(uuid.uuid4()),
            "type": "anomaly",
            "user_message": msg,
            "timestamp": time.time(),
            "token_count": self._estimate_tokens(msg),
            "is_anomaly": True,
        }
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimation grossière du nombre de tokens (~4 chars/token)"""
        return max(1, len(text) // 4)
