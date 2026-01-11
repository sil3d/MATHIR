"""
Module d'analyse intelligente des benchmarks avec Ollama
Utilise LLaMA 3.1:8b (8GB VRAM) ou LLaMA 3.2:3b (modèle léger)
"""

import json
import subprocess
import os
from typing import Dict, Optional, List
import torch


class OllamaAnalyzer:
    """Analyseur intelligent de benchmarks avec Ollama"""
    
    def __init__(self, model_name: str = "auto"):
        """
        Initialize Ollama analyzer
        
        Args:
            model_name: "llama3.1:8b" (8GB VRAM), "llama3.2:3b" (léger), ou "auto"
        """
        self.model_name = self._select_model(model_name)
        self.ollama_available = self._check_ollama()
        
        if self.ollama_available:
            print(f"✓ Ollama détecté avec modèle: {self.model_name}")
        else:
            print("⚠️ Ollama non disponible")
    
    def _select_model(self, model_name: str) -> str:
        """Sélectionne automatiquement le modèle selon VRAM disponible"""
        
        if model_name != "auto":
            return model_name
        
        # Détecte VRAM disponible
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            if vram_gb >= 8:
                print(f"🎮 GPU: {vram_gb:.1f}GB VRAM détectée")
                print("→ Utilisation de llama3.1:8b (modèle complet)")
                return "llama3.1:8b"
            else:
                print(f"🎮 GPU: {vram_gb:.1f}GB VRAM détectée")
                print("→ Utilisation de llama3.2:3b (modèle léger)")
                return "llama3.2:3b"
        else:
            print("💻 Pas de GPU détecté")
            print("→ Utilisation de llama3.2:3b (modèle léger, CPU)")
            return "llama3.2:3b"
    
    def _check_ollama(self) -> bool:
        """Vérifie si Ollama est installé et accessible"""
        try:
            result = subprocess.run(
                ['ollama', 'list'],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def _call_ollama(self, prompt: str) -> Optional[str]:
        """Appelle Ollama pour générer une analyse"""
        
        if not self.ollama_available:
            return None
        
        try:
            # Commande Ollama avec encodage UTF-8 forcé
            cmd = ['ollama', 'run', self.model_name, prompt]
            
            # Fix Windows encoding issue
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',  # Force UTF-8
                errors='ignore',   # Ignore non-UTF8 chars
                timeout=90  # 90 secondes max (augmenté)
            )
            
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                error_msg = result.stderr if result.stderr else "Unknown error"
                print(f"❌ Erreur Ollama: {error_msg}")
                return None
                
        except subprocess.TimeoutExpired:
            print("⏱️ Timeout Ollama (>60s)")
            return None
        except Exception as e:
            print(f"❌ Erreur: {e}")
            return None
    
    def analyze_retention_results(self, results: Dict) -> str:
        """Analyse les résultats de rétention"""
        
        steps = results['retention']['steps']
        mathir_scores = results['retention']['mathir']
        lstm_scores = results['retention']['lstm']
        
        # Construit le prompt
        prompt = f"""Analyse ces résultats de test de rétention de mémoire pour deux modèles d'IA :

LSTM (modèle traditionnel) :
{chr(10).join([f'- {steps[i]} steps: {lstm_scores[i]:.2%}' for i in range(len(steps))])}

MATHIR (nouveau modèle avec triple mémoire) :
{chr(10).join([f'- {steps[i]} steps: {mathir_scores[i]:.2%}' for i in range(len(steps))])}

Fournis une analyse concise en 3-4 phrases maximum :
1. Quelle est la tendance principale ?
2. À quel point MATHIR est-il meilleur ?
3. Qu'est-ce que cela signifie en pratique ?
4. Est-ce une amélioration significative ?

Réponds de manière professionnelle et technique."""
        
        print("\n🧠 Analyse Ollama des résultats de rétention...")
        return self._call_ollama(prompt)
    
    def analyze_generalization_results(self, results: Dict) -> str:
        """Analyse les résultats de généralisation"""
        
        scenarios = results['generalization']['scenarios']
        mathir_scores = results['generalization']['mathir']
        lstm_scores = results['generalization']['lstm']
        
        # Construit le prompt
        data_str = "\n".join([
            f"- {scenarios[i]}: LSTM={lstm_scores[i]:.1%}, MATHIR={mathir_scores[i]:.1%}"
            for i in range(len(scenarios))
        ])
        
        prompt = f"""Analyse ces résultats de généralisation pour deux modèles de conduite autonome testés sur différents scénarios :

{data_str}

Fournis une analyse concise en 3-4 phrases maximum :
1. Sur quels scénarios MATHIR excelle-t-il le plus ?
2. Y a-t-il des scénarios où LSTM est compétitif ?
3. Quelle est l'amélioration moyenne ?
4. Que révèle cette capacité de généralisation ?

Réponds de manière professionnelle et technique."""
        
        print("\n🧠 Analyse Ollama des résultats de généralisation...")
        return self._call_ollama(prompt)
    
    def analyze_performance_results(self, results: Dict) -> str:
        """Analyse les résultats de performance"""
        
        mathir_time = results['performance']['inference_time']['mathir']['mean']
        lstm_time = results['performance']['inference_time']['lstm']['mean']
        
        mathir_mem = results['performance']['memory']['mathir']['memory_gb'][-1]
        lstm_mem = results['performance']['memory']['lstm']['memory_gb'][-1]
        
        prompt = f"""Analyse ces résultats de performance pour deux modèles d'IA :

LSTM :
- Temps d'inférence : {lstm_time:.2f} ms
- VRAM (batch 32) : {lstm_mem:.2f} GB

MATHIR :
- Temps d'inférence : {mathir_time:.2f} ms
- VRAM (batch 32) : {mathir_mem:.2f} GB

Contexte : Limite matérielle RTX 3060/4060 = 8GB VRAM

Fournis une analyse concise en 3-4 phrases maximum :
1. Le coût en performance de MATHIR est-il acceptable ?
2. Les deux modèles sont-ils compatibles avec la limite de 8GB ?
3. Le trade-off performance/capacités est-il justifié ?
4. Recommandation finale ?

Réponds de manière professionnelle et technique."""
        
        print("\n🧠 Analyse Ollama des résultats de performance...")
        return self._call_ollama(prompt)
    
    def generate_global_summary(self, results: Dict, previous_results: Optional[Dict] = None) -> str:
        """Génère un résumé global comparant avec les résultats précédents"""
        
        # Calcul des métriques clés
        mathir_ret_1000 = results['retention']['mathir'][2]  # @ 1000 steps
        lstm_ret_1000 = results['retention']['lstm'][2]
        
        mathir_gen_avg = sum(results['generalization']['mathir']) / len(results['generalization']['mathir'])
        lstm_gen_avg = sum(results['generalization']['lstm']) / len(results['generalization']['lstm'])
        
        # Détecte si nouveaux résultats ou réutilisation
        status = "NOUVEAUX RÉSULTATS" if previous_results is None else "RÉSULTATS RÉUTILISÉS"
        
        improvement_text = ""
        if previous_results:
            prev_mathir_ret = previous_results.get('retention', {}).get('mathir', [0, 0, 0])[2]
            prev_lstm_ret = previous_results.get('retention', {}).get('lstm', [0, 0, 0])[2]
            
            if prev_mathir_ret != mathir_ret_1000:
                improvement_text = f"\nChangement vs précédent : MATHIR {prev_mathir_ret:.2%} → {mathir_ret_1000:.2%}"
        
        prompt = f"""Génère un RAPPORT DE MISSION ULTIME pour le projet MATHIR (Memory Augmented Neural Network) :
        
        CONTEXTE : MATHIR vs LSTM Classic dans le "Cognitive Labyrinth" (Test de torture mémorielle).
        STATUT : {status}
        {improvement_text}
        
        DONNÉES DU COMBAT :
        - 🧠 Rétention Longue Durée (@1000 steps) : LSTM={lstm_ret_1000:.2%} vs MATHIR={mathir_ret_1000:.2%}
        - 🌐 Généralisation (Scénarios Inédits) : LSTM={lstm_gen_avg:.2%} vs MATHIR={mathir_gen_avg:.2%}
        - 💻 Optimisation VRAM : Compatible 8GB (RTX 3060/4060) ✅
        
        Ta mission est de rédiger un verdict FINAL percutant (Style : Expert IA Futuriste & Enthousiaste).
        Structure ta réponse ainsi :
        
        1. 🏆 LE VAINQUEUR INDISCUTABLE
           - Déclare le gagnant avec emphase. Utilise des emojis (🚀, 🧠, 🥇).
           
        2. 💥 ANALYSE DU K.O. TECHNIQUE
           - Explique POURQUOI l'un a écrasé l'autre (Mémoire explicite vs Mémoire récurrente limitée).
           - Cite les chiffres clés de la Rétention.
           
        3. 🔮 VISION FUTURISTE
           - En quoi cette architecture change la donne pour les véhicules autonomes ?
           - Parle de "Neuro-Plasticité" et d'adaptation temps réel.
        
        4. 📝 RECOMMANDATION DE DÉPLOIEMENT
           - Go / No Go pour la production.
           
        Sois concis, direct, et 'WOW'. Pas de blabla inutile."""
        
        print("\n🧠 Génération du résumé global Ollama...")
        return self._call_ollama(prompt)
    
    def analyze_all(self, results: Dict, previous_results: Optional[Dict] = None) -> Dict[str, str]:
        """Génère toutes les analyses"""
        
        if not self.ollama_available:
            return {
                'retention': "⚠️ Ollama non disponible - Installez Ollama pour analyses IA",
                'generalization': "⚠️ Ollama non disponible - Installez Ollama pour analyses IA",
                'performance': "⚠️ Ollama non disponible - Installez Ollama pour analyses IA",
                'global_summary': "⚠️ Ollama non disponible - Installez Ollama pour analyses IA"
            }
        
        analyses = {}
        
        # Analyse rétention
        retention_analysis = self.analyze_retention_results(results)
        analyses['retention'] = retention_analysis or "❌ Échec de l'analyse"
        
        # Analyse généralisation
        gen_analysis = self.analyze_generalization_results(results)
        analyses['generalization'] = gen_analysis or "❌ Échec de l'analyse"
        
        # Analyse performance
        perf_analysis = self.analyze_performance_results(results)
        analyses['performance'] = perf_analysis or "❌ Échec de l'analyse"
        
        # Résumé global
        global_summary = self.generate_global_summary(results, previous_results)
        analyses['global_summary'] = global_summary or "❌ Échec de l'analyse"
        
        return analyses


def check_ollama_status() -> Dict:
    """Vérifie le statut d'Ollama et des modèles"""
    
    status = {
        'installed': False,
        'models_available': [],
        'recommended_model': None,
        'vram_available': 0.0,
        'can_use_8b': False
    }
    
    # Vérifie installation Ollama
    try:
        result = subprocess.run(
            ['ollama', 'list'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            status['installed'] = True
            
            # Parse les modèles disponibles
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            for line in lines:
                if line.strip():
                    model_name = line.split()[0]
                    status['models_available'].append(model_name)
        
    except (subprocess.TimeoutExpired, FileNotFoundError):
        status['installed'] = False
    
    # Détecte VRAM
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        status['vram_available'] = vram_gb
        status['can_use_8b'] = vram_gb >= 8
    
    # Recommande modèle
    if status['can_use_8b']:
        status['recommended_model'] = 'llama3.1:8b'
    else:
        status['recommended_model'] = 'llama3.2:3b'
    
    return status


def print_ollama_setup_guide():
    """Affiche le guide d'installation Ollama"""
    
    status = check_ollama_status()
    
    print("\n" + "="*60)
    print("  GUIDE D'INSTALLATION OLLAMA")
    print("="*60)
    
    if status['installed']:
        print("\n✅ Ollama est installé!")
        print(f"\nModèles disponibles: {', '.join(status['models_available']) or 'Aucun'}")
    else:
        print("\n❌ Ollama n'est pas installé")
        print("\nÉtapes d'installation:")
        print("1. Téléchargez Ollama: https://ollama.ai/download")
        print("2. Installez Ollama")
        print("3. Redémarrez votre terminal")
    
    print(f"\n🎮 VRAM disponible: {status['vram_available']:.1f} GB")
    
    if status['can_use_8b']:
        print("✅ Vous pouvez utiliser llama3.1:8b (modèle complet)")
        print("\nCommande pour télécharger:")
        print("   ollama pull llama3.1:8b")
    else:
        print("⚠️ VRAM insuffisante pour llama3.1:8b (nécessite 8GB)")
        print("✅ Utilisez llama3.2:3b (modèle léger)")
        print("\nCommande pour télécharger:")
        print("   ollama pull llama3.2:3b")
    
    print(f"\n📌 Modèle recommandé: {status['recommended_model']}")
    
    print("\n💡 Pour tester Ollama:")
    print(f"   ollama run {status['recommended_model']} 'Hello!'")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    # Test du module
    print("=== Test Module Ollama ===\n")
    
    # Affiche le guide
    print_ollama_setup_guide()
    
    # Test de l'analyseur
    analyzer = OllamaAnalyzer()
    
    if analyzer.ollama_available:
        print("\n✓ Analyseur Ollama prêt!")
        print(f"  Modèle: {analyzer.model_name}")
    else:
        print("\n⚠️ Ollama non disponible")
        print("  Installez Ollama pour activer les analyses IA")
