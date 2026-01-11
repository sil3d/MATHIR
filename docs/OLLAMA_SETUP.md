# 🤖 Guide d'Installation et Configuration Ollama

## Qu'est-ce qu'Ollama ?

Ollama est un outil qui permet d'exécuter des modèles LLM (Large Language Models) localement sur votre machine. Il est utilisé dans MATHIR benchmark pour générer des analyses intelligentes des résultats.

## 📥 Installation

### Windows

1. **Téléchargez Ollama**
   - Visitez: https://ollama.ai/download
   - Téléchargez l'installeur Windows
   - Exécutez l'installeur

2. **Vérifiez l'installation**
   ```cmd
   ollama --version
   ```

### Linux

```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

### macOS

```bash
brew install ollama
```

## 🧠 Choix du Modèle LLM

MATHIR supporte deux modèles LLaMA :

### Option 1: LLaMA 3.1:8b (Recommandé si 8GB+ VRAM)

**Configuration requise:**
- GPU: RTX 3060/4060 ou équivalent
- VRAM: 8GB minimum
- Meilleur pour: Analyses détaillées et précises

**Installation:**
```bash
ollama pull llama3.1:8b
```

**Taille du téléchargement:** ~4.7 GB

### Option 2: LLaMA 3.2:3b (Modèle léger)

**Configuration requise:**
- GPU: Optionnel (fonctionne sur CPU)
- VRAM: 3GB minimum (ou CPU uniquement)
- Meilleur pour: Machines avec ressources limitées

**Installation:**
```bash
ollama pull llama3.2:3b
```

**Taille du téléchargement:** ~2.0 GB

## ⚙️ Configuration Automatique

MATHIR détecte automatiquement votre VRAM et sélectionne le modèle approprié :

```python
# Automatique dans le code
if VRAM >= 8GB:
    → Utilise llama3.1:8b
else:
    → Utilise llama3.2:3b
```

## 🧪 Test de Fonctionnement

### 1. Vérifier Ollama

```bash
ollama list
```

Devrait afficher vos modèles installés.

### 2. Tester un modèle

```bash
# Test LLaMA 3.1:8b
ollama run llama3.1:8b "Hello!"

# Ou test LLaMA 3.2:3b
ollama run llama3.2:3b "Hello!"
```

### 3. Test avec MATHIR

```bash
python ollama_analyzer.py
```

Devrait afficher:
```
=== Test Module Ollama ===

====================...====================
  GUIDE D'INSTALLATION OLLAMA
====================...====================

✅ Ollama est installé!

Modèles disponibles: llama3.1:8b

🎮 VRAM disponible: X.X GB
✅ Vous pouvez utiliser llama3.1:8b (modèle complet)
```

## 🚀 Utilisation dans MATHIR

### Lancement du Benchmark avec Ollama

```bash
# Méthode 1: Script Python
python benchmark.py

# Méthode 2: Interface Streamlit
streamlit run app_streamlit.py
```

### Dans le Code

```python
from ollama_analyzer import OllamaAnalyzer

# Initialisation automatique
analyzer = OllamaAnalyzer()  # Sélection auto du modèle

# Ou forcer un modèle spécifique
analyzer = OllamaAnalyzer(model_name="llama3.1:8b")
analyzer = OllamaAnalyzer(model_name="llama3.2:3b")

# Analyser des résultats
analyses = analyzer.analyze_all(results, previous_results)
```

## 📊 Ce Que Fait Ollama dans MATHIR

### 1. Analyse de Rétention
Examine les scores de rétention MATHIR vs LSTM et génère un résumé :
- Tendances principales
- Amélioration quantitative
- Signification pratique

### 2. Analyse de Généralisation
Évalue les performances par scénario :
- Scénarios où MATHIR excelle
- Comparaison avec LSTM
- Insights sur la capacité de généralisation

### 3. Analyse de Performance
Évalue le trade-off performance/capacités :
- Temps d'inférence acceptable ?
- Compatibilité VRAM
- Recommandations

### 4. Résumé Global
Génère un verdict exécutif :
- MATHIR est-il supérieur ?
- Points forts et limitations
- Recommandation pour production
- **Détecte si les résultats sont nouveaux ou réutilisés** ✨

## 🔍 Détection Nouveaux vs Anciens Résultats

Ollama compare automatiquement les benchmarks actuels avec les précédents :

```
📂 Résultats précédents détectés
   → Les analyses Ollama compareront avec les résultats passés

STATUT : RÉSULTATS RÉUTILISÉS
Changement vs précédent : MATHIR 0.82% → 0.85%
```

## ⚠️ Problèmes Courants

### "ollama: command not found"

**Solution:**
- Redémarrez votre terminal après installation
- Vérifiez que Ollama est dans le PATH
- Windows: Relancer CMD/PowerShell

### "Error: model 'llama3.1:8b' not found"

**Solution:**
```bash
ollama pull llama3.1:8b
```

### Ollama lent / timeout

**Causes:**
- Modèle trop gros pour votre config
- Première utilisation (chargement)

**Solution:**
- Utilisez llama3.2:3b sur machines modestes
- Augmentez le timeout dans `ollama_analyzer.py` (ligne 58)

### Analyses non disponibles

Si l'onglet "Analyse Ollama" n'apparaît pas :

1. Vérifiez installation Ollama : `ollama --version`
2. Vérifiez modèle téléchargé : `ollama list`
3. Testez manuellement : `python ollama_analyzer.py`
4. Relancez le benchmark : `python benchmark.py`

## 💡 Conseils d'Optimisation

### Pour GPU Limité (<8GB)

```bash
# Utilisez le modèle léger
ollama pull llama3.2:3b
```

### Pour Analyses Plus Rapides

- Gardez Ollama en arrière-plan
- Première analyse = lent (chargement modèle)
- Analyses suivantes = rapides

### Pour Meilleure Qualité

```bash
# Utilisez le modèle complet si possible
ollama pull llama3.1:8b
```

## 📈 Comparaison des Modèles

| Caractéristique | LLaMA 3.1:8b | LLaMA 3.2:3b |
|-----------------|--------------|--------------|
| **VRAM requise** | 8GB | 3GB |
| **Taille téléchargement** | ~4.7 GB | ~2.0 GB |
| **Qualité analyses** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Vitesse** | Moyenne | Rapide |
| **CPU uniquement** | ❌ Lent | ✅ OK |
| **Recommandé pour** | RTX 3060+ | Config modeste |

## 🎯 Résumé Rapide

```bash
# 1. Installer Ollama
# Téléchargez depuis https://ollama.ai/download

# 2. Choisir et télécharger modèle
ollama pull llama3.1:8b  # Si 8GB+ VRAM
# OU
ollama pull llama3.2:3b  # Si moins de VRAM

# 3. Tester
ollama run llama3.1:8b "Test"

# 4. Lancer MATHIR
python benchmark.py
# OU
streamlit run app_streamlit.py

# 5. Profiter des analyses IA! 🎉
```

## 📚 Ressources

- **Site officiel Ollama**: https://ollama.ai
- **Documentation**: https://github.com/ollama/ollama
- **Modèles disponibles**: https://ollama.ai/library
- **LLaMA 3.1**: https://ollama.ai/library/llama3.1
- **LLaMA 3.2**: https://ollama.ai/library/llama3.2

---

## ✅ Checklist de Validation

Avant de lancer MATHIR avec Ollama :

- [ ] Ollama installé (`ollama --version`)
- [ ] Modèle téléchargé (`ollama list`)
- [ ] Test réussi (`ollama run <model> "Test"`)
- [ ] GPU détecté si disponible
- [ ] Benchmark lancé avec Ollama actif

**Prêt !** Les analyses IA amélioreront automatiquement vos benchmarks. 🚀

---

<div align="center">

**MATHIR + Ollama = Analyses Intelligentes Automatiques** 🧠

*Transformez vos benchmarks en insights actionnables*

</div>
