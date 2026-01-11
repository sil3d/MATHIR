# MATHIR - Guide de Démarrage Rapide

## 🚀 Configuration Initiale (Une Seule Fois)

```bash
# Lancer le setup
setup_cuda_env.bat
```

Cela créera l'environnement `mathir_cuda` avec PyTorch + CUDA.

---

## 💻 Utilisation Quotidienne

### 1. Activer l'Environnement

**À faire dans CHAQUE nouveau terminal :**

```bash
conda activate mathir_cuda
```

Vous devriez voir `(mathir_cuda)` apparaître dans votre prompt.

---

### 2. Lancer les Scripts

Une fois l'environnement activé, utilisez les scripts simplifiés :

#### 🏋️ Entraînement
```bash
train.bat
```
Lance `train_evolution.py` - Entraînement évolutif MATHIR vs LSTM avec checkpoints live.

#### 📊 Dashboard Scientifique
```bash
dashboard.bat
```
Lance le dashboard Streamlit avec :
- Graphiques de performance
- Analyse IA (Ollama)
- **Brain Scan** (visualisation des poids neuronaux)
- Export PDF

💡 **Astuce** : Activez le **"Mode Live"** dans la sidebar pour voir l'entraînement en temps réel !

#### 🧪 Benchmark
```bash
benchmark.bat
```
Lance `benchmark.py` - Tests de rétention mémoire pure (1000 steps).

#### ⚙️ Optimiseur
```bash
optimize.bat
```
Lance `optimize_mathir.py` - Optimisation des hyperparamètres.

---

## 🎯 Workflow Complet (Formation + Monitoring)

### Terminal 1 : Entraînement
```bash
conda activate mathir_cuda
train.bat
```

### Terminal 2 : Dashboard Live
```bash
conda activate mathir_cuda
dashboard.bat
```

Puis dans le dashboard :
1. Activez **🔄 Mode Live Training**
2. Allez dans l'onglet **🧠 Brain Scan**
3. Regardez les poids évoluer en temps réel !

---

## 📁 Fichiers Importants

- `training_log.json` : Historique d'entraînement
- `checkpoints/mathir_live.pth` : Checkpoint rafraîchi toutes les 500 steps
- `checkpoints/mathir_step_*.pth` : Sauvegardes périodiques (tous les 5000 steps)

---

## ⚠️ Troubleshooting

### "Conda not found"
Assurez-vous que conda est dans votre PATH ou utilisez **Anaconda Prompt**.

### "Module not found"
L'environnement n'est pas activé. Vérifiez que vous voyez `(mathir_cuda)` dans le prompt.

### Scripts batch ne fonctionnent pas
Lancez directement les commandes Python :
```bash
python train_evolution.py
python benchmark.py
streamlit run final_report_streamlit.py
```

---

## 🎓 Documentation Complète

- `MATHIR.md` : README principal du projet
- `MATHIR_VS_LSTM.md` : Comparaison empirique finale
- `MATHIR_JOURNAL_DE_BORD.md` : Journal scientifique pédagogique
- `MATHIR_Preuves_Mathematiques.tex` : Preuves formelles (LaTeX)

---

## 🔬 Architecture

MATHIR v3.3 utilise :
- **DeepSeek-mHC** (Manifold-Constrained Hyper-Connections)
- **Sinkhorn Warm-Start** (initialisation géométrique)
- **Mémoire Épisodique/Sémantique** (10k+ slots)

Voir `mathir_lib/` pour les détails d'implémentation.
