# 🔧 Guide Rapide: Activer PyTorch CUDA

## 🚨 Problème Actuel

Vous avez **PyTorch CPU-only** installé:
```
PyTorch version: 2.9.1+cpu  ← Pas de CUDA!
CUDA available: False
```

Même si vous avez CUDA installé sur votre système, Python utilise un environnement sans support GPU.

---

## ✅ Solution: Environnement Conda avec PyTorch CUDA

### Option 1: Script Automatique (Recommandé)

**Double-cliquez sur:**
```
setup_cuda_env.bat
```

Ce script va:
1. ✅ Créer un environnement conda `mathir_cuda`
2. ✅ Installer PyTorch avec CUDA 12.1
3. ✅ Installer toutes les dépendances
4. ✅ Vérifier que le GPU est détecté

**Temps**: ~5-10 minutes (téléchargement PyTorch)

---

### Option 2: Installation Manuelle

#### Étape 1: Créer l'environnement

```bash
conda create -n mathir_cuda python=3.10 -y
conda activate mathir_cuda
```

#### Étape 2: Installer PyTorch avec CUDA

**Pour CUDA 12.1** (le plus récent):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Pour CUDA 11.8** (si 12.1 ne marche pas):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Étape 3: Installer les dépendances MATHIR

```bash
pip install streamlit plotly pandas numpy
```

#### Étape 4: Vérifier

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

Devrait afficher: `CUDA: True`

---

## 🎯 Configuration VSCode

Une fois l'environnement créé:

### Méthode 1: Sélection Manuelle

1. **Ctrl + Shift + P**
2. Tapez: `Python: Select Interpreter`
3. Sélectionnez: `mathir_cuda` (Python 3.10)

### Méthode 2: Click en Bas à Droite

1. Cliquez sur la **version Python** en bas à droite
2. Sélectionnez `mathir_cuda` dans la liste

### Vérification

En bas à droite de VSCode, vous devriez voir:
```
Python 3.10.x ('mathir_cuda')
```

---

## 🚀 Lancer MATHIR avec CUDA

### Terminal VSCode

```bash
# Activez l'environnement (si pas déjà fait)
conda activate mathir_cuda

# Lancez Streamlit
streamlit run app_streamlit.py

# OU lancez le benchmark
python benchmark.py
```

### VSCode Intégré

Si vous avez bien sélectionné l'environnement `mathir_cuda`:
- Les scripts Python utiliseront automatiquement cet environnement
- Pas besoin d'activer manuellement

---

## 🔍 Vérification CUDA

### Test Rapide

```bash
conda activate mathir_cuda
python test_quick.py
```

**Sortie attendue:**
```
Device: cuda  ← Devrait dire "cuda" maintenant!
✓ MATHIR: 1,310,071 paramètres
✓ LSTM: 1,728,722 paramètres
```

### Test Complet

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'); print('VRAM:', torch.cuda.get_device_properties(0).total_memory / 1024**3, 'GB' if torch.cuda.is_available() else '')"
```

**Sortie attendue:**
```
PyTorch: 2.x.x+cu121  ← Notez le "+cu121"
CUDA: True
GPU: NVIDIA GeForce RTX XXXX
VRAM: X.X GB
```

---

## 📊 Dans Streamlit

Après configuration, vous verrez:

**Avant (actuel):**
```
⚠️ Aucun GPU CUDA détecté
💻 Exécution sur CPU (plus lent)
Device: cpu
```

**Après (avec CUDA):**
```
✅ GPU détecté: NVIDIA GeForce RTX XXXX
💾 VRAM: X.X GB
Device: cuda (ou cpu)
```

---

## ❓ Quelle Version CUDA ?

### Vérifier Votre Version CUDA

```bash
nvidia-smi
```

Regardez la ligne:
```
CUDA Version: 12.1  ← Votre version
```

### Tableau de Compatibilité

| CUDA Système | PyTorch à Installer |
|--------------|---------------------|
| 12.x | `cu121` (recommandé) |
| 11.8 | `cu118` |
| 11.7 ou moins | `cu117` |

### Si Vous Ne Savez Pas

Utilisez **CUDA 12.1** (la plus récente):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

PyTorch est rétro-compatible.

---

## 🐛 Problèmes Courants

### "conda: command not found"

**Cause:** Conda pas installé ou pas dans PATH

**Solution:**
1. Installez Miniconda: https://docs.conda.io/en/latest/miniconda.html
2. Redémarrez le terminal

### "CUDA out of memory"

**Cause:** VRAM insuffisante

**Solution:**
- Réduisez `batch_size` dans `benchmark.py`
- Fermez autres applications GPU

### VSCode n'utilise pas le bon environnement

**Solution:**
1. Fermez tous les terminaux VSCode
2. Ctrl+Shift+P → "Python: Select Interpreter"
3. Sélectionnez `mathir_cuda`
4. Ouvrez un nouveau terminal

### PyTorch toujours CPU-only

**Vérifiez:**
```bash
# Quel Python est utilisé?
where python

# Devrait pointer vers l'environnement conda
# C:\Users\...\anaconda3\envs\mathir_cuda\python.exe
```

**Si ce n'est pas le cas:**
```bash
conda activate mathir_cuda
```

---

## 📝 Checklist de Configuration

- [ ] Conda installé
- [ ] Environnement `mathir_cuda` créé
- [ ] PyTorch CUDA installé (vérifier avec `import torch; torch.cuda.is_available()`)
- [ ] VSCode pointe vers `mathir_cuda`
- [ ] Test rapide réussi (`python test_quick.py` → "cuda")
- [ ] Streamlit affiche le GPU

---

## 🎉 Résumé: 3 Étapes

```bash
# 1. Lancer le script de configuration
setup_cuda_env.bat

# 2. Dans VSCode: Sélectionner interpréteur "mathir_cuda"
#    (Ctrl+Shift+P → Python: Select Interpreter)

# 3. Lancer MATHIR
conda activate mathir_cuda
streamlit run app_streamlit.py
```

**C'est tout!** Votre GPU sera maintenant utilisé. 🚀

---

## 💡 Pourquoi c'est Important?

| Aspect | CPU | GPU (CUDA) |
|--------|-----|------------|
| **Benchmark 1000 steps** | ~10 min | ~2 min |
| **Entraînement** | Plusieurs heures | Minutes |
| **Ollama LLaMA 3.1:8b** | ❌ Trop lent | ✅ Rapide |
| **Inférence temps réel** | ❌ 50+ms | ✅ <10ms |

---

<div align="center">

**Configurez CUDA maintenant pour des performances optimales!** ⚡

</div>
