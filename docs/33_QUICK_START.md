# MATHIR - Quick Start Guide

## 🚀 Initial Setup (Once Only)

```bash
# Launch setup
setup_cuda_env.bat
```

This will create the `mathir_cuda` environment with PyTorch + CUDA.

---

## 💻 Daily Usage

### 1. Activate Environment

**To do in EVERY new terminal:**

```bash
conda activate mathir_cuda
```

You should see `(mathir_cuda)` appear in your prompt.

---

### 2. Launch Scripts

Once the environment is activated, use the simplified scripts:

#### 🏋️ Training
```bash
train.bat
```
Runs `train_evolution.py` - Evolutionary training for MATHIR with live checkpoints.

#### 📊 Scientific Dashboard
```bash
dashboard.bat
```
Runs the Streamlit dashboard with:
- Performance charts
- AI Analysis (Ollama)
- **Brain Scan** (visualization of neural weights)
- PDF Export

💡 **Tip**: Enable **"Live Mode"** in the sidebar to watch training in real-time!

#### 🧪 Benchmark
```bash
benchmark.bat
```
Runs `benchmark.py` - Pure memory retention tests (1000 steps).

#### ⚙️ Optimizer
```bash
optimize.bat
```
Runs `optimize_mathir.py` - Hyperparameter optimization.

---

## 🎯 Full Workflow (Training + Monitoring)

### Terminal 1 : Training
```bash
conda activate mathir_cuda
train.bat
```

### Terminal 2 : Live Dashboard
```bash
conda activate mathir_cuda
dashboard.bat
```

Then in the dashboard:
1. Enable **🔄 Live Training Mode**
2. Go to **🧠 Brain Scan** tab
3. Watch weights evolve in real-time!

---

## 📁 Important Files

- `training_log.json` : Training history
- `checkpoints/mathir_live.pth` : Checkpoint refreshed every 500 steps
- `checkpoints/mathir_step_*.pth` : Periodic saves (every 5000 steps)

---

## ⚠️ Troubleshooting

### "Conda not found"
Make sure conda is in your PATH or use **Anaconda Prompt**.

### "Module not found"
The environment is not activated. Verify you see `(mathir_cuda)` in the prompt.

### Batch scripts not working
Run Python commands directly:
```bash
python train_evolution.py
python benchmark.py
streamlit run final_report_streamlit.py
```

---

## 🎓 Complete Documentation

- `MATHIR.md` : Main project README
- `MATHIR_Preuves_Mathematiques.tex` : Formal proofs (LaTeX)

---

## 🔬 Architecture

MATHIR v3.3 uses:
- **DeepSeek-mHC** (Manifold-Constrained Hyper-Connections)
- **Sinkhorn Warm-Start** (Geometric initialization)
- **Episodic/Semantic Memory** (10k+ slots)

See `mathir_lib/` for implementation details.
