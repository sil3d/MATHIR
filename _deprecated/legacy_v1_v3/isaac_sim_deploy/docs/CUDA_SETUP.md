# 🔧 Quick Guide: Activate PyTorch CUDA

## 🚨 Current Problem

You have **PyTorch CPU-only** installed:
```
PyTorch version: 2.9.1+cpu  ← No CUDA!
CUDA available: False
```

Even if you have CUDA installed on your system, Python is using an environment without GPU support.

---

## ✅ Solution: Conda Environment with PyTorch CUDA

### Option 1: Automatic Script (Recommended)

**Double-click on:**
```
setup_cuda_env.bat
```

This script will:
1. ✅ Create a `mathir_cuda` conda environment
2. ✅ Install PyTorch with CUDA 12.1
3. ✅ Install all dependencies
4. ✅ Verify that the GPU is detected

**Time**: ~5-10 minutes (PyTorch download)

---

### Option 2: Manual Installation

#### Step 1: Create the environment

```bash
conda create -n mathir_cuda python=3.10 -y
conda activate mathir_cuda
```

#### Step 2: Install PyTorch with CUDA

**For CUDA 12.1** (newest):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**For CUDA 11.8** (if 12.1 doesn't work):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Step 3: Install MATHIR dependencies

```bash
pip install streamlit plotly pandas numpy
```

#### Step 4: Verify

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

Should display: `CUDA: True`

---

## 🎯 VSCode Configuration

Once the environment is created:

### Method 1: Manual Selection

1. **Ctrl + Shift + P**
2. Type: `Python: Select Interpreter`
3. Select: `mathir_cuda` (Python 3.10)

### Method 2: Click Bottom Right

1. Click on the **Python version** at the bottom right
2. Select `mathir_cuda` from the list

### Verification

At the bottom right of VSCode, you should see:
```
Python 3.10.x ('mathir_cuda')
```

---

## 🚀 Launch MATHIR with CUDA

### VSCode Terminal

```bash
# Activate the environment (if not already done)
conda activate mathir_cuda

# Launch Streamlit
streamlit run app_streamlit.py

# OR launch the benchmark
python benchmark.py
```

### Integrated VSCode

If you selected the `mathir_cuda` environment:
- Python scripts will automatically use this environment
- No need to manually activate

---

## 🔍 CUDA Verification

### Quick Test

```bash
conda activate mathir_cuda
python test_quick.py
```

**Expected Output:**
```
Device: cuda  ← Should say "cuda" now!
✓ MATHIR: 1,310,071 parameters
✓ LSTM: 1,728,722 parameters
```

### Full Test

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'); print('VRAM:', torch.cuda.get_device_properties(0).total_memory / 1024**3, 'GB' if torch.cuda.is_available() else '')"
```

**Expected Output:**
```
PyTorch: 2.x.x+cu121  ← Note the "+cu121"
CUDA: True
GPU: NVIDIA GeForce RTX XXXX
VRAM: X.X GB
```

---

## 📊 In Streamlit

After configuration, you will see:

**Before (current):**
```
⚠️ No CUDA GPU detected
💻 Running on CPU (slower)
Device: cpu
```

**After (with CUDA):**
```
✅ GPU detected: NVIDIA GeForce RTX XXXX
💾 VRAM: X.X GB
Device: cuda (or cpu)
```

---

## ❓ Which CUDA Version?

### Check Your CUDA Version

```bash
nvidia-smi
```

Look at the line:
```
CUDA Version: 12.1  ← Your version
```

### Compatibility Table

| System CUDA   | PyTorch to Install    |
| ------------- | --------------------- |
| 12.x          | `cu121` (recommended) |
| 11.8          | `cu118`               |
| 11.7 or lower | `cu117`               |

### If You Don't Know

Use **CUDA 12.1** (newest):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

PyTorch is backward compatible.

---

## 🐛 Common Issues

### "conda: command not found"

**Cause:** Conda not installed or not in PATH

**Solution:**
1. Install Miniconda: https://docs.conda.io/en/latest/miniconda.html
2. Restart terminal

### "CUDA out of memory"

**Cause:** Insufficient VRAM

**Solution:**
- Reduce `batch_size` in `benchmark.py`
- Close other GPU applications

### VSCode not using the right environment

**Solution:**
1. Close all VSCode terminals
2. Ctrl+Shift+P → "Python: Select Interpreter"
3. Select `mathir_cuda`
4. Open a new terminal

### PyTorch still CPU-only

**Check:**
```bash
# Which Python is used?
where python

# Should point to the conda environment
# C:\Users\...\anaconda3\envs\mathir_cuda\python.exe
```

**If not:**
```bash
conda activate mathir_cuda
```

---

## 📝 Configuration Checklist

- [ ] Conda installed
- [ ] `mathir_cuda` environment created
- [ ] PyTorch CUDA installed (verify with `import torch; torch.cuda.is_available()`)
- [ ] VSCode points to `mathir_cuda`
- [ ] Quick test passed (`python test_quick.py` → "cuda")
- [ ] Streamlit shows the GPU

---

## 🎉 Summary: 3 Steps

```bash
# 1. Run configuration script
setup_cuda_env.bat

# 2. In VSCode: Select interpreter "mathir_cuda"
#    (Ctrl+Shift+P → Python: Select Interpreter)

# 3. Launch MATHIR
conda activate mathir_cuda
streamlit run app_streamlit.py
```

**That's it!** Your GPU will now be used. 🚀

---

## 💡 Why is this Important?

| Aspect                   | CPU           | GPU (CUDA) |
| ------------------------ | ------------- | ---------- |
| **Benchmark 1000 steps** | ~10 min       | ~2 min     |
| **Training**             | Several hours | Minutes    |
| **Ollama LLaMA 3.1:8b**  | ❌ Too slow    | ✅ Fast     |
| **Real-time Inference**  | ❌ 50+ms       | ✅ <10ms    |

---

<div align="center">

**Configure CUDA now for optimal performance!** ⚡

</div>
