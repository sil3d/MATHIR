# 🤖 Ollama Installation and Configuration Guide

## What is Ollama?

Ollama is a tool that allows you to run LLMs (Large Language Models) locally on your machine. It is used in the MATHIR benchmark to generate intelligent analyses of the results.

## 📥 Installation

### Windows

1. **Download Ollama**
   - Visit: https://ollama.ai/download
   - Download the Windows installer
   - Run the installer

2. **Verify Installation**
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

## 🧠 LLM Model Choice

MATHIR supports two LLaMA models:

### Option 1: LLaMA 3.1:8b (Recommended if 8GB+ VRAM)

**Requirements:**
- GPU: RTX 3060/4060 or equivalent
- VRAM: 8GB minimum
- Best for: Detailed and precise analyses

**Installation:**
```bash
ollama pull llama3.1:8b
```

**Download Size:** ~4.7 GB

### Option 2: LLaMA 3.2:3b (Lightweight Model)

**Requirements:**
- GPU: Optional (works on CPU)
- VRAM: 3GB minimum (or CPU only)
- Best for: Machines with limited resources

**Installation:**
```bash
ollama pull llama3.2:3b
```

**Download Size:** ~2.0 GB

## ⚙️ Automatic Configuration

MATHIR automatically detects your VRAM and selects the appropriate model:

```python
# Automatic in code
if VRAM >= 8GB:
    → Uses llama3.1:8b
else:
    → Uses llama3.2:3b
```

## 🧪 Functionality Test

### 1. Verify Ollama

```bash
ollama list
```

Should display your installed models.

### 2. Test a model

```bash
# Test LLaMA 3.1:8b
ollama run llama3.1:8b "Hello!"

# Or test LLaMA 3.2:3b
ollama run llama3.2:3b "Hello!"
```

### 3. Test with MATHIR

```bash
python ollama_analyzer.py
```

Should display:
```
=== Ollama Module Test ===

====================...====================
  OLLAMA INSTALLATION GUIDE
====================...====================

✅ Ollama is installed!

Available models: llama3.1:8b

🎮 Available VRAM: X.X GB
✅ You can use llama3.1:8b (full model)
```

## 🚀 Usage in MATHIR

### Launching Benchmark with Ollama

```bash
# Method 1: Python Script
python benchmark.py

# Method 2: Streamlit Interface
streamlit run app_streamlit.py
```

### In Code

```python
from ollama_analyzer import OllamaAnalyzer

# Automatic initialization
analyzer = OllamaAnalyzer()  # Auto model selection

# Or force a specific model
analyzer = OllamaAnalyzer(model_name="llama3.1:8b")
analyzer = OllamaAnalyzer(model_name="llama3.2:3b")

# Analyze results
analyses = analyzer.analyze_all(results, previous_results)
```

## 📊 What Ollama Does in MATHIR

### 1. Retention Analysis
Examines MATHIR vs LSTM retention scores and generates a summary:
- Main trends
- Quantitative improvement
- Practical significance

### 2. Generalization Analysis
Evaluates performance by scenario:
- Scenarios where MATHIR excels
- Comparison with LSTM
- Insights on generalization capacity

### 3. Performance Analysis
Evaluates performance/capability trade-off:
- Inference time acceptable?
- VRAM compatibility
- Recommendations

### 4. Global Summary
Generates an executive verdict:
- Is MATHIR superior?
- Strengths and limitations
- Recommendation for production
- **Detects if results are new or reused** ✨

## 🔍 Detection New vs Old Results

Ollama automatically compares current benchmarks with previous ones:

```
📂 Previous results detected
   → Ollama analyses will compare with past results

STATUS: REUSED RESULTS
Change vs previous: MATHIR 0.82% → 0.85%
```

## ⚠️ Common Issues

### "ollama: command not found"

**Solution:**
- Restart your terminal after installation
- Check that Ollama is in your PATH
- Windows: Relaunch CMD/PowerShell

### "Error: model 'llama3.1:8b' not found"

**Solution:**
```bash
ollama pull llama3.1:8b
```

### Ollama slow / timeout

**Causes:**
- Model too big for your config
- First use (loading)

**Solution:**
- Use llama3.2:3b on modest machines
- Increase timeout in `ollama_analyzer.py` (line 58)

### Analyses not available

If the "Ollama Analysis" tab does not appear:

1. Check Ollama installation: `ollama --version`
2. Check downloaded model: `ollama list`
3. Test manually: `python ollama_analyzer.py`
4. Rerun benchmark: `python benchmark.py`

## 💡 Optimization Tips

### For Limited GPU (<8GB)

```bash
# Use lightweight model
ollama pull llama3.2:3b
```

### For Faster Analyses

- Keep Ollama in background
- First analysis = slow (model loading)
- Subsequent analyses = fast

### For Better Quality

```bash
# Use full model if possible
ollama pull llama3.1:8b
```

## 📈 Model Comparison

| Feature              | LLaMA 3.1:8b | LLaMA 3.2:3b  |
| -------------------- | ------------ | ------------- |
| **Req VRAM**         | 8GB          | 3GB           |
| **Download Size**    | ~4.7 GB      | ~2.0 GB       |
| **Analysis Quality** | ⭐⭐⭐⭐⭐        | ⭐⭐⭐⭐          |
| **Speed**            | Average      | Fast          |
| **CPU Only**         | ❌ Slow       | ✅ OK          |
| **Recommended For**  | RTX 3060+    | Modest Config |

## 🎯 Quick Summary

```bash
# 1. Install Ollama
# Download from https://ollama.ai/download

# 2. Choose and download model
ollama pull llama3.1:8b  # If 8GB+ VRAM
# OR
ollama pull llama3.2:3b  # If less VRAM

# 3. Test
ollama run llama3.1:8b "Test"

# 4. Launch MATHIR
python benchmark.py
# OR
streamlit run app_streamlit.py

# 5. Enjoy AI analyses! 🎉
```

## 📚 Resources

- **Official Ollama Site**: https://ollama.ai
- **Documentation**: https://github.com/ollama/ollama
- **Available Models**: https://ollama.ai/library
- **LLaMA 3.1**: https://ollama.ai/library/llama3.1
- **LLaMA 3.2**: https://ollama.ai/library/llama3.2

---

## ✅ Validation Checklist

Before launching MATHIR with Ollama:

- [ ] Ollama installed (`ollama --version`)
- [ ] Model downloaded (`ollama list`)
- [ ] Test successful (`ollama run <model> "Test"`)
- [ ] GPU detected if available
- [ ] Benchmark launched with Ollama active

**Ready!** AI analyses will automatically improve your benchmarks. 🚀

---

<div align="center">

**MATHIR + Ollama = Automatic Intelligent Analyses** 🧠

*Transform your benchmarks into actionable insights*

</div>
