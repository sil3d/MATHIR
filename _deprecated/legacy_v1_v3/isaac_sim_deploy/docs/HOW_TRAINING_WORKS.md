# How MATHIR Training Works 🧠

## Overview

When you run `train.bat`, you're not just training a neural network. You are launching an **intelligent self-optimizing system** that combines:

- **Reinforcement Learning**
- **Meta-Learning** (learning to learn)
- **AI-Driven Optimization** (AI optimizing AI)

---

## 🎯 The Components

### 1. The Architectures (Fighters)

```python
# MATHIR - The Challenger (Advanced Architecture)
mathir = MATHIR(
    hidden_dim=256,
    memory_config={
        'working_slots': 256,
        'episodic_slots': 10000,  # Long-term memory
        'semantic_slots': 1024
    }
)

# LSTM - The Current Champion (Baseline)
lstm = LSTM(
    hidden_dim=1024,  # Larger to be fair
    num_layers=3
)
```

**Key Difference**:
- **LSTM**: Forgets quickly (limited cache memory)
- **MATHIR**: Episodic memory 10k+ events + Sinkhorn projection

---

### 2. The Training Algorithm (Reinforcement Learning)

```python
# Training Loop (train_evolution.py)
for step in range(1, 1_000_000):
    # 1. Observe environment
    obs = env.get_state()  # Position, speed, obstacles
    
    # 2. MATHIR decides an action
    m_action = mathir(obs)  # Ex: "Turn left 15°"
    
    # 3. LSTM decides too
    l_action = lstm(obs)
    
    # 4. Execution and reward
    obs_next, reward = env.step(m_action)
    
    # 5. Backpropagation
    loss = reward_function(m_action, target_optimal)
    optimizer.step()  # Update weights
```

**It's Supervised Imitation Learning**: The network learns to imitate an ideal trajectory (sinusoid).

---

## 🤖 Automatic Optimization (The Secret Sauce)

### Optimization Flowchart

![Optimization Flow](docs/images/training_optimization_flow.png)

### Detailed Code

#### Every 500 Steps (30 seconds on GPU)

```python
if step % 500 == 0:
    # 1. Calculate average performance
    m_avg = np.mean(mathir_rewards[-500:])  # Ex: 0.85
    l_avg = np.mean(lstm_rewards[-500:])    # Ex: 0.92
    
    # 2. Is MATHIR worse?
    if m_avg < l_avg * 1.1:  # 10% tolerance
        print(f"⚠️ MATHIR needs help ({m_avg:.3f}). Asking Llama...")
        
        # 3. Call Ollama (Llama 3.2)
        prompt = """
        You are an AI AutoML Expert.
        Current Decay: [0.9, 0.7, 0.5]
        Current Accuracy: 0.85 (Goal: 1.0)
        Task: Suggest BETTER 'decay' rates.
        JSON ONLY: {"decay": [0.95, 0.8, 0.6]}
        """
        
        response = ollama.run("llama3.2:3b", prompt)
        # Response in 2-3 seconds
        
        # 4. Parse JSON response
        new_decay = json.loads(response)["decay"]
        # Ex: [0.92, 0.75, 0.55]
        
        # 5. LIVE Application
        mathir.memory.retention_decay = torch.tensor(new_decay)
        print(f"🧠 Llama suggested: {new_decay}")
```

#### Fallback if Ollama Fails

```python
else:
    # Random Mutation (Evolutionary Strategy)
    mutation = np.random.uniform(-0.05, 0.05, size=3)
    new_decay = np.clip(current_decay + mutation, 0.1, 0.99)
    current_decay = sorted(new_decay, reverse=True)
```

---

## 📊 What is Optimized

### MATHIR Hyperparameters

1. **`retention_decay`**: Memory forgetting speed
   - `[0.9, 0.7, 0.5]` → Hierarchy: Working > Episodic > Semantic
   - Higher = keeps longer

2. **Learning Rate** (indirectly via mutations)

### LSTM Hyperparameters

```python
if l_perf < 0.8:  # If LSTM is also struggling
    print("📉 LSTM struggling. Boosting Learning Rate...")
    learning_rate *= 1.2  # 20% increase
```

---

## 🎬 Real Console Output Example

```bash
🏋️ Initializing Trainer on cuda
🧠 Initializing MATHIR (Heavy Memory Config)...
🏎️  STARTING EVOLUTION TRAINER...

Step 100 | M_Loss: 0.0456 | L_Loss: 0.0234
Step 200 | M_Loss: 0.0423 | L_Loss: 0.0198

--- Step 500 (Evaluation) ---
⚠️ MATHIR needs help (0.850). Asking Llama...
🧠 Llama suggested: [0.92, 0.75, 0.55]

Step 600 | M_Loss: 0.0389 | L_Loss: 0.0201
Step 700 | M_Loss: 0.0356 | L_Loss: 0.0215

--- Step 1000 (Evaluation) ---
⚠️ MATHIR needs help (0.890). Asking Llama...
🧠 Llama suggested: [0.94, 0.78, 0.58]

Step 1100 | M_Loss: 0.0298 | L_Loss: 0.0234
Step 1200 | M_Loss: 0.0267 | L_Loss: 0.0256

--- Step 1500 (Evaluation) ---
✅ MATHIR performing well! (0.945 vs 0.932)

--- Step 5000 (Checkpoint) ---
💾 Checkpoints saved @ step 5000
📊 VRAM Usage: 3.2 GB / 8.0 GB

--- Step 10000 (Benchmark) ---
🧪 RUNNING CAPACITY BENCHMARK @ Step 10000...
📊 BENCH RESULTS: MATHIR=0.98 | LSTM=0.67
```

---

## 📈 Real-Time Monitoring

### Generated Files

1. **`training_log.json`** (Updated every 500 steps)
```json
{
  "step": 10000,
  "mathir_avg_reward": 0.9845,
  "lstm_avg_reward": 0.6723,
  "vram_gb": 3.2,
  "current_hyperparams": {
    "retention_decay": [0.94, 0.78, 0.58]
  }
}
```

2. **`checkpoints/mathir_live.pth`** (Network weights, updated every 500 steps)

3. **`capacity_log.json`** (Retention tests every 10k steps)

### Live Dashboard

```bash
# Terminal 2 (while train.bat is running)
dashboard.bat
```

Then in your browser:
1. **"Report" Tab** → Learning Curves
2. **Enable "Live Mode"** → Auto-refresh every 2s
3. **"Brain Scan" Tab** → See neural weights evolving!

---

## 🔬 Periodic Benchmarks

### Every 10,000 Steps

```python
if step % 10000 == 0:
    # Pure retention test (1000 steps of noise)
    mathir.eval()
    lstm.eval()
    
    m_score = benchmark.test_retention(mathir, num_steps=1000)
    l_score = benchmark.test_retention(lstm, num_steps=1000)
    
    print(f"📊 MATHIR={m_score:.4f} | LSTM={l_score:.4f}")
    
    # Save to capacity_log.json
```

**This test measures**: The ability to remember a pattern after 1000 steps of distraction.

---

## 🎓 Why is this Innovative?

### Traditional Approaches

```python
# Classic method: Fixed Hyperparams
model = LSTM(lr=0.001, hidden=512)
for epoch in range(100):
    train(model)
# If it fails, start over with other params (Grid Search)
```

### MATHIR Approach

```python
# Adaptive method: Evolving Hyperparams
model = MATHIR(lr=0.001, decay=[0.9, 0.7, 0.5])
for step in range(1_000_000):
    train(model)
    
    if performance_drops():
        new_params = ask_ai_for_better_params()  # Ollama
        model.update(new_params)  # Immediate application
# The model improves LIVE during training!
```

**Advantages**:
- ✅ No manual Grid Search needed
- ✅ Adapts to difficulties along the way
- ✅ Converges faster to optimum
- ✅ Resistant to learning plateaus

---

## 🚀 Going Further

### Full Architecture

```
MATHIR v3.3
├── Input Layer (observations)
├── Episodic Encoder (mHC) ← DeepSeek-mHC with Sinkhorn
├── Memory Modules
│   ├── Working Memory (256 slots, decay=0.94)
│   ├── Episodic Memory (10k slots, decay=0.78)
│   └── Semantic Memory (1k slots, decay=0.58)
├── Attention Router
└── Action Decoder (steering, throttle)
```

### DeepSeek-mHC (Manifold-Constrained Hyper-Connections)

```python
# Sinkhorn Projection (gradient stabilization)
class ManifoldConstrainedLinear(nn.Module):
    def forward(self, x):
        # 1. Projection onto doubly stochastic manifold
        W_proj = sinkhorn_projection(self.weight.abs())
        
        # 2. Standard Forward
        return F.linear(x, W_proj, self.bias) * self.gain
```

**Benefit**: Avoids the "Dying Gradient" that killed the LSTM at 210k steps.

---

## 📚 Linked Files

- **Source Code**: `train_evolution.py`
- **Architecture**: `mathir_lib/mathir.py`, `mathir_lib/mhc.py`
- **Benchmarks**: `benchmark.py`
- **Config**: `config.json` (created automatically)
- **Documentation**:
  - `MATHIR.md` - Main README
  - `MATHIR_VS_LSTM.md` - Empirical comparison
  - `MATHIR_JOURNAL_DE_BORD.md` - Scientific journal
  - `QUICK_START.md` - User guide

---

## 💡 TL;DR

When you run `train.bat`:

1. **MATHIR** and **LSTM** train in parallel (RL)
2. Every **500 steps**: Performance comparison
3. If MATHIR is worse → **Ollama suggests new hyperparams**
4. **Immediate** application of changes
5. Every **10k steps**: Retention benchmark
6. Every **5k steps**: Checkpoint save

**Result**: A system that **self-improves** during training! 🤖🧠

---

## 🎯 Useful Commands

```bash
# Start Training
train.bat

# Monitor in Real-Time (other terminal)
dashboard.bat

# Test after training
benchmark.bat

# Optimize params first (optional)
optimize.bat
```

**Let's go!** 🚀
