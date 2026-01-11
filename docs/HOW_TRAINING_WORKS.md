# Comment Fonctionne l'Entraînement MATHIR ? 🧠

## Vue d'Ensemble

Quand tu lances `train.bat`, tu ne fais pas juste entraîner un réseau de neurones. Tu lances un **système d'auto-optimisation intelligent** qui combine :

- **Reinforcement Learning** (apprentissage par renforcement)
- **Meta-Learning** (apprentissage à apprendre)
- **AI-Driven Optimization** (IA qui optimise l'IA)

---

## 🎯 Les Composants

### 1. Les Architectures (Combattants)

```python
# MATHIR - Le Challenger (Architecture Avancée)
mathir = MATHIR(
    hidden_dim=256,
    memory_config={
        'working_slots': 256,
        'episodic_slots': 10000,  # Mémoire à long terme
        'semantic_slots': 1024
    }
)

# LSTM - Le Champion Actuel (Baseline)
lstm = LSTM(
    hidden_dim=1024,  # Plus gros pour être équitable
    num_layers=3
)
```

**Différence clé** :
- **LSTM** : Oublie rapidement (mémoire cache limitée)
- **MATHIR** : Mémoire épisodique 10k+ événements + projection Sinkhorn

---

### 2. L'Algorithme d'Entraînement (Reinforcement Learning)

```python
# Boucle d'entraînement (train_evolution.py)
for step in range(1, 1_000_000):
    # 1. Observation de l'environnement
    obs = env.get_state()  # Position, vitesse, obstacles
    
    # 2. MATHIR décide d'une action
    m_action = mathir(obs)  # Ex: "Tourner à gauche 15°"
    
    # 3. LSTM décide aussi
    l_action = lstm(obs)
    
    # 4. Exécution et récompense
    obs_next, reward = env.step(m_action)
    
    # 5. Backpropagation
    loss = reward_function(m_action, target_optimal)
    optimizer.step()  # Mise à jour des poids
```

**C'est du Supervised Imitation Learning** : Le réseau apprend à imiter une trajectoire idéale (sinusoïde).

---

## 🤖 L'Optimisation Automatique (Le Secret Sauce)

### Diagramme du Flux

![Flux d'Optimisation](docs/images/training_optimization_flow.png)

### Code Détaillé

#### Tous les 500 Steps (30 secondes sur GPU)

```python
if step % 500 == 0:
    # 1. Calculer les performances moyennes
    m_avg = np.mean(mathir_rewards[-500:])  # Ex: 0.85
    l_avg = np.mean(lstm_rewards[-500:])    # Ex: 0.92
    
    # 2. MATHIR est-il moins bon ?
    if m_avg < l_avg * 1.1:  # Tolérance 10%
        print(f"⚠️ MATHIR needs help ({m_avg:.3f}). Asking Llama...")
        
        # 3. Appel à Ollama (Llama 3.2)
        prompt = """
        You are an AI AutoML Expert.
        Current Decay: [0.9, 0.7, 0.5]
        Current Accuracy: 0.85 (Goal: 1.0)
        Task: Suggest BETTER 'decay' rates.
        JSON ONLY: {"decay": [0.95, 0.8, 0.6]}
        """
        
        response = ollama.run("llama3.2:3b", prompt)
        # Réponse en 2-3 secondes
        
        # 4. Parser la réponse JSON
        new_decay = json.loads(response)["decay"]
        # Ex: [0.92, 0.75, 0.55]
        
        # 5. Application EN DIRECT
        mathir.memory.retention_decay = torch.tensor(new_decay)
        print(f"🧠 Llama suggested: {new_decay}")
```

#### Fallback si Ollama Échoue

```python
else:
    # Mutation aléatoire (Evolutionary Strategy)
    mutation = np.random.uniform(-0.05, 0.05, size=3)
    new_decay = np.clip(current_decay + mutation, 0.1, 0.99)
    current_decay = sorted(new_decay, reverse=True)
```

---

## 📊 Ce qui est Optimisé

### Hyperparamètres MATHIR

1. **`retention_decay`** : Vitesse d'oubli de la mémoire
   - `[0.9, 0.7, 0.5]` → Hiérarchie : Working > Episodic > Semantic
   - Plus haut = garde plus longtemps

2. **Learning Rate** (indirectement via mutations)

### Hyperparamètres LSTM

```python
if l_perf < 0.8:  # Si LSTM lutte aussi
    print("📉 LSTM struggling. Boosting Learning Rate...")
    learning_rate *= 1.2  # Augmentation de 20%
```

---

## 🎬 Exemple de Sortie Console Réelle

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

## 📈 Monitoring en Temps Réel

### Fichiers Générés

1. **`training_log.json`** (Mis à jour tous les 500 steps)
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

2. **`checkpoints/mathir_live.pth`** (Poids du réseau, mis à jour tous les 500 steps)

3. **`capacity_log.json`** (Tests de rétention tous les 10k steps)

### Dashboard Live

```bash
# Terminal 2 (pendant que train.bat tourne)
dashboard.bat
```

Puis dans le navigateur :
1. **Onglet "Rapport"** → Courbes d'apprentissage
2. **Activer "Mode Live"** → Rafraîchissement auto toutes les 2s
3. **Onglet "Brain Scan"** → Voir les poids neuronaux évoluer !

---

## 🔬 Les Benchmarks Périodiques

### Tous les 10,000 Steps

```python
if step % 10000 == 0:
    # Test de rétention pure (1000 steps de bruit)
    mathir.eval()
    lstm.eval()
    
    m_score = benchmark.test_retention(mathir, num_steps=1000)
    l_score = benchmark.test_retention(lstm, num_steps=1000)
    
    print(f"📊 MATHIR={m_score:.4f} | LSTM={l_score:.4f}")
    
    # Sauvegarde dans capacity_log.json
```

**Ce test mesure** : La capacité à se souvenir d'un pattern après 1000 steps de distraction.

---

## 🎓 Pourquoi C'est Innovant ?

### Approches Traditionnelles

```python
# Méthode classique : Hyperparams fixes
model = LSTM(lr=0.001, hidden=512)
for epoch in range(100):
    train(model)
# Si ça marche pas, on recommence avec d'autres params (Grid Search)
```

### Approche MATHIR

```python
# Méthode adaptative : Hyperparams évolutifs
model = MATHIR(lr=0.001, decay=[0.9, 0.7, 0.5])
for step in range(1_000_000):
    train(model)
    
    if performance_drops():
        new_params = ask_ai_for_better_params()  # Ollama
        model.update(new_params)  # Application immédiate
# Le modèle s'améliore EN DIRECT pendant l'entraînement !
```

**Avantages** :
- ✅ Pas besoin de Grid Search manuel
- ✅ S'adapte aux difficultés en cours de route
- ✅ Converge plus vite vers l'optimum
- ✅ Résistant aux plateaux d'apprentissage

---

## 🚀 Pour Aller Plus Loin

### Architecture Complète

```
MATHIR v3.3
├── Input Layer (observations)
├── Episodic Encoder (mHC) ← DeepSeek-mHC avec Sinkhorn
├── Memory Modules
│   ├── Working Memory (256 slots, decay=0.94)
│   ├── Episodic Memory (10k slots, decay=0.78)
│   └── Semantic Memory (1k slots, decay=0.58)
├── Attention Router
└── Action Decoder (steering, throttle)
```

### DeepSeek-mHC (Manifold-Constrained Hyper-Connections)

```python
# Projection Sinkhorn (stabilisation du gradient)
class ManifoldConstrainedLinear(nn.Module):
    def forward(self, x):
        # 1. Projection sur variété doublement stochastique
        W_proj = sinkhorn_projection(self.weight.abs())
        
        # 2. Forward standard
        return F.linear(x, W_proj, self.bias) * self.gain
```

**Bénéfice** : Évite le "Dying Gradient" qui a tué le LSTM à 210k steps.

---

## 📚 Fichiers Liés

- **Code source** : `train_evolution.py`
- **Architecture** : `mathir_lib/mathir.py`, `mathir_lib/mhc.py`
- **Benchmarks** : `benchmark.py`
- **Config** : `config.json` (créé automatiquement)
- **Documentation** :
  - `MATHIR.md` - README principal
  - `MATHIR_VS_LSTM.md` - Comparaison empirique
  - `MATHIR_JOURNAL_DE_BORD.md` - Journal scientifique
  - `QUICK_START.md` - Guide utilisateur

---

## 💡 TL;DR

Quand tu lances `train.bat` :

1. **MATHIR** et **LSTM** s'entraînent en parallèle (RL)
2. Tous les **500 steps** : Comparaison des perfs
3. Si MATHIR est moins bon → **Ollama suggère de nouveaux hyperparams**
4. Application **immédiate** des changements
5. Tous les **10k steps** : Benchmark de rétention
6. Tous les **5k steps** : Sauvegarde checkpoint

**Résultat** : Un système qui **s'auto-améliore** pendant l'entraînement ! 🤖🧠

---

## 🎯 Commandes Utiles

```bash
# Lancer l'entraînement
train.bat

# Suivre en temps réel (autre terminal)
dashboard.bat

# Tester après l'entraînement
benchmark.bat

# Optimiser les params avant (optionnel)
optimize.bat
```

**C'est parti !** 🚀
