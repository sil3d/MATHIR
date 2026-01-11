"""
MATHIR Evolution Suite - Master Trainer (Reinforcement Learning)
================================================================

Entraînement continue comparatif (MATHIR vs LSTM) sur simulateur physique.
Ajustement dynamique des hyperparamètres (Evolutionary Strategy) via Ollama.
Monitoring de capacité mémoire tous les 10k steps.

Usage:
    python train_evolution.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import json
import time
import os
import random
import math
import subprocess
import re
import torch.nn.functional as F
import psutil # For System RAM
from datetime import datetime
from collections import deque
import numpy as np

from mathir_lib import MATHIR, LSTM
from driving_env import DrivingSimulator
from benchmark import RetentionBenchmark

# --- CONFIGURATION ---
CHECKPOINT_DIR = "checkpoints"
LOG_FILE = "training_log.json"
CAPACITY_LOG_FILE = "capacity_log.json"
ITERATIONS = 1000000
SAVE_EVERY = 5000 
EVAL_EVERY = 500   # Optimisation Hyperparam (Ollama - Lent)
LOG_EVERY = 10     # Mise à jour Dashboard (Rapide)
BENCHMARK_EVERY = 5000 # Torture Test (Strict comparison)

# Assurer existence dossiers
if os.path.exists(CHECKPOINT_DIR) and os.path.isfile(CHECKPOINT_DIR):
    print(f"⚠️ A file named '{CHECKPOINT_DIR}' prevents directory creation. Renaming to '{CHECKPOINT_DIR}_OLD.txt'")
    try:
        os.rename(CHECKPOINT_DIR, f"{CHECKPOINT_DIR}_OLD_{int(time.time())}.txt")
    except Exception as e:
        print(f"❌ Failed to rename conflicting file: {e}")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

class EvolutionTrainer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Environment
        print(f"🚀 Initializing Trainer on {self.device}")
        self.env = DrivingSimulator(device=self.device)
        self.env_lstm = DrivingSimulator(device=self.device) # Dual Environment for fairness
        
        # 1. Models - COMBAT DES TITANS (Heavy LSTM vs MATHIR)
        print("🏋️ Initializing HEAVY LSTM (1024 hidden, 3 layers)...")
        self.lstm = LSTM(
            action_dim=2,
            state_dim=9, # COGNITIVE LABYRINTH UPGRADE
            hidden_dim=1024,  
            num_layers=3      
        ).to(self.device)
        
        print("🧠 Initializing MATHIR (Heavy Memory Config)...")
        # Débridage total de la mémoire pour matcher la VRAM disponible
        heavy_memory_config = {
            'working_slots': 256,      # x4
            'episodic_slots': 10000,   # x10
            'semantic_slots': 1024     # x4
        }
        
        self.mathir = MATHIR(
            action_dim=2,
            state_dim=9, # COGNITIVE LABYRINTH UPGRADE
            hidden_dim=256,
            memory_config=heavy_memory_config
        ).to(self.device)
        
        # 2. Optimizers
        self.lr_mathir = 1e-4
        self.lr_lstm = 1e-4
        self.opt_mathir = optim.Adam(self.mathir.parameters(), lr=self.lr_mathir)
        self.opt_lstm = optim.Adam(self.lstm.parameters(), lr=self.lr_lstm)
        
        # 3. Environment
        self.env = DrivingSimulator(device=self.device)
        
        # 4. Metrics & History
        self.history = {
            "timestamps": [],
            "mathir_rewards": [],
            "lstm_rewards": [],
            "vram_usage": [],
            "ram_usage": []
        }
        
        self.benchmarks = [] # Persistent store for benchmarks
        self.current_decay = [0.9, 0.7, 0.5]
        
        # 6. Auto-Resume Logic
        self.start_step = 0
        self._try_load_logs() # Load benchmarks/history if restart
        self._try_load_latest_checkpoint()

    def _try_load_latest_checkpoint(self):
        """Tente de charger le dernier checkpoint disponible"""
        import glob
        import re
        
        ckpts = glob.glob(os.path.join(CHECKPOINT_DIR, "mathir_step_*.pth"))
        if not ckpts:
            print("🆕 Aucun checkpoint trouvé. Démarrage à zéro.")
            return
            
        # Trouver le step le plus élevé
        latest_ckpt = max(ckpts, key=os.path.getctime)
        match = re.search(r'step_(\d+)', latest_ckpt)
        if match:
            step = int(match.group(1))
            lstm_ckpt = os.path.join(CHECKPOINT_DIR, f"lstm_step_{step}.pth")
            
            if os.path.exists(lstm_ckpt):
                print(f"🔄 Reprise de l'entraînement au step {step}...")
                try:
                    self.mathir.load_state_dict(torch.load(latest_ckpt))
                    self.lstm.load_state_dict(torch.load(lstm_ckpt))
                    self.start_step = step
                    print("✅ Modèles chargés avec succès !")
                except Exception as e:
                    print(f"⚠️ Erreur lors du chargement : {e}")
                    self.start_step = 0

    def get_action_dist(self, out_dict):
        """Créé une distribution normale pour échantillonner l'action"""
        mean = torch.tanh(out_dict['action_mean']) # Bound -1 to 1
        std = torch.exp(out_dict['log_std'])
        return torch.distributions.Normal(mean, std)

        return torch.distributions.Normal(mean, std)

    def _try_load_logs(self):
        """Charge l'historique des logs pour ne pas perdre les graphes au restart"""
        try:
            if os.path.exists(LOG_FILE):
                with open(LOG_FILE, 'r') as f:
                    data = json.load(f)
                    
                # Load Benchmarks (Hall of Fame)
                if "benchmarks" in data:
                    self.benchmarks = data["benchmarks"]
                    print(f"📜 Loaded {len(self.benchmarks)} historical benchmarks.")
                    
                # Load History (Optional, for continuity)
                if "history" in data:
                    h = data["history"]
                    # We append only if lists are empty (start of run)
                    if not self.history["mathir_rewards"]:
                        self.history["mathir_rewards"] = h.get("mathir", [])
                        self.history["lstm_rewards"] = h.get("lstm", [])
                        self.history["vram_usage"] = h.get("vram", [])
                        self.history["ram_usage"] = h.get("ram", [])
                        print(f"📈 Restored training history ({len(self.history['mathir_rewards'])} points).")
        except Exception as e:
            print(f"⚠️ Log Load Error: {e}")

    def get_resource_usage(self):
        """Retourne (VRAM_Reservée, RAM_Sys_Utilisée, RAM_Sys_Percent)"""
        vram = 0.0
        if torch.cuda.is_available():
            # Memory Reserved is closer to Task Manager than Memory Allocated
            vram = torch.cuda.memory_reserved(self.device) / 1024**3 
            
        ram = psutil.virtual_memory()
        return vram, ram.used / 1024**3, ram.percent

    def save_checkpoint(self, step):
        path_m = os.path.join(CHECKPOINT_DIR, f"mathir_step_{step}.pth")
        path_l = os.path.join(CHECKPOINT_DIR, f"lstm_step_{step}.pth")
        torch.save(self.mathir.state_dict(), path_m)
        torch.save(self.lstm.state_dict(), path_l)
        print(f"💾 Checkpoints saved @ step {step}")

    def update_logs(self, step):
        if len(self.history["mathir_rewards"]) > 0:
            last_m_rew = np.mean(self.history["mathir_rewards"][-50:])
            last_l_rew = np.mean(self.history["lstm_rewards"][-50:])
            current_vram, current_ram, ram_pct = self.get_resource_usage()
            
            # Update history
            self.history["vram_usage"].append(current_vram)
            self.history["ram_usage"].append(current_ram)
            
            log_data = {
                "step": step,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "mathir_avg_reward": float(last_m_rew),
                "lstm_avg_reward": float(last_l_rew),
                "vram_gb": float(current_vram),
                "ram_gb": float(current_ram),
                "ram_percent": float(ram_pct),
                "current_hyperparams": {
                    "retention_decay": self.current_decay,
                    "lstm_lr": self.lr_lstm
                },
                "history": {
                    "mathir": self.history["mathir_rewards"][-200:], 
                    "lstm": self.history["lstm_rewards"][-200:],
                    "steps": list(range(max(0, step-200), step)),
                    "vram": self.history["vram_usage"][-200:],
                    "ram": self.history["ram_usage"][-200:]
                },
                "benchmarks": self.benchmarks # Preserve benchmarks
            }
            try:
                with open(LOG_FILE, 'w') as f:
                    json.dump(log_data, f)
            except:
                pass

    def run_torture_test(self, step):
        """
        ⚔️ TORTURE TEST PROTOCOL
        Runs models on EXACTLY the same 5 distinct seeds to strictly compare them.
        No training, just pure performance evaluation.
        """
        print(f"\n⚔️ STARTING TORTURE TEST @ Step {step}...")
        self.mathir.eval()
        self.lstm.eval()
        
        seeds = [101, 202, 303, 404, 505]
        scores_m = []
        scores_l = []
        
        for seed in seeds:
            # Strict Seeding for identical environment
            np.random.seed(seed) 
            torch.manual_seed(seed)
            
            # Test MATHIR
            
            self.env.reset()
            # Capture state for LSTM replay if needed, but seeding numpy should be enough if reset() uses np.random
            # The env uses np.random.uniform etc. So np.random.seed(seed) is sufficient.
            
            obs = self.env._get_observation() 
            total_r_m = 0
            self.mathir.reset_memory()
            for _ in range(200): # Limit max steps
                with torch.no_grad():
                    out = self.mathir(obs)
                    action = torch.tanh(out['action_mean'])
                    obs, reward, done, _ = self.env.step(action)
                    total_r_m += reward.item()
                    if done: break
            scores_m.append(total_r_m)

            # Test LSTM (Exact same condition)
            # Re-seed for LSTM (Crucial for fairness)
            np.random.seed(seed)
            torch.manual_seed(seed)
            # Actually easier to just force seeds inside env if we updated reset logic to accept seed
            # But the new env reset() is random. 
            # WORKAROUND: We trust the seed fixes numpy.random
            # The new env uses np.random. So strict seeding is needed.
            
            
            # Re-seed for LSTM already handled above, now executing
            self.env.reset()
            obs = self.env._get_observation()
            total_r_l = 0
            
            for _ in range(200):
                with torch.no_grad():
                    out = self.lstm(obs)
                    action = torch.tanh(out['action_mean'])
                    obs, reward, done, _ = self.env.step(action)
                    total_r_l += reward.item()
                    if done: break
            scores_l.append(total_r_l)

        avg_m = np.mean(scores_m)
        avg_l = np.mean(scores_l)
        winner = "MATHIR" if avg_m > avg_l else "LSTM"
        print(f"🏆 TORTURE RESULT: MATHIR={avg_m:.2f} vs LSTM={avg_l:.2f} => Winner: {winner}")

        # Log Result into main log file for Dashboard
        # Append to internal list first
        bench_entry = {
            "step": step,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "mathir_score": float(avg_m),
            "lstm_score": float(avg_l),
            "winner": winner,
            "params": self.current_decay
        }
        self.benchmarks.append(bench_entry)
        # Keep only last 50
        self.benchmarks = self.benchmarks[-50:]
        
        # Then save via update_logs
        self.update_logs(step)

        self.mathir.train()
        self.lstm.train()

    def ask_ollama(self, current_params, m_perf, lstm_lr):
        try:
            prompt = f"""
            You are an AI AutoML Expert optimizing a Neural Memory (MATHIR).
            CONTEXT:
            - Your model (MATHIR) is fighting against an LSTM.
            - The LSTM is cheating! Its Learning Rate has been boosted to: {lstm_lr:.6f}
            - MATHIR is currently struggling (Score: {m_perf:.3f}).
            
            Current Decay Rates: {current_params}
            
            Task: Output a JSON with new 'decay' list (3 floats 0.1-0.99) to beat the boosted LSTM.
            Strategy: If LSTM is fast, we need STABLE long-term memory to outsmart it (higher decay).
            JSON ONLY: {{ "decay": [0.95, 0.8, 0.6] }}
            """
            result = subprocess.run(
                ["ollama", "run", "llama3.2:3b", prompt],
                capture_output=True, text=True, encoding='utf-8', errors='ignore'
            )
            match = re.search(r'\{.*\}', result.stdout, re.DOTALL)
            if match:
                new_data = json.loads(match.group(0))
                return new_data.get('decay')
        except:
            return None

    def evolve_hyperparameters(self, m_perf, l_perf):
        if m_perf < l_perf * 1.1:
            print(f"⚠️ MATHIR needs help ({m_perf:.3f}). Asking Llama...")
            new_decay = self.ask_ollama(self.current_decay, m_perf, self.lr_lstm)
            if new_decay:
                print(f"🧠 Llama suggested: {new_decay}")
                self.current_decay = sorted([float(x) for x in new_decay], reverse=True)
                if hasattr(self.mathir, 'memory'):
                    self.mathir.memory.retention_decay = torch.tensor(
                        self.current_decay, device=self.device, dtype=torch.float
                    )
            else:
                mutation = np.random.uniform(-0.05, 0.05, size=3)
                new_decay = np.clip(np.array(self.current_decay) + mutation, 0.1, 0.99)
                self.current_decay = sorted([float(x) for x in new_decay], reverse=True)

        if l_perf < 0.8:
            print("📉 LSTM struggling. Boosting Learning Rate...")
            self.lr_lstm *= 1.2
            if self.lr_lstm > 1e-3: self.lr_lstm = 1e-4
            for param_group in self.opt_lstm.param_groups:
                param_group['lr'] = self.lr_lstm

    def train_loop(self):
        print("\n🏎️  STARTING EVOLUTION TRAINER (Heavy Class)...")
        obs = self.env.reset()
        obs_l = self.env_lstm.reset()
        
        self.mathir.reset_memory()
        self.lstm.reset_memory()
        
        done = False
        done_l = False
        
        # Resume step
        steps_range = range(self.start_step + 1, ITERATIONS + 1)
        
        for step in steps_range:
            # Forward & Action Sampling (Stochastic for exploration)
            # Use average reward from last 10 steps as performance cue (or 0 initially)
            perf_cue = np.mean(self.history["mathir_rewards"][-10:]) if len(self.history["mathir_rewards"]) > 10 else 0.0
            m_out = self.mathir(obs, step=step, performance_cue=perf_cue)
            m_dist = self.get_action_dist(m_out)
            m_action = m_dist.sample()
            m_log_prob = m_dist.log_prob(m_action).sum(dim=-1, keepdim=True)
            m_action_clamped = torch.clamp(m_action, -1.0, 1.0) # Physics need valid range

            l_out = self.lstm(obs_l)
            l_dist = self.get_action_dist(l_out)
            l_action = l_dist.sample()
            l_log_prob = l_dist.log_prob(l_action).sum(dim=-1, keepdim=True)
            l_action_clamped = torch.clamp(l_action, -1.0, 1.0)

            # Step (Using clamped actions)
            # Parallel Universes: MATHIR and LSTM drive in their own worlds
            obs_next, reward, done, _ = self.env.step(m_action_clamped)
            obs_l_next, reward_l, done_l, _ = self.env_lstm.step(l_action_clamped)
            
            # --- RL LOSS (REINFORCE / Policy Gradient) ---
            # Maximiser reward => Minimiser Loss (-log_prob * reward)
            
            m_loss = -m_log_prob * reward
            l_loss = -l_log_prob * reward_l # LSTM is now driving properly in its own env

            # --- OPTIMIZATION ---
            # Training MATHIR
            self.opt_mathir.zero_grad()
            m_loss.backward()
            self.opt_mathir.step()
            
            # Training LSTM (Baseline needs to be competent to be a benchmark)
            self.opt_lstm.zero_grad()
            l_loss.backward()
            self.opt_lstm.step()
            
            # Log Rewards (Actual raw rewards from env)
            self.history["mathir_rewards"].append(reward.item())
            self.history["lstm_rewards"].append(reward_l.item())
            
            # Resource Tracking
            vram, ram, _ = self.get_resource_usage()
            self.history["vram_usage"].append(vram)
            self.history["ram_usage"].append(ram)
            
            if step % 100 == 0:
                print(f"Step {step} | M_Loss: {m_loss.item():.4f} | L_Loss: {l_loss.item():.4f}")
            
            # --- FAST LOGGING (REAL-TIME DASHBOARD) ---
            if step % LOG_EVERY == 0:
                self.update_logs(step)
                # LIVE SNAPSHOT for Dashboard (Brain Scan)
                torch.save(self.mathir.state_dict(), os.path.join(CHECKPOINT_DIR, "mathir_live.pth"))

            # --- SLOW EVALUATION (EVOLUTION) ---
            if step % EVAL_EVERY == 0:
                m_avg = np.mean(self.history["mathir_rewards"][-EVAL_EVERY:])
                l_avg = np.mean(self.history["lstm_rewards"][-EVAL_EVERY:])
                self.evolve_hyperparameters(m_avg, l_avg)
                
            if step % SAVE_EVERY == 0:
                self.save_checkpoint(step)

            # --- PERIODIC TORTURE TEST ---
            if step % BENCHMARK_EVERY == 0:
                self.run_torture_test(step)
                
            obs = obs_next
            obs_l = obs_l_next
            
            if done:
                obs = self.env.reset()
                self.mathir.reset_memory()
            if done_l:
                obs_l = self.env_lstm.reset()
                self.lstm.reset_memory()

if __name__ == "__main__":
    trainer = EvolutionTrainer()
    trainer.train_loop()
