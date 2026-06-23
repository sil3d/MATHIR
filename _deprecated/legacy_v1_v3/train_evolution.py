"""
MATHIR Evolution Suite - Master Trainer (Reinforcement Learning)
================================================================

Continuous comparative training (MATHIR vs LSTM) on physical simulator.
Dynamic adjustment of hyperparameters (Evolutionary Strategy) via Ollama.
Memory capacity monitoring every 10k steps.

Usage:
    python train_evolution.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import json
import yaml
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


# --- CONFIGURATION ---
CHECKPOINT_DIR = "checkpoints"
LOG_FILE = "training_log.json"
CAPACITY_LOG_FILE = "capacity_log.json"
ITERATIONS = 1000000
SAVE_EVERY = 5000 
EVAL_EVERY = 500   # Hyperparam Optimization (Ollama - Slow)
LOG_EVERY = 10     # Dashboard Update (Fast)
BENCHMARK_EVERY = 5000 # Torture Test (Strict comparison)

# Ensure directories exist
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
        
        # USE ULTIMATE TORTURE SIMULATOR
        from mathir_lib.torture_sim import UltimateDrivingSimulator
        
        # Load V5 Config EARLY to configure Simulator
        with open("config/mathir_v5.yaml", "r") as f:
            v5_config = yaml.safe_load(f)
            
        sim_config = v5_config.get('training', {}).get('simulator_config', {})
        self.env = UltimateDrivingSimulator(device=self.device, config=sim_config)
        self.env_lstm = UltimateDrivingSimulator(device=self.device, config=sim_config) # Dual Environment for fairness
        
        # 1. Models - CLASH OF TITANS (Heavy LSTM vs MATHIR)
        print("🏋️ Initializing HEAVY LSTM (1024 hidden, 3 layers)...")
        self.lstm = LSTM(
            action_dim=2,
            state_dim=23, # COGNITIVE TORTURE UPGRADE (State vector is bigger now)
            hidden_dim=1024,  
            num_layers=3      
        ).to(self.device)
        
        print("🧠 Initializing MATHIR V5 (Production Ready)...")
        # v5_config is already loaded above
        print(f"🔧 Loaded MATHIR V5 Config with env state_dim={v5_config['model']['state_dim']}")
        
        # Override with Torture settings if needed, but V5 config should be authoritative
        # Checking compatibility
        if v5_config['model']['state_dim'] != 23:
             print("⚠️ Warning: Config state_dim != 23. Overriding for Torture Test.")
             v5_config['model']['state_dim'] = 23
             v5_config['mhc']['rank_ratio'] = 0.3 # Ensure MHC strong ranking
        
        from mathir_lib import MATHIRv5
        self.mathir = MATHIRv5(v5_config).to(self.device)
        
        # 2. Optimizers
        self.initial_lstm_lr = 1e-4
        self.lr_mathir = 1e-4
        self.lr_lstm = self.initial_lstm_lr
        self.opt_mathir = optim.Adam(self.mathir.parameters(), lr=self.lr_mathir)
        self.opt_lstm = optim.Adam(self.lstm.parameters(), lr=self.lr_lstm)
        
        # 3. Environment (Already Init)
        
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
        
        # BEST PARAMETERS TRACKING
        self.best_run = {
            "score": -float('inf'),
            "params": {},
            "step": 0,
            "timestamp": ""
        }
        
        # 6. Auto-Resume Logic
        self.start_step = 0
        self._try_load_logs() # Load benchmarks/history if restart
        self._try_load_latest_checkpoint()

    def _try_load_latest_checkpoint(self):
        """Attempts to load the latest available checkpoint"""
        import glob
        import re
        
        ckpts = glob.glob(os.path.join(CHECKPOINT_DIR, "mathir_step_*.pth"))
        if not ckpts:
            print("🆕 No checkpoint found. Starting from scratch.")
            return
            
        # Find the highest step
        latest_ckpt = max(ckpts, key=os.path.getctime)
        match = re.search(r'step_(\d+)', latest_ckpt)
        if match:
            step = int(match.group(1))
            lstm_ckpt = os.path.join(CHECKPOINT_DIR, f"lstm_step_{step}.pth")
            
            if os.path.exists(lstm_ckpt):
                print(f"🔄 Resuming training at step {step}...")
                try:
                    # Note: We might need to adjust state_dict loading if model changed
                    # But since we updated dims in init, strict=False helps if partial match
                    # However, changing state_dim usually breaks weights incompatible. 
                    # Assuming we start fresh or user accepts mismatch if previous weights exist.
                    self.mathir.load_state_dict(torch.load(latest_ckpt, weights_only=True), strict=False)
                    self.lstm.load_state_dict(torch.load(lstm_ckpt, weights_only=True), strict=False)
                    self.start_step = step
                    print("✅ Models loaded successfully (partial or complete)!")
                except Exception as e:
                    print(f"⚠️ Error during loading: {e}")
                    self.start_step = 0

    def get_action_dist(self, out_dict):
        """Creates a normal distribution to sample the action"""
        mean = torch.tanh(out_dict['action_mean']) # Bound -1 to 1
        std = torch.exp(out_dict['log_std'])
        return torch.distributions.Normal(mean, std)

    def _try_load_logs(self):
        """Loads log history to preserve graphs on restart"""
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
        """Returns (VRAM_Reserved, RAM_Sys_Used, RAM_Sys_Percent)"""
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
        
    def save_best_params(self):
        """Saves the best parameters found"""
        if self.best_run["score"] > -100:
            print(f"\n💎 Saving BEST PARAMETERS found at step {self.best_run['step']} (Score: {self.best_run['score']:.4f})")
            try:
                with open("mathir_best_params.json", "w") as f:
                    json.dump(self.best_run, f, indent=4)
                print("✅ mathir_best_params.json generated.")
            except Exception as e:
                print(f"❌ Failed to save best params: {e}")

    def update_logs(self, step, m_rew, l_rew, vram, ram, ram_pct, current_phase="INIT"):
        """Updates the JSON file for the dashboard"""
        
        # Data validation (NaN check)
        m_rew = 0.0 if np.isnan(m_rew) else m_rew
        l_rew = 0.0 if np.isnan(l_rew) else l_rew
        
        # Moving Average (visual smoothing)
        avg_len = 50
        last_m_rew = np.mean(self.history["mathir_rewards"][-avg_len:]) if len(self.history["mathir_rewards"]) >= avg_len else m_rew
        last_l_rew = np.mean(self.history["lstm_rewards"][-avg_len:]) if len(self.history["lstm_rewards"]) >= avg_len else l_rew

        # CHECK FOR NEW BEST
        if last_m_rew > self.best_run["score"]:
            self.best_run = {
                "score": float(last_m_rew),
                "params": {
                    "retention_decay": self.current_decay,
                    "hidden_dim": 256
                },
                "step": step,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            self.save_best_params()

        log_data = {
            "step": step,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "mathir_avg_reward": float(last_m_rew),
            "lstm_avg_reward": float(last_l_rew),
            "vram_gb": vram,
            "ram_gb": ram,
            "ram_percent": ram_pct,
            "current_hyperparams": {
                "retention_decay": self.current_decay,
                "lstm_lr": self.lr_lstm,
                "mathir_lr": self.lr_mathir,
                "scenario": current_phase
            },
            "history": {
                "mathir": self.history["mathir_rewards"][-1000:], 
                "lstm": self.history["lstm_rewards"][-1000:],
                "steps": self.history["timestamps"][-1000:],
                "vram": self.history["vram_usage"][-1000:],
                "ram": self.history["ram_usage"][-1000:]
            },
            "benchmarks": self.benchmarks
        }
        
        try:
            with open(LOG_FILE, 'w') as f:
                json.dump(log_data, f)
        except Exception as e:
            print(f"⚠️ Failed to write logs: {e}")

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
            self.env.config.seed = seed # Use Torture Config seed
            self.env.reset()
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
            np.random.seed(seed)
            torch.manual_seed(seed)
            self.env_lstm.config.seed = seed
            self.env_lstm.reset()
            obs = self.env_lstm._get_observation()
            total_r_l = 0
            
            for _ in range(200):
                with torch.no_grad():
                    out = self.lstm(obs)
                    action = torch.tanh(out['action_mean'])
                    obs, reward, done, _ = self.env_lstm.step(action)
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
        
        self.benchmarks = self.benchmarks[-50:]
        
        # Then save via update_logs using the torture test results as current performance
        vram, ram, ram_pct = self.get_resource_usage()
        # Determine current phase for logging (re-calculate or access if stored)
        # For simplicity in torture test (which happens at interval), we can just log "TORTURE TEST" or the current scheduled phase
        # Let's re-use the scheduler logic briefly or just pass a special flag
        phase_idx = (step // 30000) % 4
        scenarios = [
            "PHASE 1: EVOLUTION (Baseline vs Doped)",
            "PHASE 2: SCIENTIFIC STANDARD (Fair Fight)",
            "PHASE 3: UNLEASHED (Mathir 3e-4 vs Doped)",
            "PHASE 4: CHAOS (Both 1e-3)"
        ]
        current_phase = scenarios[phase_idx]
        
        self.update_logs(step, avg_m, avg_l, vram, ram, ram_pct, current_phase)

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

    def evolve_hyperparameters(self, m_perf, l_perf, force_lr_update=False):
        # 1. Ask Ollama if MATHIR is struggling (Internal Plasticity still allowed)
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

        # 2. Dynamic Handicap for LSTM (ONLY if not controlled by Scenario)
        if force_lr_update:
            print(f"🔒 Learning Rates locked by Scenario Scheduler. Skipping dynamic adjustment.")
        else:
            if m_perf > l_perf * 1.5:
                 # MATHIR Dominates -> Reset
                 print(f"⚖️ MATHIR Dominating (>50% gap). Resetting LSTM Handicap.")
                 self.lr_lstm = self.initial_lstm_lr 
            elif l_perf < 0.8:
                # LSTM struggles -> Boost it
                print("📉 LSTM struggling. Boosting Learning Rate...")
                self.lr_lstm *= 1.2
                if self.lr_lstm > 1e-3: self.lr_lstm = 1e-3 

            # Apply changes
            for param_group in self.opt_lstm.param_groups:
                param_group['lr'] = self.lr_lstm

    def train_loop(self):
        print("\n🏎️  STARTING EVOLUTION TRAINER (Heavy Class)...")
        try:
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
                
                # MATHIR V5 Loss Components (CRITICAL FIX)
                m_policy_loss = -m_log_prob * reward
                
                # Add Stability Terms (KL Constraint + Entropy Bonus)
                # We want to minimize router_loss (keep close to prior)
                # We want to maximize router_entropy (minimize -entropy)
                router_loss = m_out.get('router_loss', 0.0)
                entropy_loss = -0.01 * m_out.get('router_entropy', 0.0)
                
                m_loss = m_policy_loss + router_loss + entropy_loss
                
                l_loss = -l_log_prob * reward_l # LSTM (Standard)

                # --- OPTIMIZATION ---
                # Training MATHIR
                self.opt_mathir.zero_grad()
                m_loss.backward()
                # 🛡️ GRADIENT CLIPPING (Prevents "Gradient Cliff" / Explosion)
                torch.nn.utils.clip_grad_norm_(self.mathir.parameters(), max_norm=1.0)
                self.opt_mathir.step()
                
                # Training LSTM (Baseline needs to be competent to be a benchmark)
                self.opt_lstm.zero_grad()
                l_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.lstm.parameters(), max_norm=1.0)
                self.opt_lstm.step()
                
                # Log Rewards (Actual raw rewards from env)
                self.history["mathir_rewards"].append(reward.item())
                self.history["lstm_rewards"].append(reward_l.item())
                self.history["timestamps"].append(step)
                
                # Resource Tracking
                vram, ram, _ = self.get_resource_usage()
                self.history["vram_usage"].append(vram)
                self.history["ram_usage"].append(ram)
                
                if step % 100 == 0:
                    print(f"Step {step} | M_Loss: {m_loss.item():.4f} | L_Loss: {l_loss.item():.4f}")
                
                # --- SCENARIO SCHEDULER (Every 30k Steps) ---
                current_phase = "INIT"
                phase_idx = (step // 30000) % 4
                
                if phase_idx == 0:
                    current_phase = "PHASE 1: EVOLUTION (Baseline vs Doped)"
                elif phase_idx == 1:
                    current_phase = "PHASE 2: SCIENTIFIC STANDARD (Fair Fight)"
                elif phase_idx == 2:
                    current_phase = "PHASE 3: UNLEASHED (Mathir 3e-4 vs Doped)"
                elif phase_idx == 3:
                    current_phase = "PHASE 4: CHAOS (Both 1e-3)"

                # --- FAST LOGGING (REAL-TIME DASHBOARD) ---
                if step % LOG_EVERY == 0:
                    # Calc momentary averages for log
                    m_avg_log = np.mean(self.history["mathir_rewards"][-50:]) if len(self.history["mathir_rewards"]) > 0 else 0
                    l_avg_log = np.mean(self.history["lstm_rewards"][-50:]) if len(self.history["lstm_rewards"]) > 0 else 0
                    vram_log, ram_log, ram_pct_log = self.get_resource_usage()
                    
                    self.update_logs(step, m_avg_log, l_avg_log, vram_log, ram_log, ram_pct_log, current_phase)
                    
                    # LIVE SNAPSHOT for Dashboard (Brain Scan)
                    torch.save(self.mathir.state_dict(), os.path.join(CHECKPOINT_DIR, "mathir_live.pth"))

                # --- SLOW EVALUATION (EVOLUTION) ---
                # Calculate averages here, as they are needed by the Scenario Scheduler
                avg_mathir = np.mean(self.history["mathir_rewards"][-EVAL_EVERY:])
                avg_lstm = np.mean(self.history["lstm_rewards"][-EVAL_EVERY:])

                # --- SCENARIO SCHEDULER (Every 30k Steps) ---
                current_phase = "UNKNOWN"
                phase_idx = (step // 30000) % 4
                
                if phase_idx == 0:
                    current_phase = "PHASE 1: EVOLUTION (Baseline vs Doped)"
                    # Standard behavior (AutoML manages LRs -> Enable Dynamic Update)
                    if step % 500 == 0:
                        self.evolve_hyperparameters(avg_mathir, avg_lstm, force_lr_update=False)
                        # Add this update_logs call as per instruction
                        vram_gb, ram_gb, ram_percent = self.get_resource_usage() # Re-get for this specific log
                        self.update_logs(step, avg_mathir, avg_lstm, vram_gb, ram_gb, ram_percent, current_phase)
                
                elif phase_idx == 1:
                    current_phase = "PHASE 2: SCIENTIFIC STANDARD (Fair Fight)"
                    # User Request: No Doping, MATHIR Boosted
                    self.lr_lstm = 1e-4 # Reset to Initial
                    self.lr_mathir = 3e-4 # Karpathy Constant
                    
                    # Force apply
                    for pg in self.opt_lstm.param_groups: pg['lr'] = self.lr_lstm
                    for pg in self.opt_mathir.param_groups: pg['lr'] = self.lr_mathir
                    
                    if step % 500 == 0:
                        # Only evolve decay, not LRs (Locked for Scientific Standard)
                        self.evolve_hyperparameters(avg_mathir, avg_lstm, force_lr_update=True) 

                elif phase_idx == 2:
                    current_phase = "PHASE 3: UNLEASHED (Mathir 3e-4 vs Doped)"
                    self.lr_lstm = 1e-3 # Max Doping
                    self.lr_mathir = 3e-4 # Boosted Mathir
                    
                    for pg in self.opt_lstm.param_groups: pg['lr'] = self.lr_lstm
                    for pg in self.opt_mathir.param_groups: pg['lr'] = self.lr_mathir
                    
                    if step % 500 == 0:
                        self.evolve_hyperparameters(avg_mathir, avg_lstm, force_lr_update=True)

                elif phase_idx == 3:
                    current_phase = "PHASE 4: CHAOS (Both 1e-3)"
                    self.lr_lstm = 1e-3
                    self.lr_mathir = 1e-3
                    
                    for pg in self.opt_lstm.param_groups: pg['lr'] = self.lr_lstm
                    for pg in self.opt_mathir.param_groups: pg['lr'] = self.lr_mathir
                    
                    if step % 500 == 0:
                        self.evolve_hyperparameters(avg_mathir, avg_lstm, force_lr_update=True)

                # --- END SCENARIO ---
                    
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
                    
        finally:
            self.save_best_params()
            print("🛑 Training Stopped.")

if __name__ == "__main__":
    trainer = EvolutionTrainer()
    trainer.train_loop()
