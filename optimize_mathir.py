"""
MATHIR Hyperparameter Optimizer (Powered by Ollama)
===================================================

Ce script utilise une boucle d'optimisation intelligente:
1. Lance un benchmark rapide
2. Analyse les résultats
3. Demande à Ollama (Llama 3.2) de proposer de meilleurs hyperparamètres
4. Met à jour la configuration et recommence
5. S'arrête quand la performance n'évolue plus (convergence)

Usage:
    python optimize_mathir.py
"""

import json
import subprocess
import time
import re
import os
from copy import deepcopy
import torch
import numpy as np

# Import du benchmark rapide
from benchmark import RetentionBenchmark, GeneralizationBenchmark
from mathir_lib import MATHIR, LSTM

CONFIG_FILE = "config.json"
OLLAMA_MODEL = "llama3.2:3b"

def load_config():
    with open(CONFIG_FILE, 'r') as f:
        return json.load(f)

def save_config(config):
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=4)

def run_evaluation(config):
    """
    Lance une évaluation rapide de la configuration actuelle.
    Retourne un score global (0-100).
    """
    print("  Running evaluation...", end="", flush=True)
    
    # 1. Instancier MATHIR avec la config actuelle (simulé via modification de params)
    # Note: Dans une vraie implémentation, MATHIR devrait lire config.json.
    # Ici on suppose que le code utilise les valeurs par défaut qui viennent d'être updatées
    # ou on passe les params explicitement.
    
    try:
        mathir = MATHIR(
            hidden_dim=config.get('mathir_settings', {}).get('hidden_dim', 256)
        )
        # Hack pour injecter les hyperparamètres dynamiquement dans l'instance
        if hasattr(mathir, 'memory'):
            decay_str = config.get('memory', {}).get('retention_decay', "[0.9, 0.7, 0.5]")
            decay_list = json.loads(decay_str) if isinstance(decay_str, str) else decay_str
            mathir.memory.retention_decay = torch.tensor(decay_list)
            
            mathir.memory.working_capacity = config.get('memory', {}).get('working_memory_size', 64)
            mathir.memory.episodic_capacity = config.get('memory', {}).get('episodic_memory_size', 1000)
    
        lstm = LSTM()
        
        # 2. Test Rétention (Rapide: 500 steps)
        retention_bench = RetentionBenchmark()
        ret_score = retention_bench.test_retention(mathir, num_steps=500)
        
        # 3. Test Généralisation (Rapide: 1 scénario)
        gen_bench = GeneralizationBenchmark()
        gen_score = gen_bench.test_scenario(mathir, 'city')
        
        # Score combiné
        total_score = (ret_score * 0.6) + (gen_score * 0.4)
        print(f" Score: {total_score:.4f} (Ret: {ret_score:.2f}, Gen: {gen_score:.2f})")
        
        return total_score, ret_score, gen_score
        
    except Exception as e:
        print(f" Failed: {e}")
        return 0.0, 0.0, 0.0

def ask_ollama(current_config, current_score, history):
    """
    Demande à Ollama de proposer de nouveaux paramètres.
    """
    prompt = f"""
    You are an AI Hyperparameter Optimizer for a Neural Memory architecture called MATHIR.
    
    Current Performance Score: {current_score:.4f} (Higher is better, max 1.0)
    
    History of attempts (Score -> Params):
    {history[:3]} ... (showing last 3)
    
    Current Configuration (JSON):
    {json.dumps(current_config['memory'], indent=2)}
    
    Your task: Suggest BETTER hyperparameters to improve the Score.
    
    Focus on:
    1. 'retention_decay': A list of 3 floats between 0.1 and 0.99. Controls memory fade.
    2. 'working_memory_size': Int between 32 and 128.
    3. 'episodic_memory_size': Int between 500 and 5000.
    
    RETURN ONLY VALID JSON matching the structure of the 'memory' block above. 
    DO NOT explain. JUST JSON.
    """
    
    print("  asking Ollama...", end="", flush=True)
    
    try:
        result = subprocess.run(
            ["ollama", "run", OLLAMA_MODEL, prompt],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='ignore',
            timeout=30
        )
        
        response = result.stdout.strip()
        
        # Extraction JSON bourrin (cherche entre { et })
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            json_str = match.group(0)
            new_params = json.loads(json_str)
            print(" Done.")
            return new_params
        else:
            print(" Failed to parse JSON.")
            return None
            
    except Exception as e:
        print(f" Error: {e}")
        return None

def main():
    print("=== MATHIR AUTO-OPTIMIZER (with Ollama) ===\n")
    
    # Check Ollama
    try:
        subprocess.run(["ollama", "--version"], capture_output=True, check=True)
    except:
        print("❌ Ollama not found. Please install Ollama.")
        return

    config = load_config()
    best_config = deepcopy(config)
    best_score = -1.0
    history = []
    
    patience = 3
    no_improve_count = 0
    
    for iteration in range(10): # Max 10 itérations
        print(f"\n--- Iteration {iteration+1} ---")
        
        # 1. Évaluer
        score, ret, gen = run_evaluation(config)
        
        if score > best_score:
            print(f"🚀 NEW BEST SCORE! ({score:.4f} > {best_score:.4f})")
            best_score = score
            best_config = deepcopy(config)
            save_config(best_config) # Sauvegarde les meilleurs
            no_improve_count = 0
        else:
            print(f"  No improvement.")
            no_improve_count += 1
        
        history.append(f"Score {score:.3f}: decay={config['memory'].get('retention_decay')}")
        
        # Condition d'arrêt
        if no_improve_count >= patience:
            print("\n🛑 CONVERGENCE REACHED. Stopping.")
            break
            
        # 2. Demander nouveaux paramètres
        new_memory_params = ask_ollama(config, score, history[-3:])
        
        if new_memory_params:
            # Appliquer changements
            # Validation basique
            if 'retention_decay' in new_memory_params:
                config['memory']['retention_decay'] = new_memory_params['retention_decay']
            if 'working_memory_size' in new_memory_params:
                 config['memory']['working_memory_size'] = new_memory_params['working_memory_size']
            
            print(f"  New Decay: {config['memory'].get('retention_decay')}")
            
    print("\n=== OPTIMIZATION COMPLETE ===")
    print(f"Best Score: {best_score:.4f}")
    print("Configuration saved to config.json")

if __name__ == "__main__":
    main()
