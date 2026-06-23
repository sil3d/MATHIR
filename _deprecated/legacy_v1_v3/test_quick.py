"""
Script de test rapide pour vérifier le benchmark
Sans lancer l'interface Streamlit
"""

import torch
from mathir_model import MATHIRAgent, LSTMBaseline, count_parameters
from benchmark import CompleteBenchmarkSuite
import json


def quick_test():
    """Test rapide de fonctionnalité"""
    
    print("="*60)
    print("  MATHIR vs LSTM - Test Rapide")
    print("="*60)
    print()
    
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print()
    
    # Initialisation modèles
    print("1. Initialisation des modèles...")
    mathir = MATHIRAgent()
    lstm = LSTMBaseline()
    
    print(f"   ✓ MATHIR: {count_parameters(mathir):,} paramètres")
    print(f"   ✓ LSTM:   {count_parameters(lstm):,} paramètres")
    print()
    
    # Test forward pass
    print("2. Test forward pass...")
    batch_size = 4
    dummy_obs = {
        'camera': torch.randn(batch_size, 1, 84, 84),
        'state': torch.randn(batch_size, 5)
    }
    
    with torch.no_grad():
        mathir_out = mathir(dummy_obs)
        lstm_out = lstm(dummy_obs, reset_hidden=True)
    
    print(f"   ✓ MATHIR output: {mathir_out['action_mean'].shape}")
    print(f"   ✓ LSTM output:   {lstm_out['action_mean'].shape}")
    print()
    
    # Test mémoire
    print("3. Test statistiques mémoire...")
    mathir_stats = mathir.get_memory_stats()
    lstm_stats = lstm.get_memory_stats()
    
    print(f"   MATHIR:")
    print(f"      - Working usage: {mathir_stats['working_usage']}")
    print(f"      - Episodic usage: {mathir_stats['episodic_usage']}")
    print(f"      - Semantic usage: {mathir_stats['semantic_usage']:.0f}")
    
    print(f"   LSTM:")
    print(f"      - Hidden norm: {lstm_stats['hidden_norm']:.4f}")
    print(f"      - Cell norm: {lstm_stats['cell_norm']:.4f}")
    print()
    
    print("="*60)
    print("  ✓ Tous les tests passés!")
    print("="*60)
    print()
    print("Pour lancer le benchmark complet:")
    print("  python benchmark.py")
    print()
    print("Pour lancer l'interface Streamlit:")
    print("  streamlit run app_streamlit.py")
    print("  ou double-cliquez sur run_benchmark.bat")
    print()


if __name__ == "__main__":
    quick_test()
