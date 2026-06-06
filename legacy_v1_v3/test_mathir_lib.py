"""
Tests d'intégration pour mathir_lib
Test que la bibliothèque fonctionne correctement
"""

import torch
from mathir_lib import MATHIR, LSTM, count_parameters, estimate_memory_usage

def test_import():
    """Test que les imports fonctionnent"""
    print("✓ Imports OK")

def test_mathir_instantiation():
    """Test création modèle MATHIR"""
    model = MATHIR(
        camera_shape=(3, 84, 84),
        state_dim=5,
        action_dim=2
    )
    print(f"✓ MATHIR créé: {count_parameters(model):,} paramètres")
    return model

def test_lstm_instantiation():
    """Test création modèle LSTM"""
    model = LSTM(
        camera_shape=(3, 84, 84),
        state_dim=5,
        action_dim=2
    )
    print(f"✓ LSTM créé: {count_parameters(model):,} paramètres")
    return model

def test_mathir_forward():
    """Test forward pass MATHIR"""
    model = MATHIR()
    
    obs = {
        'camera': torch.randn(8, 1, 84, 84),
        'state': torch.randn(8, 5)
    }
    
    output = model(obs, step=0)
    
    assert 'action_mean' in output
    assert 'features' in output
    assert output['action_mean'].shape == (8, 2)
    
    print(f"✓ MATHIR forward: {output['action_mean'].shape}")

def test_lstm_forward():
    """Test forward pass LSTM"""
    model = LSTM()
    
    obs = {
        'camera': torch.randn(8, 1, 84, 84),
        'state': torch.randn(8, 5)
    }
    
    output = model(obs)
    
    assert 'action_mean' in output
    assert 'features' in output
    assert output['action_mean'].shape == (8, 2)
    
    print(f"✓ LSTM forward: {output['action_mean'].shape}")

def test_mathir_memory_stats():
    """Test statistiques mémoire MATHIR"""
    model = MATHIR()
    
    # Forward pour remplir un peu la mémoire
    for i in range(10):
        obs = {
            'camera': torch.randn(1, 1, 84, 84),
            'state': torch.randn(1, 5)
        }
        _ = model(obs, step=i)
    
    stats = model.get_memory_stats()
    
    assert 'working_usage' in stats
    assert 'episodic_usage' in stats
    assert 'semantic_usage' in stats
    
    print(f"✓ MATHIR memory stats: {stats}")

def test_memory_reset():
    """Test reset mémoire"""
    mathir = MATHIR()
    lstm = LSTM()
    
    # Remplir mémoires
    for i in range(20):
        obs = {
            'camera': torch.randn(1, 1, 84, 84),
            'state': torch.randn(1, 5)
        }
        _ = mathir(obs, step=i)
        _ = lstm(obs)
    
    # Reset
    mathir.reset_memory()
    lstm.reset_memory()
    
    # Vérifier reset
    mathir_stats = mathir.get_memory_stats()
    assert mathir_stats['working_usage'] == 0
    assert mathir_stats['episodic_usage'] == 0
    
    lstm_stats = lstm.get_memory_stats()
    assert lstm_stats['hidden_norm'] == 0.0
    assert lstm_stats['cell_norm'] == 0.0
    
    print("✓ Memory reset OK")

def test_custom_config():
    """Test configuration personnalisée"""
    model = MATHIR(
        camera_shape=(3, 224, 224),  # Haute résolution
        state_dim=10,                # Plus d'état
        action_dim=4,                # Plus d'actions
        hidden_dim=512               # Plus de capacité
    )
    
    obs = {
        'camera': torch.randn(4, 3, 224, 224),
        'state': torch.randn(4, 10)
    }
    
    output = model(obs)
    assert output['action_mean'].shape == (4, 4)
    
    print(f"✓ Custom config OK: {count_parameters(model):,} params")

def test_batch_sizes():
    """Test différentes tailles de batch"""
    model = MATHIR()
    
    for batch_size in [1, 8, 16, 32]:
        obs = {
            'camera': torch.randn(batch_size, 1, 84, 84),
            'state': torch.randn(batch_size, 5)
        }
        
        output = model(obs)
        assert output['action_mean'].shape == (batch_size, 2)
    
    print("✓ Batch sizes OK: 1, 8, 16, 32")

def run_all_tests():
    """Lance tous les tests"""
    print("\n" + "="*60)
    print("  TESTS MATHIR_LIB")
    print("="*60 + "\n")
    
    try:
        test_import()
        mathir = test_mathir_instantiation()
        lstm = test_lstm_instantiation()
        test_mathir_forward()
        test_lstm_forward()
        test_mathir_memory_stats()
        test_memory_reset()
        test_custom_config()
        test_batch_sizes()
        
        print("\n" + "="*60)
        print("  ✅ TOUS LES TESTS PASSENT!")
        print("="*60)
        print("\nmathir_lib est prêt à l'emploi! 🚀")
        print("\nUtilisation:")
        print("  from mathir_lib import MATHIR, LSTM")
        print("  model = MATHIR()")
        
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_all_tests()
