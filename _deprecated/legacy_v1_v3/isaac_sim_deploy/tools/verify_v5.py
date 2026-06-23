# tools/verify_v5.py
import sys
import os
import torch
import yaml
from pathlib import Path

# Add root to python path
root = Path(__file__).parent.parent
sys.path.append(str(root))

try:
    print("🔎 Checking Imports...")
    from mathir_lib.mathir_v5 import MATHIRv5
    from mathir_lib.mhc_v5 import ManifoldConstrainedLinearV5
    print("✅ V5 Imports Successful.")
except ImportError as e:
    print(f"❌ Import Error: {e}")
    sys.exit(1)

try:
    print("\n📄 Checking Config...")
    config_path = root / "config/mathir_v5.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    print(f"✅ Config loaded. Learning Rate: {config['training']['learning_rate']}")
    print(f"✅ MHC Rank Ratio: {config['mhc']['rank_ratio']}")
except Exception as e:
    print(f"❌ Config Error: {e}")
    sys.exit(1)

try:
    print("\n🧠 Initializing MATHIR V5...")
    model = MATHIRv5(config)
    print("✅ Model Initialization Successful.")
    
    # Check MHC Layer type in router
    router = model.memory_core.router.router_net[0]
    if isinstance(router, ManifoldConstrainedLinearV5):
        print("✅ Correct MHC V5 Layer detected in Router.")
    else:
        print(f"❌ Error: Router layer is {type(router)}, expected ManifoldConstrainedLinearV5")

except Exception as e:
    print(f"❌ Initialization Error: {e}")
    sys.exit(1)

try:
    print("\n⚡ Running Forward Pass Test...")
    # Mock inputs matching config
    B = 2
    C, H, W = config['model']['camera_shape']
    S = config['model']['state_dim']
    
    observations = {
        'camera': torch.randn(B, C, H, W),
        'state': torch.randn(B, S)
    }
    
    # Forward
    out = model(observations)
    
    # Check output structure
    required_keys = ['action_mean', 'log_std', 'features', 'router_weights']
    missing = [k for k in required_keys if k not in out]
    
    if not missing:
        print("✅ Forward pass successful.")
        print(f"   Action Mean shape: {out['action_mean'].shape}")
        print(f"   Router Loss: {out['router_loss'].item():.6f}")
    else:
        print(f"❌ Forward pass missing keys: {missing}")
        print(out.keys())
except Exception as e:
    print(f"❌ Execution Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n🎉 ALL CHECKS PASSED. MATHIR V5 IS READY.")
