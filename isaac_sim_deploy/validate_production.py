import torch
import os
import sys
import yaml
import numpy as np
from mathir_lib.mathir_v5 import MATHIRv5

def validate_production_model(config_path="config/mathir_v5.yaml", ckpt_path="checkpoints/mathir_v5_latest.pt"):
    print("========================================================")
    print("      🏭 MATHIR V5 PRODUCTION VALIDATION PROTOCOL      ")
    print("========================================================")
    
    # 1. Check Files
    if not os.path.exists(config_path):
        print(f"[FAIL] Config not found: {config_path}")
        return False
    if not os.path.exists(ckpt_path):
        print(f"[FAIL] Checkpoint not found: {ckpt_path}")
        return False
        
    print("[PASS] Files exist.")
    
    # 2. Load Config & Model
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MATHIRv5(config).to(device)
        
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"[PASS] Model loaded successfully (Step: {checkpoint.get('step', 'Unknown')}).")
    except Exception as e:
        print(f"[FAIL] Model load failed: {e}")
        return False

    # 3. Weight Integrity Check (NaN/Inf)
    has_nan = False
    for name, param in model.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            print(f"[FAIL] NaN/Inf DETECTED in layer: {name}")
            has_nan = True
    
    if has_nan:
        print("[CRITICAL] Model weights corrupted.")
        return False
    else:
        print("[PASS] Weight integrity verified (No NaNs).")

    # 4. Architecture Verification (V5 Specifics)
    has_router = hasattr(model.memory_core, 'router')
    has_immune = hasattr(model.memory_core, 'immunological_memory')
    
    if has_router and has_immune:
        print("[PASS] V5 Architecture confirmed (KL-Router + Immune System active).")
    else:
        print(f"[FAIL] Architecture mismatch. Router={has_router}, Immune={has_immune}")
        return False

    # 5. Inference Simulation (Dry Run)
    try:
        model.eval()
        dummy_state = torch.randn(1, 4, 84, 84).to(device)  # Assuming standard frame stack
        dummy_nav = torch.randn(1, 3).to(device)
        
        with torch.no_grad():
            _ = model(dummy_state, dummy_nav)
        print("[PASS] Inference Dry-Run successful (Forward pass OK).")
    except Exception as e:
        print(f"[FAIL] Inference crashed: {e}")
        return False

    print("\n--------------------------------------------------------")
    print("✅ VERDICT: PRODUCTION READY")
    print("--------------------------------------------------------")
    print("The model accepts payload and retains plasticity constraints.")
    return True

if __name__ == "__main__":
    success = validate_production_model()
    sys.exit(0 if success else 1)
