"""
generate_dr_pool.py
Generates offline Domain Randomization pool for MATHIR V5.
Uses UltimateDrivingSimulator to create diverse training samples.

simulation sans données réelles en utilisant le simulateur torture_sim.py : 
python tools/generate_dr_pool.py --size 100000

imulateur (Abstraction / Logique) : 
python tools/generate_dr_pool.py --source sim --output dr_pool_sim.pt

Webcam (Calibration Réelle Rapide) :
Branchez une webcam et pointez-la vers un circuit ou la route. python tools/generate_dr_pool.py --source camera --output dr_pool_real.pt

Vidéo GoPro / Youtube (Dataset Ultra-Réaliste) : 
python tools/generate_dr_pool.py --source "C:/Users/.../mon_trajet.mp4" --output dr_pool_high_fi.pt


"""

import sys
import os
import torch
import torch.nn as nn
from tqdm import tqdm
import yaml
from pathlib import Path

# Add root to python path
root = Path(__file__).parent.parent
sys.path.append(str(root))

try:
    from mathir_lib.torture_sim import UltimateDrivingSimulator
    from mathir_lib.mathir_v5 import DomainRandomizationManager
except ImportError as e:
    print(f"❌ Import Error: {e}")
    sys.exit(1)

def generate_pool(config_path, output_path, pool_size=1000, source='sim'):
    print(f"🚀 Generating DR Pool...")
    print(f"   Config: {config_path}")
    print(f"   Output: {output_path}")
    print(f"   Size:   {pool_size}")
    
    # Load Config
    with open(config_path) as f:
        config = yaml.safe_load(f)
        
    # Setup Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Device: {device}")
    
    # Initialize Simulator or Real Camera
    if source == 'sim':
        try:
            env = UltimateDrivingSimulator(device=device)
            # Sim Reset
            obs = env.reset()
            # In sim mode, we loop until pool is full
            source_iter = None
        except Exception as e:
            print(f"❌ Simulator Init Error: {e}")
            env = UltimateDrivingSimulator(device=device)
    else:
        # Real World Source (Webcam or Video)
        import cv2
        print(f"🎥 Opening video source: {source}")
        cap = cv2.VideoCapture(0 if source == 'camera' else source)
        if not cap.isOpened():
            print(f"❌ Failed to open source {source}")
            sys.exit(1)
        env = None
        source_iter = cap

    # Initialize DR Manager (for online augmentations which we will burn in)
    dr_config = config.get('perception', {})
    dr_config['dr_mode'] = 'online'
    dr_manager = DomainRandomizationManager(dr_config)
    
    # Storage
    pool = []
    
    print("\n📸 Collecting observations...")
    pbar = tqdm(total=pool_size)
    
    # Resize transform for real world
    import torchvision.transforms as T
    H, W = config['model']['camera_shape'][1], config['model']['camera_shape'][2]
    resize = T.Compose([
        T.ToPILImage(),
        T.Resize((H, W)),
        T.ToTensor()
    ])
    
    while len(pool) < pool_size:
        camera_tensor = None
        
        if env:
            # Simulator Mode
            camera_tensor = obs['camera'] # [1, C, H, W]
            
            # Step env
            action = torch.zeros(1, 2).to(device)
            action[0, 0] = torch.rand(1).item() * 2 - 1 
            action[0, 1] = torch.rand(1).item() * 1.0 
            obs, _, done, _ = env.step(action)
            if done: obs = env.reset()
            
        elif source_iter:
            # Real World Mode
            ret, frame = source_iter.read()
            if not ret:
                print("End of video stream.")
                break
                
            # OpenCV is BGR, convert to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Preprocess: [H, W, C] -> [C, H, W] resized
            # Using torchvision transforms
            camera_tensor = resize(frame).unsqueeze(0).to(device) # [1, C, H, W]
            
        if camera_tensor is not None:
            # Apply Online Augmentation to create diversity over real data
            # e.g. different lighting on top of real frames
            augmented_img = dr_manager._apply_online_augmentations(camera_tensor)
            
            # Store (move to CPU)
            pool.append(augmented_img.squeeze(0).cpu())
            pbar.update(1)
        
    pbar.close()
    if source_iter: source_iter.release()
    
    # Stack into a single tensor [N, C, H, W]
    print("\n📦 Stacking and saving...")
    full_pool = torch.stack(pool)
    torch.save(full_pool, output_path)
    
    print(f"✅ Saved DR Pool to {output_path}")
    print(f"   Source: {source}")
    print(f"   Shape: {full_pool.shape}")
    print(f"   Size: {full_pool.element_size() * full_pool.nelement() / 1024**2:.2f} MB")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=1000, help="Number of samples")
    parser.add_argument("--config", type=str, default="config/mathir_v5.yaml")
    parser.add_argument("--source", type=str, default="sim", 
                      help="'sim' for TortureSim, 'camera' for Webcam, or path to video file")
    parser.add_argument("--output", type=str, default="dr_pool.pt", help="Output file path")
    args = parser.parse_args()
    
    # Ensure config exists
    if not os.path.exists(args.config):
        # Fallback path if running from root
         if os.path.exists(os.path.join("MATHIR", args.config)):
            args.config = os.path.join("MATHIR", args.config)
            
    generate_pool(args.config, args.output, args.size, args.source)

