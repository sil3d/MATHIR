#!/usr/bin/env python3
"""
Install dependencies based on detected hardware and configuration
"""

import subprocess
import sys
import json
from pathlib import Path

def install_based_on_config(config_path="config/mathir_optimized.yaml"):
    """Install dependencies based on configuration"""
    
    # Read configuration
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("📦 Installing dependencies based on configuration...")
    
    # Base dependencies (always required)
    base_packages = [
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "opencv-python>=4.7.0",
        "Pillow>=9.5.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
        "psutil>=5.9.0",
        "GPUtil>=1.4.0"
    ]
    
    # RL dependencies if training
    if config.get('training'):
        base_packages.extend([
            "gym>=0.26.0",
            "stable-baselines3>=2.0.0",
            "tensorboard>=2.13.0",
            "wandb>=0.15.0"
        ])
    
    # Simulator-specific dependencies
    simulator = config.get('training', {}).get('simulator')
    if simulator == 'carla':
        base_packages.append("carla>=0.9.15")
    
    # Edge deployment dependencies
    deployment = config.get('deployment', {})
    if deployment.get('optimizations', {}).get('precision') == 'int8':
        base_packages.extend([
            "onnx>=1.14.0",
            "onnxruntime>=1.15.0"
        ])
    
    if 'tensorrt' in str(deployment.get('target', '')).lower():
        base_packages.append("pycuda>=2022.2.2")
    
    if deployment.get('tflite'):
        base_packages.append("tflite-runtime>=2.14.0")
    
    # Install packages
    print(f"Installing {len(base_packages)} packages...")
    
    for package in base_packages:
        print(f"  Installing {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except subprocess.CalledProcessError:
            print(f"  ⚠️  Failed to install {package}")
    
    print("\n✅ Dependencies installed!")
    
    # Platform-specific instructions
    target = deployment.get('target', '')
    if 'jetson' in str(target).lower():
        print("\n📝 For Jetson devices, also run:")
        print("  sudo apt-get update")
        print("  sudo apt-get install python3-pip python3-opencv")
        print("  pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/ros2")
    
    elif 'raspberry' in str(target).lower():
        print("\n📝 For Raspberry Pi, also run:")
        print("  sudo apt-get update")
        print("  sudo apt-get install python3-opencv libopenblas-dev")
        print("  pip3 install numpy --no-binary numpy")  # Build from source for ARM

if __name__ == "__main__":
    install_based_on_config()