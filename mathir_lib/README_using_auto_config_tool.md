# MATHIR Configuration Wizard

A hardware-aware configuration system that automatically detects your hardware and suggests optimized settings for training and deploying MATHIR-based agents.

## Features

- 🔍 **Automatic Hardware Detection**: Detects CPU, GPU, RAM, and installed simulators
- 🎯 **Optimized Presets**: Pre-configured settings for common hardware combinations
- 🎮 **Simulator Integration**: Auto-detects Isaac Sim, CARLA, Gazebo
- 💻 **Edge Device Support**: Optimized configurations for Jetson, Raspberry Pi
- ⚡ **Performance-Aware**: Adjusts batch sizes, model variants, and optimizations based on capabilities

## Quick Start

```bash
# Make setup script executable
chmod +x quick_setup.sh

# Run the setup wizard
./quick_setup.sh

# Or run manually
python3 configure_mathir.py









# **Hardware-Aware Configuration Wizard for MATHIR**

## **configure_mathir.py**
```python
#!/usr/bin/env python3
"""
MATHIR Configuration Wizard - Hardware-Aware Configuration Generator
Detects hardware, recommends optimized settings, and generates YAML configuration.
"""

import os
import sys
import platform
import subprocess
import json
import yaml
import torch
import psutil
import GPUtil
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

class HardwareDetector:
    """Detects hardware capabilities and available simulation environments"""
    
    def __init__(self):
        self.system_info = self._get_system_info()
        self.gpu_info = self._get_gpu_info()
        self.simulators = self._detect_simulators()
        
    def _get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        return {
            'os': platform.system(),
            'os_version': platform.version(),
            'arch': platform.machine(),
            'cpu': platform.processor(),
            'cpu_count_physical': psutil.cpu_count(logical=False),
            'cpu_count_logical': psutil.cpu_count(logical=True),
            'ram_gb': round(psutil.virtual_memory().total / (1024**3), 2),
            'ram_available_gb': round(psutil.virtual_memory().available / (1024**3), 2),
            'python_version': platform.python_version(),
            'torch_available': torch.cuda.is_available() if torch.cuda.is_available() else False,
            'torch_version': torch.__version__,
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None
        }
    
    def _get_gpu_info(self) -> Dict[str, Any]:
        """Detect GPU information using multiple methods"""
        gpu_info = {
            'count': 0,
            'models': [],
            'memory_gb': 0,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None
        }
        
        try:
            # Try GPUtil first
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_info['count'] = len(gpus)
                gpu_info['models'] = [gpu.name for gpu in gpus]
                gpu_info['memory_gb'] = round(sum(gpu.memoryTotal for gpu in gpus) / 1024, 2)
            elif torch.cuda.is_available():
                # Fallback to torch
                gpu_info['count'] = torch.cuda.device_count()
                gpu_info['models'] = [torch.cuda.get_device_name(i) for i in range(gpu_info['count'])]
                if gpu_info['count'] > 0:
                    gpu_info['memory_gb'] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        except Exception as e:
            print(f"⚠️  Could not detect GPU details: {e}")
        
        return gpu_info
    
    def _detect_simulators(self) -> Dict[str, Dict[str, Any]]:
        """Detect installed simulation environments"""
        simulators = {
            'isaac_sim': {'installed': False, 'path': None, 'version': None},
            'carla': {'installed': False, 'path': None, 'version': None},
            'gazebo': {'installed': False, 'path': None, 'version': None},
            'unity': {'installed': False, 'path': None, 'version': None}
        }
        
        # Check for Isaac Sim
        isaac_paths = [
            Path.home() / '.local/share/ov/pkg/isaac_sim*',
            '/isaac_sim',
            '/opt/isaac_sim'
        ]
        for path_pattern in isaac_paths:
            matches = list(Path(path_pattern.parent).glob(path_pattern.name))
            if matches:
                simulators['isaac_sim']['installed'] = True
                simulators['isaac_sim']['path'] = str(matches[0])
                # Try to get version
                version_file = matches[0] / 'version.txt'
                if version_file.exists():
                    simulators['isaac_sim']['version'] = version_file.read_text().strip()
                break
        
        # Check for CARLA
        try:
            import carla
            simulators['carla']['installed'] = True
            simulators['carla']['version'] = carla.__version__
        except ImportError:
            # Check for CARLA Docker
            result = subprocess.run(['docker', 'images', 'carlasim/carla', '--format', '{{.Tag}}'], 
                                  capture_output=True, text=True)
            if result.returncode == 0 and result.stdout.strip():
                simulators['carla']['installed'] = True
                simulators['carla']['version'] = result.stdout.strip().split('\n')[0]
        
        # Check for Gazebo
        try:
            result = subprocess.run(['gz', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                simulators['gazebo']['installed'] = True
                simulators['gazebo']['version'] = result.stdout.strip()
        except FileNotFoundError:
            pass
        
        # Check for Unity (MARS/UnitySim)
        unity_paths = [
            Path('/Applications/Unity/Hub/Editor'),
            Path.home() / 'Unity/Hub/Editor',
            Path('C:/Program Files/Unity/Hub/Editor')
        ]
        for path in unity_paths:
            if path.exists():
                simulators['unity']['installed'] = True
                simulators['unity']['path'] = str(path)
                break
        
        return simulators
    
    def print_detected_hardware(self):
        """Display detected hardware information"""
        print("\n" + "="*60)
        print("🤖 DETECTED HARDWARE & SIMULATORS")
        print("="*60)
        
        # System info
        print("\n📊 SYSTEM INFORMATION:")
        print(f"  OS: {self.system_info['os']} {self.system_info['os_version']}")
        print(f"  Architecture: {self.system_info['arch']}")
        print(f"  CPU: {self.system_info['cpu']}")
        print(f"  CPU Cores: {self.system_info['cpu_count_physical']} physical, {self.system_info['cpu_count_logical']} logical")
        print(f"  RAM: {self.system_info['ram_gb']} GB total, {self.system_info['ram_available_gb']} GB available")
        
        # GPU info
        print("\n🎮 GPU INFORMATION:")
        if self.gpu_info['count'] > 0:
            print(f"  GPU Count: {self.gpu_info['count']}")
            for i, model in enumerate(self.gpu_info['models']):
                print(f"  GPU {i+1}: {model}")
            print(f"  Total VRAM: {self.gpu_info['memory_gb']} GB")
            if self.gpu_info['cuda_available']:
                print(f"  CUDA Version: {self.gpu_info['cuda_version']}")
        else:
            print("  No GPU detected")
        
        # Simulators
        print("\n🔄 DETECTED SIMULATORS:")
        for sim_name, sim_info in self.simulators.items():
            status = "✅ Installed" if sim_info['installed'] else "❌ Not installed"
            version = f" (v{sim_info['version']})" if sim_info['version'] else ""
            print(f"  {sim_name.title()}: {status}{version}")
            if sim_info['path']:
                print(f"    Path: {sim_info['path']}")
        
        print("="*60)

class ConfigurationWizard:
    """Interactive wizard for generating optimized configurations"""
    
    def __init__(self, hardware_detector: HardwareDetector):
        self.hardware = hardware_detector
        self.config = {}
        
        # Hardware presets for known platforms
        self.hardware_presets = {
            # Training/Development PCs
            'high_end_pc': {
                'description': 'High-end PC (RTX 4090, 64GB RAM, i9)',
                'min_gpu_memory': 16,
                'min_ram': 32,
                'cpu_cores': 16,
                'training_batch_size': 32,
                'num_envs': 16,
                'perception_backbone': 'efficientnet_b2',
                'model_variant': 'xlarge'
            },
            'mid_range_pc': {
                'description': 'Mid-range PC (RTX 3080, 32GB RAM, i7)',
                'min_gpu_memory': 8,
                'min_ram': 16,
                'cpu_cores': 8,
                'training_batch_size': 16,
                'num_envs': 8,
                'perception_backbone': 'efficientnet_b0',
                'model_variant': 'standard'
            },
            'entry_level_pc': {
                'description': 'Entry-level PC (RTX 3060, 16GB RAM, i5)',
                'min_gpu_memory': 6,
                'min_ram': 8,
                'cpu_cores': 6,
                'training_batch_size': 8,
                'num_envs': 4,
                'perception_backbone': 'mobilenet_v3',
                'model_variant': 'tiny'
            },
            'cpu_only': {
                'description': 'CPU-only system',
                'min_gpu_memory': 0,
                'min_ram': 16,
                'cpu_cores': 8,
                'training_batch_size': 4,
                'num_envs': 2,
                'perception_backbone': 'mobilenet_v3',
                'model_variant': 'tiny'
            },
            
            # Edge devices
            'jetson_agx_orin': {
                'description': 'NVIDIA Jetson AGX Orin (64GB)',
                'is_edge': True,
                'target_precision': 'fp16',
                'inference_batch_size': 1,
                'model_variant': 'standard',
                'optimization_level': 3,
                'use_tensorrt': True
            },
            'jetson_orin_nx': {
                'description': 'NVIDIA Jetson Orin NX (16GB)',
                'is_edge': True,
                'target_precision': 'fp16',
                'inference_batch_size': 1,
                'model_variant': 'standard',
                'optimization_level': 2,
                'use_tensorrt': True
            },
            'jetson_nano': {
                'description': 'NVIDIA Jetson Nano (4GB)',
                'is_edge': True,
                'target_precision': 'int8',
                'inference_batch_size': 1,
                'model_variant': 'tiny',
                'optimization_level': 1,
                'use_tensorrt': True
            },
            'raspberry_pi_5': {
                'description': 'Raspberry Pi 5 (8GB)',
                'is_edge': True,
                'target_precision': 'int8',
                'inference_batch_size': 1,
                'model_variant': 'tiny',
                'optimization_level': 1,
                'use_tensorrt': False,
                'use_tflite': True
            },
            'raspberry_pi_4': {
                'description': 'Raspberry Pi 4 (4GB)',
                'is_edge': True,
                'target_precision': 'int8',
                'inference_batch_size': 1,
                'model_variant': 'tiny',
                'optimization_level': 1,
                'use_tensorrt': False,
                'use_tflite': True
            }
        }
        
        # Simulator presets
        self.simulator_presets = {
            'isaac_sim': {
                'perception': {
                    'camera_resolution': [256, 256],
                    'lidar_channels': 16,
                    'imu_frequency': 100
                },
                'training': {
                    'physics_dt': 0.01,
                    'rendering_dt': 0.016,
                    'max_episode_length': 1000
                },
                'env_config': {
                    'envs_per_stage': 16,
                    'env_spacing': 2.0
                }
            },
            'carla': {
                'perception': {
                    'camera_resolution': [256, 256],
                    'camera_fov': 90,
                    'lidar_channels': 32
                },
                'training': {
                    'town': 'Town10HD',
                    'weather_dynamic': True,
                    'max_episode_length': 2000
                },
                'env_config': {
                    'traffic_vehicles': 20,
                    'traffic_pedestrians': 30
                }
            },
            'gazebo': {
                'perception': {
                    'camera_resolution': [224, 224],
                    'lidar_channels': 8
                },
                'training': {
                    'physics_update_rate': 1000,
                    'real_time_factor': 1.0,
                    'max_episode_length': 500
                },
                'env_config': {
                    'robot_model': 'turtlebot3_waffle',
                    'world': 'empty.world'
                }
            }
        }
    
    def run_interactive_setup(self):
        """Run interactive configuration wizard"""
        print("\n" + "="*60)
        print("⚙️  MATHIR CONFIGURATION WIZARD")
        print("="*60)
        
        # Step 1: Detect current hardware
        print("\n📊 STEP 1: DETECTING CURRENT HARDWARE...")
        self.hardware.print_detected_hardware()
        
        # Step 2: Ask for use case
        print("\n🎯 STEP 2: SELECT USE CASE")
        print("What is your primary use case?")
        print("  1. Training agent (using simulator)")
        print("  2. Deploying trained agent (inference only)")
        print("  3. Both training and deployment")
        
        while True:
            try:
                use_case = int(input("\nSelect option (1-3): "))
                if 1 <= use_case <= 3:
                    break
                else:
                    print("Please enter a number between 1 and 3")
            except ValueError:
                print("Please enter a valid number")
        
        # Step 3: Ask about simulation
        simulators_available = [k for k, v in self.hardware.simulators.items() if v['installed']]
        
        if use_case in [1, 3] and simulators_available:
            print(f"\n🔄 STEP 3: SELECT SIMULATOR (Detected: {', '.join(simulators_available)})")
            print("Available simulators:")
            for i, sim in enumerate(simulators_available, 1):
                print(f"  {i}. {sim.replace('_', ' ').title()}")
            print(f"  {len(simulators_available) + 1}. Other/Manual configuration")
            
            sim_choice = int(input(f"\nSelect simulator (1-{len(simulators_available) + 1}): "))
            if sim_choice <= len(simulators_available):
                selected_simulator = simulators_available[sim_choice - 1]
            else:
                selected_simulator = input("Enter simulator name (isaac_sim/carla/gazebo): ")
        else:
            selected_simulator = None
        
        # Step 4: Ask about target hardware
        print("\n💻 STEP 4: SELECT TARGET HARDWARE")
        print("Training/Development hardware (for running simulators):")
        
        # Auto-detect best training preset
        training_preset = self._auto_detect_training_preset()
        print(f"  Auto-detected: {training_preset['description']}")
        
        print("\nDeployment/Target hardware (for inference):")
        print("Edge devices:")
        for i, (key, preset) in enumerate(self.hardware_presets.items(), 1):
            if preset.get('is_edge', False):
                print(f"  {i}. {preset['description']}")
        
        print(f"  {len([p for p in self.hardware_presets.values() if p.get('is_edge', False)]) + 1}. Custom/Other")
        
        target_choice = int(input(f"\nSelect target hardware (1-{len([p for p in self.hardware_presets.values() if p.get('is_edge', False)]) + 1}): "))
        
        edge_presets = [k for k, v in self.hardware_presets.items() if v.get('is_edge', False)]
        if target_choice <= len(edge_presets):
            target_hardware = edge_presets[target_choice - 1]
        else:
            print("\nEnter custom hardware specifications:")
            target_hardware = {
                'description': input("Description: "),
                'is_edge': True,
                'target_precision': input("Target precision (fp32/fp16/int8): "),
                'inference_batch_size': int(input("Inference batch size (usually 1): ")),
                'model_variant': input("Model variant (tiny/standard/xlarge): "),
                'ram_gb': float(input("RAM in GB: "))
            }
        
        # Step 5: Generate configuration
        print("\n⚡ STEP 5: GENERATING OPTIMIZED CONFIGURATION...")
        self._generate_configuration(use_case, selected_simulator, training_preset, target_hardware)
        
        # Step 6: Show and confirm
        print("\n📋 GENERATED CONFIGURATION:")
        print("="*60)
        print(yaml.dump(self.config, default_flow_style=False, sort_keys=False))
        print("="*60)
        
        save = input("\n💾 Save configuration to config/mathir_optimized.yaml? (y/n): ")
        if save.lower() == 'y':
            self._save_configuration()
            print("✅ Configuration saved!")
        else:
            print("Configuration not saved.")
        
        return self.config
    
    def _auto_detect_training_preset(self) -> Dict[str, Any]:
        """Auto-detect the best training preset based on hardware"""
        gpu_memory = self.hardware.gpu_info['memory_gb']
        system_ram = self.hardware.system_info['ram_gb']
        
        if gpu_memory >= 16 and system_ram >= 32:
            return self.hardware_presets['high_end_pc']
        elif gpu_memory >= 8 and system_ram >= 16:
            return self.hardware_presets['mid_range_pc']
        elif gpu_memory >= 6 and system_ram >= 8:
            return self.hardware_presets['entry_level_pc']
        else:
            return self.hardware_presets['cpu_only']
    
    def _generate_configuration(self, use_case: int, simulator: Optional[str], 
                               training_preset: Dict[str, Any], target_hardware: Dict[str, Any]):
        """Generate optimized YAML configuration"""
        
        # Base configuration
        self.config = {
            'metadata': {
                'generated_by': 'MATHIR Configuration Wizard',
                'timestamp': subprocess.getoutput('date'),
                'use_case': ['training', 'deployment', 'both'][use_case - 1],
                'training_hardware': training_preset['description'],
                'target_hardware': target_hardware['description'] if isinstance(target_hardware, str) else target_hardware.get('description', 'Custom'),
                'simulator': simulator
            }
        }
        
        # Model configuration based on target
        if isinstance(target_hardware, str):
            target_preset = self.hardware_presets[target_hardware]
        else:
            target_preset = target_hardware
        
        self.config['model'] = {
            'variant': target_preset.get('model_variant', 'standard'),
            'perception': {
                'backbone': training_preset.get('perception_backbone', 'efficientnet_b0'),
                'input_resolution': [256, 256] if not simulator else self.simulator_presets.get(simulator, {}).get('perception', {}).get('camera_resolution', [256, 256])
            },
            'memory': {
                'mathir_variant': target_preset.get('model_variant', 'standard')
            }
        }
        
        # Training configuration (if needed)
        if use_case in [1, 3]:
            self.config['training'] = {
                'simulator': simulator,
                'hardware_optimized': True,
                'batch_size': training_preset.get('training_batch_size', 16),
                'parallel_environments': training_preset.get('num_envs', 8),
                'use_mixed_precision': self.hardware.gpu_info['cuda_available'],
                'gradient_accumulation_steps': 2 if training_preset.get('training_batch_size', 16) > 16 else 1
            }
            
            # Simulator-specific settings
            if simulator and simulator in self.simulator_presets:
                self.config['training']['simulator_config'] = self.simulator_presets[simulator]
        
        # Deployment configuration (if needed)
        if use_case in [2, 3]:
            self.config['deployment'] = {
                'target': target_hardware if isinstance(target_hardware, str) else target_hardware.get('description', 'Custom'),
                'optimizations': {
                    'precision': target_preset.get('target_precision', 'fp16'),
                    'batch_size': target_preset.get('inference_batch_size', 1),
                    'optimization_level': target_preset.get('optimization_level', 2)
                },
                'runtime': {
                    'max_latency_ms': self._get_target_latency(target_preset),
                    'memory_budget_mb': self._get_memory_budget(target_preset),
                    'power_mode': 'MAXN' if 'jetson' in str(target_hardware).lower() else 'balanced'
                }
            }
            
            # Edge-specific optimizations
            if target_preset.get('use_tensorrt', False):
                self.config['deployment']['tensorrt'] = {
                    'workspace_size': 1024 if target_preset.get('target_precision') == 'int8' else 2048,
                    'optimization_level': target_preset.get('optimization_level', 2),
                    'enable_dla': 'agx' in str(target_hardware).lower()
                }
            
            if target_preset.get('use_tflite', False):
                self.config['deployment']['tflite'] = {
                    'optimizations': ['DEFAULT', 'EXPERIMENTAL_SPARSITY'],
                    'num_threads': 2
                }
        
        # Hardware-aware optimizations
        self.config['hardware_optimizations'] = {
            'cpu': {
                'num_workers': min(8, self.hardware.system_info['cpu_count_physical']),
                'pin_memory': self.hardware.gpu_info['cuda_available']
            },
            'gpu': {
                'enabled': self.hardware.gpu_info['cuda_available'],
                'memory_fraction': 0.9,
                'allow_growth': True
            },
            'memory': {
                'gradient_checkpointing': self.hardware.gpu_info['memory_gb'] < 8,
                'offload_to_cpu': self.hardware.gpu_info['memory_gb'] < 6
            }
        }
        
        # Domain randomization based on hardware
        if use_case in [1, 3] and self.hardware.gpu_info['cuda_available']:
            dr_intensity = 'high' if self.hardware.gpu_info['memory_gb'] > 8 else 'medium'
            self.config['domain_randomization'] = {
                'mode': 'online' if self.hardware.gpu_info['memory_gb'] > 12 else 'hybrid',
                'intensity': dr_intensity,
                'augmentations': self._get_augmentation_preset(dr_intensity)
            }
    
    def _get_target_latency(self, target_preset: Dict[str, Any]) -> int:
        """Get target latency based on hardware"""
        if 'jetson' in str(target_preset.get('description', '')).lower():
            if 'agx' in str(target_preset.get('description', '')).lower():
                return 20  # ms
            elif 'nx' in str(target_preset.get('description', '')).lower():
                return 30
            else:
                return 50
        elif 'raspberry' in str(target_preset.get('description', '')).lower():
            return 100
        else:
            return 50
    
    def _get_memory_budget(self, target_preset: Dict[str, Any]) -> int:
        """Get memory budget based on hardware"""
        desc = str(target_preset.get('description', '')).lower()
        if '64gb' in desc:
            return 4000  # MB
        elif '16gb' in desc:
            return 2000
        elif '8gb' in desc:
            return 1000
        elif '4gb' in desc:
            return 500
        else:
            return 1000
    
    def _get_augmentation_preset(self, intensity: str) -> Dict[str, Any]:
        """Get domain randomization augmentations based on intensity"""
        presets = {
            'low': {
                'color_jitter': 0.1,
                'gaussian_noise': 0.01,
                'motion_blur': False,
                'random_crop': False
            },
            'medium': {
                'color_jitter': 0.2,
                'gaussian_noise': 0.02,
                'motion_blur': True,
                'random_crop': True,
                'probability': 0.5
            },
            'high': {
                'color_jitter': 0.3,
                'gaussian_noise': 0.03,
                'motion_blur': True,
                'random_crop': True,
                'random_rotation': 5.0,
                'random_scale': 0.1,
                'probability': 0.7
            }
        }
        return presets.get(intensity, presets['medium'])
    
    def _save_configuration(self):
        """Save configuration to YAML file"""
        config_dir = Path('config')
        config_dir.mkdir(exist_ok=True)
        
        config_path = config_dir / 'mathir_optimized.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
        
        # Also save hardware info for reference
        hardware_path = config_dir / 'hardware_info.json'
        with open(hardware_path, 'w') as f:
            json.dump({
                'system': self.hardware.system_info,
                'gpu': self.hardware.gpu_info,
                'simulators': self.hardware.simulators
            }, f, indent=2)
        
        print(f"\n📁 Configuration saved to: {config_path}")
        print(f"📊 Hardware info saved to: {hardware_path}")

def main():
    """Main entry point"""
    try:
        print("🚀 Starting MATHIR Configuration Wizard...")
        
        # Detect hardware
        detector = HardwareDetector()
        wizard = ConfigurationWizard(detector)
        
        # Run interactive setup
        config = wizard.run_interactive_setup()
        
        # Show next steps
        print("\n🎉 CONFIGURATION COMPLETE!")
        print("\nNext steps:")
        print("  1. Review the generated config/mathir_optimized.yaml")
        print("  2. Modify any settings as needed")
        print("  3. Use the config with:")
        print("     python train_agent.py --config config/mathir_optimized.yaml")
        print("  4. For deployment:")
        print("     python deploy_agent.py --config config/mathir_optimized.yaml")
        
        return 0
    
    except KeyboardInterrupt:
        print("\n\n❌ Configuration cancelled by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
```

## **install_dependencies.py**
```python
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
```

## **quick_setup.sh**
```bash
#!/bin/bash
# Quick setup script for MATHIR

echo "🚀 MATHIR Quick Setup"
echo "===================="

# Check Python version
python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
if [[ $(echo "$python_version < 3.8" | bc -l) -eq 1 ]]; then
    echo "❌ Python 3.8+ required (found $python_version)"
    exit 1
fi
echo "✅ Python $python_version detected"

# Create virtual environment (optional)
read -p "Create virtual environment? (y/n): " create_venv
if [[ $create_venv == "y" ]]; then
    python3 -m venv mathir_env
    source mathir_env/bin/activate
    echo "✅ Virtual environment activated"
fi

# Run hardware detection
echo "🔍 Detecting hardware..."
python3 configure_mathir.py

# Install dependencies
read -p "Install dependencies based on configuration? (y/n): " install_deps
if [[ $install_deps == "y" ]]; then
    python3 install_dependencies.py
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "1. Review config/mathir_optimized.yaml"
echo "2. For training: python train_agent.py"
echo "3. For deployment: python deploy_agent.py"
echo ""
echo "Need help? Check config/hardware_info.json for detected hardware."
```

## **README.md**
```markdown
# MATHIR Configuration Wizard

A hardware-aware configuration system that automatically detects your hardware and suggests optimized settings for training and deploying MATHIR-based agents.

## Features

- 🔍 **Automatic Hardware Detection**: Detects CPU, GPU, RAM, and installed simulators
- 🎯 **Optimized Presets**: Pre-configured settings for common hardware combinations
- 🎮 **Simulator Integration**: Auto-detects Isaac Sim, CARLA, Gazebo
- 💻 **Edge Device Support**: Optimized configurations for Jetson, Raspberry Pi
- ⚡ **Performance-Aware**: Adjusts batch sizes, model variants, and optimizations based on capabilities

## Quick Start

```bash
# Make setup script executable
chmod +x quick_setup.sh

# Run the setup wizard
./quick_setup.sh

# Or run manually
python3 configure_mathir.py
```

## Hardware Presets

The wizard includes presets for:

### Training/Development Systems:
- **High-end PC** (RTX 4090, 64GB RAM): Full model, large batch sizes
- **Mid-range PC** (RTX 3080, 32GB RAM): Standard model, moderate batch sizes
- **Entry-level PC** (RTX 3060, 16GB RAM): Tiny model, small batch sizes
- **CPU-only**: Optimized for CPU training

### Edge Deployment:
- **NVIDIA Jetson AGX Orin**: FP16 precision, TensorRT optimization
- **NVIDIA Jetson Orin NX**: FP16 precision, moderate optimization
- **NVIDIA Jetson Nano**: INT8 precision, aggressive optimization
- **Raspberry Pi 5**: INT8 precision, TFLite optimization
- **Raspberry Pi 4**: INT8 precision, minimal configuration

## Generated Configuration

The wizard creates `config/mathir_optimized.yaml` with:

```yaml
metadata:
  generated_by: MATHIR Configuration Wizard
  timestamp: ...
  use_case: training/deployment/both
  training_hardware: ...
  target_hardware: ...
  simulator: ...

model:
  variant: tiny/standard/xlarge
  perception:
    backbone: mobilenet_v3/efficientnet_b0/efficientnet_b2
    input_resolution: [256, 256]

training:
  batch_size: auto-calculated
  parallel_environments: auto-calculated
  simulator_config: simulator-specific

deployment:
  target: target hardware
  optimizations:
    precision: fp32/fp16/int8
    batch_size: 1
    optimization_level: 1-3

hardware_optimizations:
  cpu:
    num_workers: optimized
  gpu:
    memory_fraction: optimized
```

## Manual Override

After generation, you can manually edit `config/mathir_optimized.yaml`:

```yaml
# Example manual adjustments
model:
  variant: standard  # Change from auto-detected

training:
  batch_size: 32  # Override auto-calculated
  learning_rate: 0.0003  # Add custom parameters

deployment:
  optimizations:
    precision: fp16  # Force FP16 even if INT8 was suggested
```

## Integration with Training Scripts

Use the generated configuration:

```python
import yaml
from mathir_agent import MATHIRAgent

# Load optimized config
with open('config/mathir_optimized.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Create agent with optimized settings
agent = MATHIRAgent(config=config)

# Training
if config['metadata']['use_case'] in ['training', 'both']:
    agent.train()

# Deployment
if config['metadata']['use_case'] in ['deployment', 'both']:
    agent.optimize_for_edge()
    agent.deploy()
```

## Troubleshooting

### Hardware Not Detected
- Ensure `nvidia-smi` works for NVIDIA GPUs
- Install missing system packages: `sudo apt-get install python3-dev`

### Simulator Not Detected
- Set environment variables:
  ```bash
  export ISAAC_SIM_PATH=/path/to/isaac_sim
  export CARLA_ROOT=/path/to/carla
  ```

### Performance Issues
- Review `config/hardware_info.json` for detected capabilities
- Manually adjust batch sizes in configuration
- Consider using a smaller model variant

## Advanced Usage

### Multiple Configurations
```bash
# Generate config for different targets
python3 configure_mathir.py
# Saves as config/mathir_optimized.yaml

# Generate specific config
cp config/mathir_optimized.yaml config/mathir_jetson.yaml
# Edit for Jetson-specific optimizations
```

### CI/CD Integration
```yaml
# GitHub Actions example
- name: Configure MATHIR
  run: python3 configure_mathir.py --non-interactive --preset jetson_agx_orin
```

## Support

For issues or questions:
1. Check `config/hardware_info.json` for detection details
2. Review generated YAML configuration
3. Submit issue on GitHub with hardware information

## License

MIT License - See LICENSE file for details.
```

## **Usage Example**

```bash
# Make scripts executable
chmod +x configure_mathir.py
chmod +x quick_setup.sh

# Run the wizard
python3 configure_mathir.py

# Or use the quick setup
./quick_setup.sh

# Output example:
"""
🤖 DETECTED HARDWARE & SIMULATORS
============================================================
📊 SYSTEM INFORMATION:
  OS: Linux 5.15.0-91-generic
  Architecture: x86_64
  CPU: Intel(R) Core(TM) i9-13900K
  CPU Cores: 24 physical, 32 logical
  RAM: 64.0 GB total, 42.3 GB available

🎮 GPU INFORMATION:
  GPU Count: 1
  GPU 1: NVIDIA GeForce RTX 4090
  Total VRAM: 24.0 GB
  CUDA Version: 12.1

🔄 DETECTED SIMULATORS:
  Isaac_sim: ✅ Installed (v2023.1.1)
    Path: /home/user/.local/share/ov/pkg/isaac_sim-2023.1.1
  Carla: ❌ Not installed
  Gazebo: ❌ Not installed
  Unity: ❌ Not installed

⚙️ MATHIR CONFIGURATION WIZARD
============================================================
📊 STEP 1: DETECTING CURRENT HARDWARE...
(Shows detected hardware)

🎯 STEP 2: SELECT USE CASE
What is your primary use case?
  1. Training agent (using simulator)
  2. Deploying trained agent (inference only)
  3. Both training and deployment

Select option (1-3): 3

🔄 STEP 3: SELECT SIMULATOR (Detected: isaac_sim)
Available simulators:
  1. Isaac Sim
  2. Other/Manual configuration

Select simulator (1-2): 1

💻 STEP 4: SELECT TARGET HARDWARE
Training/Development hardware (for running simulators):
  Auto-detected: High-end PC (RTX 4090, 64GB RAM, i9)

Deployment/Target hardware (for inference):
Edge devices:
  1. NVIDIA Jetson AGX Orin (64GB)
  2. NVIDIA Jetson Orin NX (16GB)
  3. NVIDIA Jetson Nano (4GB)
  4. Raspberry Pi 5 (8GB)
  5. Raspberry Pi 4 (4GB)
  6. Custom/Other

Select target hardware (1-6): 1

⚡ STEP 5: GENERATING OPTIMIZED CONFIGURATION...

📋 GENERATED CONFIGURATION:
============================================================
metadata:
  generated_by: MATHIR Configuration Wizard
  timestamp: Mon Jan 12 15:30:45 UTC 2026
  use_case: both
  training_hardware: High-end PC (RTX 4090, 64GB RAM, i9)
  target_hardware: NVIDIA Jetson AGX Orin (64GB)
  simulator: isaac_sim

model:
  variant: standard
  perception:
    backbone: efficientnet_b2
    input_resolution: [256, 256]
    
(Continues with full configuration...)

💾 Save configuration to config/mathir_optimized.yaml? (y/n): y
✅ Configuration saved!
"""

# The configuration will be saved and ready to use
```

This system provides an intelligent, interactive way to configure MATHIR based on actual hardware capabilities, avoiding the common issues of using generic configurations that don't match the target hardware.