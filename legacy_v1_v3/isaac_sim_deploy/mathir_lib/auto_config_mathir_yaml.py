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
        if platform.system() == "Windows":
            isaac_paths = [
                Path(os.environ.get('LOCALAPPDATA', 'C:/')) / 'ov/pkg/isaac_sim*',
                Path('C:/Program Files/Omniverse/pkg/isaac_sim*')
            ]
        else:
            isaac_paths = [
                Path.home() / '.local/share/ov/pkg/isaac_sim*',
                Path('/isaac_sim'),
                Path('/opt/isaac_sim')
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
            try:
                result = subprocess.run(['docker', 'images', 'carlasim/carla', '--format', '{{.Tag}}'], 
                                    capture_output=True, text=True)
                if result.returncode == 0 and result.stdout.strip():
                    simulators['carla']['installed'] = True
                    simulators['carla']['version'] = result.stdout.strip().split('\n')[0]
            except (FileNotFoundError, Exception):
                pass
        
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
            },
            'torture_sim': {
                'perception': {
                    'camera_resolution': [84, 84], # Internal Sim native resolution
                    'lidar_channels': 0, # Not used
                },
                'training': {
                    'simulator': 'torture_sim',
                    'physics_dt': 0.1,
                    # Torture sim is lightweight -> higher batch size possible
                    'training_batch_size': 64 
                },
                'env_config': {
                    'difficulty': 'torture'
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
        
        if use_case in [1, 3]:
            print(f"\n🔄 STEP 3: SELECT SIMULATOR")
            print("  1. 🧠 Internal Torture Simulator (Python-only, Highly Optimized for Logic/Memory)")
            
            offset = 2
            for i, sim in enumerate(simulators_available):
                print(f"  {i + offset}. {sim.replace('_', ' ').title()} (Detected)")
            
            print(f"  {len(simulators_available) + offset}. Other/Manual configuration")
            
            sim_choice = int(input(f"\nSelect simulator (1-{len(simulators_available) + offset}): "))
            
            if sim_choice == 1:
                selected_simulator = 'torture_sim'
            elif sim_choice < len(simulators_available) + offset:
                selected_simulator = simulators_available[sim_choice - offset]
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
        
        print(f"  {len([p for p in self.hardware_presets.values() if p.get('is_edge', False)]) + 1}. Current Workstation (Use detected specs)")
        print(f"  {len([p for p in self.hardware_presets.values() if p.get('is_edge', False)]) + 2}. Custom/Other")
        
        target_choice = int(input(f"\nSelect target hardware (1-{len([p for p in self.hardware_presets.values() if p.get('is_edge', False)]) + 2}): "))
        
        edge_presets = [k for k, v in self.hardware_presets.items() if v.get('is_edge', False)]
        
        if target_choice <= len(edge_presets):
            target_hardware = edge_presets[target_choice - 1]
        elif target_choice == len(edge_presets) + 1:
            # Current Workstation
            target_hardware = training_preset
            target_hardware['description'] = f"Current PC ({self.hardware.system_info['os']})"
        else:
            print("\nEnter custom hardware specifications (Press Enter for Auto/Defaults):")
            
            p_prec = input("Target precision (fp32/fp16/int8) [default: fp16]: ") or "fp16"
            p_batch = input("Inference batch size [default: 1]: ") or "1"
            p_model = input("Model variant (tiny/standard/xlarge) [default: standard]: ") or "standard"
            p_ram = input("RAM in GB [default: 8]: ") or "8"

            target_hardware = {
                'description': input("Description [default: Custom Device]: ") or "Custom Device",
                'is_edge': True,
                'target_precision': p_prec,
                'inference_batch_size': int(p_batch),
                'model_variant': p_model,
                'ram_gb': float(p_ram)
            }
        
        # Step 5: Generate configuration
        print("\n⚡ STEP 5: GENERATING OPTIMIZED CONFIGURATION...")
        self._generate_configuration(use_case, selected_simulator, training_preset, target_hardware)
        
        # Step 6: Show and confirm
        print(f"\n📋 GENERATED CONFIGURATION:")
        print("="*60)
        print(yaml.dump(self.config, default_flow_style=False, sort_keys=False))
        print("="*60)
        
        save = input("\n💾 Write configuration to config/mathir_v5.yaml? (y/n) [Existing file will be backed up]: ")
        if save.lower() == 'y':
            self._save_configuration()
            print("✅ Configuration saved! (Backup created)")
        else:
            print("Configuration not saved.")
        
        return self.config
    
    def _auto_detect_training_preset(self) -> Dict[str, Any]:
        """Auto-detect the best training preset based on hardware"""
        gpu_memory = self.hardware.gpu_info['memory_gb']
        system_ram = self.hardware.system_info['ram_gb']
        
        print(f"  [Debug] Checking resources: VRAM={gpu_memory}GB, RAM={system_ram}GB")
        
        if gpu_memory >= 16 and system_ram >= 30: # Relaxed slightly
            return self.hardware_presets['high_end_pc']
        elif gpu_memory >= 8 and system_ram >= 15: # Relaxed 16 -> 15 (Matches your 15.62 GB)
            return self.hardware_presets['mid_range_pc']
        elif gpu_memory >= 6 and system_ram >= 8:
            return self.hardware_presets['entry_level_pc']
        else:
            return self.hardware_presets['cpu_only']
    
    def _generate_configuration(self, use_case: int, simulator: Optional[str], 
                               training_preset: Dict[str, Any], target_hardware: Dict[str, Any]):
        """Generate optimized YAML configuration"""
        
        # Define variant specifications
        VARIANT_SPECS = {
            'tiny': {
                'working_capacity': 32,
                'episodic_capacity': 500,
                'semantic_prototypes': 128,
                'immunological_capacity': 50,
                'feature_dim': 128,
                'hidden_dim': 128
            },
            'standard': {
                'working_capacity': 64,
                'episodic_capacity': 1000,
                'semantic_prototypes': 256,
                'immunological_capacity': 100,
                'feature_dim': 256,
                'hidden_dim': 256
            },
            'xlarge': {
                'working_capacity': 128,
                'episodic_capacity': 5000,
                'semantic_prototypes': 512,
                'immunological_capacity': 200,
                'feature_dim': 512,
                'hidden_dim': 512
            }
        }
        
        # Base configuration
        self.config = {
            'metadata': {
                'generated_by': 'MATHIR Configuration Wizard',
                'timestamp': __import__('datetime').datetime.now().isoformat(),
                'use_case': ['training', 'deployment', 'both'][use_case - 1],
                'training_hardware': training_preset['description'],
                'target_hardware': target_hardware['description'] if isinstance(target_hardware, str) else target_hardware.get('description', 'Custom'),
                'simulator': simulator
            }
        }
        
        # Get target preset and variant specs
        if isinstance(target_hardware, str):
            target_preset = self.hardware_presets[target_hardware]
        else:
            target_preset = target_hardware
        
        variant_name = target_preset.get('model_variant', 'standard')
        variant_specs = VARIANT_SPECS.get(variant_name, VARIANT_SPECS['standard'])
        
        self.config['model'] = {
            'variant': variant_name,
            'feature_dim': variant_specs['feature_dim'],
            'action_dim': 2,
            'state_dim': 23,
            'perception': {
                'backbone': training_preset.get('perception_backbone', 'efficientnet_b0'),
                'input_resolution': [256, 256] if not simulator else self.simulator_presets.get(simulator, {}).get('perception', {}).get('camera_resolution', [256, 256])
            }
        }
        
        # Explicit Memory Configuration
        self.config['memory'] = {
            'mathir_variant': variant_name,
            'working_capacity': variant_specs['working_capacity'],
            'episodic_capacity': variant_specs['episodic_capacity'],
            'semantic_prototypes': variant_specs['semantic_prototypes'],
            'immunological_capacity': variant_specs['immunological_capacity'],
            'router': {
                'kl_divergence_constraint': True,
                'kl_target': "uniform",
                'kl_coefficient': 0.01,
                'kl_margin': 0.05
            },
            'retention_decay': [0.95, 0.8, 0.6]
        }
        
        # MHC Configuration
        self.config['mhc'] = {
            'rank_ratio': 0.3,
            'use_cache': True,
            'projection': {
                'max_iter': 10,
                'tol': 1e-4,
                'epsilon': 1e-8,
                'overrelaxation': {
                    'enabled': True,
                    'initial_omega': 1.4,
                    'adaptative': True
                }
            }
        }
        
        # Training configuration (if needed)
        if use_case in [1, 3]:
            self.config['training'] = {
                'simulator': simulator,
                'hardware_optimized': True,
                'batch_size': training_preset.get('training_batch_size', 16),
                'learning_rate': 1e-4,
                'optimizer': "adamw",
                'weight_decay': 1e-5,
                'gradient_clip': 0.5,
                'dropout': 0.1,
                'num_episodes': 10000,
                'parallel_environments': training_preset.get('num_envs', 8),
                'use_mixed_precision': self.hardware.gpu_info['cuda_available'],
                'gradient_accumulation_steps': 2 if training_preset.get('training_batch_size', 16) > 16 else 1,
                'kl_annealing': {
                    'enabled': True,
                    'start_weight': 0.001,
                    'end_weight': 0.01,
                    'anneal_episodes': 2000
                }
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
        """Save configuration to YAML file with backup"""
        config_dir = Path('config')
        config_dir.mkdir(exist_ok=True)
        
        target_path = config_dir / 'mathir_v5.yaml'
        
        # Backup existing file
        if target_path.exists():
            timestamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = config_dir / f"mathir_v5.yaml.{timestamp}.bak"
            try:
                import shutil
                shutil.copy2(target_path, backup_path)
                print(f"📦 Existing config backed up to: {backup_path.name}")
            except Exception as e:
                print(f"⚠️ Warning: Could not create backup: {e}")

        # Write new config
        with open(target_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
        
        # Also save hardware info for reference (kept separate)
        hardware_path = config_dir / 'hardware_info.json'
        with open(hardware_path, 'w') as f:
            json.dump({
                'system': self.hardware.system_info,
                'gpu': self.hardware.gpu_info,
                'simulators': self.hardware.simulators
            }, f, indent=2)
        
        print(f"\n📁 Configuration written to: {target_path}")
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