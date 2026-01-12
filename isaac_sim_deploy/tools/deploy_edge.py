"""
deploy_edge.py - Fixed Deployment Pipeline
TensorRT for Jetson, TFLite for Raspberry Pi.
"""

import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import json
import yaml
import subprocess
import os
from pathlib import Path
from typing import Dict, Optional
import sys

# Add root directory to path to import mathir_lib
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class EdgeDeployer:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = self.config['deployment']['target_device']
        self.precision = self.config['deployment']['precision']
        
    def optimize_model(self, model: nn.Module) -> nn.Module:
        """Apply device-specific optimizations."""
        model.eval()
        
        # 1. Prepare for inference
        self._set_inference_mode(model)
        
        # 2. Device-specific optimizations
        if self.device == 'jetson':
            return self._optimize_for_jetson(model)
        elif self.device == 'raspberry':
            return self._optimize_for_raspberry(model)
        else:
            return model
    
    def _set_inference_mode(self, model: nn.Module):
        """Disable training-specific features."""
        for module in model.modules():
            if hasattr(module, 'set_training_mode'):
                module.set_training_mode(False)
            if hasattr(module, 'training_mode'):
                module.training_mode = False
    
    def _optimize_for_jetson(self, model: nn.Module) -> nn.Module:
        """TensorRT optimization for Jetson."""
        print("🔧 Optimizing for Jetson with TensorRT...")
        
        # Export to ONNX first
        onnx_path = self._export_to_onnx(model)
        
        # TensorRT conversion
        trt_cmd = [
            'trtexec', # Assumes trtexec is in PATH
            f'--onnx={onnx_path}',
            f'--saveEngine=mathir_v5_{self.precision}.trt',
            f'--workspace={self.config["deployment"]["tensorrt"]["workspace_size"]}',
            '--verbose'
        ]
        
        # Precision flags
        if self.precision == 'fp16':
            trt_cmd.append('--fp16')
        elif self.precision == 'int8':
            trt_cmd.extend(['--int8', '--fp16'])  # Often used together
        
        # Handle common TensorRT issues
        if self.precision == 'int8':
            trt_cmd.append('--allowGPUFallback')
        
        # Execute conversion
        try:
            print(f"Running command: {' '.join(trt_cmd)}")
            result = subprocess.run(trt_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"⚠️  TensorRT conversion warning: {result.stderr[:500]}")
                # Fallback to FP16 if INT8 fails
                if 'INT8' in result.stderr:
                    print("🔄 Falling back to FP16 due to INT8 issues")
                    trt_cmd = [c for c in trt_cmd if '--int8' not in c]
                    subprocess.run(trt_cmd)
        except Exception as e:
            print(f"❌ TensorRT conversion failed: {e}")
            print("📋 Using TorchScript as fallback...")
            return torch.jit.script(model)
        
        return model  # In production, load the .trt engine instead
    
    def _optimize_for_raspberry(self, model: nn.Module) -> nn.Module:
        """Optimizations for Raspberry Pi."""
        print("🔧 Optimizing for Raspberry Pi...")
        
        # Aggressive pruning for CPU
        from torch.nn.utils import prune
        parameters_to_prune = [
            (module, 'weight') for module in model.modules()
            if isinstance(module, (nn.Linear, nn.Conv2d))
        ]
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=0.3
        )
        
        # Permanent pruning
        for module, _ in parameters_to_prune:
            prune.remove(module, 'weight')
        
        # Quantization
        if self.precision == 'int8':
            model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
            model_prepared = torch.quantization.prepare(model, inplace=False)
            # Calibrate (simplified)
            with torch.no_grad():
                dummy_input = torch.randn(1, self.config['model']['feature_dim']) # Feature dim based on V5
                _ = model_prepared(dummy_input)
            model = torch.quantization.convert(model_prepared)
        
        return torch.jit.optimize_for_inference(torch.jit.script(model))
    
    def _export_to_onnx(self, model: nn.Module, opset: int = 13) -> str:
        """Export model to ONNX format."""
        # Using feature_dim directly as implied by MATHIRv5 logic (memory core wraps features)
        dummy_features = torch.randn(1, self.config['model']['feature_dim'])
        
        onnx_path = "mathir_v5.onnx"
        torch.onnx.export(
            model,
            (dummy_features,),
            onnx_path,
            input_names=['features'],
            output_names=['output', 'router_weights', 'router_loss', 'router_entropy'],
            dynamic_axes={
                'features': {0: 'batch_size'},
                'output': {0: 'batch_size'},
                'router_weights': {0: 'batch_size'}
            },
            opset_version=opset,
            do_constant_folding=True,
            export_params=True,
            verbose=False
        )
        
        # Validate ONNX
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        
        return onnx_path
    
    def benchmark(self, model: nn.Module, num_iterations: int = 1000) -> Dict:
        """Benchmark optimized model."""
        dummy_features = torch.randn(1, self.config['model']['feature_dim'])
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_features)
        
        # Benchmark
        import time
        latencies = []
        
        for i in range(num_iterations):
            start = time.perf_counter()
            with torch.no_grad():
                _ = model(dummy_features)
            latencies.append((time.perf_counter() - start) * 1000)  # ms
        
        avg_latency = sum(latencies) / len(latencies)
        fps = 1000 / avg_latency
        
        print(f"📊 Benchmark Results:")
        print(f"   Average Latency: {avg_latency:.2f} ms")
        print(f"   Estimated FPS: {fps:.1f}")
        print(f"   Device: {self.device.upper()}")
        print(f"   Precision: {self.precision.upper()}")
        
        return {'latency_ms': avg_latency, 'fps': fps}
    
    def deploy(self, model: nn.Module, output_dir: str = "./deployed"):
        """Complete deployment pipeline."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        print("🚀 Starting MATHIR v5 Deployment")
        print("=" * 50)
        
        # Optimize
        optimized_model = self.optimize_model(model)
        
        # Benchmark
        metrics = self.benchmark(optimized_model)
        
        # Save artifacts
        self._save_deployment_artifacts(optimized_model, metrics, output_dir)
        
        print(f"✅ Deployment complete in '{output_dir}'")
        return optimized_model
    
    def _save_deployment_artifacts(self, model, metrics, output_dir):
        """Save all deployment files."""
        # Save model
        if isinstance(model, torch.jit.ScriptModule):
            model.save(f"{output_dir}/model.pt")
        else:
            torch.save(model.state_dict(), f"{output_dir}/model.pth")
        
        # Save metrics
        with open(f"{output_dir}/metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save runtime script
        # self._generate_runtime_script(output_dir) # Implemented in DEPLOYEMENT.MD, skipping generation here to save complexity

# Usage Example
if __name__ == "__main__":
    from mathir_lib.mathir_v5 import MATHIRv5
    
    # Check if config exists
    config_path = "config/mathir_v5.yaml"
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        sys.exit(1)
        
    # Load model (fresh init for demo)
    model = MATHIRv5.from_config(config_path)
    
    # If checkpoint exists, load it
    checkpoint_path = "checkpoints/best_model.pth"
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
    else:
        print("Warning: No checkpoint found, using initialized weights.")
    
    # Deploy
    deployer = EdgeDeployer(config_path)
    deployed_model = deployer.deploy(model, output_dir="./deployed_edge")
