"""
Intelligent Benchmark Analysis Module with Ollama
Uses LLaMA 3.1:8b (8GB VRAM) or LLaMA 3.2:3b (lightweight model)
"""

import json
import subprocess
import os
from typing import Dict, Optional, List
import torch


class OllamaAnalyzer:
    """Intelligent Benchmark Analyzer with Ollama"""
    
    def __init__(self, model_name: str = "auto"):
        """
        Initialize Ollama analyzer
        
        Args:
            model_name: "llama3.1:8b" (8GB VRAM), "llama3.2:3b" (lightweight), or "auto"
        """
        self.model_name = self._select_model(model_name)
        self.ollama_available = self._check_ollama()
        
        if self.ollama_available:
            print(f"✓ Ollama detected with model: {self.model_name}")
        else:
            print("⚠️ Ollama not available")
    
    def _select_model(self, model_name: str) -> str:
        """Automatically selects model based on available VRAM"""
        
        if model_name != "auto":
            return model_name
        
        # Detect available VRAM
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            if vram_gb >= 8:
                print(f"🎮 GPU: {vram_gb:.1f}GB VRAM detected")
                print("→ Using llama3.1:8b (full model)")
                return "llama3.1:8b"
            else:
                print(f"🎮 GPU: {vram_gb:.1f}GB VRAM detected")
                print("→ Using llama3.2:3b (lightweight model)")
                return "llama3.2:3b"
        else:
            print("💻 No GPU detected")
            print("→ Using llama3.2:3b (lightweight model, CPU)")
            return "llama3.2:3b"
    
    def _check_ollama(self) -> bool:
        """Checks if Ollama is installed and accessible"""
        try:
            result = subprocess.run(
                ['ollama', 'list'],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def _call_ollama(self, prompt: str) -> Optional[str]:
        """Calls Ollama to generate an analysis"""
        
        if not self.ollama_available:
            return None
        
        try:
            # Ollama command with forced UTF-8 encoding
            cmd = ['ollama', 'run', self.model_name, prompt]
            
            # Fix Windows encoding issue
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',  # Force UTF-8
                errors='ignore',   # Ignore non-UTF8 chars
                timeout=90  # 90 seconds max (increased)
            )
            
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                error_msg = result.stderr if result.stderr else "Unknown error"
                print(f"❌ Ollama Error: {error_msg}")
                return None
                
        except subprocess.TimeoutExpired:
            print("⏱️ Ollama Timeout (>60s)")
            return None
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    
    def analyze_retention_results(self, results: Dict) -> str:
        """Analyzes retention results"""
        
        steps = results['retention']['steps']
        mathir_scores = results['retention']['mathir']
        lstm_scores = results['retention']['lstm']
        
        # Build prompt
        prompt = f"""Analyze these memory retention test results for two AI models:

LSTM (traditional model, baseline):
{chr(10).join([f'- {steps[i]} steps: {lstm_scores[i]:.2%}' for i in range(len(steps))])}

MATHIR V5 ("Anti-Fragile" Next Generation Architecture):
- Features: MHC Overrelaxed Sinkhorn, KL-constrained Router, Immunological Memory.
{chr(10).join([f'- {steps[i]} steps: {mathir_scores[i]:.2%}' for i in range(len(steps))])}

Provide a concise analysis in 3-4 sentences maximum:
1. How does MATHIR V5's plasticity compare to LSTM's rigidity?
2. Does the hierarchical architecture (V5) show superiority over the long term?
3. Quick technical verdict.

Respond in a professional and technical manner."""
        
        print("\n🧠 Ollama analysis of retention results (V5 Context)...")
        return self._call_ollama(prompt)
    
    def analyze_generalization_results(self, results: Dict) -> str:
        """Analyzes generalization results"""
        
        scenarios = results['generalization']['scenarios']
        mathir_scores = results['generalization']['mathir']
        lstm_scores = results['generalization']['lstm']
        
        # Build prompt
        data_str = "\n".join([
            f"- {scenarios[i]}: LSTM={lstm_scores[i]:.1%}, MATHIR V5={mathir_scores[i]:.1%}"
            for i in range(len(scenarios))
        ])
        
        prompt = f"""Analyze these generalization results for MATHIR V5 vs LSTM:

{data_str}

V5 Context: Uses a "Semantic Memory" based on prototypes to generalize.

Provide a concise analysis in 3-4 sentences maximum:
1. Does V5 Semantic Memory improve zero-shot transfer?
2. Comparison of robustness against unforeseen events.
3. Conclusion on adaptability.

Respond in a professional and technical manner."""
        
        print("\n🧠 Ollama analysis of generalization results (V5 Context)...")
        return self._call_ollama(prompt)
    
    def analyze_performance_results(self, results: Dict) -> str:
        """Analyzes performance results"""
        
        mathir_time = results['performance']['inference_time']['mathir']['mean']
        lstm_time = results['performance']['inference_time']['lstm']['mean']
        
        mathir_mem = results['performance']['memory']['mathir']['memory_gb'][-1]
        lstm_mem = results['performance']['memory']['lstm']['memory_gb'][-1]
        
        prompt = f"""Analyze these performance results (MATHIR V5 vs LSTM) on Edge Device (Jetson/RTX):

LSTM:
- Inference time: {lstm_time:.2f} ms
- VRAM (batch 32): {lstm_mem:.2f} GB

MATHIR V5 ("Production Ready" Optimized):
- Inference time: {mathir_time:.2f} ms
- VRAM (batch 32): {mathir_mem:.2f} GB
- Technologies: Sinkhorn Overrelaxed (10x speed gain), Intelligent Cache.

Provide a concise analysis in 3-4 sentences maximum:
1. Did the "Overrelaxed Sinkhorn" optimization solve V4 latency issues?
2. Is the memory cost justified by the cognitive capabilities?
3. Is it deployable on Jetson Orin (Target < 30ms)?

Respond in a professional and technical manner."""
        
        print("\n🧠 Ollama analysis of performance results (V5 Context)...")
        return self._call_ollama(prompt)
    
    def generate_global_summary(self, results: Dict, previous_results: Optional[Dict] = None) -> str:
        """Generates a global summary comparing with previous results"""
        
        # Calculate key metrics
        mathir_ret_1000 = results['retention']['mathir'][2]  # @ 1000 steps
        lstm_ret_1000 = results['retention']['lstm'][2]
        
        mathir_gen_avg = sum(results['generalization']['mathir']) / len(results['generalization']['mathir'])
        lstm_gen_avg = sum(results['generalization']['lstm']) / len(results['generalization']['lstm'])
        
        # Detect if new results or reuse
        status = "NEW RESULTS" if previous_results is None else "REUSED RESULTS"
        
        improvement_text = ""
        if previous_results:
            prev_mathir_ret = previous_results.get('retention', {}).get('mathir', [0, 0, 0])[2]
            prev_lstm_ret = previous_results.get('retention', {}).get('lstm', [0, 0, 0])[2]
            
            if prev_mathir_ret != mathir_ret_1000:
                improvement_text = f"\nChange vs V4: MATHIR {prev_mathir_ret:.2%} → {mathir_ret_1000:.2%}"
        
        prompt = f"""Generate a FINAL DEPLOYMENT REPORT for MATHIR V5 (Production Ready):
        
        CONTEXT: Final "V5 Enterprise" Certification vs LSTM Legacy.
        STATUS: {status}
        {improvement_text}
        
        BATTLE DATA:
        - 🧠 Long-Term Retention (@1000 steps): LSTM={lstm_ret_1000:.2%} vs MATHIR V5={mathir_ret_1000:.2%}
        - 🌐 Generalization (Unseen Scenarios): LSTM={lstm_gen_avg:.2%} vs MATHIR V5={mathir_gen_avg:.2%}
        - 💻 V5 Technologies: MHC Overrelaxed, KL Router, Digital Immunology.
        
        Your mission is to validate the passage to PRODUCTION.
        Structure your response like this:
        
        1. 🚀 V5 CERTIFICATION
           - Declare if the model is ready for the road (Production Ready).
           - Mention stability gains (KL-Router).
           
        2. ⚡ PERFORMANCE ENGINE
           - Confirm the efficiency of Overrelaxed Sinkhorn.
           
        3. 🔮 AUTONOMOUS FLEET IMPACT
           - How this V5 changes fleet management (Sim-to-Real, Edge Deployment).
           
        4. ✅ DEPLOYMENT VERDICT
           - Clear and definitive GO / NO-GO.
           
        Be expert, technical, and business/production impact oriented."""
        
        print("\n🧠 Generating global Ollama summary (V5 Enterprise)...")
        return self._call_ollama(prompt)
    
    def analyze_all(self, results: Dict, previous_results: Optional[Dict] = None) -> Dict[str, str]:
        """Generates all analyses"""
        
        if not self.ollama_available:
            return {
                'retention': "⚠️ Ollama not available - Install Ollama for AI analysis",
                'generalization': "⚠️ Ollama not available - Install Ollama for AI analysis",
                'performance': "⚠️ Ollama not available - Install Ollama for AI analysis",
                'global_summary': "⚠️ Ollama not available - Install Ollama for AI analysis"
            }
        
        analyses = {}
        
        # Retention analysis
        retention_analysis = self.analyze_retention_results(results)
        analyses['retention'] = retention_analysis or "❌ Analysis failed"
        
        # Generalization analysis
        gen_analysis = self.analyze_generalization_results(results)
        analyses['generalization'] = gen_analysis or "❌ Analysis failed"
        
        # Performance analysis
        perf_analysis = self.analyze_performance_results(results)
        analyses['performance'] = perf_analysis or "❌ Analysis failed"
        
        # Global summary
        global_summary = self.generate_global_summary(results, previous_results)
        analyses['global_summary'] = global_summary or "❌ Analysis failed"
        
        return analyses


def check_ollama_status() -> Dict:
    """Checks the status of Ollama and models"""
    
    status = {
        'installed': False,
        'models_available': [],
        'recommended_model': None,
        'vram_available': 0.0,
        'can_use_8b': False
    }
    
    # Check Ollama installation
    try:
        result = subprocess.run(
            ['ollama', 'list'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            status['installed'] = True
            
            # Parse available models
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            for line in lines:
                if line.strip():
                    model_name = line.split()[0]
                    status['models_available'].append(model_name)
        
    except (subprocess.TimeoutExpired, FileNotFoundError):
        status['installed'] = False
    
    # Detect VRAM
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        status['vram_available'] = vram_gb
        status['can_use_8b'] = vram_gb >= 8
    
    # Recommend model
    if status['can_use_8b']:
        status['recommended_model'] = 'llama3.1:8b'
    else:
        status['recommended_model'] = 'llama3.2:3b'
    
    return status


def print_ollama_setup_guide():
    """Displays the Ollama installation guide"""
    
    status = check_ollama_status()
    
    print("\n" + "="*60)
    print("  OLLAMA INSTALLATION GUIDE")
    print("="*60)
    
    if status['installed']:
        print("\n✅ Ollama is installed!")
        print(f"\nAvailable models: {', '.join(status['models_available']) or 'None'}")
    else:
        print("\n❌ Ollama is not installed")
        print("\nInstallation steps:")
        print("1. Download Ollama: https://ollama.ai/download")
        print("2. Install Ollama")
        print("3. Restart your terminal")
    
    print(f"\n🎮 Available VRAM: {status['vram_available']:.1f} GB")
    
    if status['can_use_8b']:
        print("✅ You can use llama3.1:8b (full model)")
        print("\nCommand to download:")
        print("   ollama pull llama3.1:8b")
    else:
        print("⚠️ Insufficient VRAM for llama3.1:8b (requires 8GB)")
        print("✅ Use llama3.2:3b (lightweight model)")
        print("\nCommand to download:")
        print("   ollama pull llama3.2:3b")
    
    print(f"\n📌 Recommended model: {status['recommended_model']}")
    
    print("\n💡 To test Ollama:")
    print(f"   ollama run {status['recommended_model']} 'Hello!'")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    # Module test
    print("=== Ollama Module Test ===\n")
    
    # Display guide
    print_ollama_setup_guide()
    
    # Analyzer test
    analyzer = OllamaAnalyzer()
    
    if analyzer.ollama_available:
        print("\n✓ Ollama Analyzer ready!")
        print(f"  Model: {analyzer.model_name}")
    else:
        print("\n⚠️ Ollama not available")
        print("  Install Ollama to activate AI analyses")
