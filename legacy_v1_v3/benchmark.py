"""
MATHIR Benchmark Suite
======================

Provides benchmark classes for evaluating MATHIR vs LSTM:
- RetentionBenchmark: Tests memory retention at different time steps
- GeneralizationBenchmark: Tests across driving scenarios
- CompleteBenchmarkSuite: Combines both benchmarks into a full report

Usage:
    from benchmark import CompleteBenchmarkSuite
    suite = CompleteBenchmarkSuite(device='cuda')
    results = suite.run_all_benchmarks()
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
from mathir_model import MATHIRAgent, LSTMBaseline, count_parameters


def _make_obs(batch_size=1, device='cpu'):
    """Create synthetic observations compatible with both MATHIR and LSTM."""
    return {
        'camera': torch.randn(batch_size, 1, 84, 84, device=device),
        'state': torch.randn(batch_size, 5, device=device),
    }


def _cosine_similarity(a, b):
    """Compute mean cosine similarity between two feature tensors."""
    a_flat = a.flatten(1)
    b_flat = b.flatten(1)
    return F.cosine_similarity(a_flat, b_flat, dim=1).mean().item()


class RetentionBenchmark:
    """
    Tests memory retention at different step counts.
    
    Measures how well a model retains information about early observations
    after processing many subsequent steps, using cosine similarity.
    """

    def test_retention(self, model, num_steps=500):
        """
        Run a retention test on the given model.
        
        Args:
            model: A MATHIR or LSTM model instance.
            num_steps: Number of sequential steps to process.
        
        Returns:
            float: Retention score in [0, 1]. Higher = better retention.
        """
        device = next(model.parameters()).device
        model.eval()
        model.reset_memory()

        with torch.no_grad():
            # Step 1: Encode the reference observation
            ref_obs = _make_obs(1, device)
            ref_out = model(ref_obs, reset_hidden=True) if isinstance(model, LSTMBaseline) else model(ref_obs, step=0)
            ref_features = ref_out['features'].detach().clone()

            # Steps 2..num_steps: process noise observations
            for step in range(1, num_steps):
                noise_obs = _make_obs(1, device)
                if isinstance(model, LSTMBaseline):
                    model(noise_obs, reset_hidden=False)
                else:
                    model(noise_obs, step=step)

            # Final step: encode same reference, measure retention
            final_out = model(ref_obs, reset_hidden=False) if isinstance(model, LSTMBaseline) else model(ref_obs, step=num_steps)
            final_features = final_out['features']

            score = _cosine_similarity(ref_features, final_features)

        return score


class GeneralizationBenchmark:
    """
    Tests generalization across different driving scenarios.
    
    Each scenario applies different noise patterns to synthetic observations,
    simulating varied driving conditions (highway, city, etc.).
    """

    # Scenario configs: name -> noise_std for camera observations
    SCENARIOS = {
        'highway':      0.05,   # Clean, consistent visuals
        'city':         0.15,   # Moderate noise (traffic, pedestrians)
        'country':      0.20,   # Unstructured terrain
        'tunnel':       0.25,   # Lighting changes, occlusion
        'intersection': 0.30,   # High complexity, occlusion
    }

    def test_scenario(self, model, scenario_name):
        """
        Evaluate model on a specific scenario.
        
        Args:
            model: A MATHIR or LSTM model instance.
            scenario_name: One of 'highway', 'city', 'country', 'tunnel', 'intersection'.
        
        Returns:
            float: Scenario score in [0, 1]. Higher = better generalization.
        """
        noise_std = self.SCENARIOS.get(scenario_name, 0.15)
        device = next(model.parameters()).device
        model.eval()
        model.reset_memory()

        num_episodes = 10
        action_consistencies = []

        with torch.no_grad():
            for ep in range(num_episodes):
                # Create a "consistent" base observation for this episode
                base_obs = _make_obs(1, device)

                if isinstance(model, LSTMBaseline):
                    out_base = model(base_obs, reset_hidden=True)
                else:
                    out_base = model(base_obs, step=0)
                
                base_action = out_base['action_mean'].clone()

                # Process several noisy variants
                consistency_scores = []
                for step in range(20):
                    noisy_obs = {
                        'camera': base_obs['camera'] + torch.randn_like(base_obs['camera']) * noise_std,
                        'state': base_obs['state'] + torch.randn_like(base_obs['state']) * noise_std * 0.5,
                    }
                    if isinstance(model, LSTMBaseline):
                        out_noisy = model(noisy_obs, reset_hidden=False)
                    else:
                        out_noisy = model(noisy_obs, step=step)

                    # Measure how consistent the action is (inverse of action difference)
                    action_diff = F.mse_loss(out_noisy['action_mean'], base_action).item()
                    # Convert to a 0-1 score (small diff = high score)
                    consistency = max(0.0, 1.0 - action_diff)
                    consistency_scores.append(consistency)

                action_consistencies.append(np.mean(consistency_scores))

        return float(np.mean(action_consistencies))


class CompleteBenchmarkSuite:
    """
    Full benchmark suite combining Retention, Generalization, and Performance tests.
    
    Produces a results dict compatible with the Streamlit dashboard (app_streamlit.py).
    """

    def __init__(self, device='cpu'):
        self.device = device

    def run_all_benchmarks(self):
        """
        Run all benchmarks and return a comprehensive results dict.
        
        Returns:
            dict with keys: model_info, retention, generalization, performance, ollama_analyses
        """
        # Initialize models
        mathir = MATHIRAgent().to(self.device)
        lstm = LSTMBaseline().to(self.device)
        mathir.eval()
        lstm.eval()

        results = {}

        # --- Model Info ---
        results['model_info'] = {
            'mathir_params': count_parameters(mathir),
            'lstm_params': count_parameters(lstm),
        }

        # --- Retention Benchmark ---
        results['retention'] = self._run_retention(mathir, lstm)

        # --- Generalization Benchmark ---
        results['generalization'] = self._run_generalization(mathir, lstm)

        # --- Performance Benchmark ---
        results['performance'] = self._run_performance(mathir, lstm)

        # --- Ollama Analyses (optional, not auto-generated) ---
        results['ollama_analyses'] = {}

        return results

    def _run_retention(self, mathir, lstm):
        """Run retention benchmark at standard step counts."""
        step_counts = [100, 500, 1000, 2000, 5000]
        retention_bench = RetentionBenchmark()

        mathir_scores = []
        lstm_scores = []

        for steps in step_counts:
            m_score = retention_bench.test_retention(mathir, num_steps=steps)
            l_score = retention_bench.test_retention(lstm, num_steps=steps)
            mathir_scores.append(m_score)
            lstm_scores.append(l_score)

        return {
            'steps': step_counts,
            'mathir': mathir_scores,
            'lstm': lstm_scores,
        }

    def _run_generalization(self, mathir, lstm):
        """Run generalization benchmark across all scenarios."""
        gen_bench = GeneralizationBenchmark()
        scenarios = list(GeneralizationBenchmark.SCENARIOS.keys())

        mathir_scores = []
        lstm_scores = []

        for scenario in scenarios:
            m_score = gen_bench.test_scenario(mathir, scenario)
            l_score = gen_bench.test_scenario(lstm, scenario)
            mathir_scores.append(m_score)
            lstm_scores.append(l_score)

        return {
            'scenarios': scenarios,
            'mathir': mathir_scores,
            'lstm': lstm_scores,
        }

    def _run_performance(self, mathir, lstm):
        """Run memory usage and inference time benchmarks."""
        batch_sizes = [1, 4, 8, 16, 32]

        # Memory usage estimation
        mathir_mem = []
        lstm_mem = []

        for bs in batch_sizes:
            m_mem = self._estimate_memory(mathir, bs)
            l_mem = self._estimate_memory(lstm, bs)
            mathir_mem.append(m_mem)
            lstm_mem.append(l_mem)

        # Inference time (average over 50 runs, batch_size=1)
        m_time = self._measure_inference_time(mathir, num_runs=50)
        l_time = self._measure_inference_time(lstm, num_runs=50)

        return {
            'memory': {
                'mathir': {'batch_sizes': batch_sizes, 'memory_gb': mathir_mem},
                'lstm': {'batch_sizes': batch_sizes, 'memory_gb': lstm_mem},
            },
            'inference_time': {
                'mathir': {'mean': m_time},
                'lstm': {'mean': l_time},
            },
        }

    def _estimate_memory(self, model, batch_size):
        """Estimate VRAM usage in GB for a given batch size."""
        obs = _make_obs(batch_size, self.device)
        model.eval()

        # Reset for clean measurement
        model.reset_memory()

        torch.cuda.empty_cache() if self.device == 'cuda' else None

        if self.device == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
            with torch.no_grad():
                if isinstance(model, LSTMBaseline):
                    model(obs, reset_hidden=True)
                else:
                    model(obs, step=0)
            peak_bytes = torch.cuda.max_memory_allocated(self.device)
            return peak_bytes / (1024 ** 3)
        else:
            # CPU fallback: estimate from parameters + activations
            param_bytes = sum(p.numel() * 4 for p in model.parameters())
            # Rough activation estimate
            activation_bytes = batch_size * 272 * 4  # combined_dim * float32
            total_bytes = param_bytes + activation_bytes
            return total_bytes / (1024 ** 3)

    def _measure_inference_time(self, model, num_runs=50):
        """Measure average inference time in milliseconds."""
        obs = _make_obs(1, self.device)
        model.eval()
        model.reset_memory()

        # Warmup
        with torch.no_grad():
            for _ in range(5):
                if isinstance(model, LSTMBaseline):
                    model(obs, reset_hidden=False)
                else:
                    model(obs, step=0)

        # Timed runs
        times = []
        with torch.no_grad():
            for i in range(num_runs):
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                start = time.perf_counter()

                if isinstance(model, LSTMBaseline):
                    model(obs, reset_hidden=False)
                else:
                    model(obs, step=i)

                if self.device == 'cuda':
                    torch.cuda.synchronize()
                elapsed = (time.perf_counter() - start) * 1000  # ms
                times.append(elapsed)

        return float(np.mean(times))


if __name__ == '__main__':
    print("Running CompleteBenchmarkSuite...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    suite = CompleteBenchmarkSuite(device=device)
    results = suite.run_all_benchmarks()

    print(f"\nModel Info:")
    print(f"  MATHIR params: {results['model_info']['mathir_params']:,}")
    print(f"  LSTM params:   {results['model_info']['lstm_params']:,}")

    print(f"\nRetention:")
    for i, s in enumerate(results['retention']['steps']):
        m = results['retention']['mathir'][i]
        l = results['retention']['lstm'][i]
        print(f"  {s:>5} steps — MATHIR: {m:.4f}, LSTM: {l:.4f}")

    print(f"\nGeneralization:")
    for i, sc in enumerate(results['generalization']['scenarios']):
        m = results['generalization']['mathir'][i]
        l = results['generalization']['lstm'][i]
        print(f"  {sc:>12} — MATHIR: {m:.4f}, LSTM: {l:.4f}")

    print(f"\nDone!")
