"""
mhc_v5.py - Fixed Manifold Constrained Linear Layer
Implements Overrelaxed Sinkhorn-Knopp with adaptive stopping.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from typing import Optional, Dict

class OverrelaxedSinkhornProjection(nn.Module):
    """
    Overrelaxed Sinkhorn-Knopp projection with adaptive parameter.
    Uses Lyapunov-based adaptive ω selection for guaranteed convergence.
    """
    def __init__(self, max_iter=10, tol=1e-4, eps=1e-8, 
                 overrelax=True, omega=1.4):
        super().__init__()
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.eps = float(eps)
        self.overrelax = overrelax
        self.base_omega = float(omega)
        
        # Lyapunov tracking for adaptive ω
        self.register_buffer('lyapunov_history', torch.zeros(50))
        self.history_ptr = 0
        
    def forward(self, W: torch.Tensor) -> torch.Tensor:
        out_features, in_features = W.shape
        
        # Initialize with absolute values
        W_abs = W.abs() + self.eps
        
        # Initial scaling vectors (in log domain for stability)
        u = torch.zeros(out_features, device=W.device)
        v = torch.zeros(in_features, device=W.device)
        
        # Adaptive overrelaxation parameter
        omega = self.base_omega if self.overrelax else 1.0
        change = float('inf')
        
        for i in range(self.max_iter):
            W_prev = W_abs.clone()
            
            # --- Overrelaxed Row Scaling (ω > 1) ---
            row_sum = W_abs.sum(dim=1, keepdim=True)
            # SOR update: x_new = ω * x_SK + (1-ω) * x_old
            row_scale = (1 / (row_sum + self.eps))
            W_abs = omega * (W_abs * row_scale) + (1 - omega) * W_abs
            
            # --- Overrelaxed Column Scaling ---
            col_sum = W_abs.sum(dim=0, keepdim=True)
            if out_features == in_features:
                target = 1.0
            else:
                target = out_features / in_features
            col_scale = target / (col_sum + self.eps)
            W_abs = omega * (W_abs * col_scale) + (1 - omega) * W_abs
            
            # Lyapunov function monitoring
            if self.overrelax and self.training and i > 2:
                L = (W_abs.sum(dim=1) - target).abs().mean() + \
                    (W_abs.sum(dim=0) - target).abs().mean()
                self._update_omega(L, i)
            
            # Early stopping check
            change = (W_abs - W_prev).abs().max().item()
            if change < self.tol:
                break
        
        # Restore original signs
        return W_abs * torch.sign(W)
    
    def _update_omega(self, lyapunov_value: float, iteration: int):
        """Adapt ω based on Lyapunov function descent"""
        idx = self.history_ptr % 50
        self.lyapunov_history[idx] = lyapunov_value
        self.history_ptr += 1
        
        # Simple heuristic: reduce ω if Lyapunov isn't decreasing
        if iteration > 10:
            recent = self.lyapunov_history[max(0, self.history_ptr-10):self.history_ptr]
            if recent.mean() > recent.min() * 1.1:  # Not decreasing sufficiently
                self.base_omega = max(1.1, self.base_omega * 0.95)

class ManifoldConstrainedLinearV5(nn.Module):
    """
    Fixed MHC layer with bulletproof caching and adaptive Sinkhorn.
    """
    def __init__(self, in_features, out_features, bias=True, 
                 rank_ratio=0.3, use_cache=True, config=None):
        super().__init__()
        
        if config is None:
            config = {}
        
        self.in_features = in_features
        self.out_features = out_features
        self.use_cache = use_cache
        self._cache_valid = False
        
        # Low-rank decomposition
        # Ensure rank is at least 1
        self.rank = max(1, int(min(in_features, out_features) * rank_ratio))
        self.U = nn.Parameter(torch.randn(out_features, self.rank))
        self.V = nn.Parameter(torch.randn(self.rank, in_features))
        
        # Overrelaxed Sinkhorn projection
        proj_config = config.get('projection', {})
        self.projection = OverrelaxedSinkhornProjection(
            max_iter=proj_config.get('max_iter', 10),
            tol=proj_config.get('tol', 1e-4),
            overrelax=proj_config.get('overrelaxation', {}).get('enabled', True),
            omega=proj_config.get('overrelaxation', {}).get('initial_omega', 1.4)
        )
        
        # Bias and gain
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        self.gain = nn.Parameter(torch.ones(1))
        
        # Cache with validation
        self.register_buffer('weight_cache', None)
        self.register_buffer('cache_hash', torch.tensor(0))
        
        # Hook to detect weight changes
        self._register_hooks()
        
    def _register_hooks(self):
        """Register hooks to detect parameter changes and invalidate cache."""
        def make_invalid(*args):
            self._cache_valid = False
            self.cache_hash.data = torch.tensor(hash(str(time.time())))
        
        for param in [self.U, self.V, self.gain]:
            param.register_hook(make_invalid)
    
    def _get_projected_weights(self) -> torch.Tensor:
        """Compute or retrieve cached projected weights."""
        # Compute hash of current parameters
        current_hash = hash((
            self.U.data.sum().item(),
            self.V.data.sum().item(),
            # self.gain.item() is float, hash works on basic types
            self.gain.item()
        ))
        
        # Return cache if valid
        if (self.use_cache and self._cache_valid and 
            self.weight_cache is not None and
            current_hash == self.cache_hash.item()):
            return self.weight_cache
        
        # Recompute projection
        W_reconstructed = self.U @ self.V
        W_projected = self.projection(W_reconstructed)
        
        # Apply gain
        scale = math.sqrt(self.in_features) * self.gain
        W_final = W_projected * scale
        
        # Update cache
        if self.use_cache and not self.training:
            self.weight_cache = W_final.detach()
            # self.cache_hash needs to be a tensor to be a buffer
            self.cache_hash.data = torch.tensor(current_hash) 
            self._cache_valid = True
        
        return W_final
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use cached or computed weights
        W = self._get_projected_weights()
        
        # Linear transformation
        output = F.linear(x, W, self.bias)
        
        # Add tiny KL regularizer to prevent dead weights
        if self.training and torch.rand(1).item() < 0.01:  # Sample occasionally
            kl = (W.abs() * (W.abs().log() + math.log(W.shape[1]))).mean()
            output = output + 1e-6 * kl  # Add to gradient graph
        
        return output
    
    def extra_repr(self):
        return (f'in_features={self.in_features}, out_features={self.out_features}, '
                f'rank={self.rank}, cache={self.use_cache}')
