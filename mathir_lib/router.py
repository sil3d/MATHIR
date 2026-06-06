"""
KL-Constrained Router
Allocates memory access across tiers with KL divergence constraint.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Any


class KLConstrainedRouter(nn.Module):
    """
    Router with KL-divergence constraint to prevent collapse.
    
    Uses PPO-style trust region optimization:
        - Forward: compute logits, softmax to get weights
        - KL loss: KL(weights || prev_weights) or KL(weights || uniform)
        - Update: adaptively scale KL coefficient
    
    The constraint prevents the router from collapsing to a single
    memory tier, ensuring all tiers get used.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_memories: int = 4,
        kl_coefficient: float = 0.01,
        kl_target: str = "uniform",
        kl_margin: float = 0.05,
        hidden_dim: int = 256,  # Increased from128 for more capacity
        entropy_coefficient: float = 0.01,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.num_memories = num_memories
        self.kl_coefficient = kl_coefficient
        self.kl_target = kl_target
        self.kl_margin = kl_margin
        self.entropy_coefficient = entropy_coefficient
        self.temperature = temperature
        
        # Router network - deeper with more capacity
        self.router_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_memories),
        )
        
        # Previous policy for KL constraint
        self.register_buffer(
            "prev_probs",
            torch.ones(num_memories) / num_memories
        )
        
        # Adaptive KL coefficient
        self.kl_adaptive_beta = kl_coefficient
    
    def forward(
        self,
        x: torch.Tensor,
        prev_weights: Optional[torch.Tensor] = None,
        training: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute router weights with KL constraint.
        
        Args:
            x: [B, D] input features
            prev_weights: optional previous weights for KL target
            training: whether in training mode (applies KL constraint)
            
        Returns:
            dict with:
                weights: [B, num_memories] allocation weights
                kl_loss: scalar KL divergence loss
                entropy: scalar entropy (for exploration bonus)
        """
        # Raw logits
        logits = self.router_net(x)
        
        # Temperature scaling
        if training:
            logits = logits / self.temperature
        
        # Softmax weights
        weights = F.softmax(logits, dim=-1)
        
        # KL divergence constraint
        kl_loss = torch.tensor(0.0, device=x.device)
        if training:
            if self.kl_target == "uniform":
                target_probs = torch.ones_like(weights) / self.num_memories
            elif self.kl_target == "prev_policy" and prev_weights is not None:
                target_probs = prev_weights.detach()
            else:
                target_probs = self.prev_probs.unsqueeze(0).expand_as(weights)
            
            # Compute KL divergence
            kl_div = F.kl_div(
                F.log_softmax(logits, dim=-1),
                target_probs,
                reduction="batchmean",
                log_target=False,
            )
            
            # Adaptive penalty
            if kl_div > self.kl_margin:
                self.kl_adaptive_beta *= 1.5
            elif kl_div < self.kl_margin / 1.5:
                self.kl_adaptive_beta *= 0.5
            
            self.kl_adaptive_beta = max(0.001, min(1.0, self.kl_adaptive_beta))
            kl_loss = self.kl_adaptive_beta * kl_div
            
            # Update previous policy
            with torch.no_grad():
                self.prev_probs = weights.mean(dim=0).detach()
        
        # Entropy bonus (for exploration)
        entropy = -(weights * (weights + 1e-8).log()).sum(dim=-1).mean()
        
        return {
            "weights": weights,
            "kl_loss": kl_loss,
            "entropy": entropy,
        }
    
    def set_training_mode(self, training: bool) -> None:
        """Switch between training and inference modes."""
        if not training:
            for param in self.parameters():
                param.requires_grad = False
        else:
            for param in self.parameters():
                param.requires_grad = True
    
    def get_stats(self) -> Dict[str, float]:
        """Get router statistics."""
        return {
            "kl_coefficient": self.kl_adaptive_beta,
            "prev_probs": self.prev_probs.cpu().tolist(),
        }
