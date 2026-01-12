"""
LSTM Baseline - Architecture traditionnelle pour comparaison
============================================================

LSTM standard pour benchmarking contre MATHIR.

Usage:
    from mathir_lib import LSTM
    
    model = LSTM(
        camera_shape=(1, 84, 84),
        state_dim=5,
        action_dim=2
    )
    
    output = model(observations)
"""

import torch
import torch.nn as nn
from typing import Dict

from .components import QuantumVisionEncoder


class LSTM(nn.Module):
    """
    Architecture LSTM standard
    
    Args:
        camera_shape: Forme de l'entrée caméra (C, H, W)
        state_dim: Dimension du vecteur d'état
        action_dim: Dimension de l'action
        hidden_dim: Dimension du hidden state LSTM
        num_layers: Nombre de couches LSTM
    """
    
    def __init__(
        self,
        camera_shape=(3, 84, 84),
        state_dim=5,
        action_dim=2,
        hidden_dim=256,
        num_layers=2
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Vision encoder (même que MATHIR pour comparaison équitable)
        self.vision_encoder = QuantumVisionEncoder(input_shape=camera_shape)
        
        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.GELU(),
            nn.Linear(32, 16)
        )
        
        # LSTM core
        combined_dim = 256 + 16
        self.lstm = nn.LSTM(
            input_size=combined_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0
        )
        
        # Tête d'action
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, action_dim)
        )
        
        # RL Policy Gradient (Learnable Standard Deviation)
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))
        
        # Hidden state
        self.register_buffer('h', torch.zeros(num_layers, 1, hidden_dim))
        self.register_buffer('c', torch.zeros(num_layers, 1, hidden_dim))
    
    def forward(self, observations, reset_hidden=False):
        """
        Args:
            observations: dict avec
                - camera: [B, C, H, W]
                - state: [B, state_dim]
            reset_hidden: Si True, reset le hidden state
        
        Returns:
            dict avec 'action_mean' et 'features'
        """
        batch_size = observations['camera'].size(0)
        
        # Reset hidden state si demandé
        if reset_hidden or self.h.size(1) != batch_size:
            self.h = torch.zeros(
                self.num_layers, batch_size, self.hidden_dim,
                device=observations['camera'].device
            )
            self.c = torch.zeros(
                self.num_layers, batch_size, self.hidden_dim,
                device=observations['camera'].device
            )
        
        # Encode observations
        vision_feat = self.vision_encoder(observations['camera'])
        state_feat = self.state_encoder(observations['state'])
        
        # Concaténation
        combined = torch.cat([vision_feat, state_feat], dim=-1)
        
        # LSTM
        lstm_out, (self.h, self.c) = self.lstm(
            combined.unsqueeze(1),
            (self.h.detach(), self.c.detach())
        )
        lstm_out = lstm_out.squeeze(1)
        
        # Prédictions d'action
        action_mean = self.actor(lstm_out)
        
        return {
            'action_mean': action_mean,
            'log_std': self.log_std,
            'features': lstm_out
        }
    
    def get_memory_stats(self):
        """Statistiques sur l'état caché"""
        return {
            'hidden_norm': torch.norm(self.h).item(),
            'cell_norm': torch.norm(self.c).item()
        }
    
    def reset_memory(self):
        """Réinitialise les hidden states"""
        self.h.zero_()
        self.c.zero_()


def count_parameters(model):
    """Compte le nombre de paramètres"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
