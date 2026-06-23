"""
Composants Partagés - Vision Encoder et Utilitaires
===================================================

Composants utilisés par MATHIR et LSTM.
"""

import torch
import torch.nn as nn


class QuantumVisionEncoder(nn.Module):
    """
    CNN optimisée pour extraction de features visuelles
    
    Architecture:
        - 3 couches Conv2d (32→64→128 channels)
        - BatchNorm + GELU + Dropout après chaque couche
        - AdaptiveAvgPool2d final
        - Projection linéaire vers 256 dimensions
    
    Args:
        input_shape: (C, H, W) forme de l'image d'entrée
    """
    
    def __init__(self, input_shape=(1, 84, 84)):
        super().__init__()
        
        self.cnn = nn.Sequential(
            # Couche 1: Extraction bas-niveau
            nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Dropout2d(0.05),
            
            # Couche 2: Features moyennes
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Dropout2d(0.1),
            
            # Couche 3: Haut-niveau
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Dropout2d(0.15),
            
            # Pooling adaptatif
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Calcul dimension sortie
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            out = self.cnn(dummy)
            self.feature_dim = out.view(1, -1).shape[1]
        
        # Projection finale
        self.output_proj = nn.Linear(self.feature_dim, 256)
    
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] images
        
        Returns:
            [B, 256] features
        """
        features = self.cnn(x)
        features = features.view(features.size(0), -1)
        return self.output_proj(features)


def count_parameters(model):
    """
    Compte le nombre de paramètres entraînables
    
    Args:
        model: nn.Module
    
    Returns:
        int: Nombre de paramètres
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_memory_usage(model, batch_size=32, input_shape=(1, 84, 84)):
    """
    Estime l'utilisation mémoire VRAM
    
    Args:
        model: nn.Module
        batch_size: Taille de batch pour estimation
        input_shape: Forme de l'entrée caméra
    
    Returns:
        float: Mémoire estimée en GB
    """
    # Paramètres (float32 = 4 bytes)
    params_memory = sum(p.numel() * 4 for p in model.parameters()) / 1024**3
    
    # Activations (estimation avec forward pass)
    dummy_obs = {
        'camera': torch.randn(batch_size, *input_shape),
        'state': torch.randn(batch_size, 5)
    }
    
    model.eval()
    with torch.no_grad():
        _ = model(dummy_obs)
    
    return params_memory
