"""
MATHIR: Memory-Augmented Transformer with Hierarchical Retention
Implémentation complète pour benchmark vs LSTM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple


class QuantumVisionEncoder(nn.Module):
    """CNN optimisée pour extraction de features visuelles"""
    
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
        features = self.cnn(x)
        features = features.view(features.size(0), -1)
        return self.output_proj(features)


class MATHIRMemory(nn.Module):
    """Memory-Augmented Transformer with Hierarchical Retention"""
    
    def __init__(self, input_dim=256, mem_slots=512):
        super().__init__()
        
        # === Mémoire de Travail (Court Terme) ===
        self.working_capacity = 64
        self.register_buffer(
            'working_buffer',
            torch.zeros(1, self.working_capacity, input_dim)
        )
        self.working_ptr = 0
        
        # Attention fenêtre glissante
        self.working_attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=4,
            batch_first=True,
            dropout=0.1
        )
        
        # === Mémoire Épisodique (Moyen Terme) ===
        self.episodic_capacity = 1000
        self.register_buffer(
            'episodic_keys',
            torch.zeros(mem_slots, 64)
        )
        self.register_buffer(
            'episodic_values',
            torch.zeros(mem_slots, 256)
        )
        self.episodic_ptr = 0
        self.episodic_count = 0
        
        # Auto-encodeur pour compression
        self.episodic_encoder = nn.Sequential(
            nn.Linear(input_dim + 2, 128),
            nn.GELU(),
            nn.Linear(128, 64)
        )
        
        # === Mémoire Sémantique (Long Terme) ===
        self.semantic_slots = 256
        self.register_buffer(
            'semantic_prototypes',
            torch.randn(self.semantic_slots, 64)
        )
        self.register_buffer(
            'semantic_usage',
            torch.zeros(self.semantic_slots)
        )
        
        # === Routeur Hiérarchique ===
        self.router = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Linear(128, 3),  # [working, episodic, semantic]
            nn.Softmax(dim=-1)
        )
        
        # === Rétention Temporelle ===
        self.time_embedding = nn.Embedding(1000, input_dim)
        self.register_buffer(
            'retention_decay',
            torch.tensor([0.9, 0.7, 0.5])  # 3 échelles temporelles
        )
        
        # Normalisation finale
        self.layer_norm = nn.LayerNorm(input_dim)
    
    def forward(self, x, actions=None, step=0):
        """
        x: [batch, features]
        actions: [batch, 2] optionnel
        step: pas temporel pour rétention
        """
        batch_size = x.size(0)
        
        # === 1. Mémoire de Travail ===
        self.update_working_memory(x)
        working_size = min(self.working_ptr, self.working_capacity)
        
        if working_size > 0:
            working_context = self.working_buffer[:, :working_size, :].expand(batch_size, -1, -1)
            x_working, _ = self.working_attention(
                x.unsqueeze(1),
                working_context,
                working_context
            )
            x_working = x_working.squeeze(1)
        else:
            x_working = x
        
        # === 2. Mémoire Épisodique ===
        if actions is not None and self.episodic_count > 0:
            # Encode épisode actuel
            episode_input = torch.cat([x, actions], dim=-1)
            episode_code = self.episodic_encoder(episode_input)
            
            # Stocke dans buffer (détaché pour éviter gradients)
            with torch.no_grad():
                idx = self.episodic_ptr % self.episodic_capacity
                self.episodic_keys[idx] = episode_code.mean(0).detach()
                self.episodic_ptr = (self.episodic_ptr + 1) % self.episodic_capacity
                self.episodic_count = min(self.episodic_count + 1, self.episodic_capacity)
            
            # Récupère K plus similaires
            if self.episodic_count > 10:
                similarities = F.cosine_similarity(
                    episode_code.unsqueeze(1),
                    self.episodic_keys[:self.episodic_count].unsqueeze(0),
                    dim=-1
                )
                top_k = similarities.topk(min(3, self.episodic_count), dim=1)[1]
                episodic_features = self.episodic_values[top_k].mean(1)
                x_episodic = episodic_features
            else:
                x_episodic = torch.zeros_like(x)
        else:
            x_episodic = torch.zeros_like(x)
        
        # === 3. Mémoire Sémantique ===
        # Mise à jour online des prototypes (k-means)
        if self.training and step % 100 == 0:
            self.update_semantic_prototypes(x.detach())
        
        # Association au prototype le plus proche
        x_downsampled = F.adaptive_avg_pool1d(x.unsqueeze(1), 64).squeeze(1)
        semantic_scores = F.cosine_similarity(
            x_downsampled.unsqueeze(1),
            self.semantic_prototypes.unsqueeze(0),
            dim=-1
        )
        semantic_idx = semantic_scores.argmax(dim=1)
        x_semantic = self.semantic_prototypes[semantic_idx]
        
        # Upsample to match x dimension
        x_semantic = F.interpolate(
            x_semantic.unsqueeze(1), 
            size=x.size(1), 
            mode='linear', 
            align_corners=False
        ).squeeze(1)
        
        # === 4. Fusion Hiérarchique ===
        # Poids du routeur
        router_weights = self.router(x)
        w_work, w_epi, w_sem = router_weights.chunk(3, dim=-1)
        
        # Facteurs de rétention temporelle
        decay_factors = self.retention_decay ** (step / 100)
        
        # Fusion pondérée
        output = (w_work * x_working * decay_factors[0] +
                 w_epi * x_episodic * decay_factors[1] +
                 w_sem * x_semantic * decay_factors[2])
        
        # Normalisation finale
        return self.layer_norm(output + x)  # Residual connection
    
    def update_working_memory(self, x):
        """Buffer circulaire O(1)"""
        batch_size = x.size(0)
        
        with torch.no_grad():
            for i in range(batch_size):
                idx = (self.working_ptr + i) % self.working_capacity
                self.working_buffer[0, idx] = x[i]
            
            self.working_ptr = (self.working_ptr + batch_size) % self.working_capacity
    
    def update_semantic_prototypes(self, x):
        """K-means online pour prototypes sémantiques"""
        with torch.no_grad():
            x_downsampled = F.adaptive_avg_pool1d(x.unsqueeze(1), 64).squeeze(1)
            for i in range(x.size(0)):
                distances = torch.cdist(x_downsampled[i:i+1], self.semantic_prototypes)
                closest = distances.argmin()
                
                # Mise à jour avec momentum
                alpha = 0.01
                self.semantic_prototypes[closest] = (
                    (1 - alpha) * self.semantic_prototypes[closest] + 
                    alpha * x_downsampled[i]
                )
                self.semantic_usage[closest] += 1


class MATHIRAgent(nn.Module):
    """Agent complet MATHIR pour conduite autonome"""
    
    def __init__(self, input_dim=256, hidden_dim=256, action_dim=2):
        super().__init__()
        
        # Vision encoder
        self.vision_encoder = QuantumVisionEncoder(input_shape=(1, 84, 84))
        
        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(5, 32),  # speed, steering, throttle, x, y
            nn.GELU(),
            nn.Linear(32, 16)
        )
        
        # Cœur MATHIR
        combined_dim = 256 + 16
        self.memory = MATHIRMemory(
            input_dim=combined_dim,
            mem_slots=512
        )
        
        # Tête d'action
        self.actor = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, action_dim)
        )
        
        # Log-std apprenable
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))
    
    def forward(self, observations, actions=None, step=0):
        """
        observations: dict avec
            - camera: [B, 1, H, W]
            - state: [B, 5]
        """
        # Encode observations
        vision_feat = self.vision_encoder(observations['camera'])
        state_feat = self.state_encoder(observations['state'])
        
        # Concaténation
        combined = torch.cat([vision_feat, state_feat], dim=-1)
        
        # Mémoire
        combined = self.memory(combined, actions, step)
        
        # Prédictions d'action
        action_mean = self.actor(combined)
        
        return {
            'action_mean': action_mean,
            'features': combined
        }
    
    def get_memory_stats(self):
        """Statistiques sur l'utilisation de la mémoire"""
        return {
            'working_usage': self.memory.working_ptr,
            'episodic_usage': self.memory.episodic_count,
            'semantic_usage': self.memory.semantic_usage.sum().item()
        }
    
    def reset_memory(self):
        """Réinitialise toutes les mémoires pour un nouveau test"""
        # Reset working memory
        self.memory.working_buffer.zero_()
        self.memory.working_ptr = 0
        
        # Reset episodic memory
        self.memory.episodic_keys.zero_()
        self.memory.episodic_values.zero_()
        self.memory.episodic_ptr = 0
        self.memory.episodic_count = 0
        
        # Semantic prototypes gardent leur état (long terme)


class LSTMBaseline(nn.Module):
    """LSTM baseline pour comparaison"""
    
    def __init__(self, input_dim=256, hidden_dim=256, action_dim=2):
        super().__init__()
        
        # Vision encoder (même que MATHIR)
        self.vision_encoder = QuantumVisionEncoder(input_shape=(1, 84, 84))
        
        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(5, 32),
            nn.GELU(),
            nn.Linear(32, 16)
        )
        
        # LSTM core
        combined_dim = 256 + 16
        self.lstm = nn.LSTM(
            input_size=combined_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # Tête d'action
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, action_dim)
        )
        
        # Hidden state
        self.register_buffer('h', torch.zeros(2, 1, hidden_dim))
        self.register_buffer('c', torch.zeros(2, 1, hidden_dim))
    
    def forward(self, observations, reset_hidden=False):
        """
        observations: dict avec
            - camera: [B, 1, H, W]
            - state: [B, 5]
        """
        batch_size = observations['camera'].size(0)
        
        # Reset hidden state si demandé
        if reset_hidden or self.h.size(1) != batch_size:
            self.h = torch.zeros(2, batch_size, 256, device=observations['camera'].device)
            self.c = torch.zeros(2, batch_size, 256, device=observations['camera'].device)
        
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


def estimate_memory_usage(model, batch_size=32, input_shape=(1, 84, 84)):
    """Estime l'utilisation mémoire VRAM"""
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


if __name__ == "__main__":
    # Test instantiation
    print("=== Test MATHIR vs LSTM ===\n")
    
    mathir = MATHIRAgent()
    lstm = LSTMBaseline()
    
    print(f"MATHIR Parameters: {count_parameters(mathir):,}")
    print(f"LSTM Parameters: {count_parameters(lstm):,}")
    
    print(f"\nMATHIR Memory: {estimate_memory_usage(mathir):.2f} GB")
    print(f"LSTM Memory: {estimate_memory_usage(lstm):.2f} GB")
    
    # Test forward pass
    dummy_obs = {
        'camera': torch.randn(8, 1, 84, 84),
        'state': torch.randn(8, 5)
    }
    
    mathir_out = mathir(dummy_obs)
    lstm_out = lstm(dummy_obs)
    
    print(f"\nMATHIR output shape: {mathir_out['action_mean'].shape}")
    print(f"LSTM output shape: {lstm_out['action_mean'].shape}")
    
    print("\n✓ Models initialized successfully!")
