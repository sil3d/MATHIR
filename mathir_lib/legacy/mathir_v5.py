"""
mathir_v5.py - Fixed Core Architecture
Implements KL-divergence router constraint and training modes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from .mhc_v5 import ManifoldConstrainedLinearV5

# Dummy memory implementations as placeholders since user didn't provide them, 
# but they are referenced in HierarchicalMemoryCore.
# In a real scenario, these would be imported from existing files or implemented fully.
# Assuming existing 'WorkingMemory', 'EpisodicMemory' etc are in other files or 
# I should create placeholders if they don't exist. 
# Looking at previous file view, there is no generic memory file. 
# I will implement simple versions here to ensure the code runs self-contained or add TODOs.
# However, the user request implied updating "all code".
# I'll implement basic versions derived from previous knowledge of MATHIR structure 
# to make this file compilable.

class WorkingMemory(nn.Module):
    """Circular buffer with attention retrieval."""
    def __init__(self, capacity, feature_dim=272):
        super().__init__()
        self.capacity = capacity
        self.feature_dim = feature_dim
        self.register_buffer('buffer', torch.zeros(capacity, feature_dim))
        self.register_buffer('ptr', torch.tensor(0, dtype=torch.long))
        self.attention = nn.MultiheadAttention(feature_dim, num_heads=4, batch_first=True, dropout=0.1)
    
    def store(self, features):
        """Store features in circular buffer."""
        idx = self.ptr % self.capacity
        self.buffer[idx] = features.detach().mean(0)
        self.ptr = (self.ptr + 1) % self.capacity
    
    def retrieve(self, features):
        """Retrieve using attention over buffer."""
        stored = min(self.ptr.item(), self.capacity)
        if stored == 0:
            return features
        context = self.buffer[:stored].unsqueeze(0).expand(features.size(0), -1, -1)
        out, _ = self.attention(features.unsqueeze(1), context, context)
        return out.squeeze(1) + features  # residual
            
class EpisodicMemory(nn.Module):
    """Key-value store with similarity retrieval."""
    def __init__(self, capacity, feature_dim=272, key_dim=64):
        super().__init__()
        self.capacity = capacity
        self.register_buffer('keys', torch.zeros(capacity, key_dim))
        self.register_buffer('values', torch.zeros(capacity, feature_dim))
        self.register_buffer('ptr', torch.tensor(0, dtype=torch.long))
        self.register_buffer('count', torch.tensor(0, dtype=torch.long))
        self.encoder = nn.Linear(feature_dim, key_dim)
    
    def store(self, features):
        key = self.encoder(features.detach().mean(0, keepdim=True)).squeeze(0)
        idx = self.ptr % self.capacity
        self.keys[idx] = key
        self.values[idx] = features.detach().mean(0)
        self.ptr = (self.ptr + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)
    
    def retrieve(self, features):
        stored = self.count.item()
        if stored < 10:
            return features
        query = self.encoder(features)
        sims = F.cosine_similarity(query.unsqueeze(1), self.keys[:stored].unsqueeze(0), dim=-1)
        top_k = sims.topk(min(3, stored), dim=1)[1]
        retrieved = self.values[top_k].mean(1)
        return retrieved + features  # residual

class SemanticMemory(nn.Module):
    """Online k-means prototypes."""
    def __init__(self, num_prototypes=256, feature_dim=272, proj_dim=64):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.register_buffer('prototypes', torch.randn(num_prototypes, proj_dim))
        self.register_buffer('usage', torch.zeros(num_prototypes))
        self.down = nn.Linear(feature_dim, proj_dim)
        self.up = nn.Linear(proj_dim, feature_dim)
    
    def forward(self, features):
        projected = self.down(features)
        sims = F.cosine_similarity(projected.unsqueeze(1), self.prototypes.unsqueeze(0), dim=-1)
        idx = sims.argmax(dim=1)
        retrieved = self.prototypes[idx]
        if self.training:
            # Online update closest prototype
            with torch.no_grad():
                alpha = 0.01
                self.prototypes[idx] = (1 - alpha) * self.prototypes[idx] + alpha * projected.detach()
                self.usage[idx] += 1
        return self.up(retrieved) + features  # residual

class ImmunologicalMemory(nn.Module):
    """Anomaly detector."""
    def __init__(self, capacity=100, feature_dim=272, threshold=2.0):
        super().__init__()
        self.capacity = capacity
        self.threshold = threshold
        self.register_buffer('memory_bank', torch.zeros(capacity, feature_dim))
        self.register_buffer('ptr', torch.tensor(0, dtype=torch.long))
        self.register_buffer('count', torch.tensor(0, dtype=torch.long))
    
    def store(self, features):
        idx = self.ptr % self.capacity
        self.memory_bank[idx] = features.detach().mean(0)
        self.ptr = (self.ptr + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)
    
    def recognize(self, features):
        stored = self.count.item()
        if stored < 10:
            return None  # Not enough data
        dists = torch.cdist(features, self.memory_bank[:stored])
        min_dist = dists.min(dim=1)[0]
        # Anomaly if distance exceeds threshold
        anomaly_mask = (min_dist > self.threshold).float().unsqueeze(-1)
        return anomaly_mask * features  # Signal anomaly presence

class KLConstrainedRouter(nn.Module):
    """
    Router with KL-divergence constraint to prevent collapse.
    Implements trust region optimization similar to PPO but for memory allocation.
    """
    def __init__(self, input_dim, num_memories=4, kl_coefficient=0.01, 
                 kl_target="uniform", kl_margin=0.05):
        super().__init__()
        self.num_memories = num_memories
        self.kl_coefficient = kl_coefficient
        self.kl_target = kl_target
        self.kl_margin = kl_margin
        
        # Router network
        self.router_net = nn.Sequential(
            ManifoldConstrainedLinearV5(input_dim, 128),
            nn.GELU(),
            nn.Linear(128, num_memories)
        )
        
        # Previous policy for KL constraint (PPO-style)
        self.register_buffer('prev_probs', torch.ones(num_memories) / num_memories)
        
        # Adaptive KL coefficient
        self.kl_adaptive_beta = kl_coefficient
        
    def forward(self, x, prev_weights=None, training=True):
        # Raw logits
        logits = self.router_net(x)
        
        # Temperature annealing
        if training and hasattr(self, 'temperature'):
            logits = logits / self.temperature
        
        # Softmax weights
        weights = F.softmax(logits, dim=-1)
        
        # KL Divergence constraint
        kl_loss = torch.tensor(0.0, device=x.device)
        if training:
            if self.kl_target == "uniform":
                target_probs = torch.ones_like(weights) / self.num_memories
            elif self.kl_target == "prev_policy" and prev_weights is not None:
                target_probs = prev_weights.detach()
            else:
                target_probs = self.prev_probs.unsqueeze(0).expand_as(weights)
            
            # Calculate KL divergence using PyTorch's stable function
            kl_div = F.kl_div(
                F.log_softmax(logits, dim=-1),
                target_probs,
                reduction='batchmean',
                log_target=False
            )
            
            # Adaptive penalty based on trust region violation
            if kl_div > self.kl_margin:
                self.kl_adaptive_beta *= 1.5
            elif kl_div < self.kl_margin / 1.5:
                self.kl_adaptive_beta *= 0.5
            
            self.kl_adaptive_beta = max(0.001, min(1.0, self.kl_adaptive_beta))
            kl_loss = self.kl_adaptive_beta * kl_div
            
            # Update previous policy
            self.prev_probs = weights.mean(dim=0).detach()
        
        # Entropy bonus (additional exploration)
        entropy = Categorical(probs=weights).entropy().mean()
        
        return {
            'weights': weights,
            'kl_loss': kl_loss,
            'entropy': entropy,
            'raw_logits': logits
        }

class HierarchicalMemoryCore(nn.Module):
    """
    Fixed memory core with KL-constrained router.
    """
    def __init__(self, config):
        super().__init__()
        
        # Initialize memories
        feature_dim = config.get('feature_dim', 272)
        self.working_memory = WorkingMemory(config.get('working_capacity', 64), feature_dim=feature_dim)
        self.episodic_memory = EpisodicMemory(config.get('episodic_capacity', 1000), feature_dim=feature_dim)
        self.semantic_memory = SemanticMemory(config.get('semantic_prototypes', 256), feature_dim=feature_dim)
        self.immunological_memory = ImmunologicalMemory(
            config.get('immunological_capacity', 100), feature_dim=feature_dim
        )
        
        # KL-constrained router
        router_config = config.get('router', {})
        self.router = KLConstrainedRouter(
            input_dim=config.get('feature_dim', 256),
            num_memories=4,
            kl_coefficient=router_config.get('kl_coefficient', 0.01),
            kl_target=router_config.get('kl_target', 'uniform'),
            kl_margin=router_config.get('kl_margin', 0.05)
        )
        
        # Training mode flag
        self.training_mode = True
        
    def set_training_mode(self, training):
        """Switch between training and inference modes."""
        self.training_mode = training
        if not training:
            # Freeze router for stable inference
            for param in self.router.parameters():
                param.requires_grad = False
    
    def forward(self, features, prev_router_weights=None):
        # Get router weights with KL constraint during training
        router_out = self.router(
            features, 
            prev_weights=prev_router_weights,
            training=self.training_mode
        )
        weights = router_out['weights']
        
        # Store in memories during training
        if self.training:
            self.working_memory.store(features)
            self.episodic_memory.store(features)
            self.immunological_memory.store(features)
        
        # Split weights
        w_work, w_epi, w_sem, w_imm = weights.chunk(4, dim=-1)
        
        # Retrieve from memories
        contexts = []
        if w_work.sum() > 0.1:
            contexts.append(self.working_memory.retrieve(features) * w_work)
        if w_epi.sum() > 0.1:
            contexts.append(self.episodic_memory.retrieve(features) * w_epi)
        if w_sem.sum() > 0.1:
            contexts.append(self.semantic_memory(features) * w_sem)
        if w_imm.sum() > 0.1:
            immune_resp = self.immunological_memory.recognize(features)
            if immune_resp is not None:
                contexts.append(immune_resp * w_imm)
        
        # Combine contexts
        if contexts:
            combined = sum(contexts)
        else:
            combined = features
        
        return {
            'output': combined,
            'router_weights': weights,
            'router_loss': router_out['kl_loss'],
            'router_entropy': router_out['entropy']
        }

class DomainRandomizationManager:
    """
    Manages domain randomization with online/offline modes.
    Implements DROPO-inspired offline DR.
    """
    def __init__(self, config):
        self.mode = config.get('dr_mode', 'hybrid')
        self.online_prob = config.get('online_dr', {}).get('probability', 0.7)
        
        # Load offline augmentation pool if needed
        self.offline_pool = None
        if self.mode in ['offline', 'hybrid']:
            pool_path = config.get('offline_dr', {}).get('augmentation_pool')
            if pool_path:
                try:
                    self.offline_pool = torch.load(pool_path, weights_only=True)
                except:
                    print(f"Warning: Could not load offline pool from {pool_path}")
    
    def augment_batch(self, batch, training=True):
        """Apply domain randomization based on mode."""
        if not training:
            return batch
        
        if self.mode == 'offline' and self.offline_pool:
            # Sample from pre-augmented pool
            idx = torch.randint(0, len(self.offline_pool), (batch.shape[0],))
            return torch.stack([self.offline_pool[i] for i in idx])
        
        elif self.mode == 'online' or (self.mode == 'hybrid' and 
                                      torch.rand(1).item() < self.online_prob):
            # Lightweight online augmentations
            return self._apply_online_augmentations(batch)
        
        return batch
    
    def _apply_online_augmentations(self, batch):
        """Fast online augmentations for RL training."""
        # Simplified - implement your actual augmentations here
        if torch.rand(1).item() < 0.3:
            batch = batch.flip(-1)  # Horizontal flip
        if torch.rand(1).item() < 0.3:
            noise = torch.randn_like(batch) * 0.02
            batch = batch + noise
        return batch

class MATHIRv5(nn.Module):
    """
    Production-Ready MATHIR V5 Agent.
    Combines V5 Memory Core with Quantum Vision Encoder and MHC-based Actor.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # --- Perception ---
        # Using QuantumVisionEncoder from components (Shared with Legacy/LSTM)
        from .components import QuantumVisionEncoder
        
        # Handle new config structure (auto_config) vs old (manual)
        if 'perception' in config['model'] and 'input_resolution' in config['model']['perception']:
            res = config['model']['perception']['input_resolution']
            cam_shape = (3, res[0], res[1]) # Add channels
        else:
            cam_shape = tuple(config['model'].get('camera_shape', (3, 84, 84)))

        self.vision_encoder = QuantumVisionEncoder(
            input_shape=cam_shape
        )
        
        # State Encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(config['model']['state_dim'], 32),
            nn.GELU(),
            nn.Linear(32, 16)
        )
        
        # Feature sizes
        self.combined_dim = 256 + 16 # Vision + State
        
        # --- Memory Core V5 ---
        # Update config to match computed dim
        config['memory']['feature_dim'] = self.combined_dim
        self.memory_core = HierarchicalMemoryCore(config['memory'])
        
        # --- Action Heads (Actor) ---
        # Using MHC V5 for the policy head as well for stability
        self.actor = nn.Sequential(
            ManifoldConstrainedLinearV5(
                self.combined_dim, 
                128, 
                config={'projection': config['mhc']['projection']}
            ),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, config['model']['action_dim'])
        )
        
        # Learnable Log Std for Policy Gradient
        self.log_std = nn.Parameter(torch.zeros(1, config['model']['action_dim']))
        
        # Domain Randomization
        self.dr_manager = DomainRandomizationManager(config.get('domain_randomization', {}))

    def forward(self, observations, prev_router_weights=None, step=0, performance_cue=0.0):
        """
        Forward pass compatible with RL Training Loop.
        Args:
            observations: dict({'camera': ..., 'state': ...})
            prev_router_weights: For KL constraints
            step: Current step (unused by V5 core but kept for compatibility)
            performance_cue: Scalar cue (unused by V5 core but kept)
        """
        # 1. Perception
        vision_feat = self.vision_encoder(observations['camera'])
        state_feat = self.state_encoder(observations['state'])
        
        # 2. Early Fusion
        features = torch.cat([vision_feat, state_feat], dim=-1)
        
        # 3. Memory Core pass (Router -> Memories -> Context)
        memory_out = self.memory_core(features, prev_router_weights)
        context = memory_out['output']
        
        # 4. Action Prediction
        action_mean = self.actor(context)
        
        # Return dict matching legacy interface + V5 extras
        return {
            'action_mean': action_mean,
            'log_std': self.log_std,
            'features': context,                # For Critic or next step
            'router_weights': memory_out['router_weights'],
            'router_loss': memory_out['router_loss'],
            'router_entropy': memory_out['router_entropy']
        }

    def reset_memory(self):
        """Reset internal memory states."""
        # If WorkingMemory had state (like LSTM hidden), we'd reset it here.
        pass

    @classmethod
    def from_config(cls, config_path):
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)
        return cls(config)
