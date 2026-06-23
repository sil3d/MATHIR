# train_mathir_v5.py - Complete training with all fixes
import torch
import torch.optim as optim
import yaml
import os
import sys

# Ensure mathir_lib is importable
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from mathir_lib.mathir_v5 import MATHIRv5, DomainRandomizationManager

class KLAnnealer:
    """KL coefficient annealer."""
    def __init__(self, start=0.001, end=0.01, steps=2000):
        self.start = start
        self.end = end
        self.steps = steps
        self.current = start
        self.step_count = 0
    
    def step(self):
        self.current = min(self.end, self.start + 
                          (self.end - self.start) * (self.step_count / self.steps))
        self.step_count += 1
        return self.current

def validate_model(model, dr_manager):
    """
    Validation placeholder.
    In a real scenario, run episodes in simulation without training.
    """
    model.eval()
    print("Running validation...")
    # Placeholder validation logic
    pass

def train_with_fixes(config_path):
    print(f"Loading config from {config_path}")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Initialize
    model = MATHIRv5(config)
    # config['perception'] might be missing if we didn't add it in the yaml properly or use defaults.
    # config/mathir_v5.yaml has it.
    dr_manager = DomainRandomizationManager(config.get('perception', {}))
    optimizer = optim.Adam(model.parameters(), 
                         lr=config['training']['learning_rate'])
    
    # KL Annealing
    kl_scheduler = KLAnnealer(
        start=config['training']['kl_annealing']['start_weight'],
        end=config['training']['kl_annealing']['end_weight'],
        steps=config['training']['kl_annealing']['anneal_episodes']
    )
    
    print("Starting training loop...")
    for episode in range(config['training']['num_episodes']):
        # Training step
        model.train()
        
        # Synthesize dummy data for demonstration since we don't have the simulator loop here
        # In real use, this comes from the env (see driving_env._get_observation).
        # MATHIRv5.forward() expects dict with 'camera' [B,1,84,84] and 'state' [B,state_dim]
        batch_size = config['training']['batch_size']
        observations = {
            'camera': torch.randn(batch_size, 1, 84, 84),
            'state': torch.randn(batch_size, config['model']['state_dim'])
        }
        
        # Apply domain randomization to camera only
        observations['camera'] = dr_manager.augment_batch(observations['camera'], training=True)
        
        # Forward with router constraint
        outputs = model(observations)
        
        # Dummy losses for demonstration
        policy_loss = torch.tensor(0.5, requires_grad=True)
        value_loss = torch.tensor(0.2, requires_grad=True)
        
        # KL-constrained loss
        kl_weight = kl_scheduler.step()
        total_loss = (policy_loss + 
                     value_loss + 
                     kl_weight * outputs['router_loss'] -
                     0.001 * outputs['router_entropy'])  # Entropy bonus
        
        # Optimize
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        
        # Validation every 1000 episodes (reduced for demo)
        if episode % 1000 == 0:
            print(f"Episode {episode}: Loss={total_loss.item():.4f}, RouterLoss={outputs['router_loss'].item():.4f}")
            validate_model(model, dr_manager)

if __name__ == "__main__":
    config_path = "config/mathir_v5.yaml"
    if not os.path.exists(config_path):
        # Fallback if running from root without config dir being current
        if os.path.exists(os.path.join("MATHIR", config_path)):
            config_path = os.path.join("MATHIR", config_path)
    
    train_with_fixes(config_path)
