"""
MATHIR Library - Memory-Augmented Transformer with Hierarchical Retention
=========================================================================

Bibliothèque modulaire pour utiliser MATHIR et LSTM dans vos projets.

Usage:
    # Import MATHIR
    from mathir_lib import MATHIR
    model = MATHIR()
    
    # Ou LSTM
    from mathir_lib import LSTM
    baseline = LSTM()
    
    # Ou les deux
    from mathir_lib import MATHIR, LSTM, count_parameters
    
    mathir = MATHIR(camera_shape=(3, 224, 224), action_dim=4)
    lstm = LSTM(camera_shape=(3, 224, 224), action_dim=4)
    
    print(f"MATHIR: {count_parameters(mathir):,} params")
    print(f"LSTM: {count_parameters(lstm):,} params")

Modules:
    - MATHIR: Agent avec triple mémoire hiérarchique
    - LSTM: Baseline LSTM traditionnel  
    - components: Vision encoder et utilitaires partagés
    
Version: 2.0
Author: MATHIR Team
"""

__version__ = "2.0.0"
__author__ = "MATHIR Team"

from .mathir import MATHIR, MATHIRMemory
from .mathir_v5 import MATHIRv5
from .lstm import LSTM
from .components import (
    QuantumVisionEncoder,
    count_parameters,
    estimate_memory_usage
)

__all__ = [
    # Models
    'MATHIR',
    'MATHIRv5',
    'LSTM',
    'MATHIRMemory',
    
    # Components
    'QuantumVisionEncoder',
    
    # Utils
    'count_parameters',
    'estimate_memory_usage',
]
