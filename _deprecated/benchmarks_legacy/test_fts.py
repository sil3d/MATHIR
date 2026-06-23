import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from mathir_dropin import MATHIRMemory
import torch

m = MATHIRMemory(embedding_dim=384, db_path='test_fts3.db')

# Store
emb = torch.randn(1, 384)
m.store(embedding=emb, metadata={'text': 'Python has closures for functional programming', 'concept': 'test'})

# Test recall_text directly
print('Test recall_text:')
results = m.recall_text(query_text='python', k=3)
print('results for python:', len(results))
for r in results:
    print('  -', r.get('memory_id'), r.get('modality_text', '')[:50])

results2 = m.recall_text(query_text='closures', k=3)
print('results for closures:', len(results2))
for r in results2:
    print('  -', r.get('memory_id'), r.get('modality_text', '')[:50])