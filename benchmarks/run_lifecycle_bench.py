#!/usr/bin/env python3
"""Quick MATHIR lifecycle benchmark."""
import sys
sys.path.insert(0, 'D:\\SECRET_PROJECT\\MATHIR\\mathir_mcp\\mathir_lib')
import time, numpy as np
from mathir_vec import VecMemory
from pathlib import Path
import tempfile

db = Path(tempfile.mktemp(suffix='.db'))
vm = VecMemory(db, 384)

print("=== LIFECYCLE BENCHMARK ===")

start = time.time()
for i in range(100):
    emb = np.random.randn(384).astype('float32')
    vm.store(f'mem_{i}', emb, {'content': f'Memory {i}: topic {i%10}', 'agent': 'bench', 'block_type': 'episodic', 'label': f'bench-{i}', 'priority': 5})
store_time = time.time() - start
print(f"Store 100 memories: {store_time:.3f}s ({100/store_time:.0f} ops/s)")

start = time.time()
for i in range(50):
    q = np.random.randn(384).astype('float32')
    vm.search(q, k=5)
recall_time = time.time() - start
print(f"Recall 50 queries: {recall_time:.3f}s ({50/recall_time:.0f} ops/s)")

promoted = vm.auto_promote_all()
print(f"Auto-promote: {len(promoted)} promoted")

stats = vm.stats()
total = stats.get("total", 0)
db_size = stats.get("db_size_bytes", 0)
print(f"Final stats: {total} memories, DB: {db_size} bytes")
print("=== BENCHMARK COMPLETE ===")
try: db.unlink()
except: pass
