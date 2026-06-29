#!/usr/bin/env python3
"""Inspect VecMemory schema."""
import sys
sys.path.insert(0, r'C:\Users\So-i-learn-3D\.config\mimocode\tools\mathir_mcp\mathir_lib')
from mathir_vec import VecMemory
import tempfile

db = tempfile.mktemp(suffix='.db')
vm = VecMemory(db, 384)
schema = vm._schema_kind()
print('Schema kind:', schema)
conn = vm._get_conn()
tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
print('Tables:', tables)
cols = [dict(r) for r in conn.execute('PRAGMA table_info(memories)').fetchall()]
for c in cols:
    print(' ', c['name'], c['type'], 'default=', c.get('dflt_value'))
