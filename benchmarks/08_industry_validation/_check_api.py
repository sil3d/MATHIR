import sys, json, requests
sys.path.insert(0, r'D:\SECRET_PROJECT\MATHIR\benchmarks\08_industry_validation')
import mathir_adapter as ma
adapter = ma.MathirAdapter()

# Direct API call
r = requests.post('http://127.0.0.1:7338/api/memory/hybrid_search', json={'project': 'fluid_mechanics_bench', 'query': 'multigridding CFD', 'k': 3})
j = r.json()
print('Response keys:', list(j.keys()))
print('First result:')
import pprint
pprint.pprint(j['results'][0])
