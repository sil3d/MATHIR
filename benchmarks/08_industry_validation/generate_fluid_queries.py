import json, random, sys, os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

CORPUS_FILE = r'D:\SECRET_PROJECT\MATHIR\benchmarks\05_test_data\beir_data\fluid_mechanics\fluid_mechanics\corpus.jsonl'
OUT_QUERIES = r'D:\SECRET_PROJECT\MATHIR\benchmarks\05_test_data\beir_data\fluid_mechanics\fluid_mechanics\queries.jsonl'
OUT_QRELS_DIR = r'D:\SECRET_PROJECT\MATHIR\benchmarks\05_test_data\beir_data\fluid_mechanics\fluid_mechanics\qrels'
OUT_QRELS = os.path.join(OUT_QRELS_DIR, 'test.tsv')

# 50 questions generated from the actual 50 corpus chunks sampled
# (seed=42, proportional across White + Cengel books)
QUESTIONS = [
    ("yunus_p846_c0",    "What is multigridding in computational fluid dynamics and why does it reduce convergence time?"),
    ("white_p284_c0",    "What is the velocity potential function in irrotational flow and how does it relate to the stream function?"),
    ("yunus_p729_c0",    "How are open channel flows classified by slope (M, C, S, H, A) and what determines each classification?"),
    ("yunus_p854_c0",    "What are periodic boundary conditions in CFD and for what type of flows are they useful?"),
    ("yunus_p465_c0",    "What is Couette flow and what are the key assumptions used in its analysis?"),
    ("white_p583_c1",    "What is the Kutta condition for flow around an airfoil with a sharp trailing edge?"),
    ("yunus_p552_c0",    "What is the displacement thickness effect in boundary layers and how can it be eliminated in a wind tunnel test section?"),
    ("yunus_p225_c0",    "How does the Bernoulli equation for compressible flow differ between isothermal and isentropic processes for an ideal gas?"),
    ("white_p98_c0",     "How do you calculate pressure at a point in a hydrostatic fluid using a manometer?"),
    ("white_p91_c0",     "What is the hydrostatic pressure distribution formula for lakes and oceans and how accurate is it in the atmosphere up to 1000 m?"),
    ("white_p256_c1",    "What is stratified flow and why must vertical density changes be accounted for even when velocities are small?"),
    ("white_p337_c0",    "What is the pressure coefficient Cp on an airfoil surface and how is it used to calculate lift?"),
    ("white_p256_c0",    "What is the continuity equation for compressible flow and how does it relate to conservation of mass?"),
    ("white_p482_c0",    "What is the momentum integral equation for a boundary layer and what physical principles does it express?"),
    ("white_p48_c0",     "What is the definition of the Reynolds number and what physical interpretation does it have for flow in a pipe?"),
    ("yunus_p215_c0",    "How does the Bernoulli equation apply to a Pitot tube and how can it be used to measure flow velocity?"),
    ("white_p630_c1",    "What are the differences between subsonic, transonic, supersonic, and hypersonic flows in terms of Mach number ranges?"),
    ("white_p380_c0",    "What is the definition of the Froude number and what does it represent in free-surface flows?"),
    ("yunus_p490_c0",    "How does the energy equation for steady incompressible flow account for pump or turbine work?"),
    ("white_p530_c0",    "What is the difference between laminar and turbulent flow in a boundary layer and what triggers transition?"),
    ("white_p262_c0",    "What is the physical meaning of the Navier-Stokes equations and what fundamental principles do they express?"),
    ("yunus_p507_c1",    "How is the Euler equation derived from the Navier-Stokes equations and what assumption does it rely on?"),
    ("yunus_p355_c0",    "What is the concept of hydraulic diameter and how is it used to treat non-circular ducts as equivalent circular pipes?"),
    ("yunus_p610_c1",    "What is the physical mechanism of drag crisis and at what Reynolds number does it typically occur for a sphere?"),
    ("yunus_p382_c1",    "How does a centrifugal pump characteristic curve (head versus flow rate) relate to the pump affinity laws?"),
    ("white_p324_c0",    "What is the definition of the Weber number and when is it important in fluid mechanics?"),
    ("yunus_p580_c0",    "What is the difference between streamline curvature and vorticity as measures of rotationality in a flow field?"),
    ("white_p389_c0",    "How does the Moody chart relate the friction factor to Reynolds number and relative roughness for pipe flow?"),
    ("white_p191_c0",    "What is the physical meaning of the Bernoulli equation along a streamline and what are its key limitations?"),
    ("white_p25_c0",     "What is the concept of similitude and dimensional analysis in scaling up small-scale experiments to full-scale prototype flows?"),
    ("white_p506_c0",    "How does the drag force on a bluff body change with Reynolds number around the critical drag crisis range?"),
    ("yunus_p90_c0",     "What is the difference between absolute pressure and gage pressure, and how does pressure vary with depth in a fluid at rest?"),
    ("yunus_p855_c0",    "What are fan boundary conditions and sliding mesh interfaces in CFD, and for what applications are they used?"),
    ("yunus_p893_c0",    "How is the momentum equation applied to a sluice gate to determine the forces on the gate and downstream flow velocity?"),
    ("yunus_p285_c1",    "Why can atmospheric pressure usually be disregarded when applying the momentum equation to open-channel flows?"),
    ("yunus_p566_c1",    "What is the shape factor H in boundary layer theory and how does it relate to momentum thickness and displacement thickness?"),
    ("white_p291_c0",    "What is plane Couette flow and how is the velocity profile derived from the Navier-Stokes equations for steady incompressible flow between two parallel plates?"),
    ("yunus_p484_c0",    "What is the stream function in two-dimensional incompressible flow and why is it useful in fluid mechanics?"),
    ("white_p321_c1",    "How is dimensional analysis used to form dimensionless groups like the drag coefficient and Reynolds number in fluid flow problems?"),
    ("white_p384_c0",    "What is the logarithmic law of the wall for turbulent boundary layers and how does it describe the velocity profile near a solid surface?"),
    ("white_p610_c1",    "What is the pressure coefficient Cp in aerodynamics and how is it used to compare experimental and computational results for lift on an airfoil?"),
    ("white_p639_c0",    "How are stagnation pressure and stagnation density defined in compressible flow and what is their relationship to static properties through the Mach number?"),
    ("white_p711_c0",    "How does cooling in a constant-area duct affect the Mach number and temperature of a supersonic flow?"),
    ("white_p146_c0",    "How can Archimedes' principle be used to determine whether a crown is pure gold by measuring its weight in air and in water?"),
    ("white_p355_c2",    "How is dimensional analysis applied to form dimensionless pi groups for the vibration frequency of a beam or the lift force on a missile?"),
    ("yunus_p651_c1",    "What is the relationship between throat area and exit area in a converging-diverging nozzle for choked flow, and how does Mach number vary through the nozzle?"),
    ("white_p26_c1",     "What is the definition of a fluid and how does it differ from a solid in terms of response to shear stress?"),
    ("white_p170_c0",    "How does the conservation of mass principle lead to the continuity equation for steady incompressible flow?"),
    ("yunus_p120_c0",    "What is Pascal's law and how does pressure act in all directions at a point in a fluid at rest?"),
    ("yunus_p74_c1",     "What is the definition of viscosity in a Newtonian fluid and how does shear stress relate to the velocity gradient?"),
    ("yunus_p420_c1",    "What is the energy equation for steady incompressible flow and how does it account for changes in kinetic and potential energy?"),
]

# Validate no duplicates
chunk_ids = [c for c, q in QUESTIONS]
if len(chunk_ids) != len(set(chunk_ids)):
    from collections import Counter
    dups = [c for c, n in Counter(chunk_ids).items() if n > 1]
    print(f'WARNING: {len(dups)} duplicate chunk IDs: {dups}')

# Validate all chunk IDs exist in corpus
corpus = []
with open(CORPUS_FILE, encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            corpus.append(json.loads(line))
chunk_ids_in_corpus = {c['_id'] for c in corpus}
valid_qs = [(cid, q) for cid, q in QUESTIONS if cid in chunk_ids_in_corpus]
missing = [cid for cid, q in QUESTIONS if cid not in chunk_ids_in_corpus]
if missing:
    print(f'WARNING: {len(missing)} chunk IDs not found in corpus: {missing[:5]}')
print(f'Writing {len(valid_qs)} queries for {len(chunk_ids_in_corpus)} total corpus chunks')

# Write queries.jsonl
os.makedirs(os.path.dirname(OUT_QUERIES), exist_ok=True)
qrels_rows = []
queries_out = []

for i, (chunk_id, question) in enumerate(valid_qs):
    qid = str(i + 1)
    queries_out.append({"_id": qid, "text": question, "metadata": {"chunk_id": chunk_id}})
    qrels_rows.append((qid, chunk_id, 1))

with open(OUT_QUERIES, 'w', encoding='utf-8') as f:
    for q in queries_out:
        f.write(json.dumps(q, ensure_ascii=False) + '\n')

print(f'Wrote {len(queries_out)} queries to {OUT_QUERIES}')

# Write qrels/test.tsv
os.makedirs(OUT_QRELS_DIR, exist_ok=True)
with open(OUT_QRELS, 'w', encoding='utf-8') as f:
    f.write('query-id\tcorpus-id\tscore\n')
    for qid, cid, score in qrels_rows:
        f.write(f'{qid}\t{cid}\t{score}\n')

print(f'Wrote {len(qrels_rows)} qrels rows to {OUT_QRELS}')
print()
print('=== 50 Generated Questions ===')
for i, (cid, q) in enumerate(QUESTIONS):
    print(f'Q{i+1:02d} [{cid}]: {q}')
