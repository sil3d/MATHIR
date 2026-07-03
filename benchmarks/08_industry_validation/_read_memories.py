import sqlite3
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

conn = sqlite3.connect(r'C:\Users\So-i-learn-3D\.config\mathir\data\projects\MATHIR\mathir.db')
cursor = conn.cursor()

# Read the most important memories from claude-code
important_ids = [
    'mem_dfb9736d788f4de783cc43f2de5bfa88',  # final conclusion
    'mem_2109039ae9b4460cad08bbd59c579139',  # handoff
    'mem_df505017ee4a4a0db66793ebb9317dd7',  # ppr-lte reconciled
    'mem_ed5820bcb9114db0867099a8b0d41ab1',  # benchmark status corrected
    'mem_36b97d5054554f509ce6643427fc5c6a',  # p0 infra bugs fixed
    'mem_bad8da4f3cd448139dc24f65d3ff44cb',  # lifecycle bug fixed
    'mem_5505e33aaf794dc5a0ecc5c580ed1298',  # embedder e5-small
    'mem_fe72ce83c0c54746bad522821c98737c',  # entity RRF negative
    'mem_ae25c7c2829e45478c87e7fd3a52055d',  # entity graph negative
    'mem_17de41c26cf34d0b8a123068e37abbba',  # hotpot fake regression
    'mem_205cb0884a8d4ccd998845363a1c718c',  # ppr-lte refuted
    'mem_cd94dd1d45a246a290496718d5fef195',  # vision vs reality
    'mem_9109e51948a64f2ba758953d12a22cc5',  # research session final
]

for mid in important_ids:
    cursor.execute('SELECT memory_id, agent, label, block_type, priority, content FROM memories WHERE memory_id = ?', (mid,))
    row = cursor.fetchone()
    if row:
        print(f"\n{'='*80}")
        print(f"ID: {row[0]}")
        print(f"Agent: {row[1]}")
        print(f"Label: {row[2]}")
        print(f"Type: {row[3]} | Priority: {row[4]}")
        print(f"{'='*80}")
        content = row[5] if row[5] else '(no content)'
        print(content[:3000])
        if len(content) > 3000:
            print(f"\n... [TRUNCATED - {len(content)} total chars]")

conn.close()
