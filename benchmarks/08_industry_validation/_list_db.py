import sqlite3

conn = sqlite3.connect(r'C:\Users\So-i-learn-3D\.config\mathir\data\projects\MATHIR\mathir.db')
cursor = conn.cursor()

# Get all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()
print('Tables:', [t[0] for t in tables])

# Count memories
cursor.execute('SELECT COUNT(*) FROM memories')
total = cursor.fetchone()[0]
print(f'Total memories: {total}')

# List all memories with key info
cursor.execute('SELECT memory_id, agent, label, block_type, priority, created_at FROM memories ORDER BY created_at DESC')
rows = cursor.fetchall()
print()
print(f"{'ID':<42} {'Agent':<15} {'Label':<55} {'Type':<12} {'Pri':<4} {'Created'}")
print('-' * 170)
for row in rows:
    print(f"{row[0]:<42} {row[1]:<15} {(row[2] or ''):<55} {(row[3] or ''):<12} {row[4] or 0:<4} {(row[5] or '')[:19]}")
conn.close()
