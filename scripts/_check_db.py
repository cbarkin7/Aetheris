import sqlite3
import pathlib

db_path = pathlib.Path("data/sqlite/memory.db")
con = sqlite3.connect(str(db_path))

tables = [r[0] for r in con.execute(
    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
).fetchall()]
print("Tablas encontradas:", tables)

for t in tables:
    cols = [(r[1], r[2]) for r in con.execute(f"PRAGMA table_info({t})").fetchall()]
    count = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    print(f"\n  [{t}]  ({count} filas)")
    for name, typ in cols:
        print(f"    {name:20s}  {typ}")

con.close()
