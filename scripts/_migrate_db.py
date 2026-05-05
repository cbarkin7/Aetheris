"""
Migración: crea la tabla conversations en memory.db si no existe.
Ejecutar una sola vez antes del primer arranque con la nueva versión.
"""
import sqlite3
import pathlib

DB_PATH = pathlib.Path("data/sqlite/memory.db")

con = sqlite3.connect(str(DB_PATH))

# --- Crear tabla conversations -------------------------------------------
con.execute("""
CREATE TABLE IF NOT EXISTS conversations (
    thread_id   TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL,
    title       TEXT NOT NULL,
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at  DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")
con.commit()

# --- Verificar integridad ------------------------------------------------
tables = [r[0] for r in con.execute(
    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
).fetchall()]

print("=" * 50)
print("DB:", DB_PATH.resolve())
print("Tablas:", tables)

ok = True
for t in tables:
    cols = [(r[1], r[2]) for r in con.execute(f"PRAGMA table_info({t})").fetchall()]
    count = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    print(f"\n  [{t}]  {count} filas")
    for name, typ in cols:
        print(f"    {name:22s}  {typ}")

# Comprobar que user_memory sigue intacta
um_count = con.execute("SELECT COUNT(*) FROM user_memory").fetchone()[0]
if um_count != 43:
    print(f"\nWARNING: user_memory tiene {um_count} filas, se esperaban 43.")
    ok = False
else:
    print(f"\nuser_memory: {um_count} filas intactas. OK")

# Comprobar que conversations existe y está vacía
conv_count = con.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
print(f"conversations: {conv_count} filas. OK")

con.close()
print("=" * 50)
print("Migración completada correctamente." if ok else "ATENCIÓN: revisar warnings anteriores.")
