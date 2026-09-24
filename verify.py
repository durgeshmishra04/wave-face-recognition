import os
import sqlite3
import hashlib
import psycopg
from dotenv import load_dotenv
load_dotenv()

SQLITE_DB = "known_faces.db"


def sha256(data):
    if data is None:
        return None
    return hashlib.sha256(bytes(data)).hexdigest()


def get_pg_connection():
    return psycopg.connect(
        host=os.getenv("PGHOST", "localhost"),
        port=int(os.getenv("PGPORT", "5432")),
        dbname=os.environ["PGDATABASE"],
        user=os.environ["PGUSER"],
        password=os.environ["PGPASSWORD"],
    )


sqlite_conn = sqlite3.connect(SQLITE_DB)
sqlite_cur = sqlite_conn.cursor()

pg_conn = get_pg_connection()
pg_cur = pg_conn.cursor()

print("\n=== PER-ROW KNOWN EMBEDDING VERIFICATION ===\n")

sqlite_rows = sqlite_cur.execute("""
    SELECT person, embedding, image_paths, fingerprint, updated_at
    FROM known_embeddings
    ORDER BY person
""").fetchall()

pg_cur.execute("""
    SELECT person, embedding, image_paths, fingerprint, updated_at
    FROM known_embeddings
    ORDER BY person
""")

pg_rows = pg_cur.fetchall()

print(f"SQLite rows     : {len(sqlite_rows)}")
print(f"PostgreSQL rows : {len(pg_rows)}")
print()

sqlite_dict = {
    row[0]: row
    for row in sqlite_rows
}

pg_dict = {
    row[0]: row
    for row in pg_rows
}

all_people = sorted(set(sqlite_dict) | set(pg_dict))

errors = 0

for person in all_people:

    s = sqlite_dict.get(person)
    p = pg_dict.get(person)

    if s is None:
        print(f"❌ Missing in SQLite: {person}")
        errors += 1
        continue

    if p is None:
        print(f"❌ Missing in PostgreSQL: {person}")
        errors += 1
        continue

    # Compare embedding bytes
    sqlite_embedding_hash = sha256(s[1])
    pg_embedding_hash = sha256(p[1])

    if sqlite_embedding_hash != pg_embedding_hash:
        print(f"❌ EMBEDDING MISMATCH: {person}")
        print(f"   SQLite     : {sqlite_embedding_hash}")
        print(f"   PostgreSQL : {pg_embedding_hash}")
        errors += 1
        continue

    # Compare image paths
    if s[2] != p[2]:
        print(f"❌ IMAGE PATH MISMATCH: {person}")
        errors += 1
        continue

    # Compare fingerprint
    if s[3] != p[3]:
        print(f"❌ FINGERPRINT MISMATCH: {person}")
        errors += 1
        continue

    # Compare timestamp
    if s[4] != p[4]:
        print(f"❌ TIMESTAMP MISMATCH: {person}")
        print(f"   SQLite     : {s[4]}")
        print(f"   PostgreSQL : {p[4]}")
        errors += 1
        continue

    print(f"✓ {person}")


print("\n======================================")

if errors == 0:
    print("SUCCESS")
    print("Every known_embeddings record matches exactly.")
else:
    print(f"FAILED: {errors} mismatched records found.")

print("======================================\n")

pg_cur.close()
pg_conn.close()
sqlite_cur.close()
sqlite_conn.close()