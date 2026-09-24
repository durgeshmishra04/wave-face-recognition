# SQLite -> PostgreSQL migration test

Put these files together:

    migration_project/
    ├── known_faces.db
    ├── migrate_sqlite_to_postgres.py
    └── requirements.txt

Install:

    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt

Set PostgreSQL connection variables in CMD:

    set PGHOST=localhost
    set PGPORT=5432
    set PGDATABASE=vms_test
    set PGUSER=postgres
    set PGPASSWORD=YOUR_POSTGRES_PASSWORD

Run:

    python migrate_sqlite_to_postgres.py

The script creates the VMS tables, copies all rows and BLOB embeddings, preserves image paths/URLs, resets identity sequences, and performs row-count, BLOB-byte, and exact-data hash verification.

IMPORTANT:
- It does not modify the SQLite database.
- It does not move image files.
- Run it first against the empty `vms_test` database.
- Only use `--truncate` when you intentionally want to replace data already in the target database.
