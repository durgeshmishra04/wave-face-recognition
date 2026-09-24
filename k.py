import os
import sqlite3
import hashlib

import psycopg
from dotenv import load_dotenv
from psycopg.rows import dict_row

load_dotenv()

SQLITE_DB = "known_faces.db"

TABLES = [
    "auth_accounts",
    "detection_events",
    "detection_records",
    "fcm_device_tokens",
    "known_embeddings",
    "registered_face_embeddings",
    "registered_persons",
]


def normalize_value(value):
    """
    Convert database values into deterministic comparable values.
    """
    if isinstance(value, bytes):
        return ("BLOB", len(value), hashlib.sha256(value).hexdigest())

    if value is None:
        return None

    return value


def sqlite_connection():
    conn = sqlite3.connect(SQLITE_DB)
    conn.row_factory = sqlite3.Row
    return conn


def postgres_connection():
    return psycopg.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        row_factory=dict_row,
    )


def get_columns_sqlite(conn, table):
    rows = conn.execute(
        f'PRAGMA table_info("{table}")'
    ).fetchall()

    return [row["name"] for row in rows]


def get_columns_postgres(conn, table):
    rows = conn.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = %s
        ORDER BY ordinal_position
        """,
        (table,),
    ).fetchall()

    return [row["column_name"] for row in rows]


def get_primary_key_sqlite(conn, table):
    rows = conn.execute(
        f'PRAGMA table_info("{table}")'
    ).fetchall()

    pk_columns = [
        (row["pk"], row["name"])
        for row in rows
        if row["pk"] > 0
    ]

    pk_columns.sort()

    return [name for _, name in pk_columns]


def get_primary_key_postgres(conn, table):
    rows = conn.execute(
        """
        SELECT kcu.column_name
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu
          ON tc.constraint_name = kcu.constraint_name
         AND tc.table_schema = kcu.table_schema
        WHERE tc.table_schema = 'public'
          AND tc.table_name = %s
          AND tc.constraint_type = 'PRIMARY KEY'
        ORDER BY kcu.ordinal_position
        """,
        (table,),
    ).fetchall()

    return [row["column_name"] for row in rows]


def fetch_sqlite_rows(conn, table, columns, order_columns):
    column_sql = ", ".join(f'"{c}"' for c in columns)

    order_sql = ""
    if order_columns:
        order_sql = " ORDER BY " + ", ".join(
            f'"{c}"' for c in order_columns
        )

    return conn.execute(
        f'SELECT {column_sql} FROM "{table}"{order_sql}'
    ).fetchall()


def fetch_postgres_rows(conn, table, columns, order_columns):
    column_sql = ", ".join(
        '"' + c.replace('"', '""') + '"'
        for c in columns
    )

    order_sql = ""
    if order_columns:
        order_sql = " ORDER BY " + ", ".join(
            '"' + c.replace('"', '""') + '"'
            for c in order_columns
        )

    return conn.execute(
        f'SELECT {column_sql} FROM "{table}"{order_sql}'
    ).fetchall()


def compare_table(sqlite_conn, pg_conn, table):
    print()
    print("=" * 70)
    print(f"TABLE: {table}")
    print("=" * 70)

    sqlite_columns = get_columns_sqlite(sqlite_conn, table)
    pg_columns = get_columns_postgres(pg_conn, table)

    if sqlite_columns != pg_columns:
        print("COLUMN STRUCTURE: MISMATCH")
        print("SQLite :", sqlite_columns)
        print("Postgres:", pg_columns)
        return False

    print("Columns: OK")

    sqlite_pk = get_primary_key_sqlite(sqlite_conn, table)
    pg_pk = get_primary_key_postgres(pg_conn, table)

    print("SQLite PK :", sqlite_pk)
    print("Postgres PK:", pg_pk)

    if sqlite_pk != pg_pk:
        print("PRIMARY KEY: MISMATCH")
        return False

    print("Primary key: OK")

    columns = sqlite_columns

    # Prefer primary key for deterministic ordering.
    # If a table has no PK, use all columns.
    order_columns = sqlite_pk if sqlite_pk else columns

    sqlite_rows = fetch_sqlite_rows(
        sqlite_conn,
        table,
        columns,
        order_columns,
    )

    pg_rows = fetch_postgres_rows(
        pg_conn,
        table,
        columns,
        order_columns,
    )

    print(f"SQLite rows    : {len(sqlite_rows)}")
    print(f"PostgreSQL rows: {len(pg_rows)}")

    if len(sqlite_rows) != len(pg_rows):
        print("ROW COUNT: MISMATCH")
        return False

    for index, (sqlite_row, pg_row) in enumerate(
        zip(sqlite_rows, pg_rows),
        start=1,
    ):
        for column in columns:

            sqlite_value = normalize_value(
                sqlite_row[column]
            )

            pg_value = normalize_value(
                pg_row[column]
            )

            if sqlite_value != pg_value:
                print()
                print("DATA MISMATCH")
                print(f"Row      : {index}")
                print(f"Column   : {column}")
                print(f"SQLite   : {sqlite_value}")
                print(f"Postgres : {pg_value}")
                return False

    print("All column values: OK")
    print("TABLE STATUS: PASS")

    return True


def verify_foreign_keys(sqlite_conn, pg_conn):
    print()
    print("=" * 70)
    print("FOREIGN KEY VERIFICATION")
    print("=" * 70)

    sqlite_fk_count = 0
    pg_fk_count = 0

    for table in TABLES:

        sqlite_rows = sqlite_conn.execute(
            f'PRAGMA foreign_key_list("{table}")'
        ).fetchall()

        sqlite_fk_count += len(sqlite_rows)

        pg_rows = pg_conn.execute(
            """
            SELECT
                tc.table_name,
                kcu.column_name,
                ccu.table_name AS foreign_table_name,
                ccu.column_name AS foreign_column_name
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
              ON tc.constraint_name = kcu.constraint_name
             AND tc.table_schema = kcu.table_schema
            JOIN information_schema.constraint_column_usage ccu
              ON tc.constraint_name = ccu.constraint_name
             AND tc.table_schema = ccu.table_schema
            WHERE tc.constraint_type = 'FOREIGN KEY'
              AND tc.table_schema = 'public'
              AND tc.table_name = %s
            """,
            (table,),
        ).fetchall()

        pg_fk_count += len(pg_rows)

    print(f"SQLite foreign keys    : {sqlite_fk_count}")
    print(f"PostgreSQL foreign keys: {pg_fk_count}")

    if sqlite_fk_count == pg_fk_count:
        print("Foreign key count: OK")
        return True

    print("Foreign key count: MISMATCH")
    return False


def main():
    print()
    print("=" * 70)
    print("FULL SQLITE → POSTGRESQL MIGRATION VERIFICATION")
    print("=" * 70)

    sqlite_conn = sqlite_connection()
    pg_conn = postgres_connection()

    print()
    print("Connected to both databases.")

    all_ok = True

    try:
        for table in TABLES:
            try:
                result = compare_table(
                    sqlite_conn,
                    pg_conn,
                    table,
                )

                if not result:
                    all_ok = False

            except Exception as exc:
                print()
                print(f"ERROR verifying {table}:")
                print(exc)
                all_ok = False

        fk_result = verify_foreign_keys(
            sqlite_conn,
            pg_conn,
        )

        if not fk_result:
            all_ok = False

    finally:
        sqlite_conn.close()
        pg_conn.close()

    print()
    print("=" * 70)

    if all_ok:
        print("FINAL RESULT: PASS")
        print("SQLite and PostgreSQL data match.")
        print("IDs, columns, values and BLOBs were verified.")
        print("The VMS can now proceed to PostgreSQL connection testing.")
    else:
        print("FINAL RESULT: FAILED")
        print("Do NOT switch the VMS to PostgreSQL yet.")

    print("=" * 70)


if __name__ == "__main__":
    main()