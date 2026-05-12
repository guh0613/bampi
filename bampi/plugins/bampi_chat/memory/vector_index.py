from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass


VECTOR_TABLE = "archive_embedding_vec"
VECTOR_META_TABLE = "archive_embedding_vec_meta"


class SqliteVecUnavailableError(RuntimeError):
    """Raised when sqlite-vec cannot be loaded for embedding search."""


@dataclass(slots=True)
class SqliteVecArchiveIndex:
    provider: str
    model: str

    def load_connection(self, conn: sqlite3.Connection) -> None:
        load_sqlite_vec(conn)

    def initialize_connection(self, conn: sqlite3.Connection) -> None:
        self.load_connection(conn)
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {VECTOR_META_TABLE} (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )

    def ensure_ready(self, conn: sqlite3.Connection, *, dimension: int) -> bool:
        dimension = int(dimension)
        if dimension <= 0:
            raise ValueError("embedding dimension must be positive")
        self.initialize_connection(conn)
        state = self._read_state(conn)
        table_exists = self._table_exists(conn)
        expected_state = {
            "provider": self.provider,
            "model": self.model,
            "dimension": str(dimension),
        }
        if table_exists and state == expected_state:
            return False

        if table_exists:
            conn.execute(f"DROP TABLE {VECTOR_TABLE}")
        self._create_vector_table(conn, dimension=dimension)
        self._write_state(conn, expected_state)
        self._backfill_from_archive_embeddings(conn, dimension=dimension)
        return True

    def upsert(
        self,
        conn: sqlite3.Connection,
        *,
        archive_id: int,
        group_id: str,
        vector: list[float],
    ) -> None:
        dimension = len(vector)
        self.ensure_ready(conn, dimension=dimension)
        conn.execute(f"DELETE FROM {VECTOR_TABLE} WHERE archive_id = ?", (int(archive_id),))
        conn.execute(
            f"""
            INSERT INTO {VECTOR_TABLE}(archive_id, group_id, embedding)
            VALUES (?, ?, ?)
            """,
            (int(archive_id), str(group_id), serialize_float32(vector)),
        )

    def delete(self, conn: sqlite3.Connection, *, archive_id: int) -> None:
        self.initialize_connection(conn)
        if not self._table_exists(conn):
            return
        conn.execute(f"DELETE FROM {VECTOR_TABLE} WHERE archive_id = ?", (int(archive_id),))

    def search(
        self,
        conn: sqlite3.Connection,
        *,
        group_id: str,
        query_vector: list[float],
        limit: int,
    ) -> list[tuple[int, float]]:
        changed = self.ensure_ready(conn, dimension=len(query_vector))
        if changed:
            conn.commit()
        rows = conn.execute(
            f"""
            SELECT archive_id, distance
            FROM {VECTOR_TABLE}
            WHERE embedding MATCH ?
              AND k = ?
              AND group_id = ?
            ORDER BY distance
            """,
            (serialize_float32(query_vector), max(1, int(limit)), str(group_id)),
        ).fetchall()
        return [
            (int(row["archive_id"]), 1.0 - float(row["distance"]))
            for row in rows
        ]

    def _create_vector_table(self, conn: sqlite3.Connection, *, dimension: int) -> None:
        conn.execute(
            f"""
            CREATE VIRTUAL TABLE {VECTOR_TABLE} USING vec0(
                archive_id INTEGER PRIMARY KEY,
                group_id TEXT partition key,
                embedding FLOAT[{int(dimension)}] distance_metric=cosine
            )
            """
        )

    def _backfill_from_archive_embeddings(
        self,
        conn: sqlite3.Connection,
        *,
        dimension: int,
    ) -> None:
        for row in conn.execute(
            """
            SELECT archive_id, group_id, vector
            FROM archive_embeddings
            WHERE provider = ?
              AND model = ?
              AND dimension = ?
            ORDER BY archive_id
            """,
            (self.provider, self.model, int(dimension)),
        ):
            vector = _loads_vector(row["vector"], dimension=dimension)
            if vector is None or not any(vector):
                continue
            conn.execute(
                f"""
                INSERT INTO {VECTOR_TABLE}(archive_id, group_id, embedding)
                VALUES (?, ?, ?)
                """,
                (
                    int(row["archive_id"]),
                    str(row["group_id"]),
                    serialize_float32(vector),
                ),
            )

    def _read_state(self, conn: sqlite3.Connection) -> dict[str, str]:
        return {
            str(row["key"]): str(row["value"])
            for row in conn.execute(f"SELECT key, value FROM {VECTOR_META_TABLE}")
        }

    def _write_state(self, conn: sqlite3.Connection, state: dict[str, str]) -> None:
        conn.execute(f"DELETE FROM {VECTOR_META_TABLE}")
        conn.executemany(
            f"INSERT INTO {VECTOR_META_TABLE}(key, value) VALUES (?, ?)",
            tuple(state.items()),
        )

    def _table_exists(self, conn: sqlite3.Connection) -> bool:
        row = conn.execute(
            """
            SELECT 1
            FROM sqlite_master
            WHERE type = 'table' AND name = ?
            """,
            (VECTOR_TABLE,),
        ).fetchone()
        return row is not None


def load_sqlite_vec(conn: sqlite3.Connection) -> None:
    try:
        conn.execute("SELECT vec_version()").fetchone()
        return
    except sqlite3.OperationalError:
        pass

    try:
        import sqlite_vec
    except Exception as exc:  # pragma: no cover - exercised only without dependency
        raise SqliteVecUnavailableError(
            "sqlite-vec is required when memory embeddings are enabled"
        ) from exc

    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
    except AttributeError as exc:
        raise SqliteVecUnavailableError(
            "this Python sqlite3 build does not support loadable extensions; "
            "use a Python/SQLite build that can load sqlite-vec"
        ) from exc
    except Exception as exc:
        raise SqliteVecUnavailableError("failed to load sqlite-vec") from exc
    finally:
        try:
            conn.enable_load_extension(False)
        except Exception:
            pass


def serialize_float32(vector: list[float]) -> bytes:
    from sqlite_vec import serialize_float32 as _serialize_float32

    return _serialize_float32([float(value) for value in vector])


def _loads_vector(value: object, *, dimension: int) -> list[float] | None:
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError):
        return None
    if not isinstance(parsed, list) or len(parsed) != dimension:
        return None
    vector: list[float] = []
    for item in parsed:
        if not isinstance(item, (int, float)):
            return None
        vector.append(float(item))
    return vector
