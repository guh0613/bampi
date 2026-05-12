from __future__ import annotations

import sqlite3


CURRENT_SCHEMA_VERSION = 2


SCHEMA_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS conversation_archives (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        group_id       TEXT NOT NULL,
        started_at     TEXT NOT NULL,
        ended_at       TEXT NOT NULL,
        participants   TEXT NOT NULL DEFAULT '[]',
        title          TEXT NOT NULL DEFAULT '',
        summary        TEXT NOT NULL DEFAULT '',
        keywords       TEXT NOT NULL DEFAULT '[]',
        message_count  INTEGER NOT NULL DEFAULT 0,
        created_at     TEXT NOT NULL
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_archives_group_time
        ON conversation_archives(group_id, ended_at DESC)
    """,
    """
    CREATE TABLE IF NOT EXISTS messages (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        archive_id  INTEGER NOT NULL REFERENCES conversation_archives(id) ON DELETE CASCADE,
        group_id    TEXT NOT NULL,
        user_id     TEXT NOT NULL DEFAULT '',
        nickname    TEXT NOT NULL DEFAULT '',
        role        TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
        content     TEXT NOT NULL,
        timestamp   TEXT NOT NULL
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_messages_archive
        ON messages(archive_id)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_messages_group_user
        ON messages(group_id, user_id)
    """,
    """
    CREATE TABLE IF NOT EXISTS tool_events (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        archive_id     INTEGER NOT NULL REFERENCES conversation_archives(id) ON DELETE CASCADE,
        group_id       TEXT NOT NULL,
        tool_call_id   TEXT NOT NULL DEFAULT '',
        tool_name      TEXT NOT NULL DEFAULT '',
        arguments_text TEXT NOT NULL DEFAULT '',
        result_preview TEXT NOT NULL DEFAULT '',
        result_full    TEXT NOT NULL DEFAULT '',
        is_error       INTEGER NOT NULL DEFAULT 0,
        timestamp      TEXT NOT NULL
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_tool_events_archive
        ON tool_events(archive_id)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_tool_events_group_tool
        ON tool_events(group_id, tool_name)
    """,
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS archive_fts USING fts5(
        archive_id UNINDEXED,
        group_id UNINDEXED,
        search_text,
        tokenize='unicode61'
    )
    """,
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS message_fts USING fts5(
        message_id UNINDEXED,
        archive_id UNINDEXED,
        group_id UNINDEXED,
        search_text,
        tokenize='unicode61'
    )
    """,
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS tool_event_fts USING fts5(
        tool_event_id UNINDEXED,
        archive_id UNINDEXED,
        group_id UNINDEXED,
        search_text,
        tokenize='unicode61'
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS archive_embeddings (
        archive_id INTEGER PRIMARY KEY REFERENCES conversation_archives(id) ON DELETE CASCADE,
        group_id   TEXT NOT NULL,
        provider   TEXT NOT NULL DEFAULT '',
        model      TEXT NOT NULL DEFAULT '',
        dimension  INTEGER NOT NULL DEFAULT 0,
        vector     TEXT NOT NULL DEFAULT '[]',
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_archive_embeddings_group
        ON archive_embeddings(group_id)
    """,
    """
    CREATE TABLE IF NOT EXISTS archive_embedding_vec_meta (
        key   TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS user_profiles (
        user_id          TEXT NOT NULL,
        group_id         TEXT NOT NULL,
        nickname         TEXT NOT NULL DEFAULT '',
        profile          TEXT NOT NULL DEFAULT '',
        version          INTEGER NOT NULL DEFAULT 1,
        pending_sessions INTEGER NOT NULL DEFAULT 0,
        updated_at       TEXT NOT NULL,
        last_active_at   TEXT NOT NULL DEFAULT '',
        PRIMARY KEY (user_id, group_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS profile_edits (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id       TEXT NOT NULL,
        group_id      TEXT NOT NULL,
        edit_type     TEXT NOT NULL CHECK(edit_type IN ('add', 'update', 'delete')),
        content       TEXT NOT NULL,
        created_at    TEXT NOT NULL,
        consolidated  INTEGER NOT NULL DEFAULT 0
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_profile_edits_group_user
        ON profile_edits(group_id, user_id, consolidated, created_at)
    """,
]


MIGRATIONS: dict[int, list[str]] = {
    2: [],
}


def initialize_memory_schema(conn: sqlite3.Connection) -> None:
    current_version = _schema_version(conn)
    if current_version > CURRENT_SCHEMA_VERSION:
        raise RuntimeError(
            "memory database schema is newer than this code supports: "
            f"{current_version} > {CURRENT_SCHEMA_VERSION}"
        )

    for statement in SCHEMA_STATEMENTS:
        conn.execute(statement)

    if current_version == 0:
        current_version = 1
        _set_schema_version(conn, current_version)

    for version in range(current_version + 1, CURRENT_SCHEMA_VERSION + 1):
        for statement in MIGRATIONS.get(version, []):
            conn.execute(statement)
        _set_schema_version(conn, version)


def _schema_version(conn: sqlite3.Connection) -> int:
    row = conn.execute("PRAGMA user_version").fetchone()
    return int(row[0]) if row is not None else 0


def _set_schema_version(conn: sqlite3.Connection, version: int) -> None:
    conn.execute(f"PRAGMA user_version = {int(version)}")
