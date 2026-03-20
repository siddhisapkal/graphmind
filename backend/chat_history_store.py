from __future__ import annotations

import sqlite3
from pathlib import Path


def _db_path() -> Path:
    return Path(__file__).resolve().parent / "chat_history.sqlite3"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_db_path()))
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, id)"
    )
    return conn


def ensure_conversation(*, conversation_id: str, user_id: str) -> None:
    conn = _connect()
    try:
        conn.execute(
            """
            INSERT OR IGNORE INTO conversations (id, user_id)
            VALUES (?, ?)
            """,
            (conversation_id, user_id),
        )
        conn.commit()
    finally:
        conn.close()


def save_message(*, conversation_id: str, user_id: str, role: str, content: str) -> None:
    conn = _connect()
    try:
        conn.execute(
            """
            INSERT OR IGNORE INTO conversations (id, user_id)
            VALUES (?, ?)
            """,
            (conversation_id, user_id),
        )
        conn.execute(
            """
            INSERT INTO messages (conversation_id, role, content)
            VALUES (?, ?, ?)
            """,
            (conversation_id, role, content),
        )
        conn.commit()
    finally:
        conn.close()


def get_chat_history(*, conversation_id: str, user_id: str, limit: int | None = None) -> list[dict[str, object]]:
    conn = _connect()
    try:
        convo = conn.execute(
            "SELECT id FROM conversations WHERE id = ? AND user_id = ?",
            (conversation_id, user_id),
        ).fetchone()
        if convo is None:
            return []

        query = """
            SELECT id, conversation_id, role, content, timestamp
            FROM messages
            WHERE conversation_id = ?
            ORDER BY id ASC
        """
        params: tuple[object, ...] = (conversation_id,)
        if limit is not None:
            query = """
                SELECT * FROM (
                    SELECT id, conversation_id, role, content, timestamp
                    FROM messages
                    WHERE conversation_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                ) ordered
                ORDER BY id ASC
            """
            params = (conversation_id, max(1, min(limit, 200)))
        rows = conn.execute(query, params).fetchall()
        return [
            {
                "id": int(row["id"]),
                "conversation_id": str(row["conversation_id"]),
                "role": str(row["role"]),
                "content": str(row["content"]),
                "timestamp": str(row["timestamp"]),
            }
            for row in rows
        ]
    finally:
        conn.close()


def list_conversations(*, user_id: str, limit: int = 50) -> list[dict[str, object]]:
    conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT
                c.id,
                c.created_at,
                MAX(m.timestamp) AS last_message_at,
                COALESCE(
                    (
                        SELECT m2.content
                        FROM messages m2
                        WHERE m2.conversation_id = c.id
                        ORDER BY m2.id DESC
                        LIMIT 1
                    ),
                    ''
                ) AS preview
            FROM conversations c
            LEFT JOIN messages m ON m.conversation_id = c.id
            WHERE c.user_id = ?
            GROUP BY c.id, c.created_at
            ORDER BY COALESCE(MAX(m.timestamp), c.created_at) DESC, c.id DESC
            LIMIT ?
            """,
            (user_id, max(1, min(limit, 200))),
        ).fetchall()
        return [
            {
                "conversation_id": str(row["id"]),
                "created_at": str(row["created_at"]),
                "last_message_at": str(row["last_message_at"] or row["created_at"]),
                "preview": str(row["preview"] or ""),
            }
            for row in rows
        ]
    finally:
        conn.close()
