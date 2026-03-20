from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any
from uuid import uuid4


def _db_path() -> Path:
    return Path(__file__).resolve().parent / "events.sqlite3"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_db_path()))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS raw_events (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            conversation_id TEXT,
            source_type TEXT NOT NULL,
            source_ref TEXT,
            role TEXT,
            content TEXT NOT NULL,
            metadata_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS promotion_log (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            source_event_id TEXT,
            entity TEXT NOT NULL,
            entity_type TEXT NOT NULL,
            relation TEXT NOT NULL,
            confidence REAL NOT NULL,
            status TEXT NOT NULL,
            metadata_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_raw_events_user ON raw_events(user_id)")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_raw_events_convo ON raw_events(conversation_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_promotion_log_user ON promotion_log(user_id)"
    )
    return conn


def log_raw_event(
    *,
    user_id: str,
    source_type: str,
    content: str,
    created_at: str,
    conversation_id: str | None = None,
    source_ref: str | None = None,
    role: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    event_id = str(uuid4())
    conn = _connect()
    try:
        conn.execute(
            """
            INSERT INTO raw_events (
                id, user_id, conversation_id, source_type, source_ref,
                role, content, metadata_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                user_id,
                conversation_id,
                source_type,
                source_ref,
                role,
                content,
                json.dumps(metadata or {}),
                created_at,
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return event_id


def log_promotions(
    *,
    user_id: str,
    source_event_id: str | None,
    created_at: str,
    raw_signals: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    promoted_lookup = {
        (
            str(item.get("relation") or ""),
            str(item.get("entity_type") or ""),
            str(item.get("entity") or ""),
        ): item
        for item in list(summary.get("promoted_items") or [])
    }
    conn = _connect()
    try:
        for signal in raw_signals:
            relation = str(signal.get("relation") or "")
            entity_type = str(signal.get("entity_type") or "Entity")
            entity = str(signal.get("entity") or "")
            status = (
                "promoted"
                if (relation, entity_type, entity) in promoted_lookup
                else "pending"
            )
            conn.execute(
                """
                INSERT INTO promotion_log (
                    id, user_id, source_event_id, entity, entity_type, relation,
                    confidence, status, metadata_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid4()),
                    user_id,
                    source_event_id,
                    entity,
                    entity_type,
                    relation,
                    float(signal.get("confidence") or 0.0),
                    status,
                    json.dumps(
                        {
                            "source": signal.get("source"),
                            "linked_to_action": bool(signal.get("linked_to_action")),
                            "raw_text": signal.get("raw_text"),
                        }
                    ),
                    created_at,
                ),
            )
        conn.commit()
    finally:
        conn.close()


def recent_raw_events(*, user_id: str, limit: int = 20) -> list[dict[str, Any]]:
    conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT id, user_id, conversation_id, source_type, source_ref, role, content, metadata_json, created_at
            FROM raw_events
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (user_id, max(1, min(limit, 100))),
        ).fetchall()
        items: list[dict[str, Any]] = []
        for row in rows:
            items.append(
                {
                    "id": row[0],
                    "user_id": row[1],
                    "conversation_id": row[2],
                    "source_type": row[3],
                    "source_ref": row[4],
                    "role": row[5],
                    "content": row[6],
                    "metadata": json.loads(row[7] or "{}"),
                    "created_at": row[8],
                }
            )
        return items
    finally:
        conn.close()


def delete_user_events(*, user_id: str) -> dict[str, int]:
    conn = _connect()
    try:
        raw_deleted = conn.execute(
            "DELETE FROM raw_events WHERE user_id = ?",
            (user_id,),
        ).rowcount
        promotion_deleted = conn.execute(
            "DELETE FROM promotion_log WHERE user_id = ?",
            (user_id,),
        ).rowcount
        conn.commit()
        return {
            "raw_events_deleted": int(raw_deleted or 0),
            "promotion_logs_deleted": int(promotion_deleted or 0),
        }
    finally:
        conn.close()
