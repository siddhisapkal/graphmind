from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


def _db_path() -> Path:
    return Path(__file__).resolve().parent / "relations.sqlite3"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_db_path()))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS relation_metadata (
            relation_key TEXT PRIMARY KEY,
            relation TEXT NOT NULL,
            entity_type TEXT NOT NULL,
            family TEXT NOT NULL,
            polarity TEXT NOT NULL,
            section_tags_json TEXT NOT NULL,
            strength REAL NOT NULL,
            source TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def get_relation_metadata(*, relation_key: str) -> dict[str, Any] | None:
    conn = _connect()
    try:
        row = conn.execute(
            """
            SELECT relation, entity_type, family, polarity, section_tags_json, strength, source, updated_at
            FROM relation_metadata
            WHERE relation_key = ?
            """,
            (relation_key,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    return {
        "relation": row[0],
        "entity_type": row[1],
        "family": row[2],
        "polarity": row[3],
        "section_tags": json.loads(row[4] or "[]"),
        "strength": float(row[5]),
        "source": row[6],
        "updated_at": row[7],
    }


def upsert_relation_metadata(
    *,
    relation_key: str,
    relation: str,
    entity_type: str,
    family: str,
    polarity: str,
    section_tags: list[str],
    strength: float,
    source: str,
    updated_at: str,
) -> None:
    conn = _connect()
    try:
        conn.execute(
            """
            INSERT INTO relation_metadata (
                relation_key, relation, entity_type, family, polarity,
                section_tags_json, strength, source, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(relation_key) DO UPDATE SET
                family = excluded.family,
                polarity = excluded.polarity,
                section_tags_json = excluded.section_tags_json,
                strength = excluded.strength,
                source = excluded.source,
                updated_at = excluded.updated_at
            """,
            (
                relation_key,
                relation,
                entity_type,
                family,
                polarity,
                json.dumps(section_tags),
                float(strength),
                source,
                updated_at,
            ),
        )
        conn.commit()
    finally:
        conn.close()
