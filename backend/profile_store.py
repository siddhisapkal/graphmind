from __future__ import annotations

import sqlite3
from pathlib import Path


def _db_path() -> Path:
    return Path(__file__).resolve().parent / "profile_cache.sqlite3"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_db_path()))
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS skill_profile (
            user_id TEXT NOT NULL,
            entity TEXT NOT NULL,
            entity_key TEXT NOT NULL,
            entity_type TEXT NOT NULL,
            score REAL NOT NULL DEFAULT 0,
            strength_score REAL NOT NULL DEFAULT 0,
            weakness_score REAL NOT NULL DEFAULT 0,
            improving_score REAL NOT NULL DEFAULT 0,
            last_signal TEXT NOT NULL DEFAULT '',
            evidence_count INTEGER NOT NULL DEFAULT 0,
            last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (user_id, entity_key)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS skill_profile_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            entity TEXT NOT NULL,
            entity_key TEXT NOT NULL,
            entity_type TEXT NOT NULL,
            signal_type TEXT NOT NULL,
            delta REAL NOT NULL,
            rationale TEXT NOT NULL DEFAULT '',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    return conn


def upsert_profile_observations(*, user_id: str, observations: list[dict[str, object]]) -> None:
    if not observations:
        return
    conn = _connect()
    try:
        for item in observations:
            entity = str(item.get("entity") or "").strip()
            entity_key = str(item.get("entity_key") or "").strip()
            entity_type = str(item.get("entity_type") or "Skill").strip() or "Skill"
            signal_type = str(item.get("signal_type") or "neutral").strip().lower()
            rationale = str(item.get("rationale") or "").strip()
            try:
                delta = float(item.get("delta") or 0.0)
            except (TypeError, ValueError):
                delta = 0.0
            if not entity or not entity_key:
                continue
            if signal_type == "neutral":
                continue

            existing = conn.execute(
                """
                SELECT score, strength_score, weakness_score, improving_score, evidence_count
                FROM skill_profile
                WHERE user_id = ? AND entity_key = ?
                """,
                (user_id, entity_key),
            ).fetchone()

            score = float(existing["score"]) if existing else 0.0
            strength_score = float(existing["strength_score"]) if existing else 0.0
            weakness_score = float(existing["weakness_score"]) if existing else 0.0
            improving_score = float(existing["improving_score"]) if existing else 0.0
            evidence_count = int(existing["evidence_count"]) if existing else 0

            if signal_type == "strength":
                score += abs(delta) or 0.8
                strength_score += abs(delta) or 0.8
            elif signal_type == "weakness":
                score -= abs(delta) or 0.8
                weakness_score += abs(delta) or 0.8
            elif signal_type == "improving":
                score += abs(delta) or 0.45
                improving_score += abs(delta) or 0.45
                weakness_score = max(0.0, weakness_score - min(abs(delta) or 0.45, 0.6))
            else:
                score += delta

            conn.execute(
                """
                INSERT INTO skill_profile (
                    user_id, entity, entity_key, entity_type, score,
                    strength_score, weakness_score, improving_score, last_signal, evidence_count, last_updated
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(user_id, entity_key) DO UPDATE SET
                    entity = excluded.entity,
                    entity_type = excluded.entity_type,
                    score = excluded.score,
                    strength_score = excluded.strength_score,
                    weakness_score = excluded.weakness_score,
                    improving_score = excluded.improving_score,
                    last_signal = excluded.last_signal,
                    evidence_count = excluded.evidence_count,
                    last_updated = CURRENT_TIMESTAMP
                """,
                (
                    user_id,
                    entity,
                    entity_key,
                    entity_type,
                    round(score, 4),
                    round(strength_score, 4),
                    round(weakness_score, 4),
                    round(improving_score, 4),
                    signal_type,
                    evidence_count + 1,
                ),
            )
            conn.execute(
                """
                INSERT INTO skill_profile_events (
                    user_id, entity, entity_key, entity_type, signal_type, delta, rationale
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (user_id, entity, entity_key, entity_type, signal_type, delta, rationale),
            )
        conn.commit()
    finally:
        conn.close()


def fetch_profile_summary(*, user_id: str, limit: int = 8) -> dict[str, list[dict[str, object]]]:
    conn = _connect()
    try:
        strong_rows = conn.execute(
            """
            SELECT entity, entity_type, score, strength_score, weakness_score, improving_score, evidence_count, last_signal
            FROM skill_profile
            WHERE user_id = ? AND score > 0.35
            ORDER BY score DESC, evidence_count DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()
        weak_rows = conn.execute(
            """
            SELECT entity, entity_type, score, strength_score, weakness_score, improving_score, evidence_count, last_signal
            FROM skill_profile
            WHERE user_id = ? AND score < -0.35
            ORDER BY score ASC, evidence_count DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()
        improving_rows = conn.execute(
            """
            SELECT entity, entity_type, score, strength_score, weakness_score, improving_score, evidence_count, last_signal
            FROM skill_profile
            WHERE user_id = ? AND improving_score > 0.3
            ORDER BY improving_score DESC, evidence_count DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()
        return {
            "strengths": [_row_to_dict(row) for row in strong_rows],
            "weaknesses": [_row_to_dict(row) for row in weak_rows],
            "improving": [_row_to_dict(row) for row in improving_rows],
        }
    finally:
        conn.close()


def delete_user_profile(*, user_id: str) -> dict[str, int]:
    conn = _connect()
    try:
        profile_deleted = conn.execute("DELETE FROM skill_profile WHERE user_id = ?", (user_id,)).rowcount
        events_deleted = conn.execute("DELETE FROM skill_profile_events WHERE user_id = ?", (user_id,)).rowcount
        conn.commit()
        return {"profile_deleted": int(profile_deleted or 0), "events_deleted": int(events_deleted or 0)}
    finally:
        conn.close()


def _row_to_dict(row: sqlite3.Row) -> dict[str, object]:
    return {
        "entity": str(row["entity"]),
        "entity_type": str(row["entity_type"]),
        "score": float(row["score"] or 0.0),
        "strength_score": float(row["strength_score"] or 0.0),
        "weakness_score": float(row["weakness_score"] or 0.0),
        "improving_score": float(row["improving_score"] or 0.0),
        "evidence_count": int(row["evidence_count"] or 0),
        "last_signal": str(row["last_signal"] or ""),
    }
