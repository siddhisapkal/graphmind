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
            update_mode = str(item.get("update_mode") or "reinforce").strip().lower()
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
                if update_mode == "replace_opposite":
                    weakness_score = max(0.0, weakness_score - max(abs(delta), 0.9))
                    improving_score = max(0.0, improving_score - min(max(abs(delta), 0.7), 0.9))
                    score = max(score, 0.0)
                score += abs(delta) or 0.8
                strength_score += abs(delta) or 0.8
            elif signal_type == "weakness":
                if update_mode == "replace_opposite":
                    strength_score = max(0.0, strength_score - max(abs(delta), 0.9))
                    improving_score = max(0.0, improving_score - min(max(abs(delta), 0.7), 0.9))
                    score = min(score, 0.0)
                score -= abs(delta) or 0.8
                weakness_score += abs(delta) or 0.8
            elif signal_type == "improving":
                if update_mode == "replace_opposite":
                    strength_score = max(0.0, strength_score - min(abs(delta) or 0.45, 0.5))
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
        rows = conn.execute(
            """
            SELECT entity, entity_type, score, strength_score, weakness_score, improving_score, evidence_count, last_signal
            FROM skill_profile
            WHERE user_id = ?
            ORDER BY evidence_count DESC, ABS(score) DESC, improving_score DESC, strength_score DESC, weakness_score DESC
            """,
            (user_id,),
        ).fetchall()
        grouped = _group_profile_rows(rows)
        return {
            "strengths": grouped["strengths"][:limit],
            "weaknesses": grouped["weaknesses"][:limit],
            "improving": grouped["improving"][:limit],
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


def _group_profile_rows(rows: list[sqlite3.Row]) -> dict[str, list[dict[str, object]]]:
    strengths: list[dict[str, object]] = []
    weaknesses: list[dict[str, object]] = []
    improving: list[dict[str, object]] = []
    seen_strengths: set[str] = set()
    seen_weaknesses: set[str] = set()
    seen_improving: set[str] = set()

    for row in rows:
        item = _row_to_dict(row)
        strength_score = float(item["strength_score"] or 0.0)
        weakness_score = float(item["weakness_score"] or 0.0)
        improving_score = float(item["improving_score"] or 0.0)
        score = float(item["score"] or 0.0)
        key = _profile_display_key(str(item["entity"]))

        if strength_score > 0.35 and strength_score >= max(weakness_score, improving_score):
            if key not in seen_strengths:
                seen_strengths.add(key)
                strengths.append(item)

        if weakness_score > 0.35 and weakness_score >= max(strength_score, improving_score):
            if key not in seen_weaknesses:
                seen_weaknesses.add(key)
                weaknesses.append(item)

        if improving_score > 0.3 and improving_score >= weakness_score:
            if key not in seen_improving:
                seen_improving.add(key)
                improving.append(item)

        if (
            key not in seen_weaknesses
            and weakness_score > 0.25
            and score < -0.2
            and weakness_score >= strength_score
            and weakness_score >= improving_score
        ):
            seen_weaknesses.add(key)
            weaknesses.append(item)

    return {
        "strengths": strengths,
        "weaknesses": weaknesses,
        "improving": improving,
    }


def _profile_display_key(entity: str) -> str:
    normalized = " ".join((entity or "").lower().split()).strip()
    normalized = normalized.replace("-", " ")
    normalized = normalized.replace("problem solving", "problem-solving")
    normalized = normalized.replace("advanced dsa", "advanced dsa")
    normalized = normalized.replace("timed coding rounds", "timed coding")
    normalized = normalized.replace("timed coding", "timed coding")
    normalized = normalized.replace("improving advanced dsa", "advanced dsa")
    return normalized
