from __future__ import annotations

import json
from functools import lru_cache
import hashlib
import os
import re
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from .gemini_chat import embed_texts_with_kind

FALLBACK_EMBEDDING_KIND = "local-hash-v1"
TOPIC_STOPWORDS = {
    "a", "an", "and", "are", "at", "be", "for", "from", "how", "i", "in", "is",
    "it", "me", "my", "of", "on", "or", "the", "this", "to", "we", "with", "you",
}


@dataclass
class VectorRecord:
    id: str
    user_id: str
    conversation_id: str
    role: str
    created_at: str | None
    text: str
    embedding: list[float]
    embedding_kind: str
    topic_key: str


@dataclass
class FaissBucket:
    index: faiss.Index
    records: list[VectorRecord]


_CACHE_LOCK = threading.Lock()
_USER_BUCKETS: dict[tuple[str, str], FaissBucket] = {}
_TOPIC_BUCKETS: dict[tuple[str, str, str], FaissBucket] = {}


def _db_path() -> Path:
    return Path(__file__).resolve().parent / "vector_store.sqlite3"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_db_path()))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS message_vectors (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL,
            created_at TEXT,
            text TEXT NOT NULL,
            embedding_json TEXT NOT NULL,
            embedding_kind TEXT NOT NULL DEFAULT 'local-hash-v1',
            topic_key TEXT NOT NULL DEFAULT ''
        )
        """
    )
    columns = {
        row[1]
        for row in conn.execute("PRAGMA table_info(message_vectors)").fetchall()
    }
    if "embedding_kind" not in columns:
        conn.execute(
            "ALTER TABLE message_vectors ADD COLUMN embedding_kind TEXT NOT NULL DEFAULT 'local-hash-v1'"
        )
        conn.commit()
    if "topic_key" not in columns:
        conn.execute(
            "ALTER TABLE message_vectors ADD COLUMN topic_key TEXT NOT NULL DEFAULT ''"
        )
        conn.commit()
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_message_vectors_user ON message_vectors(user_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_message_vectors_convo ON message_vectors(conversation_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_message_vectors_topic ON message_vectors(user_id, topic_key)"
    )
    _backfill_topic_keys(conn)
    return conn


def _current_embedding_kind() -> str:
    mode = os.getenv("GRAPHMIND_VECTOR_EMBEDDINGS", "local").strip().lower()
    if mode == "gemini":
        _, kind = embed_texts_with_kind(["graphmind embedding probe"])
        return kind
    return FALLBACK_EMBEDDING_KIND


def _fallback_embedding(text: str, dimensions: int = 64) -> list[float]:
    values = [0.0] * dimensions
    tokens = re.findall(r"\w+", text.lower())
    if not tokens:
        return values

    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = digest[0] % dimensions
        sign = 1.0 if digest[1] % 2 == 0 else -1.0
        values[index] += sign

    norm = sum(value * value for value in values) ** 0.5
    if norm == 0:
        return values
    return [value / norm for value in values]


def _resolve_embeddings(texts: list[str]) -> tuple[list[list[float]], str]:
    if len(texts) == 1:
        embedding, kind = _cached_single_embedding(texts[0])
        return [list(embedding)], kind
    mode = os.getenv("GRAPHMIND_VECTOR_EMBEDDINGS", "local").strip().lower()
    if mode == "gemini":
        return embed_texts_with_kind(texts)
    return [_fallback_embedding(text) for text in texts], FALLBACK_EMBEDDING_KIND


@lru_cache(maxsize=1024)
def _cached_single_embedding(text: str) -> tuple[tuple[float, ...], str]:
    mode = os.getenv("GRAPHMIND_VECTOR_EMBEDDINGS", "local").strip().lower()
    if mode == "gemini":
        embeddings, kind = embed_texts_with_kind([text])
        return tuple(float(value) for value in embeddings[0]), kind
    return tuple(_fallback_embedding(text)), FALLBACK_EMBEDDING_KIND


def _normalize_topic_terms(text: str) -> list[str]:
    normalized = re.sub(r"[^a-z0-9]+", " ", (text or "").lower())
    terms = []
    for token in normalized.split():
        if len(token) < 3 or token in TOPIC_STOPWORDS:
            continue
        terms.append(token)
    return terms


def _backfill_topic_keys(conn: sqlite3.Connection) -> None:
    rows = conn.execute(
        """
        SELECT id, text
        FROM message_vectors
        WHERE topic_key IS NULL OR topic_key = ''
        LIMIT 500
        """
    ).fetchall()
    if not rows:
        return
    for message_id, text in rows:
        conn.execute(
            "UPDATE message_vectors SET topic_key = ? WHERE id = ?",
            (_topic_key_for_text(str(text or "")), str(message_id)),
        )
    conn.commit()


def _topic_key_for_text(text: str) -> str:
    terms = _normalize_topic_terms(text)
    if not terms:
        return ""
    unique = []
    seen: set[str] = set()
    for term in terms:
        if term in seen:
            continue
        seen.add(term)
        unique.append(term)
    return "|".join(unique[:3])


def _query_topic_keys(text: str) -> list[str]:
    terms = _normalize_topic_terms(text)
    if not terms:
        return []
    keys: list[str] = []
    for i in range(min(len(terms), 3)):
        key = "|".join(terms[i:i + 3])
        if key and key not in keys:
            keys.append(key)
    singles = [term for term in terms[:3] if term not in keys]
    return keys + singles


def _vector_dimension(kind: str) -> int:
    _, resolved_kind = _resolve_embeddings(["dimension probe"])
    if resolved_kind == kind:
        probe, _ = _resolve_embeddings(["dimension probe"])
        return len(probe[0])
    probe = _fallback_embedding("dimension probe")
    return len(probe)


def _rows_to_records(rows: list[tuple[Any, ...]]) -> list[VectorRecord]:
    records: list[VectorRecord] = []
    for row in rows:
        try:
            embedding = json.loads(row[6])
        except Exception:
            continue
        if not isinstance(embedding, list) or not embedding:
            continue
        records.append(
            VectorRecord(
                id=str(row[0]),
                user_id=str(row[1]),
                conversation_id=str(row[2]),
                role=str(row[3]),
                created_at=str(row[4]) if row[4] is not None else None,
                text=str(row[5]),
                embedding=[float(value) for value in embedding],
                embedding_kind=str(row[7] or FALLBACK_EMBEDDING_KIND),
                topic_key=str(row[8] or ""),
            )
        )
    return records


def _build_bucket(records: list[VectorRecord]) -> FaissBucket | None:
    if not records:
        return None
    dim = len(records[0].embedding)
    matrix = np.asarray([record.embedding for record in records], dtype="float32")
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(matrix)
    index.add(matrix)
    return FaissBucket(index=index, records=records)


def _load_user_bucket(user_id: str, embedding_kind: str) -> FaissBucket | None:
    cache_key = (user_id, embedding_kind)
    with _CACHE_LOCK:
        cached = _USER_BUCKETS.get(cache_key)
        if cached is not None:
            return cached

    conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT id, user_id, conversation_id, role, created_at, text, embedding_json, embedding_kind, topic_key
            FROM message_vectors
            WHERE user_id = ? AND embedding_kind = ?
            ORDER BY rowid DESC
            LIMIT 2000
            """,
            (user_id, embedding_kind),
        ).fetchall()
    finally:
        conn.close()

    bucket = _build_bucket(_rows_to_records(rows))
    if bucket is None:
        return None

    with _CACHE_LOCK:
        _USER_BUCKETS[cache_key] = bucket
    return bucket


def _load_topic_bucket(user_id: str, topic_key: str, embedding_kind: str) -> FaissBucket | None:
    cache_key = (user_id, topic_key, embedding_kind)
    with _CACHE_LOCK:
        cached = _TOPIC_BUCKETS.get(cache_key)
        if cached is not None:
            return cached

    conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT id, user_id, conversation_id, role, created_at, text, embedding_json, embedding_kind, topic_key
            FROM message_vectors
            WHERE user_id = ? AND embedding_kind = ? AND topic_key LIKE ?
            ORDER BY rowid DESC
            LIMIT 500
            """,
            (user_id, embedding_kind, f"%{topic_key}%"),
        ).fetchall()
    finally:
        conn.close()

    bucket = _build_bucket(_rows_to_records(rows))
    if bucket is None:
        return None

    with _CACHE_LOCK:
        _TOPIC_BUCKETS[cache_key] = bucket
    return bucket


def _append_to_bucket(bucket: FaissBucket | None, record: VectorRecord) -> FaissBucket:
    if bucket is None:
        return _build_bucket([record])  # type: ignore[return-value]

    vector = np.asarray([record.embedding], dtype="float32")
    faiss.normalize_L2(vector)
    bucket.index.add(vector)
    bucket.records.append(record)
    return bucket


def _invalidate_user_caches(user_id: str) -> None:
    with _CACHE_LOCK:
        for key in [key for key in _USER_BUCKETS if key[0] == user_id]:
            _USER_BUCKETS.pop(key, None)
        for key in [key for key in _TOPIC_BUCKETS if key[0] == user_id]:
            _TOPIC_BUCKETS.pop(key, None)


def add_message(*, message_id: str, text: str, metadata: dict[str, Any]) -> None:
    embeddings, embedding_kind = _resolve_embeddings([text])
    emb = embeddings[0]
    user_id = str(metadata.get("user_id") or "")
    conversation_id = str(metadata.get("conversation_id") or "")
    role = str(metadata.get("role") or "")
    created_at = metadata.get("created_at")
    topic_key = _topic_key_for_text(text)

    conn = _connect()
    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO message_vectors
              (id, user_id, conversation_id, role, created_at, text, embedding_json, embedding_kind, topic_key)
            VALUES
              (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                message_id,
                user_id,
                conversation_id,
                role,
                str(created_at) if created_at is not None else None,
                text,
                json.dumps(emb),
                embedding_kind,
                topic_key,
            ),
        )
        conn.commit()
    finally:
        conn.close()

    record = VectorRecord(
        id=message_id,
        user_id=user_id,
        conversation_id=conversation_id,
        role=role,
        created_at=str(created_at) if created_at is not None else None,
        text=text,
        embedding=emb,
        embedding_kind=embedding_kind,
        topic_key=topic_key,
    )
    with _CACHE_LOCK:
        user_key = (user_id, embedding_kind)
        _USER_BUCKETS[user_key] = _append_to_bucket(_USER_BUCKETS.get(user_key), record)
        if topic_key:
            topic_bucket_key = (user_id, topic_key, embedding_kind)
            _TOPIC_BUCKETS[topic_bucket_key] = _append_to_bucket(_TOPIC_BUCKETS.get(topic_bucket_key), record)


def search(
    *,
    query: str,
    user_id: str,
    conversation_id: str | None = None,
    k: int = 5,
) -> list[dict[str, Any]]:
    query_embeddings, embedding_kind = _resolve_embeddings([query])
    query_vector = np.asarray(query_embeddings, dtype="float32")
    faiss.normalize_L2(query_vector)
    k = max(1, min(int(k), 20))

    topic_keys = _query_topic_keys(query)
    bucket = None
    for topic_key in topic_keys:
        candidate = _load_topic_bucket(user_id, topic_key, embedding_kind)
        if candidate is not None and len(candidate.records) >= max(3, k):
            bucket = candidate
            break
    if bucket is None:
        bucket = _load_user_bucket(user_id, embedding_kind)
    if bucket is None or not bucket.records:
        return []

    search_k = min(max(k * 4, 20), len(bucket.records))
    distances, indices = bucket.index.search(query_vector, search_k)
    results: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for score, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(bucket.records):
            continue
        record = bucket.records[int(idx)]
        if conversation_id and record.conversation_id != conversation_id:
            continue
        if record.id in seen_ids:
            continue
        seen_ids.add(record.id)
        results.append(
            {
                "id": record.id,
                "text": record.text,
                "metadata": {
                    "user_id": record.user_id,
                    "conversation_id": record.conversation_id,
                    "role": record.role,
                    "created_at": record.created_at,
                    "topic_key": record.topic_key,
                },
                "score": float(score),
            }
        )
        if len(results) >= k:
            break

    return results


def list_vector_users(limit: int = 200) -> list[str]:
    conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT DISTINCT user_id
            FROM message_vectors
            WHERE user_id != ''
            ORDER BY user_id
            LIMIT ?
            """,
            (max(1, min(limit, 1000)),),
        ).fetchall()
    finally:
        conn.close()
    return [str(row[0]) for row in rows if row and row[0]]


def list_embedding_kinds(limit: int = 20) -> list[str]:
    conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT DISTINCT embedding_kind
            FROM message_vectors
            WHERE embedding_kind != ''
            ORDER BY embedding_kind
            LIMIT ?
            """,
            (max(1, min(limit, 100)),),
        ).fetchall()
    finally:
        conn.close()
    kinds = [str(row[0]) for row in rows if row and row[0]]
    return kinds or [_current_embedding_kind()]


def warm_user_indexes(*, limit_users: int = 100) -> dict[str, int]:
    warmed = 0
    kinds = list_embedding_kinds(limit=10)
    for user_id in list_vector_users(limit=limit_users):
        user_warmed = False
        for embedding_kind in kinds:
            bucket = _load_user_bucket(user_id, embedding_kind)
            if bucket is not None:
                user_warmed = True
        if user_warmed:
            warmed += 1
    return {"users_warmed": warmed}


def delete_user_messages(*, user_id: str) -> int:
    conn = _connect()
    try:
        deleted = conn.execute(
            "DELETE FROM message_vectors WHERE user_id = ?",
            (user_id,),
        ).rowcount
        conn.commit()
    finally:
        conn.close()
    _invalidate_user_caches(user_id)
    return int(deleted or 0)


def rebuild_user_index(*, user_id: str) -> dict[str, int]:
    embedding_kind = _current_embedding_kind()
    _invalidate_user_caches(user_id)
    bucket = _load_user_bucket(user_id, embedding_kind)
    return {
        "user_records": len(bucket.records) if bucket is not None else 0,
        "topic_buckets": 0,
    }
