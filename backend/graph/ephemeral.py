from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Protocol

from dotenv import load_dotenv

from .models import EphemeralAggregate, MemorySignal

try:
    import redis
except Exception:  # pragma: no cover
    redis = None


load_dotenv(Path(__file__).resolve().parent.parent / ".env")


class SupportsGetSet(Protocol):
    def get(self, key: str) -> str | None: ...
    def set(self, key: str, value: str, ex: int | None = None) -> None: ...


class InMemoryTTLStore:
    def __init__(self) -> None:
        self._data: dict[str, tuple[float, str]] = {}

    def get(self, key: str) -> str | None:
        item = self._data.get(key)
        if item is None:
            return None
        expires_at, payload = item
        if expires_at < time.time():
            self._data.pop(key, None)
            return None
        return payload

    def set(self, key: str, value: str, ex: int | None = None) -> None:
        ttl = ex if ex is not None else 3600
        self._data[key] = (time.time() + ttl, value)

    def delete(self, key: str) -> None:
        self._data.pop(key, None)


class SQLiteTTLStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._ensure_table()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    def _ensure_table(self) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ephemeral_memory (
                    key TEXT PRIMARY KEY,
                    expires_at REAL NOT NULL,
                    payload TEXT NOT NULL
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    def get(self, key: str) -> str | None:
        now = time.time()
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT expires_at, payload FROM ephemeral_memory WHERE key = ?",
                (key,),
            ).fetchone()
            if row is None:
                return None
            expires_at, payload = row
            if expires_at < now:
                conn.execute("DELETE FROM ephemeral_memory WHERE key = ?", (key,))
                conn.commit()
                return None
            return str(payload)
        finally:
            conn.close()

    def set(self, key: str, value: str, ex: int | None = None) -> None:
        ttl = ex if ex is not None else 3600
        expires_at = time.time() + ttl
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO ephemeral_memory (key, expires_at, payload)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    expires_at = excluded.expires_at,
                    payload = excluded.payload
                """,
                (key, expires_at, value),
            )
            conn.commit()
        finally:
            conn.close()

    def list_items(self, prefix: str, limit: int) -> list[str]:
        now = time.time()
        conn = self._connect()
        try:
            conn.execute("DELETE FROM ephemeral_memory WHERE expires_at < ?", (now,))
            rows = conn.execute(
                """
                SELECT payload
                FROM ephemeral_memory
                WHERE key LIKE ?
                ORDER BY expires_at DESC
                LIMIT ?
                """,
                (f"graphmind:ephemeral:{prefix}%", max(1, min(limit, 100))),
            ).fetchall()
            conn.commit()
            return [str(row[0]) for row in rows]
        finally:
            conn.close()

    def delete_prefix(self, prefix: str) -> int:
        conn = self._connect()
        try:
            deleted = conn.execute(
                "DELETE FROM ephemeral_memory WHERE key LIKE ?",
                (f"{prefix}%",),
            ).rowcount
            conn.commit()
            return int(deleted or 0)
        finally:
            conn.close()


class EphemeralMemoryStore:
    def __init__(self, ttl_seconds: int = 86400) -> None:
        self.ttl_seconds = ttl_seconds
        self._client, self.backend_name = self._build_client()

    def _build_client(self) -> tuple[SupportsGetSet, str]:
        redis_url = os.getenv("REDIS_URL")
        if redis and redis_url:
            client = redis.Redis.from_url(redis_url, decode_responses=True)
            try:
                client.ping()
                return client, "redis"
            except Exception:
                pass
        sqlite_path = Path(__file__).resolve().parent.parent / "ephemeral_memory.sqlite3"
        return SQLiteTTLStore(sqlite_path), "sqlite"

    def upsert_signal(self, signal: MemorySignal) -> EphemeralAggregate:
        key = f"graphmind:ephemeral:{signal.storage_key()}"
        existing_raw = self._client.get(key)

        if existing_raw:
            payload = json.loads(existing_raw)
            aggregate = EphemeralAggregate(**payload)
            aggregate.max_confidence = max(aggregate.max_confidence, signal.confidence)
            aggregate.mention_count += 1
            aggregate.last_seen = signal.timestamp
            aggregate.last_raw_text = signal.raw_text
            aggregate.linked_to_action = aggregate.linked_to_action or signal.linked_to_action
            if signal.source not in aggregate.sources:
                aggregate.sources.append(signal.source)
        else:
            aggregate = EphemeralAggregate.from_signal(signal)

        self._client.set(key, json.dumps(aggregate.to_dict()), ex=self.ttl_seconds)
        return aggregate

    def mark_promoted(self, aggregate: EphemeralAggregate) -> None:
        aggregate.promoted = True
        key = f"graphmind:ephemeral:{aggregate.key}"
        self._client.set(key, json.dumps(aggregate.to_dict()), ex=self.ttl_seconds)

    def list_user_items(self, user_id: str, limit: int = 50) -> list[EphemeralAggregate]:
        prefix = f"{user_id.strip().lower()}|"
        if self.backend_name == "redis":
            return self._list_user_items_redis(prefix=prefix, limit=limit)
        if self.backend_name == "sqlite":
            return self._list_user_items_sqlite(prefix=prefix, limit=limit)
        return self._list_user_items_memory(prefix=prefix, limit=limit)

    def _list_user_items_memory(self, *, prefix: str, limit: int) -> list[EphemeralAggregate]:
        items: list[EphemeralAggregate] = []
        for key, (_, payload) in list(self._client._data.items()):  # type: ignore[attr-defined]
            if not key.startswith("graphmind:ephemeral:"):
                continue
            storage_key = key.split("graphmind:ephemeral:", 1)[1]
            if not storage_key.startswith(prefix):
                continue
            try:
                items.append(EphemeralAggregate(**json.loads(payload)))
            except Exception:
                continue
        items.sort(key=lambda item: item.last_seen, reverse=True)
        return items[: max(1, min(limit, 100))]

    def _list_user_items_sqlite(self, *, prefix: str, limit: int) -> list[EphemeralAggregate]:
        items: list[EphemeralAggregate] = []
        for payload in self._client.list_items(prefix=prefix, limit=limit):  # type: ignore[attr-defined]
            try:
                items.append(EphemeralAggregate(**json.loads(payload)))
            except Exception:
                continue
        items.sort(key=lambda item: item.last_seen, reverse=True)
        return items[: max(1, min(limit, 100))]

    def _list_user_items_redis(self, *, prefix: str, limit: int) -> list[EphemeralAggregate]:
        items: list[EphemeralAggregate] = []
        pattern = f"graphmind:ephemeral:{prefix}*"
        for key in self._client.scan_iter(match=pattern):  # type: ignore[attr-defined]
            payload = self._client.get(key)
            if not payload:
                continue
            try:
                items.append(EphemeralAggregate(**json.loads(payload)))
            except Exception:
                continue
        items.sort(key=lambda item: item.last_seen, reverse=True)
        return items[: max(1, min(limit, 100))]

    def reset_user_items(self, *, user_id: str) -> dict[str, int]:
        prefix = f"{user_id.strip().lower()}|"
        deleted = 0
        if self.backend_name == "redis":
            ephemeral_pattern = f"graphmind:ephemeral:{prefix}*"
            cache_patterns = [
                ephemeral_pattern,
                f"graphmind:graph_context:{user_id}:*",
                f"graphmind:graph_evidence:{user_id}:*",
                f"graphmind:graph_memory:{user_id}:*",
                f"graphmind:graph_view:{user_id}:*",
                f"graphmind:graph_version:{user_id}",
            ]
            for pattern in cache_patterns:
                keys = list(self._client.scan_iter(match=pattern))  # type: ignore[attr-defined]
                if keys:
                    deleted += int(self._client.delete(*keys) or 0)  # type: ignore[attr-defined]
            return {"deleted_keys": deleted}
        if self.backend_name == "sqlite":
            cache_prefixes = [
                f"graphmind:ephemeral:{prefix}",
                f"graphmind:graph_context:{user_id}:",
                f"graphmind:graph_evidence:{user_id}:",
                f"graphmind:graph_memory:{user_id}:",
                f"graphmind:graph_view:{user_id}:",
                f"graphmind:graph_version:{user_id}",
            ]
            for cache_prefix in cache_prefixes:
                deleted += self._client.delete_prefix(cache_prefix)  # type: ignore[attr-defined]
            return {"deleted_keys": deleted}
        for key in [
            key
            for key in list(self._client._data.keys())  # type: ignore[attr-defined]
            if key.startswith("graphmind:ephemeral:") and key.split("graphmind:ephemeral:", 1)[1].startswith(prefix)
        ]:
            self._client.delete(key)  # type: ignore[attr-defined]
            deleted += 1
        return {"deleted_keys": deleted}
