from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import threading
from typing import Iterable

import faiss
import numpy as np

from .graph.models import normalize_text_key

GENERIC_TOPIC_KEYS = {
    "study",
    "studies",
    "learning",
    "preparation",
    "practice",
    "revision",
    "topic",
    "topics",
    "concept",
    "concepts",
}


@dataclass
class TopicMatch:
    topic: str
    score: float
    source: str


class TopicSemanticRouter:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._topics: list[str] = []
        self._sources: list[str] = []
        self._index: faiss.Index | None = None
        self._dim = 96

    def refresh_from_session(self, session) -> dict[str, int]:
        rows = session.run(
            """
            MATCH (n)
            WHERE NOT n:User AND NOT n:Conversation AND NOT n:Message
            RETURN DISTINCT coalesce(n.name, '') AS name,
                            coalesce(n.aliases, []) AS aliases,
                            coalesce(n.kind, head(labels(n)), 'Entity') AS kind
            LIMIT 1000
            """
        )
        values: list[tuple[str, str]] = []
        seen: set[str] = set()
        for row in rows:
            kind = str(row.get("kind") or "Entity")
            if kind.lower() in {"company"}:
                continue
            candidates: list[str] = [str(row.get("name") or "").strip()]
            candidates.extend(str(alias).strip() for alias in list(row.get("aliases") or []))
            for candidate in candidates:
                if not candidate:
                    continue
                key = normalize_text_key(candidate)
                if not key or key in seen or key in GENERIC_TOPIC_KEYS:
                    continue
                seen.add(key)
                values.append((candidate, kind))

        with self._lock:
            self._topics = [item[0] for item in values]
            self._sources = [item[1] for item in values]
            self._index = None
            if self._topics:
                matrix = np.asarray([self._embed_text(topic) for topic in self._topics], dtype="float32")
                faiss.normalize_L2(matrix)
                self._index = faiss.IndexFlatIP(self._dim)
                self._index.add(matrix)
        return {"topics_indexed": len(values)}

    def detect(self, message: str, min_score: float = 0.48) -> TopicMatch | None:
        text = (message or "").strip()
        if not text:
            return None
        with self._lock:
            if self._index is None or not self._topics:
                return None
            query = np.asarray([self._embed_text(text)], dtype="float32")
            faiss.normalize_L2(query)
            distances, indices = self._index.search(query, 1)
            idx = int(indices[0][0])
            score = float(distances[0][0])
            if idx < 0 or idx >= len(self._topics) or score < min_score:
                return None
            return TopicMatch(topic=self._topics[idx], score=round(score, 4), source=self._sources[idx])

    def _embed_text(self, text: str) -> list[float]:
        values = [0.0] * self._dim
        for token in self._tokens(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = digest[0] % self._dim
            sign = 1.0 if digest[1] % 2 == 0 else -1.0
            values[index] += sign
        norm = math.sqrt(sum(value * value for value in values))
        if norm == 0.0:
            return values
        return [value / norm for value in values]

    @staticmethod
    def _tokens(text: str) -> Iterable[str]:
        normalized = normalize_text_key(text)
        return [token for token in normalized.split() if token]


topic_semantic_router = TopicSemanticRouter()
