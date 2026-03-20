from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import re


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_text_key(value: str) -> str:
    lowered = (value or "").strip().lower()
    lowered = lowered.replace("&", " and ")
    lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered


@dataclass
class MemorySignal:
    user_id: str
    entity: str
    relation: str
    source: str
    raw_text: str
    confidence: float
    entity_type: str = "Entity"
    linked_to_action: bool = False
    timestamp: str = field(default_factory=utc_now_iso)

    def normalized_relation(self) -> str:
        cleaned = "".join(ch if ch.isalnum() else "_" for ch in self.relation.upper())
        cleaned = "_".join(part for part in cleaned.split("_") if part)
        return cleaned or "RELATED_TO"

    def storage_key(self) -> str:
        return "|".join(
            [
                self.user_id.strip().lower(),
                self.normalized_relation(),
                self.entity_type.strip().lower(),
                self.entity.strip().lower(),
            ]
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class TripleCandidate:
    user_id: str
    subject_type: str
    subject_name: str
    relation: str
    object_type: str
    object_name: str
    confidence: float
    source: str
    raw_text: str
    source_event_id: str | None = None
    linked_to_action: bool = False
    promotion_hint: str = "default"
    timestamp: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class ResolvedNodeRef:
    canonical_id: str
    name: str
    kind: str
    aliases: list[str] = field(default_factory=list)
    matched: bool = False
    score: float = 1.0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class ResolvedTripleCandidate:
    user_id: str
    subject: ResolvedNodeRef
    relation: str
    object: ResolvedNodeRef
    confidence: float
    source: str
    raw_text: str
    source_event_id: str | None = None
    linked_to_action: bool = False
    promotion_hint: str = "default"
    timestamp: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        return payload


@dataclass
class EphemeralAggregate:
    key: str
    user_id: str
    entity: str
    entity_type: str
    relation: str
    max_confidence: float
    mention_count: int
    first_seen: str
    last_seen: str
    sources: list[str] = field(default_factory=list)
    linked_to_action: bool = False
    promoted: bool = False
    last_raw_text: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_signal(cls, signal: MemorySignal) -> "EphemeralAggregate":
        return cls(
            key=signal.storage_key(),
            user_id=signal.user_id,
            entity=signal.entity,
            entity_type=signal.entity_type,
            relation=signal.normalized_relation(),
            max_confidence=signal.confidence,
            mention_count=1,
            first_seen=signal.timestamp,
            last_seen=signal.timestamp,
            sources=[signal.source],
            linked_to_action=signal.linked_to_action,
            promoted=False,
            last_raw_text=signal.raw_text,
        )
