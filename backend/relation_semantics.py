from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from .relation_store import get_relation_metadata, upsert_relation_metadata


@dataclass(frozen=True)
class RelationSemantics:
    family: str
    polarity: str
    section_tags: tuple[str, ...]
    strength: float
    source: str = "heuristic"
    confidence: float = 0.0


def classify_relation_semantics(relation: str, *, entity_type: str = "") -> RelationSemantics:
    normalized = "".join(ch if ch.isalnum() else "_" for ch in (relation or "").upper())
    normalized = "_".join(part for part in normalized.split("_") if part)
    lowered_entity = (entity_type or "").strip().lower()
    relation_key = _relation_key(normalized, lowered_entity)
    cached = get_relation_metadata(relation_key=relation_key)
    if cached:
        return RelationSemantics(
            family=str(cached["family"]),
            polarity=str(cached["polarity"]),
            section_tags=tuple(str(tag) for tag in list(cached.get("section_tags") or [])),
            strength=float(cached["strength"]),
            source=str(cached.get("source") or "cache"),
            confidence=0.98,
        )

    if any(token in normalized for token in ("WEAK", "STRUGGLE", "LACK", "CONFUSE", "DIFFICULT")):
        semantics = RelationSemantics(
            family="capability",
            polarity="negative",
            section_tags=("weakness", "practice_priority", "capability"),
            strength=0.96,
            confidence=0.94,
        )
        _store_semantics(relation_key, relation, entity_type, semantics)
        return semantics
    if any(token in normalized for token in ("STRONG", "STRENGTH", "IMPROVED", "CONFIDENT")):
        semantics = RelationSemantics(
            family="capability",
            polarity="positive",
            section_tags=("strength", "revision", "capability"),
            strength=0.88,
            confidence=0.9,
        )
        _store_semantics(relation_key, relation, entity_type, semantics)
        return semantics
    if any(token in normalized for token in ("TARGET", "AIM", "PREPARE", "APPLY", "GOAL")):
        tags = ("goal", "target")
        if lowered_entity == "company":
            tags = ("goal", "company", "target")
        semantics = RelationSemantics(
            family="goal",
            polarity="neutral",
            section_tags=tags,
            strength=0.9,
            confidence=0.9,
        )
        _store_semantics(relation_key, relation, entity_type, semantics)
        return semantics
    if any(token in normalized for token in ("STUDY", "LEARN", "PRACTICE", "WORK", "BUILD")):
        semantics = RelationSemantics(
            family="learning",
            polarity="neutral",
            section_tags=("topic", "learning", "study"),
            strength=0.78,
            confidence=0.84,
        )
        _store_semantics(relation_key, relation, entity_type, semantics)
        return semantics
    if any(token in normalized for token in ("PART_OF", "USED_IN", "DEPENDS_ON", "PREREQUISITE", "REQUIRES", "INCLUDES")):
        semantics = RelationSemantics(
            family="structure",
            polarity="neutral",
            section_tags=("topic", "structure", "related"),
            strength=0.72,
            confidence=0.82,
        )
        _store_semantics(relation_key, relation, entity_type, semantics)
        return semantics
    if normalized == "RELATED_TO":
        semantics = RelationSemantics(
            family="association",
            polarity="neutral",
            section_tags=("related",),
            strength=0.45,
            confidence=0.8,
        )
        _store_semantics(relation_key, relation, entity_type, semantics)
        return semantics
    semantics = RelationSemantics(
        family="general",
        polarity="neutral",
        section_tags=("general",),
        strength=0.5,
        confidence=0.35,
    )
    _store_semantics(relation_key, relation, entity_type, semantics)
    return semantics


def should_background_enrich(semantics: RelationSemantics) -> bool:
    return semantics.source == "heuristic" and semantics.confidence < 0.75


def store_llm_relation_semantics(
    *,
    relation: str,
    entity_type: str,
    family: str,
    polarity: str,
    section_tags: list[str],
    strength: float,
) -> None:
    relation_key = _relation_key(
        "".join(ch if ch.isalnum() else "_" for ch in (relation or "").upper()),
        (entity_type or "").strip().lower(),
    )
    upsert_relation_metadata(
        relation_key=relation_key,
        relation=relation,
        entity_type=entity_type,
        family=family,
        polarity=polarity,
        section_tags=section_tags,
        strength=float(strength),
        source="llm",
        updated_at=datetime.now(timezone.utc).isoformat(),
    )


def _store_semantics(relation_key: str, relation: str, entity_type: str, semantics: RelationSemantics) -> None:
    upsert_relation_metadata(
        relation_key=relation_key,
        relation=relation,
        entity_type=entity_type,
        family=semantics.family,
        polarity=semantics.polarity,
        section_tags=list(semantics.section_tags),
        strength=semantics.strength,
        source=semantics.source,
        updated_at=datetime.now(timezone.utc).isoformat(),
    )


def _relation_key(relation: str, entity_type: str) -> str:
    return f"{relation}|{entity_type}"
