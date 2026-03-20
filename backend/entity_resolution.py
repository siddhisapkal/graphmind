from __future__ import annotations

import re

from .graph.models import ResolvedNodeRef, ResolvedTripleCandidate, TripleCandidate, normalize_text_key

TOPIC_ALIASES = {
    "signal and systems": "Signals and Systems",
    "signals and systems": "Signals and Systems",
    "signal systems": "Signals and Systems",
    "fourier transform": "Fourier Transform",
    "fourier transforms": "Fourier Transform",
}

CONCEPTUAL_ENTITY_TYPES = {"topic", "skill", "entity", "document", "concept"}


def resolve_triples(*, session, candidates: list[TripleCandidate]) -> list[ResolvedTripleCandidate]:
    resolved: list[ResolvedTripleCandidate] = []
    for candidate in candidates:
        subject = _resolve_node(
            session=session,
            name=candidate.subject_name,
            entity_type=candidate.subject_type,
            user_id=candidate.user_id,
        )
        object_ref = _resolve_node(
            session=session,
            name=candidate.object_name,
            entity_type=candidate.object_type,
            user_id=candidate.user_id,
        )
        resolved.append(
            ResolvedTripleCandidate(
                user_id=candidate.user_id,
                subject=subject,
                relation=_sanitize_relation(candidate.relation),
                object=object_ref,
                confidence=max(0.0, min(float(candidate.confidence), 1.0)),
                source=candidate.source,
                raw_text=candidate.raw_text,
                source_event_id=candidate.source_event_id,
                linked_to_action=candidate.linked_to_action,
                promotion_hint=candidate.promotion_hint,
                timestamp=candidate.timestamp,
            )
        )
    return resolved


def _resolve_node(*, session, name: str, entity_type: str, user_id: str) -> ResolvedNodeRef:
    canonical_name = canonical_entity_name(name, entity_type)
    canonical_id = canonical_entity_id(canonical_name, entity_type, user_id=user_id)
    canonical_name_key = normalize_text_key(canonical_name)
    expected_kind = canonical_kind(entity_type, user_id=user_id)
    exact = session.run(
        """
        MATCH (n)
        WHERE n.canonical_key = $canonical_id
           OR (
                n.name_key = $canonical_name_key
                AND coalesce(n.kind, $expected_kind) = $expected_kind
           )
        RETURN n.canonical_key AS canonical_key,
               coalesce(n.name, $canonical_name) AS name,
               coalesce(n.kind, $entity_type) AS kind,
               coalesce(n.aliases, []) AS aliases
        LIMIT 1
        """,
        canonical_id=canonical_id,
        canonical_name=canonical_name,
        canonical_name_key=canonical_name_key,
        entity_type=entity_type,
        expected_kind=expected_kind,
    ).single()
    if exact:
        return ResolvedNodeRef(
            canonical_id=str(exact["canonical_key"] or canonical_id),
            name=str(exact["name"] or canonical_name),
            kind=str(exact["kind"] or entity_type),
            aliases=list(exact["aliases"] or []),
            matched=True,
            score=1.0,
        )

    return ResolvedNodeRef(
        canonical_id=canonical_id,
        name=canonical_name,
        kind=expected_kind,
        aliases=[canonical_name] if canonical_name else [],
        matched=False,
        score=1.0,
    )


def canonical_entity_name(name: str, entity_type: str) -> str:
    cleaned = re.sub(r"\s+", " ", (name or "").strip())
    lowered_type = entity_type.strip().lower()
    key = normalize_text_key(cleaned)
    if lowered_type == "topic":
        return TOPIC_ALIASES.get(key, cleaned.title())
    if lowered_type in {"company", "skill", "goal", "document", "domain", "concept"}:
        return cleaned.title()
    return cleaned


def canonical_kind(entity_type: str, *, user_id: str) -> str:
    lowered_type = entity_type.strip().lower()
    if lowered_type == "user":
        return "User"
    if lowered_type in CONCEPTUAL_ENTITY_TYPES:
        return "Concept"
    if lowered_type == "company":
        return "Company"
    if lowered_type == "goal":
        return "Goal"
    if lowered_type == "domain":
        return "Domain"
    return entity_type.strip() or "Entity"


def canonical_entity_id(name: str, entity_type: str, *, user_id: str) -> str:
    lowered_type = entity_type.strip().lower()
    user_scope = normalize_text_key(user_id)
    if lowered_type == "user":
        return f"user::{user_scope}"
    if lowered_type in CONCEPTUAL_ENTITY_TYPES:
        return f"user::{user_scope}::concept::{normalize_text_key(name)}"
    return f"user::{user_scope}::{lowered_type}::{normalize_text_key(name)}"


def _sanitize_relation(relation: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in relation.upper())
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return cleaned or "RELATED_TO"
