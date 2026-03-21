from __future__ import annotations

import heapq
import hashlib
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone

from ..entity_resolution import canonical_entity_id, canonical_entity_name, canonical_kind, resolve_triples
from ..relation_semantics import classify_relation_semantics
from .ephemeral import EphemeralMemoryStore
from .models import EphemeralAggregate, MemorySignal, ResolvedTripleCandidate, TripleCandidate, normalize_text_key

ENTITY_LABELS = {
    "company": "Company",
    "topic": "Topic",
    "skill": "Skill",
    "goal": "Goal",
    "document": "Document",
    "entity": "Entity",
}

RELATION_ALIASES = {
    "LEARNING": "STUDIES",
    "LEARNS": "STUDIES",
    "LEARNED": "STUDIES",
    "STUDIED": "STUDIES",
    "STUDYING": "STUDIES",
    "WORKS_ON": "STUDIES",
    "PREPARING_FOR": "STUDIES",
    "PREPARES_FOR": "STUDIES",
    "INTERVIEWING_WITH": "TARGETS",
    "INTERVIEWS_WITH": "TARGETS",
}

TOPIC_ALIASES = {
    "signal and systems": "Signals and Systems",
    "signals and systems": "Signals and Systems",
    "signal systems": "Signals and Systems",
    "fourier transform": "Fourier Transform",
    "fourier transforms": "Fourier Transform",
}

CONCEPTUAL_ENTITY_TYPES = {"topic", "skill", "entity", "document"}

STRUCTURAL_EDGE_PATTERNS = (
    ("PART_OF", ("part of", "inside", "within")),
    ("USED_IN", ("used in", "applied in", "application of", " in ")),
    ("DEPENDS_ON", ("depends on", "based on", "requires")),
)

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "for", "from", "i", "in", "is",
    "it", "me", "my", "of", "on", "or", "the", "to", "we", "with", "you",
}

RELATION_STRENGTHS = {
    "TARGETS": 1.0,
    "STRENGTH_IN": 0.96,
    "IMPROVED_IN": 0.9,
    "STUDIES": 0.82,
    "STRUGGLES_WITH": 0.88,
    "PART_OF": 0.76,
    "USED_IN": 0.72,
    "DEPENDS_ON": 0.74,
    "RELATED_TO": 0.45,
}


class GraphMemoryService:
    def __init__(
        self,
        *,
        confidence_threshold: float = 0.8,
        repeated_mentions_threshold: int = 2,
        ephemeral_store: EphemeralMemoryStore | None = None,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.repeated_mentions_threshold = repeated_mentions_threshold
        self.ephemeral_store = ephemeral_store or EphemeralMemoryStore()

    @property
    def ephemeral_backend(self) -> str:
        return self.ephemeral_store.backend_name

    def _cache_get(self, key: str) -> dict | list | None:
        payload = self.ephemeral_store._client.get(key)  # type: ignore[attr-defined]
        if not payload:
            return None
        try:
            return json.loads(payload)
        except Exception:
            return None

    def _cache_set(self, key: str, value: dict | list, ttl_seconds: int = 900) -> None:
        self.ephemeral_store._client.set(  # type: ignore[attr-defined]
            key,
            json.dumps(value),
            ex=ttl_seconds,
        )

    def _graph_version(self, user_id: str) -> int:
        raw = self.ephemeral_store._client.get(f"graphmind:graph_version:{user_id}")  # type: ignore[attr-defined]
        return int(raw or 0)

    def _bump_graph_version(self, user_id: str) -> None:
        key = f"graphmind:graph_version:{user_id}"
        client = self.ephemeral_store._client  # type: ignore[attr-defined]
        try:
            client.incr(key)
        except Exception:
            current = int(client.get(key) or 0)
            client.set(key, str(current + 1), ex=self.ephemeral_store.ttl_seconds)

    def ensure_schema(self, session) -> None:
        self._drop_legacy_name_constraints(session)
        self._deduplicate_entities_by_name(session)
        self._backfill_canonical_keys(session)
        self._deduplicate_entities(session)
        for label in ENTITY_LABELS.values():
            session.run(
                f"CREATE CONSTRAINT {label.lower()}_canonical_key_unique IF NOT EXISTS FOR (n:{label}) REQUIRE n.canonical_key IS UNIQUE"
            )
            session.run(
                f"CREATE INDEX {label.lower()}_name_key_idx IF NOT EXISTS FOR (n:{label}) ON (n.name_key)"
            )

    def _drop_legacy_name_constraints(self, session) -> None:
        constraints = session.run("SHOW CONSTRAINTS YIELD name, labelsOrTypes, properties")
        valid_labels = set(ENTITY_LABELS.values())
        for record in constraints:
            labels = set(record.get("labelsOrTypes") or [])
            properties = set(record.get("properties") or [])
            if properties == {"name"} and labels and labels.issubset(valid_labels):
                constraint_name = record["name"]
                session.run(f"DROP CONSTRAINT `{constraint_name}` IF EXISTS")

    def _deduplicate_entities_by_name(self, session) -> None:
        for label in ENTITY_LABELS.values():
            duplicates = session.run(
                f"""
                MATCH (e:{label})
                WITH toLower(trim(coalesce(e.name, ''))) AS normalized_name, collect(elementId(e)) AS ids
                WHERE normalized_name <> '' AND size(ids) > 1
                RETURN normalized_name, ids
                """
            )
            for row in duplicates:
                ids = list(row["ids"])
                keep_id = ids[0]
                for duplicate_id in ids[1:]:
                    self._merge_duplicate_entity_node(session, label=label, keep_id=keep_id, duplicate_id=duplicate_id)

    def _backfill_canonical_keys(self, session) -> None:
        for label in ENTITY_LABELS.values():
            records = session.run(
                f"""
                MATCH (e:{label})
                RETURN elementId(e) AS node_id,
                       coalesce(e.name, '') AS name,
                       coalesce(e.kind, $label) AS entity_type,
                       coalesce(e.aliases, []) AS aliases
                """,
                label=label,
            )
            for record in records:
                entity_type = str(record["entity_type"] or label)
                canonical_name = self._canonical_entity_name(str(record["name"] or ""), entity_type)
                canonical_key = self._entity_key(canonical_name, entity_type)
                aliases = list(record["aliases"] or [])
                if canonical_name and canonical_name not in aliases:
                    aliases.append(canonical_name)
                session.run(
                    f"""
                    MATCH (e:{label})
                    WHERE elementId(e) = $node_id
                    SET e.canonical_key = $canonical_key,
                        e.name_key = $name_key,
                        e.kind = $entity_type,
                        e.aliases = $aliases,
                        e.updated_at = datetime()
                    """,
                    node_id=record["node_id"],
                    canonical_key=canonical_key,
                    name_key=normalize_text_key(canonical_name),
                    entity_type=entity_type,
                    aliases=aliases,
                )

    def _deduplicate_entities(self, session) -> None:
        conceptual_labels = ["Entity", "Topic", "Skill", "Document"]
        duplicates = session.run(
            """
            MATCH (e)
            WHERE any(label IN labels(e) WHERE label IN $labels)
              AND e.canonical_key STARTS WITH 'concept::'
            WITH e.canonical_key AS canonical_key, collect(elementId(e)) AS ids
            WHERE size(ids) > 1
            RETURN canonical_key, ids
            """,
            labels=conceptual_labels,
        )
        for row in duplicates:
            ids = list(row["ids"])
            keep_id = ids[0]
            for duplicate_id in ids[1:]:
                self._merge_duplicate_entity_node(session, label=None, keep_id=keep_id, duplicate_id=duplicate_id)

        for label in ENTITY_LABELS.values():
            duplicates = session.run(
                f"""
                MATCH (e:{label})
                WHERE e.canonical_key IS NOT NULL
                WITH e.canonical_key AS canonical_key, collect(elementId(e)) AS ids
                WHERE size(ids) > 1
                RETURN canonical_key, ids
                """
            )
            for row in duplicates:
                ids = list(row["ids"])
                keep_id = ids[0]
                for duplicate_id in ids[1:]:
                    self._merge_duplicate_entity_node(session, label=label, keep_id=keep_id, duplicate_id=duplicate_id)

    def _merge_duplicate_entity_node(self, session, *, label: str | None, keep_id: str, duplicate_id: str) -> None:
        label_match = f":{label}" if label else ""
        session.run(
            f"""
            MATCH (keep{label_match}), (dup{label_match})
            WHERE elementId(keep) = $keep_id AND elementId(dup) = $duplicate_id
            SET keep.name = coalesce(keep.name, dup.name),
                keep.kind = coalesce(keep.kind, dup.kind),
                keep.updated_at = datetime(),
                keep.aliases = reduce(
                    merged = coalesce(keep.aliases, []),
                    alias IN coalesce(keep.aliases, []) + coalesce(dup.aliases, []) |
                    CASE
                        WHEN alias IN merged THEN merged
                        ELSE merged + alias
                    END
                )
            """,
            keep_id=keep_id,
            duplicate_id=duplicate_id,
        )
        if label is None:
            session.run(
                """
                MATCH (keep)
                WHERE elementId(keep) = $keep_id
                SET keep:Entity
                """,
                keep_id=keep_id,
            )
        outgoing = session.run(
            """
            MATCH (dup)-[r]->(target)
            WHERE elementId(dup) = $duplicate_id
            RETURN type(r) AS rel_type, properties(r) AS rel_props, elementId(target) AS target_id
            """,
            duplicate_id=duplicate_id,
        )
        for record in outgoing:
            self._merge_relationship(
                session,
                start_id=keep_id,
                end_id=record["target_id"],
                rel_type=record["rel_type"],
                rel_props=record["rel_props"] or {},
            )

        incoming = session.run(
            """
            MATCH (source)-[r]->(dup)
            WHERE elementId(dup) = $duplicate_id
            RETURN elementId(source) AS source_id, type(r) AS rel_type, properties(r) AS rel_props
            """,
            duplicate_id=duplicate_id,
        )
        for record in incoming:
            self._merge_relationship(
                session,
                start_id=record["source_id"],
                end_id=keep_id,
                rel_type=record["rel_type"],
                rel_props=record["rel_props"] or {},
            )

        session.run(
            """
            MATCH (dup)
            WHERE elementId(dup) = $duplicate_id
            DETACH DELETE dup
            """,
            duplicate_id=duplicate_id,
        )

    @staticmethod
    def _merge_relationship(session, *, start_id: str, end_id: str, rel_type: str, rel_props: dict) -> None:
        escaped_type = rel_type.replace("`", "``")
        session.run(
            f"""
            MATCH (start_node), (end_node)
            WHERE elementId(start_node) = $start_id AND elementId(end_node) = $end_id
            MERGE (start_node)-[r:`{escaped_type}`]->(end_node)
            SET r += $rel_props
            """,
            start_id=start_id,
            end_id=end_id,
            rel_props=rel_props,
        )

    def process_signals(self, *, session, raw_signals: list[dict]) -> dict[str, object]:
        triples = [
            TripleCandidate(
                user_id=str(item.get("user_id") or "").strip(),
                subject_type="User",
                subject_name=str(item.get("user_id") or "").strip(),
                relation=str(item.get("relation") or "RELATED_TO").strip(),
                object_type=str(item.get("entity_type") or "Entity").strip(),
                object_name=str(item.get("entity") or "").strip(),
                confidence=float(item.get("confidence") or 0.0),
                source=str(item.get("source") or "chat").strip(),
                raw_text=str(item.get("raw_text") or "").strip(),
                source_event_id=str(item.get("source_event_id") or "") or None,
                linked_to_action=bool(item.get("linked_to_action")),
            )
            for item in raw_signals
            if self._is_valid_signal(item)
        ]
        return self.process_triples(session=session, triples=triples)

    def process_triples(self, *, session, triples: list[TripleCandidate]) -> dict[str, object]:
        promoted_items: list[dict[str, object]] = []
        deduped_signal_map: dict[tuple[str, str, str, str], MemorySignal] = {}
        for triple in triples:
            if triple.subject_type.strip().lower() != "user" or not triple.subject_name.strip():
                continue
            signal = self._to_signal(
                {
                    "user_id": triple.user_id,
                    "entity": triple.object_name,
                    "relation": triple.relation,
                    "source": triple.source,
                    "raw_text": triple.raw_text,
                    "confidence": triple.confidence,
                    "entity_type": triple.object_type,
                    "linked_to_action": triple.linked_to_action,
                }
            )
            key = (
                signal.user_id.strip().lower(),
                signal.normalized_relation(),
                signal.entity_type.strip().lower(),
                normalize_text_key(signal.entity),
            )
            existing = deduped_signal_map.get(key)
            if existing is None:
                deduped_signal_map[key] = signal
                continue
            existing.confidence = max(existing.confidence, signal.confidence)
            existing.linked_to_action = existing.linked_to_action or signal.linked_to_action
            existing.raw_text = signal.raw_text or existing.raw_text
            if signal.source != existing.source:
                existing.source = f"{existing.source},{signal.source}"
        signals = list(deduped_signal_map.values())
        promoted_aggregates: list[EphemeralAggregate] = []

        for signal in signals:
            aggregate = self.ephemeral_store.upsert_signal(signal)
            if self._should_promote(aggregate):
                self._promote_to_graph(session=session, aggregate=aggregate)
                self.ephemeral_store.mark_promoted(aggregate)
                promoted_items.append(aggregate.to_dict())
                promoted_aggregates.append(aggregate)

        if len(promoted_aggregates) > 1:
            self._link_comentioned_entities(session=session, aggregates=promoted_aggregates)
            self._link_structural_entities(session=session, aggregates=promoted_aggregates)

        structural_candidates = [
            triple for triple in triples if triple.subject_type.strip().lower() != "user"
        ]
        resolved_triples = resolve_triples(session=session, candidates=structural_candidates) if structural_candidates else []
        structural_promoted = 0
        for triple in resolved_triples:
            if triple.subject.kind == "User":
                continue
            if not self._should_promote_triple(triple):
                continue
            self._promote_resolved_triple_to_graph(session=session, triple=triple)
            structural_promoted += 1

        if promoted_items or structural_promoted:
            self._bump_graph_version(signals[0].user_id if signals else triples[0].user_id)

        return {
            "ephemeral_count": len(signals),
            "promoted_count": len(promoted_items) + structural_promoted,
            "promoted_items": promoted_items,
        }

    def _should_promote_triple(self, triple: ResolvedTripleCandidate) -> bool:
        return (
            triple.confidence >= self.confidence_threshold
            or triple.linked_to_action
            or triple.promotion_hint in {"structural", "user_explicit"}
        )

    def _should_promote(self, aggregate: EphemeralAggregate) -> bool:
        return (
            aggregate.max_confidence >= self.confidence_threshold
            or aggregate.mention_count >= self.repeated_mentions_threshold
            or aggregate.linked_to_action
        )

    def _promote_to_graph(self, *, session, aggregate: EphemeralAggregate) -> None:
        relation = self._canonical_relation(aggregate.relation)
        label = ENTITY_LABELS.get(aggregate.entity_type.strip().lower(), "Entity")
        entity_name = self._canonical_entity_name(aggregate.entity, aggregate.entity_type)
        entity_key = self._entity_key(entity_name, aggregate.entity_type, user_id=aggregate.user_id)
        semantics = classify_relation_semantics(relation, entity_type=aggregate.entity_type)

        session.run(
            f"""
            MERGE (u:User {{id: $user_id}})
            SET u.last_seen = datetime()
            MERGE (e:{label} {{canonical_key: $entity_key}})
            ON CREATE SET
                e.created_at = datetime(),
                e.kind = $entity_type,
                e.name = $entity,
                e.name_key = $entity_name_key,
                e.aliases = [$raw_entity]
            SET
                e.name = $entity,
                e.name_key = $entity_name_key,
                e.kind = coalesce(e.kind, $entity_type),
                e.updated_at = datetime(),
                e.aliases = CASE
                    WHEN $raw_entity IN coalesce(e.aliases, []) THEN e.aliases
                    ELSE coalesce(e.aliases, []) + $raw_entity
                END
            MERGE (u)-[r:{relation}]->(e)
            ON CREATE SET
                r.created_at = datetime(),
                r.first_promoted_at = datetime(),
                r.reinforcement_count = 0
            SET
                r.confidence = $confidence,
                r.reinforcement_count = coalesce(r.reinforcement_count, 0) + 1,
                r.last_reinforced = datetime(),
                r.mention_count = $mention_count,
                r.sources = $sources,
                r.linked_to_action = $linked_to_action,
                r.last_signal_text = $last_raw_text,
                r.relation_family = $relation_family,
                r.relation_polarity = $relation_polarity,
                r.section_tags = $section_tags,
                r.relation_strength = $relation_strength
            """,
            user_id=aggregate.user_id,
            entity=entity_name,
            entity_key=entity_key,
            entity_name_key=normalize_text_key(entity_name),
            raw_entity=aggregate.entity,
            entity_type=aggregate.entity_type,
            confidence=aggregate.max_confidence,
            mention_count=aggregate.mention_count,
            sources=aggregate.sources,
            linked_to_action=aggregate.linked_to_action,
            last_raw_text=aggregate.last_raw_text,
            relation_family=semantics.family,
            relation_polarity=semantics.polarity,
            section_tags=list(semantics.section_tags),
            relation_strength=semantics.strength,
        )

    def _promote_resolved_triple_to_graph(self, *, session, triple: ResolvedTripleCandidate) -> None:
        subject_label = self._label_for_kind(triple.subject.kind)
        object_label = self._label_for_kind(triple.object.kind)
        relation = self._canonical_relation(triple.relation)
        semantics = classify_relation_semantics(relation, entity_type=triple.object.kind)
        session.run(
            f"""
            MERGE (subject:{subject_label} {{canonical_key: $subject_id}})
            ON CREATE SET
                subject.created_at = datetime(),
                subject.name = $subject_name,
                subject.name_key = $subject_name_key,
                subject.kind = $subject_kind,
                subject.aliases = $subject_aliases
            SET
                subject.updated_at = datetime(),
                subject.name = $subject_name,
                subject.name_key = $subject_name_key,
                subject.kind = coalesce(subject.kind, $subject_kind),
                subject.aliases = CASE
                    WHEN size($subject_aliases) = 0 THEN coalesce(subject.aliases, [])
                    ELSE reduce(merged = coalesce(subject.aliases, []), alias IN $subject_aliases |
                        CASE WHEN alias IN merged THEN merged ELSE merged + alias END)
                END
            MERGE (object:{object_label} {{canonical_key: $object_id}})
            ON CREATE SET
                object.created_at = datetime(),
                object.name = $object_name,
                object.name_key = $object_name_key,
                object.kind = $object_kind,
                object.aliases = $object_aliases
            SET
                object.updated_at = datetime(),
                object.name = $object_name,
                object.name_key = $object_name_key,
                object.kind = coalesce(object.kind, $object_kind),
                object.aliases = CASE
                    WHEN size($object_aliases) = 0 THEN coalesce(object.aliases, [])
                    ELSE reduce(merged = coalesce(object.aliases, []), alias IN $object_aliases |
                        CASE WHEN alias IN merged THEN merged ELSE merged + alias END)
                END
            MERGE (subject)-[r:{relation}]->(object)
            ON CREATE SET
                r.created_at = datetime(),
                r.reinforcement_count = 0
            SET
                r.confidence = $confidence,
                r.reinforcement_count = coalesce(r.reinforcement_count, 0) + 1,
                r.last_seen = datetime(),
                r.last_signal_text = $raw_text,
                r.relation_family = $relation_family,
                r.relation_polarity = $relation_polarity,
                r.section_tags = $section_tags,
                r.relation_strength = $relation_strength,
                r.sources = CASE
                    WHEN $source IN coalesce(r.sources, []) THEN coalesce(r.sources, [])
                    ELSE coalesce(r.sources, []) + $source
                END
            """,
            subject_id=triple.subject.canonical_id,
            subject_name=triple.subject.name,
            subject_name_key=normalize_text_key(triple.subject.name),
            subject_kind=triple.subject.kind,
            subject_aliases=triple.subject.aliases,
            object_id=triple.object.canonical_id,
            object_name=triple.object.name,
            object_name_key=normalize_text_key(triple.object.name),
            object_kind=triple.object.kind,
            object_aliases=triple.object.aliases,
            confidence=triple.confidence,
            raw_text=triple.raw_text,
            source=triple.source,
            relation_family=semantics.family,
            relation_polarity=semantics.polarity,
            section_tags=list(semantics.section_tags),
            relation_strength=semantics.strength,
        )

    def _link_comentioned_entities(self, *, session, aggregates: list[EphemeralAggregate]) -> None:
        unique_entities: dict[str, tuple[str, str]] = {}
        for aggregate in aggregates:
            entity_name = self._canonical_entity_name(aggregate.entity, aggregate.entity_type)
            entity_key = self._entity_key(entity_name, aggregate.entity_type, user_id=aggregate.user_id)
            unique_entities[entity_key] = (entity_name, aggregate.entity_type)

        entity_keys = list(unique_entities.keys())
        for index, left_key in enumerate(entity_keys):
            for right_key in entity_keys[index + 1 :]:
                session.run(
                    """
                    MATCH (left {canonical_key: $left_key}), (right {canonical_key: $right_key})
                    MERGE (left)-[r:RELATED_TO]->(right)
                    ON CREATE SET r.created_at = datetime(), r.weight = 0
                    SET r.weight = coalesce(r.weight, 0) + 1,
                        r.last_seen = datetime(),
                        r.relation_family = 'association',
                        r.relation_polarity = 'neutral',
                        r.section_tags = ['related'],
                        r.relation_strength = 0.45
                    MERGE (right)-[r2:RELATED_TO]->(left)
                    ON CREATE SET r2.created_at = datetime(), r2.weight = 0
                    SET r2.weight = coalesce(r2.weight, 0) + 1,
                        r2.last_seen = datetime(),
                        r2.relation_family = 'association',
                        r2.relation_polarity = 'neutral',
                        r2.section_tags = ['related'],
                        r2.relation_strength = 0.45
                    """,
                    left_key=left_key,
                    right_key=right_key,
                )

    def _link_structural_entities(self, *, session, aggregates: list[EphemeralAggregate]) -> None:
        unique_entities: list[tuple[str, str, str, str]] = []
        seen: set[str] = set()
        message_text = " ".join(
            aggregate.last_raw_text.strip()
            for aggregate in aggregates
            if aggregate.last_raw_text.strip()
        ).lower()
        if not message_text:
            return

        for aggregate in aggregates:
            entity_name = self._canonical_entity_name(aggregate.entity, aggregate.entity_type)
            entity_key = self._entity_key(entity_name, aggregate.entity_type, user_id=aggregate.user_id)
            if entity_key in seen:
                continue
            seen.add(entity_key)
            unique_entities.append((entity_key, entity_name, aggregate.entity_type, normalize_text_key(entity_name)))

        for left_key, left_name, _left_type, left_token in unique_entities:
            for right_key, right_name, _right_type, right_token in unique_entities:
                if left_key == right_key:
                    continue
                if not left_token or not right_token:
                    continue
                left_pos = message_text.find(left_token)
                right_pos = message_text.find(right_token)
                if left_pos < 0 or right_pos < 0 or left_pos >= right_pos:
                    continue
                between = message_text[left_pos + len(left_token):right_pos]
                structural_relation = self._classify_structural_relation(between)
                if not structural_relation:
                    continue
                session.run(
                    f"""
                    MATCH (left {{canonical_key: $left_key}}), (right {{canonical_key: $right_key}})
                    MERGE (left)-[r:{structural_relation}]->(right)
                    ON CREATE SET r.created_at = datetime(), r.weight = 0
                    SET r.weight = coalesce(r.weight, 0) + 1,
                        r.last_seen = datetime(),
                        r.evidence = $between,
                        r.relation_family = 'structure',
                        r.relation_polarity = 'neutral',
                        r.section_tags = ['topic', 'structure', 'related'],
                        r.relation_strength = 0.72
                    """,
                    left_key=left_key,
                    right_key=right_key,
                    between=between.strip(),
                )

    @staticmethod
    def _classify_structural_relation(text_between: str) -> str | None:
        cleaned = re.sub(r"\s+", " ", text_between.strip().lower())
        if not cleaned:
            return None
        for relation, phrases in STRUCTURAL_EDGE_PATTERNS:
            if any(phrase in cleaned for phrase in phrases):
                return relation
        if " of " in f" {cleaned} ":
            return "PART_OF"
        return None

    @staticmethod
    def _query_terms(query: str) -> set[str]:
        return {
            token
            for token in normalize_text_key(query).split()
            if token and token not in STOPWORDS and len(token) > 1
        }

    @classmethod
    def _node_overlap_score(cls, *, node: dict[str, object], terms: set[str]) -> float:
        if not terms:
            return 0.0
        texts = [str(node.get("label") or "")]
        texts.extend(str(alias) for alias in list(node.get("aliases") or []))
        best = 0.0
        for text in texts:
            words = set(normalize_text_key(text).split())
            overlap = terms & words
            if overlap:
                best = max(best, len(overlap) / max(1, len(terms)))
        return best

    def fetch_graph_context(self, *, session, user_id: str, limit: int = 6) -> list[str]:
        cache_key = f"graphmind:graph_context:{user_id}:{self._graph_version(user_id)}:{limit}"
        cached = self._cache_get(cache_key)
        if isinstance(cached, list):
            return [str(item) for item in cached]
        records = self.fetch_graph_memory(session=session, user_id=user_id, limit=limit)
        context: list[str] = []
        for record in records:
            if record["entity"]:
                context.append(
                    f"{record['relation']} -> {record['entity']} ({record['entity_type']}, confidence {record['confidence']:.2f}, reinforcements {record['reinforcement_count']})"
                )
        linked_records = session.run(
            """
            MATCH (u:User {id: $user_id})-->(e)-[r]->(other)
            WHERE type(r) IN ['PART_OF', 'USED_IN', 'DEPENDS_ON', 'RELATED_TO']
              AND NOT e:Conversation AND NOT e:Message
              AND NOT other:Conversation AND NOT other:Message
            RETURN e.name AS source,
                   type(r) AS relation,
                   other.name AS target
            LIMIT $limit
            """,
            user_id=user_id,
            limit=max(1, min(limit, 20)),
        )
        for record in linked_records:
            context.append(f"{record['source']} {record['relation']} {record['target']}")
        self._cache_set(cache_key, context, ttl_seconds=600)
        return context

    def fetch_section_context(
        self,
        *,
        session,
        user_id: str,
        section_tags: list[str],
        section_families: list[str],
        focus_entity: str | None = None,
        query: str | None = None,
        limit: int = 8,
    ) -> list[str]:
        if not section_tags and not section_families:
            return []
        focus_key = normalize_text_key(focus_entity or "")
        query_terms = self._query_terms(query or "")
        query_embedding = self._hash_embedding(query or "")
        strict_topic_relevance = bool(({"topic", "learning"} & set(section_tags)) and not focus_key and query_terms)
        records = session.run(
            """
            MATCH (u:User {id: $user_id})-[r]->(e)
            WHERE (
                size($section_tags) = 0
                OR any(tag IN coalesce(r.section_tags, []) WHERE tag IN $section_tags)
            ) OR (
                size($section_families) = 0
                OR coalesce(r.relation_family, '') IN $section_families
            )
            OPTIONAL MATCH (e)-[rel]->(related)
            WHERE type(rel) IN ['PART_OF', 'USED_IN', 'DEPENDS_ON', 'RELATED_TO']
              AND NOT related:Conversation AND NOT related:Message
            RETURN type(r) AS relation,
                   coalesce(r.relation_family, '') AS relation_family,
                   coalesce(r.relation_polarity, 'neutral') AS relation_polarity,
                   coalesce(r.section_tags, []) AS section_tags,
                   e.name AS entity,
                   coalesce(e.kind, head(labels(e)), 'Entity') AS entity_type,
                   coalesce(r.confidence, 0.0) AS confidence,
                   coalesce(r.reinforcement_count, 0) AS reinforcement_count,
                   related.name AS related_entity,
                   type(rel) AS related_relation
            LIMIT $limit
            """,
            user_id=user_id,
            section_tags=section_tags,
            section_families=section_families,
            limit=max(1, min(limit * 3, 24)),
        )
        lines: list[tuple[float, str]] = []
        for record in records:
            entity = str(record.get("entity") or "").strip()
            if not entity:
                continue
            relation = self._canonical_relation(str(record.get("relation") or "RELATED_TO"))
            confidence = float(record.get("confidence") or 0.0)
            reinforcement = float(record.get("reinforcement_count") or 0.0)
            score = confidence + min(reinforcement / 4.0, 1.0)
            entity_node = {"label": entity, "aliases": []}
            relevance = 0.0
            if query_terms:
                relevance = max(
                    self._node_overlap_score(node=entity_node, terms=query_terms),
                    self._node_semantic_score(node=entity_node, query_embedding=query_embedding),
                )
            if focus_key and focus_key in normalize_text_key(entity):
                score += 0.5
                relevance = max(relevance, 0.75)
            line = f"{relation} -> {entity} ({record.get('entity_type')}, family {record.get('relation_family')}, polarity {record.get('relation_polarity')})"
            related_entity = str(record.get("related_entity") or "").strip()
            related_relation = str(record.get("related_relation") or "").strip()
            if related_entity and related_relation:
                line += f" | related: {entity} {related_relation} {related_entity}"
                if focus_key and focus_key in normalize_text_key(related_entity):
                    score += 0.35
                    relevance = max(relevance, 0.6)
                elif query_terms:
                    related_node = {"label": related_entity, "aliases": []}
                    relevance = max(
                        relevance,
                        self._node_overlap_score(node=related_node, terms=query_terms),
                        self._node_semantic_score(node=related_node, query_embedding=query_embedding),
                    )
            if strict_topic_relevance and relevance < 0.24:
                continue
            score += relevance * 0.9
            lines.append((score, line))
        lines.sort(key=lambda item: item[0], reverse=True)
        deduped: list[str] = []
        for _, line in lines:
            if line not in deduped:
                deduped.append(line)
            if len(deduped) >= limit:
                break
        return deduped

    def fetch_graph_evidence(
        self,
        *,
        session,
        user_id: str,
        query: str,
        limit: int = 6,
    ) -> dict[str, list[dict[str, object]] | list[str]]:
        cache_key = f"graphmind:graph_evidence:{user_id}:{self._graph_version(user_id)}:{normalize_text_key(query)}:{limit}"
        cached = self._cache_get(cache_key)
        if isinstance(cached, dict):
            return cached
        graph = self.fetch_graph_view(session=session, user_id=user_id, limit=60)
        nodes = list(graph.get("nodes") or [])
        edges = list(graph.get("edges") or [])
        user_node_id = f"user::{user_id}"
        node_map = {str(node["id"]): node for node in nodes}
        terms = self._query_terms(query)
        query_embedding = self._hash_embedding(query)

        outgoing_memory: dict[str, list[dict[str, object]]] = defaultdict(list)
        adjacency: dict[str, list[dict[str, object]]] = defaultdict(list)
        for edge in edges:
            source = str(edge.get("source") or "")
            target = str(edge.get("target") or "")
            if edge.get("kind") == "memory" and source == user_node_id:
                outgoing_memory[target].append(edge)
            elif edge.get("kind") == "entity":
                adjacency[source].append(edge)
                adjacency[target].append(edge)

        ranked: list[tuple[float, str]] = []
        for node in nodes:
            node_id = str(node.get("id") or "")
            if node_id == user_node_id:
                continue
            score = self._retrieval_score(
                node=node,
                query_terms=terms,
                query_embedding=query_embedding,
                memory_edges=outgoing_memory.get(node_id, []),
                related_edges=adjacency.get(node_id, []),
            )
            if score > 0:
                ranked.append((score, node_id))

        if not ranked:
            for node_id, memory_edges in outgoing_memory.items():
                ranked.append(
                    (
                        self._retrieval_score(
                            node=node_map.get(node_id, {"id": node_id, "label": node_id}),
                            query_terms=terms,
                            query_embedding=query_embedding,
                            memory_edges=memory_edges,
                            related_edges=adjacency.get(node_id, []),
                        ),
                        node_id,
                    )
                )

        ranked.sort(reverse=True)
        seed_node_ids = [node_id for _, node_id in ranked[: max(2, min(limit, 10))]]
        chosen_node_ids = self._best_first_subgraph(
            user_node_id=user_node_id,
            seed_node_ids=seed_node_ids,
            outgoing_memory=outgoing_memory,
            adjacency=adjacency,
            node_map=node_map,
            query_terms=terms,
            query_embedding=query_embedding,
            limit=max(1, min(limit, 10)),
            max_hops=3,
        )

        facts: list[dict[str, object]] = []
        paths: list[str] = []
        citations: list[str] = []
        seen_paths: set[str] = set()

        for node_id in chosen_node_ids:
            node = node_map.get(node_id)
            if not node:
                continue
            for memory_edge in outgoing_memory.get(node_id, []):
                path = f"{user_id} -[{memory_edge['label']}]-> {node['label']}"
                if path not in seen_paths:
                    seen_paths.add(path)
                    paths.append(path)
                    citations.append(path)
                facts.append(
                    {
                        "kind": "memory",
                        "text": path,
                        "edge": memory_edge["label"],
                        "node": node["label"],
                        "node_type": node.get("type"),
                        "weight": memory_edge.get("weight", 1),
                        "score": round(self._weighted_edge_score(memory_edge, overlap=self._node_overlap_score(node=node, terms=terms)), 4),
                    }
                )

            for edge in adjacency.get(node_id, []):
                source = str(edge.get("source") or "")
                target = str(edge.get("target") or "")
                other_id = target if source == node_id else source
                other = node_map.get(other_id)
                if not other:
                    continue
                if edge.get("label") == "RELATED_TO" and any(
                    candidate.get("label") != "RELATED_TO"
                    and {str(candidate.get("source") or ""), str(candidate.get("target") or "")} == {node_id, other_id}
                    for candidate in adjacency.get(node_id, [])
                ):
                    continue
                if source == node_id:
                    path = f"{node['label']} -[{edge['label']}]-> {other['label']}"
                else:
                    path = f"{other['label']} -[{edge['label']}]-> {node['label']}"
                if path in seen_paths:
                    continue
                seen_paths.add(path)
                paths.append(path)
                citations.append(path)
                facts.append(
                    {
                        "kind": "graph",
                        "text": path,
                        "edge": edge["label"],
                        "node": node["label"],
                        "related_node": other["label"],
                        "node_type": node.get("type"),
                        "related_type": other.get("type"),
                        "weight": edge.get("weight", 1),
                        "score": round(self._weighted_edge_score(edge, overlap=self._node_overlap_score(node=node, terms=terms)), 4),
                    }
                )

        result = {
            "facts": facts[: max(1, min(limit * 2, 20))],
            "paths": paths[: max(1, min(limit, 12))],
            "citations": citations[: max(1, min(limit, 12))],
        }
        self._cache_set(cache_key, result, ttl_seconds=600)
        return result

    def fetch_graph_memory(self, *, session, user_id: str, limit: int = 10) -> list[dict[str, object]]:
        cache_key = f"graphmind:graph_memory:{user_id}:{self._graph_version(user_id)}:{limit}"
        cached = self._cache_get(cache_key)
        if isinstance(cached, list):
            return cached
        records = session.run(
            """
            MATCH (u:User {id: $user_id})-[r]->(e)
            WHERE NOT e:Conversation AND NOT e:Message
            OPTIONAL MATCH (e)-[:RELATED_TO]->(related)
            WHERE NOT related:Conversation AND NOT related:Message
            RETURN type(r) AS relation,
                   e.name AS entity,
                   coalesce(e.aliases, []) AS aliases,
                   coalesce(e.kind, head(labels(e)), "Entity") AS entity_type,
                   coalesce(r.confidence, 0.0) AS confidence,
                   coalesce(r.reinforcement_count, 0) AS reinforcement_count,
                   coalesce(r.mention_count, 0) AS mention_count,
                   coalesce(r.sources, []) AS sources,
                   coalesce(r.section_tags, []) AS section_tags,
                   coalesce(r.relation_family, '') AS relation_family,
                   coalesce(r.relation_polarity, 'neutral') AS relation_polarity,
                   count(related) AS related_count,
                   r.last_reinforced AS last_reinforced
            ORDER BY coalesce(last_reinforced, datetime("1970-01-01T00:00:00Z")) DESC,
                     reinforcement_count DESC
            LIMIT $limit
            """,
            user_id=user_id,
            limit=max(1, min(limit, 50)),
        )
        merged: dict[tuple[str, str, str], dict[str, object]] = {}
        for record in records:
            item = dict(record)
            item["relation"] = self._canonical_relation(str(item.get("relation") or "RELATED_TO"))
            item["entity_type"] = str(item.get("entity_type") or "Entity")
            item["entity"] = self._canonical_entity_name(
                str(item.get("entity") or ""),
                item["entity_type"],
            )
            if item.get("last_reinforced") is not None:
                item["last_reinforced"] = str(item["last_reinforced"])
            key = (
                str(item["relation"]),
                str(item["entity_type"]).lower(),
                normalize_text_key(str(item["entity"])),
            )
            existing = merged.get(key)
            if existing is None:
                merged[key] = item
                continue

            existing["confidence"] = max(
                float(existing.get("confidence") or 0.0),
                float(item.get("confidence") or 0.0),
            )
            existing["reinforcement_count"] = int(existing.get("reinforcement_count") or 0) + int(
                item.get("reinforcement_count") or 0
            )
            existing["mention_count"] = int(existing.get("mention_count") or 0) + int(
                item.get("mention_count") or 0
            )
            existing["related_count"] = max(
                int(existing.get("related_count") or 0),
                int(item.get("related_count") or 0),
            )
            existing["sources"] = sorted(
                {
                    *list(existing.get("sources") or []),
                    *list(item.get("sources") or []),
                }
            )
            existing["aliases"] = sorted(
                {
                    *list(existing.get("aliases") or []),
                    *list(item.get("aliases") or []),
                    str(existing.get("entity") or ""),
                    str(item.get("entity") or ""),
                }
            )
            existing_last = str(existing.get("last_reinforced") or "")
            item_last = str(item.get("last_reinforced") or "")
            if item_last > existing_last:
                existing["last_reinforced"] = item_last

        items = list(merged.values())
        items.sort(
            key=lambda item: (
                str(item.get("last_reinforced") or ""),
                int(item.get("reinforcement_count") or 0),
            ),
            reverse=True,
        )
        result = items[: max(1, min(limit, 50))]
        self._cache_set(cache_key, result, ttl_seconds=600)
        return result

    def fetch_ephemeral_memory(self, *, user_id: str, limit: int = 20) -> list[dict[str, object]]:
        items = self.ephemeral_store.list_user_items(user_id=user_id, limit=limit)
        return [item.to_dict() for item in items]

    def reset_user_memory(self, *, session, user_id: str) -> dict[str, int]:
        attached_node_ids = [
            str(record["node_id"])
            for record in session.run(
                """
                MATCH (:User {id: $user_id})--(n)
                WHERE NOT n:Conversation AND NOT n:Message
                RETURN DISTINCT elementId(n) AS node_id
                """,
                user_id=user_id,
            )
        ]
        conversations_deleted = session.run(
            """
            MATCH (u:User {id: $user_id})-[:HAS_CONVERSATION]->(c:Conversation)
            OPTIONAL MATCH (c)-[:HAS_MESSAGE]->(m:Message)
            DETACH DELETE c, m
            RETURN count(DISTINCT c) AS conversations, count(DISTINCT m) AS messages
            """,
            user_id=user_id,
        ).single()
        relationship_deleted = session.run(
            """
            MATCH (:User {id: $user_id})-[r]-()
            DELETE r
            RETURN count(r) AS count
            """,
            user_id=user_id,
        ).single()
        user_deleted = session.run(
            """
            MATCH (u:User {id: $user_id})
            DETACH DELETE u
            RETURN count(u) AS count
            """,
            user_id=user_id,
        ).single()
        orphan_cleanup = session.run(
            """
            UNWIND $node_ids AS node_id
            MATCH (n)
            WHERE elementId(n) = node_id
              AND NOT n:User
              AND NOT n:Conversation
              AND NOT n:Message
              AND NOT (n)--()
            DETACH DELETE n
            RETURN count(n) AS count
            """,
            node_ids=attached_node_ids,
        ).single()
        cache_summary = self.ephemeral_store.reset_user_items(user_id=user_id)
        return {
            "user_nodes_deleted": int((user_deleted or {}).get("count", 0)),
            "user_relationships_deleted": int((relationship_deleted or {}).get("count", 0)),
            "conversations_deleted": int((conversations_deleted or {}).get("conversations", 0)),
            "messages_deleted": int((conversations_deleted or {}).get("messages", 0)),
            "orphan_nodes_deleted": int((orphan_cleanup or {}).get("count", 0)),
            "ephemeral_keys_deleted": int(cache_summary.get("deleted_keys", 0)),
        }

    def fetch_graph_view(self, *, session, user_id: str, limit: int = 24) -> dict[str, list[dict[str, object]]]:
        cache_key = f"graphmind:graph_view:{user_id}:{self._graph_version(user_id)}:{limit}"
        cached = self._cache_get(cache_key)
        if isinstance(cached, dict):
            return cached
        records = session.run(
            """
            MATCH (u:User {id: $user_id})-[r]->(e)
            WHERE NOT e:Conversation AND NOT e:Message
            RETURN u.id AS user_id,
                   type(r) AS relation,
                   coalesce(r.reinforcement_count, 0) AS reinforcement_count,
                   coalesce(r.confidence, 0.0) AS confidence,
                   r.last_reinforced AS last_reinforced,
                   e.canonical_key AS entity_key,
                   e.name AS entity,
                   coalesce(e.aliases, []) AS aliases,
                   coalesce(e.kind, head(labels(e)), "Entity") AS entity_type
            ORDER BY coalesce(r.last_reinforced, datetime("1970-01-01T00:00:00Z")) DESC,
                     reinforcement_count DESC,
                     confidence DESC
            LIMIT $limit
            """,
            user_id=user_id,
            limit=max(1, min(limit, 100)),
        )

        nodes: dict[str, dict[str, object]] = {
            f"user::{user_id}": {
                "id": f"user::{user_id}",
                "label": user_id,
                "type": "User",
                "size": 28,
            }
        }
        edges: dict[tuple[str, str, str], dict[str, object]] = {}

        for record in records:
            entity_key = str(record.get("entity_key") or "")
            entity_name = self._canonical_entity_name(
                str(record.get("entity") or ""),
                str(record.get("entity_type") or "Entity"),
            )
            entity_type = str(record.get("entity_type") or "Entity")
            if not entity_key:
                continue
            nodes[entity_key] = {
                "id": entity_key,
                "label": entity_name,
                "type": entity_type,
                "aliases": list(record.get("aliases") or []),
                "size": 16 + min(int(record.get("reinforcement_count") or 0) * 2, 14),
            }
            user_edge_key = (f"user::{user_id}", entity_key, self._canonical_relation(str(record.get("relation") or "RELATED_TO")))
            edges[user_edge_key] = {
                "source": f"user::{user_id}",
                "target": entity_key,
                "label": user_edge_key[2],
                "weight": max(1, int(record.get("reinforcement_count") or 1)),
                "confidence": float(record.get("confidence") or 0.0),
                "last_seen": str(record.get("last_reinforced") or ""),
                "relation_strength": self._relation_strength(user_edge_key[2]),
                "kind": "memory",
            }

        linked_records = session.run(
            """
            MATCH (u:User {id: $user_id})-->(left)-[r]->(right)
            WHERE type(r) IN ['RELATED_TO', 'PART_OF', 'USED_IN', 'DEPENDS_ON']
              AND NOT left:Conversation AND NOT left:Message
              AND NOT right:Conversation AND NOT right:Message
            RETURN left.canonical_key AS left_key,
                   left.name AS left_name,
                   coalesce(left.kind, head(labels(left)), "Entity") AS left_type,
                   right.canonical_key AS right_key,
                   right.name AS right_name,
                   coalesce(right.kind, head(labels(right)), "Entity") AS right_type,
                   type(r) AS relation,
                   coalesce(r.weight, 0) AS weight,
                   r.last_seen AS last_seen,
                   coalesce(r.confidence, 0.0) AS confidence
            LIMIT $limit
            """,
            user_id=user_id,
            limit=max(1, min(limit * 3, 150)),
        )
        structural_pairs: set[tuple[str, str]] = set()
        buffered_edges: list[dict[str, object]] = []
        for record in linked_records:
            left_key = str(record.get("left_key") or "")
            right_key = str(record.get("right_key") or "")
            if not left_key or not right_key or left_key == right_key:
                continue
            left_name = self._canonical_entity_name(str(record.get("left_name") or ""), str(record.get("left_type") or "Entity"))
            right_name = self._canonical_entity_name(str(record.get("right_name") or ""), str(record.get("right_type") or "Entity"))
            left_type = str(record.get("left_type") or "Entity")
            right_type = str(record.get("right_type") or "Entity")
            nodes.setdefault(left_key, {"id": left_key, "label": left_name, "type": left_type, "size": 16})
            nodes.setdefault(right_key, {"id": right_key, "label": right_name, "type": right_type, "size": 16})
            relation = str(record.get("relation") or "RELATED_TO")
            source, target = (sorted([left_key, right_key]) if relation == "RELATED_TO" else (left_key, right_key))
            if relation != "RELATED_TO":
                structural_pairs.add(tuple(sorted([left_key, right_key])))
            buffered_edges.append({
                "source": source,
                "target": target,
                "label": relation,
                "weight": max(1, int(record.get("weight") or 1)),
                "confidence": float(record.get("confidence") or 0.0),
                "last_seen": str(record.get("last_seen") or ""),
                "relation_strength": self._relation_strength(relation),
                "kind": "entity",
            })

        for edge in buffered_edges:
            pair_key = tuple(sorted([str(edge["source"]), str(edge["target"])]))
            if edge["label"] == "RELATED_TO" and pair_key in structural_pairs:
                continue
            edge_key = (str(edge["source"]), str(edge["target"]), str(edge["label"]))
            edges[edge_key] = edge

        result = {
            "nodes": list(nodes.values()),
            "edges": list(edges.values()),
        }
        self._cache_set(cache_key, result, ttl_seconds=600)
        return result

    @staticmethod
    def _sanitize_relation(relation: str) -> str:
        cleaned = "".join(ch if ch.isalnum() else "_" for ch in relation.upper())
        cleaned = "_".join(part for part in cleaned.split("_") if part)
        return cleaned or "RELATED_TO"

    @classmethod
    def _canonical_relation(cls, relation: str) -> str:
        sanitized = cls._sanitize_relation(relation)
        return RELATION_ALIASES.get(sanitized, sanitized)

    @staticmethod
    def _canonical_entity_name(entity: str, entity_type: str) -> str:
        cleaned = re.sub(r"\s+", " ", (entity or "").strip())
        lowered_type = entity_type.strip().lower()
        text_key = normalize_text_key(cleaned)

        if lowered_type == "company":
            return cleaned.title()
        if lowered_type == "topic":
            return TOPIC_ALIASES.get(text_key, cleaned.title())
        if lowered_type in {"skill", "goal", "document"}:
            return cleaned.title()
        return cleaned

    @staticmethod
    def _entity_key(entity: str, entity_type: str, user_id: str | None = None) -> str:
        lowered_type = entity_type.strip().lower()
        key_type = "concept" if lowered_type in CONCEPTUAL_ENTITY_TYPES else lowered_type
        entity_key = normalize_text_key(entity)
        if user_id:
            return f"user::{normalize_text_key(user_id)}::{key_type}::{entity_key}"
        return f"{key_type}::{entity_key}"

    @staticmethod
    def _label_for_kind(kind: str) -> str:
        normalized = (kind or "Entity").strip().lower()
        if normalized == "concept":
            return "Entity"
        return ENTITY_LABELS.get(normalized, kind.strip() or "Entity")

    @staticmethod
    def _is_valid_signal(raw_signal: dict) -> bool:
        return bool(str(raw_signal.get("entity") or "").strip())

    @staticmethod
    def _to_signal(raw_signal: dict) -> MemorySignal:
        return MemorySignal(
            user_id=str(raw_signal.get("user_id") or "").strip(),
            entity=str(raw_signal.get("entity") or "").strip(),
            relation=str(raw_signal.get("relation") or "RELATED_TO").strip(),
            source=str(raw_signal.get("source") or "chat").strip(),
            raw_text=str(raw_signal.get("raw_text") or "").strip(),
            confidence=float(raw_signal.get("confidence") or 0.0),
            entity_type=str(raw_signal.get("entity_type") or "Entity").strip(),
            linked_to_action=bool(raw_signal.get("linked_to_action")),
        )

    @staticmethod
    def _safe_datetime(value: str | object | None) -> datetime | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        normalized = text.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    @classmethod
    def _relation_strength(cls, relation: str) -> float:
        return RELATION_STRENGTHS.get(cls._canonical_relation(relation), 0.5)

    @classmethod
    def _recency_score(cls, value: str | object | None, *, half_life_days: float = 14.0) -> float:
        dt = cls._safe_datetime(value)
        if dt is None:
            return 0.0
        age_days = max(0.0, (datetime.now(timezone.utc) - dt).total_seconds() / 86400.0)
        return math.exp(-math.log(2) * age_days / max(0.5, half_life_days))

    @classmethod
    def _weighted_edge_score(cls, edge: dict[str, object], *, overlap: float = 0.0) -> float:
        confidence = max(0.0, min(float(edge.get("confidence") or 0.0), 1.0))
        recency = cls._recency_score(edge.get("last_seen"))
        reinforcement = min(float(edge.get("weight") or 1.0) / 5.0, 1.0)
        relation_strength = float(edge.get("relation_strength") or cls._relation_strength(str(edge.get("label") or "")))
        return (
            0.34 * confidence
            + 0.22 * recency
            + 0.24 * reinforcement
            + 0.20 * relation_strength
            + 0.30 * overlap
        )

    def _best_first_subgraph(
        self,
        *,
        user_node_id: str,
        seed_node_ids: list[str],
        outgoing_memory: dict[str, list[dict[str, object]]],
        adjacency: dict[str, list[dict[str, object]]],
        node_map: dict[str, dict[str, object]],
        query_terms: set[str],
        query_embedding: list[float],
        limit: int,
        max_hops: int,
    ) -> list[str]:
        if not seed_node_ids:
            seed_node_ids = list(outgoing_memory.keys())[:limit]

        heap: list[tuple[float, int, str]] = []
        best_hops: dict[str, int] = {}
        chosen: list[str] = []
        chosen_set: set[str] = set()

        for node_id in seed_node_ids:
            node = node_map.get(node_id)
            if not node:
                continue
            seed_score = max(
                [
                    self._retrieval_score(
                        node=node,
                        query_terms=query_terms,
                        query_embedding=query_embedding,
                        memory_edges=outgoing_memory.get(node_id, []),
                        related_edges=adjacency.get(node_id, []),
                    )
                ]
                or [0.0]
            )
            heapq.heappush(heap, (-seed_score, 1, node_id))

        while heap and len(chosen) < limit:
            neg_score, hops, node_id = heapq.heappop(heap)
            if hops > max_hops:
                continue
            if node_id in best_hops and hops >= best_hops[node_id]:
                continue
            best_hops[node_id] = hops
            if node_id != user_node_id and node_id not in chosen_set:
                chosen.append(node_id)
                chosen_set.add(node_id)

            node = node_map.get(node_id)
            if not node:
                continue

            for edge in adjacency.get(node_id, []):
                source = str(edge.get("source") or "")
                target = str(edge.get("target") or "")
                neighbor_id = target if source == node_id else source
                if neighbor_id == user_node_id:
                    continue
                neighbor = node_map.get(neighbor_id)
                if not neighbor:
                    continue
                neighbor_score = self._retrieval_score(
                    node=neighbor,
                    query_terms=query_terms,
                    query_embedding=query_embedding,
                    memory_edges=outgoing_memory.get(neighbor_id, []),
                    related_edges=adjacency.get(neighbor_id, []),
                )
                edge_score = self._weighted_edge_score(
                    edge,
                    overlap=self._node_overlap_score(node=neighbor, terms=query_terms),
                )
                priority = 0.55 * neighbor_score + 0.25 * edge_score + 0.20 * max(0.0, float(-neg_score))
                heapq.heappush(heap, (-priority, hops + 1, neighbor_id))

        return chosen[:limit]

    @staticmethod
    def _hash_embedding(text: str, dimensions: int = 96) -> list[float]:
        values = [0.0] * dimensions
        tokens = [token for token in normalize_text_key(text).split() if token]
        if not tokens:
            return values
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = digest[0] % dimensions
            sign = 1.0 if digest[1] % 2 == 0 else -1.0
            values[index] += sign
        norm = math.sqrt(sum(value * value for value in values))
        if norm == 0.0:
            return values
        return [value / norm for value in values]

    @classmethod
    def _semantic_similarity(cls, left: list[float], right: list[float]) -> float:
        if not left or not right or len(left) != len(right):
            return 0.0
        return max(0.0, min(1.0, sum(a * b for a, b in zip(left, right))))

    @classmethod
    def _node_semantic_score(cls, *, node: dict[str, object], query_embedding: list[float]) -> float:
        candidates = [str(node.get("label") or "").strip()]
        candidates.extend(str(alias).strip() for alias in list(node.get("aliases") or []))
        best = 0.0
        for candidate in candidates:
            if not candidate:
                continue
            best = max(best, cls._semantic_similarity(cls._hash_embedding(candidate), query_embedding))
        return best

    @classmethod
    def _retrieval_score(
        cls,
        *,
        node: dict[str, object],
        query_terms: set[str],
        query_embedding: list[float],
        memory_edges: list[dict[str, object]],
        related_edges: list[dict[str, object]],
    ) -> float:
        overlap = cls._node_overlap_score(node=node, terms=query_terms)
        lexical = max(0.0, min(1.0, overlap))
        semantic = max(lexical, cls._node_semantic_score(node=node, query_embedding=query_embedding))
        memory_weight = max(
            [cls._weighted_edge_score(edge, overlap=overlap) for edge in memory_edges] or [0.0]
        )
        graph_bonus = max(
            [cls._weighted_edge_score(edge, overlap=overlap) for edge in related_edges] or [0.0]
        )
        memory_component = max(0.0, min(1.0, memory_weight))
        graph_component = max(0.0, min(1.0, graph_bonus))
        freshness = max(
            [cls._recency_score(edge.get("last_seen")) for edge in memory_edges] or [0.0]
        )
        base_score = 0.50 * semantic + 0.30 * memory_component + 0.20 * graph_component
        recency_multiplier = 1.15 if freshness >= 0.86 else (0.80 + 0.35 * freshness)
        return base_score * recency_multiplier


graph_memory_service = GraphMemoryService()
