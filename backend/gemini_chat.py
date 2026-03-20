import hashlib
import json
import os
import re
import sqlite3
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from groq import Groq

from .graph.models import TripleCandidate

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

_client: genai.Client | None = None
_groq_client: Groq | None = None
CHAT_PROVIDER = os.getenv("GRAPHMIND_CHAT_PROVIDER", "groq").strip().lower()
CHAT_MODEL = os.getenv("GROQ_CHAT_MODEL", "llama-3.1-8b-instant")
SIGNAL_MODEL = os.getenv("GEMINI_SIGNAL_MODEL", "gemini-flash-latest")
EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
REPLY_MAX_TOKENS = int(os.getenv("GRAPHMIND_REPLY_MAX_TOKENS", "180"))
REPLY_CONTEXT_ITEMS = int(os.getenv("GRAPHMIND_REPLY_CONTEXT_ITEMS", "4"))
REPLY_ITEM_CHARS = int(os.getenv("GRAPHMIND_REPLY_ITEM_CHARS", "120"))
SEMANTIC_RELATION_ALIASES = {
    "WORKED_ON": "STUDIES",
    "WORKS_ON": "STUDIES",
    "STUDIED": "STUDIES",
    "LEARNED": "STUDIES",
    "LEARNING": "STUDIES",
}

COMPANY_STOPWORDS = {
    "a",
    "an",
    "hr",
    "interview",
    "interviews",
    "technical",
    "behavioral",
    "behavioural",
    "system",
    "design",
    "coding",
    "software",
    "the",
    "role",
    "job",
}


def _extraction_cache_path() -> Path:
    return BASE_DIR / "extraction_cache.sqlite3"


def _connect_extraction_cache() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_extraction_cache_path()))
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS extraction_cache (
            hash TEXT PRIMARY KEY,
            user_id TEXT NOT NULL DEFAULT '',
            source TEXT NOT NULL DEFAULT 'chat',
            message TEXT NOT NULL,
            result TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_extraction_cache_created_at ON extraction_cache(created_at)"
    )
    return conn


def _extraction_cache_key(*, user_id: str, source: str, message: str) -> str:
    payload = f"{user_id.strip().lower()}|{source.strip().lower()}|{' '.join(message.split()).strip().lower()}"
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def _load_cached_extraction(*, user_id: str, source: str, message: str) -> list[TripleCandidate] | None:
    cache_key = _extraction_cache_key(user_id=user_id, source=source, message=message)
    conn = _connect_extraction_cache()
    try:
        row = conn.execute(
            "SELECT result FROM extraction_cache WHERE hash = ?",
            (cache_key,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    try:
        payload = json.loads(str(row["result"] or "[]"))
    except json.JSONDecodeError:
        return None
    triples: list[TripleCandidate] = []
    for item in payload if isinstance(payload, list) else []:
        if not isinstance(item, dict):
            continue
        try:
            triples.append(
                TripleCandidate(
                    user_id=str(item.get("user_id") or user_id),
                    subject_type=str(item.get("subject_type") or "User"),
                    subject_name=str(item.get("subject_name") or user_id),
                    relation=str(item.get("relation") or "RELATED_TO"),
                    object_type=str(item.get("object_type") or "Entity"),
                    object_name=str(item.get("object_name") or ""),
                    confidence=float(item.get("confidence") or 0.0),
                    source=str(item.get("source") or source),
                    raw_text=str(item.get("raw_text") or message),
                    source_event_id=str(item.get("source_event_id")) if item.get("source_event_id") else None,
                    linked_to_action=bool(item.get("linked_to_action")),
                    promotion_hint=str(item.get("promotion_hint") or "default"),
                    timestamp=str(item.get("timestamp") or ""),
                )
            )
        except Exception:
            continue
    return triples or None


def _store_cached_extraction(*, user_id: str, source: str, message: str, triples: list[TripleCandidate]) -> None:
    if not triples:
        return
    cache_key = _extraction_cache_key(user_id=user_id, source=source, message=message)
    payload = json.dumps([triple.to_dict() for triple in triples])
    conn = _connect_extraction_cache()
    try:
        conn.execute(
            """
            INSERT INTO extraction_cache (hash, user_id, source, message, result)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(hash) DO UPDATE SET
                result = excluded.result,
                message = excluded.message,
                user_id = excluded.user_id,
                source = excluded.source
            """,
            (cache_key, user_id, source, message, payload),
        )
        conn.commit()
    finally:
        conn.close()


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. Put it in backend/.env or your environment."
            )
        _client = genai.Client(api_key=api_key)
    return _client


def _get_groq_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. Put it in backend/.env or your environment."
            )
        _groq_client = Groq(api_key=api_key)
    return _groq_client


def generate_reply(
    *,
    user_message: str,
    retrieved_snippets: list[str],
    recent_history: list[dict[str, str]] | None = None,
    graph_facts: list[str] | None = None,
    evidence_paths: list[str] | None = None,
    web_facts: list[str] | None = None,
) -> str:
    return generate_reply_bundle(
        user_message=user_message,
        retrieved_snippets=retrieved_snippets,
        recent_history=recent_history,
        graph_facts=graph_facts,
        evidence_paths=evidence_paths,
        web_facts=web_facts,
    )["text"]


def generate_reply_bundle(
    *,
    user_message: str,
    retrieved_snippets: list[str],
    recent_history: list[dict[str, str]] | None = None,
    graph_facts: list[str] | None = None,
    evidence_paths: list[str] | None = None,
    web_facts: list[str] | None = None,
    memory_found: bool = False,
) -> dict[str, str]:
    def compact_items(items: list[str] | None, *, limit: int) -> list[str]:
        compacted: list[str] = []
        for item in list(items or [])[:limit]:
            cleaned = " ".join(str(item or "").split()).strip()
            if not cleaned:
                continue
            if len(cleaned) > REPLY_ITEM_CHARS:
                cleaned = cleaned[: REPLY_ITEM_CHARS - 3].rstrip() + "..."
            compacted.append(cleaned)
        return compacted

    context_block = ""
    compact_snippets = compact_items(retrieved_snippets, limit=REPLY_CONTEXT_ITEMS)
    compact_graph = compact_items(graph_facts, limit=REPLY_CONTEXT_ITEMS)
    compact_paths = compact_items(evidence_paths, limit=3)
    compact_web = compact_items(web_facts, limit=3)
    query_text = " ".join(str(user_message or "").split()).strip()
    compact_history = [
        {
            "role": str(item.get("role") or "user"),
            "content": " ".join(str(item.get("content") or "").split())[:REPLY_ITEM_CHARS],
        }
        for item in list(recent_history or [])[-8:]
        if str(item.get("content") or "").strip()
    ]
    filtered_history: list[dict[str, str]] = []
    for item in compact_history:
        relevance = _history_relevance_score(query_text, item["content"])
        threshold = 0.18 if memory_found else 0.3
        if relevance >= threshold:
            filtered_history.append(item)

    if compact_snippets:
        joined = "\n".join(f"- {snippet}" for snippet in compact_snippets)
        context_block = f"\n\nRelevant past messages:\n{joined}\n"

    history_block = ""
    if filtered_history:
        joined = "\n".join(
            f"- {item['role']}: {item['content']}" for item in filtered_history
        )
        history_block = f"\nRecent conversation:\n{joined}\n"

    graph_block = ""
    if compact_graph:
        joined = "\n".join(f"- {fact}" for fact in compact_graph)
        graph_block = f"\nKnown graph memory:\n{joined}\n"

    evidence_block = ""
    if compact_paths:
        joined = "\n".join(f"- {path}" for path in compact_paths)
        evidence_block = f"\nEvidence paths:\n{joined}\n"

    web_block = ""
    if compact_web:
        joined = "\n".join(f"- {fact}" for fact in compact_web)
        web_block = f"\nLive web findings:\n{joined}\n"

    prompt = f"""Answer using only the supplied memory and optional web findings.
Be concise, practical, and specific.
Prefer short answers unless the user asks for depth.
Use memory only when it is directly relevant to the user's current question.
If the supplied memory is weak, generic, or unrelated, ignore it instead of forcing it into the answer.
Do not say things like "I've saved your previous conversation" or narrate the retrieval process unless the user explicitly asks.
For learning questions, answer with a direct beginner-friendly explanation or roadmap.
Do not repeat the memory back verbatim unless it materially improves the answer.
If relevant memory is not provided, answer from general knowledge without mentioning missing memory, lacking memory, recent conversation analysis, or unsupported preparation claims.
Never claim the user is preparing for, weak in, or related to a topic unless that is directly supported by the supplied memory.
Do not stitch unrelated topics together just because they appeared earlier in the conversation.
{graph_block}
{web_block}
{evidence_block}
{history_block}
{context_block}
User: {user_message}
Assistant:"""

    try:
        if CHAT_PROVIDER == "groq":
            client = _get_groq_client()
            response = client.chat.completions.create(
                model=CHAT_MODEL,
                temperature=0.2,
                max_tokens=REPLY_MAX_TOKENS,
                messages=[
                    {"role": "system", "content": "Use provided memory only when it is directly relevant. If memory is absent or weak, answer from general knowledge. Do not invent user facts or preparation claims."},
                    {"role": "user", "content": prompt},
                ],
            )
            text = response.choices[0].message.content if response.choices else ""
            return {
                "text": text or "I am here. Tell me more.",
                "provider": "groq",
                "model": CHAT_MODEL,
            }

        client = _get_client()
        response = client.models.generate_content(
            model=os.getenv("GEMINI_CHAT_MODEL", "gemini-flash-latest"),
            contents=prompt,
        )
        return {
            "text": response.text or "I am here. Tell me more.",
            "provider": "gemini",
            "model": os.getenv("GEMINI_CHAT_MODEL", "gemini-flash-latest"),
        }
    except Exception:
        return {
            "text": _fallback_reply(user_message=user_message, graph_facts=graph_facts),
            "provider": "fallback",
            "model": "fallback",
        }


def configured_models() -> dict[str, str]:
    return {
        "chat_provider": CHAT_PROVIDER,
        "chat_model": CHAT_MODEL,
        "signal_model": SIGNAL_MODEL,
        "embed_model": EMBED_MODEL,
    }


def classify_relation_with_llm(*, relation: str, entity_type: str = "") -> dict[str, object] | None:
    prompt = f"""
Classify this relation into semantic retrieval metadata.
Return strict JSON with keys:
family, polarity, section_tags, strength

Allowed family examples: capability, goal, learning, structure, association, general
Allowed polarity: positive, negative, neutral
section_tags should be a short list like ["weakness","practice_priority"] or ["goal","company","target"]
strength should be a float between 0 and 1

Relation: {relation}
Entity type: {entity_type or "Entity"}
""".strip()
    try:
        client = _get_groq_client()
        response = client.chat.completions.create(
            model=os.getenv("GROQ_ROUTER_MODEL", "llama-3.1-8b-instant"),
            temperature=0.0,
            max_tokens=120,
            messages=[
                {
                    "role": "system",
                    "content": "You classify graph relations for retrieval metadata. Return JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        text = response.choices[0].message.content if response.choices else ""
        match = re.search(r"\{.*\}", text or "", flags=re.DOTALL)
        if not match:
            return None
        payload = json.loads(match.group(0))
        if not isinstance(payload, dict):
            return None
        tags = payload.get("section_tags") or []
        if not isinstance(tags, list):
            tags = []
        return {
            "family": str(payload.get("family") or "general").strip().lower(),
            "polarity": str(payload.get("polarity") or "neutral").strip().lower(),
            "section_tags": [str(tag).strip().lower() for tag in tags if str(tag).strip()],
            "strength": max(0.0, min(float(payload.get("strength") or 0.5), 1.0)),
        }
    except Exception:
        return None


def extract_triple_candidates(*, user_id: str, message: str, source: str = "chat") -> list[TripleCandidate]:
    cached = _load_cached_extraction(user_id=user_id, source=source, message=message)
    if cached is not None:
        return cached

    signal_mode = os.getenv("GRAPHMIND_SIGNAL_MODE", "auto").strip().lower()
    heuristic_triples: list[TripleCandidate] | None = None

    if signal_mode == "heuristic":
        heuristic_triples = _heuristic_triple_candidates(user_id=user_id, message=message, source=source)
        _store_cached_extraction(user_id=user_id, source=source, message=message, triples=heuristic_triples)
        return heuristic_triples

    prompt = f"""
Understand the meaning of the user message and extract durable memory in structured form.
Return strict JSON with this shape:
{{
  "user_facts": [
    {{
      "relation": "TARGETS",
      "object_type": "Company",
      "object_name": "Google",
      "confidence": 0.84,
      "linked_to_action": false
    }}
  ],
  "concept_relations": [
    {{
      "subject_type": "Concept",
      "subject_name": "Fourier Transform",
      "relation": "PART_OF",
      "object_type": "Concept",
      "object_name": "Signals and Systems",
      "confidence": 0.9
    }}
  ]
}}

Rules:
- Understand meaning, not keywords. Do not infer the opposite sentiment.
- If user says they are good/confident/strong in something, use STRENGTH_IN or IMPROVED_IN, never STRUGGLES_WITH.
- If user says they are weak/confused/bad at something, use STRUGGLES_WITH.
- If the user is asking for help learning, preparing, understanding, revising, or assessing themselves on a specific topic, you may infer a soft STUDIES fact for that topic even if they did not literally say "I am studying it".
- If the user asks about their own preparation level for a topic, infer a user_fact about that topic rather than returning empty.
- For direct topic-help requests like "tell me about opamp" or "help me prepare opamp", prefer Topic/Concept nodes, not Company.
- Keep entity names short and atomic. Never include whole clauses.
- subject/object types should be one of Company, Topic, Skill, Goal, Document, Entity, Domain, Concept.
- user_facts are always from User({user_id}) to an object.
- concept_relations are only for concept/domain/resource structure such as PART_OF, USED_IN, PREREQUISITE_FOR, COVERS, EMPHASIZES.
- relation should be an uppercase snake case relation.
- confidence must be between 0 and 1.
- linked_to_action is true when the message indicates concrete work, practice, upload, scheduling, or execution.
- Do not output vague nodes like "study", "learning", "preparation", "thing", "problem".
- If nothing should be remembered, return empty arrays.

User message: {message}
""".strip()

    try:
        client = _get_client()
        response = client.models.generate_content(
            model=SIGNAL_MODEL,
            contents=prompt,
        )
        parsed = _parse_semantic_response(response.text or "")
        semantic_triples = _semantic_response_to_triples(
            user_id=user_id,
            message=message,
            source=source,
            payload=parsed,
        )
        if semantic_triples:
            _store_cached_extraction(user_id=user_id, source=source, message=message, triples=semantic_triples)
            return semantic_triples
    except Exception:
        pass

    semantic_interest_triples = _semantic_interest_fallback(
        user_id=user_id,
        message=message,
        source=source,
    )
    if semantic_interest_triples:
        _store_cached_extraction(user_id=user_id, source=source, message=message, triples=semantic_interest_triples)
        return semantic_interest_triples

    broad_interest_triples = _broad_interest_fallback(
        user_id=user_id,
        message=message,
        source=source,
    )
    if broad_interest_triples:
        _store_cached_extraction(user_id=user_id, source=source, message=message, triples=broad_interest_triples)
        return broad_interest_triples

    heuristic_triples = _heuristic_triple_candidates(user_id=user_id, message=message, source=source)
    _store_cached_extraction(user_id=user_id, source=source, message=message, triples=heuristic_triples)
    return heuristic_triples


def extract_memory_signals(*, user_id: str, message: str, source: str = "chat") -> list[dict]:
    signals: list[dict] = []
    for triple in extract_triple_candidates(user_id=user_id, message=message, source=source):
        if triple.subject_type.strip().lower() != "user":
            continue
        signals.append(
            {
                "user_id": triple.user_id,
                "entity": triple.object_name,
                "entity_type": triple.object_type,
                "relation": triple.relation,
                "confidence": triple.confidence,
                "linked_to_action": triple.linked_to_action,
                "source": triple.source,
                "raw_text": triple.raw_text,
                "source_event_id": triple.source_event_id,
            }
        )
    return signals


def _parse_semantic_response(text: str) -> dict:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {}
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _semantic_interest_fallback(*, user_id: str, message: str, source: str) -> list[TripleCandidate]:
    prompt = f"""
Decide whether this user message implies durable user-interest or preparation memory.
Return strict JSON with this exact shape:
{{
  "should_store": true,
  "relation": "STUDIES",
  "object_type": "Topic",
  "object_name": "Operational Amplifiers",
  "confidence": 0.82,
  "linked_to_action": false
}}

Rules:
- Use should_store=false if the message does not imply durable user interest, preparation, weakness, goal, or study intent.
- If the message asks for help learning, preparing, revising, understanding, teaching, or assessing a specific topic, field, concept, technology, company, exam, or domain, you may store a soft memory signal.
- If the message asks about the user's own preparation level for something, store a soft memory signal for that subject.
- Prefer STUDIES for topics/fields/concepts, TARGETS for companies, and PREPARES_FOR for roles/exams/goals.
- Never label academic topics as Company.
- Prefer Topic, Skill, Domain, Goal, or Entity for subjects like opamp, graphs, heaps, dynamic programming, os, dbms, recursion, analog electronics, system design.
- object_name must be short and atomic.
- Good examples that should_store=true:
  - "tell me about opamp"
  - "help me prepare os"
  - "do you know my preparation level of dbms"
  - "i need help with analog electronics"
  - "what should i study for compiler design"

User message: {message}
""".strip()

    try:
        client = _get_client()
        response = client.models.generate_content(
            model=SIGNAL_MODEL,
            contents=prompt,
        )
        payload = _parse_semantic_response(response.text or "")
        if not bool(payload.get("should_store")):
            return []
        triple = _normalize_triple_candidate(
            user_id=user_id,
            message=message,
            source=source,
            triple={
                "subject_type": "User",
                "subject_name": user_id,
                "relation": str(payload.get("relation") or "STUDIES"),
                "object_type": str(payload.get("object_type") or "Topic"),
                "object_name": str(payload.get("object_name") or ""),
                "confidence": max(float(payload.get("confidence") or 0.58), 0.82),
                "linked_to_action": bool(payload.get("linked_to_action")),
            },
        )
        if triple.object_type.strip().lower() == "company" and not _is_valid_company_name(triple.object_name):
            return []
        if not _is_valid_memory_span(triple.object_name):
            return []
        return [triple]
    except Exception:
        return []


def _broad_interest_fallback(*, user_id: str, message: str, source: str) -> list[TripleCandidate]:
    lowered = " ".join((message or "").lower().split())
    if not lowered:
        return []

    patterns = [
        r"(?:tell me about|teach me about|explain|help me with|help me understand|i need help with|guide me in)\s+([a-z0-9 +#&/().,\-]{2,80})",
        r"(?:help me prepare|prepare me for|what should i study for|how do i prepare for|my preparation level of)\s+([a-z0-9 +#&/().,\-]{2,80})",
    ]

    raw_subject = ""
    for pattern in patterns:
        match = re.search(pattern, lowered, flags=re.IGNORECASE)
        if match:
            raw_subject = match.group(1).strip(" .?")
            break
    if not raw_subject:
        return []

    raw_subject = re.split(r"\b(?:for interviews?|for interview|interviews?|interview|please|pls|now)\b", raw_subject, maxsplit=1, flags=re.IGNORECASE)[0].strip(" .?")
    if not raw_subject:
        return []

    looks_like_company = _is_valid_company_name(raw_subject.title()) and raw_subject.lower() not in COMPANY_STOPWORDS and raw_subject[:1].isupper()
    object_type = "Company" if looks_like_company else "Topic"
    relation = "TARGETS" if object_type == "Company" else "STUDIES"
    object_name = _clean_entity_text(raw_subject, object_type)
    if object_type == "Company" and not _is_valid_company_name(object_name):
        return []
    if not _is_valid_memory_span(object_name):
        return []
    return [
        TripleCandidate(
            user_id=user_id,
            subject_type="User",
            subject_name=user_id,
            relation=relation,
            object_type=object_type,
            object_name=object_name,
            confidence=0.82,
            source=source,
            raw_text=message,
            linked_to_action=bool(re.search(r"\bprepare|study|help|understand|explain|guide\b", lowered)),
        )
    ]


def _semantic_response_to_triples(*, user_id: str, message: str, source: str, payload: dict) -> list[TripleCandidate]:
    triples: list[TripleCandidate] = []
    for fact in payload.get("user_facts") or []:
        object_type = str(fact.get("object_type") or "Entity").strip() or "Entity"
        object_name = _clean_entity_text(str(fact.get("object_name") or "").strip(), object_type)
        relation = _normalize_semantic_relation(str(fact.get("relation") or "RELATED_TO"))
        if object_type.strip().lower() == "company" and not _is_valid_company_name(object_name):
            continue
        if not _is_valid_memory_span(object_name):
            continue
        try:
            confidence = float(fact.get("confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        triples.append(
            TripleCandidate(
                user_id=user_id,
                subject_type="User",
                subject_name=user_id,
                relation=relation,
                object_type=object_type,
                object_name=object_name,
                confidence=max(0.0, min(confidence, 1.0)),
                source=source,
                raw_text=message,
                linked_to_action=bool(fact.get("linked_to_action")),
            )
        )

    for relation_item in payload.get("concept_relations") or []:
        subject_type = str(relation_item.get("subject_type") or "Concept").strip() or "Concept"
        object_type = str(relation_item.get("object_type") or "Concept").strip() or "Concept"
        subject_name = _clean_entity_text(str(relation_item.get("subject_name") or "").strip(), subject_type)
        object_name = _clean_entity_text(str(relation_item.get("object_name") or "").strip(), object_type)
        relation = _normalize_semantic_relation(str(relation_item.get("relation") or "RELATED_TO"))
        if not _is_valid_memory_span(subject_name) or not _is_valid_memory_span(object_name):
            continue
        try:
            confidence = float(relation_item.get("confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        triples.append(
            TripleCandidate(
                user_id=user_id,
                subject_type=subject_type,
                subject_name=subject_name,
                relation=relation,
                object_type=object_type,
                object_name=object_name,
                confidence=max(0.0, min(confidence, 1.0)),
                source=source,
                raw_text=message,
                promotion_hint="structural",
            )
        )

    return _merge_triple_candidates(triples)


def _normalize_triple_candidate(*, user_id: str, message: str, source: str, triple: dict) -> TripleCandidate:
    try:
        confidence = float(triple.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0

    object_type = str(triple.get("object_type") or "Entity").strip() or "Entity"
    return TripleCandidate(
        user_id=user_id,
        subject_type=str(triple.get("subject_type") or "User").strip() or "User",
        subject_name=str(triple.get("subject_name") or user_id).strip() or user_id,
        relation=str(triple.get("relation") or "RELATED_TO").strip() or "RELATED_TO",
        object_type=object_type,
        object_name=_clean_entity_text(str(triple.get("object_name") or "").strip(), object_type),
        confidence=max(0.0, min(confidence, 1.0)),
        source=source,
        raw_text=message,
        linked_to_action=bool(triple.get("linked_to_action")),
    )


def _heuristic_triple_candidates(*, user_id: str, message: str, source: str) -> list[TripleCandidate]:
    clauses = _split_clauses(message)
    triples: list[TripleCandidate] = []

    for company in _extract_company_targets(message):
        triples.append(
            TripleCandidate(
                user_id=user_id,
                subject_type="User",
                subject_name=user_id,
                relation="TARGETS",
                object_type="Company",
                object_name=company,
                confidence=0.78,
                source=source,
                raw_text=message,
                linked_to_action=True,
            )
        )
        if re.search(r"\b(interview|interviewed|interviewing|attended .* interview)\b", message, flags=re.IGNORECASE):
            triples.append(
                TripleCandidate(
                    user_id=user_id,
                    subject_type="User",
                    subject_name=user_id,
                    relation="INTERVIEWED_AT",
                    object_type="Company",
                    object_name=company,
                    confidence=0.86,
                    source=source,
                    raw_text=message,
                    linked_to_action=True,
                )
            )

    for clause in clauses:
        struggle_in_match = re.search(
            r"(?:bad|weak|poor|struggling)\s+at\s+([a-zA-Z0-9 +#-]+?)\s+in\s+([a-zA-Z0-9 +#-]+)",
            clause,
            flags=re.IGNORECASE,
        )
        if struggle_in_match:
            sub_skill = _clean_entity_text(struggle_in_match.group(1).strip(" ."), "Skill")
            parent_topic = _clean_entity_text(struggle_in_match.group(2).strip(" ."), "Topic")
            if _is_valid_memory_span(sub_skill):
                triples.append(
                    TripleCandidate(
                        user_id=user_id,
                        subject_type="User",
                        subject_name=user_id,
                        relation="STRUGGLES_WITH",
                        object_type="Skill",
                        object_name=sub_skill,
                        confidence=0.95,
                        source=source,
                        raw_text=message,
                    )
                )
            if _is_valid_memory_span(parent_topic):
                triples.append(
                    TripleCandidate(
                        user_id=user_id,
                        subject_type="User",
                        subject_name=user_id,
                        relation="STUDIES",
                        object_type="Topic",
                        object_name=parent_topic,
                        confidence=0.82,
                        source=source,
                        raw_text=message,
                    )
                )
                if _is_valid_memory_span(sub_skill):
                    triples.append(
                        TripleCandidate(
                            user_id=user_id,
                            subject_type="Concept",
                            subject_name=sub_skill,
                            relation="USED_IN",
                            object_type="Concept",
                            object_name=parent_topic,
                            confidence=0.78,
                            source=source,
                            raw_text=message,
                            promotion_hint="structural",
                        )
                    )

        strength_match = re.search(
            r"(?:good|strong|confident)\s+at\s+([a-zA-Z0-9 +#-]+)",
            clause,
            flags=re.IGNORECASE,
        )
        if strength_match:
            skill = _clean_entity_text(strength_match.group(1).strip(" ."), "Skill")
            if _is_valid_memory_span(skill):
                triples.append(
                    TripleCandidate(
                        user_id=user_id,
                        subject_type="User",
                        subject_name=user_id,
                        relation="STRENGTH_IN",
                        object_type="Skill",
                        object_name=skill,
                        confidence=0.88,
                        source=source,
                        raw_text=message,
                    )
                )

        struggle_topic_match = re.search(
            r"(?:struggling|struggle|weak|bad|poor)\s+(?:in|with)\s+(?:the\s+)?(?:topic|topics|concept|concepts|area|areas)?\s*of?\s*([a-zA-Z0-9 +#,\-]{2,80})",
            clause,
            flags=re.IGNORECASE,
        )
        if struggle_topic_match:
            for object_name in _split_memory_objects(struggle_topic_match.group(1).strip(" ."), "Topic"):
                if not _is_valid_memory_span(object_name):
                    continue
                triples.append(
                    TripleCandidate(
                        user_id=user_id,
                        subject_type="User",
                        subject_name=user_id,
                        relation="STUDIES",
                        object_type="Topic",
                        object_name=object_name,
                        confidence=0.8,
                        source=source,
                        raw_text=message,
                        linked_to_action=False,
                    )
                )
                triples.append(
                    TripleCandidate(
                        user_id=user_id,
                        subject_type="User",
                        subject_name=user_id,
                        relation="STRUGGLES_WITH",
                        object_type="Skill",
                        object_name=object_name,
                        confidence=0.9,
                        source=source,
                        raw_text=message,
                        linked_to_action=False,
                    )
                )

        patterns = [
            (r"target(?:ing)? ([A-Z][A-Za-z0-9&.-]+)", "TARGETS", 0.68, "Company", False),
            (r"applying to ([A-Z][A-Za-z0-9&.-]+)", "TARGETS", 0.78, "Company", True),
            (r"(?:learning|studying|studied|learned) ([a-zA-Z0-9 +#,\-]{2,80})", "STUDIES", 0.84, "Topic", True),
            (r"(?:want to learn|want to study|want to prepare|preparing for|prepare) ([a-zA-Z0-9 +#,\-]{2,80})", "STUDIES", 0.84, "Topic", True),
            (r"(?:practiced|practising|practicing|working on|building) ([a-zA-Z0-9 +#-]{2,60})", "STUDIES", 0.82, "Topic", True),
            (r"struggling with ([a-zA-Z0-9 +#-]{2,60})", "STRUGGLES_WITH", 0.88, "Skill", False),
            (r"improved in ([a-zA-Z0-9 +#-]{2,60})", "IMPROVED_IN", 0.84, "Skill", True),
        ]

        for pattern, relation, confidence, object_type, linked_to_action in patterns:
            match = re.search(pattern, clause, flags=re.IGNORECASE)
            if not match:
                continue
            for object_name in _split_memory_objects(match.group(1).strip(" ."), object_type):
                if object_type == "Company" and not _is_valid_company_name(object_name):
                    continue
                if not _is_valid_memory_span(object_name):
                    continue
                triples.append(
                    TripleCandidate(
                        user_id=user_id,
                        subject_type="User",
                        subject_name=user_id,
                        relation=relation,
                        object_type=object_type,
                        object_name=object_name.title() if object_type == "Company" else object_name,
                        confidence=confidence,
                        source=source,
                        raw_text=message,
                        linked_to_action=linked_to_action,
                    )
                )

        asked_about_match = re.search(
            r"(?:asked about|question(?:ed)? on)\s+([a-zA-Z0-9 +#,\-]{2,80})",
            clause,
            flags=re.IGNORECASE,
        )
        if asked_about_match:
            raw_topic = re.split(r"\b(?:but|and|so|because)\b", asked_about_match.group(1), maxsplit=1, flags=re.IGNORECASE)[0]
            for object_name in _split_memory_objects(raw_topic.strip(" ."), "Topic"):
                if not _is_valid_memory_span(object_name):
                    continue
                triples.append(
                    TripleCandidate(
                        user_id=user_id,
                        subject_type="User",
                        subject_name=user_id,
                        relation="STUDIES",
                        object_type="Topic",
                        object_name=object_name,
                        confidence=0.83,
                        source=source,
                        raw_text=message,
                        linked_to_action=True,
                    )
                )
                if re.search(r"(?:failed to explain|couldn't explain|could not explain)", message, flags=re.IGNORECASE):
                    triples.append(
                        TripleCandidate(
                            user_id=user_id,
                            subject_type="User",
                            subject_name=user_id,
                            relation="STRUGGLES_WITH",
                            object_type="Skill",
                            object_name=object_name,
                            confidence=0.88,
                            source=source,
                            raw_text=message,
                        )
                    )

        failed_to_explain_match = re.search(
            r"(?:failed to explain|couldn't explain|could not explain)\s+([a-zA-Z0-9 +#,\-]{2,80})",
            clause,
            flags=re.IGNORECASE,
        )
        if failed_to_explain_match:
            raw_skill = re.split(r"\b(?:but|and|so|because)\b", failed_to_explain_match.group(1), maxsplit=1, flags=re.IGNORECASE)[0]
            for object_name in _split_memory_objects(raw_skill.strip(" ."), "Skill"):
                if not _is_valid_memory_span(object_name):
                    continue
                triples.append(
                    TripleCandidate(
                        user_id=user_id,
                        subject_type="User",
                        subject_name=user_id,
                        relation="STRUGGLES_WITH",
                        object_type="Skill",
                        object_name=object_name,
                        confidence=0.9,
                        source=source,
                        raw_text=message,
                    )
                )

    return _merge_triple_candidates(triples)


def _normalize_semantic_relation(relation: str) -> str:
    normalized = (relation or "RELATED_TO").strip().upper() or "RELATED_TO"
    return SEMANTIC_RELATION_ALIASES.get(normalized, normalized)


def _merge_triple_candidates(triples: list[TripleCandidate]) -> list[TripleCandidate]:
    merged: dict[tuple[str, str, str, str, str], TripleCandidate] = {}
    for triple in triples:
        key = (
            triple.subject_type.lower(),
            triple.subject_name.strip().lower(),
            triple.relation.upper(),
            triple.object_type.lower(),
            triple.object_name.strip().lower(),
        )
        existing = merged.get(key)
        if existing is None:
            merged[key] = triple
            continue
        if triple.confidence > existing.confidence:
            merged[key] = triple
            existing = merged[key]
        existing.linked_to_action = existing.linked_to_action or triple.linked_to_action
        if triple.promotion_hint == "structural":
            existing.promotion_hint = "structural"
    return list(merged.values())


def _split_memory_objects(raw_text: str, object_type: str) -> list[str]:
    cleaned = re.sub(
        r"\b(?:today|tonight|yesterday|now|recently|lately|this morning|this evening)\b",
        "",
        raw_text or "",
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .,")
    if not cleaned:
        return []

    parts = re.split(r"\s*(?:,| and | & )\s*", cleaned, flags=re.IGNORECASE)
    results: list[str] = []
    for part in parts:
        candidate = _clean_entity_text(part.strip(" .,"), object_type)
        if candidate:
            results.append(candidate)
    return results or [_clean_entity_text(cleaned, object_type)]


def _extract_company_targets(text: str) -> list[str]:
    candidates: list[str] = []
    patterns = [
        r"\b(?:at|with|for)\s+([A-Z][A-Za-z0-9&.-]+)\b(?=\s*(?:interviews?|role|job|placement|$|[.,]))",
        r"\b([A-Za-z][A-Za-z0-9&.-]+)\s+interview\b",
        r"\btarget(?:ing)?\s+([A-Z][A-Za-z0-9&.-]+)\b",
        r"\bapplying\s+to\s+([A-Z][A-Za-z0-9&.-]+)\b",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            company = _clean_entity_text(match.group(1).strip(" ."), "Company")
            if _is_valid_company_name(company):
                candidates.append(company)
    deduped: list[str] = []
    seen: set[str] = set()
    for company in candidates:
        key = company.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(company)
    return deduped


def _is_valid_company_name(text: str) -> bool:
    cleaned = text.strip()
    if not cleaned:
        return False
    if cleaned.lower() in COMPANY_STOPWORDS:
        return False
    if len(cleaned) < 3:
        return False
    if len(cleaned.split()) > 3:
        return False
    return True


def embed_texts(texts: list[str]) -> list[list[float]]:
    return embed_texts_with_kind(texts)[0]


def embed_texts_with_kind(texts: list[str]) -> tuple[list[list[float]], str]:
    try:
        client = _get_client()
        resp = client.models.embed_content(
            model=EMBED_MODEL,
            contents=texts,
        )

        if hasattr(resp, "embeddings") and resp.embeddings:
            out: list[list[float]] = []
            for embedding in resp.embeddings:
                if hasattr(embedding, "values"):
                    out.append(list(embedding.values))
                elif isinstance(embedding, dict) and "values" in embedding:
                    out.append(list(embedding["values"]))
            if out:
                return out, EMBED_MODEL

        if hasattr(resp, "embedding") and hasattr(resp.embedding, "values"):
            return [list(resp.embedding.values)], EMBED_MODEL
    except Exception:
        pass

    return [_fallback_embedding(text) for text in texts], "local-hash-v1"


def _fallback_reply(*, user_message: str, graph_facts: list[str] | None = None) -> str:
    cleaned = " ".join((user_message or "").split()).strip()
    if graph_facts:
        joined = "; ".join(list(graph_facts or [])[:2])
        return f"I can help based on the relevant memory I found: {joined}"
    return (
        "I can help, but the live model is unavailable right now. "
        f"Please try again, or ask a shorter direct question like: {cleaned}"
    )


def _history_relevance_score(query: str, candidate: str) -> float:
    query_clean = " ".join((query or "").split()).strip()
    candidate_clean = " ".join((candidate or "").split()).strip()
    if not query_clean or not candidate_clean:
        return 0.0
    query_tokens = set(re.findall(r"\w+", query_clean.lower()))
    candidate_tokens = set(re.findall(r"\w+", candidate_clean.lower()))
    if not query_tokens or not candidate_tokens:
        overlap = 0.0
    else:
        overlap = len(query_tokens & candidate_tokens) / max(1, len(query_tokens))
    query_embedding = _fallback_embedding(query_clean)
    candidate_embedding = _fallback_embedding(candidate_clean)
    semantic = sum(a * b for a, b in zip(query_embedding, candidate_embedding))
    return max(overlap, semantic)


def _fallback_embedding(text: str, dimensions: int = 32) -> list[float]:
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


def _clean_entity_text(entity: str, entity_type: str) -> str:
    cleaned = re.sub(r"\s+", " ", entity.strip(" .,\t\r\n"))
    if not cleaned:
        return cleaned

    lowered_type = entity_type.strip().lower()
    if lowered_type in {"topic", "concept"}:
        cleaned = re.sub(
            r"\b(today|tonight|now|currently|lately|recently|this week|this month)\b$",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip(" .,\t\r\n")
        cleaned = re.sub(r"\b(problem|problems|stuff|things)\b$", "", cleaned, flags=re.IGNORECASE).strip(" .,\t\r\n")
    if lowered_type in {"company", "skill", "goal", "document", "concept"}:
        cleaned = cleaned.title()

    return cleaned


def _split_clauses(text: str) -> list[str]:
    chunks = re.split(r"[.;\n]+|\s+(?:but|while|however)\s+", text)
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def _is_valid_memory_span(text: str) -> bool:
    if not text:
        return False
    stripped = text.strip()
    normalized = " " + stripped.lower() + " "
    banned_fragments = [
        " currently ",
        " yesterday ",
        " recently ",
        " and ",
        " but ",
        " because ",
        " improved ",
        " studying ",
        " preparing ",
        " interviews ",
    ]
    if len(stripped.split()) > 6:
        return False
    if len(stripped) > 48:
        return False
    if any(fragment in normalized for fragment in banned_fragments):
        return False
    if re.search(r"[,.!?;:]", stripped):
        return False
    if re.search(r"\b(?:for|with|while|after|before|during|currently|yesterday|today|tomorrow|because|that|which)\b", normalized):
        return False
    if re.search(r"\b(?:good|bad|weak|strong|improved|studying|preparing|targeting|practice|practiced)\b", normalized):
        return False
    if stripped.lower().startswith(("i ", "my ", "we ", "it ")):
        return False
    return True
