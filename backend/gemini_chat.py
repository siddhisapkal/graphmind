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
from .graph.models import normalize_text_key

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
CACHE_VERSION = "v3"
_COMPANY_CLASSIFICATION_CACHE: dict[str, str | None] = {}


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
    payload = f"{CACHE_VERSION}|{user_id.strip().lower()}|{source.strip().lower()}|{' '.join(message.split()).strip().lower()}"
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
You are an intelligent semantic memory extraction system.

Your task is to understand the MEANING of the user message and extract durable memory.

IMPORTANT:
- Understand intent, not exact wording.
- The same meaning can be expressed in different ways.

RELATION STRATEGY:
- First, try to map the meaning to one of these CORE relations:

  STRUGGLES_WITH
  STUDIES
  TARGETS
  PREPARING_FOR
  STRENGTH_IN
  IMPROVING_IN
  INTERESTED_IN

- If the meaning clearly fits one of these, USE it.

- If none of these fit well, then you MAY create a new relation dynamically that best describes the meaning.

Examples:
- "economics is confusing" -> STRUGGLES_WITH Economics
- "i am studying polity" -> STUDIES Polity
- "trying to get better at writing" -> IMPROVING_IN Answer Writing
- "i revise daily" -> HAS_HABIT Revision

Instructions:
- Treat full message as one semantic unit
- Combine sentences if needed
- Resolve references like "it", "this"
- Subject is always User({user_id})
- Object must be short and atomic
- Do NOT include full sentences
- Do NOT extract meaningless data
- Prefer Topic, Skill, Goal, Company, Domain, Concept, Entity, or Document as object_type
- Do not turn academic topics into Company

Confidence:
- 0.8-1.0 -> clear
- 0.5-0.7 -> inferred

Output STRICT JSON:
{{
  "user_facts": [
    {{
      "relation": "...",
      "object_type": "Topic",
      "object_name": "...",
      "confidence": 0.0,
      "linked_to_action": false
    }}
  ],
  "concept_relations": []
}}

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


def generate_company_planner(
    *,
    company: str,
    days_left: int,
    web_results: list[dict[str, str]] | None = None,
    memory_facts: list[str] | None = None,
    profile_summary: dict[str, list[dict[str, object]]] | None = None,
) -> dict[str, object]:
    normalized_company = " ".join((company or "").split()).strip()
    normalized_days = max(1, int(days_left))
    compact_web = []
    for item in list(web_results or [])[:6]:
        title = " ".join(str(item.get("title") or "").split()).strip()
        snippet = " ".join(str(item.get("snippet") or "").split()).strip()
        url = " ".join(str(item.get("url") or "").split()).strip()
        if title:
            compact_web.append(f"{title}: {snippet} ({url})".strip())
    compact_memory = [" ".join(str(item or "").split()).strip() for item in list(memory_facts or [])[:6] if str(item or "").strip()]
    compact_profile = _compact_profile_summary(profile_summary)

    prompt = f"""
Create a company-wise placement planner.
Return strict JSON only with this exact shape:
{{
  "company": "{normalized_company}",
  "overview": "short summary",
  "web_focus_topics": ["topic 1", "topic 2"],
  "personalized_focus": {{
    "strengths_to_use": ["strength 1"],
    "weaknesses_to_focus": ["weakness 1"],
    "improving_now": ["improving 1"]
  }},
  "fit_analysis": {{
    "matched_strengths": ["strength matched to company topics"],
    "matched_weaknesses": ["weakness matched to company topics"],
    "strategic_summary": "short personalized analysis"
  }},
  "stages": [
    {{
      "name": "Coding Round",
      "focus": "what to focus on",
      "resource": "what to practice",
      "expected_questions": ["arrays", "graphs"]
    }}
  ],
  "daily_plan": [
    {{
      "day": 1,
      "title": "Aptitude and reasoning",
      "tasks": ["task 1", "task 2"],
      "goal": "daily goal"
    }}
  ],
  "recommendations": ["tip 1", "tip 2"],
  "likely_previous_question_patterns": ["pattern 1", "pattern 2"]
}}

Rules:
- Use the web findings to infer likely rounds, repeated interview themes, and preparation areas.
- Use the memory facts to personalize the plan if they indicate strengths or weaknesses.
- Keep stages practical and atomic.
- The daily_plan must have exactly {normalized_days} entries.
- Make the plan realistic for the actual company process found in the evidence.
- Do not assume coding rounds unless the web findings suggest coding, DSA, online assessment, programming, or technical screening.
- The company may be from any domain, not only software.
- Different companies should produce materially different stages and tasks when the evidence differs.
- Do not mention unavailable tools.
- recommendations and likely_previous_question_patterns should be concise.
- web_focus_topics should summarize the main areas repeatedly mentioned in the web evidence.
- personalized_focus must map the user's strengths, weaknesses, and improving areas from the cached profile first, then memory facts if needed.
- fit_analysis must compare company/web topics with the precomputed profile information.
- matched_strengths should include strengths that help with the company topics.
- matched_weaknesses should include weaknesses that may hurt performance for this company.
- strategic_summary should clearly explain what to lean on and what to fix first.

Memory facts:
{json.dumps(compact_memory, ensure_ascii=True)}

Cached profile summary:
{json.dumps(compact_profile, ensure_ascii=True)}

Web findings:
{json.dumps(compact_web, ensure_ascii=True)}
""".strip()

    try:
        client = _get_client()
        response = client.models.generate_content(
            model=SIGNAL_MODEL,
            contents=prompt,
        )
        payload = _parse_semantic_response(response.text or "")
        if isinstance(payload.get("daily_plan"), list) and isinstance(payload.get("stages"), list):
            payload["company"] = str(payload.get("company") or normalized_company)
            payload["daily_plan"] = _normalize_daily_plan(payload.get("daily_plan"), normalized_days)
            payload["stages"] = _normalize_planner_stages(payload.get("stages"))
            payload["web_focus_topics"] = _normalize_string_list(
                payload.get("web_focus_topics"),
                fallback=_fallback_web_focus_topics(list(web_results or [])),
            )
            payload["personalized_focus"] = _normalize_personalized_focus(
                payload.get("personalized_focus"),
                memory_facts=compact_memory,
                profile_summary=compact_profile,
            )
            payload["fit_analysis"] = _normalize_fit_analysis(
                payload.get("fit_analysis"),
                web_topics=payload["web_focus_topics"],
                personalized_focus=payload["personalized_focus"],
                profile_summary=compact_profile,
            )
            payload["recommendations"] = _normalize_string_list(payload.get("recommendations"), fallback=["Prioritize the rounds most often mentioned in the evidence"])
            payload["likely_previous_question_patterns"] = _normalize_string_list(
                payload.get("likely_previous_question_patterns"),
                fallback=["Review repeated themes from recent interview experiences"],
            )
            return payload
    except Exception:
        pass

    default_stages = _fallback_planner_stages(company=normalized_company, web_results=list(web_results or []))
    daily_plan = _fallback_daily_plan(
        company=normalized_company,
        days=normalized_days,
        stages=default_stages,
        web_results=list(web_results or []),
        memory_facts=compact_memory,
    )
    return {
        "company": normalized_company,
        "overview": f"{normalized_company} planner generated from current interview-process evidence and personalized memory hints.",
        "web_focus_topics": _fallback_web_focus_topics(list(web_results or [])),
        "personalized_focus": _normalize_personalized_focus({}, memory_facts=compact_memory, profile_summary=compact_profile),
        "fit_analysis": _normalize_fit_analysis(
            {},
            web_topics=_fallback_web_focus_topics(list(web_results or [])),
            personalized_focus=_normalize_personalized_focus({}, memory_facts=compact_memory, profile_summary=compact_profile),
            profile_summary=compact_profile,
        ),
        "stages": default_stages,
        "daily_plan": daily_plan,
        "recommendations": ["Prioritize weak topics first", "Do one timed practice block daily"],
        "likely_previous_question_patterns": _fallback_question_patterns(list(web_results or [])),
    }


def analyze_strength_weakness_profile(
    *,
    message: str,
    triples: list[TripleCandidate] | None = None,
    web_facts: list[str] | None = None,
    seed_observations: list[dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    compact_triples = [
        {
            "relation": triple.relation,
            "object_type": triple.object_type,
            "object_name": triple.object_name,
            "confidence": triple.confidence,
        }
        for triple in list(triples or [])[:10]
        if triple.subject_type.strip().lower() == "user"
    ]
    compact_web = [
        " ".join(str(item or "").split()).strip()
        for item in list(web_facts or [])[:5]
        if str(item or "").strip()
    ]
    compact_seed = [
        {
            "entity": str(item.get("entity") or "").strip(),
            "entity_type": str(item.get("entity_type") or "Skill").strip() or "Skill",
            "signal_type": str(item.get("signal_type") or "").strip().lower(),
            "delta": float(item.get("delta") or 0.0),
        }
        for item in list(seed_observations or [])[:8]
        if isinstance(item, dict) and str(item.get("entity") or "").strip()
    ]
    prompt = f"""
Analyze the user's message and classify any self-assessment into strengths, weaknesses, and improving areas.
Return strict JSON only with this exact shape:
{{
  "strengths": [
    {{"entity": "Problem Solving", "entity_type": "Skill", "delta": 0.8, "rationale": "user says they are good at it"}}
  ],
  "weaknesses": [
    {{"entity": "Dynamic Programming", "entity_type": "Skill", "delta": 0.9, "rationale": "user says it is challenging"}}
  ],
  "improving": [
    {{"entity": "Timed Coding Rounds", "entity_type": "Skill", "delta": 0.7, "rationale": "user is actively working on it"}}
  ]
}}

Rules:
- Extract only atomic skills or topics.
- Do not return sentence wrappers like "to improve in these areas", "advanced topics", "topics like", or "understanding core concepts".
- Split grouped phrases into separate items.
- Use `strengths` for areas the user says they are good at, strong in, comfortable with, or confident in.
- Use `weaknesses` for areas the user says are challenging, weak, difficult, confusing, or where they struggle.
- Use `improving` for areas the user says they are currently practicing, improving, working on, or building confidence in.
- The same entity can appear in both `weaknesses` and `improving` if the user implies both.
- Use the web context only to disambiguate the topic, not to invent signals.
- If the user clearly gives self-assessment, do not return empty arrays for everything.

Extracted user triples:
{json.dumps(compact_triples, ensure_ascii=True)}

Preliminary observations:
{json.dumps(compact_seed, ensure_ascii=True)}

Relevant web context:
{json.dumps(compact_web, ensure_ascii=True)}

User message:
{message}
""".strip()

    try:
        client = _get_client()
        observations = _extract_bucketed_profile_observations_with_gemini(client=client, prompt=prompt)
        filtered_observations = _filter_profile_observations(observations)
        if filtered_observations:
            return filtered_observations

        candidates = _candidate_profile_entities_from_message(
            message=message,
            triples=triples,
            seed_observations=seed_observations,
        )
        if candidates:
            classified = _classify_profile_candidates_with_gemini(
                client=client,
                message=message,
                candidates=candidates,
                web_facts=compact_web,
            )
            filtered_classified = _filter_profile_observations(classified)
            if filtered_classified:
                return filtered_classified
    except Exception:
        pass

    heuristic_profile = _self_assessment_fallback_observations(message)
    if heuristic_profile:
        return _filter_profile_observations(heuristic_profile)

    fallback: list[dict[str, object]] = []
    for triple in list(triples or []):
        if triple.subject_type.strip().lower() != "user":
            continue
        relation = triple.relation.strip().upper()
        signal_type = "neutral"
        delta = 0.0
        if relation == "STRUGGLES_WITH":
            signal_type = "weakness"
            delta = 0.9
        elif relation == "STRENGTH_IN":
            signal_type = "strength"
            delta = 0.95
        elif relation in {"IMPROVED_IN", "IMPROVING_IN"}:
            signal_type = "improving"
            delta = 0.55
        elif relation == "STUDIES":
            signal_type = "neutral"
            delta = 0.1
        if signal_type == "neutral" and delta == 0.0:
            continue
        fallback.append(
            {
                "entity": _clean_profile_entity_text(triple.object_name, triple.object_type),
                "entity_type": triple.object_type,
                "entity_key": normalize_text_key(triple.object_name),
                "signal_type": signal_type,
                "delta": delta,
                "rationale": relation.lower(),
            }
        )
    deduped: dict[tuple[str, str], dict[str, object]] = {}
    for item in fallback:
        if not _is_useful_profile_entity(str(item.get("entity") or "")):
            continue
        key = (str(item.get("entity_key") or ""), str(item.get("signal_type") or ""))
        existing = deduped.get(key)
        if existing is None or float(item.get("delta") or 0.0) > float(existing.get("delta") or 0.0):
            deduped[key] = item
    return list(deduped.values())


def _normalize_string_list(value: object, *, fallback: list[str]) -> list[str]:
    items = [str(item).strip() for item in list(value or []) if str(item).strip()]
    return items[:6] or fallback


def _normalize_planner_stages(value: object) -> list[dict[str, object]]:
    stages: list[dict[str, object]] = []
    for item in list(value or []):
        if not isinstance(item, dict):
            continue
        stages.append(
            {
                "name": str(item.get("name") or "Stage").strip() or "Stage",
                "focus": str(item.get("focus") or "").strip(),
                "resource": str(item.get("resource") or "").strip(),
                "expected_questions": [str(question).strip() for question in list(item.get("expected_questions") or []) if str(question).strip()][:5],
            }
        )
    return stages[:8]


def _normalize_personalized_focus(
    value: object,
    *,
    memory_facts: list[str],
    profile_summary: dict[str, list[dict[str, object]]] | None = None,
) -> dict[str, list[str]]:
    payload = value if isinstance(value, dict) else {}
    profile = profile_summary or {}
    strengths = _normalize_string_list(
        payload.get("strengths_to_use"),
        fallback=_profile_entities_from_summary(profile.get("strengths")) or _profile_terms(memory_facts, prefix="STRENGTH_PROFILE"),
    )
    weaknesses = _normalize_string_list(
        payload.get("weaknesses_to_focus"),
        fallback=_profile_entities_from_summary(profile.get("weaknesses")) or _profile_terms(memory_facts, prefix="WEAKNESS_PROFILE"),
    )
    improving = _normalize_string_list(
        payload.get("improving_now"),
        fallback=_profile_entities_from_summary(profile.get("improving")) or _profile_terms(memory_facts, prefix="IMPROVING_PROFILE"),
    )
    return {
        "strengths_to_use": strengths[:5],
        "weaknesses_to_focus": weaknesses[:5],
        "improving_now": improving[:5],
    }


def _normalize_fit_analysis(
    value: object,
    *,
    web_topics: list[str],
    personalized_focus: dict[str, list[str]],
    profile_summary: dict[str, list[dict[str, object]]] | None = None,
) -> dict[str, object]:
    payload = value if isinstance(value, dict) else {}
    strengths = list(personalized_focus.get("strengths_to_use") or [])
    weaknesses = list(personalized_focus.get("weaknesses_to_focus") or [])
    improving = list(personalized_focus.get("improving_now") or [])
    profile = profile_summary or {}
    profile_strengths = list(profile.get("strengths") or [])
    profile_weaknesses = list(profile.get("weaknesses") or [])
    profile_improving = list(profile.get("improving") or [])
    fallback_strength_matches = _match_profile_summary_to_topics(profile_strengths, web_topics)
    fallback_weakness_matches = _match_profile_summary_to_topics(profile_weaknesses + profile_improving, web_topics)
    matched_strengths = _normalize_string_list(
        payload.get("matched_strengths"),
        fallback=fallback_strength_matches or _match_profile_to_topics(strengths, web_topics),
    )
    matched_weaknesses = _normalize_string_list(
        payload.get("matched_weaknesses"),
        fallback=fallback_weakness_matches or _match_profile_to_topics(weaknesses + improving, web_topics),
    )
    strategic_summary = str(payload.get("strategic_summary") or "").strip()
    if not strategic_summary:
        if matched_strengths and matched_weaknesses:
            strategic_summary = (
                f"Use {', '.join(matched_strengths[:2])} as your advantage, and start by fixing "
                f"{', '.join(matched_weaknesses[:2])} because those areas are likely to show up in this company process."
            )
        elif matched_weaknesses:
            strategic_summary = (
                f"Start with {', '.join(matched_weaknesses[:2])} because that overlaps with the company topics and is the clearest risk area right now."
            )
        elif matched_strengths:
            strategic_summary = (
                f"Lean on {', '.join(matched_strengths[:2])} because your cached strengths already align with the company topics."
            )
        elif strengths or weaknesses or improving:
            strategic_summary = (
                "Your cached profile exists, but the overlap with the detected company topics is still broad. Use the web topics to guide preparation and keep updating your profile through chat."
            )
        else:
            strategic_summary = (
                "No strong cached profile signals are available yet, so this planner is leaning mostly on the current web evidence."
            )
    return {
        "matched_strengths": matched_strengths[:5],
        "matched_weaknesses": matched_weaknesses[:5],
        "strategic_summary": strategic_summary,
    }


def _normalize_daily_plan(value: object, days: int) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for raw in list(value or []):
        if not isinstance(raw, dict):
            continue
        try:
            day = int(raw.get("day") or 0)
        except (TypeError, ValueError):
            day = 0
        items.append(
            {
                "day": day,
                "title": str(raw.get("title") or "Preparation block").strip() or "Preparation block",
                "tasks": [str(task).strip() for task in list(raw.get("tasks") or []) if str(task).strip()][:5],
                "goal": str(raw.get("goal") or "").strip(),
            }
        )
    items = [item for item in items if item["tasks"] or item["goal"] or item["title"]]
    items = items[:days]
    while len(items) < days:
        next_day = len(items) + 1
        items.append(
            {
                "day": next_day,
                "title": "Focused practice",
                "tasks": ["Review the most repeated round themes from the company evidence"],
                "goal": "Close one gap before the next round",
            }
        )
    for index, item in enumerate(items, start=1):
        item["day"] = index
    return items


def _fallback_planner_stages(*, company: str, web_results: list[dict[str, str]]) -> list[dict[str, object]]:
    combined = " ".join(
        " ".join(
            [
                str(item.get("title") or ""),
                str(item.get("snippet") or ""),
            ]
        )
        for item in web_results
    ).lower()
    stages: list[dict[str, object]] = []

    if re.search(r"\b(aptitude|quant|reasoning|psychometric|assessment)\b", combined):
        stages.append(
            {
                "name": "Assessment Round",
                "focus": "speed, reasoning, and accuracy",
                "resource": "timed practice sets",
                "expected_questions": ["aptitude", "logical reasoning", "short assessments"],
            }
        )
    coding_allowed = not re.search(r"\b(no coding|without coding|no coding questions)\b", combined)
    if coding_allowed and re.search(r"\b(coding|programming|dsa|algorithm|hackerrank|leetcode|online assessment)\b", combined):
        stages.append(
            {
                "name": "Coding Round",
                "focus": "problem solving under time pressure",
                "resource": "timed coding questions",
                "expected_questions": ["arrays", "strings", "graphs"],
            }
        )
    if re.search(r"\b(mcq|multiple choice|written test|technical mcq)\b", combined):
        stages.append(
            {
                "name": "Written Technical Test",
                "focus": "objective technical concepts and fast elimination",
                "resource": "MCQ revision sets",
                "expected_questions": ["technical MCQs", "fundamental concepts", "short applied questions"],
            }
        )
    if re.search(r"\b(analog|digital|semiconductor|electronics|vlsi|circuits|embedded)\b", combined):
        stages.append(
            {
                "name": "Domain Fundamentals",
                "focus": "company-specific core domain topics",
                "resource": "domain revision and concept review",
                "expected_questions": ["core fundamentals", "domain scenarios", "design basics"],
            }
        )
    if re.search(r"\b(case study|guesstimate|analysis|business case|analyst)\b", combined):
        stages.append(
            {
                "name": "Case or Analysis Round",
                "focus": "structured reasoning and communication",
                "resource": "case frameworks and mock analysis",
                "expected_questions": ["case walkthroughs", "analysis questions", "decision making"],
            }
        )
    if re.search(r"\b(group discussion|presentation|communication|hr|behavioral|behavioural)\b", combined):
        stages.append(
            {
                "name": "Communication Round",
                "focus": "clear speaking and behavioral responses",
                "resource": "mock speaking and behavioral prompts",
                "expected_questions": ["introduce yourself", "teamwork", "conflict handling"],
            }
        )
    if re.search(r"\b(technical|interview|domain|fundamentals)\b", combined):
        stages.append(
            {
                "name": "Technical or Domain Interview",
                "focus": "core concepts and explanation clarity",
                "resource": "topic revision and mock interview",
                "expected_questions": ["fundamentals", "applied questions", "scenario-based questions"],
            }
        )
    if not stages:
        stages = [
            {
                "name": "Screening Round",
                "focus": "understand the likely first-stage filters",
                "resource": "company-specific experience writeups",
                "expected_questions": ["screening questions", "common themes"],
            },
            {
                "name": "Interview Round",
                "focus": "structured answers and topic revision",
                "resource": "mock interview practice",
                "expected_questions": ["repeated interview themes", "company-specific questions"],
            },
        ]
    return stages[:5]


def _fallback_question_patterns(web_results: list[dict[str, str]]) -> list[str]:
    combined = " ".join(
        " ".join([str(item.get("title") or ""), str(item.get("snippet") or "")])
        for item in web_results
    ).lower()
    patterns: list[str] = []
    if "aptitude" in combined or "reasoning" in combined:
        patterns.append("Aptitude and reasoning screening")
    if re.search(r"\b(coding|programming|dsa|algorithm)\b", combined):
        patterns.append("Problem-solving and coding questions")
    if re.search(r"\b(mcq|multiple choice|written test)\b", combined):
        patterns.append("Written technical MCQs and quick concept checks")
    if re.search(r"\b(hr|behavioral|behavioural|communication)\b", combined):
        patterns.append("Behavioral and communication prompts")
    if re.search(r"\b(case|analysis|analyst)\b", combined):
        patterns.append("Case-style or analytical reasoning questions")
    if re.search(r"\b(technical|domain|fundamentals)\b", combined):
        patterns.append("Core technical or domain fundamentals")
    if re.search(r"\b(analog|digital|semiconductor|electronics|vlsi|circuits|embedded)\b", combined):
        patterns.append("Domain-specific fundamentals and applied concept questions")
    return patterns or ["Repeated round themes from company interview experiences"]


def _fallback_web_focus_topics(web_results: list[dict[str, str]]) -> list[str]:
    combined = " ".join(
        " ".join([str(item.get("title") or ""), str(item.get("snippet") or "")])
        for item in web_results
    ).lower()
    topics: list[str] = []
    if re.search(r"\b(aptitude|reasoning|quant)\b", combined):
        topics.append("Aptitude and reasoning")
    if re.search(r"\b(coding|dsa|algorithm|problem solving|online assessment)\b", combined):
        topics.append("Coding and problem solving")
    if re.search(r"\b(hr|behavioral|behavioural|communication)\b", combined):
        topics.append("Behavioral and communication")
    if re.search(r"\b(system design|design)\b", combined):
        topics.append("System or design thinking")
    if re.search(r"\b(technical|dbms|os|oops|cn|fundamentals)\b", combined):
        topics.append("Core technical fundamentals")
    if re.search(r"\b(analog|digital|semiconductor|electronics|vlsi|circuits|embedded)\b", combined):
        topics.append("Domain-specific fundamentals")
    return topics or ["Recruitment-process topics inferred from interview experiences"]


def _profile_terms(memory_facts: list[str], *, prefix: str) -> list[str]:
    results: list[str] = []
    for fact in memory_facts:
        if prefix not in fact:
            continue
        match = re.search(r"->\s*(.*?)\s*\(", fact)
        if match:
            value = match.group(1).strip()
            if value:
                results.append(value)
    return results


def _profile_entities_from_summary(items: object) -> list[str]:
    results: list[str] = []
    for item in list(items or []):
        if not isinstance(item, dict):
            continue
        entity = str(item.get("entity") or "").strip()
        if entity:
            results.append(entity)
    return results


def _match_profile_to_topics(profile_items: list[str], web_topics: list[str]) -> list[str]:
    results: list[str] = []
    normalized_topics = [(topic, set(normalize_text_key(topic).split())) for topic in web_topics if topic]
    for item in profile_items:
        item_key = normalize_text_key(item)
        item_tokens = set(item_key.split())
        if not item_tokens:
            continue
        matched = False
        for topic, topic_tokens in normalized_topics:
            if item_key in normalize_text_key(topic) or normalize_text_key(topic) in item_key or (item_tokens & topic_tokens):
                results.append(f"{item} -> {topic}")
                matched = True
                break
        if not matched and any(token in {"coding", "problem", "technical", "design", "communication", "reasoning"} for token in item_tokens):
            results.append(item)
    deduped: list[str] = []
    seen: set[str] = set()
    for item in results:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _compact_profile_summary(
    profile_summary: dict[str, list[dict[str, object]]] | None,
) -> dict[str, list[dict[str, object]]]:
    compact: dict[str, list[dict[str, object]]] = {"strengths": [], "weaknesses": [], "improving": []}
    for section in ("strengths", "weaknesses", "improving"):
        for item in list((profile_summary or {}).get(section) or [])[:5]:
            if not isinstance(item, dict):
                continue
            entity = str(item.get("entity") or "").strip()
            if not entity:
                continue
            compact[section].append(
                {
                    "entity": entity,
                    "entity_type": str(item.get("entity_type") or "Skill").strip() or "Skill",
                    "score": round(float(item.get("score") or 0.0), 2),
                    "improving_score": round(float(item.get("improving_score") or 0.0), 2),
                    "evidence_count": int(item.get("evidence_count") or 0),
                }
            )
    return compact


def _extract_bucketed_profile_observations_with_gemini(*, client: genai.Client, prompt: str) -> list[dict[str, object]]:
    response = client.models.generate_content(
        model=SIGNAL_MODEL,
        contents=prompt,
    )
    payload = _parse_semantic_response(response.text or "")
    observations: list[dict[str, object]] = []
    for signal_type, bucket in (
        ("strength", payload.get("strengths")),
        ("weakness", payload.get("weaknesses")),
        ("improving", payload.get("improving")),
    ):
        for item in list(bucket or []):
            if not isinstance(item, dict):
                continue
            entity = str(item.get("entity") or "").strip()
            entity_type = str(item.get("entity_type") or "Skill").strip() or "Skill"
            rationale = str(item.get("rationale") or "").strip()
            try:
                delta = float(item.get("delta") or 0.0)
            except (TypeError, ValueError):
                delta = 0.0
            if not entity:
                continue
            observations.append(
                {
                    "entity": _clean_profile_entity_text(entity, entity_type),
                    "entity_type": entity_type,
                    "entity_key": normalize_text_key(entity),
                    "signal_type": signal_type,
                    "delta": max(-1.5, min(delta, 1.5)),
                    "rationale": rationale,
                }
            )
    return observations


def _classify_profile_candidates_with_gemini(
    *,
    client: genai.Client,
    message: str,
    candidates: list[str],
    web_facts: list[str] | None = None,
) -> list[dict[str, object]]:
    prompt = f"""
Classify whether each candidate topic from the user's message should count as a strength, weakness, improving area, or be ignored.
Return strict JSON only with this exact shape:
{{
  "items": [
    {{
      "entity": "Problem Solving",
      "entity_type": "Skill",
      "signal_type": "strength",
      "delta": 0.8,
      "rationale": "user explicitly says they are strong in it"
    }}
  ]
}}

Rules:
- signal_type must be one of: strength, weakness, improving, ignore
- Use ignore for vague, generic, or unsupported candidates.
- Classify based on the user's wording first.
- Use the web context only to understand whether the candidate is a relevant interview/preparation topic.
- Do not invent candidates.
- Keep entity short and atomic.

Candidates:
{json.dumps(candidates[:10], ensure_ascii=True)}

Relevant web context:
{json.dumps(list(web_facts or [])[:5], ensure_ascii=True)}

User message:
{message}
""".strip()
    response = client.models.generate_content(
        model=SIGNAL_MODEL,
        contents=prompt,
    )
    payload = _parse_semantic_response(response.text or "")
    observations: list[dict[str, object]] = []
    for item in list(payload.get("items") or []):
        if not isinstance(item, dict):
            continue
        entity = str(item.get("entity") or "").strip()
        entity_type = str(item.get("entity_type") or "Skill").strip() or "Skill"
        signal_type = str(item.get("signal_type") or "ignore").strip().lower()
        rationale = str(item.get("rationale") or "").strip()
        try:
            delta = float(item.get("delta") or 0.0)
        except (TypeError, ValueError):
            delta = 0.0
        if not entity or signal_type not in {"strength", "weakness", "improving"}:
            continue
        observations.append(
            {
                "entity": _clean_profile_entity_text(entity, entity_type),
                "entity_type": entity_type,
                "entity_key": normalize_text_key(entity),
                "signal_type": signal_type,
                "delta": max(-1.5, min(delta or 0.7, 1.5)),
                "rationale": rationale or "candidate_classification",
            }
        )
    return observations


def _message_has_self_assessment_cues(message: str) -> bool:
    lowered = " ".join((message or "").lower().split())
    patterns = (
        r"\bi am good at\b",
        r"\bi am strong in\b",
        r"\bi am confident in\b",
        r"\bi am comfortable with\b",
        r"\bi am weak in\b",
        r"\bi struggle with\b",
        r"\bi am working on\b",
        r"\bi am improving\b",
        r"\bi am practicing\b",
        r"\bmy strength\b",
        r"\bmy weakness\b",
        r"\bi find .* challenging\b",
    )
    return any(re.search(pattern, lowered) for pattern in patterns)


def _candidate_profile_entities_from_message(
    *,
    message: str,
    triples: list[TripleCandidate] | None = None,
    seed_observations: list[dict[str, object]] | None = None,
) -> list[str]:
    results: list[str] = []
    text = " ".join((message or "").split())
    cue_patterns = (
        r"\b(?:i am|i'm)\s+(?:good at|strong in|confident in|comfortable with)\s+(.+?)(?:[.!]|$)",
        r"\b(?:i am|i'm)\s+(?:weak in|struggling with|working on|improving|actively improving)\s+(.+?)(?:[.!]|$)",
        r"\b(?:i find)\s+(.+?)\s+(?:challenging|difficult|hard|tricky)(?:[.!]|$)",
        r"\b(?:focusing on improving|aiming to become|building confidence in)\s+(.+?)(?:[.!]|$)",
    )
    for pattern in cue_patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            results.extend(_split_profile_entities(match.group(1)))
    for item in list(seed_observations or [])[:8]:
        if not isinstance(item, dict):
            continue
        entity = _clean_profile_entity_text(
            str(item.get("entity") or ""),
            str(item.get("entity_type") or "Skill"),
        )
        if _is_useful_profile_entity(entity):
            results.append(entity)
    for triple in list(triples or [])[:10]:
        if triple.subject_type.strip().lower() != "user":
            continue
        entity = _clean_profile_entity_text(triple.object_name, triple.object_type)
        if _is_useful_profile_entity(entity):
            results.append(entity)
    return _dedupe_strings(results)


def _self_assessment_fallback_observations(message: str) -> list[dict[str, object]]:
    text = " ".join((message or "").split())
    if not _message_has_self_assessment_cues(text):
        return []
    observations: list[dict[str, object]] = []
    patterns = [
        (r"\b(?:i am|i'm)\s+(?:good at|strong in|confident in|comfortable with)\s+(.+?)(?:[.!]|$)", "strength", 0.85),
        (r"\b(?:i am|i'm)\s+(?:weak in|struggling with)\s+(.+?)(?:[.!]|$)", "weakness", 0.9),
        (r"\b(?:i am|i'm)\s+(?:working on|improving|actively improving)\s+(.+?)(?:[.!]|$)", "improving", 0.78),
        (r"\b(?:i find)\s+(.+?)\s+(?:challenging|difficult|hard|tricky)(?:[.!]|$)", "weakness", 0.82),
        (r"\b(?:aiming to become|focusing on improving|build(?:ing)? confidence in)\s+(.+?)(?:[.!]|$)", "improving", 0.72),
    ]
    for pattern, signal_type, delta in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            phrase = match.group(1).strip(" .,-")
            for entity in _split_profile_entities(phrase):
                observations.append(
                    {
                        "entity": entity,
                        "entity_type": "Skill",
                        "entity_key": normalize_text_key(entity),
                        "signal_type": signal_type,
                        "delta": delta,
                        "rationale": "self_assessment_fallback",
                    }
                )
    return observations


def _split_profile_entities(text: str) -> list[str]:
    cleaned = re.sub(r"\b(?:my|skills? in|performance in|knowledge of|topics like|areas like)\b", " ", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(?:and|or)\b", ",", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("/", ",")
    parts = [part.strip(" .,-") for part in cleaned.split(",")]
    results: list[str] = []
    for part in parts:
        candidate = _clean_profile_entity_text(part, "Skill")
        if _is_useful_profile_entity(candidate):
            results.append(candidate)
    return _dedupe_strings(results)


def _clean_profile_entity_text(entity: str, entity_type: str) -> str:
    cleaned = _clean_entity_text(entity, entity_type)
    cleaned = re.sub(r"^(topics?|skills?|areas?)\s+(like|such as)\s+", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"^(working on|improving in|good at|confident in|comfortable with)\s+", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"\b(for interviews?|in these areas|these areas|those areas)\b", "", cleaned, flags=re.IGNORECASE).strip(" ,.-")
    return cleaned


def _filter_profile_observations(observations: list[dict[str, object]]) -> list[dict[str, object]]:
    filtered: list[dict[str, object]] = []
    for item in observations:
        entity = _clean_profile_entity_text(
            str(item.get("entity") or ""),
            str(item.get("entity_type") or "Skill"),
        )
        if not _is_useful_profile_entity(entity):
            continue
        updated = dict(item)
        updated["entity"] = entity
        updated["entity_key"] = normalize_text_key(entity)
        if str(updated.get("signal_type") or "").strip().lower() == "neutral":
            continue
        filtered.append(updated)
    deduped: dict[tuple[str, str], dict[str, object]] = {}
    for item in filtered:
        key = (str(item.get("entity_key") or ""), str(item.get("signal_type") or ""))
        existing = deduped.get(key)
        if existing is None or float(item.get("delta") or 0.0) > float(existing.get("delta") or 0.0):
            deduped[key] = item
    return list(deduped.values())


def _is_useful_profile_entity(entity: str) -> bool:
    value = " ".join((entity or "").split()).strip()
    if not value:
        return False
    lowered = value.lower()
    blocked_phrases = {
        "these areas",
        "those areas",
        "to improve in these areas",
        "topics like",
        "skills like",
        "core concepts",
        "understanding core concepts",
        "programming fundamentals and problem solving",
        "advanced topics",
        "coding interviews",
    }
    if lowered in blocked_phrases:
        return False
    if len(lowered) < 3:
        return False
    if len(lowered) > 48:
        return False
    if re.search(r"\b(this|that|these|those|something|anything|everything)\b", lowered):
        return False
    if lowered.startswith(("to ", "for ", "in ", "with ", "on ")):
        return False
    alpha_tokens = re.findall(r"[a-zA-Z0-9\+\#]+", lowered)
    if not alpha_tokens:
        return False
    if len(alpha_tokens) > 7:
        return False
    return True


def _match_profile_summary_to_topics(
    profile_items: list[dict[str, object]],
    web_topics: list[str],
) -> list[str]:
    results: list[str] = []
    for item in profile_items:
        if not isinstance(item, dict):
            continue
        entity = str(item.get("entity") or "").strip()
        if not entity:
            continue
        topic = _best_topic_match(entity, web_topics)
        if not topic:
            continue
        score = float(item.get("score") or 0.0)
        improving_score = float(item.get("improving_score") or 0.0)
        evidence_count = int(item.get("evidence_count") or 0)
        meta_bits: list[str] = []
        if abs(score) >= 0.35:
            meta_bits.append(f"score {score:.2f}")
        if improving_score > 0.3:
            meta_bits.append(f"improving {improving_score:.2f}")
        if evidence_count > 0:
            meta_bits.append(f"{evidence_count} signals")
        meta = f" ({', '.join(meta_bits)})" if meta_bits else ""
        results.append(f"{entity} -> {topic}{meta}")
    return _dedupe_strings(results)


def _best_topic_match(entity: str, web_topics: list[str]) -> str | None:
    entity_key = normalize_text_key(entity)
    entity_tokens = set(entity_key.split())
    if not entity_tokens:
        return None
    entity_categories = _semantic_categories(entity_key)
    best_topic = None
    best_score = 0
    for topic in web_topics:
        topic_key = normalize_text_key(topic)
        topic_tokens = set(topic_key.split())
        topic_categories = _semantic_categories(topic_key)
        overlap = len(entity_tokens & topic_tokens)
        category_overlap = len(entity_categories & topic_categories)
        contains_bonus = 2 if (entity_key in topic_key or topic_key in entity_key) else 0
        score = overlap + (category_overlap * 3) + contains_bonus
        if score > best_score:
            best_score = score
            best_topic = topic
    return best_topic if best_score > 0 else None


def _semantic_categories(text: str) -> set[str]:
    categories: set[str] = set()
    checks = {
        "coding": r"\b(coding|code|programming|problem|problem solving|dsa|algorithm|arrays?|strings?|graphs?|trees?|dp|dynamic programming|leetcode|debugging)\b",
        "technical": r"\b(technical|fundamentals|oops|os|dbms|cn|computer science|core concepts|optimization)\b",
        "design": r"\b(system design|design|architecture|scalability|distributed)\b",
        "communication": r"\b(communication|behavioral|behavioural|interview|confidence|speaking|presentation|hr)\b",
        "assessment": r"\b(aptitude|reasoning|quant|assessment|mcq|written)\b",
        "domain": r"\b(domain|analog|digital|semiconductor|electronics|embedded|vlsi|circuits)\b",
    }
    for name, pattern in checks.items():
        if re.search(pattern, text):
            categories.add(name)
    return categories


def _dedupe_strings(items: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for item in items:
        key = item.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(item.strip())
    return deduped


def _fallback_daily_plan(
    *,
    company: str,
    days: int,
    stages: list[dict[str, object]],
    web_results: list[dict[str, str]],
    memory_facts: list[str],
) -> list[dict[str, object]]:
    if not stages:
        stages = [
            {
                "name": "Preparation Round",
                "focus": "company-specific interview readiness",
                "resource": "practice and revision",
                "expected_questions": ["common interview themes"],
            }
        ]
    weak_hint = next((fact for fact in memory_facts if "STRUGGLES_WITH" in fact or "STUDIES" in fact), "")
    stage_cycle = [stages[index % len(stages)] for index in range(days)]
    question_patterns = _fallback_question_patterns(web_results)
    daily_plan: list[dict[str, object]] = []
    for day in range(1, days + 1):
        stage = stage_cycle[day - 1]
        stage_name = str(stage.get("name") or "Preparation")
        focus = str(stage.get("focus") or "focused practice")
        expected = [str(item) for item in list(stage.get("expected_questions") or []) if str(item)]
        primary_question = expected[0] if expected else "common questions"
        if day <= max(2, days // 3):
            title = f"Foundation: {stage_name}"
            tasks = [
                f"Map the {company} {stage_name.lower()} expectations from recent interview experiences",
                f"Revise {focus}",
                f"Practice 8-12 questions around {primary_question}",
            ]
            goal = f"Build baseline confidence for {stage_name.lower()}."
        elif day <= max(4, (2 * days) // 3):
            title = f"Practice Sprint: {stage_name}"
            tasks = [
                f"Do one timed practice block for {stage_name.lower()}",
                f"Review mistakes from {primary_question}",
                f"Summarize repeat patterns: {question_patterns[(day - 1) % len(question_patterns)]}",
            ]
            goal = f"Convert revision into performance for {stage_name.lower()}."
        elif day < days:
            title = f"Mock and Review: {stage_name}"
            tasks = [
                f"Run one mock session focused on {stage_name.lower()}",
                f"Fix two weak areas from earlier attempts",
                f"Prepare concise answers for {primary_question}",
            ]
            goal = f"Simulate the real round and tighten weak spots."
        else:
            title = "Final Revision and Confidence Check"
            tasks = [
                f"Review the highest-probability rounds for {company}",
                f"Revisit the most repeated question patterns from the web evidence",
                f"Do a calm final pass on {weak_hint or 'your weakest area'}",
            ]
            goal = f"Enter the {company} process with a clean revision summary and confidence."
        daily_plan.append(
            {
                "day": day,
                "title": title,
                "tasks": tasks,
                "goal": goal,
            }
        )
    return daily_plan


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
- Read the whole message semantically, including 2-3 sentence messages and conjunction-heavy phrasing, before deciding.
- If the user names a topic in one part of the message and expresses preparation, interest, confusion, or weakness in another part, connect them.
- Do not require rigid sentence forms. The user may express preparation or weakness in any wording.
- Abstract abilities and academic habits count too, for example answer writing, revision, communication, analysis, time management, and consistency.
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
  - "i want to improve my answer writing"
  - "my revision is weak"
  - "i need to work on communication"

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
        r"(?:i want to improve|i need to improve|i need to work on|i want to work on)\s+(?:my\s+)?([a-z0-9 +#&/().,\-]{2,80})",
        r"(?:my\s+)([a-z0-9 +#&/().,\-]{2,80})\s+(?:is|feels)\s+(?:weak|poor|bad|difficult|hard|confusing|tricky|tough)",
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

    resolved_company = _resolve_company_name(raw_subject)
    looks_like_company = bool(resolved_company)
    abstract_skill_terms = {"answer writing", "revision", "communication", "analysis", "time management", "consistency", "problem solving", "writing speed", "presentation", "clarity"}
    object_type = "Company" if looks_like_company else "Topic"
    relation = "TARGETS" if object_type == "Company" else "STUDIES"
    normalized_subject = raw_subject.strip().lower()
    if normalized_subject in abstract_skill_terms:
        object_type = "Skill"
        relation = "IMPROVED_IN" if re.search(r"\bimprove|work on\b", lowered) else "STRUGGLES_WITH"
    object_name = resolved_company if resolved_company else _clean_entity_text(raw_subject, object_type)
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
    relation = str(triple.get("relation") or "RELATED_TO").strip() or "RELATED_TO"
    object_name, object_type = _resolve_entity_type_and_name(
        object_name=str(triple.get("object_name") or "").strip(),
        object_type=object_type,
        relation=relation,
    )
    return TripleCandidate(
        user_id=user_id,
        subject_type=str(triple.get("subject_type") or "User").strip() or "User",
        subject_name=str(triple.get("subject_name") or user_id).strip() or user_id,
        relation=relation,
        object_type=object_type,
        object_name=object_name,
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

        focus_match = re.search(
            r"(?:focusing on|focused on|currently focusing on|working through)\s+([a-zA-Z0-9 +#,\-]{2,80})",
            clause,
            flags=re.IGNORECASE,
        )
        if focus_match:
            for object_name in _split_memory_objects(focus_match.group(1).strip(" ."), "Topic"):
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
                        confidence=0.84,
                        source=source,
                        raw_text=message,
                        linked_to_action=True,
                    )
                )

        difficulty_match = re.search(
            r"([a-zA-Z0-9 +#,\-]{2,60})\s+(?:is|feels)\s+(?:a bit |kind of |quite )?(?:difficult|hard|confusing|tricky|tough)",
            clause,
            flags=re.IGNORECASE,
        )
        if not difficulty_match:
            difficulty_match = re.search(
                r"find\s+([a-zA-Z0-9 +#,\-]{2,60}?)(?:\s+(?:a bit|kind of|quite))?\s+(?:difficult|hard|confusing|tricky|tough)",
                clause,
                flags=re.IGNORECASE,
            )
        if difficulty_match:
            for object_name in _split_memory_objects(difficulty_match.group(1).strip(" ."), "Skill"):
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
            company = _resolve_company_name(match.group(1).strip(" ."))
            if company and _is_valid_company_name(company):
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


def _resolve_entity_type_and_name(*, object_name: str, object_type: str, relation: str) -> tuple[str, str]:
    cleaned_type = object_type.strip() or "Entity"
    cleaned_name = _clean_entity_text(object_name, cleaned_type)
    relation_key = relation.strip().upper()
    should_try_company = (
        cleaned_type.lower() == "company"
        or relation_key in {"TARGETS", "INTERVIEWED_AT"}
        or bool(re.search(r"\binterview\b", cleaned_name, flags=re.IGNORECASE))
    )
    if should_try_company:
        resolved_company = _resolve_company_name(cleaned_name)
        if resolved_company:
            return resolved_company, "Company"
    return cleaned_name, cleaned_type


def _resolve_company_name(text: str) -> str | None:
    cleaned = re.sub(r"\s+", " ", (text or "").strip(" .,\t\r\n"))
    if not cleaned:
        return None
    cache_key = cleaned.lower()
    if cache_key in _COMPANY_CLASSIFICATION_CACHE:
        return _COMPANY_CLASSIFICATION_CACHE[cache_key]

    prompt = f"""
Decide whether the given text refers to a real company or organization.
Return strict JSON only with this exact shape:
{{
  "is_company": true,
  "company_name": "Microsoft"
}}

Rules:
- If the text includes extra context like interview, role, job, or preparation, extract only the company name when it is clearly a company.
- If the text is a topic, domain, skill, exam, field, or generic phrase, return is_company=false and company_name="".
- company_name must be short and atomic.
- Do not guess when unsure.

Text: {cleaned}
""".strip()

    resolved: str | None = None
    try:
        client = _get_client()
        response = client.models.generate_content(
            model=SIGNAL_MODEL,
            contents=prompt,
        )
        payload = _parse_semantic_response(response.text or "")
        if bool(payload.get("is_company")):
            candidate = _clean_entity_text(str(payload.get("company_name") or "").strip(), "Company")
            if _is_valid_company_name(candidate):
                resolved = candidate
    except Exception:
        resolved = None

    _COMPANY_CLASSIFICATION_CACHE[cache_key] = resolved
    return resolved


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
