from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from html import unescape
import hashlib
import math
import re
import time
from urllib.parse import parse_qs, quote_plus, unquote, urlparse

import httpx


SEARCH_INTENT_PROTOTYPES = {
    "prep_guidance": "user asking what to study practice or prepare for an organization company exam institution role or target",
    "current_info": "user asking for latest recent current news updates live information",
    "general_learning": "user asking what to study or practice for a topic subject or concept",
}

PREP_PATTERNS = (
    r"\bwhat should i study\b",
    r"\bwhat should i practice\b",
    r"\bhow do i prepare\b",
    r"\bhow should i prepare\b",
    r"\bwhat topics should i focus\b",
    r"\bprepare for\b",
    r"\bfocusing on\b",
    r"\btargeting\b",
)

ROLE_PATTERNS = (
    "software engineer",
    "backend engineer",
    "frontend engineer",
    "data scientist",
    "ml engineer",
    "embedded engineer",
    "firmware engineer",
    "analog engineer",
    "digital design",
    "civil services",
)

EXAM_HINTS = {"upsc", "gate", "jee", "neet", "cat", "gre", "toefl", "ielts"}
INSTITUTION_HINTS = {"iit", "nit", "mit", "stanford", "harvard"}
_SEARCH_CACHE: dict[str, tuple[float, list["WebResult"]]] = {}
_CACHE_TTL_SECONDS = 6 * 60 * 60


@dataclass
class WebResult:
    title: str
    snippet: str
    url: str

    def to_text(self) -> str:
        return f"{self.title}: {self.snippet} ({self.url})"


@dataclass
class SearchPlan:
    should_search: bool
    intent: str
    confidence: float
    entities: dict[str, str]
    queries: list[str]
    reason: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_search_plan(
    *,
    message: str,
    semantic_topic: str | None = None,
    route_intent: str | None = None,
) -> SearchPlan:
    text = " ".join((message or "").split())
    lowered = text.lower()

    intent_scores = _intent_scores(lowered)
    intent = max(intent_scores, key=intent_scores.get)
    confidence = intent_scores[intent]

    entity = _extract_target_entity(text)
    role = _extract_role(lowered)
    entity_type = _infer_entity_type(entity.lower() if entity else "")
    should_search = False
    reason = "no web search needed"
    queries: list[str] = []

    if entity and intent == "prep_guidance":
        should_search = True
        reason = f"semantic prep intent for {entity_type}"
        queries = _prep_queries(
            entity=entity,
            entity_type=entity_type,
            role=role,
            semantic_topic=semantic_topic,
        )
    elif intent == "current_info":
        should_search = True
        reason = "current-info intent"
        queries = [text]
    elif semantic_topic and intent == "general_learning" and route_intent in {"respond_and_retrieve", "respond_retrieve_and_update"}:
        should_search = True
        reason = "semantic topic-learning intent"
        queries = [
            f"{semantic_topic} important practice topics",
            f"{semantic_topic} most important concepts to practice",
        ]

    return SearchPlan(
        should_search=should_search,
        intent=intent,
        confidence=round(confidence, 4),
        entities={
            "entity": entity or "",
            "entity_type": entity_type,
            "role": role or "",
            "topic": semantic_topic or "",
        },
        queries=queries[:4],
        reason=reason,
    )


def search_from_plan(plan: SearchPlan, limit: int = 4) -> list[WebResult]:
    if not plan.should_search:
        return []
    cache_key = _cache_key_for_plan(plan, limit=limit)
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached
    results: list[WebResult] = []
    seen_urls: set[str] = set()
    for query in plan.queries:
        for result in _duckduckgo_search(query, limit=limit):
            if result.url in seen_urls:
                continue
            seen_urls.add(result.url)
            results.append(result)
            if len(results) >= limit:
                _cache_set(cache_key, results)
                return results
    _cache_set(cache_key, results)
    return results


def _intent_scores(text: str) -> dict[str, float]:
    query_vec = _hash_embed(text)
    scores = {intent: _cosine_similarity(query_vec, list(proto_vec)) for intent, proto_vec in _prototype_vectors().items()}
    if any(re.search(pattern, text) for pattern in PREP_PATTERNS):
        scores["prep_guidance"] += 0.32
        scores["general_learning"] += 0.14
    if re.search(r"\b(latest|recent|today|current|news|update)\b", text):
        scores["current_info"] += 0.34
    return {key: round(min(value, 1.0), 4) for key, value in scores.items()}


def _prep_queries(*, entity: str, entity_type: str, role: str | None, semantic_topic: str | None) -> list[str]:
    queries: list[str] = []
    if entity_type == "exam":
        queries.extend(
            [
                f"{entity} syllabus important topics",
                f"{entity} preparation strategy important subjects",
                f"{entity} common preparation roadmap",
            ]
        )
    elif entity_type == "institution":
        queries.extend(
            [
                f"{entity} admission preparation important topics",
                f"{entity} entrance preparation roadmap",
                f"{entity} interview or selection preparation tips",
            ]
        )
    else:
        queries.extend(
            [
                f"{entity} preparation topics",
                f"{entity} interview or selection preparation areas",
                f"{entity} commonly expected skills and topics",
            ]
        )
    if role:
        queries.insert(0, f"{entity} {role} preparation topics")
    if semantic_topic:
        queries.append(f"{entity} {semantic_topic} topic importance")
    return queries


def _extract_target_entity(text: str) -> str | None:
    patterns = [
        r"\b(?:for|at|with|targeting|applying to)\s+([A-Za-z][A-Za-z0-9&.\- ]{1,40})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        entity = match.group(1).strip(" .?")
        entity = re.split(r"\b(?:role|roles|job|jobs|interview|interviews)\b", entity, maxsplit=1, flags=re.IGNORECASE)[0].strip(" .")
        if entity:
            return entity
    return None


def infer_memory_signals_from_plan(
    *,
    message: str,
    user_id: str,
    plan: SearchPlan,
) -> list[dict[str, object]]:
    if not plan.should_search:
        return []
    lowered = " ".join((message or "").lower().split())
    if " i " not in f" {lowered} " and " my " not in f" {lowered} ":
        return []

    entity = str(plan.entities.get("entity") or "").strip()
    entity_type = str(plan.entities.get("entity_type") or "").strip().lower()
    topic = str(plan.entities.get("topic") or "").strip()
    raw_signals: list[dict[str, object]] = []

    if entity:
        relation = "TARGETS" if entity_type == "organization" else "PREPARES_FOR"
        signal_type = "Company" if entity_type == "organization" else "Goal"
        raw_signals.append(
            {
                "user_id": user_id,
                "entity": entity,
                "entity_type": signal_type,
                "relation": relation,
                "confidence": 0.83,
                "linked_to_action": True,
                "source": "search_plan",
                "raw_text": message,
            }
        )

    if topic and re.search(r"\b(study|practice|prepare|focus)\b", lowered):
        raw_signals.append(
            {
                "user_id": user_id,
                "entity": topic,
                "entity_type": "Topic",
                "relation": "STUDIES",
                "confidence": 0.74,
                "linked_to_action": True,
                "source": "search_plan",
                "raw_text": message,
            }
        )

    return raw_signals


def _infer_entity_type(entity: str) -> str:
    normalized = entity.strip().lower()
    if normalized in EXAM_HINTS:
        return "exam"
    if normalized in INSTITUTION_HINTS:
        return "institution"
    return "organization"


def _extract_role(text: str) -> str | None:
    for role in ROLE_PATTERNS:
        if role in text:
            return role
    return None


def _duckduckgo_search(query: str, limit: int = 4) -> list[WebResult]:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
    }
    with httpx.Client(timeout=8.0, follow_redirects=True, headers=headers) as client:
        html_results = _duckduckgo_html_search(client=client, query=query, limit=limit)
        if html_results:
            return html_results[:limit]
        lite_results = _duckduckgo_lite_search(client=client, query=query, limit=limit)
        if lite_results:
            return lite_results[:limit]
    return []


def _duckduckgo_html_search(*, client: httpx.Client, query: str, limit: int) -> list[WebResult]:
    url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
    try:
        response = client.get(url)
        response.raise_for_status()
        html = response.text
    except Exception:
        return []

    title_matches = list(
        re.finditer(
            r'<a[^>]*class="result__a"[^>]*href="(?P<url>[^"]+)"[^>]*>(?P<title>.*?)</a>',
            html,
            flags=re.IGNORECASE | re.DOTALL,
        )
    )
    snippet_matches = re.findall(
        r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>|<div[^>]*class="result__snippet"[^>]*>(.*?)</div>',
        html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    snippets = []
    for left, right in snippet_matches:
        snippet = _clean_html(left or right)
        if snippet:
            snippets.append(snippet)

    results: list[WebResult] = []
    for index, match in enumerate(title_matches[:limit]):
        raw_url = _decode_duckduckgo_url(unescape(match.group("url")))
        title = _clean_html(match.group("title"))
        snippet = snippets[index] if index < len(snippets) else ""
        if title and raw_url and _is_useful_search_result(raw_url, title=title, snippet=snippet):
            results.append(WebResult(title=title, snippet=snippet, url=raw_url))
    return results


def _duckduckgo_lite_search(*, client: httpx.Client, query: str, limit: int) -> list[WebResult]:
    url = f"https://lite.duckduckgo.com/lite/?q={quote_plus(query)}"
    try:
        response = client.get(url)
        response.raise_for_status()
        html = response.text
    except Exception:
        return []

    results: list[WebResult] = []
    pattern = re.compile(
        r'<a[^>]*href="(?P<url>[^"]+)"[^>]*>(?P<title>.*?)</a>',
        flags=re.IGNORECASE | re.DOTALL,
    )
    for match in pattern.finditer(html):
        raw_url = _decode_duckduckgo_url(unescape(match.group("url")))
        title = _clean_html(match.group("title"))
        if not title or not raw_url:
            continue
        if not _is_useful_search_result(raw_url, title=title, snippet=""):
            continue
        results.append(WebResult(title=title, snippet="", url=raw_url))
        if len(results) >= limit:
            break
    return results


def _decode_duckduckgo_url(url: str) -> str:
    parsed = urlparse(url)
    if "duckduckgo.com" not in (parsed.netloc or ""):
        return url
    query = parse_qs(parsed.query or "")
    uddg = query.get("uddg")
    if uddg and uddg[0]:
        return unquote(uddg[0])
    return url


def _is_useful_search_result(url: str, *, title: str, snippet: str) -> bool:
    lowered_url = (url or "").lower()
    lowered_text = f"{title} {snippet}".lower()
    blocked_fragments = (
        "duckduckgo.com/y.js",
        "jobrapido",
        "talent.com",
        "foundit",
        "naukri.com",
        "simplyhired",
        "job openings",
        "apply now",
        "urgent hiring",
    )
    if any(fragment in lowered_url for fragment in blocked_fragments):
        return False
    if any(fragment in lowered_text for fragment in blocked_fragments):
        return False
    return True


def _cache_key_for_plan(plan: SearchPlan, *, limit: int) -> str:
    normalized_queries = "|".join(query.strip().lower() for query in plan.queries[:limit])
    entity = str(plan.entities.get("entity") or "").strip().lower()
    entity_type = str(plan.entities.get("entity_type") or "").strip().lower()
    return f"{plan.intent}|{entity_type}|{entity}|{normalized_queries}|{limit}"


def _cache_get(key: str) -> list[WebResult] | None:
    payload = _SEARCH_CACHE.get(key)
    if payload is None:
        return None
    expires_at, results = payload
    if expires_at < time.time():
        _SEARCH_CACHE.pop(key, None)
        return None
    return results


def _cache_set(key: str, results: list[WebResult]) -> None:
    _SEARCH_CACHE[key] = (time.time() + _CACHE_TTL_SECONDS, list(results))


def _clean_html(value: str) -> str:
    text = re.sub(r"<.*?>", " ", value or "")
    text = unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _hash_embed(text: str, dimensions: int = 96) -> list[float]:
    values = [0.0] * dimensions
    for token in re.findall(r"\w+", text.lower()):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = digest[0] % dimensions
        sign = 1.0 if digest[1] % 2 == 0 else -1.0
        values[index] += sign
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0.0:
        return values
    return [value / norm for value in values]


@lru_cache(maxsize=8)
def _prototype_vectors() -> dict[str, tuple[float, ...]]:
    return {intent: tuple(_hash_embed(text)) for intent, text in SEARCH_INTENT_PROTOTYPES.items()}


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return max(0.0, min(dot / (norm_a * norm_b), 1.0))
