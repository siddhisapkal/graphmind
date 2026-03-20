from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import json
import math
import os
import re

from dotenv import load_dotenv
from groq import Groq


MEMORY_PATTERNS = (
    r"\bi am\b",
    r"\bi'?m\b",
    r"\bi have\b",
    r"\bi like\b",
    r"\bi love\b",
    r"\bi prefer\b",
    r"\bi struggle with\b",
    r"\bi am struggling with\b",
    r"\bi am weak at\b",
    r"\bi am weak in\b",
    r"\bi am good at\b",
    r"\bi am strong at\b",
    r"\bi am targeting\b",
    r"\bi am applying to\b",
    r"\bi am learning\b",
    r"\bi am studying\b",
    r"\bmy goal\b",
    r"\bmy weakness\b",
)

RETRIEVAL_PATTERNS = (
    r"\bwhat did i\b",
    r"\bwhat do i\b",
    r"\bwhich\b",
    r"\bremember\b",
    r"\brecall\b",
    r"\bpast\b",
    r"\bprevious\b",
    r"\bweak\b",
    r"\bstrength\b",
    r"\btarget\b",
    r"\bgoal\b",
    r"\bcompany\b",
    r"\btopic\b",
    r"\bskill\b",
    r"\bmy weakness\b",
    r"\bmy strengths?\b",
)

ACTION_PATTERNS = (
    r"\bexplain\b",
    r"\bhelp\b",
    r"\bsolve\b",
    r"\bwrite\b",
    r"\bsummarize\b",
    r"\bplan\b",
    r"\bhow\b",
    r"\bwhy\b",
    r"\bteach\b",
)

QUESTION_WORDS = {"what", "why", "how", "when", "where", "which", "who", "can", "should"}
INTENT_PROTOTYPES = {
    "MEMORY": "user stating personal facts preferences weaknesses strengths goals targets study habits or durable self information",
    "RETRIEVAL": "user asking about past information stored memory user history weaknesses strengths targets goals or previous facts",
    "RESPONSE": "user asking for explanation help solving teaching summarizing or general answer without depending on memory",
}

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))


@dataclass
class PromptRouteDecision:
    intent: str
    needs_retrieval: bool
    needs_memory_update: bool
    use_graph: bool
    use_vector: bool
    confidence: float
    scores: dict[str, float]
    layer: str
    semantic_topic: str | None = None
    reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def route_prompt(message: str, semantic_topic: str | None = None) -> PromptRouteDecision:
    text = " ".join((message or "").strip().split())
    lowered = text.lower()
    tokens = re.findall(r"[a-z0-9']+", lowered)
    is_question = "?" in text or (tokens[0] in QUESTION_WORDS if tokens else False)

    rule_scores = _rule_scores(lowered=lowered, is_question=is_question)
    rule_top_intent = max(rule_scores, key=rule_scores.get)
    rule_top_score = rule_scores[rule_top_intent]

    embedding_scores = _embedding_scores(lowered)
    embedding_top_intent = max(embedding_scores, key=embedding_scores.get)
    embedding_top_score = embedding_scores[embedding_top_intent]

    combined_scores = {
        intent: 0.6 * embedding_scores[intent] + 0.4 * rule_scores[intent]
        for intent in INTENT_PROTOTYPES
    }
    combined_top_intent = max(combined_scores, key=combined_scores.get)
    combined_top_score = combined_scores[combined_top_intent]

    layer = "rule"
    chosen_intent = rule_top_intent
    if rule_top_score >= 0.48:
        chosen_intent = rule_top_intent
        layer = "rule"
        confidence = rule_top_score
    elif embedding_top_score >= 0.16:
        chosen_intent = embedding_top_intent
        layer = "embedding"
        confidence = embedding_top_score
    elif combined_top_score >= 0.22:
        chosen_intent = combined_top_intent
        layer = "combined"
        confidence = combined_top_score
    else:
        chosen_intent = _llm_fallback_intent(text)
        layer = "llm_fallback"
        confidence = max(combined_top_score, 0.45)

    decision = _intent_to_route(
        top_intent=chosen_intent,
        confidence=confidence,
        scores={
            "rule_memory": round(rule_scores["MEMORY"], 4),
            "rule_retrieval": round(rule_scores["RETRIEVAL"], 4),
            "rule_response": round(rule_scores["RESPONSE"], 4),
            "embed_memory": round(embedding_scores["MEMORY"], 4),
            "embed_retrieval": round(embedding_scores["RETRIEVAL"], 4),
            "embed_response": round(embedding_scores["RESPONSE"], 4),
            "combined_memory": round(combined_scores["MEMORY"], 4),
            "combined_retrieval": round(combined_scores["RETRIEVAL"], 4),
            "combined_response": round(combined_scores["RESPONSE"], 4),
        },
        text=lowered,
        is_question=is_question,
        layer=layer,
        semantic_topic=semantic_topic,
    )
    return decision


def _rule_scores(*, lowered: str, is_question: bool) -> dict[str, float]:
    memory_score = _pattern_score(lowered, MEMORY_PATTERNS)
    retrieval_score = _pattern_score(lowered, RETRIEVAL_PATTERNS)
    action_score = _pattern_score(lowered, ACTION_PATTERNS)

    if is_question:
        retrieval_score += 0.16
        action_score += 0.10

    if re.search(r"\b(my|me|i)\b", lowered) and re.search(r"\b(weak|strong|goal|target|remember|said|skills?)\b", lowered):
        retrieval_score += 0.26

    if re.search(r"\b(i|my)\b", lowered) and re.search(r"\b(am|like|love|prefer|studying|learning|targeting|applying|struggling|weak|strong)\b", lowered):
        memory_score += 0.34

    if re.search(r"\b(i struggle with|i am weak in|i am weak at|i am not good at|my weak area)\b", lowered):
        memory_score += 0.34

    if re.search(r"\b(explain|teach|solve|help me|how to)\b", lowered):
        action_score += 0.28

    return {
        "MEMORY": min(memory_score, 1.0),
        "RETRIEVAL": min(retrieval_score, 1.0),
        "RESPONSE": min(action_score, 1.0),
    }


def _intent_to_route(
    *,
    top_intent: str,
    confidence: float,
    scores: dict[str, float],
    text: str,
    is_question: bool,
    layer: str,
    semantic_topic: str | None,
) -> PromptRouteDecision:
    memory_hint = scores.get("combined_memory", 0.0) >= 0.5
    retrieval_hint = scores.get("combined_retrieval", 0.0) >= 0.5 or is_question

    if top_intent == "MEMORY":
        intent = "respond_retrieve_and_update" if retrieval_hint else "respond_and_update_memory_async"
        needs_memory_update = True
        needs_retrieval = retrieval_hint
    elif top_intent == "RETRIEVAL":
        intent = "respond_retrieve_and_update" if memory_hint else "respond_and_retrieve"
        needs_memory_update = memory_hint and re.search(r"\b(i|my)\b", text) is not None
        needs_retrieval = True
    else:
        intent = "respond_only"
        needs_memory_update = memory_hint and re.search(r"\b(i|my)\b", text) is not None and not is_question
        needs_retrieval = False

    if re.search(r"\bwhat('?s| is) my\b", text) or re.search(r"\bmy weakness\b", text):
        needs_retrieval = True
        intent = "respond_retrieve_and_update" if needs_memory_update else "respond_and_retrieve"

    topic_help_pattern = re.search(
        r"\b(what should i practice|what should i study|how do i improve|what topics should i focus|what should i focus)\b",
        text,
    )
    reason = None
    if semantic_topic and topic_help_pattern:
        needs_retrieval = True
        intent = "respond_retrieve_and_update" if needs_memory_update else "respond_and_retrieve"
        reason = "semantic topic match"

    use_graph = needs_retrieval
    use_vector = needs_retrieval and confidence < 0.92

    return PromptRouteDecision(
        intent=intent,
        needs_retrieval=needs_retrieval,
        needs_memory_update=needs_memory_update,
        use_graph=use_graph,
        use_vector=use_vector,
        confidence=round(confidence, 4),
        scores=scores,
        layer=layer,
        semantic_topic=semantic_topic,
        reason=reason,
    )


def _pattern_score(text: str, patterns: tuple[str, ...]) -> float:
    score = 0.0
    for pattern in patterns:
        if re.search(pattern, text):
            score += 0.18
    return score


def _token_hash_embedding(text: str, dimensions: int = 96) -> list[float]:
    values = [0.0] * dimensions
    for token in re.findall(r"\w+", text.lower()):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = digest[0] % dimensions
        sign = 1.0 if digest[1] % 2 == 0 else -1.0
        values[index] += sign
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0:
        return values
    return [value / norm for value in values]


@lru_cache(maxsize=8)
def _prototype_embeddings() -> dict[str, tuple[float, ...]]:
    return {
        intent: tuple(_token_hash_embedding(text))
        for intent, text in INTENT_PROTOTYPES.items()
    }


@lru_cache(maxsize=512)
def _embedding_scores(text: str) -> dict[str, float]:
    query_vector = _token_hash_embedding(text)
    scores: dict[str, float] = {}
    for intent, prototype in _prototype_embeddings().items():
        scores[intent] = round(_cosine_similarity(query_vector, list(prototype)), 4)
    return scores


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return max(0.0, min(dot / (norm_a * norm_b), 1.0))


@lru_cache(maxsize=128)
def _llm_fallback_intent(text: str) -> str:
    if os.getenv("GRAPHMIND_ROUTER_LLM_FALLBACK", "true").strip().lower() not in {"1", "true", "yes"}:
        return "RESPONSE"
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "RESPONSE"
    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=os.getenv("GROQ_ROUTER_MODEL", "llama-3.1-8b-instant"),
            temperature=0.0,
            max_tokens=8,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Classify the user message into exactly one label: MEMORY, RETRIEVAL, RESPONSE. "
                        "MEMORY means durable user facts/preferences/weaknesses/goals. "
                        "RETRIEVAL means asking about stored memory or prior user facts. "
                        "RESPONSE means general explanation/help that does not require memory."
                    ),
                },
                {"role": "user", "content": text},
            ],
        )
        content = (response.choices[0].message.content or "").strip().upper()
        if "MEMORY" in content:
            return "MEMORY"
        if "RETRIEVAL" in content:
            return "RETRIEVAL"
    except Exception:
        pass
    return "RESPONSE"
