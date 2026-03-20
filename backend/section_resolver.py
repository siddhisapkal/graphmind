from __future__ import annotations

from dataclasses import dataclass
import re


SECTION_CONFIGS = {
    "weaknesses": {"tags": ("weakness", "practice_priority"), "families": ("capability",)},
    "skills": {"tags": ("strength", "revision"), "families": ("capability",)},
    "topics": {"tags": ("topic", "learning"), "families": ("learning", "structure")},
    "companies": {"tags": ("company", "target"), "families": ("goal",)},
    "goals": {"tags": ("goal", "target"), "families": ("goal",)},
}


@dataclass
class SectionPlan:
    sections: list[str]
    focus_entity: str | None = None
    reason: str | None = None

    def query_tags(self) -> list[str]:
        tags: set[str] = set()
        for section in self.sections:
            tags.update(SECTION_CONFIGS.get(section, {}).get("tags", ()))
        return sorted(tags)

    def query_families(self) -> list[str]:
        families: set[str] = set()
        for section in self.sections:
            families.update(SECTION_CONFIGS.get(section, {}).get("families", ()))
        return sorted(families)


def resolve_sections(*, message: str, route_intent: str, semantic_topic: str | None = None, target_entity: str | None = None) -> SectionPlan:
    lowered = " ".join((message or "").lower().split())
    sections: list[str] = []
    reason = None

    if re.search(r"\bweak(ness|nesses)?\b", lowered):
        sections.append("weaknesses")
    if re.search(r"\b(strength|strengths|good at|strong at)\b", lowered):
        sections.append("skills")
    if re.search(r"\b(what do i know|what have i studied|what should i practice|what should i study|improve|practice|study)\b", lowered):
        sections.append("topics")
    if re.search(r"\b(target|company|interview|prepare for|focusing on|applying to)\b", lowered):
        sections.extend(["companies", "goals"])

    if route_intent in {"respond_and_retrieve", "respond_retrieve_and_update"} and not sections:
        sections.append("topics")

    if semantic_topic and "topics" not in sections:
        sections.append("topics")
        reason = "semantic topic section"

    deduped: list[str] = []
    for section in sections:
        if section not in deduped:
            deduped.append(section)

    return SectionPlan(
        sections=deduped,
        focus_entity=semantic_topic or target_entity,
        reason=reason,
    )
