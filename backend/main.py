from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import BackgroundTasks, Cookie, FastAPI, HTTPException, Response, status
from fastapi.responses import HTMLResponse
from neo4j.exceptions import DatabaseError
from pydantic import BaseModel, Field

from .auth_store import AuthUser, authenticate_user, create_session, delete_session, get_user_by_session_token, register_user
from .chat_history_store import ensure_conversation, get_chat_history, list_conversations, save_message as save_chat_message
from .db import get_session
from .event_store import delete_user_events, log_promotions, log_raw_event, recent_raw_events
from .gemini_chat import analyze_strength_weakness_profile, classify_relation_with_llm, configured_models, extract_triple_candidates, generate_company_planner, generate_reply_bundle
from .profile_store import delete_user_profile, fetch_profile_summary, upsert_profile_observations
from .graph.service import graph_memory_service
from .prompt_router import route_prompt
from .relation_semantics import classify_relation_semantics, should_background_enrich, store_llm_relation_semantics
from .section_resolver import resolve_sections
from .topic_router import topic_semantic_router
from .vector_store import add_message, delete_user_messages, search as vector_search, warm_user_indexes
from .web_research import SearchPlan, build_search_plan, infer_memory_signals_from_plan, search_from_plan

app = FastAPI(title="GraphMind", version="0.4.0")
RETRIEVAL_EXECUTOR = ThreadPoolExecutor(max_workers=4)
MEMORY_EXECUTOR = ThreadPoolExecutor(max_workers=2)
SESSION_COOKIE_NAME = "graphmind_session"


class ChatRequest(BaseModel):
    user_id: str | None = None
    message: str = Field(min_length=1)
    conversation_id: str | None = None
    source: str = "chat"
    allow_web_search: bool = False


class SearchRequest(BaseModel):
    user_id: str | None = None
    query: str
    conversation_id: str | None = None
    k: int = 5


class MemorySignalInput(BaseModel):
    entity: str
    relation: str
    confidence: float = Field(ge=0.0, le=1.0)
    entity_type: str = "Entity"
    linked_to_action: bool = False
    raw_text: str | None = None


class MemoryIngestRequest(BaseModel):
    user_id: str | None = None
    source: str = "external_api"
    signals: list[MemorySignalInput] = Field(default_factory=list)


class AuthCredentials(BaseModel):
    username: str = Field(min_length=3)
    password: str = Field(min_length=8)


class CompanyPlannerRequest(BaseModel):
    user_id: str | None = None
    company: str = Field(min_length=2)
    days_left: int | None = Field(default=14)


@app.on_event("startup")
def _startup_create_constraints() -> None:
    with get_session() as session:
        _deduplicate_user_nodes(session)
        _drop_legacy_chat_graph_schema(session)
        _cleanup_legacy_chat_graph(session)
        _ensure_constraint(
            session,
            "CREATE CONSTRAINT user_id_unique IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE",
            "user_id_unique",
        )
        graph_memory_service.ensure_schema(session)
        topic_semantic_router.refresh_from_session(session)
    warm_user_indexes(limit_users=100)


def _ensure_constraint(session, query: str, name: str) -> None:
    try:
        session.run(query)
    except DatabaseError as exc:
        print(f"Skipping constraint {name}: {exc}")


def _drop_legacy_chat_graph_schema(session) -> None:
    constraints = session.run("SHOW CONSTRAINTS YIELD name, labelsOrTypes")
    for record in constraints:
        labels = set(record.get("labelsOrTypes") or [])
        if labels & {"Conversation", "Message"}:
            session.run(f"DROP CONSTRAINT `{record['name']}` IF EXISTS")
    indexes = session.run("SHOW INDEXES YIELD name, labelsOrTypes")
    for record in indexes:
        labels = set(record.get("labelsOrTypes") or [])
        if labels & {"Conversation", "Message"}:
            session.run(f"DROP INDEX `{record['name']}` IF EXISTS")


def _cleanup_legacy_chat_graph(session) -> None:
    session.run(
        """
        MATCH (n)
        WHERE n:Conversation OR n:Message
        DETACH DELETE n
        """
    )


def _ensure_user_node(session, *, user_id: str) -> None:
    session.run(
        """
        MERGE (u:User {id: $user_id})
        ON CREATE SET u.created_at = datetime()
        SET u.last_seen = datetime()
        """,
        user_id=user_id,
    )


def _deduplicate_user_nodes(session) -> None:
    rows = session.run(
        """
        MATCH (u:User)
        WHERE u.id IS NOT NULL
        WITH u.id AS id, collect(elementId(u)) AS ids
        WHERE size(ids) > 1
        RETURN id, ids
        """
    )
    for row in rows:
        ids = list(row["ids"])
        if len(ids) < 2:
            continue
        keep_id = ids[0]
        for duplicate_id in ids[1:]:
            _merge_duplicate_user_into(session, keep_id=keep_id, duplicate_id=duplicate_id)


def _merge_duplicate_user_into(session, *, keep_id: str, duplicate_id: str) -> None:
    session.run(
        """
        MATCH (keep:User), (dup:User)
        WHERE elementId(keep) = $keep_id AND elementId(dup) = $duplicate_id
        SET keep += properties(dup)
        """,
        keep_id=keep_id,
        duplicate_id=duplicate_id,
    )

    outgoing = session.run(
        """
        MATCH (dup:User)-[r]->(target)
        WHERE elementId(dup) = $duplicate_id
        RETURN type(r) AS rel_type, properties(r) AS rel_props, elementId(target) AS target_id
        """,
        duplicate_id=duplicate_id,
    )
    for record in outgoing:
        _merge_relationship(
            session,
            start_id=keep_id,
            end_id=record["target_id"],
            rel_type=record["rel_type"],
            rel_props=record["rel_props"] or {},
        )

    incoming = session.run(
        """
        MATCH (source)-[r]->(dup:User)
        WHERE elementId(dup) = $duplicate_id
        RETURN elementId(source) AS source_id, type(r) AS rel_type, properties(r) AS rel_props
        """,
        duplicate_id=duplicate_id,
    )
    for record in incoming:
        _merge_relationship(
            session,
            start_id=record["source_id"],
            end_id=keep_id,
            rel_type=record["rel_type"],
            rel_props=record["rel_props"] or {},
        )

    session.run(
        """
        MATCH (dup:User)
        WHERE elementId(dup) = $duplicate_id
        DETACH DELETE dup
        """,
        duplicate_id=duplicate_id,
    )


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


def _resolve_user_id(requested_user_id: str | None, current_user: AuthUser | None) -> str:
    requested = (requested_user_id or "").strip()
    if current_user is not None:
        if requested and requested != current_user.user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Authenticated users can only access their own memory space.",
            )
        return current_user.user_id
    if not requested:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Login required or provide a user_id explicitly.",
        )
    return requested


def _set_session_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=token,
        httponly=True,
        samesite="lax",
        max_age=60 * 60 * 24 * 30,
    )


def _auth_payload(user: AuthUser) -> dict[str, object]:
    return {
        "authenticated": True,
        "user": {
            "user_id": user.user_id,
            "username": user.username,
            "created_at": user.created_at,
        },
    }


def _message_requests_web(message: str) -> bool:
    lowered = " ".join((message or "").lower().split())
    web_phrases = (
        "from web",
        "from the web",
        "from internet",
        "from the internet",
        "search web",
        "search the web",
        "search internet",
        "look it up",
        "look this up",
        "latest",
        "current",
        "today",
        "recent",
    )
    return any(phrase in lowered for phrase in web_phrases)




@app.get("/")
def root() -> dict[str, str]:
    return {"message": "GraphMind backend is running"}


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "ephemeral_backend": graph_memory_service.ephemeral_backend,
        "llm": configured_models(),
    }


@app.post("/auth/register")
def auth_register(req: AuthCredentials, response: Response) -> dict[str, object]:
    try:
        user = register_user(username=req.username, password=req.password)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    with get_session() as session:
        _ensure_user_node(session, user_id=user.user_id)
    token, _expires_at = create_session(user_id=user.user_id)
    _set_session_cookie(response, token)
    return _auth_payload(user)


@app.post("/auth/login")
def auth_login(req: AuthCredentials, response: Response) -> dict[str, object]:
    user = authenticate_user(username=req.username, password=req.password)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password.",
        )
    with get_session() as session:
        _ensure_user_node(session, user_id=user.user_id)
    token, _expires_at = create_session(user_id=user.user_id)
    _set_session_cookie(response, token)
    return _auth_payload(user)


@app.post("/auth/logout")
def auth_logout(response: Response, graphmind_session: str | None = Cookie(default=None)) -> dict[str, object]:
    delete_session(graphmind_session)
    response.delete_cookie(SESSION_COOKIE_NAME)
    return {"authenticated": False}


@app.get("/auth/me")
def auth_me(graphmind_session: str | None = Cookie(default=None, alias=SESSION_COOKIE_NAME)) -> dict[str, object]:
    user = get_user_by_session_token(graphmind_session)
    if user is None:
        return {"authenticated": False, "user": None}
    return _auth_payload(user)


@app.get("/chat-ui", response_class=HTMLResponse)
def chat_ui() -> str:
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>GraphMind Memory Console</title>
  <style>
    :root {
      color-scheme: dark;
      --bg:#06131f;
      --bg2:#081a29;
      --panel:#0d1826;
      --panel2:#132235;
      --panel3:#0a1220;
      --border:rgba(148,163,184,.16);
      --border-strong:rgba(125,211,252,.32);
      --text:#ecf7ff;
      --muted:#8fa8bc;
      --accent:#7dd3fc;
      --accent-strong:#38bdf8;
      --accent-soft:rgba(125,211,252,.12);
      --good:#34d399;
      --warn:#fbbf24;
      --shadow:0 30px 80px rgba(1,8,20,.46);
    }
    *{box-sizing:border-box}
    body{
      margin:0;
      font-family:"Segoe UI Variable Display","Segoe UI",Tahoma,sans-serif;
      background:
        radial-gradient(circle at top left, rgba(56,189,248,.12), transparent 28%),
        radial-gradient(circle at top right, rgba(52,211,153,.08), transparent 22%),
        linear-gradient(180deg, var(--bg2), var(--bg));
      color:var(--text);
      height:100vh;
      overflow:hidden;
      padding:16px;
    }
    .app{
      width:100%;
      max-width:1540px;
      height:calc(100vh - 32px);
      margin:0 auto;
      border:1px solid var(--border);
      border-radius:28px;
      background:
        linear-gradient(180deg, rgba(11,26,40,.92), rgba(6,15,24,.96)),
        radial-gradient(circle at top, rgba(125,211,252,.08), transparent 34%);
      box-shadow:var(--shadow);
      padding:18px;
      display:flex;
      flex-direction:column;
      gap:16px;
      backdrop-filter:blur(10px);
      overflow:hidden;
    }
    .top{
      display:flex;
      flex-direction:column;
      gap:14px;
      align-items:stretch;
      padding:4px 2px 2px;
    }
    .title{
      display:flex;
      align-items:flex-end;
      justify-content:space-between;
      gap:18px;
      padding:4px 4px 0;
    }
    .title b{
      font-size:30px;
      letter-spacing:.01em;
      font-weight:750;
    }
    .title span{
      font-size:13px;
      color:var(--muted);
      max-width:720px;
      line-height:1.5;
      text-align:right;
    }
    .controls{
      display:flex;
      gap:10px;
      align-items:center;
      justify-content:flex-start;
      flex-wrap:wrap;
      padding:12px 14px;
      border:1px solid rgba(148,163,184,.12);
      border-radius:24px;
      background:linear-gradient(180deg, rgba(19,34,53,.9), rgba(10,18,32,.94));
    }
    .authBar{
      display:flex;
      gap:8px;
      align-items:center;
      flex-wrap:wrap;
      margin-right:8px;
      padding-right:8px;
      border-right:1px solid rgba(148,163,184,.12);
    }
    .authBar.signedIn .field,
    .authBar.signedIn #loginBtn,
    .authBar.signedIn #registerBtn{
      display:none;
    }
    .authBar.signedOut #logoutBtn{
      display:none;
    }
    .pill{
      border:1px solid rgba(148,163,184,.18);
      background:rgba(8,18,30,.78);
      color:var(--muted);
      padding:8px 11px;
      border-radius:999px;
      font-size:12px;
    }
    .btn{
      border:1px solid rgba(148,163,184,.16);
      background:linear-gradient(180deg, rgba(24,40,59,.96), rgba(10,18,30,.94));
      color:var(--text);
      padding:9px 12px;
      border-radius:14px;
      font-size:12px;
      font-weight:600;
      cursor:pointer;
      transition:transform .14s ease, border-color .14s ease, background .14s ease;
    }
    .btn:hover{
      border-color:var(--border-strong);
      transform:translateY(-1px);
    }
    .field{
      border:1px solid rgba(148,163,184,.16);
      background:rgba(7,16,26,.88);
      color:var(--text);
      padding:9px 12px;
      border-radius:14px;
      font-size:12px;
    }
    .field::placeholder{color:rgba(148,163,184,.9)}
    .authStatus{font-size:12px; color:var(--muted);}
    .main{
      display:grid;
      grid-template-columns:260px minmax(0,1.75fr) minmax(350px,.92fr);
      gap:16px;
      flex:1;
      min-height:0;
      overflow:hidden;
    }
    .sidebar,
    .panel,
    .composer,
    .msgs{
      box-shadow:inset 0 1px 0 rgba(255,255,255,.03);
    }
    .sidebar{
      border:1px solid var(--border);
      border-radius:24px;
      background:linear-gradient(180deg, rgba(18,31,47,.96), rgba(9,17,29,.96));
      padding:14px;
      min-height:0;
      display:flex;
      flex-direction:column;
      gap:12px;
    }
    .sidebarList{display:flex; flex-direction:column; gap:10px; overflow:auto; min-height:0; padding-right:2px;}
    .convoBtn{
      width:100%;
      text-align:left;
      border:1px solid rgba(148,163,184,.12);
      background:linear-gradient(180deg, rgba(19,33,50,.92), rgba(10,18,29,.92));
      color:var(--text);
      border-radius:16px;
      padding:12px;
      cursor:pointer;
      transition:transform .14s ease, border-color .14s ease, background .14s ease;
    }
    .convoBtn:hover{
      border-color:rgba(125,211,252,.44);
      transform:translateY(-1px);
    }
    .convoBtn.active{
      border-color:rgba(125,211,252,.68);
      background:
        linear-gradient(180deg, rgba(10,56,83,.86), rgba(10,24,38,.96)),
        radial-gradient(circle at left top, rgba(125,211,252,.16), transparent 50%);
    }
    .convoTitle{font-size:12px; color:var(--text); margin-bottom:6px; line-height:1.4; font-weight:600;}
    .convoMeta{font-size:11px; color:var(--muted);}
    .chatCol{
      display:grid;
      grid-template-rows:minmax(0,1fr) auto;
      gap:14px;
      min-height:0;
      overflow:hidden;
    }
    .msgs{
      border:1px solid rgba(125,211,252,.18);
      background:
        radial-gradient(circle at top, rgba(125,211,252,.08), transparent 30%),
        linear-gradient(180deg, rgba(10,20,31,.98), rgba(6,13,22,.98));
      border-radius:28px;
      padding:22px 18px;
      overflow:auto;
      display:flex;
      flex-direction:column;
      gap:14px;
      scroll-behavior:smooth;
    }
    .msg{
      max-width:min(84%, 760px);
      padding:14px 16px;
      border-radius:22px;
      font-size:15px;
      line-height:1.65;
      white-space:pre-wrap;
    }
    .user{
      margin-left:auto;
      background:linear-gradient(135deg, #34d399, #14b8a6);
      color:#03211e;
      box-shadow:0 12px 28px rgba(20,184,166,.18);
    }
    .bot{
      background:linear-gradient(180deg, rgba(18,31,47,.98), rgba(10,18,29,.98));
      border:1px solid rgba(148,163,184,.16);
    }
    .choiceRow{display:flex; gap:8px; margin-top:10px; flex-wrap:wrap;}
    .choiceBtn{
      border:1px solid rgba(148,163,184,.18);
      background:rgba(4,13,22,.92);
      color:var(--text);
      padding:7px 11px;
      border-radius:999px;
      font-size:12px;
      font-weight:600;
      cursor:pointer;
    }
    .choiceBtn:hover{border-color:rgba(125,211,252,.55);}
    .composer{
      border:1px solid var(--border);
      background:linear-gradient(180deg, rgba(18,31,47,.94), rgba(8,16,27,.96));
      border-radius:24px;
      padding:12px;
      display:flex;
      gap:10px;
      align-items:flex-end;
    }
    textarea{
      flex:1;
      resize:none;
      border:1px solid rgba(148,163,184,.16);
      background:rgba(5,14,23,.92);
      color:var(--text);
      border-radius:16px;
      padding:12px 14px;
      font:inherit;
      min-height:56px;
      max-height:160px;
      outline:none;
      line-height:1.45;
    }
    textarea::placeholder{color:rgba(148,163,184,.9)}
    .send{
      border:none;
      background:linear-gradient(135deg, #7dd3fc, #38bdf8 52%, #0ea5e9);
      color:#032033;
      padding:12px 18px;
      border-radius:999px;
      cursor:pointer;
      font-weight:700;
      min-width:92px;
      box-shadow:0 14px 30px rgba(56,189,248,.22);
    }
    .send:disabled{opacity:.55; cursor:default}
    .meta{display:flex; justify-content:space-between; gap:10px; color:var(--muted); font-size:12px; padding:0 4px;}
    code{background:rgba(148,163,184,.12); padding:2px 6px; border-radius:8px;}
    .memory{
      display:flex;
      flex-direction:column;
      gap:14px;
      min-height:0;
      overflow:auto;
      padding-right:4px;
      overscroll-behavior:contain;
    }
    .memory > *{
      flex:0 0 auto;
      min-height:0;
    }
    .memoryGrid{
      display:grid;
      grid-template-columns:1fr;
      gap:14px;
      min-height:auto;
      overflow:visible;
      flex:0 0 auto;
    }
    .panel{
      border:1px solid var(--border);
      border-radius:24px;
      background:linear-gradient(180deg, rgba(18,31,47,.96), rgba(8,16,27,.96));
      padding:14px;
      min-height:0;
      display:flex;
      flex-direction:column;
      gap:12px;
    }
    .panel h3{margin:0; font-size:14px; letter-spacing:.01em;}
    .panelHead{display:flex; align-items:center; justify-content:space-between; gap:8px;}
    .tiny{font-size:11px; color:var(--muted);}
    .topMeta{display:flex; align-items:center; gap:8px; flex-wrap:wrap;}
    .statusChip{
      display:inline-flex;
      align-items:center;
      gap:8px;
      padding:8px 12px;
      border-radius:999px;
      border:1px solid rgba(148,163,184,.14);
      background:rgba(6,14,24,.72);
      color:var(--muted);
      font-size:12px;
    }
    .statusChip strong{color:var(--text); font-weight:600;}
    .statusDot{
      width:8px;
      height:8px;
      border-radius:999px;
      background:#f59e0b;
      box-shadow:0 0 0 6px rgba(245,158,11,.12);
      flex:0 0 auto;
    }
    .statusChip.live .statusDot{
      background:#34d399;
      box-shadow:0 0 0 6px rgba(52,211,153,.12);
    }
    .hiddenMeta{display:none;}
    .list{display:flex; flex-direction:column; gap:10px; overflow:auto; min-height:0;}
    .item{
      padding:11px 12px;
      border-radius:16px;
      background:linear-gradient(180deg, rgba(18,31,47,.92), rgba(10,18,29,.92));
      border:1px solid rgba(148,163,184,.12);
    }
    .item b{display:block; margin-bottom:5px; font-size:13px;}
    .row{display:flex; gap:8px; flex-wrap:wrap; font-size:12px; color:var(--muted);}
    .tag{padding:3px 8px; border-radius:999px; background:var(--accent-soft); border:1px solid rgba(125,211,252,.24); font-size:11px;}
    .ok{color:var(--good)}
    .warn{color:var(--warn)}
    .summary{display:grid; gap:10px; min-height:28px;}
    .summaryHero{
      display:flex;
      align-items:flex-start;
      justify-content:space-between;
      gap:10px;
      padding:12px 14px;
      border-radius:18px;
      background:linear-gradient(135deg, rgba(125,211,252,.11), rgba(16,185,129,.07));
      border:1px solid rgba(125,211,252,.16);
    }
    .summaryHero b{display:block; font-size:14px; margin-bottom:4px;}
    .summaryHero span{font-size:12px; color:var(--muted); line-height:1.55;}
    .summaryStats{display:flex; gap:8px; flex-wrap:wrap;}
    .summaryStats .tag{background:rgba(34,197,94,.1); border-color:rgba(34,197,94,.25);}
    .mutedTag{background:rgba(148,163,184,.08) !important; border-color:rgba(148,163,184,.16) !important; color:var(--muted);}
    .evidenceList{
      display:flex;
      flex-direction:column;
      gap:10px;
      overflow:visible;
      min-height:auto;
      flex:0 0 auto;
      padding-right:2px;
    }
    .graphCanvas{width:100%; min-height:250px; border-radius:20px; border:1px solid rgba(125,211,252,.14); background:
      radial-gradient(circle at top, rgba(125,211,252,.1), transparent 42%),
      linear-gradient(180deg, rgba(11,24,37,.96), rgba(6,14,24,.94));
      overflow:hidden;
      position:relative;
    }
    .graphCanvas svg{width:100%; height:100%; display:block; min-height:250px;}
    .graphActions{display:flex; align-items:center; gap:8px;}
    .iconBtn{
      border:1px solid rgba(148,163,184,.16);
      background:linear-gradient(180deg, rgba(24,40,59,.96), rgba(10,18,30,.94));
      color:var(--text);
      padding:7px 10px;
      border-radius:12px;
      font-size:11px;
      font-weight:700;
      cursor:pointer;
      transition:transform .14s ease, border-color .14s ease;
    }
    .iconBtn:hover{
      border-color:var(--border-strong);
      transform:translateY(-1px);
    }
    .graphLegend{display:flex; gap:10px; flex-wrap:wrap; padding-bottom:2px;}
    .legendDot{display:inline-flex; align-items:center; gap:6px; font-size:11px; color:var(--muted);}
    .legendDot i{display:inline-block; width:10px; height:10px; border-radius:999px;}
    .graphEmpty{display:flex; align-items:center; justify-content:center; min-height:250px; color:var(--muted); font-size:12px;}
    .graphHint{font-size:11px; color:var(--muted); padding-top:2px;}
    .graphModal{
      position:fixed;
      inset:0;
      display:none;
      align-items:center;
      justify-content:center;
      padding:20px;
      background:rgba(2,8,16,.76);
      backdrop-filter:blur(8px);
      z-index:50;
    }
    .graphModal.open{display:flex;}
    .graphModalCard{
      width:min(1100px, calc(100vw - 40px));
      height:min(780px, calc(100vh - 40px));
      border:1px solid rgba(148,163,184,.18);
      border-radius:28px;
      background:linear-gradient(180deg, rgba(10,21,33,.98), rgba(6,14,24,.98));
      box-shadow:0 30px 80px rgba(1,8,20,.56);
      display:grid;
      grid-template-rows:auto auto minmax(0,1fr);
      gap:12px;
      padding:18px;
    }
    .graphModalHead{
      display:flex;
      align-items:center;
      justify-content:space-between;
      gap:12px;
    }
    .graphModalTitle{display:flex; flex-direction:column; gap:4px;}
    .graphModalTitle b{font-size:18px;}
    .graphModalCanvas{
      width:100%;
      height:100%;
      min-height:0;
      border-radius:22px;
      border:1px solid rgba(125,211,252,.14);
      background:
        radial-gradient(circle at top, rgba(125,211,252,.1), transparent 42%),
        linear-gradient(180deg, rgba(11,24,37,.96), rgba(6,14,24,.94));
      overflow:hidden;
      position:relative;
    }
    .graphModalCanvas svg{width:100%; height:100%; display:block;}
    .plannerModal{
      position:fixed;
      inset:0;
      display:none;
      align-items:center;
      justify-content:center;
      padding:20px;
      background:rgba(2,8,16,.76);
      backdrop-filter:blur(8px);
      z-index:55;
    }
    .plannerModal.open{display:flex;}
    .plannerModalCard{
      width:min(920px, calc(100vw - 40px));
      height:min(780px, calc(100vh - 40px));
      border:1px solid rgba(148,163,184,.18);
      border-radius:28px;
      background:linear-gradient(180deg, rgba(10,21,33,.99), rgba(6,14,24,.99));
      box-shadow:0 30px 80px rgba(1,8,20,.56);
      display:grid;
      grid-template-rows:auto auto minmax(0,1fr);
      gap:14px;
      padding:18px;
    }
    .plannerBar{display:flex; gap:10px; flex-wrap:wrap; align-items:center;}
    .plannerInput{
      border:1px solid rgba(148,163,184,.16);
      background:rgba(5,14,23,.92);
      color:var(--text);
      border-radius:14px;
      padding:10px 12px;
      font:inherit;
      min-width:150px;
    }
    .plannerBody{
      display:grid;
      grid-template-columns:1.05fr .95fr;
      gap:14px;
      min-height:0;
    }
    .plannerPanel{
      min-height:0;
      overflow:auto;
      border:1px solid rgba(148,163,184,.12);
      border-radius:20px;
      background:linear-gradient(180deg, rgba(16,28,43,.94), rgba(8,16,27,.96));
      padding:14px;
      display:flex;
      flex-direction:column;
      gap:10px;
    }
    .plannerSectionTitle{font-size:12px; color:var(--muted); text-transform:uppercase; letter-spacing:.08em;}
    .plannerBlock{
      padding:12px;
      border-radius:16px;
      border:1px solid rgba(148,163,184,.12);
      background:rgba(8,16,27,.78);
    }
    .plannerBlock b{display:block; margin-bottom:5px; font-size:13px;}
    .plannerBlock ul{margin:6px 0 0 18px; padding:0;}
    .plannerBlock li{margin:4px 0; color:var(--muted); font-size:12px; line-height:1.55;}
    .plannerPlaceholder{color:var(--muted); font-size:12px; padding:18px 6px;}
    .plannerStartRow{display:flex; gap:10px; align-items:center; justify-content:space-between; flex-wrap:wrap;}
    .plannerProgressLine{display:flex; gap:8px; flex-wrap:wrap; align-items:center;}
    .plannerDayCard{
      padding:14px;
      border-radius:18px;
      border:1px solid rgba(125,211,252,.18);
      background:linear-gradient(135deg, rgba(125,211,252,.09), rgba(16,185,129,.07));
    }
    .plannerDayCard b{display:block; margin-bottom:6px; font-size:15px;}
    .plannerSummaryGrid{display:grid; gap:10px;}
    .plannerActiveBtn{display:none;}
    .plannerActiveBtn.show{display:inline-flex;}
    @media (max-width: 1240px){
      body{overflow:auto; height:auto;}
      .app{height:auto; min-height:calc(100vh - 32px);}
      .main{grid-template-columns:220px minmax(0,1.35fr) minmax(290px,.92fr);}
      .title{flex-direction:column; align-items:flex-start;}
      .title span{text-align:left;}
      .authBar{border-right:none; padding-right:0; margin-right:0;}
      .topMeta{width:100%;}
    }
    @media (max-width: 980px){
      body{padding:12px; overflow:auto; height:auto;}
      .app{height:auto; min-height:calc(100vh - 24px);}
      .main{grid-template-columns:1fr;}
      .chatCol{order:1;}
      .sidebar{order:2; min-height:220px;}
      .memory{order:3; min-height:420px;}
      .sidebar{min-height:220px;}
      .title b{font-size:24px;}
      .app{padding:14px;}
      .msgs{min-height:52vh;}
    }
    @media (max-width: 720px){
      .controls{padding:12px;}
      .field{width:100% !important;}
      .authBar{width:100%;}
      .msg{max-width:92%;}
      .summaryHero{flex-direction:column;}
      .plannerBody{grid-template-columns:1fr;}
    }
  </style>
</head>
<body>
  <div class="app">
    <div class="top">
      <div class="title">
        <b>GraphMind</b>
      </div>
      <div class="controls">
        <div class="authBar signedOut" id="authBar">
          <input type="text" id="authUsername" class="field" placeholder="username" style="width:140px;">
          <input type="password" id="authPassword" class="field" placeholder="password" style="width:140px;">
          <button class="btn" id="loginBtn">Login</button>
          <button class="btn" id="registerBtn">Register</button>
          <button class="btn" id="logoutBtn">Logout</button>
        </div>
        <div class="topMeta">
          <span class="statusChip" id="workspaceStatus"><i class="statusDot"></i><strong id="uname"></strong><span id="authStatus">Checking session...</span></span>
          <span class="statusChip"><strong>Model</strong><span id="llmDisplay">unknown</span></span>
        </div>
        <span class="hiddenMeta pill">User: <code id="uid"></code></span>
        <span class="hiddenMeta pill">Conversation: <code id="cid"></code></span>
        <span class="hiddenMeta pill">Ephemeral: <code id="backendLabel"></code></span>
        <span class="hiddenMeta pill">LLM: <code id="llmLabel">unknown</code></span>
        <button class="btn" id="newChat">New chat</button>
        <button class="btn" id="openPlannerBtn">Company planner</button>
        <button class="btn plannerActiveBtn" id="activePlannerBtn">Open planner</button>
        <button class="btn" id="refreshMemory">Refresh memory</button>
        <button class="btn" id="resetMemory">Reset user memory</button>
      </div>
    </div>

    <div class="main">
      <div class="sidebar">
        <div class="panelHead">
          <h3>Conversations</h3>
          <span class="tiny">SQLite history</span>
        </div>
        <div class="sidebarList" id="conversationList"></div>
      </div>
      <div class="chatCol">
        <div class="msgs" id="msgs">
          <div class="msg bot">Log in or register to start building a private memory space. Each account keeps its own graph, events, and vector history.</div>
        </div>

        <div>
          <div class="composer">
            <textarea id="input" placeholder="Type here... (Enter to send, Shift+Enter for newline)"></textarea>
            <button class="send" id="send">Send</button>
          </div>
          <div class="meta">
            <span id="status">Ready</span>
            <span id="lat">Idle</span>
          </div>
        </div>
      </div>
      <div class="memory">
        <div class="panel">
          <div class="panelHead">
            <h3>Last Promotion</h3>
            <span class="tiny" id="memoryUpdated">Waiting for activity</span>
          </div>
          <div class="summary" id="summary"></div>
        </div>
        <div class="panel">
          <div class="panelHead">
            <h3>Evidence Paths</h3>
            <span class="tiny">Grounding used for answers</span>
          </div>
          <div class="evidenceList" id="evidenceList"></div>
        </div>
        <div class="memoryGrid">
          <div class="panel">
            <div class="panelHead">
              <h3>Memory Graph</h3>
              <div class="graphActions">
                <span class="tiny">Relationship view</span>
                <button class="iconBtn" id="maximizeGraph" type="button">Maximize</button>
              </div>
            </div>
            <div class="graphHint">Core entities only. Lower-signal metadata stays out of the graph view.</div>
            <div class="graphLegend">
              <span class="legendDot"><i style="background:#f59e0b"></i>User</span>
              <span class="legendDot"><i style="background:#22c55e"></i>Skill</span>
              <span class="legendDot"><i style="background:#38bdf8"></i>Topic</span>
              <span class="legendDot"><i style="background:#a78bfa"></i>Company</span>
              <span class="legendDot"><i style="background:#fb7185"></i>Goal</span>
            </div>
            <div class="graphCanvas" id="graphCanvas">
              <div class="graphEmpty">No graph nodes yet.</div>
            </div>
          </div>
          <div class="panel">
            <div class="panelHead">
              <h3>Ephemeral Memory</h3>
              <span class="tiny">Short-term signals</span>
            </div>
            <div class="list" id="ephemeralList"></div>
          </div>
          <div class="panel">
            <div class="panelHead">
              <h3>Graph Memory</h3>
              <span class="tiny">Promoted long-term knowledge</span>
            </div>
            <div class="list" id="graphList"></div>
          </div>
          <div class="panel">
            <div class="panelHead">
              <h3>Cached Profile</h3>
              <span class="tiny">Strengths, weaknesses, improving</span>
            </div>
            <div class="list" id="profileList"></div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <div class="graphModal" id="graphModal" aria-hidden="true">
    <div class="graphModalCard" role="dialog" aria-modal="true" aria-labelledby="graphModalTitle">
      <div class="graphModalHead">
        <div class="graphModalTitle">
          <b id="graphModalTitle">Memory Graph</b>
          <span class="tiny">Expanded relationship view of your strongest saved memory.</span>
        </div>
        <button class="iconBtn" id="closeGraphModal" type="button">Close</button>
      </div>
      <div class="graphLegend">
        <span class="legendDot"><i style="background:#f59e0b"></i>User</span>
        <span class="legendDot"><i style="background:#38bdf8"></i>Topic</span>
        <span class="legendDot"><i style="background:#22c55e"></i>Skill</span>
        <span class="legendDot"><i style="background:#fb7185"></i>Goal</span>
        <span class="legendDot"><i style="background:#a78bfa"></i>Company</span>
      </div>
      <div class="graphModalCanvas" id="graphModalCanvas">
        <div class="graphEmpty">No graph nodes yet.</div>
      </div>
    </div>
  </div>

  <div class="plannerModal" id="plannerModal" aria-hidden="true">
    <div class="plannerModalCard" role="dialog" aria-modal="true" aria-labelledby="plannerModalTitle">
      <div class="graphModalHead">
        <div class="graphModalTitle">
          <b id="plannerModalTitle">Company Planner</b>
          <span class="tiny">Generate a company-wise placement plan using Gemini and live web research.</span>
        </div>
        <button class="iconBtn" id="closePlannerModal" type="button">Close</button>
      </div>
      <div class="plannerBar">
        <input class="plannerInput" id="plannerCompanyInput" type="text" placeholder="Company name">
        <input class="plannerInput" id="plannerDaysInput" type="number" min="1" max="60" value="14" placeholder="Days">
        <button class="btn" id="generatePlannerBtn" type="button">Generate plan</button>
        <span class="tiny" id="plannerState">Enter a company and number of days.</span>
      </div>
      <div class="plannerBody">
        <div class="plannerPanel" id="plannerMainPanel">
          <div class="plannerPlaceholder">Planner overview, stages, and day-by-day schedule will appear here.</div>
        </div>
        <div class="plannerPanel" id="plannerSourcePanel">
          <div class="plannerPlaceholder">Web findings and likely previous-question patterns will appear here.</div>
        </div>
      </div>
    </div>
  </div>

<script>
(() => {
  const msgs = document.getElementById('msgs');
  const input = document.getElementById('input');
  const sendBtn = document.getElementById('send');
  const statusEl = document.getElementById('status');
  const latEl = document.getElementById('lat');
  const uidEl = document.getElementById('uid');
  const cidEl = document.getElementById('cid');
  const backendLabel = document.getElementById('backendLabel');
  const llmLabel = document.getElementById('llmLabel');
  const llmDisplay = document.getElementById('llmDisplay');
  const workspaceStatus = document.getElementById('workspaceStatus');
  const graphCanvas = document.getElementById('graphCanvas');
  const graphModal = document.getElementById('graphModal');
  const graphModalCanvas = document.getElementById('graphModalCanvas');
  const maximizeGraphBtn = document.getElementById('maximizeGraph');
  const closeGraphModalBtn = document.getElementById('closeGraphModal');
  const openPlannerBtn = document.getElementById('openPlannerBtn');
  const activePlannerBtn = document.getElementById('activePlannerBtn');
  const plannerModal = document.getElementById('plannerModal');
  const closePlannerModalBtn = document.getElementById('closePlannerModal');
  const plannerCompanyInput = document.getElementById('plannerCompanyInput');
  const plannerDaysInput = document.getElementById('plannerDaysInput');
  const generatePlannerBtn = document.getElementById('generatePlannerBtn');
  const plannerMainPanel = document.getElementById('plannerMainPanel');
  const plannerSourcePanel = document.getElementById('plannerSourcePanel');
  const plannerState = document.getElementById('plannerState');
  const ephemeralList = document.getElementById('ephemeralList');
  const graphList = document.getElementById('graphList');
  const profileList = document.getElementById('profileList');
  const evidenceList = document.getElementById('evidenceList');
  const summary = document.getElementById('summary');
  const memoryUpdated = document.getElementById('memoryUpdated');
  const unameEl = document.getElementById('uname');
  const authStatusEl = document.getElementById('authStatus');
  const authBar = document.getElementById('authBar');
  const authUsername = document.getElementById('authUsername');
  const authPassword = document.getElementById('authPassword');
  const loginBtn = document.getElementById('loginBtn');
  const registerBtn = document.getElementById('registerBtn');
  const logoutBtn = document.getElementById('logoutBtn');
  const conversationList = document.getElementById('conversationList');

  const rand = () => Math.random().toString(36).slice(2,10);

  let userId = '';
  let username = '';
  let latestGraphNodes = [];
  let latestGraphEdges = [];
  let latestPlannerPayload = null;
  let activePlannerSession = null;
  let convoId = localStorage.getItem('graphmind_convo_id');
  if (!convoId) { convoId = "convo-" + rand(); localStorage.setItem('graphmind_convo_id', convoId); }

  uidEl.textContent = 'guest';
  unameEl.textContent = 'guest';
  cidEl.textContent = convoId;

  function setAuthState(payload) {
    const user = payload?.user || null;
    userId = user?.user_id || '';
    username = user?.username || '';
    uidEl.textContent = userId || 'guest';
    unameEl.textContent = username || 'Guest';
    authStatusEl.textContent = userId ? 'Private workspace active' : 'Sign in to use saved memory';
    workspaceStatus.classList.toggle('live', !!userId);
    authBar.classList.toggle('signedIn', !!userId);
    authBar.classList.toggle('signedOut', !userId);
    input.disabled = !userId;
    sendBtn.disabled = !userId;
    logoutBtn.disabled = !userId;
    document.getElementById('newChat').disabled = !userId;
    openPlannerBtn.disabled = !userId;
    activePlannerBtn.disabled = !userId;
    document.getElementById('refreshMemory').disabled = !userId;
    document.getElementById('resetMemory').disabled = !userId;
    if (!userId) {
      activePlannerSession = null;
      latestPlannerPayload = null;
      renderActivePlannerButton();
    } else {
      activePlannerSession = loadPlannerSession();
      renderActivePlannerButton();
    }
    if (!userId) {
      conversationList.innerHTML = '<div class="item tiny">Sign in to see saved chats.</div>';
      renderEmpty(ephemeralList, 'Sign in to load this user memory space.');
      renderEmpty(graphList, 'Sign in to load this user memory space.');
      renderEmpty(profileList, 'Sign in to load your cached profile.');
      renderEmpty(evidenceList, 'Sign in to see evidence paths.');
      graphCanvas.innerHTML = '<div class="graphEmpty">Sign in to view your graph.</div>';
      graphModalCanvas.innerHTML = '<div class="graphEmpty">Sign in to view your graph.</div>';
    }
  }

  async function hydrateSession() {
    try {
      const res = await fetch('/auth/me');
      const data = await res.json();
      setAuthState(data);
      if (data?.authenticated) {
        await loadConversationList();
        await loadChatHistory();
        activePlannerSession = loadPlannerSession();
        renderActivePlannerButton();
        await refreshMemory();
        input.focus();
      }
    } catch (e) {
      console.error(e);
      authStatusEl.textContent = 'Unable to restore your session.';
    }
  }

  async function requestAuth(path, payload) {
    const res = await fetch(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    return { res, data };
  }

  async function submitAuth(path) {
    const usernameValue = authUsername.value.trim();
    const passwordValue = authPassword.value;
    if (!usernameValue || !passwordValue) {
      authStatusEl.textContent = 'Enter a username and password first.';
      return;
    }
    authStatusEl.textContent = path === '/auth/register' ? 'Creating account...' : 'Signing in...';
    try {
      let { res, data } = await requestAuth(path, { username: usernameValue, password: passwordValue });
      if (!res.ok && path === '/auth/register' && String(data?.detail || '').includes('already exists')) {
        authStatusEl.textContent = 'Account exists. Trying sign in...';
        ({ res, data } = await requestAuth('/auth/login', { username: usernameValue, password: passwordValue }));
      }
      if (!res.ok) {
        const detail = data?.detail || 'Authentication failed.';
        throw new Error(detail);
      }
      setAuthState(data);
      if (!convoId) {
        convoId = 'convo-' + rand();
        localStorage.setItem('graphmind_convo_id', convoId);
      }
      cidEl.textContent = convoId;
      if (path === '/auth/register' && res.url.includes('/auth/register')) {
        convoId = 'convo-' + rand();
        localStorage.setItem('graphmind_convo_id', convoId);
        cidEl.textContent = convoId;
        msgs.innerHTML = '<div class="msg bot">Account created. Your private memory space starts now.</div>';
      } else {
        await loadConversationList();
        await loadChatHistory();
      }
      authStatusEl.textContent = 'Signed in as ' + data.user.username;
      authPassword.value = '';
      activePlannerSession = loadPlannerSession();
      renderActivePlannerButton();
      await refreshMemory();
      await loadConversationList();
      input.focus();
    } catch (e) {
      console.error(e);
      authStatusEl.textContent = e.message || 'Authentication failed.';
      msgs.innerHTML = '<div class="msg bot">' + (e.message || 'Authentication failed.') + '</div>';
    }
  }

  function renderEmpty(target, text) {
    target.innerHTML = '<div class="item tiny">' + text + '</div>';
  }

  function plannerStorageKey() {
    return 'graphmind_planner_' + (userId || 'guest');
  }

  function loadPlannerSession() {
    if (!userId) return null;
    try {
      const raw = localStorage.getItem(plannerStorageKey());
      if (!raw) return null;
      const parsed = JSON.parse(raw);
      return parsed && typeof parsed === 'object' ? parsed : null;
    } catch (e) {
      console.error(e);
      return null;
    }
  }

  function savePlannerSession(session) {
    if (!userId) return;
    activePlannerSession = session;
    localStorage.setItem(plannerStorageKey(), JSON.stringify(session));
    renderActivePlannerButton();
  }

  function clearPlannerSession() {
    if (userId) localStorage.removeItem(plannerStorageKey());
    activePlannerSession = null;
    renderActivePlannerButton();
  }

  function renderActivePlannerButton() {
    const company = activePlannerSession?.company || '';
    activePlannerBtn.classList.toggle('show', !!company);
    activePlannerBtn.textContent = company || 'Open planner';
  }

  function plannerFlowSummary(daily) {
    const items = Array.isArray(daily) ? daily : [];
    if (!items.length) return [];
    const checkpoints = [0, Math.floor(items.length / 3), Math.floor((items.length * 2) / 3), items.length - 1];
    return [...new Set(checkpoints)]
      .filter((index) => index >= 0 && index < items.length)
      .map((index) => ({
        label: `Day ${index}`,
        title: items[index]?.title || 'Preparation block',
        goal: items[index]?.goal || ''
      }));
  }

  function renderPlannerStartView(data) {
    const planner = data?.planner || {};
    const stages = Array.isArray(planner.stages) ? planner.stages : [];
    const summaryItems = plannerFlowSummary(planner.daily_plan || []);
    plannerMainPanel.innerHTML = `
      <div class="plannerBlock">
        <b>${planner.company || data.company}</b>
        <div class="tiny">${planner.overview || 'Planner generated from web-backed company research.'}</div>
      </div>
      <div class="plannerSectionTitle">14-Day Flow</div>
      <div class="plannerSummaryGrid">
        ${summaryItems.map((item) => `
          <div class="plannerBlock">
            <b>${item.label}</b>
            <div class="tiny">${item.title}</div>
            <div class="tiny" style="margin-top:6px;">${item.goal}</div>
          </div>
        `).join('')}
      </div>
      <div class="plannerSectionTitle">Stages</div>
      ${stages.map((stage) => `
        <div class="plannerBlock">
          <b>${stage.name}</b>
          <div class="tiny">${stage.focus || ''}</div>
        </div>
      `).join('')}
      <div class="plannerStartRow">
        <span class="tiny">Start the guided flow to unlock Day 0.</span>
        <button class="btn" id="startPlannerFlowBtn" type="button">Start</button>
      </div>
    `;
    const startPlannerFlowBtn = document.getElementById('startPlannerFlowBtn');
    if (startPlannerFlowBtn) {
      startPlannerFlowBtn.addEventListener('click', () => {
        const session = {
          company: planner.company || data.company,
          daysLeft: Number(data.days_left || (planner.daily_plan || []).length || 14),
          planner,
          sources: data.sources || [],
          currentDay: 0,
          started: true,
        };
        savePlannerSession(session);
        renderPlannerProgressView(session);
        plannerState.textContent = 'Planner started. Day 0 is ready.';
      });
    }
  }

  function renderPlannerProgressView(session) {
    const planner = session?.planner || {};
    const daily = Array.isArray(planner.daily_plan) ? planner.daily_plan : [];
    const currentDay = Math.max(0, Number(session?.currentDay || 0));
    const current = daily[currentDay];
    if (!current) {
      plannerMainPanel.innerHTML = `
        <div class="plannerBlock">
          <b>${planner.company || session?.company || 'Planner complete'}</b>
          <div class="tiny">All guided days are complete.</div>
        </div>
        <div class="plannerStartRow">
          <span class="tiny">You can restart this company flow anytime.</span>
          <button class="btn" id="restartPlannerBtn" type="button">Restart</button>
        </div>
      `;
      const restartPlannerBtn = document.getElementById('restartPlannerBtn');
      if (restartPlannerBtn) {
        restartPlannerBtn.addEventListener('click', () => {
          const nextSession = { ...session, currentDay: 0 };
          savePlannerSession(nextSession);
          renderPlannerProgressView(nextSession);
        });
      }
      return;
    }
    plannerMainPanel.innerHTML = `
      <div class="plannerBlock">
        <b>${planner.company || session.company}</b>
        <div class="plannerProgressLine">
          <span class="tag">Current Day ${currentDay}</span>
          <span class="tag">${Math.min(currentDay, daily.length)}/${daily.length} done</span>
        </div>
      </div>
      <div class="plannerDayCard">
        <b>Day ${currentDay}: ${current.title}</b>
        <ul>${Array.isArray(current.tasks) ? current.tasks.map((task) => `<li>${task}</li>`).join('') : ''}</ul>
        <div class="tiny" style="margin-top:8px;">${current.goal || ''}</div>
      </div>
      <div class="plannerStartRow">
        <span class="tiny">Complete this day to unlock the next one.</span>
        <div class="plannerProgressLine">
          <button class="btn" id="plannerDoneBtn" type="button">Done</button>
          <button class="btn" id="plannerResetBtn" type="button">Reset</button>
        </div>
      </div>
    `;
    const plannerDoneBtn = document.getElementById('plannerDoneBtn');
    const plannerResetBtn = document.getElementById('plannerResetBtn');
    if (plannerDoneBtn) {
      plannerDoneBtn.addEventListener('click', () => {
        const nextSession = { ...session, currentDay: currentDay + 1 };
        savePlannerSession(nextSession);
        renderPlannerProgressView(nextSession);
        plannerState.textContent = nextSession.currentDay >= daily.length ? 'Planner complete.' : `Moved to Day ${nextSession.currentDay}.`;
      });
    }
    if (plannerResetBtn) {
      plannerResetBtn.addEventListener('click', () => {
        const nextSession = { ...session, currentDay: 0 };
        savePlannerSession(nextSession);
        renderPlannerProgressView(nextSession);
        plannerState.textContent = 'Planner reset to Day 0.';
      });
    }
  }

  function renderPlannerResult(data) {
    const planner = data?.planner || {};
    const recommendations = Array.isArray(planner.recommendations) ? planner.recommendations : [];
    const patterns = Array.isArray(planner.likely_previous_question_patterns) ? planner.likely_previous_question_patterns : [];
    const webTopics = Array.isArray(planner.web_focus_topics) ? planner.web_focus_topics : [];
    const focusMap = planner.personalized_focus || {};
    const strengths = Array.isArray(focusMap.strengths_to_use) ? focusMap.strengths_to_use : [];
    const weaknesses = Array.isArray(focusMap.weaknesses_to_focus) ? focusMap.weaknesses_to_focus : [];
    const improving = Array.isArray(focusMap.improving_now) ? focusMap.improving_now : [];
    const fitAnalysis = planner.fit_analysis || {};
    const matchedStrengths = Array.isArray(fitAnalysis.matched_strengths) ? fitAnalysis.matched_strengths : [];
    const matchedWeaknesses = Array.isArray(fitAnalysis.matched_weaknesses) ? fitAnalysis.matched_weaknesses : [];
    const strategicSummary = fitAnalysis.strategic_summary || '';
    const sources = Array.isArray(data?.sources) ? data.sources : [];
    const renderPlannerList = (items, emptyText) => {
      if (!Array.isArray(items) || !items.length) {
        return `<div class="tiny">${escapeXml(emptyText)}</div>`;
      }
      return `<ul>${items.map((item) => `<li>${escapeXml(String(item || ''))}</li>`).join('')}</ul>`;
    };
    latestPlannerPayload = data;
    renderPlannerStartView(data);

    plannerSourcePanel.innerHTML = `
      <div class="plannerSectionTitle">Topics Asked From Web</div>
      <div class="plannerBlock">
        ${renderPlannerList(webTopics, 'No strong web topics were extracted.')}
      </div>
      <div class="plannerSectionTitle">Personalized Focus Map</div>
      <div class="plannerBlock">
        <b>Analysis</b>
        <div class="tiny">${escapeXml(strategicSummary || 'No personalized analysis available yet.')}</div>
      </div>
      <div class="plannerBlock">
        <b>Strengths that match company topics</b>
        ${renderPlannerList(matchedStrengths, 'No strong cached overlap found yet.')}
      </div>
      <div class="plannerBlock">
        <b>Weaknesses that need focus first</b>
        ${renderPlannerList(matchedWeaknesses, 'No weak-topic overlap detected yet.')}
      </div>
      <div class="plannerBlock">
        <b>Full cached profile snapshot</b>
        ${renderPlannerList([
          ...strengths.map((item) => `Strength: ${item}`),
          ...weaknesses.map((item) => `Weakness: ${item}`),
          ...improving.map((item) => `Improving: ${item}`)
        ], 'No cached profile signals yet. Chat more about your strengths and weak areas first.')}
      </div>
      <div class="plannerBlock">
        <b>Currently improving</b>
        ${renderPlannerList(improving, 'No active improving signals yet.')}
      </div>
      <div class="plannerSectionTitle">Recommendations</div>
      <div class="plannerBlock">
        ${renderPlannerList(recommendations, 'No recommendations returned.')}
      </div>
      <div class="plannerSectionTitle">Previous Question Patterns</div>
      <div class="plannerBlock">
        ${renderPlannerList(patterns, 'No repeated question patterns detected.')}
      </div>
      <div class="plannerSectionTitle">Web Sources</div>
      ${sources.length ? sources.map((item) => `
        <div class="plannerBlock">
          <b>${item.title || 'Source'}</b>
          <div class="tiny">${item.snippet || ''}</div>
          <div class="tiny" style="margin-top:6px;"><a href="${escapeXml(item.url || '#')}" target="_blank" rel="noopener noreferrer">${item.url || ''}</a></div>
        </div>
      `).join('') : '<div class="plannerPlaceholder">No web sources returned.</div>'}
    `;
  }

  async function generatePlanner() {
    const company = (plannerCompanyInput.value || '').trim();
    const parsedDays = parseInt(String(plannerDaysInput.value || '').trim(), 10);
    const daysLeft = Number.isFinite(parsedDays) && parsedDays > 0 ? parsedDays : 14;
    if (!company || !daysLeft) {
      plannerState.textContent = 'Enter both company and days.';
      return;
    }
    plannerDaysInput.value = String(daysLeft);
    plannerState.textContent = 'Researching company rounds and generating plan...';
    generatePlannerBtn.disabled = true;
    try {
      const res = await fetch('/planner/company', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          company,
          days_left: daysLeft
        })
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data?.detail || 'Unable to generate planner.');
      }
      clearPlannerSession();
      renderPlannerResult(data);
      plannerState.textContent = 'Planner ready.';
    } catch (e) {
      console.error(e);
      plannerState.textContent = e.message || 'Unable to generate planner.';
    } finally {
      generatePlannerBtn.disabled = false;
    }
  }

  function formatConversationPreview(item) {
    const preview = String(item?.preview || '').trim();
    if (!preview) return 'Empty conversation';
    return preview.length > 54 ? preview.slice(0, 51).trimEnd() + '...' : preview;
  }

  async function loadConversationList() {
    if (!userId) {
      conversationList.innerHTML = '<div class="item tiny">Sign in to see saved chats.</div>';
      return;
    }
    try {
      const res = await fetch('/chat/conversations');
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data?.detail || 'Unable to load conversations.');
      }
      const items = data?.items || [];
      if (!items.length) {
        conversationList.innerHTML = '<div class="item tiny">No saved conversations yet.</div>';
        return;
      }
      conversationList.innerHTML = '';
      for (const item of items) {
        const button = document.createElement('button');
        button.className = 'convoBtn' + (item.conversation_id === convoId ? ' active' : '');
        button.innerHTML =
          '<div class="convoTitle">' + formatConversationPreview(item) + '</div>' +
          '<div class="convoMeta">' + item.conversation_id + '</div>';
        button.addEventListener('click', async () => {
          convoId = item.conversation_id;
          localStorage.setItem('graphmind_convo_id', convoId);
          cidEl.textContent = convoId;
          await loadConversationList();
          await loadChatHistory();
        });
        conversationList.appendChild(button);
      }
    } catch (e) {
      console.error(e);
      conversationList.innerHTML = '<div class="item tiny">Unable to load saved chats.</div>';
    }
  }

  function renderEphemeral(items, backend) {
    backendLabel.textContent = backend || 'unknown';
    if (!items || !items.length) {
      renderEmpty(ephemeralList, 'No active short-term signals.');
      return;
    }
    ephemeralList.innerHTML = items.slice(0, 5).map((item) => `
      <div class="item">
        <b>${item.entity}</b>
        <div class="row">
          <span class="tag">${item.relation.replaceAll('_', ' ')}</span>
          <span class="tag">${item.entity_type}</span>
          <span>${item.mention_count || 0} mentions</span>
          <span class="${item.promoted ? 'ok' : 'warn'}">${item.promoted ? 'promoted' : 'warming up'}</span>
        </div>
      </div>
    `).join('');
  }

  function renderGraph(items) {
    if (!items || !items.length) {
      renderEmpty(graphList, 'No graph memory yet.');
      return;
    }
    graphList.innerHTML = items.slice(0, 6).map((item) => `
      <div class="item">
        <b>${item.entity}</b>
        <div class="row">
          <span class="tag">${item.relation.replaceAll('_', ' ')}</span>
          <span class="tag">${item.entity_type}</span>
          <span>${item.reinforcement_count || 0} reinforcements</span>
          <span>${item.related_count || 0} linked</span>
        </div>
        ${(item.aliases && item.aliases.length > 1) ? `<div class="tiny">aliases: ${item.aliases.join(', ')}</div>` : ''}
      </div>
    `).join('');
  }

  function renderProfile(profile) {
    const strengths = Array.isArray(profile?.strengths) ? profile.strengths : [];
    const weaknesses = Array.isArray(profile?.weaknesses) ? profile.weaknesses : [];
    const improving = Array.isArray(profile?.improving) ? profile.improving : [];
    const rows = [
      ...strengths.map((item) => ({ label: 'Strength', tone: 'ok', item })),
      ...weaknesses.map((item) => ({ label: 'Weakness', tone: 'warn', item })),
      ...improving.map((item) => ({ label: 'Improving', tone: '', item })),
    ];
    if (!rows.length) {
      renderEmpty(profileList, 'No cached profile signals yet.');
      return;
    }
    profileList.innerHTML = rows.slice(0, 9).map(({ label, tone, item }) => `
      <div class="item">
        <b>${escapeXml(item.entity || 'Unknown')}</b>
        <div class="row">
          <span class="tag ${tone}">${label}</span>
          <span class="tag">${escapeXml(item.entity_type || 'Skill')}</span>
          <span>score ${Number(item.score || 0).toFixed(2)}</span>
          <span>${Number(item.evidence_count || 0)} signals</span>
        </div>
      </div>
    `).join('');
  }

  function renderSummary(data) {
    const promoted = data?.promotion_summary?.promoted_count ?? 0;
    const signals = data?.signals_extracted ?? 0;
    const retrieved = data?.retrieved_count ?? 0;
    const graphFacts = data?.graph_evidence?.facts?.length ?? 0;
    const route = data?.route?.intent || 'unknown';
    const topic = data?.topic_match?.topic || '';
    const provider = data?.llm_provider || 'unknown';
    llmLabel.textContent = provider;
    llmDisplay.textContent = provider;
    const headline = promoted > 0
      ? `Memory updated with ${promoted} promoted item${promoted === 1 ? '' : 's'}.`
      : signals > 0
        ? `Signals detected and kept ready for reinforcement.`
        : `No major memory change from this turn.`;
    const supporting = graphFacts > 0
      ? `Answer grounded with ${graphFacts} graph fact${graphFacts === 1 ? '' : 's'} and ${retrieved} retrieved snippet${retrieved === 1 ? '' : 's'}.`
      : `Route: ${route.replaceAll('_', ' ')}.` + (topic ? ` Focus: ${topic}.` : '');
    const tags = [
      `<span class="tag">signals ${signals}</span>`,
      `<span class="tag">promoted ${promoted}</span>`,
      `<span class="tag">retrieved ${retrieved}</span>`,
      `<span class="tag mutedTag">${route.replaceAll('_', ' ')}</span>`
    ];
    if (topic) tags.unshift(`<span class="tag">${topic}</span>`);
    summary.innerHTML = `
      <div class="summaryHero">
        <div>
          <b>${headline}</b>
          <span>${supporting}</span>
        </div>
        <div class="summaryStats">${tags.join('')}</div>
      </div>
    `;
    memoryUpdated.textContent = 'Updated just now';
  }

  function renderEvidence(paths) {
    if (!paths || !paths.length) {
      renderEmpty(evidenceList, 'No evidence paths yet.');
      return;
    }
    evidenceList.innerHTML = paths.map((path) => `
      <div class="item">
        <div class="tiny">${path}</div>
      </div>
    `).join('');
  }

  function colorForType(type) {
    const key = String(type || '').toLowerCase();
    if (key === 'user') return '#f59e0b';
    if (key === 'company') return '#a78bfa';
    if (key === 'goal') return '#fb7185';
    if (key === 'skill') return '#22c55e';
    if (key === 'topic') return '#38bdf8';
    return '#94a3b8';
  }

  function escapeXml(value) {
    return String(value || '')
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  function renderGraphView(target, nodes, edges, options = {}) {
    if (!nodes || !nodes.length) {
      target.innerHTML = '<div class="graphEmpty">No graph nodes yet.</div>';
      return;
    }

    const width = options.width || 520;
    const height = options.height || 300;
    const paddingX = Math.max(48, Math.round(width * 0.08));
    const paddingY = Math.max(42, Math.round(height * 0.10));
    const centerX = width * 0.52;
    const centerY = height * 0.58;
    const userNode = nodes.find((node) => String(node.type) === 'User') || nodes[0];
    const otherNodes = nodes.filter((node) => node.id !== userNode.id);
    const positions = new Map();
    positions.set(userNode.id, {
      x: Math.min(width - paddingX - 24, Math.max(paddingX + 24, width * 0.72)),
      y: Math.min(height - paddingY - 24, Math.max(paddingY + 24, height * 0.72)),
    });

    const edgeList = edges || [];
    const adjacency = new Map();
    edgeList.forEach((edge) => {
      const source = String(edge.source || '');
      const target = String(edge.target || '');
      if (!adjacency.has(source)) adjacency.set(source, []);
      if (!adjacency.has(target)) adjacency.set(target, []);
      adjacency.get(source).push(edge);
      adjacency.get(target).push(edge);
    });

    const lanes = {
      company: { start: Math.PI * 1.02, end: Math.PI * 1.30, radius: 0.88 },
      goal: { start: Math.PI * 1.22, end: Math.PI * 1.52, radius: 0.78 },
      skill: { start: Math.PI * 1.55, end: Math.PI * 1.88, radius: 0.72 },
      topic: { start: Math.PI * 1.88, end: Math.PI * 2.26, radius: 0.82 },
      entity: { start: Math.PI * 1.35, end: Math.PI * 2.18, radius: 0.92 },
      other: { start: Math.PI * 1.08, end: Math.PI * 2.18, radius: 0.96 },
    };

    const buckets = {
      company: [],
      goal: [],
      skill: [],
      topic: [],
      entity: [],
      other: [],
    };

    otherNodes.forEach((node) => {
      const type = String(node.type || '').toLowerCase();
      if (type === 'company') buckets.company.push(node);
      else if (type === 'goal') buckets.goal.push(node);
      else if (type === 'skill') buckets.skill.push(node);
      else if (type === 'topic') buckets.topic.push(node);
      else if (['entity', 'concept', 'domain', 'document'].includes(type)) buckets.entity.push(node);
      else buckets.other.push(node);
    });

    function placeArc(items, laneKey) {
      if (!items.length) return;
      const lane = lanes[laneKey];
      const radiusXBase = Math.max(90, (width / 2) - paddingX - 26);
      const radiusYBase = Math.max(72, (height / 2) - paddingY - 26);
      items.forEach((node, index) => {
        const t = items.length === 1 ? 0.5 : index / Math.max(1, items.length - 1);
        const angle = lane.start + (lane.end - lane.start) * t;
        const wobble = ((index % 2 === 0) ? -1 : 1) * (10 + (index % 3) * 6);
        const radiusScale = lane.radius + ((index % 3) - 1) * 0.07;
        const x = centerX + Math.cos(angle) * radiusXBase * radiusScale + wobble * 0.18;
        const y = centerY + Math.sin(angle) * radiusYBase * radiusScale + wobble * 0.32;
        positions.set(node.id, {
          x: Math.min(width - paddingX, Math.max(paddingX, x)),
          y: Math.min(height - paddingY, Math.max(paddingY, y)),
        });
      });
    }

    placeArc(buckets.company, 'company');
    placeArc(buckets.goal, 'goal');
    placeArc(buckets.skill, 'skill');
    placeArc(buckets.topic, 'topic');
    placeArc(buckets.entity, 'entity');
    placeArc(buckets.other, 'other');

    function nodeRadius(node) {
      return Math.max(12, Number(node.size || 16));
    }

    function clampPosition(pos, radius) {
      pos.x = Math.min(width - paddingX - radius, Math.max(paddingX + radius, pos.x));
      pos.y = Math.min(height - paddingY - radius, Math.max(paddingY + radius, pos.y));
    }

    function separateNodes(iterations = 36) {
      for (let step = 0; step < iterations; step++) {
        for (let i = 0; i < nodes.length; i++) {
          const a = nodes[i];
          if (a.id === userNode.id) continue;
          const posA = positions.get(a.id);
          if (!posA) continue;
          for (let j = i + 1; j < nodes.length; j++) {
            const b = nodes[j];
            if (b.id === userNode.id) continue;
            const posB = positions.get(b.id);
            if (!posB) continue;
            const dx = posB.x - posA.x;
            const dy = posB.y - posA.y;
            const distance = Math.sqrt(dx * dx + dy * dy) || 1;
            const minDistance = nodeRadius(a) + nodeRadius(b) + 68;
            if (distance >= minDistance) continue;
            const push = (minDistance - distance) / 2;
            const nx = dx / distance;
            const ny = dy / distance;
            posA.x -= nx * push;
            posA.y -= ny * push;
            posB.x += nx * push;
            posB.y += ny * push;
            clampPosition(posA, nodeRadius(a));
            clampPosition(posB, nodeRadius(b));
          }
        }
      }
    }

    function relaxTowardNeighbors(iterations = 18) {
      for (let step = 0; step < iterations; step++) {
        otherNodes.forEach((node) => {
          const pos = positions.get(node.id);
          if (!pos) return;
          const connected = (adjacency.get(node.id) || [])
            .map((edge) => String(edge.source) === node.id ? String(edge.target) : String(edge.source))
            .map((id) => positions.get(id))
            .filter(Boolean);
          if (!connected.length) return;
          const avgX = connected.reduce((sum, item) => sum + item.x, 0) / connected.length;
          const avgY = connected.reduce((sum, item) => sum + item.y, 0) / connected.length;
          pos.x = pos.x * 0.88 + avgX * 0.12;
          pos.y = pos.y * 0.88 + avgY * 0.12;
          clampPosition(pos, nodeRadius(node));
        });
        separateNodes(3);
      }
    }

    relaxTowardNeighbors();
    separateNodes(28);

    function normalizeSpread() {
      const placed = nodes
        .map((node) => ({ node, pos: positions.get(node.id) }))
        .filter((entry) => entry.pos);
      if (!placed.length) return;

      const minX = Math.min(...placed.map((entry) => entry.pos.x));
      const maxX = Math.max(...placed.map((entry) => entry.pos.x));
      const minY = Math.min(...placed.map((entry) => entry.pos.y));
      const maxY = Math.max(...placed.map((entry) => entry.pos.y));
      const spanX = Math.max(1, maxX - minX);
      const spanY = Math.max(1, maxY - minY);
      const usableWidth = width - paddingX * 2;
      const usableHeight = height - paddingY * 2;

      const shouldStretchX = spanX < usableWidth * 0.72;
      const shouldStretchY = spanY < usableHeight * 0.72;
      if (!shouldStretchX && !shouldStretchY) return;

      placed.forEach(({ node, pos }) => {
        const radius = nodeRadius(node);
        if (shouldStretchX) {
          const tx = (pos.x - minX) / spanX;
          pos.x = paddingX + tx * usableWidth;
        }
        if (shouldStretchY) {
          const ty = (pos.y - minY) / spanY;
          pos.y = paddingY + ty * usableHeight;
        }
        clampPosition(pos, radius);
      });
    }

    normalizeSpread();
    separateNodes(18);

    function edgePath(source, target, edge) {
      const dx = target.x - source.x;
      const dy = target.y - source.y;
      const isMemory = edge.kind === 'memory';
      const curve = isMemory ? 18 : 0;
      const cx = (source.x + target.x) / 2 + (isMemory ? (dy >= 0 ? curve : -curve) : 0);
      const cy = (source.y + target.y) / 2 - (isMemory ? (dx >= 0 ? curve : -curve) : 0);
      return { d: `M ${source.x.toFixed(1)} ${source.y.toFixed(1)} Q ${cx.toFixed(1)} ${cy.toFixed(1)} ${target.x.toFixed(1)} ${target.y.toFixed(1)}`, cx, cy };
    }

    const edgeSvg = (edges || []).map((edge) => {
      const source = positions.get(edge.source);
      const target = positions.get(edge.target);
      if (!source || !target) return '';
      const path = edgePath(source, target, edge);
      const mx = path.cx.toFixed(1);
      const my = path.cy.toFixed(1);
      const stroke = edge.kind === 'entity' ? 'rgba(251,113,133,0.8)' : 'rgba(56,189,248,0.35)';
      const showLabel = edge.kind === 'entity';
      return `
        <g>
          <path d="${path.d}" fill="none" stroke="${stroke}" stroke-width="${edge.kind === 'entity' ? 2.4 : 1.5}" />
          ${showLabel ? `<text x="${mx}" y="${my}" fill="rgba(229,231,235,.96)" font-size="9" text-anchor="middle">${escapeXml(edge.label)}</text>` : ''}
        </g>
      `;
    }).join('');

    const nodeSvg = nodes.map((node) => {
      const pos = positions.get(node.id);
      if (!pos) return '';
      const color = colorForType(node.type);
      const radius = Math.max(12, Number(node.size || 16));
      const labelY = pos.y + radius + 16;
      return `
        <g>
          <circle cx="${pos.x.toFixed(1)}" cy="${pos.y.toFixed(1)}" r="${radius}" fill="${color}" fill-opacity="0.18" stroke="${color}" stroke-width="2" />
          <circle cx="${pos.x.toFixed(1)}" cy="${pos.y.toFixed(1)}" r="${Math.max(4, radius / 3)}" fill="${color}" />
          <text x="${pos.x.toFixed(1)}" y="${labelY.toFixed(1)}" fill="#e5e7eb" font-size="11" text-anchor="middle">${escapeXml(node.label)}</text>
        </g>
      `;
    }).join('');

    target.innerHTML = `
      <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Memory graph">
        ${edgeSvg}
        ${nodeSvg}
      </svg>
    `;
  }

  async function loadChatHistory() {
    if (!userId || !convoId) {
      return;
    }
    try {
      const res = await fetch('/chat/history/' + convoId);
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data?.detail || 'Unable to load chat history.');
      }
      const history = data?.messages || [];
      if (!history.length) {
        msgs.innerHTML = '<div class="msg bot">New conversation started. What are you working on now?</div>';
        return;
      }
      msgs.innerHTML = '';
      for (const message of history) {
        add(message.content || '', message.role === 'user' ? 'user' : 'bot');
      }
    } catch (e) {
      console.error(e);
      msgs.innerHTML = '<div class="msg bot">Unable to load chat history for this conversation.</div>';
    }
  }

  async function refreshMemory() {
    if (!userId) {
      return;
    }
    try {
      const [ephemeralRes, graphRes, graphViewRes, profileRes] = await Promise.all([
        fetch('/memory/ephemeral/' + userId),
        fetch('/graph/memory/' + userId),
        fetch('/graph/view/' + userId),
        fetch('/profile/summary/' + userId)
      ]);
      if (!ephemeralRes.ok || !graphRes.ok || !graphViewRes.ok || !profileRes.ok) {
        throw new Error('Unable to refresh memory for this session.');
      }
      const ephemeralData = await ephemeralRes.json();
      const graphData = await graphRes.json();
      const graphViewData = await graphViewRes.json();
      const profileData = await profileRes.json();
      renderEphemeral(ephemeralData.items || [], ephemeralData.backend);
      renderGraph(graphData.items || []);
      renderProfile(profileData.profile || {});
      latestGraphNodes = graphViewData.nodes || [];
      latestGraphEdges = graphViewData.edges || [];
      renderGraphView(graphCanvas, latestGraphNodes, latestGraphEdges);
      renderGraphView(graphModalCanvas, latestGraphNodes, latestGraphEdges, { width: 1100, height: 760 });
    } catch (e) {
      console.error(e);
    }
  }

  async function refreshMemoryEventually(delays) {
    for (const delay of delays) {
      await new Promise((resolve) => setTimeout(resolve, delay));
      await refreshMemory();
    }
  }

  document.getElementById('newChat').addEventListener('click', () => {
    if (!userId) return;
    convoId = "convo-" + rand();
    localStorage.setItem('graphmind_convo_id', convoId);
    cidEl.textContent = convoId;
    msgs.innerHTML = '<div class="msg bot">New conversation started. What are you working on now?</div>';
    loadConversationList();
    input.focus();
  });

  function add(text, role) {
    const div = document.createElement('div');
    div.className = 'msg ' + (role === 'user' ? 'user' : 'bot');
    div.textContent = text;
    msgs.appendChild(div);
    msgs.scrollTop = msgs.scrollHeight;
    return div;
  }

  function addWebChoicePrompt(questionText, originalText) {
    const container = document.createElement('div');
    container.className = 'msg bot';
    container.textContent = questionText;

    const row = document.createElement('div');
    row.className = 'choiceRow';

    const yesBtn = document.createElement('button');
    yesBtn.className = 'choiceBtn';
    yesBtn.textContent = 'Yes';
    yesBtn.addEventListener('click', async () => {
      row.remove();
      container.appendChild(document.createTextNode(' Searching the web...'));
      await send({ overrideText: originalText, allowWebSearch: true, skipUserEcho: true });
    });

    const noBtn = document.createElement('button');
    noBtn.className = 'choiceBtn';
    noBtn.textContent = 'No';
    noBtn.addEventListener('click', () => {
      row.remove();
      const note = document.createElement('div');
      note.className = 'tiny';
      note.textContent = 'Okay, staying with memory only.';
      container.appendChild(note);
    });

    row.appendChild(yesBtn);
    row.appendChild(noBtn);
    container.appendChild(row);
    msgs.appendChild(container);
    msgs.scrollTop = msgs.scrollHeight;
  }

  async function send(options = {}) {
    const text = (options.overrideText || input.value).trim();
    if (!text) return;
    if (!userId) {
      authStatusEl.textContent = 'Sign in before sending messages.';
      return;
    }
    if (!options.skipUserEcho) {
      add(text, 'user');
    }
    if (!options.overrideText) {
      input.value = '';
    }

    sendBtn.disabled = true;
    statusEl.textContent = 'Thinking...';
    const start = performance.now();
    try {
      const res = await fetch('/chat', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({
          conversation_id: convoId,
          message: text,
          allow_web_search: !!options.allowWebSearch
        })
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data?.detail || 'Chat request failed.');
      }
      if (!data?.memory_found && !data?.web_search_used) {
        add('I could not find anything relevant in your memory for that yet.', 'bot');
        if (data?.answer) {
          add(data.answer, 'bot');
        }
        addWebChoicePrompt(
          'Do you want me to search the web for this too?',
          text
        );
      } else {
        add(data.answer || 'No answer', 'bot');
      }
      renderSummary(data);
      renderEvidence(data?.graph_evidence?.paths || []);
      await loadConversationList();
      await refreshMemory();
      refreshMemoryEventually([900, 1800, 3200]);
      statusEl.textContent = 'Ready';
      const totalMs = data?.time_ms ?? Math.round(performance.now()-start);
      const retrievalMs = data?.retrieval_time_ms ?? 0;
      const graphRetrievalMs = data?.graph_retrieval_time_ms ?? 0;
      const graphPromotionMs = data?.graph_promotion_time_ms ?? 0;
      const llmMs = data?.llm_generation_time_ms ?? 0;
      const extractionMs = data?.signal_extraction_time_ms ?? 0;
      const webMs = data?.web_retrieval_time_ms ?? 0;
      latEl.textContent = 'Response ' + totalMs + ' ms | LLM ' + llmMs + ' ms | Extract ' + extractionMs + ' ms | Graph ' + graphRetrievalMs + ' ms | Promotion ' + graphPromotionMs + ' ms | Vector ' + retrievalMs + ' ms | Web ' + webMs + ' ms';
    } catch (e) {
      console.error(e);
      add(e.message || 'Server error. Check backend logs.', 'bot');
      statusEl.textContent = 'Error';
      latEl.textContent = 'Failed';
    } finally {
      sendBtn.disabled = false;
      input.focus();
    }
  }

  sendBtn.addEventListener('click', send);
  loginBtn.addEventListener('click', () => submitAuth('/auth/login'));
  registerBtn.addEventListener('click', () => submitAuth('/auth/register'));
  logoutBtn.addEventListener('click', async () => {
    await fetch('/auth/logout', { method: 'POST' });
    setAuthState({ authenticated: false, user: null });
    msgs.innerHTML = '<div class="msg bot">Signed out. Sign back in to continue with your own memory space.</div>';
  });
  authPassword.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      submitAuth('/auth/login');
    }
  });
  authUsername.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      submitAuth('/auth/login');
    }
  });
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
  });

  document.getElementById('refreshMemory').addEventListener('click', refreshMemory);
  openPlannerBtn.addEventListener('click', () => {
    plannerModal.classList.add('open');
    plannerModal.setAttribute('aria-hidden', 'false');
    if (activePlannerSession) {
      renderPlannerProgressView(activePlannerSession);
      plannerCompanyInput.value = activePlannerSession.company || '';
      plannerDaysInput.value = String(activePlannerSession.daysLeft || 14);
      plannerState.textContent = `Active planner: ${activePlannerSession.company}`;
    } else if (latestPlannerPayload) {
      renderPlannerStartView(latestPlannerPayload);
    }
    plannerCompanyInput.focus();
  });
  activePlannerBtn.addEventListener('click', () => {
    if (!activePlannerSession) return;
    plannerModal.classList.add('open');
    plannerModal.setAttribute('aria-hidden', 'false');
    renderPlannerProgressView(activePlannerSession);
    plannerCompanyInput.value = activePlannerSession.company || '';
    plannerDaysInput.value = String(activePlannerSession.daysLeft || 14);
    plannerState.textContent = `Active planner: ${activePlannerSession.company}`;
  });
  closePlannerModalBtn.addEventListener('click', () => {
    plannerModal.classList.remove('open');
    plannerModal.setAttribute('aria-hidden', 'true');
  });
  plannerModal.addEventListener('click', (event) => {
    if (event.target === plannerModal) {
      plannerModal.classList.remove('open');
      plannerModal.setAttribute('aria-hidden', 'true');
    }
  });
  generatePlannerBtn.addEventListener('click', generatePlanner);
  plannerCompanyInput.addEventListener('keydown', (event) => {
    if (event.key === 'Enter') {
      event.preventDefault();
      generatePlanner();
    }
  });
  plannerDaysInput.addEventListener('keydown', (event) => {
    if (event.key === 'Enter') {
      event.preventDefault();
      generatePlanner();
    }
  });
  maximizeGraphBtn.addEventListener('click', () => {
    renderGraphView(graphModalCanvas, latestGraphNodes, latestGraphEdges, { width: 1100, height: 760 });
    graphModal.classList.add('open');
    graphModal.setAttribute('aria-hidden', 'false');
  });
  closeGraphModalBtn.addEventListener('click', () => {
    graphModal.classList.remove('open');
    graphModal.setAttribute('aria-hidden', 'true');
  });
  graphModal.addEventListener('click', (event) => {
    if (event.target === graphModal) {
      graphModal.classList.remove('open');
      graphModal.setAttribute('aria-hidden', 'true');
    }
  });
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && graphModal.classList.contains('open')) {
      graphModal.classList.remove('open');
      graphModal.setAttribute('aria-hidden', 'true');
    }
    if (event.key === 'Escape' && plannerModal.classList.contains('open')) {
      plannerModal.classList.remove('open');
      plannerModal.setAttribute('aria-hidden', 'true');
    }
  });
  document.getElementById('resetMemory').addEventListener('click', async () => {
    const ok = window.confirm('Reset all memory for this user? This clears graph, ephemeral, vector, and event history for the current user.');
    if (!ok) return;
    if (!userId) return;
    statusEl.textContent = 'Resetting...';
    latEl.textContent = 'Clearing user memory';
    try {
      const res = await fetch('/memory/reset/' + userId, { method: 'DELETE' });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data?.detail || 'Reset failed.');
      }
      msgs.innerHTML = '<div class="msg bot">User memory reset. You can test fresh extraction now.</div>';
      renderEmpty(ephemeralList, 'No ephemeral signals yet.');
      renderEmpty(graphList, 'No graph memory yet.');
      renderEmpty(evidenceList, 'No evidence paths yet.');
      graphCanvas.innerHTML = '<div class="graphEmpty">No graph nodes yet.</div>';
      graphModalCanvas.innerHTML = '<div class="graphEmpty">No graph nodes yet.</div>';
      latestGraphNodes = [];
      latestGraphEdges = [];
      summary.innerHTML = [
        `<span class="tag">graph ${data?.graph?.user_relationships_deleted ?? 0}</span>`,
        `<span class="tag">vector ${data?.vector?.deleted_messages ?? 0}</span>`,
        `<span class="tag">events ${data?.events?.raw_events_deleted ?? 0}</span>`
      ].join('');
      memoryUpdated.textContent = 'Memory reset for ' + userId;
      statusEl.textContent = 'Ready';
      latEl.textContent = 'Reset complete';
      await refreshMemory();
    } catch (e) {
      console.error(e);
      statusEl.textContent = 'Error';
      latEl.textContent = 'Reset failed';
    }
  });

  renderEmpty(ephemeralList, 'No ephemeral signals yet.');
  renderEmpty(graphList, 'No graph memory yet.');
  renderEmpty(evidenceList, 'No evidence paths yet.');
  setAuthState({ authenticated: false, user: null });
  hydrateSession();
})();
</script>
</body>
</html>
"""

def _compress_snippet(text: str, limit: int = 180) -> str:
    cleaned = " ".join((text or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3].rstrip() + "..."


def _fetch_graph_bundle(
    *,
    user_id: str,
    query: str,
    section_tags: list[str] | None = None,
    section_families: list[str] | None = None,
    focus_entity: str | None = None,
) -> tuple[dict[str, object], list[str], int]:
    graph_evidence: dict[str, object] = {"facts": [], "paths": [], "citations": []}
    graph_context: list[str] = []
    start = time.time()
    with get_session() as session:
        graph_evidence = graph_memory_service.fetch_graph_evidence(
            session=session,
            user_id=user_id,
            query=query,
            limit=6,
        )
        if section_tags or section_families:
            graph_context = graph_memory_service.fetch_section_context(
                session=session,
                user_id=user_id,
                section_tags=section_tags or [],
                section_families=section_families or [],
                focus_entity=focus_entity,
                query=query,
                limit=6,
            )
        if not graph_context and list(graph_evidence.get("paths") or []):
            graph_context = graph_memory_service.fetch_graph_context(
                session=session,
                user_id=user_id,
                limit=6,
            )
    return graph_evidence, graph_context, int((time.time() - start) * 1000)


def _fetch_vector_bundle(
    *,
    query: str,
    user_id: str,
    conversation_id: str,
    k: int,
) -> tuple[list[dict[str, object]], int]:
    start = time.time()
    results = vector_search(
        query=query,
        user_id=user_id,
        conversation_id=conversation_id,
        k=k,
    )
    return results, int((time.time() - start) * 1000)


def _fetch_web_bundle(*, queries: list[str], intent: str, reason: str) -> tuple[list[dict[str, str]], int]:
    start = time.time()
    plan = SearchPlan(
        should_search=True,
        intent=intent,
        confidence=1.0,
        entities={},
        queries=queries,
        reason=reason,
    )
    items = [
        {"title": result.title, "snippet": result.snippet, "url": result.url}
        for result in search_from_plan(plan, limit=4)
    ]
    return items, int((time.time() - start) * 1000)


def _planner_queries(company: str) -> list[str]:
    cleaned = " ".join((company or "").split()).strip()
    return [
        f"{cleaned} recruitment process rounds",
        f"{cleaned} previous interview questions",
        f"{cleaned} aptitude technical hr questions",
        f"{cleaned} placement preparation topics",
        f"{cleaned} role interview experience",
    ]


def _planner_memory_context(*, user_id: str) -> list[str]:
    with get_session() as session:
        records = graph_memory_service.fetch_graph_memory(user_id=user_id, session=session, limit=8)
    lines: list[str] = []
    for item in records:
        entity = str(item.get("entity") or "").strip()
        relation = str(item.get("relation") or "").strip()
        entity_type = str(item.get("entity_type") or "").strip()
        if entity and relation:
            lines.append(f"{relation} -> {entity} ({entity_type})")
    profile = fetch_profile_summary(user_id=user_id, limit=4)
    for item in list(profile.get("strengths") or [])[:3]:
        lines.append(f"STRENGTH_PROFILE -> {item['entity']} ({item['entity_type']}, score {float(item['score']):.2f})")
    for item in list(profile.get("weaknesses") or [])[:3]:
        lines.append(f"WEAKNESS_PROFILE -> {item['entity']} ({item['entity_type']}, score {float(item['score']):.2f})")
    for item in list(profile.get("improving") or [])[:2]:
        lines.append(f"IMPROVING_PROFILE -> {item['entity']} ({item['entity_type']}, improving {float(item['improving_score']):.2f})")
    return lines


def _profile_signal_queries(
    *,
    message: str,
    observations: list[dict[str, object]],
    triples,
) -> list[str]:
    lowered = f" {' '.join((message or '').lower().split())} "
    if " i " not in lowered and " my " not in lowered and " me " not in lowered:
        return []
    entities: list[str] = []
    for item in list(observations or [])[:5]:
        entity = " ".join(str(item.get("entity") or "").split()).strip()
        if entity:
            entities.append(entity)
    if not entities:
        for triple in list(triples or [])[:8]:
            if getattr(triple, "subject_type", "").strip().lower() != "user":
                continue
            entity = " ".join(str(getattr(triple, "object_name", "") or "").split()).strip()
            if entity:
                entities.append(entity)
    deduped: list[str] = []
    seen: set[str] = set()
    for entity in entities:
        key = entity.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(entity)
    queries: list[str] = []
    for entity in deduped[:3]:
        queries.append(f"{entity} interview preparation important topics")
        queries.append(f"{entity} common interview questions concepts")
    return queries[:4]


def _profile_signal_web_facts(
    *,
    message: str,
    observations: list[dict[str, object]],
    triples,
) -> list[str]:
    queries = _profile_signal_queries(message=message, observations=observations, triples=triples)
    if not queries:
        return []
    plan = SearchPlan(
        should_search=True,
        intent="general_learning",
        confidence=0.78,
        entities={},
        queries=queries,
        reason="profile signal extraction context",
    )
    return [
        f"{result.title}: {result.snippet} ({result.url})"
        for result in search_from_plan(plan, limit=4)
        if result.title
    ]


def _process_memory_pipeline(
    *,
    user_id: str,
    conversation_id: str,
    message: str,
    source: str,
    source_event_id: str,
    created_at: str,
    inferred_raw_signals: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    extracted_triples = extract_triple_candidates(
        user_id=user_id,
        message=message,
        source=source,
    )
    base_profile_observations = analyze_strength_weakness_profile(
        message=message,
        triples=extracted_triples,
    )
    profile_web_facts = _profile_signal_web_facts(
        message=message,
        observations=base_profile_observations,
        triples=extracted_triples,
    )
    profile_observations = analyze_strength_weakness_profile(
        message=message,
        triples=extracted_triples,
        web_facts=profile_web_facts,
        seed_observations=base_profile_observations,
    ) or base_profile_observations
    if profile_observations:
        upsert_profile_observations(
            user_id=user_id,
            observations=profile_observations,
        )
    profile_summary = fetch_profile_summary(user_id=user_id, limit=5)
    extracted_raw_signals = [
        {
            "user_id": triple.user_id,
            "entity": triple.object_name,
            "entity_type": triple.object_type,
            "relation": triple.relation,
            "confidence": triple.confidence,
            "linked_to_action": triple.linked_to_action,
            "source": triple.source,
            "raw_text": triple.raw_text,
            "source_event_id": source_event_id,
        }
        for triple in extracted_triples
        if triple.subject_type.strip().lower() == "user"
    ]
    all_raw_signals = [*extracted_raw_signals, *(inferred_raw_signals or [])]

    with get_session() as session:
        promotion_summary = {"ephemeral_count": 0, "promoted_count": 0, "promoted_items": []}
        if extracted_triples:
            promotion_summary = graph_memory_service.process_triples(
                session=session,
                triples=extracted_triples,
            )
        if inferred_raw_signals:
            inferred_summary = graph_memory_service.process_signals(
                session=session,
                raw_signals=inferred_raw_signals,
            )
            promotion_summary = {
                "ephemeral_count": int(promotion_summary.get("ephemeral_count") or 0) + int(inferred_summary.get("ephemeral_count") or 0),
                "promoted_count": int(promotion_summary.get("promoted_count") or 0) + int(inferred_summary.get("promoted_count") or 0),
                "promoted_items": [
                    *list(promotion_summary.get("promoted_items") or []),
                    *list(inferred_summary.get("promoted_items") or []),
                ],
            }
        if int(promotion_summary.get("promoted_count") or 0) > 0:
            topic_semantic_router.refresh_from_session(session)

    log_promotions(
        user_id=user_id,
        source_event_id=source_event_id,
        created_at=created_at,
        raw_signals=all_raw_signals,
        summary=promotion_summary,
    )

    for signal in all_raw_signals:
        relation = str(signal.get("relation") or "").strip()
        entity_type = str(signal.get("entity_type") or "").strip()
        semantics = classify_relation_semantics(relation, entity_type=entity_type)
        if not should_background_enrich(semantics):
            continue
        enriched = classify_relation_with_llm(relation=relation, entity_type=entity_type)
        if not enriched:
            continue
        store_llm_relation_semantics(
            relation=relation,
            entity_type=entity_type,
            family=str(enriched.get("family") or "general"),
            polarity=str(enriched.get("polarity") or "neutral"),
            section_tags=[str(tag) for tag in list(enriched.get("section_tags") or [])],
            strength=float(enriched.get("strength") or 0.5),
        )

    return {
        "signals_extracted": len(extracted_raw_signals),
        "promotion_summary": promotion_summary,
        "profile_summary": profile_summary,
        "profile_web_facts": profile_web_facts,
    }


@app.get("/graph/memory/{user_id}")
def get_graph_memory(
    user_id: str,
    limit: int = 10,
    graphmind_session: str | None = Cookie(default=None, alias=SESSION_COOKIE_NAME),
) -> dict[str, object]:
    resolved_user_id = _resolve_user_id(user_id, get_user_by_session_token(graphmind_session))
    with get_session() as session:
        items = graph_memory_service.fetch_graph_memory(user_id=resolved_user_id, session=session, limit=limit)
    return {"user_id": resolved_user_id, "items": items}


@app.get("/graph/view/{user_id}")
def get_graph_view(
    user_id: str,
    limit: int = 24,
    graphmind_session: str | None = Cookie(default=None, alias=SESSION_COOKIE_NAME),
) -> dict[str, object]:
    resolved_user_id = _resolve_user_id(user_id, get_user_by_session_token(graphmind_session))
    with get_session() as session:
        graph = graph_memory_service.fetch_graph_view(user_id=resolved_user_id, session=session, limit=limit)
    return {"user_id": resolved_user_id, **graph}


@app.get("/memory/ephemeral/{user_id}")
def get_ephemeral_memory(
    user_id: str,
    limit: int = 20,
    graphmind_session: str | None = Cookie(default=None, alias=SESSION_COOKIE_NAME),
) -> dict[str, object]:
    resolved_user_id = _resolve_user_id(user_id, get_user_by_session_token(graphmind_session))
    items = graph_memory_service.fetch_ephemeral_memory(user_id=resolved_user_id, limit=limit)
    return {
        "user_id": resolved_user_id,
        "backend": graph_memory_service.ephemeral_backend,
        "items": items,
    }


@app.get("/profile/summary/{user_id}")
def get_profile_summary(
    user_id: str,
    graphmind_session: str | None = Cookie(default=None, alias=SESSION_COOKIE_NAME),
) -> dict[str, object]:
    resolved_user_id = _resolve_user_id(user_id, get_user_by_session_token(graphmind_session))
    return {
        "user_id": resolved_user_id,
        "profile": fetch_profile_summary(user_id=resolved_user_id, limit=8),
    }


@app.get("/events/{user_id}")
def get_recent_events(
    user_id: str,
    limit: int = 20,
    graphmind_session: str | None = Cookie(default=None, alias=SESSION_COOKIE_NAME),
) -> dict[str, object]:
    resolved_user_id = _resolve_user_id(user_id, get_user_by_session_token(graphmind_session))
    return {"user_id": resolved_user_id, "items": recent_raw_events(user_id=resolved_user_id, limit=limit)}


@app.delete("/memory/reset/{user_id}")
def reset_user_memory(
    user_id: str,
    graphmind_session: str | None = Cookie(default=None, alias=SESSION_COOKIE_NAME),
) -> dict[str, object]:
    resolved_user_id = _resolve_user_id(user_id, get_user_by_session_token(graphmind_session))
    with get_session() as session:
        graph_summary = graph_memory_service.reset_user_memory(session=session, user_id=resolved_user_id)
    event_summary = delete_user_events(user_id=resolved_user_id)
    deleted_messages = delete_user_messages(user_id=resolved_user_id)
    profile_summary = delete_user_profile(user_id=resolved_user_id)
    return {
        "user_id": resolved_user_id,
        "graph": graph_summary,
        "events": event_summary,
        "vector": {"deleted_messages": deleted_messages},
        "profile": profile_summary,
    }


@app.post("/search")
def search(
    req: SearchRequest,
    graphmind_session: str | None = Cookie(default=None, alias=SESSION_COOKIE_NAME),
) -> dict[str, object]:
    resolved_user_id = _resolve_user_id(req.user_id, get_user_by_session_token(graphmind_session))
    results = vector_search(
        query=req.query,
        user_id=resolved_user_id,
        conversation_id=req.conversation_id,
        k=req.k,
    )
    return {"results": results}


@app.get("/chat/history/{conversation_id}")
def chat_history(
    conversation_id: str,
    limit: int = 100,
    graphmind_session: str | None = Cookie(default=None, alias=SESSION_COOKIE_NAME),
) -> dict[str, object]:
    current_user = get_user_by_session_token(graphmind_session)
    resolved_user_id = _resolve_user_id(None, current_user)
    items = get_chat_history(
        conversation_id=conversation_id,
        user_id=resolved_user_id,
        limit=limit,
    )
    return {
        "conversation_id": conversation_id,
        "user_id": resolved_user_id,
        "messages": items,
    }


@app.get("/chat/conversations")
def chat_conversations(
    limit: int = 50,
    graphmind_session: str | None = Cookie(default=None, alias=SESSION_COOKIE_NAME),
) -> dict[str, object]:
    current_user = get_user_by_session_token(graphmind_session)
    resolved_user_id = _resolve_user_id(None, current_user)
    items = list_conversations(user_id=resolved_user_id, limit=limit)
    return {"user_id": resolved_user_id, "items": items}


@app.post("/memory/signals")
def ingest_memory_signals(
    req: MemoryIngestRequest,
    graphmind_session: str | None = Cookie(default=None, alias=SESSION_COOKIE_NAME),
) -> dict[str, object]:
    resolved_user_id = _resolve_user_id(req.user_id, get_user_by_session_token(graphmind_session))
    created_at = datetime.now(timezone.utc).isoformat()
    source_event_id = log_raw_event(
        user_id=resolved_user_id,
        conversation_id=None,
        source_type=req.source,
        source_ref="api:memory_signals",
        role="system",
        content=f"memory_signals:{len(req.signals)}",
        metadata={"signal_count": len(req.signals)},
        created_at=created_at,
    )
    raw_signals = [
        {
            "user_id": resolved_user_id,
            "entity": signal.entity,
            "relation": signal.relation,
            "confidence": signal.confidence,
            "entity_type": signal.entity_type,
            "linked_to_action": signal.linked_to_action,
            "source": req.source,
            "raw_text": signal.raw_text or signal.entity,
        }
        for signal in req.signals
    ]

    with get_session() as session:
        summary = graph_memory_service.process_signals(
            session=session,
            raw_signals=raw_signals,
        )
    log_promotions(
        user_id=resolved_user_id,
        source_event_id=source_event_id,
        created_at=created_at,
        raw_signals=raw_signals,
        summary=summary,
    )

    return {
        "user_id": resolved_user_id,
        "source": req.source,
        "summary": summary,
    }


@app.post("/planner/company")
def company_planner(
    req: CompanyPlannerRequest,
    graphmind_session: str | None = Cookie(default=None, alias=SESSION_COOKIE_NAME),
) -> dict[str, object]:
    resolved_user_id = _resolve_user_id(req.user_id, get_user_by_session_token(graphmind_session))
    company = " ".join(req.company.split()).strip()
    days_left = int(req.days_left or 14)
    days_left = max(1, min(days_left, 60))
    if not company:
        raise HTTPException(status_code=400, detail="Company is required.")

    planner_queries = _planner_queries(company)
    plan = SearchPlan(
        should_search=True,
        intent="prep_guidance",
        confidence=1.0,
        entities={"entity": company, "entity_type": "organization", "role": "software engineer", "topic": ""},
        queries=planner_queries,
        reason="company planner research",
    )
    web_results = [
        {"title": result.title, "snippet": result.snippet, "url": result.url}
        for result in search_from_plan(plan, limit=6)
    ]
    memory_facts = _planner_memory_context(user_id=resolved_user_id)
    profile_summary = fetch_profile_summary(user_id=resolved_user_id, limit=5)
    planner = generate_company_planner(
        company=company,
        days_left=days_left,
        web_results=web_results,
        memory_facts=memory_facts,
        profile_summary=profile_summary,
    )
    return {
        "user_id": resolved_user_id,
        "company": company,
        "days_left": days_left,
        "planner": planner,
        "sources": web_results,
        "memory_facts": memory_facts,
        "profile_summary": profile_summary,
    }


@app.post("/chat")
def chat(
    req: ChatRequest,
    background_tasks: BackgroundTasks,
    graphmind_session: str | None = Cookie(default=None, alias=SESSION_COOKIE_NAME),
) -> dict[str, object]:
    resolved_user_id = _resolve_user_id(req.user_id, get_user_by_session_token(graphmind_session))
    with get_session() as session:
        _ensure_user_node(session, user_id=resolved_user_id)
    start = time.time()
    conversation_id = req.conversation_id or f"convo-{uuid4().hex[:10]}"
    ensure_conversation(conversation_id=conversation_id, user_id=resolved_user_id)
    now_iso = datetime.now(timezone.utc).isoformat()
    source_event_id = log_raw_event(
        user_id=resolved_user_id,
        conversation_id=conversation_id,
        source_type=req.source,
        source_ref="chat:user_message",
        role="user",
        content=req.message,
        metadata={"conversation_id": conversation_id},
        created_at=now_iso,
    )

    user_msg_id = str(uuid4())

    add_message(
        message_id=user_msg_id,
        text=req.message,
        metadata={
            "user_id": resolved_user_id,
            "conversation_id": conversation_id,
            "role": "user",
            "created_at": now_iso,
        },
    )
    save_chat_message(
        conversation_id=conversation_id,
        user_id=resolved_user_id,
        role="user",
        content=req.message,
    )
    recent_history = get_chat_history(
        conversation_id=conversation_id,
        user_id=resolved_user_id,
        limit=10,
    )

    topic_match = topic_semantic_router.detect(req.message)
    route_decision = route_prompt(
        req.message,
        semantic_topic=topic_match.topic if topic_match else None,
    )
    search_plan = build_search_plan(
        message=req.message,
        semantic_topic=topic_match.topic if topic_match else None,
        route_intent=route_decision.intent,
    )
    section_plan = resolve_sections(
        message=req.message,
        route_intent=route_decision.intent,
        semantic_topic=topic_match.topic if topic_match else None,
        target_entity=str(search_plan.entities.get("entity") or "").strip() or None,
    )
    graph_evidence: dict[str, object] = {"facts": [], "paths": [], "citations": []}
    graph_context: list[str] = []
    graph_retrieval_time_ms = 0
    retrieved: list[dict[str, object]] = []
    retrieval_time_ms = 0
    retrieval_mode = "skipped"
    web_results: list[dict[str, str]] = []
    web_retrieval_time_ms = 0
    inferred_memory_signals = infer_memory_signals_from_plan(
        message=req.message,
        user_id=resolved_user_id,
        plan=search_plan,
    )

    # memory-first retrieval pipeline (graph + vector)
    graph_future = RETRIEVAL_EXECUTOR.submit(
        _fetch_graph_bundle,
        user_id=resolved_user_id,
        query=req.message,
        section_tags=section_plan.query_tags(),
        section_families=section_plan.query_families(),
        focus_entity=section_plan.focus_entity,
    )
    vector_future = RETRIEVAL_EXECUTOR.submit(
        _fetch_vector_bundle,
        query=req.message,
        user_id=resolved_user_id,
        conversation_id=conversation_id,
        k=5,
    )

    effective_allow_web_search = bool(req.allow_web_search or _message_requests_web(req.message))

    web_future = None
    web_search_used = False
    if effective_allow_web_search and search_plan.should_search and search_plan.queries:
        web_future = RETRIEVAL_EXECUTOR.submit(
            _fetch_web_bundle,
            queries=search_plan.queries,
            intent=search_plan.intent,
            reason=search_plan.reason,
        )

    if graph_future is not None:
        graph_evidence, graph_context, graph_retrieval_time_ms = graph_future.result()

    graph_max_score = max(
        (float(item.get("score") or 0.0) for item in list(graph_evidence.get("facts") or [])),
        default=0.0,
    )

    if vector_future is not None:
        retrieved, retrieval_time_ms = vector_future.result()

    if web_future is not None:
        web_results, web_retrieval_time_ms = web_future.result()
        web_search_used = bool(web_results)

    snippets: list[str] = []
    relevant_retrieved_count = 0
    for result in retrieved:
        score = float(result.get("score") or 0.0)
        if score < 0.5:
            continue
        text = _compress_snippet((result.get("text") or "").strip())
        if text and text != req.message:
            snippets.append(text)
            relevant_retrieved_count += 1
    web_facts = [
        _compress_snippet(f"{item.get('title')}: {item.get('snippet')} ({item.get('url')})", limit=220)
        for item in web_results
        if item.get("title")
    ]

    if graph_max_score < 0.45:
        graph_context = []
        graph_evidence = {"facts": [], "paths": [], "citations": []}

    relevant_graph_hit = graph_max_score >= 0.45
    relevant_vector_hit = relevant_retrieved_count > 0
    memory_hit = bool(relevant_graph_hit or relevant_vector_hit)

    llm_start = time.time()
    reply_bundle = generate_reply_bundle(
        user_message=req.message,
        retrieved_snippets=snippets,
        recent_history=recent_history,
        graph_facts=graph_context,
        evidence_paths=list(graph_evidence.get("paths") or []),
        web_facts=web_facts if web_search_used else None,
        memory_found=memory_hit,
    )
    llm_generation_time_ms = int((time.time() - llm_start) * 1000)
    answer = reply_bundle["text"]
    if memory_hit:
        retrieval_mode = "memory_hit"
    else:
        retrieval_mode = "direct_reply"
    if web_search_used:
        retrieval_mode = "memory_plus_web"
    llm_provider = reply_bundle.get("provider", "unknown")
    llm_model = reply_bundle.get("model", "unknown")

    memory_pipeline_started = time.time()
    memory_pipeline_result = _process_memory_pipeline(
        user_id=resolved_user_id,
        conversation_id=conversation_id,
        message=req.message,
        source=req.source,
        source_event_id=source_event_id,
        created_at=now_iso,
        inferred_raw_signals=inferred_memory_signals,
    )
    graph_promotion_time_ms = int((time.time() - memory_pipeline_started) * 1000)

    ephemeral_memory = graph_memory_service.fetch_ephemeral_memory(
        user_id=resolved_user_id,
        limit=10,
    )

    with get_session() as session:
        graph_memory = graph_memory_service.fetch_graph_memory(
            user_id=resolved_user_id,
            session=session,
            limit=10,
        )
    bot_msg_id = str(uuid4())

    add_message(
        message_id=bot_msg_id,
        text=answer,
        metadata={
            "user_id": resolved_user_id,
            "conversation_id": conversation_id,
            "role": "assistant",
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    save_chat_message(
        conversation_id=conversation_id,
        user_id=resolved_user_id,
        role="assistant",
        content=answer,
    )

    log_raw_event(
        user_id=resolved_user_id,
        conversation_id=conversation_id,
        source_type="assistant_reply",
        source_ref="chat:assistant_reply",
        role="assistant",
        content=answer,
        metadata={
            "conversation_id": conversation_id,
            "llm_provider": llm_provider,
            "llm_model": llm_model,
            "graph_paths": list(graph_evidence.get("paths") or []),
        },
        created_at=datetime.now(timezone.utc).isoformat(),
    )

    return {
        "user_id": resolved_user_id,
        "conversation_id": conversation_id,
        "answer": answer,
        "retrieved_count": len(snippets),
        "retrieval_time_ms": retrieval_time_ms,
        "graph_retrieval_time_ms": graph_retrieval_time_ms,
        "web_retrieval_time_ms": web_retrieval_time_ms,
        "retrieval_mode": retrieval_mode,
        "memory_found": memory_hit,
        "web_search_used": web_search_used,
        "suggest_web_search": (not memory_hit and not effective_allow_web_search and bool(search_plan.queries)),
        "graph_confidence": round(graph_max_score, 4),
        "graph_promotion_time_ms": graph_promotion_time_ms,
        "signal_extraction_time_ms": 0,
        "signals_extracted": int(memory_pipeline_result.get("signals_extracted") or 0),
        "promotion_summary": memory_pipeline_result.get("promotion_summary") or {"status": "completed"},
        "route": route_decision.to_dict(),
        "topic_match": (
            {
                "topic": topic_match.topic,
                "score": topic_match.score,
                "source": topic_match.source,
            }
            if topic_match
            else None
        ),
        "search_plan": search_plan.to_dict(),
        "section_plan": {
            "sections": section_plan.sections,
            "focus_entity": section_plan.focus_entity,
            "tags": section_plan.query_tags(),
            "families": section_plan.query_families(),
        },
        "web_results": web_results,
        "graph_evidence": graph_evidence,
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "llm_generation_time_ms": llm_generation_time_ms,
        "memory_update_mode": "inline",
        "ephemeral_backend": graph_memory_service.ephemeral_backend,
        "ephemeral_memory": ephemeral_memory,
        "graph_memory": graph_memory,
        "time_ms": int((time.time() - start) * 1000),
    }
