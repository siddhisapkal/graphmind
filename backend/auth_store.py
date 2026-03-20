from __future__ import annotations

import hashlib
import hmac
import secrets
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4


SESSION_TTL_DAYS = 30
PBKDF2_ITERATIONS = 120_000


@dataclass
class AuthUser:
    user_id: str
    username: str
    created_at: str


def _db_path() -> Path:
    return Path(__file__).resolve().parent / "users.sqlite3"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_db_path()))
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            password_salt TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS user_sessions (
            token_hash TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_user_sessions_user ON user_sessions(user_id)"
    )
    return conn


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_username(username: str) -> str:
    return " ".join((username or "").strip().split()).lower()


def _hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        PBKDF2_ITERATIONS,
    ).hex()


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def register_user(*, username: str, password: str) -> AuthUser:
    normalized_username = _normalize_username(username)
    if len(normalized_username) < 3:
        raise ValueError("Username must be at least 3 characters long.")
    if len(password or "") < 8:
        raise ValueError("Password must be at least 8 characters long.")

    user = AuthUser(
        user_id=f"user-{uuid4().hex[:12]}",
        username=normalized_username,
        created_at=_now_iso(),
    )
    salt = secrets.token_hex(16)
    password_hash = _hash_password(password, salt)

    conn = _connect()
    try:
        existing = conn.execute(
            "SELECT id FROM users WHERE username = ?",
            (normalized_username,),
        ).fetchone()
        if existing is not None:
            raise ValueError("Username already exists.")
        conn.execute(
            """
            INSERT INTO users (id, username, password_hash, password_salt, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (user.user_id, user.username, password_hash, salt, user.created_at),
        )
        conn.commit()
    finally:
        conn.close()
    return user


def authenticate_user(*, username: str, password: str) -> AuthUser | None:
    normalized_username = _normalize_username(username)
    conn = _connect()
    try:
        row = conn.execute(
            """
            SELECT id, username, password_hash, password_salt, created_at
            FROM users
            WHERE username = ?
            """,
            (normalized_username,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    expected_hash = str(row["password_hash"] or "")
    actual_hash = _hash_password(password, str(row["password_salt"] or ""))
    if not hmac.compare_digest(expected_hash, actual_hash):
        return None
    return AuthUser(
        user_id=str(row["id"]),
        username=str(row["username"]),
        created_at=str(row["created_at"]),
    )


def create_session(*, user_id: str) -> tuple[str, str]:
    token = secrets.token_urlsafe(32)
    token_hash = _hash_token(token)
    created_at = datetime.now(timezone.utc)
    expires_at = created_at + timedelta(days=SESSION_TTL_DAYS)

    conn = _connect()
    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO user_sessions (token_hash, user_id, created_at, expires_at)
            VALUES (?, ?, ?, ?)
            """,
            (token_hash, user_id, created_at.isoformat(), expires_at.isoformat()),
        )
        conn.commit()
    finally:
        conn.close()
    return token, expires_at.isoformat()


def get_user_by_session_token(token: str | None) -> AuthUser | None:
    if not token:
        return None
    token_hash = _hash_token(token)
    now_iso = _now_iso()

    conn = _connect()
    try:
        row = conn.execute(
            """
            SELECT u.id, u.username, u.created_at, s.expires_at
            FROM user_sessions s
            JOIN users u ON u.id = s.user_id
            WHERE s.token_hash = ?
            """,
            (token_hash,),
        ).fetchone()
        if row is None:
            return None
        if str(row["expires_at"]) <= now_iso:
            conn.execute("DELETE FROM user_sessions WHERE token_hash = ?", (token_hash,))
            conn.commit()
            return None
        return AuthUser(
            user_id=str(row["id"]),
            username=str(row["username"]),
            created_at=str(row["created_at"]),
        )
    finally:
        conn.close()


def delete_session(token: str | None) -> None:
    if not token:
        return
    conn = _connect()
    try:
        conn.execute(
            "DELETE FROM user_sessions WHERE token_hash = ?",
            (_hash_token(token),),
        )
        conn.commit()
    finally:
        conn.close()
