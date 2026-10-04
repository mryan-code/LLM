"""PostgreSQL integration for global and user training layers."""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from typing import Any

from storm_zero_llm.config import DatabaseConfig
from storm_zero_llm.memory import MemoryRecord, MemoryType

_CONVERSATION_HISTORY_LIMIT = 75
_CONVERSATION_SUBJECTS_CONTENT_LIMIT = 20
_BASE_OUTPUT_TOKEN_RESERVE = 256


def estimate_token_count(text: str) -> int:
    """Conservative token estimate for packing context without loading a tokenizer."""
    stripped = text.strip()
    if not stripped:
        return 0
    return max(1, (len(stripped) + 3) // 4)


def format_conversation_line(prompt: str, response: str) -> str:
    return f"- {prompt} = {response}"


def pack_conversation_history(
    turns: list[dict[str, str]],
    *,
    user_prompt: str,
    rules_block: str,
    n_ctx: int,
    output_reserve: int = _BASE_OUTPUT_TOKEN_RESERVE,
) -> str:
    """Pack newest-first turns into the remaining context budget; return chronological text."""
    budget = max(
        0,
        int(n_ctx) - int(output_reserve) - estimate_token_count(user_prompt) - estimate_token_count(rules_block),
    )
    if budget <= 0 or not turns:
        return ""

    selected: list[str] = []
    used = 0
    for turn in turns:
        line = format_conversation_line(turn.get("prompt", ""), turn.get("response", ""))
        cost = estimate_token_count(line) + 1
        if used + cost > budget:
            break
        selected.append(line)
        used += cost

    # `turns` are newest-first; reverse so the prompt reads oldest -> newest.
    return "\n".join(reversed(selected))


@dataclass(frozen=True)
class RuntimeRuleContext:
    global_hard_rules: list[str]
    global_guidelines: list[str]
    user_guidelines: list[str]
    user_p2_data: list[str]
    user_avatar_data: list[str]

    def section_data(self) -> dict[str, list[str]]:
        return {
            "global_hard_rules": list(self.global_hard_rules),
            "global_guidelines": list(self.global_guidelines),
            "user_guidelines": list(self.user_guidelines),
            "user_p2_data": list(self.user_p2_data),
            "user_avatar_data": list(self.user_avatar_data),
        }

    def to_prompt_block(self, *, template: str = "") -> str:
        from storm_zero_llm.prompts import _FALLBACK_BASE_HEADERS, fill_structured_prompt_template

        return fill_structured_prompt_template(
            template,
            self.section_data(),
            fallback_headers=_FALLBACK_BASE_HEADERS,
        )

    def to_image_prompt_block(self, *, template: str = "") -> str:
        """Profile context for image generation (p2 + avatar only)."""
        from storm_zero_llm.prompts import _FALLBACK_IMAGE_HEADERS, fill_structured_prompt_template

        return fill_structured_prompt_template(
            template,
            {
                "user_p2_data": list(self.user_p2_data),
                "user_avatar_data": list(self.user_avatar_data),
            },
            fallback_headers=_FALLBACK_IMAGE_HEADERS,
        )

    def counts(self) -> dict[str, int]:
        return {
            "global_hard_rules": len(self.global_hard_rules),
            "global_guidelines": len(self.global_guidelines),
            "user_guidelines": len(self.user_guidelines),
            "user_p2_data": len(self.user_p2_data),
            "user_avatar_data": len(self.user_avatar_data),
        }


class _LoggingCursor:
    def __init__(self, cursor: Any, log: list[dict[str, Any]]):
        self._cursor = cursor
        self._log = log

    def __enter__(self) -> "_LoggingCursor":
        self._cursor.__enter__()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self._cursor.__exit__(exc_type, exc, tb)

    def execute(self, query: str, params: tuple[object, ...] | None = None) -> Any:
        if params is None:
            params = ()
        self._log.append({"sql": query, "params": list(params)})
        return self._cursor.execute(query, params)

    def fetchall(self) -> list[dict[str, Any]]:
        return self._cursor.fetchall()

    def fetchone(self) -> dict[str, Any] | None:
        return self._cursor.fetchone()


class PostgresTrainingRepository:
    """
    PostgreSQL access for training/runtime context.

    `ThreadingHTTPServer` serves each request on its own thread. psycopg connections
    are not thread-safe, so production instances created via `from_config` keep a
    separate connection per thread. Tests may still pass a shared fake connection.
    """

    def __init__(self, connection: Any | None = None, *, config: DatabaseConfig | None = None):
        if connection is None and config is None:
            raise ValueError("PostgresTrainingRepository requires a connection or config")
        self._shared_connection = connection
        self._config = config
        self._thread_state = threading.local()
        self._request_log = threading.local()

    @property
    def connection(self) -> Any:
        if self._shared_connection is not None:
            return self._shared_connection
        conn = getattr(self._thread_state, "connection", None)
        if conn is None:
            conn = self._open_connection()
            self._thread_state.connection = conn
            return conn
        try:
            # psycopg has no ping(); a cheap query validates the connection.
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        except Exception:
            try:
                conn.close()
            except Exception:
                pass
            conn = self._open_connection()
            self._thread_state.connection = conn
        return conn

    def begin_request_queries(self) -> None:
        self._request_log.queries = []

    def get_request_queries(self) -> list[dict[str, Any]]:
        queries = getattr(self._request_log, "queries", None)
        if queries is None:
            return []
        return list(queries)

    def _cursor(self) -> Any:
        log = getattr(self._request_log, "queries", None)
        cursor = self.connection.cursor()
        if log is None:
            return cursor
        return _LoggingCursor(cursor, log)

    def _open_connection(self) -> Any:
        if self._config is None:
            raise RuntimeError("Cannot open PostgreSQL connection without database config")
        try:
            import psycopg  # type: ignore
            from psycopg.rows import dict_row  # type: ignore
        except ImportError as exc:
            raise RuntimeError("psycopg must be installed to use PostgreSQL training sync") from exc

        return psycopg.connect(
            host=self._config.host,
            port=self._config.port,
            user=self._config.user,
            password=self._config.password,
            dbname=self._config.name,
            row_factory=dict_row,
            autocommit=True,
        )

    @classmethod
    def from_config(cls, config: DatabaseConfig) -> "PostgresTrainingRepository":
        return cls(config=config)

    def close(self) -> None:
        if self._shared_connection is not None:
            self._shared_connection.close()
            self._shared_connection = None
            return
        conn = getattr(self._thread_state, "connection", None)
        if conn is not None:
            try:
                conn.close()
            finally:
                self._thread_state.connection = None

    def load_global_memories(self) -> list[MemoryRecord]:
        rows: list[MemoryRecord] = []
        rows.extend(self._load_global_rules(strict_value=1, memory_type=MemoryType.HARD_RULE, priority="critical"))
        rows.extend(self._load_global_rules(strict_value=0, memory_type=MemoryType.GUIDE, priority="normal"))
        return rows

    def _load_global_rules(self, strict_value: int, memory_type: MemoryType, priority: str) -> list[MemoryRecord]:
        db_rows = self._fetch_global_rows(strict_value)

        memories: list[MemoryRecord] = []
        for row in db_rows:
            title = _pick_first_text(row, ["title", "name", "rule_name", "guide_name", "label"]) or f"{memory_type.value}_{row.get('id', 'unknown')}"
            content = _pick_first_text(row, ["rule", "description", "content"]) or ""
            if not content:
                continue
            topics = _split_topics(row.get("topics"))
            memories.append(
                MemoryRecord(
                    memory_type=memory_type,
                    scope="global",
                    title=title,
                    content=content,
                    topics=topics,
                    priority=priority,
                    source="tblglobal_rule",
                )
            )
        return memories

    def _fetch_global_rows(self, strict_value: int) -> list[dict[str, Any]]:
        """Load rows from global rules table without assuming column names."""

        query = "SELECT * FROM tblglobal_rule WHERE strict = %s AND deleted = 0"
        with self._cursor() as cursor:
            cursor.execute(query, (strict_value,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def strict_rule_blocks_guideline(self, guideline: str) -> bool:
        rules = self._fetch_global_rows(strict_value=1)
        for row in rules:
            rule_text = _pick_first_text(row, ["rule", "description", "content"]) or ""
            if _rule_blocks_guideline(rule_text, guideline):
                return True
        return False

    def load_runtime_rule_context(self, user_id: int) -> RuntimeRuleContext:
        hard_rows = self._fetch_global_rows(strict_value=1)
        guide_rows = self._fetch_global_rows(strict_value=0)
        user_rows = self._fetch_user_guideline_rows(user_id=user_id)
        user_p2_rows = self._fetch_user_p2_rows(user_id=user_id)
        user_avatar_rows = self._fetch_user_avatar_rows(user_id=user_id)

        hard_rules = _expand_multiline_rules(
            _pick_first_text(row, ["rule", "description", "content"]) or ""
            for row in hard_rows
        )
        global_guidelines = _expand_multiline_rules(
            _pick_first_text(row, ["rule", "description", "content"]) or ""
            for row in guide_rows
        )
        user_guidelines = [
            _pick_first_text(row, ["guideline", "rule", "description", "content"]) or ""
            for row in user_rows
        ]
        user_p2_data = _build_p2_data_from_rows(user_p2_rows)
        user_avatar_data = _build_avatar_data_from_rows(user_avatar_rows)

        context = RuntimeRuleContext(
            global_hard_rules=hard_rules,
            global_guidelines=global_guidelines,
            user_guidelines=[row.strip() for row in user_guidelines if row.strip()],
            user_p2_data=user_p2_data,
            user_avatar_data=[row.strip() for row in user_avatar_data if row.strip()],
        )
        return context

    def _fetch_user_guideline_rows(self, user_id: int) -> list[dict[str, Any]]:
        query = "SELECT * FROM tbluser_guideline WHERE user_id = %s AND deleted = 0"
        with self._cursor() as cursor:
            cursor.execute(query, (user_id,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def _fetch_user_p2_rows(self, user_id: int) -> list[dict[str, Any]]:
        query = "SELECT `key`, `value` FROM tbluser_p2 WHERE user_id = %s AND deleted = 0"
        with self._cursor() as cursor:
            cursor.execute(query, (user_id,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def _fetch_user_avatar_rows(self, user_id: int) -> list[dict[str, Any]]:
        query = "SELECT * FROM tbluser_avatar WHERE user_id = %s AND deleted = 0 ORDER BY id DESC"
        with self._cursor() as cursor:
            cursor.execute(query, (user_id,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def fetch_recent_conversation_turns(
        self,
        user_id: int,
        limit: int = _CONVERSATION_HISTORY_LIMIT,
    ) -> list[dict[str, str]]:
        user_id = _require_user_id(user_id)
        limit = max(0, int(limit))
        query = (
            "SELECT `prompt`, `response` FROM tbluser_conversation_content "
            "WHERE user_id = %s AND deleted = 0 ORDER BY created DESC LIMIT %s"
        )
        with self._cursor() as cursor:
            cursor.execute(query, (user_id, limit))
            rows = [dict(row) for row in cursor.fetchall()]
        turns: list[dict[str, str]] = []
        for row in rows:
            prompt = str(row.get("prompt") or "").strip()
            response = str(row.get("response") or "").strip()
            if not prompt and not response:
                continue
            turns.append({"prompt": prompt, "response": response})
        return turns

    def fetch_conversation_subjects_content(
        self,
        user_id: int,
        limit: int | None = _CONVERSATION_SUBJECTS_CONTENT_LIMIT,
        *,
        subject_id: int | None = None,
    ) -> list[dict[str, Any]]:
        user_id = _require_user_id(user_id)
        params: list[object] = [user_id]
        query = (
            "SELECT `created`, `prompt`, `response` FROM tbluser_conversation_content "
            "WHERE user_id = %s AND deleted = 0"
        )
        if subject_id is not None:
            query += " AND user_conversation_subject_id = %s"
            params.append(int(subject_id))
        query += " ORDER BY created DESC"
        if limit is not None:
            query += " LIMIT %s"
            params.append(max(0, int(limit)))
        with self._cursor() as cursor:
            cursor.execute(query, tuple(params))
            rows = [dict(row) for row in cursor.fetchall()]
        content: list[dict[str, Any]] = []
        for row in rows:
            created = row.get("created")
            content.append(
                {
                    "created": str(created) if created is not None else "",
                    "prompt": str(row.get("prompt") or "").strip(),
                    "response": str(row.get("response") or "").strip(),
                }
            )
        return content

    def get_structured_avatar_data(self, user_id: int) -> dict[str, Any]:
        rows = self._fetch_user_avatar_rows(user_id=user_id)
        return _build_structured_avatar_data(rows)

    def get_user_p2_entries(self, user_id: int) -> list[str]:
        rows = self._fetch_user_p2_rows(user_id=user_id)
        return _build_p2_response_entries(rows)

    def get_avatar_voice(self, user_id: int) -> str | None:
        query = (
            "SELECT avatar_voice FROM tbluser_avatar "
            "WHERE user_id = %s AND deleted = 0 "
            "ORDER BY id DESC LIMIT 1"
        )
        with self._cursor() as cursor:
            cursor.execute(query, (user_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return _normalize_avatar_voice(_pick_first_text(dict(row), ["avatar_voice", "voice"]))

    def get_conversation_subjects(self, user_id: int) -> list[dict[str, Any]]:
        query = (
            "SELECT id, subject FROM tbluser_conversation_subject "
            "WHERE user_id = %s AND deleted = 0 ORDER BY id DESC"
        )
        with self._cursor() as cursor:
            cursor.execute(query, (user_id,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_conversation_subject_text(self, user_id: int, subject_id: int) -> str | None:
        user_id = _require_user_id(user_id)
        query = (
            "SELECT subject FROM tbluser_conversation_subject "
            "WHERE user_id = %s AND id = %s AND deleted = 0 LIMIT 1"
        )
        with self._cursor() as cursor:
            cursor.execute(query, (user_id, int(subject_id)))
            row = cursor.fetchone()
            if not row:
                return None
            text = str(row.get("subject") or "").strip()
            return text or None

    def create_conversation_subject(self, user_id: int, subject: str) -> int:
        user_id = _require_user_id(user_id)
        normalized = _normalize_subject(subject)
        with self._cursor() as cursor:
            cursor.execute(
                "INSERT INTO tbluser_conversation_subject (user_id, subject) VALUES (%s, %s) RETURNING id",
                (user_id, normalized),
            )
            inserted = cursor.fetchone()
            return int(inserted["id"])

    def update_conversation_subject(self, subject_id: int, subject: str) -> None:
        normalized = _normalize_subject(subject)
        with self._cursor() as cursor:
            cursor.execute(
                "UPDATE tbluser_conversation_subject SET subject = %s WHERE id = %s AND deleted = 0",
                (normalized, subject_id),
            )

    def persist_p2(self, user_id: int, key: str, value: str) -> dict[str, Any]:
        user_id = _require_user_id(user_id)
        normalized_key = _normalize_p2_key(key)
        normalized_value = _normalize_p2_value(value)
        if not normalized_key:
            raise ValueError("p2 key is required")
        if not normalized_value:
            raise ValueError("p2 value is required")
        with self._cursor() as cursor:
            cursor.execute(
                'SELECT id FROM tbluser_p2 WHERE user_id = %s AND "key" = %s AND deleted = 0 LIMIT 1',
                (user_id, normalized_key),
            )
            existing = cursor.fetchone()
            updated = bool(existing and existing.get("id"))
            cursor.execute(
                'UPDATE tbluser_p2 SET deleted = 1 WHERE user_id = %s AND "key" = %s AND deleted = 0',
                (user_id, normalized_key),
            )
            cursor.execute(
                'INSERT INTO tbluser_p2 (user_id, "key", "value") VALUES (%s, %s, %s) RETURNING id',
                (user_id, normalized_key, normalized_value),
            )
            inserted = cursor.fetchone()
            return {
                "id": int(inserted["id"]) if inserted and inserted.get("id") else None,
                "key": normalized_key,
                "value": normalized_value,
                "updated": updated,
                "created": not updated,
            }

    def find_guideline(self, user_id: int, guideline: str) -> dict[str, Any] | None:
        user_id = _require_user_id(user_id)
        normalized = _normalize_guideline_text(guideline)
        if not normalized:
            return None
        with self._cursor() as cursor:
            cursor.execute(
                "SELECT id, guideline FROM tbluser_guideline "
                "WHERE user_id = %s AND deleted = 0 AND guideline = %s LIMIT 1",
                (user_id, normalized),
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def persist_guideline(self, user_id: int, prompt: str) -> dict[str, Any]:
        user_id = _require_user_id(user_id)
        normalized = _normalize_guideline_text(prompt)
        if not normalized:
            raise ValueError("guideline is required")
        existing = self.find_guideline(user_id=user_id, guideline=normalized)
        if existing is not None:
            return {
                "id": int(existing["id"]),
                "guideline": str(existing.get("guideline") or normalized),
                "already_exists": True,
                "created": False,
                "updated": False,
                "deleted": False,
            }
        with self._cursor() as cursor:
            cursor.execute(
                "INSERT INTO tbluser_guideline (user_id, guideline) VALUES (%s, %s) RETURNING id",
                (user_id, normalized),
            )
            inserted = cursor.fetchone()
            return {
                "id": int(inserted["id"]) if inserted and inserted.get("id") else None,
                "guideline": normalized,
                "already_exists": False,
                "created": True,
                "updated": False,
                "deleted": False,
            }

    def update_guideline(self, user_id: int, guideline_id: int, prompt: str) -> dict[str, Any]:
        user_id = _require_user_id(user_id)
        normalized = _normalize_guideline_text(prompt)
        if not normalized:
            raise ValueError("guideline is required")
        with self._cursor() as cursor:
            cursor.execute(
                "UPDATE tbluser_guideline SET guideline = %s "
                "WHERE id = %s AND user_id = %s AND deleted = 0",
                (normalized, int(guideline_id), user_id),
            )
            return {
                "id": int(guideline_id),
                "guideline": normalized,
                "already_exists": False,
                "created": False,
                "updated": True,
                "deleted": False,
            }

    def delete_guideline(self, user_id: int, guideline_id: int) -> dict[str, Any]:
        user_id = _require_user_id(user_id)
        with self._cursor() as cursor:
            cursor.execute(
                "UPDATE tbluser_guideline SET deleted = 1 "
                "WHERE id = %s AND user_id = %s AND deleted = 0",
                (int(guideline_id), user_id),
            )
            return {
                "id": int(guideline_id),
                "guideline": "",
                "already_exists": False,
                "created": False,
                "updated": False,
                "deleted": True,
            }

    def persist_conversation(
        self,
        user_id: int,
        prompt: str,
        response: str,
        subject: str | None = None,
        subject_id: int | None = None,
    ) -> None:
        user_id = _require_user_id(user_id)
        needs_correction = _looks_like_correction(prompt)
        self.connection.begin()
        try:
            with self._cursor() as cursor:
                resolved_subject_id = subject_id
                if resolved_subject_id is None:
                    resolved_subject_id = self._ensure_subject(cursor, user_id=user_id, subject=subject or "general")
                if needs_correction:
                    cursor.execute(
                        "UPDATE tbluser_conversation_content SET deleted = 1 WHERE user_id = %s AND user_conversation_subject_id = %s AND deleted = 0",
                        (user_id, resolved_subject_id),
                    )
                try:
                    cursor.execute(
                        "INSERT INTO tbluser_conversation_content (user_id, user_conversation_subject_id, prompt, response) VALUES (%s, %s, %s, %s)",
                        (user_id, resolved_subject_id, prompt.strip(), response.strip()),
                    )
                except Exception as exc:
                    raise RuntimeError(
                        "Failed to insert conversation row. "
                        f"subject_id={resolved_subject_id}; error={exc!r}"
                    ) from exc
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise

    @staticmethod
    def _ensure_subject(cursor: Any, user_id: int, subject: str) -> int:
        cursor.execute(
            "SELECT id FROM tbluser_conversation_subject WHERE user_id = %s AND subject = %s AND deleted = 0 LIMIT 1",
            (user_id, subject),
        )
        row = cursor.fetchone()
        if row and row.get("id"):
            return int(row["id"])

        cursor.execute(
            "INSERT INTO tbluser_conversation_subject (user_id, subject) VALUES (%s, %s) RETURNING id",
            (user_id, subject),
        )
        inserted = cursor.fetchone()
        return int(inserted["id"])

_AVATAR_SKIP_COLUMNS = frozenset(
    {
        "id",
        "user_id",
        "deleted",
        "created",
        "updated",
        "avatar",
        "profile",
        "data",
        "content",
        "avatar_voice",
    }
)
_PERSONA_OPPOSITE_LABELS = {
    "sterile": "rational",
}
_AVATAR_FIELD_LABELS = {
    "user_name": "user_name (This is what I refer to the user as)",
    "avatar_name": "avatar_name (This is what I refer to myself(the LLM) as)",
    "user_pronouns": "user_pronouns(these are the pronouns I use to describe the user)",
}


def _format_persona_entry(column_name: str, raw_value: Any) -> str | None:
    if not column_name.startswith("persona_"):
        return None
    try:
        value = int(round(float(raw_value)))
    except (TypeError, ValueError):
        return None
    value = max(0, min(100, value))

    remainder = column_name.removeprefix("persona_")
    parts = [part for part in remainder.split("_") if part]
    if len(parts) < 2:
        return None

    first = parts[0]
    last = _PERSONA_OPPOSITE_LABELS.get(parts[-1], parts[-1])
    return f"{first} = {100 - value}%, {last} = {value}%"


def _format_p2_entry(key: str, value: str) -> str:
    return f"`{key.strip()}` = `{value.strip()}`"


def _format_p2_response_entry(key: str, value: str) -> str:
    return f"- {key.strip()} = {value.strip()}"


def _normalize_p2_key(key: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9\s_]", " ", str(key or "").strip().lower())
    words = [part for part in re.split(r"[\s_]+", cleaned) if part]
    return "_".join(words[:3])


def _normalize_p2_value(value: str) -> str:
    cleaned = " ".join(str(value or "").strip().split())
    return cleaned[:100]


def _normalize_guideline_text(prompt: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", str(prompt or ""))
    return " ".join(cleaned.split())[:100]


def _build_p2_data_from_rows(rows: list[dict[str, Any]]) -> list[str]:
    entries: list[str] = []
    for row in rows:
        key = _pick_first_text(row, ["key"]) or ""
        value = _pick_first_text(row, ["value"]) or ""
        if key and value:
            entries.append(_format_p2_entry(key, value))
    return entries


def _build_p2_response_entries(rows: list[dict[str, Any]]) -> list[str]:
    entries: list[str] = []
    for row in rows:
        key = _pick_first_text(row, ["key"]) or ""
        value = _pick_first_text(row, ["value"]) or ""
        if key and value:
            entries.append(_format_p2_response_entry(key, value))
    return entries


def _build_structured_avatar_data(rows: list[dict[str, Any]]) -> dict[str, Any]:
    user_name = ""
    avatar_name = ""
    user_pronouns = ""
    persona_lines: list[str] = []

    for row in rows:
        if not user_name:
            user_name = str(row.get("user_name") or "").strip()
        if not avatar_name:
            avatar_name = str(row.get("avatar_name") or "").strip()
        if not user_pronouns:
            user_pronouns = str(row.get("user_pronouns") or "").strip()
        for key, raw_value in row.items():
            if not str(key).startswith("persona_") or raw_value is None:
                continue
            formatted = _format_persona_entry(str(key), raw_value)
            if formatted:
                persona_lines.append(f"- {formatted}")

    return {
        "user_name": user_name,
        "avatar_name": avatar_name,
        "avatar_persona": persona_lines,
        "user_pronouns": user_pronouns,
    }


def _build_avatar_data_from_rows(rows: list[dict[str, Any]]) -> list[str]:
    field_entries: list[str] = []
    persona_lines: list[str] = []

    for row in rows:
        for key, raw_value in row.items():
            if key in _AVATAR_SKIP_COLUMNS or raw_value is None:
                continue
            if key.startswith("persona_"):
                formatted = _format_persona_entry(key, raw_value)
                if formatted:
                    persona_lines.append(formatted)
                continue
            text = str(raw_value).strip()
            if text:
                label = _AVATAR_FIELD_LABELS.get(key, key)
                field_entries.append(f"{label}: {text}")

    if persona_lines:
        nested = "\n".join(f"- {line}" for line in persona_lines)
        field_entries.append(f"avatar_persona(reply using these values):\n{nested}")

    if field_entries or persona_lines:
        return field_entries

    legacy_entries: list[str] = []
    for row in rows:
        text = _pick_first_text(row, ["avatar", "profile", "data", "content"]) or ""
        if text.strip():
            legacy_entries.append(text.strip())
    return legacy_entries


def _expand_multiline_rules(entries: Any) -> list[str]:
    """Split each global rule/guideline row on newlines into individual rules."""
    expanded: list[str] = []
    for entry in entries:
        text = str(entry or "").replace("\\n", "\n")
        for line in text.split("\n"):
            stripped = line.strip()
            if stripped:
                expanded.append(stripped)
    return expanded


def _format_rule_entries(entries: list[str], limit: int = 20) -> list[str]:
    from storm_zero_llm.prompts import format_prompt_entries

    return format_prompt_entries(entries, limit=limit)


def _normalize_subject(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
    return " ".join(cleaned.split()) or "general"


def _looks_like_correction(text: str) -> bool:
    lowered = text.lower()
    return any(word in lowered for word in ("actually", "correction", "instead", "update that"))


def _normalize_avatar_voice(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.lower().endswith(".pt"):
        text = text[:-3]
    return text


def _pick_first_text(row: dict[str, Any], keys: list[str]) -> str | None:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _split_topics(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    return [item.strip() for item in re.split(r"[;,]", str(raw)) if item.strip()]


def _rule_blocks_guideline(rule_text: str, guideline: str) -> bool:
    lowered_rule = rule_text.lower()
    lowered_guide = guideline.lower()
    if not lowered_rule.strip() or not lowered_guide.strip():
        return False
    if not any(marker in lowered_rule for marker in ("never", "do not", "must not", "forbid", "forbidden")):
        return False

    rule_tokens = {t for t in re.split(r"[^a-z0-9]+", lowered_rule) if len(t) > 3}
    guide_tokens = {t for t in re.split(r"[^a-z0-9]+", lowered_guide) if len(t) > 3}
    overlap = rule_tokens.intersection(guide_tokens)
    return len(overlap) >= 2


def _require_user_id(user_id: int | str | None) -> int:
    if user_id is None:
        raise ValueError("user_id is required")
    if isinstance(user_id, bool):
        raise ValueError("user_id must be a positive integer")
    if isinstance(user_id, int):
        if user_id <= 0:
            raise ValueError("user_id must be a positive integer")
        return user_id
    text = str(user_id).strip()
    if not text:
        raise ValueError("user_id is required")
    try:
        parsed = int(text)
    except ValueError as exc:
        raise ValueError("user_id must be a positive integer") from exc
    if parsed <= 0:
        raise ValueError("user_id must be a positive integer")
    return parsed


def parse_user_id(value: object) -> int:
    return _require_user_id(value)  # type: ignore[arg-type]
