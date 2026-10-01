"""Layered file-backed memory store."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from uuid import uuid4


class MemoryType(StrEnum):
    HARD_RULE = "hard_rule"
    GUIDE = "guide"
    REFERENCE = "reference"
    USER_RULE = "user_rule"
    PROFILE = "profile"
    CONVERSATION = "conversation"
    RELATIONSHIP = "relationship"
    TASK = "task"


AUTHORITY_ORDER: tuple[MemoryType, ...] = (
    MemoryType.HARD_RULE,
    MemoryType.USER_RULE,
    MemoryType.GUIDE,
    MemoryType.REFERENCE,
    MemoryType.PROFILE,
    MemoryType.RELATIONSHIP,
    MemoryType.TASK,
    MemoryType.CONVERSATION,
)


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "memory"


@dataclass
class MemoryRecord:
    memory_type: MemoryType
    scope: str
    title: str
    content: str
    user_id: str | None = None
    topics: list[str] = field(default_factory=list)
    priority: str = "normal"
    confidence: str = "confirmed"
    source: str = "manual"
    memory_id: str = field(default_factory=lambda: uuid4().hex)
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    enabled: bool = True

    def __post_init__(self) -> None:
        self.memory_type = MemoryType(self.memory_type)
        if self.scope not in {"global", "user"}:
            raise ValueError("scope must be 'global' or 'user'")

    def to_dict(self) -> dict[str, object]:
        return {
            "memory_id": self.memory_id,
            "memory_type": self.memory_type.value,
            "scope": self.scope,
            "user_id": self.user_id,
            "title": self.title,
            "content": self.content,
            "topics": self.topics,
            "priority": self.priority,
            "confidence": self.confidence,
            "source": self.source,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "MemoryRecord":
        return cls(
            memory_id=str(payload.get("memory_id") or uuid4().hex),
            memory_type=MemoryType(str(payload["memory_type"])),
            scope=str(payload["scope"]),
            user_id=str(payload["user_id"]) if payload.get("user_id") else None,
            title=str(payload["title"]),
            content=str(payload["content"]),
            topics=[str(item) for item in payload.get("topics", [])],
            priority=str(payload.get("priority", "normal")),
            confidence=str(payload.get("confidence", "confirmed")),
            source=str(payload.get("source", "manual")),
            created_at=str(payload.get("created_at") or utc_now()),
            updated_at=str(payload.get("updated_at") or utc_now()),
            enabled=bool(payload.get("enabled", True)),
        )

    def matches(self, topics: set[str], query: str) -> bool:
        if topics and not topics.intersection(set(self.topics)):
            return False
        if query:
            haystack = f"{self.title}\n{self.content}\n{' '.join(self.topics)}".lower()
            return query.lower() in haystack
        return True


class FileMemoryStore:
    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def add(self, record: MemoryRecord) -> Path:
        record.updated_at = utc_now()
        path = self.path_for(record)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record.to_dict(), indent=2) + "\n", encoding="utf-8")
        return path

    def upsert(self, record: MemoryRecord) -> Path:
        existing = self.find(record.scope, record.memory_type, record.title, record.user_id, record.source)
        if existing is None:
            return self.add(record)
        record.memory_id = existing.memory_id
        record.created_at = existing.created_at
        return self.update(record)

    def update(self, record: MemoryRecord) -> Path:
        existing = self.get(record.memory_id)
        if existing is None:
            raise KeyError(f"Unknown memory id: {record.memory_id}")
        old_path = self.path_for(existing)
        new_path = self.path_for(record)
        record.updated_at = utc_now()
        record.created_at = existing.created_at
        if old_path.exists() and old_path != new_path:
            old_path.unlink()
        new_path.parent.mkdir(parents=True, exist_ok=True)
        new_path.write_text(json.dumps(record.to_dict(), indent=2) + "\n", encoding="utf-8")
        return new_path

    def get(self, memory_id: str) -> MemoryRecord | None:
        for path in self.root.rglob("*.json"):
            record = self._read(path)
            if record and record.memory_id == memory_id:
                return record
        return None

    def find(
        self,
        scope: str,
        memory_type: MemoryType,
        title: str,
        user_id: str | None,
        source: str | None,
    ) -> MemoryRecord | None:
        for record in self._all_records():
            if record.scope != scope or record.memory_type != memory_type:
                continue
            if record.title != title or record.user_id != user_id:
                continue
            if source is not None and record.source != source:
                continue
            return record
        return None

    def retrieve(self, user_id: int, topics: list[str] | None = None, query: str = "") -> list[MemoryRecord]:
        topic_set = set(topics or [])
        records = [
            record
            for record in self._all_records()
            if record.enabled and self._visible(record, user_id) and record.matches(topic_set, query)
        ]
        return sorted(records, key=self._sort_key)

    def path_for(self, record: MemoryRecord) -> Path:
        name = f"{record.memory_type.value}_{slugify(record.title)}"
        if record.scope == "user":
            suffix = str(record.user_id) if record.user_id is not None else "unknown"
            return self.root / "users" / suffix / record.memory_type.value / f"{name}_{suffix}.json"
        return self.root / "global" / record.memory_type.value / f"{name}.json"

    def _all_records(self) -> list[MemoryRecord]:
        rows: list[MemoryRecord] = []
        for path in self.root.rglob("*.json"):
            record = self._read(path)
            if record:
                rows.append(record)
        return rows

    @staticmethod
    def _read(path: Path) -> MemoryRecord | None:
        try:
            return MemoryRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            return None

    @staticmethod
    def _visible(record: MemoryRecord, user_id: int) -> bool:
        if record.scope == "global":
            return True
        return bool(record.user_id) and str(record.user_id) == str(user_id)

    @staticmethod
    def _sort_key(record: MemoryRecord) -> tuple[int, int, str]:
        priority_rank = {"critical": 0, "high": 1, "normal": 2, "low": 3}
        return (
            AUTHORITY_ORDER.index(record.memory_type),
            priority_rank.get(record.priority, 2),
            record.updated_at,
        )
