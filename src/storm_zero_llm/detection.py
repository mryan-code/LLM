"""Query intent detection for routing requests."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from storm_zero_llm.prompts import (
    build_detection_template,
    build_final_detection_prompt,
    default_prompts_dir,
)
from storm_zero_llm.provider import GenerationRequest, LlamaCppProvider, LocalCompanionProvider

_DETECTION_OUTPUT_TOKEN_RESERVE = 256

_SUBJECT_REFERENCE_PATTERNS = (
    re.compile(
        r"(?:remember|recall)\s+(?:that\s+time\s+|when\s+)?we\s+"
        r"(?:talked|spoke|discussed|chatted)\s+about\s+(.+)$",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:that|our)\s+(?:conversation|chat|discussion)\s+about\s+(.+)$",
        re.IGNORECASE,
    ),
    re.compile(
        r"back\s+to\s+(?:talking\s+about\s+|our\s+conversation\s+about\s+)?(.+)$",
        re.IGNORECASE,
    ),
)


class DetectionAction(StrEnum):
    CREATE_MEDIA = "create_media"
    USER_DATA = "user_data"
    ADD_USER_DATA = "user_data"  # alias for backward compatibility
    SCHEDULED_TASK = "scheduled_task"
    GENERAL_QUERY = "general_query"


_ACTION_ALIASES = {
    "user_data": DetectionAction.USER_DATA,
    "add_user_data": DetectionAction.USER_DATA,
    "create_media": DetectionAction.CREATE_MEDIA,
    "scheduled_task": DetectionAction.SCHEDULED_TASK,
    "general_query": DetectionAction.GENERAL_QUERY,
}


class ConversationMode(StrEnum):
    NEW = "new"
    CONTINUED = "continued"


@dataclass(frozen=True)
class DetectionResult:
    action: DetectionAction
    media_type: str | None = None
    user_data_target: str | None = None
    conversation_mode: ConversationMode | None = None
    subject_summary: str | None = None
    subject_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "media_type": self.media_type,
            "user_data_target": self.user_data_target,
            "conversation_mode": self.conversation_mode.value if self.conversation_mode else None,
            "subject_id": self.subject_id,
            "subject_summary": self.subject_summary,
        }


def normalize_subject(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
    return " ".join(cleaned.split())


def estimate_token_count(text: str) -> int:
    """Conservative token estimate for packing context without loading a tokenizer."""
    stripped = text.strip()
    if not stripped:
        return 0
    return max(1, (len(stripped) + 3) // 4)


def format_subject_line(row: dict[str, object]) -> str:
    return f"- id={row.get('id')} subject={row.get('subject')}"


def pack_subjects_block(
    existing_subjects: list[dict[str, object]],
    *,
    prompt: str,
    system_prompt: str,
    n_ctx: int,
    output_reserve: int = _DETECTION_OUTPUT_TOKEN_RESERVE,
) -> str:
    """Include as many subject rows as fit in the model context window."""
    budget = max(
        0,
        int(n_ctx) - int(output_reserve) - estimate_token_count(prompt) - estimate_token_count(system_prompt),
    )
    if budget <= 0 or not existing_subjects:
        return "No existing subjects."

    lines: list[str] = []
    used = 0
    for row in existing_subjects:
        line = format_subject_line(row)
        cost = estimate_token_count(line) + 1
        if used + cost > budget:
            break
        lines.append(line)
        used += cost

    return "\n".join(lines) if lines else "No existing subjects."


def load_detection_prompt_template(
    prompts_dir: Path | None = None,
    *,
    power: bool = False,
) -> str:
    """Re-read detection.txt (or power_detection.txt) from disk on every call."""
    return build_detection_template(prompts_dir=prompts_dir, power=power)


def build_detection_prompt(
    user_prompt: str,
    subjects_block: str,
    *,
    prompts_dir: Path | None = None,
    power: bool = False,
) -> str:
    template = load_detection_prompt_template(prompts_dir, power=power).rstrip()
    base = f"{user_prompt.strip()} - {template}".rstrip()
    if subjects_block.strip():
        return f"{base}\n{subjects_block.strip()}"
    return base


def build_detection_prompt_object(
    user_prompt: str,
    existing_subjects: list[dict[str, object]] | None = None,
    *,
    prompts_dir: Path | None = None,
    power: bool = False,
) -> dict[str, Any]:
    subjects = [
        str(row.get("subject") or "").strip()
        for row in (existing_subjects or [])
        if str(row.get("subject") or "").strip()
    ]
    return build_final_detection_prompt(
        user_prompt=user_prompt,
        content=load_detection_prompt_template(prompts_dir, power=power).rstrip(),
        conversation_subjects=subjects,
    )


def compose_detection_prompt_string(
    detection_object: dict[str, Any],
    *,
    subjects_block: str = "",
) -> str:
    user_prompt = str(detection_object.get("user_prompt") or "").strip()
    content = str(detection_object.get("content") or "").rstrip()
    base = f"{user_prompt} - {content}".rstrip() if content else user_prompt
    if subjects_block.strip():
        return f"{base}\n{subjects_block.strip()}"
    subjects = detection_object.get("conversation_subjects") or []
    if subjects:
        lines = [f"- subject={subject}" for subject in subjects if str(subject).strip()]
        if lines:
            return f"{base}\n" + "\n".join(lines)
    return base


def match_referenced_subject(
    prompt: str,
    existing_subjects: list[dict[str, object]],
) -> tuple[int, str] | None:
    """If the prompt references an existing subject, return (subject_id, subject)."""
    if not existing_subjects:
        return None
    text = prompt.strip()
    referenced = ""
    for pattern in _SUBJECT_REFERENCE_PATTERNS:
        match = pattern.search(text)
        if match is None:
            continue
        referenced = normalize_subject(next((group for group in match.groups() if group), ""))
        break
    if not referenced:
        return None

    best: tuple[int, str, int] | None = None
    for row in existing_subjects:
        subject_text = normalize_subject(str(row.get("subject", "")))
        if not subject_text:
            continue
        if referenced == subject_text or referenced in subject_text or subject_text in referenced:
            score = min(len(referenced), len(subject_text))
            if best is None or score > best[2]:
                best = (int(row["id"]), str(row.get("subject") or subject_text), score)
    if best is None:
        return None
    return best[0], best[1]


_IMAGE_REQUEST_VERBS = r"(show|create|send|generate)"
_IMAGE_REQUEST_NOUNS = r"(image|photo|nude|pic|picture)"
_IMAGE_REQUEST_PATTERN = re.compile(
    rf"\b{_IMAGE_REQUEST_VERBS}\b[^.\n]*\b{_IMAGE_REQUEST_NOUNS}\b|\b{_IMAGE_REQUEST_NOUNS}\b[^.\n]*\b{_IMAGE_REQUEST_VERBS}\b",
    re.IGNORECASE,
)


def matches_image_generation_request(text: str) -> bool:
    return bool(_IMAGE_REQUEST_PATTERN.search(text.strip()))


class IntentDetector:
    def __init__(
        self,
        detection_model: Path,
        llm_runtime,
        prompts_dir: Path | None = None,
    ):
        self.detection_model = detection_model
        self.llm_runtime = llm_runtime
        self.prompts_dir = prompts_dir or default_prompts_dir()
        self.last_prompt: str | None = None
        self.last_detection: dict[str, Any] | None = None

    def detect(
        self,
        prompt: str,
        existing_subjects: list[dict[str, object]] | None = None,
        *,
        detection_model: Path | None = None,
        power: bool = False,
    ) -> DetectionResult:
        self.last_prompt = None
        self.last_detection = None
        existing_subjects = existing_subjects or []
        self.last_detection = build_detection_prompt_object(
            prompt,
            existing_subjects,
            prompts_dir=self.prompts_dir,
            power=power,
        )
        subjects_block = pack_subjects_block(
            existing_subjects,
            prompt=prompt,
            system_prompt="",
            n_ctx=getattr(self.llm_runtime, "n_ctx", 8192),
        )
        self.last_prompt = compose_detection_prompt_string(
            self.last_detection,
            subjects_block=subjects_block,
        )
        if matches_image_generation_request(prompt):
            return DetectionResult(
                action=DetectionAction.CREATE_MEDIA,
                media_type="image",
                conversation_mode=ConversationMode.NEW,
                subject_summary=normalize_subject(prompt)[:120] or "general",
            )
        referenced = match_referenced_subject(prompt, existing_subjects)
        if referenced is not None:
            subject_id, subject_text = referenced
            return DetectionResult(
                action=DetectionAction.GENERAL_QUERY,
                conversation_mode=ConversationMode.CONTINUED,
                subject_summary=normalize_subject(subject_text)[:120] or "general",
                subject_id=subject_id,
            )
        model_result = self._detect_with_model(
            prompt,
            existing_subjects,
            detection_model=detection_model,
            power=power,
        )
        if model_result is not None:
            return model_result
        return self._detect_with_heuristics(prompt, existing_subjects)

    def _detect_with_model(
        self,
        prompt: str,
        existing_subjects: list[dict[str, object]],
        *,
        detection_model: Path | None = None,
        power: bool = False,
    ) -> DetectionResult | None:
        model_path = detection_model or self.detection_model
        if not model_path.exists():
            return None

        system_prompt = ""
        template = load_detection_prompt_template(self.prompts_dir, power=power)
        # Budget uses prompt + system + subjects, matching the chat completion shape.
        subjects_block = pack_subjects_block(
            existing_subjects,
            prompt=f"{prompt} - {template}",
            system_prompt=system_prompt,
            n_ctx=getattr(self.llm_runtime, "n_ctx", 8192),
        )
        detection_prompt = compose_detection_prompt_string(
            self.last_detection
            or build_detection_prompt_object(
                prompt,
                existing_subjects,
                prompts_dir=self.prompts_dir,
                power=power,
            ),
            subjects_block=subjects_block,
        )
        # Prefer the budget-aware subjects packing used for the model call.
        self.last_prompt = detection_prompt
        try:
            provider = LlamaCppProvider(model_path, self.llm_runtime)
            raw = provider.generate(
                GenerationRequest(prompt=detection_prompt, system_prompt=system_prompt)
            ).response
        except Exception:
            return None

        return self._parse_model_detection(raw, prompt, existing_subjects)

    def _parse_model_detection(
        self,
        raw: str,
        prompt: str,
        existing_subjects: list[dict[str, object]],
    ) -> DetectionResult | None:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if match is None:
            return None
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None

        action_raw = str(payload.get("action", "general_query")).strip().lower()
        action = _ACTION_ALIASES.get(action_raw, DetectionAction.GENERAL_QUERY)

        media_type = payload.get("media_type")
        user_data_target = payload.get("user_data_target")
        conversation_mode_raw = payload.get("conversation_mode")
        subject_summary = payload.get("subject_summary")
        subject_id = payload.get("subject_id")

        conversation_mode = None
        if conversation_mode_raw:
            try:
                conversation_mode = ConversationMode(str(conversation_mode_raw).lower())
            except ValueError:
                conversation_mode = ConversationMode.NEW

        resolved_subject_id = None
        if subject_id is not None:
            try:
                resolved_subject_id = int(subject_id)
            except (TypeError, ValueError):
                resolved_subject_id = None

        if action == DetectionAction.GENERAL_QUERY and conversation_mode == ConversationMode.CONTINUED:
            if resolved_subject_id is None and existing_subjects:
                resolved_subject_id = int(existing_subjects[0]["id"])

        summary = normalize_subject(str(subject_summary or prompt))[:120] or "general"

        return DetectionResult(
            action=action,
            media_type=str(media_type).lower() if media_type and str(media_type).lower() != "null" else None,
            user_data_target=(
                str(user_data_target).lower()
                if user_data_target and str(user_data_target).lower() != "null"
                else None
            ),
            conversation_mode=conversation_mode or ConversationMode.NEW,
            subject_summary=summary,
            subject_id=resolved_subject_id,
        )

    def _detect_with_heuristics(
        self,
        prompt: str,
        existing_subjects: list[dict[str, object]],
    ) -> DetectionResult:
        text = prompt.strip().lower()
        if not text:
            return DetectionResult(
                action=DetectionAction.GENERAL_QUERY,
                conversation_mode=ConversationMode.NEW,
                subject_summary="general",
            )

        media_verbs = r"(show|create|send|generate|make|produce|render|compose|build|draw|design|craft)"
        image_nouns = r"(image|photo|picture|art|illustration|logo|icon|poster|wallpaper|picture)"
        # music_nouns = r"(music|song|beat|melody|track|instrumental|audio|jingle)"

        # if re.search(rf"{media_verbs}[^.\n]*{music_nouns}|{music_nouns}[^.\n]*{media_verbs}", text):
        #     return DetectionResult(action=DetectionAction.CREATE_MEDIA, media_type="music")
        if re.search(rf"{media_verbs}[^.\n]*{image_nouns}|{image_nouns}[^.\n]*{media_verbs}", text):
            return DetectionResult(action=DetectionAction.CREATE_MEDIA, media_type="image")

        if re.search(r"\b(remind me|reminder|every day at|every morning|schedule|at \d{1,2}:\d{2})\b", text):
            return DetectionResult(action=DetectionAction.SCHEDULED_TASK)

        guideline_markers = ("always ", "never ", "from now on", "remember to", "prefer ", "call me ")
        profile_markers = ("my name is", "i am ", "i'm ", "my favorite", "favourite", "i like ", "i love ")

        if any(marker in text for marker in guideline_markers):
            return DetectionResult(action=DetectionAction.USER_DATA, user_data_target="guideline")
        if any(marker in text for marker in profile_markers):
            return DetectionResult(action=DetectionAction.USER_DATA, user_data_target="p2")

        referenced = match_referenced_subject(prompt, existing_subjects)
        if referenced is not None:
            subject_id, subject_text = referenced
            return DetectionResult(
                action=DetectionAction.GENERAL_QUERY,
                conversation_mode=ConversationMode.CONTINUED,
                subject_summary=normalize_subject(subject_text)[:120] or "general",
                subject_id=subject_id,
            )

        conversation_mode = ConversationMode.NEW
        subject_id = None
        subject_summary = normalize_subject(prompt)[:120] or "general"
        normalized_prompt = normalize_subject(prompt)
        for row in existing_subjects:
            subject_text = normalize_subject(str(row.get("subject", "")))
            if subject_text and subject_text in normalized_prompt:
                conversation_mode = ConversationMode.CONTINUED
                subject_id = int(row["id"])
                subject_summary = subject_text[:120] or subject_summary
                break

        return DetectionResult(
            action=DetectionAction.GENERAL_QUERY,
            conversation_mode=conversation_mode,
            subject_summary=subject_summary,
            subject_id=subject_id,
        )

    def extract_p2_key_value(
        self,
        prompt: str,
        *,
        detection_model: Path | None = None,
    ) -> tuple[str, str] | None:
        heuristic = self._extract_p2_key_value_heuristic(prompt)
        model_path = detection_model or self.detection_model
        if not model_path.exists():
            return heuristic

        system_prompt = (
            "Extract user profile preference from the message. Reply with JSON only using this schema:\n"
            '{"key":"snake_case_identifier","value":"the preference value"}\n'
            "Use short lowercase snake_case keys such as name, age, or favorite_color. "
            "Extract only the stated preference value, not the full sentence."
        )

        try:
            provider = LlamaCppProvider(model_path, self.llm_runtime)
            raw = provider.generate(GenerationRequest(prompt=prompt, system_prompt=system_prompt)).response
        except Exception:
            return heuristic

        parsed = self._parse_p2_key_value(raw)
        return parsed or heuristic

    def _parse_p2_key_value(self, raw: str) -> tuple[str, str] | None:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if match is None:
            return None
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None

        key = str(payload.get("key", "")).strip()
        value = str(payload.get("value", "")).strip()
        if not key or not value or key.lower() == "null" or value.lower() == "null":
            return None
        return key, value

    def _extract_p2_key_value_heuristic(self, prompt: str) -> tuple[str, str] | None:
        text = prompt.strip()
        if not text:
            return None

        patterns: list[tuple[str, str]] = [
            (r"my name is\s+(.+)", "name"),
            (r"call me\s+(.+)", "name"),
            (r"my favorite color is\s+(.+)", "favorite_color"),
            (r"my favourite color is\s+(.+)", "favorite_color"),
            (r"my favorite colour is\s+(.+)", "favorite_color"),
            (r"my favourite colour is\s+(.+)", "favorite_colour"),
            (r"i am\s+(\d+)\s+years old", "age"),
            (r"i'm\s+(\d+)\s+years old", "age"),
            (r"i am\s+(.+)", "name"),
            (r"i'm\s+(.+)", "name"),
            (r"i like\s+(.+)", "likes"),
            (r"i love\s+(.+)", "likes"),
        ]
        lowered = text.lower()
        for pattern, key in patterns:
            match = re.search(pattern, lowered, flags=re.IGNORECASE)
            if match is None:
                continue
            value = text[match.start(1):match.end(1)].strip(" .")
            if value:
                return key, value
        return None
