"""Load and apply `prompts/*.txt` templates on each request."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

_DEFAULT_PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"

_DESCRIPTION_FILES = {
    "global_hard_rules": "global_hard_rules.txt",
    "global_guidelines": "global_guidelines.txt",
    "user_guidelines": "user_guidelines.txt",
    "user_p2": "user_p2.txt",
    "avatar_data": "avatar_data.txt",
}

# kind -> (default file, power file)
_TEMPLATE_FILES: dict[str, tuple[str, str]] = {
    "detection": ("detection.txt", "power_detection.txt"),
    "base": ("base.txt", "power_base.txt"),
    "image_generation": ("image_generation.txt", "image_generation.txt"),
}

# Match template section headers (case-insensitive substring) to rule-context fields.
_SECTION_FIELD_MATCHERS: tuple[tuple[str, str], ...] = (
    ("hard rules", "global_hard_rules"),
    ("global guidelines", "global_guidelines"),
    ("user guidelines", "user_guidelines"),
    ("user profile data", "user_p2_data"),
    ("profile data (p2)", "user_p2_data"),
    ("user avatar data", "user_avatar_data"),
)

_FALLBACK_BASE_HEADERS: dict[str, str] = {
    "global_hard_rules": "Hard rules (always listen to these):",
    "global_guidelines": "Global guidelines (listen to these if they are not overridden by user guidelines):",
    "user_guidelines": "User guidelines (these override global guidelines, not hard rules):",
    "user_p2_data": "User profile data (p2):",
    "user_avatar_data": "User avatar data:",
}

_FALLBACK_IMAGE_HEADERS: dict[str, str] = {
    "user_p2_data": "User profile data (p2):",
    "user_avatar_data": "User avatar data:",
}

# `[...]` blocks are generation instructions, never emitted into the final prompt.
_BRACKET_INSTRUCTION_RE = re.compile(r"\[[^\[\]]*\]")


def default_prompts_dir() -> Path:
    return _DEFAULT_PROMPTS_DIR


def extract_bracket_instructions(text: str) -> list[str]:
    """Return the contents of each `[...]` instruction block."""
    return [match.group(0)[1:-1].strip() for match in _BRACKET_INSTRUCTION_RE.finditer(text)]


def strip_bracket_instructions(text: str) -> str:
    """
    Remove `[...]` generation instructions from template text.

    Bracket contents guide how to build the prompt; they must not appear in
    the text sent to the model.
    """
    cleaned = _BRACKET_INSTRUCTION_RE.sub("", text)
    # Collapse whitespace left behind by removals, preserve newlines.
    cleaned = re.sub(r"[^\S\n]{2,}", " ", cleaned)
    cleaned = re.sub(r" +([,:;.])", r"\1", cleaned)
    return cleaned


def load_prompt_template(
    name: str,
    *,
    prompts_dir: Path | None = None,
) -> str:
    """Read a prompt template from disk. Always hits the filesystem (no cache)."""
    path = (prompts_dir or _DEFAULT_PROMPTS_DIR) / name
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8")
    # Support templates that store literal "\n" sequences.
    return text.replace("\\n", "\n").rstrip()


def prompt_template_filename(kind: str, *, power: bool = False) -> str:
    """
    Resolve which prompts/*.txt file to use.

    parameter.power = true  -> power_detection.txt / power_base.txt
    otherwise               -> detection.txt / base.txt
    """
    files = _TEMPLATE_FILES.get(kind)
    if files is None:
        raise ValueError(f"Unknown prompt template kind: {kind!r}")
    default_name, power_name = files
    return power_name if power else default_name


def resolve_prompt_template(
    kind: str,
    *,
    prompts_dir: Path | None = None,
    power: bool = False,
) -> str:
    """Load the template file selected by parameter.power."""
    return load_prompt_template(
        prompt_template_filename(kind, power=power),
        prompts_dir=prompts_dir,
    )


def parse_template_sections(template: str) -> list[tuple[str, list[str], list[str]]]:
    """
    Split a structural template into (output_header, instructions, example_body) triples.

    `[...]` blocks on the header line are treated as generation instructions and are
    not included in `output_header`.
    """
    sections: list[tuple[str, list[str], list[str]]] = []
    current_header: str | None = None
    current_instructions: list[str] = []
    current_body: list[str] = []
    for line in template.splitlines():
        stripped = line.strip()
        is_header = (
            bool(stripped)
            and not line[:1].isspace()
            and stripped.endswith(":")
            and not stripped.startswith("-")
        )
        if is_header:
            if current_header is not None:
                sections.append((current_header, current_instructions, current_body))
            current_instructions = extract_bracket_instructions(stripped)
            cleaned = strip_bracket_instructions(stripped).strip()
            if not cleaned.endswith(":"):
                cleaned = f"{cleaned}:"
            current_header = cleaned
            current_body = []
            continue
        if current_header is not None:
            current_body.append(line)
    if current_header is not None:
        sections.append((current_header, current_instructions, current_body))
    return sections


def field_for_section_header(header: str) -> str | None:
    lowered = strip_bracket_instructions(header).lower()
    for needle, field in _SECTION_FIELD_MATCHERS:
        if needle in lowered:
            return field
    return None


def format_prompt_entries(entries: list[str], *, limit: int = 20) -> list[str]:
    lines: list[str] = []
    for entry in entries[:limit]:
        entry_lines = [line.rstrip() for line in entry.splitlines() if line.strip()]
        if not entry_lines:
            continue
        lines.append(f"  - {entry_lines[0].strip()}")
        for continuation in entry_lines[1:]:
            stripped = continuation.strip()
            if stripped.startswith("- "):
                lines.append(f"    {stripped}")
            else:
                lines.append(f"    - {stripped}")
    return lines


def fill_structured_prompt_template(
    template: str,
    section_data: dict[str, list[str]],
    *,
    fallback_headers: dict[str, str] | None = None,
) -> str:
    """
    Build a prompt block from a structural template.

    Template section headers are preserved (so edits to prompts/*.txt apply immediately).
    `[...]` generation instructions guide which data to include but are stripped from output.
    Example bodies in the template are replaced with live `section_data` entries.
    Sections with no live data are omitted. Unknown template sections are skipped.
    """
    sections = parse_template_sections(template)
    if sections:
        lines: list[str] = []
        for header, _instructions, _example in sections:
            field = field_for_section_header(header)
            if field is None:
                continue
            entries = [item for item in section_data.get(field, []) if str(item).strip()]
            if not entries:
                continue
            lines.append(header if header.endswith(":") else f"{header}:")
            lines.extend(format_prompt_entries(entries))
        return "\n".join(lines)

    headers = fallback_headers or _FALLBACK_BASE_HEADERS
    lines = []
    for field, header in headers.items():
        entries = [item for item in section_data.get(field, []) if str(item).strip()]
        if not entries:
            continue
        lines.append(strip_bracket_instructions(header).strip())
        lines.extend(format_prompt_entries(entries))
    return "\n".join(lines)


def build_base_rules_block(
    section_data: dict[str, list[str]],
    *,
    prompts_dir: Path | None = None,
    power: bool = False,
) -> str:
    template = resolve_prompt_template("base", prompts_dir=prompts_dir, power=power)
    return fill_structured_prompt_template(
        template,
        section_data,
        fallback_headers=_FALLBACK_BASE_HEADERS,
    )


def build_image_profile_block(
    section_data: dict[str, list[str]],
    *,
    prompts_dir: Path | None = None,
) -> str:
    template = resolve_prompt_template("image_generation", prompts_dir=prompts_dir)
    # Image prompts only include profile/avatar sections even if the template grows.
    image_data = {
        "user_p2_data": section_data.get("user_p2_data", []),
        "user_avatar_data": section_data.get("user_avatar_data", []),
    }
    return fill_structured_prompt_template(
        template,
        image_data,
        fallback_headers=_FALLBACK_IMAGE_HEADERS,
    )


def build_detection_template(
    *,
    prompts_dir: Path | None = None,
    power: bool = False,
) -> str:
    text = resolve_prompt_template("detection", prompts_dir=prompts_dir, power=power)
    text = strip_bracket_instructions(text).rstrip()
    if not text:
        return ""
    return text if text.endswith("\n") else f"{text}\n"


def load_prompt_description(
    key: str,
    *,
    prompts_dir: Path | None = None,
) -> str:
    filename = _DESCRIPTION_FILES.get(key)
    if filename is None:
        return ""
    return load_prompt_template(filename, prompts_dir=prompts_dir).strip()


def build_response_prompts(
    *,
    user_prompt: str,
    global_hard_rules: list[str] | None = None,
    global_guidelines: list[str] | None = None,
    user_guidelines: list[str] | None = None,
    user_p2: list[str] | None = None,
    avatar_data: dict[str, Any] | None = None,
    conversation_subjects: list[str] | None = None,
    conversation_subjects_content: list[dict[str, Any]] | None = None,
    prompts_dir: Path | None = None,
) -> dict[str, Any]:
    """Assemble the structured `prompts` object returned by /chat and /use-model."""
    avatar = avatar_data or {}
    return {
        "user_prompt": user_prompt,
        "global_hard_rules": {
            "rules": list(global_hard_rules or []),
            "description": load_prompt_description("global_hard_rules", prompts_dir=prompts_dir),
        },
        "global_guidelines": {
            "rules": list(global_guidelines or []),
            "description": load_prompt_description("global_guidelines", prompts_dir=prompts_dir),
        },
        "user_guidelines": {
            "rules": list(user_guidelines or []),
            "description": load_prompt_description("user_guidelines", prompts_dir=prompts_dir),
        },
        "user_p2": {
            "p2": list(user_p2 or []),
            "description": load_prompt_description("user_p2", prompts_dir=prompts_dir),
        },
        "avatar_data": {
            "description": load_prompt_description("avatar_data", prompts_dir=prompts_dir),
            "data": {
                "user_name": str(avatar.get("user_name") or ""),
                "avatar_name": str(avatar.get("avatar_name") or ""),
                "avatar_persona": list(avatar.get("avatar_persona") or []),
                "user_pronouns": str(avatar.get("user_pronouns") or ""),
            },
        },
        "conversation_subjects": list(conversation_subjects or []),
        "conversation_subjects_content": list(conversation_subjects_content or []),
        "final": {
            "base": {},
            "image_generation": "",
            "detection": {},
        },
    }


def build_final_detection_prompt(
    *,
    user_prompt: str,
    content: str = "",
    conversation_subjects: list[str] | None = None,
) -> dict[str, Any]:
    """Structured detection prompt object recorded in prompts.final.detection."""
    return {
        "user_prompt": user_prompt,
        "content": content,
        "conversation_subjects": list(conversation_subjects or []),
    }


def build_final_base_prompt(
    *,
    user_prompt: str,
    global_hard_rules: list[str] | None = None,
    global_guidelines: list[str] | None = None,
    user_guidelines: list[str] | None = None,
    user_p2: list[str] | None = None,
    avatar_data: dict[str, Any] | None = None,
    conversation_subjects_content: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Structured base prompt object recorded in prompts.final.base."""
    avatar = avatar_data or {}
    return {
        "user_prompt": user_prompt,
        "global_hard_rules": list(global_hard_rules or []),
        "global_guidelines": list(global_guidelines or []),
        "user_guidelines": list(user_guidelines or []),
        "user_p2": list(user_p2 or []),
        "avatar_data": {
            "user_name": str(avatar.get("user_name") or ""),
            "avatar_name": str(avatar.get("avatar_name") or ""),
            "avatar_persona": list(avatar.get("avatar_persona") or []),
            "user_pronouns": str(avatar.get("user_pronouns") or ""),
        },
        "conversation_subjects_content": list(conversation_subjects_content or []),
    }
