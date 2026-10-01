"""Model providers for Storm Zero LLM."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

from storm_zero_llm.config import LLMRuntimeConfig

_THINK_TAG_OPEN = "<" + "think" + ">"
_THINK_TAG_CLOSE = "</" + "think" + ">"
_THINKING_OPEN_TAGS = ("<think>", _THINK_TAG_OPEN)
_THINKING_CLOSE_TAGS = ("</think>", _THINK_TAG_CLOSE)
_FINAL_RESPONSE_PATTERN = re.compile(
    r"(?:^|\n)\s*final(?:\s+)?(?:response|answer)\s*:\s*",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class GenerationRequest:
    prompt: str
    system_prompt: str
    include_reasoning: bool = False


@dataclass(frozen=True)
class GenerationResult:
    response: str
    reasoning: str | None = None


@dataclass
class _LoadedModel:
    llama: Any
    chat_template_kwargs: dict[str, Any]


class LocalCompanionProvider:
    def generate(self, request: GenerationRequest) -> GenerationResult:
        text = request.prompt.strip()
        if not text:
            return GenerationResult(response="")
        if "?" in text:
            return GenerationResult(response="I hear you. I can help reason through that.")
        return GenerationResult(response="I hear you.")


class LlamaCppProvider:
    _load_lock: Lock = Lock()
    _loaded_models: dict[tuple[str, int, int, int, int, bool], _LoadedModel] = {}
    _model_locks: dict[tuple[str, int, int, int, int, bool], Lock] = {}

    def __init__(self, model_path: Path, runtime: LLMRuntimeConfig):
        self.model_path = model_path
        self.runtime = runtime

    def generate(self, request: GenerationRequest) -> GenerationResult:
        key = self._model_key()
        loaded = self._get_or_load_model(key)
        with self._inference_lock(key):
            loaded.chat_template_kwargs["enable_thinking"] = request.include_reasoning
            # Clear KV cache so unrelated requests do not share a partial prefix match.
            loaded.llama.reset()
            completion = loaded.llama.create_chat_completion(
                messages=[
                    {"role": "system", "content": request.system_prompt},
                    {"role": "user", "content": request.prompt},
                ],
            )
        text = _completion_text(completion).strip()
        model_reasoning, response = _extract_model_output(text)
        if request.include_reasoning:
            return GenerationResult(response=response, reasoning=model_reasoning)
        return GenerationResult(response=response)

    def _model_key(self) -> tuple[str, int, int, int, int, bool]:
        return (
            str(self.model_path.resolve()),
            self.runtime.n_ctx,
            self.runtime.n_gpu_layers,
            self.runtime.n_threads,
            self.runtime.n_batch,
            self.runtime.flash_attn,
        )

    def _inference_lock(self, key: tuple[str, int, int, int, int, bool]) -> Lock:
        with self._load_lock:
            lock = self._model_locks.get(key)
            if lock is None:
                lock = Lock()
                self._model_locks[key] = lock
            return lock

    def _get_or_load_model(self, key: tuple[str, int, int, int, int, bool]) -> _LoadedModel:
        cached = self._loaded_models.get(key)
        if cached is not None:
            return cached

        with self._load_lock:
            cached = self._loaded_models.get(key)
            if cached is not None:
                return cached

            if not self.model_path.exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            try:
                from llama_cpp import Llama
            except ImportError as exc:
                raise RuntimeError("llama-cpp-python is required to run GGUF models") from exc

            chat_template_kwargs: dict[str, Any] = {"enable_thinking": False}
            llama = Llama(
                model_path=str(self.model_path),
                n_ctx=self.runtime.n_ctx,
                n_threads=self.runtime.n_threads,
                n_gpu_layers=self.runtime.n_gpu_layers,
                n_batch=self.runtime.n_batch,
                flash_attn=self.runtime.flash_attn,
            )
            _wrap_chat_handler(llama, chat_template_kwargs)
            cached = _LoadedModel(llama=llama, chat_template_kwargs=chat_template_kwargs)
            self._loaded_models[key] = cached
            self._model_locks.setdefault(key, Lock())
            return cached


def _wrap_chat_handler(model: Any, template_kwargs: dict[str, Any]) -> None:
    base_chat_handler = model.chat_handler or model._chat_handlers.get(model.chat_format)
    if base_chat_handler is None:
        import llama_cpp.llama_chat_format as llama_chat_format

        base_chat_handler = llama_chat_format.get_chat_completion_handler(model.chat_format)

    def chat_handler_with_kwargs(*args: Any, **kwargs: Any) -> Any:
        return base_chat_handler(*args, **{**template_kwargs, **kwargs})

    model.chat_handler = chat_handler_with_kwargs


def _completion_text(completion: dict[str, Any]) -> str:
    choices = completion.get("choices", [])
    if not choices:
        return ""
    first = choices[0]
    message = first.get("message") if isinstance(first, dict) else None
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content
    if isinstance(first, dict):
        text = first.get("text")
        if isinstance(text, str):
            return text
    return ""


def _extract_model_output(raw: str) -> tuple[str | None, str]:
    text = raw.strip()
    if not text:
        return None, ""

    reasoning_parts: list[str] = []
    lower = text.lower()
    for open_tag, close_tag in zip(_THINKING_OPEN_TAGS, _THINKING_CLOSE_TAGS, strict=True):
        if open_tag:
            if open_tag not in lower or not lower.startswith(open_tag):
                continue
        elif close_tag not in lower:
            continue

        close_index = lower.rfind(close_tag)
        if close_index >= 0:
            block_start = lower.find(open_tag) + len(open_tag) if open_tag else 0
            segment = text[block_start:close_index].strip()
            if segment:
                reasoning_parts.append(segment)
            text = text[close_index + len(close_tag) :].strip()
            lower = text.lower()
            continue

        block_start = lower.find(open_tag) + len(open_tag) if open_tag else 0
        segment = text[block_start:].strip()
        if segment:
            reasoning_parts.append(segment)
        return _join_reasoning(reasoning_parts), ""

    final_match = _FINAL_RESPONSE_PATTERN.search(text)
    if final_match:
        prefix = text[: final_match.start()].strip()
        if prefix:
            reasoning_parts.append(prefix)
        text = text[final_match.end() :].strip()

    return _join_reasoning(reasoning_parts), text.strip().strip('"')


def _join_reasoning(parts: list[str]) -> str | None:
    reasoning = "\n\n".join(part for part in parts if part).strip()
    return reasoning or None


def _sanitize_output(raw: str) -> str:
    return _extract_model_output(raw)[1]
