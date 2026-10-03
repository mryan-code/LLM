"""Local vision and speech adapters for realtime evaluation.

High-risk: these functions load model weights and run inference on live camera
and microphone bytes. They do not write frames or audio to disk.
"""

from __future__ import annotations

import base64
import json
import re
import threading
from typing import Any

from storm_zero_llm.config import StormZeroConfig

_VISION_PROMPT = (
    "Describe what you see. Reply with JSON only, using keys scene, people, and emotion. "
    "scene is the setting, people describes who is visible, and emotion is the apparent feeling."
)

_vision_lock = threading.Lock()
_vision_cache: dict[tuple[str, str], Any] = {}
_whisper_lock = threading.Lock()
_whisper_cache: dict[str, Any] = {}


def parse_vision_text(text: str) -> dict[str, str]:
    """Pull scene, people, and emotion out of a model reply."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            return {
                "scene": _text(payload.get("scene")),
                "people": _text(payload.get("people")),
                "emotion": _text(payload.get("emotion")),
            }
    return {"scene": text.strip(), "people": "", "emotion": ""}


def describe_frame(jpeg: bytes, config: StormZeroConfig) -> dict[str, str]:
    """Run the configured multimodal GGUF on one JPEG and return the evaluation fields."""
    llama = _vision_llama(config)
    data_uri = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")
    completion = llama.create_chat_completion(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _VISION_PROMPT},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }
        ]
    )
    text = _completion_text(completion)
    return parse_vision_text(text)


def transcribe_pcm(pcm: bytes, config: StormZeroConfig) -> str:
    """Transcribe 16 kHz mono PCM with faster-whisper. The model name comes from WHISPER_MODEL."""
    model = _whisper_model(config)
    audio = _pcm_to_float(pcm)
    segments, _info = model.transcribe(audio)
    return " ".join(segment.text.strip() for segment in segments if segment.text.strip()).strip()


def _vision_llama(config: StormZeroConfig) -> Any:
    key = (str(config.vision_model), str(config.vision_mmproj))
    cached = _vision_cache.get(key)
    if cached is not None:
        return cached
    with _vision_lock:
        cached = _vision_cache.get(key)
        if cached is not None:
            return cached
        if not config.vision_model.is_file() or not config.vision_mmproj.is_file():
            raise FileNotFoundError("Vision model files are not configured")
        try:
            from llama_cpp import Llama
            from llama_cpp.llama_chat_format import Llava15ChatHandler
        except ImportError as exc:
            raise RuntimeError("llama-cpp-python is required for realtime vision") from exc
        handler = Llava15ChatHandler(clip_model_path=str(config.vision_mmproj), verbose=False)
        llama = Llama(
            model_path=str(config.vision_model),
            chat_handler=handler,
            n_ctx=config.llm_runtime.n_ctx,
            n_threads=config.llm_runtime.n_threads,
            n_gpu_layers=config.llm_runtime.n_gpu_layers,
            n_batch=config.llm_runtime.n_batch,
            verbose=False,
        )
        _vision_cache[key] = llama
        return llama


def _whisper_model(config: StormZeroConfig) -> Any:
    name = config.whisper_model
    cached = _whisper_cache.get(name)
    if cached is not None:
        return cached
    with _whisper_lock:
        cached = _whisper_cache.get(name)
        if cached is not None:
            return cached
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError("faster-whisper is required for realtime speech") from exc
        model = WhisperModel(name, device="cpu", compute_type="int8")
        _whisper_cache[name] = model
        return model


def _pcm_to_float(pcm: bytes):
    import numpy as np

    usable = pcm[: len(pcm) - (len(pcm) % 2)]
    return np.frombuffer(usable, dtype="<i2").astype(np.float32) / 32768.0


def _completion_text(completion: Any) -> str:
    choices = completion.get("choices") if isinstance(completion, dict) else None
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content", "")
    if isinstance(content, list):
        return " ".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
    return str(content or "")


def _text(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    if value is None:
        return ""
    return str(value)
