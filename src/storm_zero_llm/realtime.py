"""Live camera and microphone evaluation session.

Frames and audio stay in memory. A spoken reply is produced only after silence
ends an utterance, and that reply is handed to the existing chat path.
"""

from __future__ import annotations

import array
import importlib
import math
import sys
import threading
from typing import Any, Callable

from storm_zero_llm.config import StormZeroConfig

FRAME_KIND = 0x01
AUDIO_KIND = 0x02
SAMPLE_RATE = 16000
# About 600 ms of quiet after speech closes the utterance so we do not reply on every chunk.
SILENCE_MS = 600
SPEECH_RMS_THRESHOLD = 500
MAX_UTTERANCE_MS = 60000

VisionFn = Callable[[bytes], dict[str, Any]]
TranscribeFn = Callable[[bytes], str]
ChatFn = Callable[..., Any]
TtsFn = Callable[[str], dict[str, Any] | None]


def configuration_error(
    config: StormZeroConfig,
    import_module: Callable[[str], Any] = importlib.import_module,
) -> str | None:
    """Return a reason the realtime session cannot see or hear, or None when it can start."""
    if not config.vision_model.is_file() or not config.vision_mmproj.is_file():
        return "Vision model is not configured. Set VISION_MODEL and VISION_MMPROJ to existing files."
    if not config.whisper_model:
        return "Speech model is not configured. Set WHISPER_MODEL."
    try:
        import_module("faster_whisper")
    except ImportError:
        return "Speech model is not configured. Install the realtime extra (faster-whisper)."
    return None


def build_realtime_prompt(transcript: str, evaluation: dict[str, str]) -> str:
    """Build the chat prompt from what was heard and the latest visual evaluation."""
    return (
        "The user is speaking during a live camera and microphone session.\n"
        f"Scene: {evaluation.get('scene', '')}\n"
        f"People: {evaluation.get('people', '')}\n"
        f"Emotion: {evaluation.get('emotion', '')}\n"
        f"The user said: {transcript}\n"
        "Reply to what they just said. Use the visual evaluation as context."
    )


def pcm_rms(pcm: bytes) -> float:
    """Root-mean-square energy of little-endian 16-bit PCM, used as a simple voice detector."""
    usable = pcm[: len(pcm) - (len(pcm) % 2)]
    if len(usable) < 2:
        return 0.0
    samples = array.array("h")
    samples.frombytes(usable)
    if sys.byteorder != "little":
        samples.byteswap()
    if not samples:
        return 0.0
    total = 0
    for sample in samples:
        total += sample * sample
    return math.sqrt(total / len(samples))


def pcm_duration_ms(pcm: bytes) -> float:
    sample_count = len(pcm) // 2
    return (sample_count / SAMPLE_RATE) * 1000


def _as_text(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    if value is None:
        return ""
    return str(value)


def _chat_response_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    response = getattr(result, "response", None)
    if response is None and isinstance(result, dict):
        response = result.get("response", "")
    return _as_text(response)


class RealtimeSession:
    """One user's live evaluation. Vision runs one frame at a time and keeps only the newest JPEG."""

    def __init__(
        self,
        *,
        user_id: int,
        vision_fn: VisionFn,
        transcribe_fn: TranscribeFn,
        chat_fn: ChatFn,
        tts_fn: TtsFn | None = None,
    ) -> None:
        self.user_id = user_id
        self.vision_fn = vision_fn
        self.transcribe_fn = transcribe_fn
        self.chat_fn = chat_fn
        self.tts_fn = tts_fn
        self.latest_jpeg: bytes | None = None
        self.latest_evaluation: dict[str, str] = {"scene": "", "people": "", "emotion": ""}
        self.latest_transcript = ""
        self._frame_dirty = False
        self._vision_running = False
        self._in_speech = False
        self._silence_ms = 0.0
        self._utterance = bytearray()
        self._lock = threading.Lock()

    def handle_binary(self, payload: bytes) -> list[dict[str, Any]]:
        """Accept one prefixed binary message and return events ready to send."""
        if not payload:
            return [{"type": "error", "error": "empty binary payload"}]
        kind = payload[0]
        body = payload[1:]
        if kind == FRAME_KIND:
            return self._handle_frame(body)
        if kind == AUDIO_KIND:
            return self._handle_audio(body)
        return [{"type": "error", "error": "unknown binary payload"}]

    def _handle_frame(self, jpeg: bytes) -> list[dict[str, Any]]:
        if not jpeg:
            return []
        with self._lock:
            # Drop older frames by keeping only the newest JPEG while a vision pass is running.
            self.latest_jpeg = jpeg
            self._frame_dirty = True
            if self._vision_running:
                return []
            self._vision_running = True
        try:
            return self._drain_vision()
        except Exception as exc:
            with self._lock:
                self._vision_running = False
            return [{"type": "error", "error": str(exc)}]

    def _drain_vision(self) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        while True:
            with self._lock:
                if not self._frame_dirty or self.latest_jpeg is None:
                    self._vision_running = False
                    return events
                jpeg = self.latest_jpeg
                self._frame_dirty = False
            described = self.vision_fn(jpeg)
            with self._lock:
                evaluation = {
                    "scene": _as_text(described.get("scene")),
                    "people": _as_text(described.get("people")),
                    "emotion": _as_text(described.get("emotion")),
                }
                self.latest_evaluation = evaluation
                heard = self.latest_transcript
            events.append(
                {
                    "type": "evaluation",
                    "scene": evaluation["scene"],
                    "people": evaluation["people"],
                    "emotion": evaluation["emotion"],
                    "heard": heard,
                }
            )

    def _handle_audio(self, pcm: bytes) -> list[dict[str, Any]]:
        with self._lock:
            utterance = self._accept_audio(pcm)
        if utterance is None:
            return []
        return self._finish_utterance(utterance)

    def _accept_audio(self, pcm: bytes) -> bytes | None:
        if len(pcm) < 2:
            return None
        rms = pcm_rms(pcm)
        duration = pcm_duration_ms(pcm)
        if rms >= SPEECH_RMS_THRESHOLD:
            self._in_speech = True
            self._silence_ms = 0.0
            self._utterance.extend(pcm)
            if self._utterance_duration_ms() >= MAX_UTTERANCE_MS:
                return self._take_utterance()
            return None
        if not self._in_speech:
            return None
        self._silence_ms += duration
        self._utterance.extend(pcm)
        if self._silence_ms >= SILENCE_MS or self._utterance_duration_ms() >= MAX_UTTERANCE_MS:
            return self._take_utterance()
        return None

    def _utterance_duration_ms(self) -> float:
        return pcm_duration_ms(bytes(self._utterance))

    def _take_utterance(self) -> bytes:
        payload = bytes(self._utterance)
        self._utterance.clear()
        self._in_speech = False
        self._silence_ms = 0.0
        return payload

    def _finish_utterance(self, pcm: bytes) -> list[dict[str, Any]]:
        """Transcribe a finished utterance and reply through chat only after the user has spoken."""
        try:
            transcript = self.transcribe_fn(pcm).strip()
        except Exception as exc:
            return [{"type": "error", "error": str(exc)}]
        with self._lock:
            self.latest_transcript = transcript
            evaluation = dict(self.latest_evaluation)
        events: list[dict[str, Any]] = [{"type": "transcript", "text": transcript, "final": True}]
        if not transcript:
            return events
        try:
            result = self.chat_fn(
                user_id=self.user_id,
                prompt=build_realtime_prompt(transcript, evaluation),
            )
            response = _chat_response_text(result)
        except Exception as exc:
            events.append({"type": "error", "error": str(exc)})
            return events
        reply: dict[str, Any] = {"type": "reply", "response": response}
        if response.strip() and self.tts_fn is not None:
            try:
                tts_payload = self.tts_fn(response)
            except Exception as exc:
                tts_payload = {"base64": None, "mime_type": None, "voice": None, "error": str(exc)}
            if tts_payload:
                reply["tts"] = tts_payload
        events.append(reply)
        return events
