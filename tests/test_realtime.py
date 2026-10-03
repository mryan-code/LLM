import array

from storm_zero_llm.config import StormZeroConfig
from storm_zero_llm.realtime import FRAME_KIND, AUDIO_KIND, RealtimeSession, configuration_error


def _pcm(value: int, duration_ms: int) -> bytes:
    count = int(16000 * duration_ms / 1000)
    return array.array("h", [value] * count).tobytes()


def _session(vision, transcribe, chat, tts):
    return RealtimeSession(
        user_id=7,
        vision_fn=vision,
        transcribe_fn=transcribe,
        chat_fn=chat,
        tts_fn=tts,
    )


def test_frame_emits_evaluation_without_a_reply() -> None:
    calls = {"chat": 0}

    def vision(_jpeg: bytes) -> dict[str, str]:
        return {"scene": "a desk", "people": "one person", "emotion": "calm"}

    def chat(**_kwargs):
        calls["chat"] += 1
        return {"response": "should not run"}

    session = _session(vision, lambda _pcm: "hello", chat, lambda _text: None)
    events = session.handle_binary(bytes([FRAME_KIND]) + b"jpeg-bytes")

    assert events == [
        {
            "type": "evaluation",
            "scene": "a desk",
            "people": "one person",
            "emotion": "calm",
            "heard": "",
        }
    ]
    assert calls["chat"] == 0


def test_reply_waits_until_silence_ends_the_utterance() -> None:
    chat_calls: list[dict] = []

    def vision(_jpeg: bytes) -> dict[str, str]:
        return {"scene": "a kitchen", "people": "one person", "emotion": "happy"}

    def transcribe(_pcm: bytes) -> str:
        return "what do you see"

    def chat(**kwargs):
        chat_calls.append(kwargs)
        return type("Chat", (), {"response": "I see a kitchen."})()

    def tts(text: str) -> dict[str, str]:
        return {"base64": "abc", "mime_type": "audio/wav", "voice": "af_heart", "text": text}

    session = _session(vision, transcribe, chat, tts)
    session.handle_binary(bytes([FRAME_KIND]) + b"frame")

    speech_events = session.handle_binary(bytes([AUDIO_KIND]) + _pcm(8000, 100))
    assert speech_events == []
    assert chat_calls == []

    short_silence = session.handle_binary(bytes([AUDIO_KIND]) + _pcm(0, 500))
    assert short_silence == []
    assert chat_calls == []

    events = session.handle_binary(bytes([AUDIO_KIND]) + _pcm(0, 100))
    assert events[0] == {"type": "transcript", "text": "what do you see", "final": True}
    assert events[1]["type"] == "reply"
    assert events[1]["response"] == "I see a kitchen."
    assert events[1]["tts"]["base64"] == "abc"
    assert len(chat_calls) == 1
    assert chat_calls[0]["user_id"] == 7
    assert "what do you see" in chat_calls[0]["prompt"]
    assert "a kitchen" in chat_calls[0]["prompt"]

    follow_up = session.handle_binary(bytes([FRAME_KIND]) + b"frame-2")
    assert follow_up[0]["heard"] == "what do you see"


def test_configuration_error_when_vision_files_are_missing(tmp_path) -> None:
    config = StormZeroConfig.load(project_root=tmp_path)
    error = configuration_error(config)
    assert error is not None
    assert "Vision" in error


def test_configuration_error_when_speech_runtime_is_missing(tmp_path) -> None:
    models = tmp_path / "dedicated_models"
    models.mkdir()
    (models / "vision.gguf").write_bytes(b"vision")
    (models / "vision-mmproj.gguf").write_bytes(b"mmproj")
    config = StormZeroConfig.load(project_root=tmp_path)

    def fail_import(name: str):
        if name == "faster_whisper":
            raise ImportError(name)
        raise AssertionError(name)

    error = configuration_error(config, import_module=fail_import)
    assert error is not None
    assert "Speech" in error
