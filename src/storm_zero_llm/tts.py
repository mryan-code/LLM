"""Text-to-speech integration using Kokoro."""

from __future__ import annotations

import base64
import importlib
import io
import os
import struct
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class TTSResult:
    voice: str
    mime_type: str
    audio_base64: str


class KokoroTTSService:
    def __init__(self, assets_path: Path, voice: str = "af_heart", lang_code: str = "a", sample_rate: int = 24000):
        self.assets_path = assets_path
        self.voice = voice
        self.lang_code = lang_code
        self.sample_rate = sample_rate
        self._pipeline: Any | None = None

    def synthesize(self, text: str) -> TTSResult:
        content = text.strip()
        if not content:
            return TTSResult(voice=self.voice, mime_type="audio/wav", audio_base64=self._encode_wav_base64([]))

        pipeline = self._pipeline_instance()
        audio_samples: list[float] = []
        for chunk in pipeline(content, voice=self.voice):
            chunk_audio = self._extract_audio_chunk(chunk)
            if chunk_audio:
                audio_samples.extend(chunk_audio)

        if hasattr(pipeline, "sample_rate"):
            sample_rate = getattr(pipeline, "sample_rate")
            if isinstance(sample_rate, int) and sample_rate > 0:
                self.sample_rate = sample_rate

        return TTSResult(
            voice=self.voice,
            mime_type="audio/wav",
            audio_base64=self._encode_wav_base64(audio_samples),
        )

    def _pipeline_instance(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline

        os.environ.setdefault("KOKORO_HOME", str(self.assets_path))
        os.environ.setdefault("KOKORO_PATH", str(self.assets_path))

        try:
            kokoro_module = importlib.import_module("kokoro")
            pipeline_class = getattr(kokoro_module, "KPipeline")
            self._pipeline = pipeline_class(lang_code=self.lang_code)
            return self._pipeline
        except ImportError:
            # Fall through to kokoro-onnx backend.
            pass
        except Exception as exc:
            raise RuntimeError(f"Kokoro backend failed to initialize: {exc}") from exc

        try:
            self._pipeline = _KokoroOnnxAdapter(self.assets_path)
            return self._pipeline
        except FileNotFoundError as exc:
            raise RuntimeError(str(exc)) from exc
        except ImportError as exc:
            raise RuntimeError(
                "No compatible Kokoro backend is available. Install either 'kokoro' or 'kokoro-onnx'."
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"kokoro-onnx backend failed to initialize: {exc}") from exc

    def diagnostics(self) -> dict[str, str | None]:
        info: dict[str, str | None] = {
            "configured_assets_path": str(self.assets_path),
            "backend": None,
            "model_path": None,
            "voices_path": None,
            "error": None,
        }
        try:
            pipeline = self._pipeline_instance()
        except Exception as exc:
            info["error"] = str(exc)
            return info

        if isinstance(pipeline, _KokoroOnnxAdapter):
            info["backend"] = "kokoro-onnx"
            info["model_path"] = str(pipeline.model_path)
            info["voices_path"] = str(pipeline.voices_path)
            return info

        info["backend"] = "kokoro"
        return info

    def _extract_audio_chunk(self, chunk: Any) -> list[float]:
        # KPipeline yields tuples like (graphemes, phonemes, audio).
        candidate = chunk[2] if isinstance(chunk, tuple) and len(chunk) >= 3 else chunk

        if hasattr(candidate, "detach") and hasattr(candidate, "cpu") and hasattr(candidate, "numpy"):
            candidate = candidate.detach().cpu().numpy()
        if hasattr(candidate, "numpy"):
            candidate = candidate.numpy()
        if hasattr(candidate, "flatten"):
            candidate = candidate.flatten().tolist()

        if isinstance(candidate, (list, tuple)):
            return [float(value) for value in candidate]

        if isinstance(candidate, Iterable):
            return [float(value) for value in candidate]

        return []

    def _encode_wav_base64(self, samples: list[float]) -> str:
        pcm = bytearray()
        for sample in samples:
            clamped = max(-1.0, min(1.0, float(sample)))
            pcm.extend(struct.pack("<h", int(clamped * 32767)))

        with io.BytesIO() as buffer:
            with wave.open(buffer, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(bytes(pcm))
            raw = buffer.getvalue()

        return base64.b64encode(raw).decode("ascii")


class _KokoroOnnxAdapter:
    def __init__(self, assets_path: Path):
        module = importlib.import_module("kokoro_onnx")
        kokoro_class = getattr(module, "Kokoro")

        roots = self._candidate_asset_roots(assets_path)
        model_path, _ = self._find_assets_file(roots, "*.onnx")
        voices_path, _ = self._find_assets_file(roots, "*.bin")

        self.model_path = model_path
        self.voices_path = voices_path
        self._validate_model_input_types(self.model_path)
        self._engine = kokoro_class(model_path=str(model_path), voices_path=str(voices_path))
        self.sample_rate = 24000

    def __call__(self, text: str, voice: str):
        audio, sample_rate = self._engine.create(text=text, voice=voice, lang="en-us")
        self.sample_rate = int(sample_rate)
        return [(None, None, audio)]

    @staticmethod
    def _candidate_asset_roots(primary: Path) -> list[Path]:
        home = Path.home()
        env_roots = [
            os.environ.get("TTS_PATH"),
            os.environ.get("KOKORO_PATH"),
            os.environ.get("KOKORO_HOME"),
        ]

        candidates: list[Path] = [primary]
        candidates.extend(Path(value).expanduser() for value in env_roots if value)
        candidates.extend(
            [
                home / "Library" / "Application Support" / "kokoro",
                home / "Library" / "Application Support" / "kokoro-onnx",
                home / "Library" / "Caches" / "kokoro",
                home / "Library" / "Caches" / "kokoro-onnx",
                home / "Library" / "Caches" / "huggingface" / "hub",
                home / ".cache" / "kokoro",
                home / ".cache" / "kokoro-onnx",
                home / ".cache" / "huggingface" / "hub",
                Path("/Library/Application Support/kokoro"),
                Path("/Library/Application Support/kokoro-onnx"),
                Path("/Library/Caches/kokoro"),
                Path("/Library/Caches/kokoro-onnx"),
                Path("/opt/homebrew/share/kokoro"),
                Path("/opt/homebrew/share/kokoro-onnx"),
                Path("/usr/local/share/kokoro"),
                Path("/usr/local/share/kokoro-onnx"),
                Path("/usr/share/kokoro"),
                Path("/usr/share/kokoro-onnx"),
            ]
        )

        unique: list[Path] = []
        seen: set[str] = set()
        for path in candidates:
            try:
                normalized = str(path.resolve())
            except Exception:
                normalized = str(path)
            if normalized in seen:
                continue
            seen.add(normalized)
            unique.append(path)
        return unique

    @staticmethod
    def _find_assets_file(roots: list[Path], pattern: str) -> tuple[Path, Path]:
        searched: list[str] = []
        for root in roots:
            if not root.exists() or not root.is_dir():
                continue
            searched.append(str(root))
            matches = sorted(root.rglob(pattern))
            if matches:
                return matches[0], root

        searched_suffix = ", ".join(searched) if searched else "no existing asset directories"
        raise FileNotFoundError(f"No matching TTS asset for pattern '{pattern}'. Searched: {searched_suffix}")

    @staticmethod
    def _validate_model_input_types(model_path: Path) -> None:
        try:
            import onnxruntime as ort
        except Exception:
            # If onnxruntime is not importable here, let kokoro-onnx emit its own init error.
            return

        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        input_types = [item.type.lower() for item in session.get_inputs()]
        if any("int" in value for value in input_types):
            return

        pretty = ", ".join(input_types) if input_types else "none"
        raise RuntimeError(
            f"Selected ONNX model appears incompatible with kokoro-onnx token input types. "
            f"model_path={model_path} input_types=[{pretty}]"
        )
