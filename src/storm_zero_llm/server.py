"""HTTP server for Storm Zero LLM."""

from __future__ import annotations

import hmac
import json
import os
import re
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from storm_zero_llm.agent import StormZeroAgent
from storm_zero_llm.image_generation import (
    ImageGenerationConfig,
    ImageGenerator,
    create_image_generator,
)
from storm_zero_llm.training_db import parse_user_id
from storm_zero_llm.tts import KokoroTTSService, TTSResult


class StormZeroRequestHandler(BaseHTTPRequestHandler):
    agent: StormZeroAgent
    _tts_service: KokoroTTSService | None = None
    _image_generator: ImageGenerator | None = None
    _file_data_url_start_pattern = re.compile(r"data:([^;,\s]+);base64,([A-Za-z0-9+/=]*)", re.IGNORECASE)

    def do_GET(self) -> None:
        if self.path == "/":
            self._send_json(
                {
                    "status": "ok",
                    "service": "storm-zero-llm",
                    "message": "Storm Zero LLM server is running.",
                    "readme": self._read_readme(),
                    "endpoints": {
                        "health": "/health",
                        "chat": "/chat",
                        "about": "/about",
                        "use-model": "/use-model",
                        "generate-image": "/generate-image",
                    },
                }
            )
            return

        if self.path == "/health":
            self._send_json(
                {
                    "success": True,
                    "status_code": 200,
                    "service": "storm-zero-llm",
                    "readme": self._read_readme(),
                }
            )
            return

        if self.path == "/about":
            body = self._read_readme()
            if body is None:
                self._send_json({"success": False, "status_code": 404, "error": "README.md not found"}, HTTPStatus.NOT_FOUND)
                return
            self._send_text(body, "text/markdown; charset=utf-8")
            return

        self._send_json({"success": False, "status_code": 404, "error": "not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if self.path == "/chat":
            self._handle_chat()
            return
        if self.path == "/use-model":
            self._handle_use_model()
            return
        if self.path == "/generate-image":
            self._handle_generate_image()
            return
        self._send_json({"success": False, "status_code": 404, "error": "not found"}, HTTPStatus.NOT_FOUND)

    def _handle_chat(self) -> None:
        try:
            payload = self._read_json()
            user_id = parse_user_id(payload.get("user_id"))
            prompt = str(payload.get("prompt", "")).strip()
            topics = [str(topic) for topic in payload.get("topics", []) if str(topic).strip()]
            proactive_allowed = bool(payload.get("proactive_allowed", False))
            include_reasoning = bool(payload.get("include_reasoning", False))
            tts_enabled = bool(payload.get("tts", False))
            voice = self._resolve_tts_voice(user_id)

            if not prompt:
                raise ValueError("prompt is required")

            chat_result = self.agent.chat(
                user_id=user_id,
                prompt=prompt,
                topics=topics,
                proactive_allowed=proactive_allowed,
                include_reasoning=include_reasoning,
                request_params=payload,
            )
            result = {
                "response": chat_result.response,
                "confidence": chat_result.confidence,
                "critique": chat_result.critique,
                "detected_type": chat_result.detected_type,
                "media": chat_result.media,
                "subject_id": chat_result.subject_id,
                "user_guideline_id": chat_result.user_guideline_id,
                "queries": chat_result.queries or [],
                "detection": chat_result.detection,
                "rules_evaluated": chat_result.rules_evaluated or {"global": False, "user": False},
                "prompts": chat_result.prompts or {},
            }
            if include_reasoning and chat_result.reasoning is not None:
                result["reasoning"] = chat_result.reasoning
        except ValueError as exc:
            self._send_json({"success": False, "status_code": 400, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        except RuntimeError as exc:
            self._send_runtime_error(str(exc), endpoint="chat")
            return
        except Exception as exc:
            self._send_json({"success": False, "status_code": 500, "error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        self._send_request_response(
            result=result,
            response_type=result["detected_type"],
            tts_enabled=tts_enabled,
            voice=voice,
            parameters=self._response_parameters(payload, result.get("detection")),
        )

    def _handle_use_model(self) -> None:
        try:
            payload = self._read_json()
            model_type = str(payload.get("type", "")).strip().lower()
            user_id = parse_user_id(payload.get("user_id"))
            model = str(payload.get("model", "")).strip() if payload.get("model") else None
            prompt = str(payload.get("prompt", "")).strip()
            include_reasoning = bool(payload.get("include_reasoning", False))
            tts_enabled = bool(payload.get("tts", False))
            voice = self._resolve_tts_voice(user_id)

            if model_type not in {"system", "user"}:
                raise ValueError("type must be 'system' or 'user'")
            if not prompt:
                raise ValueError("prompt is required")
            if not model:
                raise ValueError("model is required")

            if model_type == "system":
                model_path = self.agent.config.dedicated_models_path / model
            else:
                model_path = self.agent.config.custom_models_path / model

            use_model_result = self.agent.use_model(
                user_id=user_id,
                prompt=prompt,
                model_path=model_path,
                include_reasoning=include_reasoning,
                request_params=payload,
            )
            result = {
                "response": use_model_result.response,
                "confidence": use_model_result.confidence,
                "critique": use_model_result.critique,
                "detected_type": use_model_result.detected_type,
                "media": use_model_result.media,
                "subject_id": use_model_result.subject_id,
                "user_guideline_id": use_model_result.user_guideline_id,
                "queries": use_model_result.queries or [],
                "model": str(model_path),
                "prompt": prompt,
                "type": model_type,
                "rules_evaluated": use_model_result.rules_evaluated or {"global": False, "user": False},
                "detection": use_model_result.detection,
                "prompts": use_model_result.prompts or {},
            }
            if include_reasoning and use_model_result.reasoning is not None:
                result["reasoning"] = use_model_result.reasoning
        except ValueError as exc:
            self._send_json({"success": False, "status_code": 400, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        except RuntimeError as exc:
            self._send_runtime_error(str(exc), endpoint="use-model")
            return
        except Exception as exc:
            self._send_json({"success": False, "status_code": 500, "error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        self._send_request_response(
            result=result,
            response_type=result["detected_type"],
            tts_enabled=tts_enabled,
            voice=voice,
            include_model_fields=True,
            parameters=self._response_parameters(payload, result.get("detection")),
        )

    @staticmethod
    def _response_parameters(
        payload: dict[str, Any],
        detection: dict[str, Any] | None,
    ) -> dict[str, Any]:
        parameters = dict(payload)
        if detection is not None:
            parameters["detection"] = detection
        return parameters

    def _send_request_response(
        self,
        *,
        result: dict[str, Any],
        response_type: str,
        tts_enabled: bool,
        voice: str,
        include_model_fields: bool = False,
        parameters: dict[str, Any] | None = None,
    ) -> None:
        normalized_response = self._normalize_response_payload(result["response"])
        response_payload: dict[str, Any] = {
            "success": True,
            "status_code": 200,
            "type": response_type,
            "response": normalized_response,
            "confidence": result["confidence"],
            "critique": result["critique"],
            "queries": result.get("queries", []),
            "prompts": result.get("prompts", {}),
            "rules_evaluated": result.get("rules_evaluated", {"global": False, "user": False}),
            "parameters": parameters or {},
        }
        if result.get("reasoning") is not None:
            response_payload["reasoning"] = result["reasoning"]
        detection = result.get("detection")
        if detection is not None:
            response_payload["detection"] = detection
        subject_id = result.get("subject_id")
        if subject_id is not None:
            response_payload["subject_id"] = subject_id
        user_guideline_id = result.get("user_guideline_id")
        if user_guideline_id is not None:
            response_payload["user_guideline_id"] = user_guideline_id
        if include_model_fields:
            response_payload["model"] = result.get("model")
            response_payload["prompt"] = result.get("prompt")
            if result.get("type"):
                response_payload["type"] = result["type"]
        media_payload = result.get("media")
        if media_payload is not None:
            response_payload["media"] = media_payload
        if (
            tts_enabled
            and isinstance(normalized_response, str)
            and normalized_response.strip()
            and not self._is_data_url_response(normalized_response)
        ):
            response_payload["tts"] = self._tts_payload_from_text(normalized_response, voice=voice)
        self._send_json(response_payload)

    def _handle_generate_image(self) -> None:
        if not self._check_image_auth():
            self._send_json(
                {"success": False, "status_code": 401, "error": "unauthorized"},
                HTTPStatus.UNAUTHORIZED,
            )
            return
        try:
            payload = self._read_json()
            prompt = str(payload.get("prompt", "")).strip()
            if not prompt:
                raise ValueError("prompt is required")
            image = self._image_generator_instance().generate(prompt)
        except ValueError as exc:
            self._send_json({"success": False, "status_code": 400, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        except RuntimeError as exc:
            self._send_json({"success": False, "status_code": 502, "error": str(exc)}, HTTPStatus.BAD_GATEWAY)
            return
        except Exception as exc:
            self._send_json({"success": False, "status_code": 500, "error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        self._send_json({"success": True, "status_code": 200, "image": image})

    @classmethod
    def _image_generator_instance(cls) -> ImageGenerator:
        if cls._image_generator is None:
            config = cls.agent.config
            image_config = ImageGenerationConfig(
                flux_model=config.generate_image_model,
                coreml_path=config.image_coreml_path,
                lora_model=config.generate_image_lora_model,
                device=config.image_device,
                token=config.huggingface_token,
                cpu_offload=config.image_cpu_offload,
                sequential_cpu_offload=config.image_sequential_cpu_offload,
                compute_unit=config.image_compute_unit,
                num_inference_steps=config.image_num_inference_steps,
                guidance_scale=config.image_guidance_scale,
            )
            cls._image_generator = create_image_generator(image_config)
        return cls._image_generator

    def _check_image_auth(self) -> bool:
        expected = os.environ.get("IMAGE_API_KEY", "").strip()
        if not expected:
            return True
        authorization = self.headers.get("Authorization", "")
        return hmac.compare_digest(authorization, f"Bearer {expected}")

    def _read_json(self) -> dict[str, Any]:
        size = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(size) if size else b"{}"
        payload = json.loads(body.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Request body must be a JSON object")
        return payload

    def _read_readme(self) -> str | None:
        path = self.agent.config.readme_path
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, payload: str, content_type: str) -> None:
        body = payload.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_runtime_error(self, message: str, endpoint: str) -> None:
        if endpoint == "use-model":
            self._send_json(
                {
                    "success": False,
                    "status_code": 502,
                    "error": "Model runtime failure",
                    "error_code": "model_runtime_error",
                },
                HTTPStatus.BAD_GATEWAY,
            )
            return

        self._send_json(
            {
                "success": False,
                "status_code": 500,
                "error": message,
            },
            HTTPStatus.INTERNAL_SERVER_ERROR,
        )

    def _resolve_tts_voice(self, user_id: int) -> str:
        avatar_voice = self.agent.get_avatar_voice(user_id)
        if avatar_voice:
            return avatar_voice
        return "af_heart"

    def _tts_payload_from_text(self, text: str, voice: str = "af_heart") -> dict[str, Any]:
        try:
            result = self._tts_service_instance(voice).synthesize(text)
            return self._serialize_tts_result(result)
        except Exception as exc:
            return {
                "base64": None,
                "mime_type": None,
                "voice": voice,
                "error": str(exc),
            }

    def _tts_service_instance(self, voice: str) -> KokoroTTSService:
        return KokoroTTSService(self.agent.config.tts_path, voice=voice)

    @staticmethod
    def _serialize_tts_result(result: TTSResult) -> dict[str, Any]:
        return {
            "base64": result.audio_base64,
            "mime_type": result.mime_type,
            "voice": result.voice,
        }

    @classmethod
    def _is_data_url_response(cls, response: str) -> bool:
        first_line = response.strip().splitlines()[0] if response.strip() else ""
        return cls._file_data_url_start_pattern.match(first_line) is not None

    @classmethod
    def _normalize_response_payload(cls, response: str) -> str | dict[str, dict[str, str]]:
        options = cls._extract_options(response)
        if options:
            return {"potential_options": options}
        return response

    @staticmethod
    def _extract_options(text: str) -> dict[str, str]:
        options: dict[str, str] = {}
        pending_key: str | None = None

        for raw in text.splitlines():
            line = raw.strip().replace("*", "").strip()
            if not line:
                continue

            match = re.match(r"(?i)^option\s*([0-9]+|[a-z])\s*[:\-\)]\s*(.*)$", line)
            if match:
                suffix = match.group(1).strip().lower()
                key = f"option_{suffix}"
                value = match.group(2).strip().strip('"').strip("'")
                if value:
                    options[key] = value
                    pending_key = None
                else:
                    pending_key = key
                continue

            if pending_key is not None:
                value = line.strip().strip('"').strip("'")
                if value:
                    options[pending_key] = value
                    pending_key = None

        if len(options) < 2:
            return {}
        return options

    def log_message(self, fmt: str, *args: Any) -> None:
        return


def create_server(agent: StormZeroAgent, host: str, port: int) -> ThreadingHTTPServer:
    class BoundHandler(StormZeroRequestHandler):
        pass

    BoundHandler.agent = agent
    return ThreadingHTTPServer((host, port), BoundHandler)


def run_server(agent: StormZeroAgent, host: str, port: int) -> None:
    server = create_server(agent, host=host, port=port)
    print(f"LLM server started on http://{host}:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
