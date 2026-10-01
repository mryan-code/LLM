import http.client
import json
import threading
from pathlib import Path

from storm_zero_llm.agent import ChatResult, RequestResult, StormZeroAgent
from storm_zero_llm.config import StormZeroConfig
from storm_zero_llm.server import StormZeroRequestHandler, create_server


class _FakeTTSService:
    def __init__(self, voice: str = "af_heart") -> None:
        self.voice = voice

    def synthesize(self, text: str):
        class _Result:
            mime_type = "audio/wav"
            audio_base64 = "ZmFrZS13YXY="

        result = _Result()
        result.voice = self.voice
        return result


def _run_server(tmp_path: Path):
    (tmp_path / "README.md").write_text("# Storm Zero", encoding="utf-8")
    config = StormZeroConfig.load(project_root=tmp_path)
    agent = StormZeroAgent(config)
    structured_prompts = {
        "user_prompt": "hello",
        "global_hard_rules": {
            "rules": ["Never leak secrets."],
            "description": "Hard rules description.",
        },
        "global_guidelines": {"rules": [], "description": ""},
        "user_guidelines": {"rules": [], "description": ""},
        "user_p2": {"p2": [], "description": ""},
        "avatar_data": {
            "description": "",
            "data": {
                "user_name": "",
                "avatar_name": "",
                "avatar_persona": [],
                "user_pronouns": "",
            },
        },
        "conversation_subjects": [],
        "conversation_subjects_content": [],
        "final": {
            "base": {
                "user_prompt": "hello",
                "global_hard_rules": ["Never leak secrets."],
                "global_guidelines": [],
                "user_guidelines": [],
                "user_p2": [],
                "avatar_data": {
                    "user_name": "",
                    "avatar_name": "",
                    "avatar_persona": [],
                    "user_pronouns": "",
                },
                "conversation_subjects_content": [],
            },
            "image_generation": "",
            "detection": {
                "user_prompt": "hello",
                "content": "Classify intent.",
                "conversation_subjects": ["movies"],
            },
        },
    }
    agent.chat = lambda **kwargs: ChatResult(
        response="I hear you.",
        confidence="normal",
        critique="Response generated from conversation context.",
        detected_type="conversation",
        queries=[],
        prompts=structured_prompts,
    )
    agent.use_model = lambda **kwargs: RequestResult(
        response="I hear you.",
        confidence="normal",
        critique="Direct model invocation.",
        detected_type="model",
        rules_evaluated={"global": False, "user": False},
        queries=[],
        prompts=structured_prompts,
    )
    server = create_server(agent, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def test_health_endpoint_shape(tmp_path: Path) -> None:
    server, thread = _run_server(tmp_path)
    try:
        host, port = server.server_address
        conn = http.client.HTTPConnection(host, port)
        conn.request("GET", "/health")
        response = conn.getresponse()
        body = json.loads(response.read().decode("utf-8"))
    finally:
        server.shutdown()
        thread.join(timeout=2)

    assert response.status == 200
    assert body["success"] is True
    assert body["status_code"] == 200


def test_chat_endpoint_shape(tmp_path: Path) -> None:
    server, thread = _run_server(tmp_path)
    try:
        host, port = server.server_address
        conn = http.client.HTTPConnection(host, port)
        conn.request(
            "POST",
            "/chat",
            body=json.dumps(
                {
                    "user_id": 1,
                    "prompt": "What do you remember about movies?",
                    "topics": ["movies"],
                }
            ),
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        body = json.loads(response.read().decode("utf-8"))
    finally:
        server.shutdown()
        thread.join(timeout=2)

    assert response.status == 200
    assert body["success"] is True
    assert body["status_code"] == 200
    assert body["type"] == "conversation"
    assert body["rules_evaluated"] == {"global": False, "user": False}
    assert body["queries"] == []
    assert body["prompts"]["user_prompt"] == "hello"
    assert body["prompts"]["global_hard_rules"]["rules"] == ["Never leak secrets."]
    assert body["prompts"]["final"]["base"]["user_prompt"] == "hello"
    assert body["prompts"]["final"]["base"]["global_hard_rules"] == ["Never leak secrets."]
    assert body["prompts"]["final"]["detection"]["user_prompt"] == "hello"
    assert body["prompts"]["final"]["detection"]["content"] == "Classify intent."
    assert body["parameters"] == {
        "user_id": 1,
        "prompt": "What do you remember about movies?",
        "topics": ["movies"],
    }
    assert "response" in body
    assert "tts" not in body
    assert "reasoning" not in body


def test_chat_endpoint_uses_rules_evaluated_from_chat_result(tmp_path: Path) -> None:
    server, thread = _run_server(tmp_path)
    original_chat = server.RequestHandlerClass.agent.chat
    load_rule_context_calls = 0
    original_load_rule_context = server.RequestHandlerClass.agent._load_rule_context

    def _track_load_rule_context(user_id: int):
        nonlocal load_rule_context_calls
        load_rule_context_calls += 1
        return original_load_rule_context(user_id)

    try:
        host, port = server.server_address

        def _chat_with_rules(**kwargs):
            return ChatResult(
                response="I hear you.",
                confidence="normal",
                critique="test",
                detected_type="conversation",
                queries=[],
                rules_evaluated={"global": True, "user": True},
            )

        server.RequestHandlerClass.agent.chat = _chat_with_rules
        server.RequestHandlerClass.agent._load_rule_context = _track_load_rule_context

        conn = http.client.HTTPConnection(host, port)
        conn.request(
            "POST",
            "/chat",
            body=json.dumps({"user_id": 1, "prompt": "hello"}),
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        body = json.loads(response.read().decode("utf-8"))
    finally:
        server.RequestHandlerClass.agent.chat = original_chat
        server.RequestHandlerClass.agent._load_rule_context = original_load_rule_context
        server.shutdown()
        thread.join(timeout=2)

    assert response.status == 200
    assert body["rules_evaluated"] == {"global": True, "user": True}
    assert load_rule_context_calls == 0


def test_chat_endpoint_includes_reasoning_when_requested(tmp_path: Path) -> None:
    server, thread = _run_server(tmp_path)
    original_chat = server.RequestHandlerClass.agent.chat
    try:
        host, port = server.server_address

        def _chat_with_reasoning(**kwargs):
            return ChatResult(
                response="Hey, test received.",
                confidence="normal",
                critique="test",
                detected_type="conversation",
                reasoning="Thinking Process:\n\n1. Analyze the request.",
                queries=[],
            )

        server.RequestHandlerClass.agent.chat = _chat_with_reasoning
        conn = http.client.HTTPConnection(host, port)
        conn.request(
            "POST",
            "/chat",
            body=json.dumps(
                {
                    "user_id": 1,
                    "prompt": "this is a test",
                    "include_reasoning": True,
                }
            ),
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        body = json.loads(response.read().decode("utf-8"))
    finally:
        server.RequestHandlerClass.agent.chat = original_chat
        server.shutdown()
        thread.join(timeout=2)

    assert response.status == 200
    assert body["response"] == "Hey, test received."
    assert body["reasoning"] == "Thinking Process:\n\n1. Analyze the request."


def test_chat_endpoint_includes_detection_in_parameters(tmp_path: Path) -> None:
    server, thread = _run_server(tmp_path)
    original_chat = server.RequestHandlerClass.agent.chat
    try:
        host, port = server.server_address

        def _chat_with_detection(**kwargs):
            return ChatResult(
                response="I hear you.",
                confidence="normal",
                critique="test",
                detected_type="conversation",
                queries=[],
                detection={
                    "action": "general_query",
                    "media_type": None,
                    "user_data_target": None,
                    "conversation_mode": "new",
                    "subject_id": None,
                    "subject_summary": "hello",
                },
            )

        server.RequestHandlerClass.agent.chat = _chat_with_detection

        conn = http.client.HTTPConnection(host, port)
        conn.request(
            "POST",
            "/chat",
            body=json.dumps({"user_id": 1, "prompt": "hello"}),
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        body = json.loads(response.read().decode("utf-8"))
    finally:
        server.RequestHandlerClass.agent.chat = original_chat
        server.shutdown()
        thread.join(timeout=2)

    assert response.status == 200
    assert body["parameters"]["prompt"] == "hello"
    assert body["detection"]["action"] == "general_query"
    assert body["detection"]["conversation_mode"] == "new"
    assert body["parameters"]["detection"]["action"] == "general_query"
    assert body["parameters"]["detection"]["conversation_mode"] == "new"


def test_use_model_requires_system_option(tmp_path: Path) -> None:
    server, thread = _run_server(tmp_path)
    try:
        host, port = server.server_address
        conn = http.client.HTTPConnection(host, port)
        conn.request(
            "POST",
            "/use-model",
            body=json.dumps({"type": "system", "user_id": 1, "prompt": "hello"}),
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        body = json.loads(response.read().decode("utf-8"))
    finally:
        server.shutdown()
        thread.join(timeout=2)

    assert response.status == 400
    assert body["success"] is False


def test_use_model_requires_user_id(tmp_path: Path) -> None:
    server, thread = _run_server(tmp_path)
    try:
        host, port = server.server_address
        conn = http.client.HTTPConnection(host, port)
        conn.request(
            "POST",
            "/use-model",
            body=json.dumps({"type": "system", "option": "text", "prompt": "hello"}),
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        body = json.loads(response.read().decode("utf-8"))
    finally:
        server.shutdown()
        thread.join(timeout=2)

    assert response.status == 400
    assert body["success"] is False
    assert body["error"] == "user_id is required"


def test_use_model_response_includes_tts_shape(tmp_path: Path) -> None:
    original_tts_factory = StormZeroRequestHandler._tts_service_instance
    StormZeroRequestHandler._tts_service_instance = lambda self, voice: _FakeTTSService(voice)
    server, thread = _run_server(tmp_path)
    try:
        host, port = server.server_address
        conn = http.client.HTTPConnection(host, port)
        conn.request(
            "POST",
            "/use-model",
            body=json.dumps(
                {
                    "type": "system",
                    "user_id": 1,
                    "option": "text",
                    "model": "text.gguf",
                    "prompt": "hello",
                    "tts": True,
                }
            ),
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        body = json.loads(response.read().decode("utf-8"))
    finally:
        server.shutdown()
        thread.join(timeout=2)
        StormZeroRequestHandler._tts_service_instance = original_tts_factory

    assert response.status == 200
    assert body["success"] is True
    assert body["rules_evaluated"] == {"global": False, "user": False}
    assert body["parameters"] == {
        "type": "system",
        "user_id": 1,
        "option": "text",
        "model": "text.gguf",
        "prompt": "hello",
        "tts": True,
    }
    assert body["tts"] == {
        "base64": "ZmFrZS13YXY=",
        "mime_type": "audio/wav",
        "voice": "af_heart",
    }


def test_chat_response_omits_tts_when_disabled(tmp_path: Path) -> None:
    server, thread = _run_server(tmp_path)
    try:
        host, port = server.server_address
        conn = http.client.HTTPConnection(host, port)
        conn.request(
            "POST",
            "/chat",
            body=json.dumps(
                {
                    "user_id": 1,
                    "prompt": "Tell me something",
                    "tts": False,
                }
            ),
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        body = json.loads(response.read().decode("utf-8"))
    finally:
        server.shutdown()
        thread.join(timeout=2)

    assert response.status == 200
    assert body["success"] is True
    assert body["rules_evaluated"] == {"global": False, "user": False}
    assert "tts" not in body


def test_chat_response_includes_tts_when_enabled(tmp_path: Path) -> None:
    original_tts_factory = StormZeroRequestHandler._tts_service_instance
    StormZeroRequestHandler._tts_service_instance = lambda self, voice: _FakeTTSService(voice)
    server, thread = _run_server(tmp_path)
    try:
        host, port = server.server_address
        conn = http.client.HTTPConnection(host, port)
        conn.request(
            "POST",
            "/chat",
            body=json.dumps(
                {
                    "user_id": 1,
                    "prompt": "Tell me something",
                    "tts": True,
                }
            ),
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        body = json.loads(response.read().decode("utf-8"))
    finally:
        server.shutdown()
        thread.join(timeout=2)
        StormZeroRequestHandler._tts_service_instance = original_tts_factory

    assert response.status == 200
    assert body["success"] is True
    assert body["tts"] == {
        "base64": "ZmFrZS13YXY=",
        "mime_type": "audio/wav",
        "voice": "af_heart",
    }


def test_chat_response_uses_avatar_voice_for_tts(tmp_path: Path) -> None:
    captured_voices: list[str] = []
    original_tts_factory = StormZeroRequestHandler._tts_service_instance

    def _capture_voice(self, voice: str) -> _FakeTTSService:
        captured_voices.append(voice)
        return _FakeTTSService(voice)

    StormZeroRequestHandler._tts_service_instance = _capture_voice
    server, thread = _run_server(tmp_path)
    server.RequestHandlerClass.agent.get_avatar_voice = lambda user_id: "af_bella"
    try:
        host, port = server.server_address
        conn = http.client.HTTPConnection(host, port)
        conn.request(
            "POST",
            "/chat",
            body=json.dumps(
                {
                    "user_id": 1,
                    "prompt": "Tell me something",
                    "tts": True,
                }
            ),
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        body = json.loads(response.read().decode("utf-8"))
    finally:
        server.shutdown()
        thread.join(timeout=2)
        StormZeroRequestHandler._tts_service_instance = original_tts_factory

    assert response.status == 200
    assert captured_voices == ["af_bella"]
    assert body["tts"] == {
        "base64": "ZmFrZS13YXY=",
        "mime_type": "audio/wav",
        "voice": "af_bella",
    }


def test_chat_response_ignores_voice_parameter(tmp_path: Path) -> None:
    captured_voices: list[str] = []
    original_tts_factory = StormZeroRequestHandler._tts_service_instance

    def _capture_voice(self, voice: str) -> _FakeTTSService:
        captured_voices.append(voice)
        return _FakeTTSService(voice)

    StormZeroRequestHandler._tts_service_instance = _capture_voice
    server, thread = _run_server(tmp_path)
    server.RequestHandlerClass.agent.get_avatar_voice = lambda user_id: "af_bella"
    try:
        host, port = server.server_address
        conn = http.client.HTTPConnection(host, port)
        conn.request(
            "POST",
            "/chat",
            body=json.dumps(
                {
                    "user_id": 1,
                    "prompt": "Tell me something",
                    "tts": True,
                    "voice": "af_heart",
                }
            ),
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        json.loads(response.read().decode("utf-8"))
    finally:
        server.shutdown()
        thread.join(timeout=2)
        StormZeroRequestHandler._tts_service_instance = original_tts_factory

    assert response.status == 200
    assert captured_voices == ["af_bella"]


def test_use_model_response_omits_tts_when_disabled(tmp_path: Path) -> None:
    server, thread = _run_server(tmp_path)
    try:
        host, port = server.server_address
        conn = http.client.HTTPConnection(host, port)
        conn.request(
            "POST",
            "/use-model",
            body=json.dumps(
                {
                    "type": "system",
                    "user_id": 1,
                    "option": "text",
                    "model": "text.gguf",
                    "prompt": "hello",
                    "tts": False,
                }
            ),
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        body = json.loads(response.read().decode("utf-8"))
    finally:
        server.shutdown()
        thread.join(timeout=2)

    assert response.status == 200
    assert body["success"] is True
    assert "tts" not in body


def test_chat_omits_file_field_when_response_contains_base64_data_url(tmp_path: Path) -> None:
    server, thread = _run_server(tmp_path)
    original_chat = server.RequestHandlerClass.agent.chat
    try:
        host, port = server.server_address

        def _chat_with_file(**kwargs):
            return ChatResult(
                response=(
                    "Generated file below:\n"
                    "data:application/pdf;base64,VGhpcyBpcyBhIHRlc3Q=\n"
                    "Download complete."
                ),
                confidence="normal",
                critique="test",
                detected_type="conversation",
            )

        server.RequestHandlerClass.agent.chat = _chat_with_file

        conn = http.client.HTTPConnection(host, port)
        conn.request(
            "POST",
            "/chat",
            body=json.dumps({"user_id": 1, "prompt": "send a file"}),
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        body = json.loads(response.read().decode("utf-8"))
    finally:
        server.RequestHandlerClass.agent.chat = original_chat
        server.shutdown()
        thread.join(timeout=2)

    assert response.status == 200
    assert "file" not in body


def test_use_model_omits_file_field_when_response_contains_base64_data_url(tmp_path: Path) -> None:
    server, thread = _run_server(tmp_path)
    original_use_model = server.RequestHandlerClass.agent.use_model
    try:
        host, port = server.server_address

        def _use_model_with_file(**kwargs):
            return RequestResult(
                response=(
                    "Here is your file:\n"
                    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUA\n"
                    "AAAAFQ=="
                ),
                confidence="normal",
                critique="test",
                detected_type="model",
                rules_evaluated={"global": False, "user": False},
            )

        server.RequestHandlerClass.agent.use_model = _use_model_with_file

        conn = http.client.HTTPConnection(host, port)
        conn.request(
            "POST",
            "/use-model",
            body=json.dumps(
                {
                    "type": "system",
                    "user_id": 1,
                    "option": "text",
                    "model": "text.gguf",
                    "prompt": "send a file",
                }
            ),
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        body = json.loads(response.read().decode("utf-8"))
    finally:
        server.RequestHandlerClass.agent.use_model = original_use_model
        server.shutdown()
        thread.join(timeout=2)

    assert response.status == 200
    assert "file" not in body


def test_chat_returns_potential_options_object_when_multiple_options_detected(tmp_path: Path) -> None:
    server, thread = _run_server(tmp_path)
    original_chat = server.RequestHandlerClass.agent.chat
    try:
        host, port = server.server_address

        def _chat_with_options(**kwargs):
            return ChatResult(
                response='Option 1: "Alpha"\nOption 2: "Beta"',
                confidence="normal",
                critique="test",
                detected_type="conversation",
            )

        server.RequestHandlerClass.agent.chat = _chat_with_options

        conn = http.client.HTTPConnection(host, port)
        conn.request(
            "POST",
            "/chat",
            body=json.dumps({"user_id": 1, "prompt": "This is a test"}),
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        body = json.loads(response.read().decode("utf-8"))
    finally:
        server.RequestHandlerClass.agent.chat = original_chat
        server.shutdown()
        thread.join(timeout=2)

    assert response.status == 200
    assert body["response"] == {
        "potential_options": {
            "option_1": "Alpha",
            "option_2": "Beta",
        }
    }


def test_use_model_returns_potential_options_object_when_multiple_options_detected(tmp_path: Path) -> None:
    server, thread = _run_server(tmp_path)
    original_use_model = server.RequestHandlerClass.agent.use_model
    try:
        host, port = server.server_address

        def _use_model_with_options(**kwargs):
            return RequestResult(
                response="Option 1: Keep it short\nOption 2: Keep it detailed",
                confidence="normal",
                critique="test",
                detected_type="model",
                rules_evaluated={"global": False, "user": False},
            )

        server.RequestHandlerClass.agent.use_model = _use_model_with_options

        conn = http.client.HTTPConnection(host, port)
        conn.request(
            "POST",
            "/use-model",
            body=json.dumps(
                {
                    "type": "system",
                    "user_id": 1,
                    "option": "text",
                    "model": "text.gguf",
                    "prompt": "This is a test",
                }
            ),
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        body = json.loads(response.read().decode("utf-8"))
    finally:
        server.RequestHandlerClass.agent.use_model = original_use_model
        server.shutdown()
        thread.join(timeout=2)

    assert response.status == 200
    assert body["response"] == {
        "potential_options": {
            "option_1": "Keep it short",
            "option_2": "Keep it detailed",
        }
    }


def test_use_model_returns_502_for_model_runtime_failure(tmp_path: Path) -> None:
    server, thread = _run_server(tmp_path)
    original_use_model = server.RequestHandlerClass.agent.use_model
    try:
        host, port = server.server_address

        def _use_model_runtime_error(**kwargs):
            raise RuntimeError("model backend exploded")

        server.RequestHandlerClass.agent.use_model = _use_model_runtime_error

        conn = http.client.HTTPConnection(host, port)
        conn.request(
            "POST",
            "/use-model",
            body=json.dumps(
                {
                    "type": "system",
                    "user_id": 1,
                    "option": "text",
                    "model": "text.gguf",
                    "prompt": "This is a test",
                }
            ),
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        body = json.loads(response.read().decode("utf-8"))
    finally:
        server.RequestHandlerClass.agent.use_model = original_use_model
        server.shutdown()
        thread.join(timeout=2)

    assert response.status == 502
    assert body == {
        "success": False,
        "status_code": 502,
        "error": "Model runtime failure",
        "error_code": "model_runtime_error",
    }


def test_chat_routes_media_prompts_to_image_model(tmp_path: Path) -> None:
    server, thread = _run_server(tmp_path)
    original_chat = server.RequestHandlerClass.agent.chat
    try:
        host, port = server.server_address

        def _chat_with_media(**kwargs):
            return ChatResult(
                response="",
                confidence="normal",
                critique="Media request routed to GENERATE_IMAGE_MODEL.",
                detected_type="media",
                media={"mime_type": "image/png", "base64": "iVBORw0KGgo="},
            )

        server.RequestHandlerClass.agent.chat = _chat_with_media

        conn = http.client.HTTPConnection(host, port)
        conn.request(
            "POST",
            "/chat",
            body=json.dumps({"user_id": 1, "prompt": "generate an image of a cat", "tts": True}),
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        body = json.loads(response.read().decode("utf-8"))
    finally:
        server.RequestHandlerClass.agent.chat = original_chat
        server.shutdown()
        thread.join(timeout=2)

    assert response.status == 200
    assert body["type"] == "media"
    assert body["response"] == ""
    assert body["media"] == {
        "mime_type": "image/png",
        "base64": "iVBORw0KGgo=",
    }
    assert "file" not in body
    assert "tts" not in body


def test_use_model_routes_media_prompts_to_music_model(tmp_path: Path) -> None:
    server, thread = _run_server(tmp_path)
    original_use_model = server.RequestHandlerClass.agent.use_model
    try:
        host, port = server.server_address

        def _use_model_with_media(**kwargs):
            model_path = kwargs["model_path"]
            return RequestResult(
                response="music bytes",
                confidence="normal",
                critique="Media request routed to MUSIC_MODEL.",
                detected_type="media",
                media={"mime_type": "audio/wav", "base64": "bXVzaWM="},
                rules_evaluated={"global": False, "user": False},
            )

        server.RequestHandlerClass.agent.use_model = _use_model_with_media

        conn = http.client.HTTPConnection(host, port)
        conn.request(
            "POST",
            "/use-model",
            body=json.dumps(
                {
                    "type": "system",
                    "user_id": 1,
                    "option": "text",
                    "model": "ignored.gguf",
                    "prompt": "compose music for a game",
                }
            ),
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        body = json.loads(response.read().decode("utf-8"))
    finally:
        server.RequestHandlerClass.agent.use_model = original_use_model
        server.shutdown()
        thread.join(timeout=2)

    assert response.status == 200
    assert body["media"] == {
        "mime_type": "audio/wav",
        "base64": "bXVzaWM=",
    }
    assert "file" not in body
