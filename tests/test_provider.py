import sys
import threading
import time
from typing import Any
from unittest.mock import MagicMock, patch

from storm_zero_llm.config import LLMRuntimeConfig
from storm_zero_llm.provider import (
    GenerationRequest,
    LlamaCppProvider,
    _LoadedModel,
    _THINK_TAG_CLOSE,
    _THINK_TAG_OPEN,
    _extract_model_output,
    _sanitize_output,
    _wrap_chat_handler,
)


def test_different_models_can_run_inference_concurrently(tmp_path) -> None:
    runtime = LLMRuntimeConfig(n_ctx=512, n_threads=1, n_gpu_layers=0)
    providers = []
    loaded_models = []

    for index in range(2):
        model_path = tmp_path / f"model-{index}.gguf"
        model_path.write_bytes(b"gguf")
        provider = LlamaCppProvider(model_path, runtime)
        mock_llama = MagicMock()
        started = threading.Event()
        release = threading.Event()

        def create_chat_completion(
            *args: Any,
            _started=started,
            _release=release,
            _index=index,
            **kwargs: Any,
        ) -> dict[str, Any]:
            _started.set()
            assert _release.wait(timeout=1.0)
            return {"choices": [{"message": {"content": f"model-{_index}"}}]}

        mock_llama.create_chat_completion.side_effect = create_chat_completion
        loaded = _LoadedModel(llama=mock_llama, chat_template_kwargs={"enable_thinking": False})
        loaded_models.append(loaded)
        providers.append((provider, loaded, started, release))

    with patch.object(LlamaCppProvider, "_get_or_load_model", side_effect=[loaded for _, loaded, _, _ in providers]):
        first_thread = threading.Thread(
            target=lambda: providers[0][0].generate(GenerationRequest(prompt="a", system_prompt="sys")),
        )
        second_thread = threading.Thread(
            target=lambda: providers[1][0].generate(GenerationRequest(prompt="b", system_prompt="sys")),
        )

        first_thread.start()
        second_thread.start()
        assert providers[0][2].wait(timeout=1.0)
        assert providers[1][2].wait(timeout=1.0)

        providers[0][3].set()
        providers[1][3].set()
        deadline = time.monotonic() + 1.0
        first_thread.join(timeout=max(0.0, deadline - time.monotonic()))
        second_thread.join(timeout=max(0.0, deadline - time.monotonic()))

    assert not first_thread.is_alive()
    assert not second_thread.is_alive()


def test_generate_does_not_pass_enable_thinking_to_create_chat_completion(tmp_path) -> None:
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"gguf")

    mock_llama = MagicMock()
    mock_llama.create_chat_completion.return_value = {
        "choices": [{"message": {"content": "Hello"}}]
    }

    runtime = LLMRuntimeConfig(n_ctx=512, n_threads=1, n_gpu_layers=0)
    provider = LlamaCppProvider(model_path, runtime)
    loaded = _LoadedModel(llama=mock_llama, chat_template_kwargs={"enable_thinking": False})

    with patch.object(provider, "_get_or_load_model", return_value=loaded):
        result = provider.generate(
            GenerationRequest(prompt="hi", system_prompt="sys", include_reasoning=True)
        )

    assert result.response == "Hello"
    mock_llama.reset.assert_called_once()
    mock_llama.create_chat_completion.assert_called_once_with(
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ],
    )
    assert loaded.chat_template_kwargs == {"enable_thinking": True}


def test_get_or_load_model_passes_batch_and_flash_attn(tmp_path) -> None:
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"gguf")
    runtime = LLMRuntimeConfig(
        n_ctx=2048,
        n_threads=4,
        n_gpu_layers=32,
        n_batch=1024,
        flash_attn=True,
    )
    provider = LlamaCppProvider(model_path, runtime)
    mock_llama = MagicMock()
    mock_llama.chat_handler = None
    mock_llama._chat_handlers = {}
    mock_llama.chat_format = "chatml"
    mock_llama_cpp = MagicMock()
    mock_llama_cpp.Llama.return_value = mock_llama

    with (
        patch.dict(LlamaCppProvider._loaded_models, {}, clear=True),
        patch.dict(LlamaCppProvider._model_locks, {}, clear=True),
        patch.dict(sys.modules, {"llama_cpp": mock_llama_cpp}),
        patch("storm_zero_llm.provider._wrap_chat_handler"),
    ):
        loaded = provider._get_or_load_model(provider._model_key())

    assert loaded.llama is mock_llama
    mock_llama_cpp.Llama.assert_called_once_with(
        model_path=str(model_path.resolve()),
        n_ctx=2048,
        n_threads=4,
        n_gpu_layers=32,
        n_batch=1024,
        flash_attn=True,
    )


def test_wrap_chat_handler_forwards_template_kwargs() -> None:
    captured: dict[str, Any] = {}

    def base_handler(*args, **kwargs):
        captured.update(kwargs)
        return {"choices": []}

    model = MagicMock()
    model.chat_handler = base_handler
    model.chat_format = "chatml"
    model._chat_handlers = {}

    template_kwargs = {"enable_thinking": False}
    _wrap_chat_handler(model, template_kwargs)
    model.chat_handler(messages=[], enable_thinking=True)

    assert captured["enable_thinking"] is True
    template_kwargs["enable_thinking"] = True
    model.chat_handler(messages=[])
    assert captured["enable_thinking"] is True


def test_sanitize_output_strips_redacted_thinking_block() -> None:
    raw = (
        "<think>\nThinking Process:\n\n1. Analyze the request.\n"
        "</think>\n\nHey, test received."
    )
    assert _sanitize_output(raw) == "Hey, test received."


def test_sanitize_output_strips_think_tags() -> None:
    raw = (
        f"{_THINK_TAG_OPEN}\nReasoning here.\n{_THINK_TAG_CLOSE}\n\n"
        "Final answer here."
    )
    assert _sanitize_output(raw) == "Final answer here."


def test_sanitize_output_uses_final_response_marker() -> None:
    raw = (
        "Thinking Process:\n\n1. Analyze the request.\n\n"
        "Final Response:\nYep, I'm here."
    )
    assert _sanitize_output(raw) == "Yep, I'm here."


def test_sanitize_output_returns_empty_for_unclosed_thinking() -> None:
    raw = f"{_THINK_TAG_OPEN}\nThinking Process:\n\n1. Analyze the request."
    assert _sanitize_output(raw) == ""


def test_extract_model_output_splits_reasoning_and_response() -> None:
    reasoning, response = _extract_model_output(
        "<think>\nThinking Process:\n\n1. Analyze the request.\n"
        "</think>\n\nHey, test received."
    )
    assert reasoning == "Thinking Process:\n\n1. Analyze the request."
    assert response == "Hey, test received."

