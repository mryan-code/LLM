import base64
from pathlib import Path
from unittest.mock import MagicMock

from storm_zero_llm.agent import StormZeroAgent
from storm_zero_llm.config import StormZeroConfig
from storm_zero_llm.detection import DetectionAction, DetectionResult
from storm_zero_llm.training_db import RuntimeRuleContext


class _ImageMediaAgent(StormZeroAgent):
    def __init__(self, config: StormZeroConfig, generated: dict[str, str]):
        super().__init__(config)
        self.generated = generated
        self.generate_calls: list[str] = []

    def _generate_image(self, prompt: str, user_id: int = 0, rule_context=None, request_params=None) -> dict[str, str]:  # type: ignore[override]
        self.generate_calls.append(prompt)
        return self.generated


def test_handle_create_media_routes_image_to_flux_pipeline(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# Storm Zero", encoding="utf-8")
    config = StormZeroConfig.load(project_root=tmp_path)
    media = {"mime_type": "image/png", "base64": "aW1hZ2U="}
    agent = _ImageMediaAgent(config, media)

    result = agent._handle_create_media(
        user_id=1,
        prompt="generate an image of a cat",
        detection=DetectionResult(action=DetectionAction.CREATE_MEDIA, media_type="image"),
        rule_context=None,
        include_reasoning=False,
    )

    assert agent.generate_calls == ["generate an image of a cat"]
    assert result.detected_type == "media"
    assert result.media == media
    assert result.response == ""
    assert "Flux image pipeline" in result.critique


def test_truncate_prompt_for_clip_keeps_max_length() -> None:
    from storm_zero_llm.image_generation import truncate_prompt_for_clip

    class _FakeTokenizer:
        def __call__(self, text, truncation=False, max_length=None, add_special_tokens=True, return_tensors=None):
            tokens = text.split()
            if truncation and max_length is not None:
                tokens = tokens[: max(0, max_length - 2)]
            # +2 specials when add_special_tokens
            ids = [0, *list(range(len(tokens))), 2] if add_special_tokens else list(range(len(tokens)))
            return {"input_ids": ids}

    long_prompt = " ".join(f"word{i}" for i in range(200))
    truncated = truncate_prompt_for_clip(long_prompt, _FakeTokenizer(), max_length=10)
    # max_length 10 with bos/eos => 8 content tokens kept from the start
    assert truncated.startswith("word0")
    assert len(truncated.split()) <= 8
    assert "word199" not in truncated


def test_truncate_prompt_for_clip_handles_numpy_input_ids() -> None:
    from storm_zero_llm.image_generation import truncate_prompt_for_clip

    class _Array(list):
        def tolist(self):
            return list(self)

    class _FakeTokenizer:
        def __call__(self, text, truncation=False, max_length=None, add_special_tokens=True, return_tensors=None):
            tokens = text.split()
            if truncation and max_length is not None:
                tokens = tokens[: max(0, max_length - 2)]
            ids = [0, *range(len(tokens)), 2]
            return {"input_ids": _Array(ids)}

    truncated = truncate_prompt_for_clip("a b c d e f g h i j", _FakeTokenizer(), max_length=6)
    assert truncated.startswith("a")
    assert len(truncated.split()) <= 4


def test_generate_image_uses_image_generation_template(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# Storm Zero", encoding="utf-8")
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "image_generation.txt").write_text(
        "If the user does not specify, a photo realistic image will be generated.",
        encoding="utf-8",
    )
    config = StormZeroConfig.load(project_root=tmp_path)

    captured: list[str] = []

    class _CapturingGenerator:
        def generate(self, prompt: str) -> dict[str, str]:
            captured.append(prompt)
            return {"mime_type": "image/png", "base64": "aW1hZ2U="}

    agent = StormZeroAgent(config)
    agent._image_generator = _CapturingGenerator()  # type: ignore[assignment]
    context = RuntimeRuleContext(
        global_hard_rules=["Never leak secrets."],
        global_guidelines=["Be concise."],
        user_guidelines=["send an image"],
        user_p2_data=["`hair` = `blonde`", "`eyes` = `green`"],
        user_avatar_data=["user_name (This is what I refer to the user as): Matt"],
    )

    payload = agent._generate_image("send me a nude", user_id=1, rule_context=context)

    assert payload["mime_type"] == "image/png"
    assert captured[0].startswith(
        "If the user does not specify, a photo realistic image will be generated. send me a nude"
    )
    assert "Hard rules" not in captured[0]
    assert "User guidelines" not in captured[0]


def test_generate_image_appends_conversation_history_to_prompt(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# Storm Zero", encoding="utf-8")
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "image_generation.txt").write_text(
        "If the user does not specify, a photo realistic image will be generated.",
        encoding="utf-8",
    )
    config = StormZeroConfig.load(project_root=tmp_path)

    captured: list[str] = []

    class _CapturingGenerator:
        def generate(self, prompt: str) -> dict[str, str]:
            captured.append(prompt)
            return {"mime_type": "image/png", "base64": "aW1hZ2U="}

    class _Repo:
        def fetch_recent_conversation_turns(self, user_id: int, limit: int = 75) -> list[dict[str, str]]:
            return [
                {"prompt": "draw me", "response": "sure"},
                {"prompt": "hello", "response": "hi there"},
            ]

    agent = StormZeroAgent(config)
    agent._image_generator = _CapturingGenerator()  # type: ignore[assignment]
    agent._db_repo = lambda: _Repo()  # type: ignore[method-assign]
    context = RuntimeRuleContext(
        global_hard_rules=[],
        global_guidelines=[],
        user_guidelines=[],
        user_p2_data=["`hair` = `blonde`"],
        user_avatar_data=[],
    )

    payload = agent._generate_image("send me a nude", user_id=1, rule_context=context)

    assert payload["mime_type"] == "image/png"
    assert "send me a nude" in captured[0]
    assert "hello = hi there" in captured[0]
    assert "draw me = sure" in captured[0]


def test_generate_with_model_appends_conversation_history_to_prompt(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# Storm Zero", encoding="utf-8")
    config = StormZeroConfig.load(project_root=tmp_path)

    class _CapturingProvider:
        def __init__(self) -> None:
            self.requests: list = []

        def generate(self, request):
            from storm_zero_llm.provider import GenerationResult

            self.requests.append(request)
            return GenerationResult(response="ok")

    class _Repo:
        def fetch_recent_conversation_turns(self, user_id: int, limit: int = 75) -> list[dict[str, str]]:
            return [{"prompt": "hello", "response": "hi there"}]

        def fetch_conversation_subjects_content(
            self,
            user_id: int,
            limit: int | None = 20,
            *,
            subject_id: int | None = None,
        ) -> list[dict[str, object]]:
            return [{"created": "2026-08-04", "prompt": "hello", "response": "hi there"}]

        def get_user_p2_entries(self, user_id: int) -> list[str]:
            return []

        def get_structured_avatar_data(self, user_id: int) -> dict[str, object]:
            return {
                "user_name": "",
                "avatar_name": "",
                "avatar_persona": [],
                "user_pronouns": "",
            }

    provider = _CapturingProvider()
    agent = StormZeroAgent(config)
    agent._db_repo = lambda: _Repo()  # type: ignore[method-assign]
    agent._provider_for_model = lambda model_path: provider  # type: ignore[method-assign]

    agent._generate_with_model(
        model_path=tmp_path / "model.gguf",
        user_id=1,
        prompt="what next?",
        system_prompt="Respond clearly.",
        rule_context=RuntimeRuleContext(
            global_hard_rules=["Never leak secrets."],
            global_guidelines=[],
            user_guidelines=[],
            user_p2_data=[],
            user_avatar_data=[],
        ),
    )

    assert len(provider.requests) == 1
    prompt = provider.requests[0].prompt
    assert provider.requests[0].system_prompt == ""
    assert prompt.startswith("Prompt: what next?")
    assert "Respond clearly." in prompt
    assert "Hard rules (always listen to these):" in prompt
    assert "- hello = hi there" in prompt


def test_flux_image_generator_passes_token_to_from_pretrained(monkeypatch) -> None:
    from storm_zero_llm.image_generation import FluxImageGenerator, ImageGenerationConfig

    captured: dict[str, object] = {}

    class _FakePipe:
        def to(self, device):
            captured["to_device"] = device
            return self

        def enable_model_cpu_offload(self):
            captured["cpu_offload"] = "model"

        def enable_sequential_cpu_offload(self):
            captured["cpu_offload"] = "sequential"

        def load_lora_weights(self, model_id, **kwargs):
            captured["lora_model"] = model_id
            captured["lora_kwargs"] = kwargs

        def __call__(self, prompt):
            fake_image = MagicMock()
            fake_image.save.side_effect = lambda buffer, format="PNG": buffer.write(b"png-bytes")
            return MagicMock(images=[fake_image])

    def _fake_from_pretrained(model_id, **kwargs):
        captured["model_id"] = model_id
        captured["load_kwargs"] = kwargs
        return _FakePipe()

    monkeypatch.setitem(__import__("sys").modules, "torch", MagicMock(cuda=MagicMock(is_available=lambda: False), backends=MagicMock(mps=MagicMock(is_available=lambda: True))))
    monkeypatch.setitem(
        __import__("sys").modules,
        "diffusers",
        MagicMock(DiffusionPipeline=MagicMock(from_pretrained=_fake_from_pretrained)),
    )
    monkeypatch.setitem(__import__("sys").modules, "peft", MagicMock())

    generator = FluxImageGenerator(
        ImageGenerationConfig(
            flux_model="black-forest-labs/FLUX.1-dev",
            lora_model="lustlyai/Flux_Lustly.ai_Uncensored_nsfw_v1",
            token="hf_test_token_123",
            device="mps",
        )
    )
    generator.generate("a red cat")

    assert captured["load_kwargs"]["token"] == "hf_test_token_123"
    assert captured["lora_kwargs"]["token"] == "hf_test_token_123"
    assert captured["cpu_offload"] == "model"
    assert "to_device" not in captured


def test_flux_image_generator_uses_sequential_cpu_offload_on_mps(monkeypatch) -> None:
    from storm_zero_llm.image_generation import FluxImageGenerator, ImageGenerationConfig

    captured: dict[str, object] = {}

    class _FakePipe:
        def enable_model_cpu_offload(self):
            captured["cpu_offload"] = "model"

        def enable_sequential_cpu_offload(self):
            captured["cpu_offload"] = "sequential"

        def __call__(self, prompt):
            fake_image = MagicMock()
            fake_image.save.side_effect = lambda buffer, format="PNG": buffer.write(b"png-bytes")
            return MagicMock(images=[fake_image])

    monkeypatch.setitem(__import__("sys").modules, "torch", MagicMock(cuda=MagicMock(is_available=lambda: False), backends=MagicMock(mps=MagicMock(is_available=lambda: True))))
    monkeypatch.setitem(
        __import__("sys").modules,
        "diffusers",
        MagicMock(DiffusionPipeline=MagicMock(from_pretrained=lambda *args, **kwargs: _FakePipe())),
    )

    generator = FluxImageGenerator(
        ImageGenerationConfig(
            flux_model="black-forest-labs/FLUX.1-dev",
            device="mps",
            sequential_cpu_offload=True,
        )
    )
    generator.generate("a red cat")

    assert captured["cpu_offload"] == "sequential"


def test_flux_image_generator_wraps_mps_oom_with_actionable_message(monkeypatch) -> None:
    from storm_zero_llm.image_generation import FluxImageGenerator, ImageGenerationConfig

    fake_pipe = MagicMock()
    fake_pipe.side_effect = RuntimeError(
        "MPS backend out of memory (MPS allocated: 30.15 GiB, other allocations: 464.00 KiB, "
        "max allowed: 30.19 GiB). Tried to allocate 72.00 MiB on shared pool."
    )

    generator = FluxImageGenerator(ImageGenerationConfig(flux_model="test-model"))
    generator._pipe = fake_pipe

    try:
        generator.generate("a red cat")
    except RuntimeError as exc:
        assert "Image generation exceeded available Apple GPU memory" in str(exc)
        assert "LLM_N_GPU_LAYERS" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError for MPS OOM")


def test_flux_image_generator_raises_runtime_error_when_peft_missing(monkeypatch) -> None:
    from storm_zero_llm.image_generation import FluxImageGenerator, ImageGenerationConfig

    class _FakePipe:
        def to(self, device):
            return self

        def load_lora_weights(self, model_id, **kwargs):
            raise ValueError("PEFT backend is required for this method.")

    def _fake_from_pretrained(model_id, **kwargs):
        return _FakePipe()

    monkeypatch.setitem(__import__("sys").modules, "torch", MagicMock(cuda=MagicMock(is_available=lambda: False), backends=MagicMock(mps=MagicMock(is_available=lambda: False))))
    monkeypatch.setitem(
        __import__("sys").modules,
        "diffusers",
        MagicMock(DiffusionPipeline=MagicMock(from_pretrained=_fake_from_pretrained)),
    )
    monkeypatch.setitem(__import__("sys").modules, "peft", MagicMock())

    generator = FluxImageGenerator(
        ImageGenerationConfig(
            flux_model="black-forest-labs/FLUX.1-dev",
            lora_model="lustlyai/Flux_Lustly.ai_Uncensored_nsfw_v1",
        )
    )

    try:
        generator.generate("a red cat")
    except RuntimeError as exc:
        assert "PEFT is required for LoRA image generation" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError when diffusers requires PEFT backend")


def test_flux_image_generator_encodes_png_payload() -> None:
    from storm_zero_llm.image_generation import FluxImageGenerator, ImageGenerationConfig

    generator = FluxImageGenerator(ImageGenerationConfig(flux_model="test-model"))
    fake_image = MagicMock()
    fake_pipe = MagicMock()
    fake_pipe.return_value.images = [fake_image]
    generator._pipe = fake_pipe

    def _save(buffer, format="PNG") -> None:
        buffer.write(b"png-bytes")

    fake_image.save.side_effect = _save

    payload = generator.generate("a red cat")

    fake_pipe.assert_called_once_with("a red cat")
    assert payload["mime_type"] == "image/png"
    assert payload["base64"] == base64.b64encode(b"png-bytes").decode("ascii")


def test_create_image_generator_selects_coreml_when_path_set(tmp_path: Path) -> None:
    from storm_zero_llm.image_generation import CoreMLImageGenerator, ImageGenerationConfig, create_image_generator

    coreml_path = tmp_path / "Resources"
    coreml_path.mkdir()
    generator = create_image_generator(
        ImageGenerationConfig(
            flux_model="stabilityai/stable-diffusion-2-1-base",
            coreml_path=coreml_path,
        )
    )
    assert isinstance(generator, CoreMLImageGenerator)


def test_coreml_image_generator_passes_pipeline_options(monkeypatch, tmp_path: Path) -> None:
    from storm_zero_llm.image_generation import CoreMLImageGenerator, ImageGenerationConfig

    captured: dict[str, object] = {}
    coreml_path = tmp_path / "Resources"
    coreml_path.mkdir()
    (coreml_path / "TextEncoder.mlmodelc").mkdir()
    (coreml_path / "metadata.json").write_text('[{"inputSchema": []}]', encoding="utf-8")

    class _FakeOutput:
        images = [MagicMock()]

    class _FakePipe:
        def __call__(self, **kwargs):
            captured["call_kwargs"] = kwargs
            fake_image = MagicMock()
            fake_image.save.side_effect = lambda buffer, format="PNG": buffer.write(b"png-bytes")
            return MagicMock(images=[fake_image])

    def _fake_get_coreml_pipe(**kwargs):
        captured["coreml_kwargs"] = kwargs
        return _FakePipe()

    monkeypatch.setattr("sys.platform", "darwin")
    monkeypatch.setitem(
        __import__("sys").modules,
        "python_coreml_stable_diffusion",
        MagicMock(__file__="/tmp/python_coreml_stable_diffusion/__init__.py"),
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "python_coreml_stable_diffusion.coreml_model",
        MagicMock(get_resource_type=lambda _path: "compiled"),
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "python_coreml_stable_diffusion.pipeline",
        MagicMock(get_coreml_pipe=_fake_get_coreml_pipe),
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "diffusers",
        MagicMock(
            StableDiffusionPipeline=MagicMock(
                from_pretrained=lambda model_version, **kwargs: MagicMock(
                    config={"force_zeros_for_empty_prompt": True},
                    tokenizer=MagicMock(),
                    scheduler=MagicMock(),
                    feature_extractor=MagicMock(),
                )
            ),
            StableDiffusionXLPipeline=MagicMock(),
        ),
    )

    generator = CoreMLImageGenerator(
        ImageGenerationConfig(
            flux_model="stabilityai/stable-diffusion-2-1-base",
            coreml_path=coreml_path,
            compute_unit="CPU_AND_GPU",
            num_inference_steps=25,
            guidance_scale=8.5,
        )
    )
    payload = generator.generate("a red cat")

    assert captured["coreml_kwargs"]["model_version"] == "stabilityai/stable-diffusion-2-1-base"
    assert captured["coreml_kwargs"]["compute_unit"] == "CPU_AND_GPU"
    assert captured["coreml_kwargs"]["sources"] == "compiled"
    assert captured["call_kwargs"]["num_inference_steps"] == 25
    assert captured["call_kwargs"]["guidance_scale"] == 8.5
    assert payload["base64"] == base64.b64encode(b"png-bytes").decode("ascii")


def test_patch_coreml_pipeline_fixes_numpy_equal_truth_value(tmp_path: Path, monkeypatch) -> None:
    from storm_zero_llm.image_generation import _patch_coreml_pipeline_imports

    pkg = tmp_path / "python_coreml_stable_diffusion"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    pipeline = pkg / "pipeline.py"
    pipeline.write_text(
        "from transformers import CLIPFeatureExtractor, CLIPTokenizer\n"
        "            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not np.equal(\n"
        "                    text_input_ids, untruncated_ids\n"
        "            ):\n"
        "                removed_text = None\n",
        encoding="utf-8",
    )

    import sys
    import types

    fake_mod = types.ModuleType("python_coreml_stable_diffusion")
    fake_mod.__file__ = str(pkg / "__init__.py")
    monkeypatch.setitem(sys.modules, "python_coreml_stable_diffusion", fake_mod)

    _patch_coreml_pipeline_imports()

    text = pipeline.read_text(encoding="utf-8")
    assert "np.array_equal" in text
    assert "not np.equal(" not in text
    assert "CLIPImageProcessor as CLIPFeatureExtractor" in text


def test_coreml_image_generator_requires_macos(monkeypatch, tmp_path: Path) -> None:
    from storm_zero_llm.image_generation import CoreMLImageGenerator, ImageGenerationConfig

    monkeypatch.setattr("sys.platform", "linux")
    generator = CoreMLImageGenerator(
        ImageGenerationConfig(
            flux_model="stabilityai/stable-diffusion-2-1-base",
            coreml_path=tmp_path,
        )
    )

    try:
        generator.generate("a red cat")
    except RuntimeError as exc:
        assert "requires macOS" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError on non-macOS platforms")
