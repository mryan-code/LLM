from pathlib import Path

from storm_zero_llm.detection import (
    ConversationMode,
    DetectionAction,
    DetectionResult,
    IntentDetector,
    matches_image_generation_request,
)


def test_detection_result_to_dict() -> None:
    result = DetectionResult(
        action=DetectionAction.CREATE_MEDIA,
        media_type="image",
        conversation_mode=ConversationMode.NEW,
        subject_summary="sunset photo",
    )

    assert result.to_dict() == {
        "action": "create_media",
        "media_type": "image",
        "user_data_target": None,
        "conversation_mode": "new",
        "subject_id": None,
        "subject_summary": "sunset photo",
    }


def test_matches_image_generation_request_for_create_send_generate() -> None:
    assert matches_image_generation_request("send me a nude image")
    assert matches_image_generation_request("send me an image")
    assert matches_image_generation_request("generate an image of a cat")
    assert matches_image_generation_request("please create an image")
    assert matches_image_generation_request("show me a picture of a cat")
    assert not matches_image_generation_request("tell me about image formats")


def test_detect_routes_send_image_prompt_to_create_media(tmp_path: Path) -> None:
    detector = IntentDetector(tmp_path / "missing.gguf", llm_runtime=object())
    result = detector.detect("send me an image of a sunset", [])

    assert result.action == DetectionAction.CREATE_MEDIA
    assert result.media_type == "image"


def test_heuristic_detects_image_generation(tmp_path: Path) -> None:
    detector = IntentDetector(tmp_path / "missing.gguf", llm_runtime=object())
    result = detector.detect("please generate an image of a mountain", [])

    assert result.action == DetectionAction.CREATE_MEDIA
    assert result.media_type == "image"


def test_heuristic_detects_scheduled_task(tmp_path: Path) -> None:
    detector = IntentDetector(tmp_path / "missing.gguf", llm_runtime=object())
    result = detector.detect("remind me to take my medication at 5:00pm", [])

    assert result.action == DetectionAction.SCHEDULED_TASK


def test_heuristic_detects_user_profile_update(tmp_path: Path) -> None:
    detector = IntentDetector(tmp_path / "missing.gguf", llm_runtime=object())
    result = detector.detect("My name is Ink and my favorite color is blue", [])

    assert result.action == DetectionAction.ADD_USER_DATA
    assert result.user_data_target == "p2"


def test_heuristic_extracts_p2_key_value(tmp_path: Path) -> None:
    detector = IntentDetector(tmp_path / "missing.gguf", llm_runtime=object())

    assert detector.extract_p2_key_value("My name is Ink") == ("name", "Ink")
    assert detector.extract_p2_key_value("My favorite color is blue") == ("favorite_color", "blue")


def test_heuristic_continues_existing_conversation(tmp_path: Path) -> None:
    detector = IntentDetector(tmp_path / "missing.gguf", llm_runtime=object())
    result = detector.detect(
        "tell me more about hiking trails",
        [{"id": 42, "subject": "hiking trails"}],
    )

    assert result.action == DetectionAction.GENERAL_QUERY
    assert result.conversation_mode == ConversationMode.CONTINUED
    assert result.subject_id == 42


def test_detect_matches_referenced_subject(tmp_path: Path) -> None:
    from storm_zero_llm.detection import match_referenced_subject

    subjects = [{"id": 9, "subject": "weather"}, {"id": 3, "subject": "movies"}]
    matched = match_referenced_subject(
        "remember that time we talked about the weather",
        subjects,
    )
    assert matched == (9, "weather")

    detector = IntentDetector(tmp_path / "missing.gguf", llm_runtime=object())
    result = detector.detect(
        "remember that time we talked about the weather",
        subjects,
    )
    assert result.action == DetectionAction.GENERAL_QUERY
    assert result.conversation_mode == ConversationMode.CONTINUED
    assert result.subject_id == 9
    assert result.subject_summary == "weather"
    assert detector.last_detection is not None
    assert detector.last_detection["user_prompt"] == "remember that time we talked about the weather"
    assert detector.last_detection["conversation_subjects"] == ["weather", "movies"]


def test_build_detection_prompt_prepends_user_prompt_and_appends_subjects(tmp_path: Path) -> None:
    from storm_zero_llm.detection import build_detection_prompt

    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "detection.txt").write_text(
        "Classify intent.\nThe user has conversed with you about:\n",
        encoding="utf-8",
    )

    built = build_detection_prompt(
        "hello there",
        "- id=1 subject=movies",
        prompts_dir=prompts_dir,
    )

    assert built.startswith("hello there - Classify intent.")
    assert "Classify intent." in built
    assert built.endswith("\n- id=1 subject=movies")


def test_pack_subjects_block_fits_model_context() -> None:
    from storm_zero_llm.detection import pack_subjects_block

    subjects = [{"id": i, "subject": f"topic {i} " + ("x" * 40)} for i in range(1, 101)]
    block = pack_subjects_block(
        subjects,
        prompt="user prompt " + ("y" * 200),
        system_prompt="",
        n_ctx=200,
        output_reserve=50,
    )

    assert "No existing subjects." not in block
    lines = block.splitlines()
    assert 1 <= len(lines) < 100
    assert lines[0].startswith("- id=1 subject=")


def test_pack_subjects_block_empty_when_no_budget() -> None:
    from storm_zero_llm.detection import pack_subjects_block

    block = pack_subjects_block(
        [{"id": 1, "subject": "movies"}],
        prompt="x" * 1000,
        system_prompt="y" * 1000,
        n_ctx=100,
        output_reserve=50,
    )
    assert block == "No existing subjects."


def test_detect_with_model_uses_detection_txt_and_packed_subjects(tmp_path: Path) -> None:
    from storm_zero_llm.config import LLMRuntimeConfig
    from storm_zero_llm.provider import GenerationRequest, GenerationResult

    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "detection.txt").write_text(
        "Classify intent.\nThe user has conversed with you about:\n",
        encoding="utf-8",
    )
    model_path = tmp_path / "detection.gguf"
    model_path.write_text("", encoding="utf-8")

    captured: list[GenerationRequest] = []

    class _FakeProvider:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def generate(self, request: GenerationRequest) -> GenerationResult:
            captured.append(request)
            return GenerationResult(
                response='{"action":"general_query","conversation_mode":"new","subject_summary":"hello"}'
            )

    import storm_zero_llm.detection as detection_mod

    original = detection_mod.LlamaCppProvider
    detection_mod.LlamaCppProvider = _FakeProvider  # type: ignore[misc,assignment]
    try:
        detector = IntentDetector(
            model_path,
            LLMRuntimeConfig(n_gpu_layers=0, n_threads=1, n_ctx=512),
            prompts_dir=prompts_dir,
        )
        subjects = [{"id": i, "subject": f"subject {i}"} for i in range(1, 50)]
        result = detector.detect("hello", subjects)
    finally:
        detection_mod.LlamaCppProvider = original  # type: ignore[misc]

    assert result.action == DetectionAction.GENERAL_QUERY
    assert len(captured) == 1
    assert captured[0].system_prompt == ""
    assert captured[0].prompt.startswith("hello - Classify intent.")
    assert "The user has conversed with you about:" in captured[0].prompt
    assert "- id=1 subject=subject 1" in captured[0].prompt
