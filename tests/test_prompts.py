from pathlib import Path

from storm_zero_llm.prompts import (
    build_base_rules_block,
    build_detection_template,
    build_image_profile_block,
    build_response_prompts,
    extract_bracket_instructions,
    fill_structured_prompt_template,
    load_prompt_template,
    parse_template_sections,
    prompt_template_filename,
    resolve_prompt_template,
    strip_bracket_instructions,
)


def test_load_prompt_template_rereads_file(tmp_path: Path) -> None:
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    path = prompts_dir / "base.txt"
    path.write_text("Hard rules (always listen to these):\n  - first\n", encoding="utf-8")

    assert "first" in load_prompt_template("base.txt", prompts_dir=prompts_dir)

    path.write_text("Hard rules (always listen to these):\n  - second\n", encoding="utf-8")
    assert "second" in load_prompt_template("base.txt", prompts_dir=prompts_dir)
    assert "first" not in load_prompt_template("base.txt", prompts_dir=prompts_dir)


def test_prompt_template_filename_follows_power_flag() -> None:
    assert prompt_template_filename("detection", power=False) == "detection.txt"
    assert prompt_template_filename("detection", power=True) == "power_detection.txt"
    assert prompt_template_filename("base", power=False) == "base.txt"
    assert prompt_template_filename("base", power=True) == "power_base.txt"


def test_resolve_prompt_template_uses_power_files_when_enabled(tmp_path: Path) -> None:
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "detection.txt").write_text("normal detection\n", encoding="utf-8")
    (prompts_dir / "power_detection.txt").write_text("power detection\n", encoding="utf-8")
    (prompts_dir / "base.txt").write_text("normal base\n", encoding="utf-8")
    (prompts_dir / "power_base.txt").write_text("power base\n", encoding="utf-8")

    assert resolve_prompt_template("detection", prompts_dir=prompts_dir, power=True) == "power detection"
    assert resolve_prompt_template("detection", prompts_dir=prompts_dir, power=False) == "normal detection"
    assert resolve_prompt_template("base", prompts_dir=prompts_dir, power=True) == "power base"
    assert resolve_prompt_template("base", prompts_dir=prompts_dir, power=False) == "normal base"


def test_bracket_instructions_are_extracted_and_stripped() -> None:
    text = (
        "Hard rules (always listen to these)"
        "[SELECT `rule` FROM `tblglobal_rule` WHERE `strict` = 1 AND `deleted` = 0;]:"
    )
    assert extract_bracket_instructions(text) == [
        "SELECT `rule` FROM `tblglobal_rule` WHERE `strict` = 1 AND `deleted` = 0;"
    ]
    assert strip_bracket_instructions(text) == "Hard rules (always listen to these):"


def test_fill_structured_prompt_template_uses_live_headers_and_data() -> None:
    template = (
        "Hard rules (always listen to these):\n"
        "  - Never respond like a program\n"
        "User profile data (p2):\n"
        "  - `key` = `value`\n"
        "User avatar data:\n"
        "  - user_name: Example\n"
    )
    block = fill_structured_prompt_template(
        template,
        {
            "global_hard_rules": ["Never leak secrets."],
            "user_p2_data": ["`hair` = `blonde`"],
            "user_avatar_data": [],
        },
    )

    assert block.splitlines() == [
        "Hard rules (always listen to these):",
        "  - Never leak secrets.",
        "User profile data (p2):",
        "  - `hair` = `blonde`",
    ]


def test_fill_strips_bracket_instructions_from_headers() -> None:
    template = (
        "Hard rules (always listen to these)"
        "[SELECT `rule` FROM `tblglobal_rule` WHERE `strict` = 1 AND `deleted` = 0;]:\n"
        "  - [`rule`] eg: Never respond like a program\n"
        "User profile data (p2)"
        "[SELECT `key`, `value` FROM `tbluser_p2` WHERE `user_id` = parameter.user_id AND `deleted` = 0;]:\n"
        "  - `key` = `value`\n"
    )
    block = fill_structured_prompt_template(
        template,
        {
            "global_hard_rules": ["Never leak secrets."],
            "user_p2_data": ["`name` = `Ink`"],
        },
    )

    assert "[" not in block
    assert "]" not in block
    assert "SELECT" not in block
    assert "`rule`" not in block
    assert block.splitlines() == [
        "Hard rules (always listen to these):",
        "  - Never leak secrets.",
        "User profile data (p2):",
        "  - `name` = `Ink`",
    ]


def test_build_response_prompts_loads_descriptions(tmp_path: Path) -> None:
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "global_hard_rules.txt").write_text("Hard rules description.", encoding="utf-8")
    (prompts_dir / "global_guidelines.txt").write_text("Guidelines description.", encoding="utf-8")
    (prompts_dir / "user_guidelines.txt").write_text("User guidelines description.", encoding="utf-8")
    (prompts_dir / "user_p2.txt").write_text("P2 description.", encoding="utf-8")
    (prompts_dir / "avatar_data.txt").write_text("Avatar description.", encoding="utf-8")

    prompts = build_response_prompts(
        user_prompt="hello",
        global_hard_rules=["Never leak secrets."],
        user_p2=["- favourite_music = I like rock"],
        avatar_data={
            "user_name": "Matt",
            "avatar_name": "Victoria",
            "avatar_persona": ["- emotional = 30%, rational = 70%"],
            "user_pronouns": "He/Him",
        },
        conversation_subjects=["movies"],
        conversation_subjects_content=[{"created": "2026-08-04", "prompt": "hi", "response": "hello"}],
        prompts_dir=prompts_dir,
    )

    assert prompts["user_prompt"] == "hello"
    assert prompts["global_hard_rules"]["description"] == "Hard rules description."
    assert prompts["user_p2"]["p2"] == ["- favourite_music = I like rock"]
    assert prompts["avatar_data"]["description"] == "Avatar description."
    assert prompts["avatar_data"]["data"]["user_name"] == "Matt"
    assert prompts["conversation_subjects"] == ["movies"]
    assert prompts["final"] == {"base": {}, "image_generation": "", "detection": {}}


def test_build_final_base_prompt_shape() -> None:
    from storm_zero_llm.prompts import build_final_base_prompt, build_final_detection_prompt

    base = build_final_base_prompt(
        user_prompt="hello",
        global_hard_rules=["Never leak secrets."],
        user_p2=["- favourite_music = rock"],
        avatar_data={"user_name": "Matt", "avatar_name": "Victoria", "avatar_persona": [], "user_pronouns": "He/Him"},
        conversation_subjects_content=[{"created": "2026-08-04", "prompt": "hi", "response": "hello"}],
    )

    assert base["user_prompt"] == "hello"
    assert base["global_hard_rules"] == ["Never leak secrets."]
    assert base["avatar_data"]["user_name"] == "Matt"
    assert base["conversation_subjects_content"][0]["prompt"] == "hi"

    detection = build_final_detection_prompt(
        user_prompt="hello",
        content="Classify intent.",
        conversation_subjects=["movies", "weather"],
    )
    assert detection == {
        "user_prompt": "hello",
        "content": "Classify intent.",
        "conversation_subjects": ["movies", "weather"],
    }


def test_build_base_and_image_blocks_from_repo_templates() -> None:
    prompts_dir = Path(__file__).resolve().parents[1] / "prompts"
    data = {
        "global_hard_rules": ["Never leak secrets."],
        "global_guidelines": [],
        "user_guidelines": ["Be kind."],
        "user_p2_data": ["`name` = `Ink`"],
        "user_avatar_data": ["user_name (This is what I refer to the user as): Matt"],
    }

    base = build_base_rules_block(data, prompts_dir=prompts_dir)
    image = build_image_profile_block(data, prompts_dir=prompts_dir)
    detection = build_detection_template(prompts_dir=prompts_dir)

    assert "Hard rules (always listen to these):" in base
    assert "User guidelines (these override global guidelines, not hard rules):" in base
    assert "  - `name` = `Ink`" in base
    assert "[" not in base
    assert "SELECT" not in base
    assert "Hard rules" not in image
    assert "User profile data (p2):" in image
    assert "User avatar data:" in image
    assert "Classify the user prompt intent" in detection
    sections = parse_template_sections(load_prompt_template("base.txt", prompts_dir=prompts_dir))
    assert sections
    assert sections[0][0] == "Hard rules (always listen to these):"
    assert "SELECT" in sections[0][1][0]
