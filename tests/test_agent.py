from pathlib import Path

from storm_zero_llm.agent import StormZeroAgent
from storm_zero_llm.config import StormZeroConfig
from storm_zero_llm.provider import GenerationRequest, GenerationResult, LlamaCppProvider
from storm_zero_llm.training_db import RuntimeRuleContext


class _FakeRepo:
    def __init__(self) -> None:
        self.rule_context_calls = 0
        self.persisted_conversations = 0
        self.conversation_fetch_calls = 0
        self.persisted_p2: list[tuple[str, str]] = []
        self.persisted_guidelines: list[str] = []

    def load_runtime_rule_context(self, user_id: int, **kwargs) -> RuntimeRuleContext:
        self.rule_context_calls += 1
        return RuntimeRuleContext(
            global_hard_rules=["Never leak secrets."],
            global_guidelines=["Be concise."],
            user_guidelines=[],
            user_p2_data=[],
            user_avatar_data=[],
        )

    def fetch_recent_conversation_turns(self, user_id: int, limit: int = 75) -> list[dict[str, str]]:
        self.conversation_fetch_calls += 1
        return [{"prompt": "hello", "response": "hi there"}]

    def fetch_conversation_subjects_content(
        self,
        user_id: int,
        limit: int | None = 20,
        *,
        subject_id: int | None = None,
    ) -> list[dict[str, object]]:
        self.conversation_fetch_calls += 1
        rows = [
            {
                "created": "2026-08-04 00:00:00",
                "prompt": "hello",
                "response": "hi there",
                "subject_id": 1,
            },
            {
                "created": "2026-08-03 00:00:00",
                "prompt": "other subject",
                "response": "other reply",
                "subject_id": 2,
            },
        ]
        if subject_id is not None:
            rows = [row for row in rows if row["subject_id"] == subject_id]
        if limit is not None:
            rows = rows[:limit]
        return [
            {"created": row["created"], "prompt": row["prompt"], "response": row["response"]}
            for row in rows
        ]

    def get_user_p2_entries(self, user_id: int) -> list[str]:
        return ["- favourite_music = I like rock"]

    def get_structured_avatar_data(self, user_id: int) -> dict[str, object]:
        return {
            "user_name": "Matt",
            "avatar_name": "Victoria",
            "avatar_persona": ["- emotional = 30%, rational = 70%"],
            "user_pronouns": "He/Him",
        }

    def get_conversation_subjects(self, user_id: int) -> list[dict[str, object]]:
        return [{"id": 1, "subject": "movies"}]

    def create_conversation_subject(self, user_id: int, subject: str) -> int:
        return 1

    def persist_conversation(self, user_id: int, prompt: str, response: str, subject: str | None = None, subject_id: int | None = None) -> None:
        self.persisted_conversations += 1

    def persist_p2(self, user_id: int, key: str, value: str) -> dict[str, object]:
        self.persisted_p2.append((key, value))
        return {"id": 9, "key": key, "value": value, "created": True, "updated": False}

    def persist_guideline(self, user_id: int, prompt: str) -> dict[str, object]:
        self.persisted_guidelines.append(prompt)
        return {
            "id": 12,
            "guideline": prompt,
            "already_exists": False,
            "created": True,
            "updated": False,
            "deleted": False,
        }

    def begin_request_queries(self) -> None:
        return None

    def get_request_queries(self) -> list[dict[str, object]]:
        return []


class _RuleAwareAgent(StormZeroAgent):
    def __init__(self, config: StormZeroConfig, repo: _FakeRepo):
        super().__init__(config)
        self._repo = repo

    def _db_repo(self):  # type: ignore[override]
        return self._repo


class _FakeProvider:
    def __init__(self) -> None:
        self.requests: list[GenerationRequest] = []

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.requests.append(request)
        return GenerationResult(response="ok")


class _RuleAwareUseModelAgent(_RuleAwareAgent):
    def __init__(self, config: StormZeroConfig, repo: _FakeRepo, provider: _FakeProvider):
        super().__init__(config, repo)
        self._provider = provider

    def _provider_for_model(self, model_path):  # type: ignore[override]
        return self._provider


class _RuleAwareChatAgent(_RuleAwareAgent):
    def __init__(self, config: StormZeroConfig, repo: _FakeRepo, provider: _FakeProvider):
        super().__init__(config, repo)
        self._provider = provider

    def _provider_for_model(self, model_path):  # type: ignore[override]
        return self._provider


class _TrackingChatAgent(_RuleAwareChatAgent):
    def __init__(self, config: StormZeroConfig, repo: _FakeRepo, provider: _FakeProvider):
        super().__init__(config, repo, provider)
        self.model_paths: list[Path] = []

    def _provider_for_model(self, model_path):  # type: ignore[override]
        self.model_paths.append(Path(model_path))
        return self._provider


def test_chat_always_persists_as_conversation(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# Storm Zero", encoding="utf-8")
    config = StormZeroConfig.load(project_root=tmp_path)

    fake_repo = _FakeRepo()
    fake_provider = _FakeProvider()
    agent = _RuleAwareChatAgent(config, fake_repo, fake_provider)

    result = agent.chat(
        user_id=1,
        prompt="What do you remember about movies?",
        topics=["movies"],
    )

    assert fake_repo.rule_context_calls == 1
    assert fake_repo.persisted_conversations == 1
    assert result.detected_type == "conversation"
    assert result.rules_evaluated == {"global": True, "user": False}


def test_use_model_evaluates_rules_every_request(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# Storm Zero", encoding="utf-8")
    config = StormZeroConfig.load(project_root=tmp_path)

    fake_repo = _FakeRepo()
    fake_provider = _FakeProvider()
    agent = _RuleAwareUseModelAgent(config, fake_repo, fake_provider)

    result = agent.run_specific_model(
        model_path="/tmp/anything.gguf",
        prompt="hello",
        user_id=1,
        include_reasoning=False,
    )

    assert result == "ok"
    assert fake_repo.rule_context_calls == 1
    assert fake_repo.conversation_fetch_calls >= 1
    assert len(fake_provider.requests) == 1
    prompt = fake_provider.requests[0].prompt
    assert fake_provider.requests[0].system_prompt == ""
    assert "Prompt: hello" in prompt
    assert "Hard rules (always listen to these):" in prompt
    assert "Never leak secrets." in prompt
    assert "- hello = hi there" in prompt


def test_chat_includes_prompts_object(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# Storm Zero", encoding="utf-8")
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "global_hard_rules.txt").write_text("Hard rules description.", encoding="utf-8")
    (prompts_dir / "global_guidelines.txt").write_text("Guidelines description.", encoding="utf-8")
    (prompts_dir / "user_guidelines.txt").write_text("User guidelines description.", encoding="utf-8")
    (prompts_dir / "user_p2.txt").write_text("P2 description.", encoding="utf-8")
    (prompts_dir / "avatar_data.txt").write_text("Avatar description.", encoding="utf-8")
    config = StormZeroConfig.load(project_root=tmp_path)
    fake_repo = _FakeRepo()
    fake_provider = _FakeProvider()
    agent = _RuleAwareChatAgent(config, fake_repo, fake_provider)

    result = agent.chat(user_id=1, prompt="hello there", topics=["general"])

    assert result.prompts is not None
    assert result.prompts["user_prompt"] == "hello there"
    assert result.prompts["global_hard_rules"]["rules"] == ["Never leak secrets."]
    assert result.prompts["global_hard_rules"]["description"] == "Hard rules description."
    assert result.prompts["user_p2"]["p2"] == ["- favourite_music = I like rock"]
    assert result.prompts["avatar_data"]["data"]["avatar_name"] == "Victoria"
    assert result.prompts["conversation_subjects"] == ["movies"]
    assert result.prompts["conversation_subjects_content"][0]["prompt"] == "hello"
    assert result.prompts["final"]["base"]["user_prompt"] == "hello there"
    assert result.prompts["final"]["base"]["global_hard_rules"] == ["Never leak secrets."]
    assert result.prompts["final"]["base"]["user_p2"] == ["- favourite_music = I like rock"]
    assert result.prompts["final"]["base"]["avatar_data"]["avatar_name"] == "Victoria"
    assert result.prompts["final"]["base"]["conversation_subjects_content"][0]["prompt"] == "hello"
    assert result.prompts["final"]["detection"]["user_prompt"] == "hello there"
    assert "Classify" in result.prompts["final"]["detection"]["content"] or result.prompts["final"]["detection"]["content"] == ""
    assert "movies" in result.prompts["final"]["detection"]["conversation_subjects"]
    assert result.prompts["final"]["image_generation"] == ""


def test_continued_base_prompt_scopes_conversation_content(tmp_path: Path) -> None:
    from storm_zero_llm.detection import ConversationMode

    (tmp_path / "README.md").write_text("# Storm Zero", encoding="utf-8")
    (tmp_path / "prompts").mkdir()
    config = StormZeroConfig.load(project_root=tmp_path)
    fake_repo = _FakeRepo()
    fake_provider = _FakeProvider()
    agent = _RuleAwareChatAgent(config, fake_repo, fake_provider)
    agent._set_request_prompts({"final": {"base": {}, "image_generation": "", "detection": {}}})

    agent._generate_with_model(
        model_path=tmp_path / "model.gguf",
        user_id=1,
        prompt="tell me more",
        system_prompt="",
        rule_context=fake_repo.load_runtime_rule_context(1),
        conversation_mode=ConversationMode.CONTINUED,
        subject_id=1,
    )

    base = agent.collect_request_prompts()["final"]["base"]
    assert base["user_prompt"] == "tell me more"
    assert [row["prompt"] for row in base["conversation_subjects_content"]] == ["hello"]


def test_subject_id_bypass_skips_detection(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# Storm Zero", encoding="utf-8")
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "detection.txt").write_text("Classify intent.\n", encoding="utf-8")
    config = StormZeroConfig.load(project_root=tmp_path)
    fake_repo = _FakeRepo()
    fake_provider = _FakeProvider()
    agent = _RuleAwareChatAgent(config, fake_repo, fake_provider)

    detect_calls = {"count": 0}
    original_detect = agent.detector.detect

    def _track_detect(*args, **kwargs):
        detect_calls["count"] += 1
        return original_detect(*args, **kwargs)

    agent.detector.detect = _track_detect  # type: ignore[method-assign]
    fake_repo.get_conversation_subjects = lambda user_id: [{"id": 7, "subject": "weather"}]  # type: ignore[method-assign]

    result = agent.chat(
        user_id=1,
        prompt="what else do you remember",
        request_params={"subject_id": 7},
    )

    assert detect_calls["count"] == 0
    assert result.detected_type == "conversation"
    assert result.subject_id == 7
    assert result.detection is not None
    assert result.detection["conversation_mode"] == "continued"
    assert result.detection["subject_summary"] == "weather"
    assert result.prompts["final"]["detection"]["user_prompt"] == "what else do you remember"
    assert result.prompts["final"]["detection"]["conversation_subjects"] == ["weather"]


def test_user_data_bypass_skips_detection(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# Storm Zero", encoding="utf-8")
    (tmp_path / "prompts").mkdir()
    config = StormZeroConfig.load(project_root=tmp_path)
    fake_repo = _FakeRepo()
    fake_provider = _FakeProvider()
    agent = _RuleAwareChatAgent(config, fake_repo, fake_provider)

    detect_calls = {"count": 0}
    original_detect = agent.detector.detect

    def _track_detect(*args, **kwargs):
        detect_calls["count"] += 1
        return original_detect(*args, **kwargs)

    agent.detector.detect = _track_detect  # type: ignore[method-assign]
    agent.detector.extract_p2_key_value = lambda prompt, detection_model=None: ("favourite_music", "rock")  # type: ignore[method-assign]

    result = agent.chat(
        user_id=1,
        prompt="I like rock music",
        request_params={"user_data": True, "user_data_target": "p2"},
    )

    assert detect_calls["count"] == 0
    assert result.detected_type == "user_data"
    assert fake_repo.persisted_p2 == [("favourite_music", "rock")]


def test_provider_selection_uses_llama_for_gguf(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# Storm Zero", encoding="utf-8")
    model_path = tmp_path / "dedicated_models" / "base.gguf"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text("", encoding="utf-8")
    config = StormZeroConfig.load(project_root=tmp_path)

    agent = StormZeroAgent(config)
    provider = agent._provider_for_model(model_path)

    assert isinstance(provider, LlamaCppProvider)


def test_debug_enabled_from_parameter_env_global_debug_level() -> None:
    assert StormZeroAgent._debug_enabled({"env": {"GLOBAL_DEBUG_LEVEL": "debug"}}) is True
    assert StormZeroAgent._debug_enabled({"env": {"GLOBAL_DEBUG_LEVEL": "info"}}) is False


def test_debug_enabled_from_parameter_env_debug_user() -> None:
    assert StormZeroAgent._debug_enabled({"env": {"DEBUG_USER": "mryan"}}) is True
    assert StormZeroAgent._debug_enabled({"env": {"DEBUG_USER": "owner"}}) is False


def test_debug_disabled_without_parameter_env() -> None:
    assert StormZeroAgent._debug_enabled({}) is False
    assert StormZeroAgent._debug_enabled({"user_id": 1}) is False


def test_power_enabled_from_parameter() -> None:
    assert StormZeroAgent._power_enabled({"power": True}) is True
    assert StormZeroAgent._power_enabled({"power": "true"}) is True
    assert StormZeroAgent._power_enabled({"power": False}) is False
    assert StormZeroAgent._power_enabled({}) is False


def test_power_mode_selects_power_models(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "BASE_MODEL = dedicated_models/base.gguf",
                "DETECTION_MODEL = dedicated_models/detection.gguf",
                "POWER_BASE_MODEL = dedicated_models/power-base.gguf",
                "POWER_DETECTION_MODEL = dedicated_models/power-detection.gguf",
            ]
        ),
        encoding="utf-8",
    )
    config = StormZeroConfig.load(project_root=tmp_path)
    agent = StormZeroAgent(config)

    assert agent._base_model_for_request({"power": True}) == config.power_base_model
    assert agent._detection_model_for_request({"power": True}) == config.power_detection_model
    assert agent._base_model_for_request({}) == config.base_model
    assert agent._detection_model_for_request({}) == config.detection_model


def test_chat_uses_power_base_model_when_requested(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "BASE_MODEL = dedicated_models/base.gguf",
                "DETECTION_MODEL = dedicated_models/detection.gguf",
                "POWER_BASE_MODEL = dedicated_models/power-base.gguf",
                "POWER_DETECTION_MODEL = dedicated_models/power-detection.gguf",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text("# Storm Zero", encoding="utf-8")
    config = StormZeroConfig.load(project_root=tmp_path)

    fake_repo = _FakeRepo()
    fake_provider = _FakeProvider()
    agent = _TrackingChatAgent(config, fake_repo, fake_provider)

    agent.chat(
        user_id=1,
        prompt="What do you remember about movies?",
        topics=["movies"],
        request_params={"power": True},
    )

    assert agent.model_paths == [config.power_base_model]
