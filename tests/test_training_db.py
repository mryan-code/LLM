import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

from storm_zero_llm.config import DatabaseConfig
from storm_zero_llm.training_db import (
    MySQLTrainingRepository,
    RuntimeRuleContext,
    _build_avatar_data_from_rows,
    _expand_multiline_rules,
    _format_persona_entry,
)


class _FakeCursor:
    def __init__(self, responses: dict[str, list[dict[str, object]]]):
        self._responses = responses
        self._last_query = ""

    def __enter__(self) -> "_FakeCursor":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None

    def execute(self, query: str, params: tuple[object, ...]) -> None:
        strict = int(params[0])
        self._last_query = query % strict if "%s" in query else query

    def fetchall(self) -> list[dict[str, object]]:
        return self._responses[self._last_query]


class _FakeConnection:
    def __init__(self, responses: dict[str, list[dict[str, object]]]):
        self._responses = responses

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self._responses)


class _PersistCursor:
    def __init__(self, connection: "_PersistConnection"):
        self.connection = connection
        self._subject_inserted = False

    def __enter__(self) -> "_PersistCursor":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None

    def execute(self, query: str, params: tuple[object, ...] | None = None) -> None:
        if params is None:
            params = ()
        self.connection.executed.append((query, params))

        if query.startswith("SELECT id FROM tbluser_conversation_subject"):
            self.connection._fetchone_result = None
            return
        if query.startswith("SELECT id FROM tbluser_p2"):
            self.connection._fetchone_result = None
            return
        if query.startswith("SELECT id, guideline FROM tbluser_guideline"):
            self.connection._fetchone_result = self.connection.guideline_lookup
            return
        if query.startswith("INSERT INTO tbluser_conversation_subject"):
            self._subject_inserted = True
            self.connection._fetchone_result = None
            return
        if query.startswith("SELECT LAST_INSERT_ID() AS id"):
            self.connection._fetchone_result = {"id": 77}
            return
        if query.startswith("INSERT INTO tbluser_conversation_content "):
            raise RuntimeError("conversation insert failed")

    def fetchone(self) -> dict[str, object] | None:
        return self.connection._fetchone_result


class _PersistConnection:
    def __init__(self):
        self.executed: list[tuple[str, tuple[object, ...]]] = []
        self.began = False
        self.committed = False
        self.rolled_back = False
        self._fetchone_result: dict[str, object] | None = None
        self.guideline_lookup: dict[str, object] | None = None

    def begin(self) -> None:
        self.began = True

    def commit(self) -> None:
        self.committed = True

    def rollback(self) -> None:
        self.rolled_back = True

    def cursor(self) -> _PersistCursor:
        return _PersistCursor(self)


class _RuleContextCursor:
    def __init__(self, responses: dict[str, list[dict[str, object]]]):
        self._responses = responses
        self._last_query = ""

    def __enter__(self) -> "_RuleContextCursor":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None

    def execute(self, query: str, params: tuple[object, ...]) -> None:
        self._last_query = query

    def fetchall(self) -> list[dict[str, object]]:
        return self._responses.get(self._last_query, [])

    def fetchone(self) -> dict[str, object] | None:
        rows = self.fetchall()
        return rows[0] if rows else None


class _RuleContextConnection:
    def __init__(self, responses: dict[str, list[dict[str, object]]]):
        self._responses = responses

    def cursor(self) -> _RuleContextCursor:
        return _RuleContextCursor(self._responses)


def test_load_global_memories_without_title_column() -> None:
    responses = {
        "SELECT * FROM tblglobal_rule WHERE strict = 1 AND deleted = 0": [
            {
                "id": 11,
                "rule": "Always protect user privacy.",
                "topics": "privacy,security",
            }
        ],
        "SELECT * FROM tblglobal_rule WHERE strict = 0 AND deleted = 0": [
            {
                "id": 22,
                "description": "Use a supportive tone.",
                "topics": "tone,style",
            }
        ],
    }

    repository = MySQLTrainingRepository(_FakeConnection(responses))
    memories = repository.load_global_memories()

    assert len(memories) == 2
    assert memories[0].title == "hard_rule_11"
    assert memories[0].content == "Always protect user privacy."
    assert memories[0].topics == ["privacy", "security"]
    assert memories[0].source == "tblglobal_rule"
    assert memories[1].title == "guide_22"
    assert memories[1].content == "Use a supportive tone."


def test_persist_conversation_rolls_back_when_insert_fails() -> None:
    connection = _PersistConnection()
    repository = MySQLTrainingRepository(connection)

    try:
        repository.persist_conversation(
            user_id=1,
            prompt="Tell me about movies",
            response="Sure",
            subject="movies",
        )
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected conversation persistence to raise RuntimeError")

    assert "Failed to insert conversation row" in message
    assert "conversation insert failed" in message
    assert connection.began is True
    assert connection.committed is False
    assert connection.rolled_back is True
    content_inserts = [
        query
        for query, _params in connection.executed
        if query.startswith("INSERT INTO tbluser_conversation_content ")
    ]
    assert len(content_inserts) == 1
    assert "prompt, response" in content_inserts[0]
    assert "conversation)" not in content_inserts[0]


def test_load_runtime_rule_context_includes_user_p2_data() -> None:
    responses = {
        "SELECT * FROM tblglobal_rule WHERE strict = %s AND deleted = 0": [
            {"rule": "Never leak secrets."},
            {"rule": "Be concise."},
        ],
        "SELECT * FROM tbluser_guideline WHERE user_id = %s AND deleted = 0": [
            {"guideline": "Use nickname Ink."},
        ],
        "SELECT `key`, `value` FROM tbluser_p2 WHERE user_id = %s AND deleted = 0": [
            {"key": "name", "value": "Ink"},
        ],
        "SELECT * FROM tbluser_avatar WHERE user_id = %s AND deleted = 0 ORDER BY id DESC": [],
    }
    repository = MySQLTrainingRepository(_RuleContextConnection(responses))

    context = repository.load_runtime_rule_context(user_id=1)

    assert context.global_hard_rules == ["Never leak secrets.", "Be concise."]
    assert context.user_guidelines == ["Use nickname Ink."]
    assert context.user_p2_data == ["`name` = `Ink`"]
    assert context.user_avatar_data == []
    assert "Hard rules (always listen to these):" in context.to_prompt_block()
    assert "User profile data (p2):" in context.to_prompt_block()
    assert "  - `name` = `Ink`" in context.to_prompt_block()


def test_expand_multiline_rules_splits_rows_on_newlines() -> None:
    assert _expand_multiline_rules(
        [
            "Never leak secrets.\nBe concise.",
            "Stay calm\\nStay curious",
            "  ",
        ]
    ) == [
        "Never leak secrets.",
        "Be concise.",
        "Stay calm",
        "Stay curious",
    ]


def test_load_runtime_rule_context_splits_multiline_global_rules() -> None:
    responses = {
        "SELECT * FROM tblglobal_rule WHERE strict = %s AND deleted = 0": [
            {"rule": "Never leak secrets.\nNever pretend to be human."},
            {"rule": "Be warm.\nBe curious."},
        ],
        "SELECT * FROM tbluser_guideline WHERE user_id = %s AND deleted = 0": [],
        "SELECT `key`, `value` FROM tbluser_p2 WHERE user_id = %s AND deleted = 0": [],
        "SELECT * FROM tbluser_avatar WHERE user_id = %s AND deleted = 0 ORDER BY id DESC": [],
    }
    repository = MySQLTrainingRepository(_RuleContextConnection(responses))

    context = repository.load_runtime_rule_context(user_id=1)

    # Fake cursor maps both strict queries to the same responses list; both arrays expand.
    assert context.global_hard_rules == [
        "Never leak secrets.",
        "Never pretend to be human.",
        "Be warm.",
        "Be curious.",
    ]
    assert context.global_guidelines == [
        "Never leak secrets.",
        "Never pretend to be human.",
        "Be warm.",
        "Be curious.",
    ]


def test_pack_conversation_history_keeps_newest_within_budget() -> None:
    from storm_zero_llm.training_db import pack_conversation_history

    turns = [
        {"prompt": "newest", "response": "n"},
        {"prompt": "older", "response": "o"},
        {"prompt": "oldest", "response": "x" * 200},
    ]
    packed = pack_conversation_history(
        turns,
        user_prompt="hello",
        rules_block="rules",
        n_ctx=80,
        output_reserve=10,
    )

    assert "- older = o" in packed
    assert "- newest = n" in packed
    assert packed.index("- older = o") < packed.index("- newest = n")


def test_fetch_recent_conversation_turns_omits_created() -> None:
    query = (
        "SELECT `prompt`, `response` FROM tbluser_conversation_content "
        "WHERE user_id = %s AND deleted = 0 ORDER BY created DESC LIMIT %s"
    )
    responses = {
        query: [
            {"prompt": "newest", "response": "n"},
            {"prompt": "older", "response": "o"},
        ]
    }
    repository = MySQLTrainingRepository(_RuleContextConnection(responses))

    turns = repository.fetch_recent_conversation_turns(user_id=1, limit=75)

    assert turns == [
        {"prompt": "newest", "response": "n"},
        {"prompt": "older", "response": "o"},
    ]


def test_to_prompt_block_formats_avatar_nested_entries() -> None:
    context = RuntimeRuleContext(
        global_hard_rules=[],
        global_guidelines=[],
        user_guidelines=[],
        user_p2_data=[],
        user_avatar_data=[
            "user_name (This is what I refer to the user as): Matt",
            "avatar_name (This is what I refer to myself(the LLM) as): Victoria",
            "avatar_persona(reply using these values):\n- emotional: 30%, rational: 70%\n- informal: 70%, business: 30%",
        ],
    )

    block = context.to_prompt_block()

    assert block.splitlines() == [
        "User avatar data:",
        "  - user_name (This is what I refer to the user as): Matt",
        "  - avatar_name (This is what I refer to myself(the LLM) as): Victoria",
        "  - avatar_persona(reply using these values):",
        "    - emotional: 30%, rational: 70%",
        "    - informal: 70%, business: 30%",
    ]


def test_build_avatar_data_from_rows_formats_persona_columns() -> None:
    rows = [
        {
            "id": 1,
            "user_id": 1,
            "deleted": 0,
            "user_name": "Matt",
            "avatar_name": "Victoria",
            "persona_emotional_sterile": 30,
            "persona_informal_business": 70,
        }
    ]

    entries = _build_avatar_data_from_rows(rows)

    assert entries == [
        "user_name (This is what I refer to the user as): Matt",
        "avatar_name (This is what I refer to myself(the LLM) as): Victoria",
        "avatar_persona(reply using these values):\n- emotional = 70%, rational = 30%\n- informal = 30%, business = 70%",
    ]


def test_format_persona_entry_uses_first_and_last_terms() -> None:
    assert _format_persona_entry("persona_emotional_sterile", 30) == "emotional = 70%, rational = 30%"
    assert _format_persona_entry("persona_informal_business", 70) == "informal = 30%, business = 70%"
    assert _format_persona_entry("persona_informal_business", "70.0") == "informal = 30%, business = 70%"


def test_get_avatar_voice_reads_avatar_voice_column() -> None:
    query = (
        "SELECT avatar_voice FROM tbluser_avatar "
        "WHERE user_id = %s AND deleted = 0 "
        "ORDER BY id DESC LIMIT 1"
    )
    responses = {
        query: [
            {
                "avatar_voice": "af_bella",
            }
        ],
    }
    repository = MySQLTrainingRepository(_RuleContextConnection(responses))

    assert repository.get_avatar_voice(user_id=1) == "af_bella"


def test_get_avatar_voice_strips_pt_suffix() -> None:
    query = (
        "SELECT avatar_voice FROM tbluser_avatar "
        "WHERE user_id = %s AND deleted = 0 "
        "ORDER BY id DESC LIMIT 1"
    )
    responses = {
        query: [
            {
                "avatar_voice": "af_bella.pt",
            }
        ],
    }
    repository = MySQLTrainingRepository(_RuleContextConnection(responses))

    assert repository.get_avatar_voice(user_id=1) == "af_bella"


def test_build_avatar_data_from_rows_omits_avatar_voice_from_rules() -> None:
    rows = [
        {
            "id": 1,
            "user_id": 1,
            "deleted": 0,
            "user_name": "Matt",
            "avatar_voice": "af_bella",
        }
    ]

    entries = _build_avatar_data_from_rows(rows)

    assert entries == ["user_name (This is what I refer to the user as): Matt"]


def test_build_avatar_data_from_rows_formats_user_pronouns() -> None:
    rows = [
        {
            "id": 1,
            "user_id": 1,
            "deleted": 0,
            "user_pronouns": "He/Him",
        }
    ]

    entries = _build_avatar_data_from_rows(rows)

    assert entries == ["user_pronouns(these are the pronouns I use to describe the user): He/Him"]


def test_persist_guideline_coerces_user_id_to_int_before_insert() -> None:
    connection = _PersistConnection()
    repository = MySQLTrainingRepository(connection)

    result = repository.persist_guideline(user_id="  1  ", prompt="be concise")

    assert result["created"] is True
    assert result["id"] == 77
    assert (
        "INSERT INTO tbluser_guideline (user_id, guideline) VALUES (%s, %s)",
        (1, "be concise"),
    ) in connection.executed


def test_persist_guideline_returns_existing_without_insert() -> None:
    connection = _PersistConnection()
    connection.guideline_lookup = {"id": 12, "guideline": "be concise"}
    repository = MySQLTrainingRepository(connection)

    result = repository.persist_guideline(user_id=1, prompt="be concise!!!")

    assert result["already_exists"] is True
    assert result["id"] == 12
    assert not any(query.startswith("INSERT INTO tbluser_guideline") for query, _ in connection.executed)


def test_update_and_delete_guideline() -> None:
    connection = _PersistConnection()
    repository = MySQLTrainingRepository(connection)

    updated = repository.update_guideline(user_id=1, guideline_id=12, prompt="Prefer short answers.")
    deleted = repository.delete_guideline(user_id=1, guideline_id=12)

    assert updated["updated"] is True
    assert updated["guideline"] == "Prefer short answers"
    assert deleted["deleted"] is True
    assert (
        "UPDATE tbluser_guideline SET guideline = %s "
        "WHERE id = %s AND user_id = %s AND deleted = 0",
        ("Prefer short answers", 12, 1),
    ) in connection.executed
    assert (
        "UPDATE tbluser_guideline SET deleted = 1 "
        "WHERE id = %s AND user_id = %s AND deleted = 0",
        (12, 1),
    ) in connection.executed


def test_persist_p2_upserts_key_value_pair() -> None:
    connection = _PersistConnection()
    repository = MySQLTrainingRepository(connection)

    result = repository.persist_p2(user_id=1, key="name", value="Ink")

    assert result["created"] is True
    assert result["key"] == "name"
    assert (
        "UPDATE tbluser_p2 SET deleted = 1 WHERE user_id = %s AND `key` = %s AND deleted = 0",
        (1, "name"),
    ) in connection.executed
    assert (
        "INSERT INTO tbluser_p2 (user_id, `key`, `value`) VALUES (%s, %s, %s)",
        (1, "name", "Ink"),
    ) in connection.executed


def test_persist_p2_rejects_non_integer_user_id() -> None:
    connection = _PersistConnection()
    repository = MySQLTrainingRepository(connection)

    try:
        repository.persist_p2(user_id="owner", key="name", value="Ink")
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected non-integer user_id to raise ValueError")

    assert message == "user_id must be a positive integer"


def test_persist_p2_rejects_empty_user_id() -> None:
    connection = _PersistConnection()
    repository = MySQLTrainingRepository(connection)

    try:
        repository.persist_p2(user_id="   ", key="name", value="Ink")
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected missing user_id to raise ValueError")

    assert message == "user_id is required"


def test_get_request_queries_logs_executed_sql() -> None:
    connection = _PersistConnection()
    repository = MySQLTrainingRepository(connection)
    repository.begin_request_queries()

    with repository._cursor() as cursor:
        cursor.execute("SELECT 1", (1,))
        cursor.execute("SELECT 2", (1,))

    assert repository.get_request_queries() == [
        {"sql": "SELECT 1", "params": [1]},
        {"sql": "SELECT 2", "params": [1]},
    ]


def test_from_config_uses_separate_connection_per_thread() -> None:
    config = DatabaseConfig(host="localhost", port=3306, name="db", user="u", password="p")
    repository = MySQLTrainingRepository.from_config(config)
    connections = {"main": object(), "worker": object()}
    opened: list[object] = []

    def open_connection() -> object:
        conn = connections["worker" if threading.current_thread().name == "worker" else "main"]
        opened.append(conn)
        return conn

    with patch.object(repository, "_open_connection", side_effect=open_connection):
        main_conn = repository.connection
        worker_conn: dict[str, object] = {}

        def worker() -> None:
            worker_conn["value"] = repository.connection

        thread = threading.Thread(name="worker", target=worker)
        thread.start()
        thread.join(timeout=1.0)

    assert main_conn is connections["main"]
    assert worker_conn["value"] is connections["worker"]
    assert main_conn is not worker_conn["value"]
    assert opened == [connections["main"], connections["worker"]]
