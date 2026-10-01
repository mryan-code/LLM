from pathlib import Path

from storm_zero_llm.memory import FileMemoryStore, MemoryRecord, MemoryType


def test_user_memory_file_is_suffixed_with_user_id(tmp_path: Path) -> None:
    store = FileMemoryStore(tmp_path)
    record = MemoryRecord(
        memory_type=MemoryType.PROFILE,
        scope="user",
        user_id=1,
        title="Favorite movies",
        content="likes cyberpunk",
    )

    path = store.add(record)
    assert path.name.endswith("_1.json")
