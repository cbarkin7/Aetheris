"""Unit tests for long-term memory using in-memory SQLite."""
import pytest

from aetheris.memory.long_term import delete_user_memory, load_user_memory, upsert_user_memory


@pytest.fixture
def mem_db(tmp_path):
    return str(tmp_path / "test_memory.db")


def test_load_empty_returns_empty_dict(mem_db):
    result = load_user_memory("user1", db_path=mem_db)
    assert result == {}


def test_upsert_and_load(mem_db):
    upsert_user_memory("user1", {"language": "Spanish", "timezone": "UTC+1"}, db_path=mem_db)
    result = load_user_memory("user1", db_path=mem_db)
    assert result["language"] == "Spanish"
    assert result["timezone"] == "UTC+1"


def test_upsert_overwrites_existing_key(mem_db):
    upsert_user_memory("user1", {"language": "English"}, db_path=mem_db)
    upsert_user_memory("user1", {"language": "French"}, db_path=mem_db)
    result = load_user_memory("user1", db_path=mem_db)
    assert result["language"] == "French"


def test_user_isolation(mem_db):
    upsert_user_memory("user1", {"pref": "A"}, db_path=mem_db)
    upsert_user_memory("user2", {"pref": "B"}, db_path=mem_db)
    assert load_user_memory("user1", db_path=mem_db)["pref"] == "A"
    assert load_user_memory("user2", db_path=mem_db)["pref"] == "B"


def test_delete_user_memory(mem_db):
    upsert_user_memory("user1", {"key": "value"}, db_path=mem_db)
    delete_user_memory("user1", db_path=mem_db)
    assert load_user_memory("user1", db_path=mem_db) == {}


def test_upsert_empty_dict_is_noop(mem_db):
    upsert_user_memory("user1", {}, db_path=mem_db)
    assert load_user_memory("user1", db_path=mem_db) == {}
