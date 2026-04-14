"""Unit tests for Settings / config module."""
import pytest
from pydantic import ValidationError


def test_settings_loads_from_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    from aetheris.config import get_settings, Settings
    get_settings.cache_clear()
    s = Settings()
    assert s.openai_api_key == "sk-test"
    get_settings.cache_clear()


def test_invalid_log_level_raises(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "NONSENSE")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    from aetheris.config import Settings
    with pytest.raises(ValidationError):
        Settings()


def test_cors_origins_list(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("CORS_ORIGINS", "http://localhost:8501,http://localhost:3000")
    from aetheris.config import Settings
    s = Settings()
    assert len(s.cors_origins_list) == 2
    assert "http://localhost:8501" in s.cors_origins_list


def test_is_production_false_by_default(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    from aetheris.config import Settings
    s = Settings()
    assert s.is_production is False


def test_get_settings_cached(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    from aetheris.config import get_settings
    get_settings.cache_clear()
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
    get_settings.cache_clear()
