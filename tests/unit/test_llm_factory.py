"""
Tests unitarios para la factoría LLM (OpenAI principal + Bedrock fallback).
"""
import pytest
from unittest.mock import MagicMock, patch


def _mock_llm(name: str = "FakeLLM"):
    m = MagicMock()
    m.__class__.__name__ = name
    m.with_fallbacks = MagicMock(return_value=m)
    m.bind_tools = MagicMock(return_value=m)
    return m


class TestLLMFactory:

    def setup_method(self):
        """Limpiar caché LLM antes de cada test."""
        from aetheris.llm import clear_llm_cache
        from aetheris.config import get_settings
        clear_llm_cache()
        get_settings.cache_clear()

    def teardown_method(self):
        from aetheris.llm import clear_llm_cache
        from aetheris.config import get_settings
        clear_llm_cache()
        get_settings.cache_clear()

    def test_returns_openai_provider(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "")
        fake = _mock_llm("ChatOpenAI")
        with patch("aetheris.llm._build_openai_llm", return_value=fake):
            from aetheris.llm import get_llm
            _, provider = get_llm()
        assert provider == "openai"

    def test_returns_bedrock_only_when_no_openai(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "")
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAEXAMPLE")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
        fake = _mock_llm("ChatBedrockConverse")
        with patch("aetheris.llm._build_bedrock_llm", return_value=fake):
            from aetheris.llm import get_llm
            _, provider = get_llm()
        assert provider == "bedrock"

    def test_with_fallbacks_called_when_bedrock_available(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAEXAMPLE")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
        fake_openai = _mock_llm("ChatOpenAI")
        fake_bedrock = _mock_llm("ChatBedrockConverse")
        with patch("aetheris.llm._build_openai_llm", return_value=fake_openai), \
             patch("aetheris.llm._build_bedrock_llm", return_value=fake_bedrock):
            from aetheris.llm import get_llm
            get_llm()
        fake_openai.with_fallbacks.assert_called_once_with([fake_bedrock])

    def test_raises_without_any_provider(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "")
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "")
        from aetheris.llm import get_llm
        with pytest.raises(RuntimeError, match="No hay proveedor LLM"):
            get_llm()

    def test_bind_tools_when_provided(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "")
        fake = _mock_llm("ChatOpenAI")
        mock_tool = MagicMock()
        with patch("aetheris.llm._build_openai_llm", return_value=fake):
            from aetheris.llm import get_llm
            get_llm(tools=[mock_tool])
        fake.bind_tools.assert_called_once_with([mock_tool])

    def test_returns_tuple(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "")
        fake = _mock_llm("ChatOpenAI")
        with patch("aetheris.llm._build_openai_llm", return_value=fake):
            from aetheris.llm import get_llm
            result = get_llm()
        assert isinstance(result, tuple) and len(result) == 2
        assert isinstance(result[1], str)

    def test_llm_cached_on_second_call(self, monkeypatch):
        """El LLM base debe construirse solo una vez."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "")
        fake = _mock_llm("ChatOpenAI")
        with patch("aetheris.llm._build_openai_llm", return_value=fake) as mock_build:
            from aetheris.llm import get_llm
            get_llm()
            get_llm()
        mock_build.assert_called_once()
