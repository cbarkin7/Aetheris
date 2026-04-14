"""
Test de integración: endpoint de Speech-to-Text con faster-whisper simulado.
No carga el modelo real — mockea WhisperModel para evitar descarga en CI.
"""
import io
import pytest
from unittest.mock import MagicMock, patch


@pytest.mark.integration
def test_transcribe_returns_text(api_client):
    """El endpoint debe devolver el texto transcrito por el modelo simulado."""
    mock_segment = MagicMock()
    mock_segment.text = " Esto es una prueba de transcripción."

    mock_info = MagicMock()
    mock_info.language = "es"
    mock_info.language_probability = 0.98

    with patch("aetheris.api.routers.speech._get_whisper_model") as mock_model_fn:
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([mock_segment], mock_info)
        mock_model_fn.return_value = mock_model

        resp = api_client.post(
            "/api/v1/speech/transcribe",
            files={"file": ("audio.wav", io.BytesIO(b"fake_audio_bytes"), "audio/wav")},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert "text" in data
    assert "prueba" in data["text"]
    assert data["filename"] == "audio.wav"
    assert data["chars"] > 0


@pytest.mark.integration
def test_transcribe_rejects_unsupported_format(api_client):
    """Formatos no soportados deben devolver 400."""
    resp = api_client.post(
        "/api/v1/speech/transcribe",
        files={"file": ("audio.xyz", io.BytesIO(b"bytes"), "application/octet-stream")},
    )
    assert resp.status_code == 400
    assert "Formato no soportado" in resp.json()["detail"]


@pytest.mark.integration
def test_transcribe_rejects_empty_file(api_client):
    """Ficheros vacíos deben devolver 400."""
    with patch("aetheris.api.routers.speech._get_whisper_model"):
        resp = api_client.post(
            "/api/v1/speech/transcribe",
            files={"file": ("audio.wav", io.BytesIO(b""), "audio/wav")},
        )
    assert resp.status_code == 400
    assert "vacío" in resp.json()["detail"]


@pytest.mark.integration
def test_whisper_model_singleton():
    """El modelo debe cargarse solo una vez aunque se llame varias veces."""
    from aetheris.api.routers import speech
    speech._whisper_model = None  # reset singleton

    mock_model = MagicMock()
    with patch("aetheris.api.routers.speech.WhisperModel", return_value=mock_model):
        m1 = speech._get_whisper_model()
        m2 = speech._get_whisper_model()

    assert m1 is m2  # misma instancia
    speech._whisper_model = None  # limpiar para otros tests
