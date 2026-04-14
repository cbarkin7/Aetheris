"""
Endpoint de Speech-to-Text mediante faster-whisper (ejecución local, sin coste de API).

El modelo se descarga automáticamente en el primer uso y se cachea en memoria
durante la vida de la aplicación para evitar recargas en cada petición.

Formatos de audio soportados: mp3, wav, m4a, ogg, webm, flac.
"""
import logging
import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, UploadFile, File

from aetheris.config import get_settings

try:
    from faster_whisper import WhisperModel
except ImportError:  # pragma: no cover — faster-whisper optional in CI
    WhisperModel = None  # type: ignore[assignment,misc]

router = APIRouter(prefix="/api/v1/speech", tags=["speech"])
logger = logging.getLogger(__name__)

_SUPPORTED_EXTENSIONS = frozenset({".mp3", ".wav", ".m4a", ".ogg", ".webm", ".flac"})

# Singleton del modelo faster-whisper — se carga una sola vez
_whisper_model: Any = None


def _get_whisper_model():
    """Devuelve el modelo faster-whisper cacheado, cargándolo si es necesario."""
    global _whisper_model
    if _whisper_model is None:
        s = get_settings()
        logger.info(
            "Cargando modelo faster-whisper '%s' (device=%s, compute=%s)…",
            s.whisper_model_size, s.whisper_device, s.whisper_compute_type,
        )
        _whisper_model = WhisperModel(
            s.whisper_model_size,
            device=s.whisper_device,
            compute_type=s.whisper_compute_type,
        )
        logger.info("Modelo faster-whisper cargado correctamente.")
    return _whisper_model


def _transcribe_bytes(audio_bytes: bytes, filename: str) -> str:
    """
    Transcribe audio desde bytes usando faster-whisper.
    Escribe un fichero temporal, transcribe y lo elimina.
    """
    ext = Path(filename).suffix.lower() or ".wav"
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        model = _get_whisper_model()
        segments, info = model.transcribe(
            tmp_path,
            beam_size=5,
            language=None,  # detección automática de idioma
            vad_filter=True,  # filtro de actividad de voz (elimina silencios)
            vad_parameters={"min_silence_duration_ms": 500},
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
        logger.info(
            "STT: '%s' transcrito en idioma='%s' (probabilidad=%.2f) → %d chars",
            filename, info.language, info.language_probability, len(text),
        )
        return text
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@router.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(..., description="Fichero de audio a transcribir"),
) -> dict:
    """
    Transcribe un fichero de audio a texto usando faster-whisper (ejecución local).

    No realiza ninguna llamada a API externa — el modelo corre en el servidor.

    Formatos soportados: mp3, wav, m4a, ogg, webm, flac.

    Devuelve:
        {"text": "transcripción", "filename": "nombre_fichero", "chars": 42}
    """
    filename = file.filename or "audio.wav"
    ext = Path(filename).suffix.lower()

    if ext and ext not in _SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Formato no soportado: '{ext}'. Usa mp3, wav, m4a, ogg, webm o flac.",
        )

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="El fichero de audio está vacío.")

    try:
        text = _transcribe_bytes(audio_bytes, filename)
        return {"text": text, "filename": filename, "chars": len(text)}
    except Exception as exc:
        logger.error("STT: error en transcripción de '%s': %s", filename, exc)
        raise HTTPException(
            status_code=500,
            detail=f"Error durante la transcripción: {exc}",
        )
