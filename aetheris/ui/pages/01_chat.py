"""
AETHERIS — Página de Chat con streaming SSE, HITL y entrada de audio.
"""
import json
import os
import sys
import uuid
from pathlib import Path

# Garantiza que la raíz del proyecto esté en sys.path
_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

import requests
import streamlit as st

from aetheris.ui.components.chat_message import render_message
from aetheris.ui.components.hitl_modal import render_hitl_modal

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

st.title("AETHERIS")
st.caption("Agente Cognitivo Autónomo")

# ---------------------------------------------------------------------------
# Estado de sesión
# ---------------------------------------------------------------------------
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "user_id" not in st.session_state:
    st.session_state.user_id = "Admin-Aetheris"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "hitl_pending" not in st.session_state:
    st.session_state.hitl_pending = None
if "last_audio_key" not in st.session_state:
    st.session_state.last_audio_key = None

# ---------------------------------------------------------------------------
# Barra lateral
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Sesión")
    # Usamos una clave auxiliar para evitar que st.text_input(key="user_id")
    # sobreescriba la sesión con cadena vacía en el primer renderizado.
    new_user_id = st.text_input(
        "ID de usuario",
        value=st.session_state.user_id,
        key="_user_id_input",
    )
    if new_user_id and new_user_id != st.session_state.user_id:
        st.session_state.user_id = new_user_id

    st.text_input(
        "ID de conversación",
        value=st.session_state.thread_id,
        key="_thread_id_display",
        disabled=True,
        help="Copia este ID para retomar la conversación más tarde.",
    )
    if st.button("Nueva conversación"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.hitl_pending = None
        st.session_state.last_audio_key = None
        st.rerun()

    st.divider()

    # Recuperar conversación existente por ID
    with st.expander("🔄 Retomar conversación"):
        recovery_id = st.text_input(
            "ID de conversación",
            key="_recovery_id",
            placeholder="Pega aquí el ID…",
        )
        if st.button("Cargar", key="_load_conv"):
            if recovery_id and recovery_id.strip():
                st.session_state.thread_id = recovery_id.strip()
                st.session_state.messages = []
                st.session_state.hitl_pending = None
                st.session_state.last_audio_key = None
                st.success(f"Conversación `{recovery_id[:12]}…` cargada")
                st.rerun()
            else:
                st.warning("Introduce un ID de conversación válido.")

    st.divider()
    st.caption("Modelo: GPT-4o-mini · Fallback: Bedrock")


# ---------------------------------------------------------------------------
# Función auxiliar: transcribir audio
# ---------------------------------------------------------------------------
def _transcribe_audio(audio_file) -> str | None:
    """Envía el audio al backend y devuelve el texto transcrito."""
    try:
        resp = requests.post(
            f"{API_BASE}/api/v1/speech/transcribe",
            files={"file": (audio_file.name, audio_file.getvalue(), audio_file.type or "audio/wav")},
            timeout=30,
        )
        if resp.ok:
            text = resp.json().get("text", "").strip()
            return text or None
        st.error(f"Error en la transcripción: {resp.json().get('detail', 'Error desconocido')}")
    except Exception as exc:
        st.error(f"No se pudo conectar con el servicio de audio: {exc}")
    return None


# ---------------------------------------------------------------------------
# Función auxiliar: enviar mensaje al agente y hacer streaming
# ---------------------------------------------------------------------------
def _send_message(prompt: str) -> None:
    """Envía el mensaje al agente y muestra la respuesta en streaming."""
    st.session_state.messages.append({"role": "user", "content": prompt})
    render_message("user", prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        hitl_actions = None

        try:
            resp = requests.post(
                f"{API_BASE}/api/v1/chat",
                json={
                    "message": prompt,
                    "thread_id": st.session_state.thread_id,
                    "user_id": st.session_state.user_id,
                    "stream": True,
                },
                stream=True,
                timeout=(10, 180),  # (connect_timeout, read_timeout) — 180s para búsquedas largas
            )
            resp.raise_for_status()

            for line in resp.iter_lines():
                if not line or not line.startswith(b"data: "):
                    continue
                event = json.loads(line[6:])
                etype = event.get("type")

                if etype == "conversation_id":
                    # Actualizar el thread_id con el asignado por el backend
                    # (puede diferir si el cliente no lo envió)
                    st.session_state.thread_id = event["thread_id"]

                elif etype == "token":
                    full_response += event["content"]
                    placeholder.markdown(full_response + "▌")

                elif etype == "hitl_required":
                    hitl_actions = event.get("actions", [])
                    placeholder.markdown("⏳ **Acción lista — necesito tu aprobación para continuar.**")
                    break

                elif etype == "guardrail_blocked":
                    violations = event.get("violations", [])
                    placeholder.markdown(
                        "⚠️ **Solicitud bloqueada por seguridad.**\n\n"
                        f"Motivo: `{', '.join(violations)}`"
                    )
                    break

                elif etype == "done":
                    placeholder.markdown(full_response)
                    break

                elif etype == "error":
                    st.error(f"Error del agente: {event.get('message')}")
                    break

        except Exception as exc:
            st.error(f"Error de conexión: {exc}")

    if full_response:
        st.session_state.messages.append({"role": "assistant", "content": full_response})

    if hitl_actions:
        # Guardar en historial el aviso antes del rerun para que aparezca
        # como último mensaje del chat y el usuario vea que hay algo pendiente.
        st.session_state.messages.append({
            "role": "assistant",
            "content": "⏳ **Acción lista — necesito tu aprobación para continuar.** Revisa la solicitud a continuación.",
        })
        st.session_state.hitl_pending = hitl_actions
        st.rerun()


# ---------------------------------------------------------------------------
# Historial de conversación
# ---------------------------------------------------------------------------
for msg in st.session_state.messages:
    render_message(msg["role"], msg["content"])

# ---------------------------------------------------------------------------
# Modal HITL — se renderiza AL FINAL del historial para que el usuario lo vea
# ---------------------------------------------------------------------------
if st.session_state.hitl_pending:
    approval = render_hitl_modal(st.session_state.hitl_pending)
    if approval is not None:
        st.session_state.hitl_pending = None
        # Eliminar el mensaje "⏳ Acción lista…" del historial antes de reanudar
        st.session_state.messages = [
            m for m in st.session_state.messages
            if "Acción lista" not in m.get("content", "")
        ]

        try:
            resp = requests.post(
                f"{API_BASE}/api/v1/chat/{st.session_state.thread_id}/resume",
                json={"approved": approval, "user_id": st.session_state.user_id},
                stream=True,
                timeout=60,
            )
            resp.raise_for_status()
        except Exception as exc:
            st.error(f"No se pudo reanudar la conversación: {exc}")
            st.rerun()
        else:
            with st.chat_message("assistant"):
                placeholder = st.empty()
                full_response = ""
                action_feedback: list[str] = []

                for line in resp.iter_lines():
                    if not line or not line.startswith(b"data: "):
                        continue
                    event = json.loads(line[6:])
                    etype = event.get("type")

                    if etype == "action_result":
                        name = event.get("name", "")
                        summary = event.get("summary", "")
                        msg = f"✅ **{name}** ejecutado correctamente."
                        if summary:
                            msg += f"\n> {summary[:200]}"
                        action_feedback.append(msg)
                        placeholder.markdown("\n\n".join(action_feedback) + "\n\n_Generando resumen…_ ⏳")

                    elif etype == "action_error":
                        name = event.get("name", "")
                        error = event.get("error", "Error desconocido")
                        action_feedback.append(f"❌ **{name}** ha fallado: {error}")
                        placeholder.markdown("\n\n".join(action_feedback) + "\n\n_Continuando…_ ⏳")

                    elif etype == "token":
                        full_response += event["content"]
                        prefix = "\n\n".join(action_feedback) + "\n\n---\n" if action_feedback else ""
                        placeholder.markdown(prefix + full_response + "▌")

                    elif etype == "done":
                        prefix = "\n\n".join(action_feedback) + "\n\n---\n" if action_feedback else ""
                        placeholder.markdown(prefix + full_response)
                        break

                    elif etype == "error":
                        st.error(f"Error durante la ejecución: {event.get('message', '')}")
                        break

            combined_parts = action_feedback[:]
            if full_response:
                combined_parts.append(full_response)
            if combined_parts:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "\n\n".join(combined_parts),
                })

            st.rerun()

# ---------------------------------------------------------------------------
# Entrada de audio (sobre la entrada de texto)
# ---------------------------------------------------------------------------
st.markdown("#### Habla con AETHERIS")
col_audio, col_info = st.columns([2, 3])

with col_audio:
    audio_file = st.file_uploader(
        "🎤 Sube un audio",
        type=["mp3", "wav", "m4a", "ogg", "webm"],
        key="audio_upload",
        label_visibility="visible",
        help="Formatos: mp3, wav, m4a, ogg, webm. El audio se transcribirá automáticamente.",
    )

with col_info:
    if audio_file is not None:
        audio_key = f"{audio_file.name}_{audio_file.size}"
        if audio_key != st.session_state.last_audio_key:
            st.session_state.last_audio_key = audio_key
            st.audio(audio_file)
            with st.spinner("Transcribiendo audio…"):
                transcribed = _transcribe_audio(audio_file)
            if transcribed:
                st.success(f"**Transcripción:** _{transcribed}_")
                _send_message(transcribed)
            else:
                st.warning("No se pudo transcribir el audio. Intenta escribir el mensaje.")

# ---------------------------------------------------------------------------
# Entrada de texto — deshabilitada mientras hay una aprobación HITL pendiente
# ---------------------------------------------------------------------------
if st.session_state.hitl_pending:
    st.chat_input("Aprueba o rechaza la acción antes de continuar…", disabled=True)
else:
    if prompt := st.chat_input("O escribe tu mensaje aquí…"):
        _send_message(prompt)
