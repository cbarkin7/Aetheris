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
# Prompt pendiente generado por el uploader de audio (se procesa en el render loop)
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None


# ---------------------------------------------------------------------------
# Helpers: API
# ---------------------------------------------------------------------------
def _load_history(thread_id: str) -> list[dict]:
    """Carga el historial de mensajes desde el backend para un thread dado."""
    try:
        resp = requests.get(
            f"{API_BASE}/api/v1/chat/{thread_id}/history",
            params={"user_id": st.session_state.user_id},
            timeout=10,
        )
        if resp.ok:
            data = resp.json()
            return [
                {"role": m["role"], "content": m["content"]}
                for m in data.get("messages", [])
                if m["role"] in ("human", "ai") and m["content"].strip()
            ]
    except Exception:
        pass
    return []


def _load_conversation_list() -> list[dict]:
    """Devuelve la lista de conversaciones del usuario desde el backend."""
    try:
        resp = requests.get(
            f"{API_BASE}/api/v1/chat/threads/{st.session_state.user_id}",
            params={"limit": 30},
            timeout=5,
        )
        if resp.ok:
            return resp.json().get("conversations", [])
    except Exception:
        pass
    return []


def _delete_conversation(thread_id: str) -> bool:
    """Llama al backend para eliminar la conversación y todos sus datos."""
    try:
        resp = requests.delete(
            f"{API_BASE}/api/v1/chat/{thread_id}",
            timeout=10,
        )
        return resp.ok
    except Exception:
        return False


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
# Helper: enviar mensaje al agente y hacer streaming
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
                timeout=(10, 180),
            )
            resp.raise_for_status()

            for line in resp.iter_lines():
                if not line or not line.startswith(b"data: "):
                    continue
                event = json.loads(line[6:])
                etype = event.get("type")

                if etype == "conversation_id":
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
        st.session_state.messages.append({
            "role": "assistant",
            "content": "⏳ **Acción lista — necesito tu aprobación para continuar.** Revisa la solicitud a continuación.",
        })
        st.session_state.hitl_pending = hitl_actions
        st.rerun()


# ---------------------------------------------------------------------------
# Barra lateral — historial de conversaciones
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Sesión")
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

    col_new, col_refresh = st.columns(2)
    with col_new:
        if st.button("➕ Nueva", use_container_width=True):
            st.session_state.thread_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.session_state.hitl_pending = None
            st.session_state.last_audio_key = None
            st.session_state.pending_prompt = None
            st.rerun()

    st.divider()

    # ── Historial lateral ──────────────────────────────────────────────────
    st.subheader("💬 Conversaciones")
    conversations = _load_conversation_list()

    # Rastrear qué conversación está esperando confirmación de borrado
    if "_confirm_delete_tid" not in st.session_state:
        st.session_state._confirm_delete_tid = None

    if conversations:
        for conv in conversations:
            tid = conv["thread_id"]
            title = conv["title"] or "Sin título"
            is_active = tid == st.session_state.thread_id
            is_confirming = st.session_state._confirm_delete_tid == tid

            if is_confirming:
                # ── Estado de confirmación de borrado ──────────────────────
                st.warning(f"¿Eliminar «{title[:30]}»?", icon="⚠️")
                col_ok, col_cancel = st.columns(2)
                with col_ok:
                    if st.button("Sí, borrar", key=f"_del_ok_{tid}", use_container_width=True, type="primary"):
                        with st.spinner("Eliminando…"):
                            ok = _delete_conversation(tid)
                        st.session_state._confirm_delete_tid = None
                        if ok:
                            # Si se borró la conversación activa, crear nueva sesión
                            if is_active:
                                st.session_state.thread_id = str(uuid.uuid4())
                                st.session_state.messages = []
                                st.session_state.hitl_pending = None
                        else:
                            st.error("No se pudo eliminar la conversación.")
                        st.rerun()
                with col_cancel:
                    if st.button("Cancelar", key=f"_del_cancel_{tid}", use_container_width=True):
                        st.session_state._confirm_delete_tid = None
                        st.rerun()
            else:
                # ── Fila normal: botón de conversación + botón de borrado ──
                # Título truncado para dejar espacio al icono 🗑️
                label = title[:34] + "…" if len(title) > 34 else title
                col_conv, col_del = st.columns([5, 1])
                with col_conv:
                    button_type = "primary" if is_active else "secondary"
                    if st.button(label, key=f"_conv_{tid}", use_container_width=True, type=button_type):
                        if not is_active:
                            st.session_state.thread_id = tid
                            st.session_state.hitl_pending = None
                            st.session_state.last_audio_key = None
                            st.session_state.pending_prompt = None
                            st.session_state.messages = _load_history(tid)
                            st.rerun()
                with col_del:
                    if st.button("🗑️", key=f"_del_{tid}", help="Eliminar esta conversación"):
                        st.session_state._confirm_delete_tid = tid
                        st.rerun()
    else:
        st.caption("Aún no hay conversaciones guardadas.")

    st.divider()
    st.caption("Modelo: GPT-4o-mini · Fallback: Bedrock")


# ---------------------------------------------------------------------------
# Historial de conversación — renderizado principal
# ---------------------------------------------------------------------------
for msg in st.session_state.messages:
    render_message(msg["role"], msg["content"])

# ---------------------------------------------------------------------------
# Modal HITL
# ---------------------------------------------------------------------------
if st.session_state.hitl_pending:
    approval = render_hitl_modal(st.session_state.hitl_pending)
    if approval is not None:
        st.session_state.hitl_pending = None
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
                next_hitl_actions = None  # próximas acciones que necesitan aprobación

                for line in resp.iter_lines():
                    if not line or not line.startswith(b"data: "):
                        continue
                    event = json.loads(line[6:])
                    etype = event.get("type")

                    if etype == "action_result":
                        name = event.get("name", "")
                        summary = event.get("summary", "")
                        msg_text = f"✅ **{name}** ejecutado correctamente."
                        if summary:
                            msg_text += f"\n> {summary[:200]}"
                        action_feedback.append(msg_text)
                        placeholder.markdown("\n\n".join(action_feedback) + "\n\n_Procesando siguiente acción…_ ⏳")

                    elif etype == "action_error":
                        name = event.get("name", "")
                        error = event.get("error", "Error desconocido")
                        action_feedback.append(f"❌ **{name}** ha fallado: {error}")
                        placeholder.markdown("\n\n".join(action_feedback) + "\n\n_Continuando…_ ⏳")

                    elif etype == "hitl_required":
                        # Siguiente acción en la cola necesita aprobación del usuario.
                        # El grafo está pausado — guardar el progreso y mostrar el modal.
                        next_hitl_actions = event.get("actions", [])
                        if action_feedback:
                            placeholder.markdown("\n\n".join(action_feedback) + "\n\n⏳ **Siguiente acción lista — necesito tu aprobación para continuar.**")
                        else:
                            placeholder.markdown("⏳ **Acción lista — necesito tu aprobación para continuar.**")
                        break

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

            # Guardar el progreso intermedio (acciones ejecutadas hasta ahora)
            combined_parts = action_feedback[:]
            if full_response:
                combined_parts.append(full_response)

            if next_hitl_actions:
                # Hay más acciones en cola que necesitan aprobación.
                # Guardar el progreso y activar el modal HITL para la siguiente acción.
                if combined_parts:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "\n\n".join(combined_parts),
                    })
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "⏳ **Acción lista — necesito tu aprobación para continuar.** Revisa la solicitud a continuación.",
                })
                st.session_state.hitl_pending = next_hitl_actions
            elif combined_parts:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "\n\n".join(combined_parts),
                })

            st.rerun()

# ---------------------------------------------------------------------------
# Entrada de audio
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
                # Guardar como prompt pendiente con indicador de audio y hacer rerun
                # para que el mensaje se envíe desde el contexto principal (no desde
                # dentro del col_info), asegurando que aparece en el hilo de chat.
                st.session_state.pending_prompt = f"🎤 {transcribed}"
                st.rerun()
            else:
                st.warning("No se pudo transcribir el audio. Intenta escribir el mensaje.")

# ---------------------------------------------------------------------------
# Procesar prompt pendiente de audio (fuera del col_info, en el flujo principal)
# ---------------------------------------------------------------------------
if st.session_state.pending_prompt and not st.session_state.hitl_pending:
    prompt_to_send = st.session_state.pending_prompt
    st.session_state.pending_prompt = None
    _send_message(prompt_to_send)

# ---------------------------------------------------------------------------
# Entrada de texto
# ---------------------------------------------------------------------------
if st.session_state.hitl_pending:
    st.chat_input("Aprueba o rechaza la acción antes de continuar…", disabled=True)
else:
    if prompt := st.chat_input("O escribe tu mensaje aquí…"):
        _send_message(prompt)
