"""Página de carga y gestión de documentos."""
import os
import sys
from pathlib import Path

# Garantiza que la raíz del proyecto esté en sys.path
_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

import requests
import streamlit as st

from aetheris.ui.components.document_card import render_document_card

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

st.title("Biblioteca de documentos")
st.caption("Sube documentos para que AETHERIS los utilice como base de conocimiento.")

# ---------------------------------------------------------------------------
# Subida
# ---------------------------------------------------------------------------
st.subheader("Subir documento")
uploaded = st.file_uploader(
    "Selecciona un fichero",
    type=["pdf", "docx", "txt", "md"],
    accept_multiple_files=False,
)


def _do_upload(force: bool = False) -> None:
    """Envía el fichero al backend. Si force=True indica reingestión."""
    if not uploaded:
        return
    with st.spinner("Ingiriendo…"):
        resp = requests.post(
            f"{API_BASE}/api/v1/documents/upload",
            files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
            data={
                "user_id": st.session_state.get("user_id", "default"),
                "force": "true" if force else "false",
            },
            timeout=120,
        )

    if resp.status_code == 201:
        result = resp.json()
        st.success(
            f"✅ **{result['filename']}** ingestado correctamente — "
            f"{result['n_chunks']} fragmentos"
        )
        # Limpiar estado de conflicto si lo hubiera
        st.session_state.pop("_upload_conflict", None)
        st.rerun()

    elif resp.status_code == 409:
        # El documento ya existe: guardar info del conflicto en session_state
        conflict = resp.json().get("detail", {})
        st.session_state["_upload_conflict"] = conflict

    else:
        st.error(f"Error en la subida: {resp.text}")


# Botón principal de ingestión
if uploaded:
    if st.button("Ingestar documento", type="primary"):
        _do_upload(force=False)

# ---------------------------------------------------------------------------
# Gestión del conflicto 409 — confirmación de actualización
# ---------------------------------------------------------------------------
conflict = st.session_state.get("_upload_conflict")
if conflict and uploaded:
    ingested_str = conflict.get("ingested_at", "fecha desconocida")
    # Formatear fecha si viene en ISO
    try:
        from datetime import datetime
        dt = datetime.fromisoformat(ingested_str.replace("Z", "+00:00"))
        ingested_str = dt.strftime("%d/%m/%Y %H:%M UTC")
    except Exception:
        pass

    st.warning(
        f"⚠️ **{conflict.get('filename', uploaded.name)}** ya está indexado "
        f"({conflict.get('n_chunks', '?')} fragmentos, ingestado el {ingested_str}).\n\n"
        "¿Quieres reemplazar los datos obsoletos con la versión actual del fichero?"
    )
    col_yes, col_no = st.columns(2)
    with col_yes:
        if st.button("✅ Sí, actualizar", type="primary", key="confirm_update"):
            st.session_state.pop("_upload_conflict", None)
            _do_upload(force=True)
    with col_no:
        if st.button("❌ No, cancelar", key="cancel_update"):
            st.session_state.pop("_upload_conflict", None)
            st.rerun()

# ---------------------------------------------------------------------------
# Lista de documentos
# ---------------------------------------------------------------------------
st.subheader("Documentos indexados")

try:
    resp = requests.get(f"{API_BASE}/api/v1/documents", timeout=10)
    docs = resp.json() if resp.ok else []
except Exception:
    docs = []
    st.warning("No se ha podido conectar con la API")

if not docs:
    st.info("Aún no hay documentos indexados. Sube uno arriba.")
else:
    def delete_doc(doc_id: str) -> None:
        r = requests.delete(f"{API_BASE}/api/v1/documents/{doc_id}", timeout=10)
        if r.ok:
            st.success("Documento eliminado")
            st.rerun()
        else:
            st.error(f"Error al eliminar: {r.text}")

    for doc in docs:
        render_document_card(doc, on_delete=delete_doc)
        st.divider()
