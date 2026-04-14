"""Página de carga y gestión de documentos."""
import os

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

if uploaded and st.button("Ingestar documento", type="primary"):
    with st.spinner("Ingiriendo…"):
        resp = requests.post(
            f"{API_BASE}/api/v1/documents/upload",
            files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
            data={"user_id": st.session_state.get("user_id", "default")},
            timeout=120,
        )
    if resp.ok:
        result = resp.json()
        st.success(f"Ingestado **{result['filename']}** — {result['n_chunks']} fragmentos")
    else:
        st.error(f"Error en la subida: {resp.text}")

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
