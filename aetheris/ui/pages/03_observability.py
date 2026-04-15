"""Página de observabilidad — trazas de LangSmith y métricas."""
import os
import sys
from pathlib import Path

# Garantiza que la raíz del proyecto esté en sys.path
_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

import requests
import streamlit as st

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

st.title("Observabilidad")
st.caption("Trazas de LangSmith, métricas de coste y estado del agente.")

# ---------------------------------------------------------------------------
# Estado del sistema
# ---------------------------------------------------------------------------
st.subheader("Estado del sistema")
col1, col2 = st.columns(2)

with col1:
    try:
        resp = requests.get(f"{API_BASE}/api/v1/health", timeout=5)
        health = resp.json()
        status_color = "green" if health["status"] == "ok" else "red"
        st.markdown(f"**Estado de la API:** :{status_color}[{health['status'].upper()}]")
        st.markdown(f"**Chroma:** {'OK' if health['chroma_ok'] else 'ERROR'}")
        st.markdown(f"**SQLite:** {'OK' if health['sqlite_ok'] else 'ERROR'}")
        st.markdown(f"**Entorno:** {health['app_env']}")
    except Exception as exc:
        st.error(f"No se puede alcanzar la API: {exc}")

with col2:
    try:
        resp = requests.get(f"{API_BASE}/api/v1/health/langsmith", timeout=5)
        ls = resp.json()
        ls_color = "green" if ls["langsmith_connected"] else "red"
        st.markdown(f"**LangSmith:** :{ls_color}[{'Conectado' if ls['langsmith_connected'] else 'Desconectado'}]")
        st.markdown(f"**Proyecto:** {ls['project_name']}")
        if ls.get("error"):
            st.caption(f"Error: {ls['error']}")
    except Exception:
        st.markdown("**LangSmith:** :gray[Desconocido]")

# ---------------------------------------------------------------------------
# Ejecuciones recientes
# ---------------------------------------------------------------------------
st.subheader("Ejecuciones recientes del agente")

if st.button("Actualizar trazas"):
    st.rerun()

try:
    from aetheris.observability.tracing import get_recent_runs
    runs = get_recent_runs(limit=20)
    if not runs:
        st.info("No se han encontrado trazas. Asegúrate de que LangSmith está configurado y has realizado alguna consulta.")
    else:
        for run in runs:
            with st.expander(f"{run['name']} — {run['status']} ({run['start_time'][:19]})", expanded=False):
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Estado", run["status"])
                with col_b:
                    st.metric("Tokens", run.get("total_tokens") or "—")
                with col_c:
                    cost = run.get("total_cost")
                    st.metric("Coste", f"${cost:.4f}" if cost else "—")
                st.caption(f"ID de ejecución: {run['id']}")
except Exception as exc:
    st.warning(f"No se han podido cargar las trazas: {exc}")

# ---------------------------------------------------------------------------
# Enlace al panel de LangSmith
# ---------------------------------------------------------------------------
st.divider()
st.markdown("Abre el panel completo de LangSmith para inspección detallada de trazas y evaluación.")
st.link_button("Abrir panel de LangSmith", "https://smith.langchain.com")
