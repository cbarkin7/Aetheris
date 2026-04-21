"""Componente de confirmación Human-in-the-Loop para acciones de Google Workspace."""
import streamlit as st

# CSS inyectado una sola vez: tiñe el botón "Aprobar" de verde y el "Rechazar" de rojo
# apagado para que el contraste visual sea claro pero no agresivo.
_HITL_CSS = """
<style>
/* ── Botón Aprobar ── verde Bootstrap 5 */
div[data-testid="stHorizontalBlock"]
    div[data-testid="column"]:first-child
    button[kind="primary"] {
    background-color: #198754 !important;
    border-color:     #198754 !important;
    color:            #ffffff !important;
}
div[data-testid="stHorizontalBlock"]
    div[data-testid="column"]:first-child
    button[kind="primary"]:hover {
    background-color: #157347 !important;
    border-color:     #146c43 !important;
}

/* ── Botón Rechazar ── rojo apagado (más suave que el primary rojo por defecto) */
div[data-testid="stHorizontalBlock"]
    div[data-testid="column"]:last-child
    button[kind="secondary"] {
    background-color: transparent   !important;
    border-color:     #dc3545       !important;
    color:            #dc3545       !important;
}
div[data-testid="stHorizontalBlock"]
    div[data-testid="column"]:last-child
    button[kind="secondary"]:hover {
    background-color: #dc354520     !important;
}
</style>
"""


def render_hitl_modal(actions: list[dict]) -> bool | None:
    """
    Muestra un diálogo de confirmación para las acciones pendientes de Google Workspace.

    Cada acción incluye el campo `description` generado por el LLM para que el
    usuario entienda exactamente qué se va a ejecutar en su nombre.

    Returns:
        True  → usuario aprobó
        False → usuario rechazó
        None  → aún no ha decidido
    """
    # Inyectar estilos de botones (idempotente: Streamlit deduplicará el bloque)
    st.markdown(_HITL_CSS, unsafe_allow_html=True)

    st.warning("⚠️ **AETHERIS quiere realizar las siguientes acciones en tu nombre:**")

    for i, action in enumerate(actions, start=1):
        name = action.get("name", "acción desconocida")
        description = action.get("description", "")
        args = action.get("args", {})

        with st.container(border=True):
            st.markdown(f"**Acción {i}:** `{name}`")
            if description:
                st.markdown(f"📋 {description}")
            with st.expander("Ver parámetros detallados"):
                st.json(args)

    st.divider()
    col_ok, col_ko = st.columns(2)
    approved = None

    with col_ok:
        if st.button("✅ Aprobar", type="primary", key="hitl_approve", use_container_width=True):
            approved = True
    with col_ko:
        if st.button("❌ Rechazar", type="secondary", key="hitl_reject", use_container_width=True):
            approved = False

    return approved
