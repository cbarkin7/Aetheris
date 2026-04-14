"""Document list card component with metadata and delete action."""
import streamlit as st


def render_document_card(doc: dict, on_delete=None) -> None:
    """Render a single document entry with filename, ID, and delete button."""
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown(f"**{doc.get('filename', 'Unknown')}**")
        st.caption(f"ID: {doc.get('document_id', '')[:16]}…")
    with col2:
        if on_delete and st.button("Delete", key=f"del_{doc.get('document_id', '')}", type="secondary"):
            on_delete(doc.get("document_id"))
