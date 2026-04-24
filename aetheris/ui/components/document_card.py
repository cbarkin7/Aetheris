"""Document list card component with metadata and delete action."""
import streamlit as st


def render_document_card(doc: dict, on_delete=None) -> None:
    """Render a single document entry with filename, ingestion date, chunk count and delete button."""
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown(f"**{doc.get('filename', 'Unknown')}**")
        meta_parts = [f"ID: `{doc.get('document_id', '')[:16]}…`"]
        if doc.get("n_chunks"):
            meta_parts.append(f"{doc['n_chunks']} fragmentos")
        if doc.get("ingested_at"):
            # Accept both ISO string and datetime object
            ingested = doc["ingested_at"]
            if hasattr(ingested, "strftime"):
                ingested_str = ingested.strftime("%d/%m/%Y %H:%M UTC")
            else:
                try:
                    from datetime import datetime, timezone
                    dt = datetime.fromisoformat(str(ingested))
                    ingested_str = dt.strftime("%d/%m/%Y %H:%M UTC")
                except ValueError:
                    ingested_str = str(ingested)
            meta_parts.append(f"Ingestado: {ingested_str}")
        st.caption("  ·  ".join(meta_parts))
    with col2:
        if on_delete and st.button("Eliminar", key=f"del_{doc.get('document_id', '')}", type="secondary"):
            on_delete(doc.get("document_id"))
