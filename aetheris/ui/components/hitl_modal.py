"""Human-in-the-Loop approval dialog component."""
import streamlit as st


def render_hitl_modal(actions: list[dict]) -> bool | None:
    """
    Render a confirmation dialog for pending Google Workspace actions.
    Returns True (approved), False (rejected), or None (not yet decided).
    """
    st.warning("AETHERIS wants to perform the following action on your behalf:")

    for action in actions:
        name = action.get("name", "unknown action")
        args = action.get("args", {})
        st.markdown(f"**Action:** `{name}`")
        st.json(args)

    col1, col2 = st.columns(2)
    approved = None
    with col1:
        if st.button("Approve", type="primary", key="hitl_approve"):
            approved = True
    with col2:
        if st.button("Reject", type="secondary", key="hitl_reject"):
            approved = False

    return approved
