"""Reusable chat message bubble component."""
import streamlit as st


def render_message(role: str, content: str, tool_calls: list | None = None) -> None:
    """Render a single chat message with role-appropriate styling."""
    with st.chat_message(role):
        st.markdown(content)
        if tool_calls:
            with st.expander("Tool calls", expanded=False):
                for tc in tool_calls:
                    st.json(tc)
