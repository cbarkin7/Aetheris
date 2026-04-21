"""
AETHERIS Streamlit entry point.
Run with: streamlit run aetheris/ui/app.py
"""
import sys
import uuid
from pathlib import Path

# Ensure the project root (Aetheris/) is in sys.path so the 'aetheris'
# package is importable both in app.py and in all pages run by st.navigation.
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import streamlit as st

st.set_page_config(
    page_title="AETHERIS",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize global session state
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "user_id" not in st.session_state:
    st.session_state.user_id = "Admin-Aetheris"

# Navigation
pages = {
    "Chat": [st.Page("pages/01_chat.py", title="Chat", icon="💬")],
    "Knowledge": [st.Page("pages/02_documents.py", title="Documents", icon="📄")],
    "Monitoring": [st.Page("pages/03_observability.py", title="Observability", icon="📊")],
}

pg = st.navigation(pages)
pg.run()
