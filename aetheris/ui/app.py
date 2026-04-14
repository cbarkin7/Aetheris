"""
AETHERIS Streamlit entry point.
Run with: streamlit run aetheris/ui/app.py
"""
import uuid
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
    st.session_state.user_id = "default"

# Navigation
pages = {
    "Chat": [st.Page("pages/01_chat.py", title="Chat", icon="💬")],
    "Knowledge": [st.Page("pages/02_documents.py", title="Documents", icon="📄")],
    "Monitoring": [st.Page("pages/03_observability.py", title="Observability", icon="📊")],
}

pg = st.navigation(pages)
pg.run()
