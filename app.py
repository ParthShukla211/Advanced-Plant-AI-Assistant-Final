# app.py
# -*- coding: utf-8 -*-
import os
import sys
import streamlit as st

# --- Path Setup ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from auth import auth_gate
from styles import apply_custom_styles
from utils.nltk_setup import ensure_nltk_data
from utils.helpers import initialize_session_state

# Import page modules
from page_modules.home import home_page
from page_modules.chat import chat_page
from page_modules.models import page_models
from page_modules.index import page_index
from page_modules.database_analysis import page_database_analysis
from page_modules.presets import page_presets
from page_modules.logs import page_logs
from page_modules.analytics import page_analytics
from page_modules.exports import page_exports
from page_modules.developer_tools import page_developer_tools
from page_modules.query_analyzer import page_query_analyzer
from page_modules.rag_inspector import page_rag_inspector

# --- Initialization ---
ensure_nltk_data()
apply_custom_styles()
if not auth_gate(): st.stop()
initialize_session_state()
st.set_page_config(
    page_title="Advanced Plant AI Assistant",
    page_icon="🏭"
)

# <<< MODIFIED: Centralized navigation state management
# 1. Initialize our single source of truth for the current page.
if "current_page" not in st.session_state:
    st.session_state.current_page = "Home"

# 2. Define the list of pages and a callback function.
page_options = ["Home", "Chat", "Index",  "Models", "RAG Inspector", "Query Analyzer", "Logs", "Database Analysis", "Presets",  "Analytics", "Exports", "Developer Tools"]

def set_page_from_selectbox():
    """Callback function to update the current_page from the selectbox."""
    st.session_state.current_page = st.session_state.nav_selectbox

# --- Main App Layout ---
with st.sidebar:
    st.title("⚙️ Controls")
    
    # 3. Get the current page's index for the selectbox.
    try:
        current_page_index = page_options.index(st.session_state.current_page)
    except ValueError:
        current_page_index = 0 # Default to Home if page is not in the list

    # 4. The selectbox now uses a key and the on_change callback.
    st.selectbox(
        "📂 Menu",
        page_options,
        index=current_page_index,
        key="nav_selectbox",
        on_change=set_page_from_selectbox
    )
    
    st.markdown("---")
    from page_modules.sidebar import render_sidebar
    render_sidebar()

# --- Page Routing ---
page_router = {
    "Home": home_page,
    "Chat": chat_page,
    "Index": page_index,
    "RAG Inspector": page_rag_inspector,
    "Query Analyzer": page_query_analyzer,
    "Database Analysis": page_database_analysis,
    "Models": page_models,
    "Presets": page_presets,
    "Logs": page_logs,
    "Analytics": page_analytics,
    "Exports": page_exports,
    "Developer Tools": page_developer_tools,
}

# 5. The router now reads from our single source of truth.
page_to_show = st.session_state.current_page
if page_to_show in page_router:
    page_router[page_to_show]()
else:
    # Fallback to home page if something goes wrong
    page_router["Home"]()