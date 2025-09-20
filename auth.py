# auth.py
# Authentication functionality

import streamlit as st

ACCESS_TOKEN = "Parth123@#"

def safe_rerun():
    """Use st.rerun() if available; otherwise ignore (older builds)."""
    try:
        st.rerun()
    except Exception:
        pass

def auth_gate() -> bool:

    st.set_page_config(
        page_title="🔐 Auth | Advanced Plant AI Assistant",
        page_icon="🏭",   # you can use any emoji or path to an image
    
    )
    """
    Shows a compact login card until a valid token is entered.
    Returns True if authenticated, False otherwise.
    """
    if "authed" not in st.session_state:
        st.session_state.authed = False

    if st.session_state.authed:
        return True

    # Minimal hint in sidebar
    with st.sidebar:
        st.markdown("### 🔐 Access Required")
        st.info("Enter security token to access the app.")

    # Header (page title uses industry emoji)
    st.markdown('<h1 class="main-header">🏭 Advanced Plant AI Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">🧑‍💻 Developed by <strong>Parth Shukla</strong> ✨</div>', unsafe_allow_html=True)

    # Login card
    st.markdown('<div class="login-card">', unsafe_allow_html=True)
    st.markdown('<div class="login-title">🔐 Secure Access</div>', unsafe_allow_html=True)
    st.markdown('<div class="login-sub">Please enter your access token to continue</div>', unsafe_allow_html=True)

    token = st.text_input(
        "Access Token",
        type="password",
        placeholder="Enter token (case sensitive)",
        label_visibility="collapsed"
    )
    login = st.button("Enter", type="primary", use_container_width=True)
    if login:
        if token == ACCESS_TOKEN:
            st.session_state.authed = True
            st.success("Access granted ✅")
            safe_rerun()
        else:
            st.error("Invalid token. Access denied.")

    st.markdown("</div>", unsafe_allow_html=True)
    return False
