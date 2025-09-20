# styles.py
# UI styles and configuration

import streamlit as st

def apply_custom_styles():
    """Apply custom CSS styles to the application"""
    st.set_page_config(
        page_title="Plant AI Assistant",
        page_icon="🏭",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown("""
    <style>
    /* Headings and general layout */
    .main-header {
      background: linear-gradient(90deg, #2E86AB, #A23B72, #F18F01);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      font-size: 2.5rem;
      font-weight: 900;
      text-align: center;
      margin: 0.25rem 0 0.25rem 0;
    }
    .stSelectbox:hover,
    .stSelectbox div[data-baseweb="select"]:hover,
    .stSelectbox label:hover,
    .stSelectbox div[data-baseweb="select"] div:hover {
        cursor: pointer !important;
    }
    /* Specific fix for Streamlit select boxes */
    div[data-baseweb="select"] input {
        caret-color: transparent !important;
    }
    .sub-header {
      text-align:center; color:#444; font-weight:600; margin-bottom: 0.8rem;
    }

    /* --- THEME-AWARE ANSWER BOXES (FINAL FIX) --- */
    
    /* This rule applies to all answer boxes */
    .answer-box-plant, .answer-box-general, .answer-box-hybrid, .answer-box {
        padding: 16px; 
        border-radius: 10px; 
        margin: 8px 0;
        /* Use Streamlit's theme variables for perfect theme adaptation */
        background-color: var(--secondary-background-color);
        color: var(--text-color);
    }
    .badge{
        background-color: var(--secondary-background-color) !important;
        color: var(--text-color) !important;
        }
    
    /* We still use the colored left border to differentiate the answer types */
    .answer-box-plant { 
      border-left: 5px solid #2E86AB; 
    }
    .answer-box-general { 
      border-left: 5px solid #A23B72; 
    }
    .answer-box-hybrid { 
      border-left: 5px solid #2E8B57; 
    }
    .answer-box { 
      border-left: 5px solid #6c757d; 
    }
    
    /* Other styles remain unchanged */
    .metrics-wrap { margin: 12px 0 16px 0; }
    .badge { display:inline-block; padding:4px 10px; background:#EEF2FF; color:#334155; border:1px solid #CBD5E1; border-radius:999px; font-size:0.85rem; margin: 4px 10px 0 0; }
    .login-card { max-width: 560px; margin: 6vh auto 0 auto; padding: 22px 26px; }
    .login-title { text-align:center; font-weight:800; font-size:1.6rem; margin-bottom: 6px; }
    .login-sub  { text-align:center; color:#555; margin-bottom: 12px; }
    .sidebar-bottom { margin-top: 18px; padding-top: 12px; border-top: 1px solid #E5E7EB; }
    </style>
    """, unsafe_allow_html=True)