# utils/helpers.py
# General helper functions

import os
import streamlit as st
import time

def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def initialize_session_state():
    defaults = {
        "logs": [],
        "analytics": {"total_queries":0, "avg_response_time":0.0, "retrieval_stats":[], "model_usage":{}},
        "faiss_index": None,
        "embed_model": None,
        "all_chunks": [],
        "chat": [],
        "current_model_kind": "🟡 Local Mistral (GGUF)",
        "persona_choice": "Default: Precise Industrial Assistant",
        "custom_system_prompt": "",
        "answer_mode_main": "Plant Specific (RAG)",
        # generation control
        "is_generating": False,
        "interrupt_generation": False,
        "gen_run_id": None,
        "presets": {}
    }
    for k,v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def _start_generation():
    # mark a new run id; used to visually simulate interrupts
    st.session_state["is_generating"] = True
    st.session_state["interrupt_generation"] = False
    st.session_state["gen_run_id"] = str(time.time())

def _request_interrupt():
    # set interrupt flag and rerun; old run will continue server-side, but UI reruns and stops appending
    st.session_state["interrupt_generation"] = True


def get_answer_theme(mode):
    """Return the appropriate CSS class based on answer mode"""
    theme_map = {
        "Plant Specific (RAG)": "answer-box-plant",
        "General LLM": "answer-box-general", 
        "Hybrid": "answer-box-hybrid"
    }
    return theme_map.get(mode, "answer-box")

import re

def strip_html_tags(text):
    """
    Remove all HTML tags from text while preserving the content
    Example: <div class="test">Hello</div> becomes "Hello"
    """
    if not text:
        return text
    # Remove HTML tags but keep the content
    clean_text = re.sub('<.*?>', '', text)
    # Replace HTML entities
    clean_text = clean_text.replace('&nbsp;', ' ').replace('&amp;', '&')
    return clean_text.strip()