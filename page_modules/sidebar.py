# pages/sidebar.py
# Sidebar rendering functions

import streamlit as st
from models.adapters import MODEL_FAMILY

def render_sidebar():
    """Render the sidebar controls"""
    # ---- Model selection (sidebar) ----
    st.markdown("**🧠 Model**")
    sb_model_idx = MODEL_FAMILY.index(st.session_state["current_model_kind"]) if st.session_state["current_model_kind"] in MODEL_FAMILY else 0
    sb_model_choice = st.selectbox("Language Model", MODEL_FAMILY, index=sb_model_idx, key="model_select_sidebar")
    if sb_model_choice != st.session_state["current_model_kind"]:
        st.session_state["current_model_kind"] = sb_model_choice

    # ---- Persona / system prompt (sidebar) ----
    st.markdown("**🧭 Persona / System Prompt**")
    persona_options = [
        "Default: Precise Industrial Assistant",
        "Safety Officer",
        "Maintenance Specialist",
        "Process Engineer",
        "Custom..."
    ]
    persona_map = {
        "Default: Precise Industrial Assistant": "You are a helpful, precise industrial assistant for plant operations.",
        "Safety Officer": "You are a plant safety officer. Emphasize safety, compliance, and risk mitigation.",
        "Maintenance Specialist": "You are a senior maintenance engineer. Focus on diagnostics, preventive maintenance, and reliability.",
        "Process Engineer": "You are a process engineer. Optimize parameters, explain control logic, and discuss trade-offs."
    }
    sb_persona_idx = persona_options.index(st.session_state["persona_choice"]) if st.session_state["persona_choice"] in persona_options else 0
    sb_persona = st.selectbox("Persona", persona_options, index=sb_persona_idx, key="persona_select_sidebar")
    st.session_state["persona_choice"] = sb_persona
    if sb_persona == "Custom...":
        st.session_state["custom_system_prompt"] = st.text_area("Custom system prompt", value=st.session_state.get("custom_system_prompt", ""), height=90, placeholder="Write your system instructions...")
        system_prompt_final = st.session_state["custom_system_prompt"].strip() or "You are a helpful, precise industrial assistant."
    else:
        st.session_state["custom_system_prompt"] = ""
        system_prompt_final = persona_map[sb_persona]
    st.markdown("**🎯 Answer Mode**")
    answer_mode_options = [
        "Plant Specific (RAG)",
        "General LLM", 
        "Hybrid"
    ]
    answer_mode_map = {
        "Plant Specific (RAG)": "Uses only your uploaded plant documentation and RAG retrieval for answers.",
        "General LLM": "Uses the language model's general knowledge without document retrieval.",
        "Hybrid": "Combines both plant-specific documentation and general knowledge for comprehensive answers."
    }
    sb_answer_mode_idx = answer_mode_options.index(st.session_state["answer_mode_main"]) if st.session_state["answer_mode_main"] in answer_mode_options else 0
    sb_answer_mode = st.selectbox("Answer Mode", answer_mode_options, index=sb_answer_mode_idx, key="answer_mode_select_sidebar")
    st.session_state["answer_mode_main"] = sb_answer_mode

    st.markdown("---")
    # ---- RAG toggles (sidebar) ----
    st.markdown("**🔎 Retrieval (RAG)**")
    use_rag = st.checkbox("Enable RAG", value=(st.session_state["answer_mode_main"] in ["Plant Specific (RAG)", "Hybrid"]), key="use_rag")
    use_hybrid_search = st.checkbox("Hybrid Search (Semantic + Keyword)", value=True)
    use_reranking = st.checkbox("Document Reranking", value=True)
    use_compression = st.checkbox("Context Compression", value=True)
    top_k_retrieval = st.slider("Documents to retrieve (Top-K)", 5, 30, 12)
    chunk_size = st.slider("Chunk size (words)", 200, 800, 400, 50)
    chunk_overlap = st.slider("Chunk overlap", 0, 150, 50, 10)
    alpha_hybrid = st.slider("Semantic vs Keyword balance", 0.0, 1.0, 0.7, 0.05)
    compression_budget = st.slider("Compression context budget (chars)", 500, 8000, 2200, 100)
    st.markdown("---")
    # ---- Generation/Runtime (sidebar) ----
    st.markdown("**📝 Generation & Runtime**")
    streaming = st.checkbox("Stream tokens", value=True, key="streaming")
    max_tokens = st.number_input("Max new tokens", 64, 4096, 512, 32, key="max_tokens")
    temperature = st.slider("Temperature", 0.0, 1.5, 0.3, 0.05, key="temperature")
    top_p = st.slider("Top-p", 0.05, 1.0, 0.9, 0.05, key="top_p")
    top_k = st.slider("Top-k", 1, 200, 40, 1, key="top_k")
    repeat_penalty = st.slider("Repetition penalty", 1.0, 2.0, 1.2, 0.05, key="repeat_penalty")
    st.markdown("---")
    st.markdown("**🧪 Advanced Sampling**")
    mirostat_mode = st.selectbox("Mirostat", ["Off", "Mode 1", "Mode 2"], index=0, key="mirostat_mode")
    mirostat_mode_val = {"Off": 0, "Mode 1": 1, "Mode 2": 2}[mirostat_mode]
    st.session_state["mirostat_mode_val"] = mirostat_mode_val
    col_tau, col_eta = st.columns(2)
    with col_tau:
        mirostat_tau = st.slider("tau", 1.0, 10.0, 5.0, 0.1, key="mirostat_tau")
    with col_eta:
        mirostat_eta = st.slider("eta", 0.01, 1.0, 0.1, 0.01, key="mirostat_eta")
    seed_val = st.number_input("Seed (optional, -1=unset)", -1, 2**31-1, -1, 1, key="seed_val")
    stop_seqs_raw = st.text_input("Stop sequences (comma-separated)", value="", key="stop_seqs_raw")
    st.markdown("---")
    # ---- Local model runtime ----
    st.markdown("**🧩 Local GGUF Runtime (ctransformers)**")
    n_ctx = st.slider("Context window", 1024, 8192, 4096, 512, key="n_ctx")
    gpu_layers = st.slider("GPU layers (0=CPU)", 0, 80, 0, key="gpu_layers")

    # ---- Bottom action buttons (vertical, last) ----
    st.markdown('<div class="sidebar-bottom"></div>', unsafe_allow_html=True)
    if st.button("🧹 Clear Chat History", use_container_width=True, key="btn_clear_chat_history"):
        st.session_state["chat"] = []
        st.success("Chat history cleared.")
    if st.button("🚪 Logout", use_container_width=True, key="btn_logout"):
        st.session_state.authed = False
        try:
            st.rerun()
        except Exception:
            pass

    # Return the final system prompt
    if st.session_state["persona_choice"] == "Custom...":
        return st.session_state["custom_system_prompt"].strip() or "You are a helpful, precise industrial assistant."
    else:
        return persona_map[st.session_state["persona_choice"]]