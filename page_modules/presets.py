# pages/presets.py
# Presets page

import streamlit as st
import json

def page_presets():
    st.header("🎚️ Presets (Save / Apply)")
    st.markdown("---")
    # snapshot of current settings
    current = {
        "model": st.session_state["current_model_kind"],
        "persona": st.session_state["persona_choice"],
        "custom_system_prompt": st.session_state.get("custom_system_prompt", ""),
        "answer_mode": st.session_state["answer_mode_main"],
        "params": {
            "max_tokens": st.session_state.get("max_tokens", 512),
            "temperature": st.session_state.get("temperature", 0.3),
            "top_p": st.session_state.get("top_p", 0.9),
            "top_k": st.session_state.get("top_k", 40),
            "repeat_penalty": st.session_state.get("repeat_penalty", 1.2),
            "mirostat_mode": st.session_state.get("mirostat_mode", "Off"),
            "mirostat_tau": st.session_state.get("mirostat_tau", 5.0),
            "mirostat_eta": st.session_state.get("mirostat_eta", 0.1),
            "seed": st.session_state.get("seed_val", -1),
            "stop": st.session_state.get("stop_seqs_raw", ""),
            "n_ctx": st.session_state.get("n_ctx", 4096),
            "gpu_layers": st.session_state.get("gpu_layers", 0),
            "retrieval": {
                "use_rag": st.session_state.get("use_rag", True),
                "top_k_retrieval": st.session_state.get("top_k_retrieval", 12),
                "chunk_size": st.session_state.get("chunk_size", 400),
                "chunk_overlap": st.session_state.get("chunk_overlap", 50),
                "alpha_hybrid": st.session_state.get("alpha_hybrid", 0.7),
                "compression_budget": st.session_state.get("compression_budget", 2200),
                "use_hybrid_search": st.session_state.get("use_hybrid_search", True),
                "use_reranking": st.session_state.get("use_reranking", True),
                "use_compression": st.session_state.get("use_compression", True),
                "streaming": st.session_state.get("streaming", True),
            },
        },
    }

    name = st.text_input("Preset name")
    if st.button("💾 Save Preset", disabled=not name):
        st.session_state["presets"][name] = current
        st.success(f"Saved preset: {name}")

    if st.session_state["presets"]:
        st.markdown("#### Saved Presets")
        options = list(st.session_state["presets"].keys())
        sel = st.selectbox("Choose", options)
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("▶️ Apply", use_container_width=True):
                p = st.session_state["presets"][sel]
                st.session_state["current_model_kind"] = p["model"]
                st.session_state["persona_choice"] = p["persona"]
                st.session_state["custom_system_prompt"] = p.get("custom_system_prompt", "")
                st.session_state["answer_mode_main"] = p.get("answer_mode", "Plant Specific (RAG)")
                st.success(f"Applied preset: {sel}")
        with col2:
            if st.button("🗑️ Delete", use_container_width=True):
                del st.session_state["presets"][sel]
                st.success("Preset deleted.")
                try:
                    st.rerun()
                except Exception:
                    pass
        with col3:
            if st.button("⬇️ Export JSON", use_container_width=True):
                data = json.dumps(st.session_state["presets"][sel], indent=2)
                st.download_button(
                    "Download",
                    data=data,
                    file_name=f"preset_{sel}.json",
                    mime="application/json",
                )