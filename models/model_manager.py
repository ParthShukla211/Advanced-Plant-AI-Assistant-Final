# models/model_manager.py
# Model management functions

import streamlit as st
from models.adapters import CTRANS_CHOICES, HF_T2T_CHOICES, MODEL_FAMILY
from models.adapters import CTransChatModel, get_hf_t2t_pipeline, hf_t2t_generate

def get_llm_adapter():
    kind = st.session_state["current_model_kind"]
    stop_sequences = [
        s.strip()
        for s in (
            st.session_state.get("stop_seqs_raw", "").split(",")
            if st.session_state.get("stop_seqs_raw")
            else []
        )
        if s.strip()
    ]
    seed_param = None if st.session_state.get("seed_val", -1) == -1 else int(st.session_state["seed_val"])

    if kind in CTRANS_CHOICES:
        spec = CTRANS_CHOICES[kind]
        adapter = CTransChatModel(
            spec,
            context_length=st.session_state.get("n_ctx", 4096),
            gpu_layers=st.session_state.get("gpu_layers", 0),
            seed=seed_param,
            mirostat_mode=st.session_state.get("mirostat_mode_val", 0),
            mirostat_tau=st.session_state.get("mirostat_tau", 5.0),
            mirostat_eta=st.session_state.get("mirostat_eta", 0.1),
            stop_sequences=stop_sequences,
        )
        template = spec["template"]
        return ("ctrans", adapter, template)
    else:
        model_id = HF_T2T_CHOICES[kind]
        pipe = get_hf_t2t_pipeline(model_id)
        return ("hf_t2t", pipe, None)