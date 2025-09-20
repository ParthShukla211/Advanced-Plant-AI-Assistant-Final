# page_modules/chat.py
# The main chat interface for the application

import re
import os
import streamlit as st
import time
import numpy as np
from datetime import datetime
from models.model_manager import get_llm_adapter
from rag.retrieval import advanced_retrieve_and_generate
from rag.document_processing import load_pdfs, enhanced_chunk_text
from rag.retrieval import create_embeddings, build_faiss_index
from utils.helpers import _start_generation, _request_interrupt, get_answer_theme, strip_html_tags

def render_answer_with_theme(answer, theme_class, title=None, interrupted=False):
    """ Wraps heading and answer together in a single theme box, optionally appending interruption note """
    if not answer:
        return ""
    html = f'<div class="{theme_class}">'
    if title:
        html += f'<h4 style="margin-bottom:0.5em;">{title}</h4>'
    html += answer
    if interrupted:
        html += "<br><span style='color:#e57373;font-weight:bold;'>(Generation interrupted 🚫)</span>"
    html += '</div>'
    return html

def chat_page():
    """ Renders the main chat interface, including controls, history, and input. """
    st.markdown('<h1 style="text-align: center;">🏭 Advanced Plant AI Assistant </h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-weight: bold;">🧑‍💻 Developed by Parth Shukla ✨</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    st.session_state["interrupt_generation"] = False

    # Get system prompt from sidebar
    if st.session_state["persona_choice"] == "Custom...":
        system_prompt_final = st.session_state["custom_system_prompt"].strip() or "You are a helpful, precise industrial assistant."
    else:
        persona_map = {
            "Default: Precise Industrial Assistant": "You are a helpful,precise industrial assistant for plant operations.",
            "Safety Officer": "You are a plant safety officer. Emphasize safety, compliance, and risk mitigation.",
            "Maintenance Specialist": "You are a senior maintenance engineer. Focus on diagnostics, preventive maintenance, and reliability.",
            "Process Engineer": "You are a process engineer. Optimize parameters, explain control logic, and discuss trade-offs."
        }
        system_prompt_final = persona_map[st.session_state["persona_choice"]]

    # Main Controls (3 dropdowns)
    st.markdown("### 🎛️ Main Controls")
    m1, m2, m3 = st.columns(3)
    with m1:
        from models.adapters import MODEL_FAMILY
        idx = MODEL_FAMILY.index(st.session_state["current_model_kind"]) if st.session_state["current_model_kind"] in MODEL_FAMILY else 0
        model_selected_main = st.selectbox("Model Selection", MODEL_FAMILY, index=idx, key="model_select_main")
        if model_selected_main != st.session_state["current_model_kind"]:
            st.session_state["current_model_kind"] = model_selected_main
    with m2:
        persona_options = ["Default: Precise Industrial Assistant","Safety Officer","Maintenance Specialist","Process Engineer","Custom..."]
        persona_idx = persona_options.index(st.session_state["persona_choice"]) if st.session_state["persona_choice"] in persona_options else 0
        persona_main = st.selectbox("Persona / System Prompt", persona_options, index=persona_idx, key="persona_select_main")
        if persona_main != st.session_state["persona_choice"]:
            st.session_state["persona_choice"] = persona_main
    with m3:
        mode_options = ["Plant Specific (RAG)", "General LLM", "Hybrid"]
        mode_idx = mode_options.index(st.session_state["answer_mode_main"]) if st.session_state["answer_mode_main"] in mode_options else 0
        mode_main = st.selectbox("Answer Mode", mode_options, index=mode_idx, key="answer_mode_select_main")
        if mode_main != st.session_state["answer_mode_main"]:
            st.session_state["answer_mode_main"] = mode_main

    # Chat history
    for m in st.session_state["chat"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"], unsafe_allow_html=True)

    user_question = st.chat_input("Ask your plant engineering question...")

    if not user_question:
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(
                '<div style="text-align: center; color: #ff6b35; font-weight: 600; padding: 9px; background-color: rgb(255 223 172 / 11%); border: 1px solid #ffcc80; border-radius: 7px;">⚠️ App under development. Always verify critical information before any use.</div>',
                unsafe_allow_html=True,
            )

    if user_question:
        st.session_state["chat"].append({"role": "user", "content": user_question})
        st.session_state['run_id'] = time.time()
        with st.chat_message("user"):
            st.markdown(user_question)

        _start_generation()
        with st.chat_message("assistant"):
            thinking_ph = st.empty()
            main_ph = st.empty()
            stop_container_ph = st.empty()
            
            def html_to_log_format(text):
                if not text: return ""
                text = re.sub(r'<h4[^>]*>(.*?)</h4>', r'\n\n**\1**\n\n', text)
                text = re.sub(r'<br>', '\n', text)
                text = re.sub(r'<div class="metrics-wrap">', '\n\n', text)
                text = re.sub(r'<[^>]+>', '', text)
                return text.strip()

            thinking_ph.markdown("🤔 Thinking...")
            
            model_kind, adapter, template = get_llm_adapter()
            plant_answer, general_answer, source_pdfs = "", "", []
            docs_retrieved, start = 0, time.time()
            theme_class = get_answer_theme(st.session_state["answer_mode_main"])
            
            use_rag_now = st.session_state["answer_mode_main"] in ["Plant Specific (RAG)", "Hybrid"]
            if use_rag_now:
                if st.session_state.get("faiss_index") is None:
                    with st.spinner("📚 First-time indexing, please wait..."):
                        pdf_folder = "Database"; 
                        if not os.path.exists(pdf_folder): st.error(f"📁 Folder '{pdf_folder}' not found."); return
                        docs_texts = load_pdfs(pdf_folder)
                        if not docs_texts: st.error("No PDF files found."); return
                        all_chunks = [c for name, doc in docs_texts for c in enhanced_chunk_text(doc, name)]
                        embeddings, embed_model = create_embeddings(all_chunks)
                        faiss_index = build_faiss_index(np.array(embeddings))
                        st.session_state.update({"faiss_index": faiss_index, "embed_model": embed_model, "all_chunks": all_chunks})
                
                faiss_index, embed_model, all_chunks = st.session_state.get("faiss_index"), st.session_state.get("embed_model"), st.session_state.get("all_chunks")

                if model_kind == "ctrans" and st.session_state.get("streaming", True):
                    is_first_token = True
                    anim = ["⏳ Generating.", "⏳ Generating..", "⏳ Generating...", "⏳ Generating...."]; ai = 0
                    buffer = ""
                    
                    def _stream_rag(prompt_text, max_new_tokens=None):
                        nonlocal buffer, ai, is_first_token
                        gen_params = {"temperature": st.session_state.get("temperature", 0.3), "top_p": st.session_state.get("top_p", 0.9), "top_k": st.session_state.get("top_k", 40), "repeat_penalty": st.session_state.get("repeat_penalty", 1.2)}
                        
                        with stop_container_ph.container():
                            st.markdown("<div style='margin-top: 8px;'></div>", unsafe_allow_html=True)
                            if st.button("🚫 Stop Generation", key=f"stop_btn_rag_{st.session_state['run_id']}", use_container_width=True):
                                _request_interrupt()
                                st.rerun()

                        for tok in adapter.generate(template=template, system_prompt=system_prompt_final, user_prompt=prompt_text, max_new_tokens=max_new_tokens, stream=True, **gen_params):
                            if st.session_state.get("interrupt_generation", False): break
                            if is_first_token: 
                                thinking_ph.empty()
                                is_first_token = False
                            buffer += tok
                            content_html = render_answer_with_theme(buffer, theme_class, "📚 Plant Specific Answer")
                            animation_html = f"<br><i>{anim[ai % len(anim)]}</i>"
                            main_ph.markdown(content_html + animation_html, unsafe_allow_html=True)
                            ai += 1
                        return buffer

                    plant_answer, source_pdfs, docs_retrieved = advanced_retrieve_and_generate(query=user_question, embed_model=embed_model, index=faiss_index, chunks=all_chunks, llm_generate_fn=_stream_rag, max_length=st.session_state.get("max_tokens", 512), top_k=st.session_state.get("top_k_retrieval", 12), system_prompt=system_prompt_final)
            
            general_answer = ""
            if st.session_state["answer_mode_main"] in ["General LLM", "Hybrid"] and not st.session_state.get("interrupt_generation", False):
                gp = f"Provide a comprehensive technical answer...\n\nQuestion: {user_question}\n\nAnswer:"
                if model_kind == "ctrans" and st.session_state.get("streaming", True):
                    is_first_token = not plant_answer
                    anim = ["⏳ Generating.", "⏳ Generating..", "⏳ Generating...", "⏳ Generating...."]; ai = 0
                    buffer = ""
                    rag_html = render_answer_with_theme(plant_answer, theme_class, "📚 Plant Specific Answer")
                    gen_params = {"temperature": st.session_state.get("temperature", 0.3), "top_p": st.session_state.get("top_p", 0.9), "top_k": st.session_state.get("top_k", 40), "repeat_penalty": st.session_state.get("repeat_penalty", 1.2)}

                    with stop_container_ph.container():
                        st.markdown("<div style='margin-top: 8px;'></div>", unsafe_allow_html=True)
                        if st.button("🚫 Stop Generation", key=f"stop_btn_gen_{st.session_state['run_id']}", use_container_width=True):
                            _request_interrupt()
                            st.rerun()

                    for tok in adapter.generate(template=template, system_prompt=system_prompt_final, user_prompt=gp, max_new_tokens=st.session_state.get("max_tokens", 512), stream=True, **gen_params):
                        if st.session_state.get("interrupt_generation", False): break
                        if is_first_token: 
                            thinking_ph.empty()
                            is_first_token = False
                        buffer += tok
                        general_html = render_answer_with_theme(buffer, theme_class, "🌍 General LLM Answer")
                        combined_html = f"{rag_html}<br>{general_html}" if rag_html else general_html
                        animation_html = f"<br><i>{anim[ai % len(anim)]}</i>"
                        main_ph.markdown(combined_html + animation_html, unsafe_allow_html=True)
                        ai += 1
                    general_answer = buffer

            if not st.session_state.get("interrupt_generation", False):
                thinking_ph.empty()
                stop_container_ph.empty()
                rt = round(time.time() - start, 2)

                final_html = ""
                if plant_answer: final_html += render_answer_with_theme(plant_answer, theme_class, "📚 Plant Specific Answer")
                if general_answer:
                    if final_html: final_html += "<br>"
                    final_html += render_answer_with_theme(general_answer, theme_class, "🌍 General LLM Answer")
                
                if not final_html: final_html = "No output was generated."

                metrics_html = f'<div class="metrics-wrap"><span class="badge">⏱️ Response generated in {rt}s</span>'
                if source_pdfs: metrics_html += f'<span class="badge">📂 Sources: {", ".join(source_pdfs)}</span>'
                metrics_html += '</div>'
                
                full_content = final_html + metrics_html

                main_ph.markdown(full_content, unsafe_allow_html=True)
                st.session_state["chat"].append({"role": "assistant", "content": full_content})

                st.session_state["logs"].append({
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "model": st.session_state["current_model_kind"],
                    "mode": st.session_state["answer_mode_main"], "q": user_question, "a": html_to_log_format(full_content),
                    "sources": list(set(source_pdfs)) if source_pdfs else [], "docs_retrieved": docs_retrieved, "response_time": rt,
                })
                an = st.session_state["analytics"]
                an["total_queries"] += 1
                an["avg_response_time"] = ((an["avg_response_time"] * (an["total_queries"] - 1) + rt) / an["total_queries"])
                an["model_usage"][st.session_state["current_model_kind"]] = an["model_usage"].get(st.session_state["current_model_kind"], 0) + 1
                if use_rag_now: an["retrieval_stats"].append({"docs_retrieved": docs_retrieved, "response_time": rt})
        
        # <<< MODIFIED: Trigger a rerun after the generation is complete
        st.rerun()