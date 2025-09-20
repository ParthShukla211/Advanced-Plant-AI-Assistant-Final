# page_modules/home.py
# The landing page for the application with full content.

import streamlit as st

def home_page():
    """ Renders the landing page with title, guides, and quick actions. """
    st.markdown('<h1 style="text-align: center;">🏭 Advanced Plant AI Assistant </h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-weight: bold;">🧑‍💻 Developed by Parth Shukla ✨</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("### 🚀 Quick Actions")

    with st.expander("📖 User Guide", expanded=False):
        st.markdown("""
**What this app does**
- Answers plant engineering questions using your uploaded **PDF documentation** (RAG).
- Can also mix in **general knowledge** when needed (Hybrid mode).
- Lets you **switch models** (local GGUF via ctransformers or HF Flan‑T5) and **tune generation**.
**Recommended flow**
1. **Index your docs:** Go to **Index** page → Build/Refresh RAG Index after placing PDFs in `./Database`.
2. **Pick a model:** Local Mistral GGUF (default) or try Llama‑2 / Qwen2 (auto‑download).
3. **Set persona:** Safety Officer / Maintenance / Process Engineer or a custom system prompt.
4. **Ask a question:** Use the chat input at the bottom. Your message appears immediately; the assistant streams.
5. **Verify:** Check **Sources** and always validate with official procedures.
**Tips**
- Ask **specific, technical** questions: include equipment tags, units, ranges.
- Use **Hybrid Search** + **Reranking** for better retrieval.
- Increase **Compression budget** for richer context; lower it to keep answers tight.
- If responses are repetitive, increase **Repetition penalty** or enable **Mirostat**.
**Safety**
- The assistant may be imperfect. **Always validate** critical steps with certified personnel and official SOPs.
        """)

    with st.expander("ℹ️ Pages Info", expanded=False):
        st.markdown("""
- **🏠 Home:** The home page for the Advanced Plant AI assistant.
- **💬 Chat:** The main chat interface where you interact with the AI assistant.
- **📚 Index:** Manage the knowledge base. Upload, view, and build the search index for your PDF documents.
- **🧠 Models:** View and manage the available AI models.
- **🔎 RAG Inspector:** Page dedicated for a tool to dissect the entire Retrieval-Augmented Generation pipeline.
- **✅ Query Analyzer:** A powerful tool to see the step-by-step backend process of how a query is handled.
- **🗒️ Logs:** Review a detailed history of all your past queries and the AI's full responses.
- **🗃️ Database Analysis:** Get insights into your document library, including file types, sizes, and page counts.
- **🎚️ Presets:** Save and load your favorite combinations of models and settings.
- **📊 Analytics:** See performance metrics, such as average response time and model usage.
- **📦 Exports:** Generate and download reports of your chat history.
- **🧩 Developer Tools:** Access tools to clear caches, reset analytics, and inspect the session state.
        """)

    qcol1, qcol2, qcol3, qcol4 = st.columns(4)
    if qcol1.button("🛠️ Maintenance Guide", use_container_width=True):
        st.session_state["chat"].append({"role": "user", "content": "What are the routine maintenance procedures?"})
        st.session_state.current_page = "Chat"
        st.rerun()
    if qcol2.button("⚡ Safety Protocols", use_container_width=True):
        st.session_state["chat"].append({"role": "user", "content": "What are the key safety protocols and emergency procedures?"})
        st.session_state.current_page = "Chat"
        st.rerun()
    if qcol3.button("📈 KPIs & Parameters", use_container_width=True):
        st.session_state["chat"].append({"role": "user", "content": "What are the key performance indicators and operational parameters?"})
        st.session_state.current_page = "Chat"
        st.rerun()
    if qcol4.button("⚠️ Troubleshooting", use_container_width=True):
        st.session_state["chat"].append({"role": "user", "content": "What are common issues and troubleshooting steps?"})
        st.session_state.current_page = "Chat"
        st.rerun()

    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("💬 Go to Chat Page", use_container_width=True, type="primary"):
            st.session_state.current_page = "Chat"
            st.rerun()