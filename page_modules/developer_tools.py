# pages/developer_tools.py
# Developer tools page

import streamlit as st

def page_developer_tools():
    st.header("🧩 Developer Tools")
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🧽 Clear Index & Cache", use_container_width=True):
            st.session_state["faiss_index"] = None
            st.session_state["embed_model"] = None
            st.session_state["all_chunks"] = []
            st.cache_resource.clear()
            st.success("Cleared FAISS index, embeddings and caches.")
    with col2:
        if st.button("🧹 Clear Chat", use_container_width=True):
            st.session_state["chat"] = []
            st.success("Chat cleared.")
    with col3:
        if st.button("♻️ Reset Analytics", use_container_width=True):
            st.session_state["analytics"] = {
                "total_queries": 0,
                "avg_response_time": 0.0,
                "retrieval_stats": [],
                "model_usage": {},
            }
            st.success("Analytics reset.")

    st.markdown("#### Session Keys")
    st.code(", ".join(list(st.session_state.keys())) or "None", language="text")
    if st.session_state.get("all_chunks"):
        st.markdown(f"**Total Chunks Indexed:** {len(st.session_state['all_chunks'])}")
        if st.checkbox("Show sample chunks"):
            for i, chunk in enumerate(st.session_state["all_chunks"][:3]):
                st.text_area(
                    f"Sample Chunk {i+1}",
                    chunk[0][:500] + " ...",
                    height=120,
                )