# page_modules/rag_inspector.py
# An advanced, interactive tool to inspect the RAG retrieval process

import streamlit as st
import pandas as pd
from rag.core import QueryExpander, HybridRetriever, DocumentReranker, ContextualCompressor

def page_rag_inspector():
    st.header("🔎 Advanced RAG Inspector")
    st.markdown("Use this tool to dissect the entire Retrieval-Augmented Generation pipeline. Enter a query, adjust the parameters, and see exactly how the system retrieves and processes information from your documents before sending it to the AI.")

    # Check if the RAG index is ready
    if not st.session_state.get("faiss_index"):
        st.warning("The RAG index has not been built yet. Please go to the **Index** page and build it first.")
        return

    st.markdown("---")

    # --- Interactive Controls ---
    with st.container(border=True):
        st.markdown("#### 1. Configure Inspection")
        query = st.text_input("Enter your query here:", key="rag_inspector_query")
        
        c1, c2 = st.columns(2)
        with c1:
            top_k = st.slider("Documents to retrieve (Top-K)", 5, 30, st.session_state.get("top_k_retrieval", 12), key="inspector_top_k")
            alpha = st.slider("Hybrid Search Balance (Alpha)", 0.0, 1.0, st.session_state.get("alpha_hybrid", 0.7), 0.05, key="inspector_alpha",
                              help="1.0 = Purely semantic search. 0.0 = Purely keyword search.")
        with c2:
            simulate_rerank = st.checkbox("Simulate Reranking", st.session_state.get("use_reranking", True), key="inspector_rerank")
            simulate_compress = st.checkbox("Simulate Contextual Compression", st.session_state.get("use_compression", True), key="inspector_compress")
            
        run_inspection = st.button("🔬 Inspect RAG Pipeline", use_container_width=True, type="primary")

    if run_inspection and query:
        # Load necessary components from session state
        embed_model = st.session_state.get("embed_model")
        faiss_index = st.session_state.get("faiss_index")
        all_chunks = st.session_state.get("all_chunks")

        # --- 2. Query Expansion ---
        with st.expander("2: Query Expansion", expanded=True):
            expander = QueryExpander()
            expanded_queries = expander.expand_query(query)
            st.write("**Original Query:**", query)
            st.write("**Expanded Queries for Search:**", expanded_queries)

        # --- 3. Document Retrieval ---
        st.markdown("#### 3: Document Retrieval (Semantic + Keyword)")
        retriever = HybridRetriever(embed_model, all_chunks)
        
        sem_results = retriever.retrieve_semantic(query, faiss_index, top_k=top_k)
        key_results = retriever.retrieve_keyword(query, top_k=top_k)

        sem_scores = {idx: 1 - dist for idx, dist in sem_results}
        key_scores = {idx: score for idx, score in key_results}
        
        all_indices = set(sem_scores.keys()) | set(key_scores.keys())
        
        retrieved_data = []
        for idx in all_indices:
            hybrid_score = (alpha * sem_scores.get(idx, 0.0)) + ((1 - alpha) * key_scores.get(idx, 0.0))
            if hybrid_score > 0:
                retrieved_data.append({
                    "Source": all_chunks[idx][1],
                    "Chunk Text": all_chunks[idx][0],
                    "Semantic Score": f"{sem_scores.get(idx, 0.0):.4f}",
                    "Keyword Score": f"{key_scores.get(idx, 0.0):.4f}",
                    "Hybrid Score": f"{hybrid_score:.4f}",
                    "Original Index": idx
                })
        
        # --- FIX: Check if retrieved_data is empty before creating DataFrame ---
        if not retrieved_data:
            st.warning("No documents were retrieved for this query with the current settings.")
            return
            
        retrieved_df = pd.DataFrame(retrieved_data).sort_values(by="Hybrid Score", ascending=False).reset_index(drop=True)
        st.dataframe(retrieved_df, use_container_width=True)

        initial_chunks = [(row["Chunk Text"], row["Source"]) for i, row in retrieved_df.iterrows()]

        # --- 4. Reranking (Optional) ---
        if simulate_rerank:
            st.markdown("#### 4: Document Reranking")
            if len(initial_chunks) > 1:
                reranker = DocumentReranker()
                reranked_chunks = reranker.rerank_documents(query, initial_chunks, top_k=min(8, len(initial_chunks)))
                st.info("Chunks have been re-ordered by the cross-encoder for higher relevance. Displaying the new order:")
                for i, (text, source) in enumerate(reranked_chunks):
                    with st.expander(f"**Rank {i+1}** | Source: `{source}`"):
                        st.write(text)
                final_chunks_for_compression = reranked_chunks
            else:
                st.write("Not enough documents to rerank.")
                final_chunks_for_compression = initial_chunks
        else:
            final_chunks_for_compression = initial_chunks

        # --- 5. Contextual Compression (Optional) ---
        if simulate_compress:
            st.markdown("#### 5: Contextual Compression")
            if final_chunks_for_compression:
                compressor = ContextualCompressor(max_context_length=st.session_state.get("compression_budget", 2000))
                compressed_chunks = compressor.compress_context(query, final_chunks_for_compression)
                st.info("The following compressed context is what would be passed to the language model:")
                final_context = "\n\n---\n\n".join([f"**Source: `{source}`**\n\n> {text}" for text, source in compressed_chunks])
                st.markdown(final_context)
            else:
                st.write("No documents to compress.")