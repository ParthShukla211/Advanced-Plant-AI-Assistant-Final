# rag/retrieval.py
# Retrieval functions

import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Tuple


from rag.core import QueryExpander, HybridRetriever, DocumentReranker, ContextualCompressor

@st.cache_resource(show_spinner=False)
def create_embeddings(chunks: List[Tuple[str, str]], model_name="sentence-transformers/all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    texts = [c[0] for c in chunks]
    emb = model.encode(texts, show_progress_bar=False)
    return emb, model

@st.cache_resource(show_spinner=False)
def build_faiss_index(embeddings: np.ndarray):
    emb = np.array(embeddings)
    dim = emb.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(emb)
    return index

def advanced_retrieve_and_generate(
    query: str,
    embed_model: SentenceTransformer,
    index,
    chunks: List[Tuple[str, str]],
    llm_generate_fn,
    *,
    max_length=512,
    top_k=10,
    use_reranking=True,
    use_compression=True,
    use_hybrid=True,
    alpha_hybrid=0.7,
    compression_budget=2000,
    system_prompt: str = ""
):
    # Check for interruption at the very start
    if st.session_state.get("interrupt_generation", False):
        return "", [], 0
    
    expander = QueryExpander()
    expanded = expander.expand_query(query)

    # Check for interruption after query expansion
    if st.session_state.get("interrupt_generation", False):
        return "", [], 0
    
    retriever = HybridRetriever(embed_model, chunks)
    all_docs = []
    for q in expanded:
        # Check for interruption before each query
        if st.session_state.get("interrupt_generation", False):
            return "", [], 0
            
        res = retriever.hybrid_retrieve(q, index, top_k=top_k, alpha=alpha_hybrid, use_hybrid=use_hybrid)
        for idx,_ in res:
            if idx < len(chunks):
                all_docs.append(chunks[idx])

    # Check for interruption after retrieval
    if st.session_state.get("interrupt_generation", False):
        return "", [], 0
    
    seen = set(); uniq = []
    for d in all_docs:
        t = d[0]
        if t not in seen:
            seen.add(t); uniq.append(d)

    if use_reranking and len(uniq) > 5:
        # Check for interruption before reranking
        if st.session_state.get("interrupt_generation", False):
            return "", [], 0
        uniq = DocumentReranker().rerank_documents(query, uniq, top_k=min(8, len(uniq)))
    else:
        uniq = uniq[:8]

    # Check for interruption after reranking
    if st.session_state.get("interrupt_generation", False):
        return "", [], 0
    
    if use_compression and uniq:
        # Check for interruption before compression
        if st.session_state.get("interrupt_generation", False):
            return "", [], 0
        uniq = ContextualCompressor(max_context_length=compression_budget).compress_context(query, uniq)

    # Final check before generation
    if st.session_state.get("interrupt_generation", False):
        return "", [], 0
    
    if not uniq:
        return "No relevant information found in the plant documentation.", [], 0

    context = "\n\n".join([d[0] for d in uniq])
    source_pdfs = list(set([d[1] for d in uniq]))

    prompt = f"""
{system_prompt.strip()}

You are the Plant AI Assistant. Answer **only** using the documentation provided below.
If the documentation is insufficient, say so clearly.

### Documentation
{context}

### Question
{query}

### Instructions
- Be specific and technical where appropriate
- Include relevant details and specifications
- Mention safety considerations if applicable
- Structure your answer clearly
- If you cite, refer to the source file names

### Answer
""".strip()

    # Check one more time before calling the LLM
    if st.session_state.get("interrupt_generation", False):
        return "", [], 0
        
    answer = llm_generate_fn(prompt, max_new_tokens=max_length)
    
    # Final check after generation
    if st.session_state.get("interrupt_generation", False):
        return "", [], 0
        
    return answer.strip(), source_pdfs, len(uniq)