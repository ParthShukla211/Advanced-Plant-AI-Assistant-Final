# page_modules/query_analyzer.py
# An advanced, interactive tool to analyze the backend RAG and generation process

import streamlit as st
import pandas as pd
import time
import re
import nltk
from models.model_manager import get_llm_adapter
from rag.core import QueryExpander, HybridRetriever, DocumentReranker, ContextualCompressor
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# --- Augmented Model Data ---
ALL_MODELS = {
    "🟢 Local Mistral (GGUF)": {"emoji": "💨", "family": "Mistral", "size": "8B", "use_case": "High-quality chat and complex instruction following."},
    "🟡 Mistral-7B-Instruct (Q4_K_M, auto)": {"emoji": "🌬️", "family": "Mistral", "size": "7B", "use_case": "A great balance of high performance and resource usage."},
    "🟠 Local Phi-2 (Q5_0)": {"emoji": "💡", "family": "Phi (Microsoft)", "size": "2.7B", "use_case": "Excellent for logical reasoning in a compact size."},
    "🟣 Llama-2-7B-Chat (Q3_K_S, auto)": {"emoji": "🦙", "family": "Llama (Meta)", "size": "7B", "use_case": "A reliable and safe choice for general-purpose chat."},
    "🟤 Qwen2-7B-Instruct (Q4_K_M, auto)": {"emoji": "🐉", "family": "Qwen (Alibaba)", "size": "7B", "use_case": "Excels in multilingual tasks and complex instructions."},
    "🔵 Flan-T5 Small (HF)": {"emoji": "🍮", "family": "Flan-T5 (Google)", "size": "80M", "use_case": "Extremely fast for simple tasks like classification."},
    "🔵 Flan-T5 Base (HF)": {"emoji": "🍮", "family": "Flan-T5 (Google)", "size": "250M", "use_case": "A solid baseline for various NLP tasks."},
    "🔵 Flan-T5 Large (HF)": {"emoji": "🍮", "family": "Flan-T5 (Google)", "size": "780M", "use_case": "Higher quality for nuanced text-to-text tasks."}
}

def highlight_keywords(text, keywords):
    """Highlights keywords in a body of text using markdown."""
    for keyword in keywords:
        text = re.sub(f"({re.escape(keyword)})", r"`\1`", text, flags=re.IGNORECASE)
    return text

def page_query_analyzer():
    st.header("✅ Live Query Analyzer", divider="rainbow")
    st.markdown("Trace a query's journey from your input to the final AI-generated answer. Enter a query below and scroll down to see each of the 15 steps of the backend process unfold.")

    st.markdown("---")
    
    if not st.session_state.get("faiss_index"):
        st.warning("The RAG index has not been built yet. Please go to the **Index** page and build it first.")
        return

    # --- 1. Input & Configuration ---
    st.subheader("💬 Enter Your Prompt")
    with st.container(border=True):
        st.markdown("##### Configure Your Query")
        query = st.text_input("Enter a technical question to analyze:", key="analyzer_query_input")
        c1, c2, c3 = st.columns(3)
        with c1:
            persona_options = ["Default: Precise Industrial Assistant", "Safety Officer", "Maintenance Specialist", "Process Engineer"]
            st.selectbox("Select Persona", persona_options, key="analyzer_persona")
        with c2:
            from models.adapters import MODEL_FAMILY
            st.selectbox("Select Model", MODEL_FAMILY, key="analyzer_model")
        with c3:
            st.selectbox("Select Answer Mode", ["Plant Specific (RAG)", "General LLM", "Hybrid"], key="analyzer_mode")
        
        if st.button("🔬 Analyze Backend Process", use_container_width=True, type="primary"):
            if query:
                st.session_state.run_analyzer = True
                st.session_state.analyzer_query_to_run = query
            else:
                st.warning("Please enter a query to analyze.")

    # --- Main Analysis Flow ---
    if st.session_state.get("run_analyzer"):
        query = st.session_state.analyzer_query_to_run
        
        st.markdown("---")
        st.markdown("### ⚙️ Backend Process Breakdown (15 Steps)")

        embed_model, faiss_index, all_chunks = st.session_state.get("embed_model"), st.session_state.get("faiss_index"), st.session_state.get("all_chunks")
        
        # --- Display steps sequentially ---
        with st.container(border=True):
            st.markdown("#### 📥 Step 1: Query Received & Configured")
            model_info = ALL_MODELS.get(st.session_state.analyzer_model, {})
            c1, c2 = st.columns([1, 3])
            with c1: st.markdown(f"<p style='font-size: 52px; text-align: center;'>{model_info.get('emoji', '🧠')}</p>", unsafe_allow_html=True)
            with c2:
                st.markdown(f"**Selected Model:** `{st.session_state.analyzer_model}`")
                st.markdown(f"**Good for:** *{model_info.get('use_case', 'N/A')}*")
            st.info(f"**Query:** \"{query}\"")
            st.info(f"**Persona:** `{st.session_state.analyzer_persona}` | **Answer Mode:** `{st.session_state.analyzer_mode}`")
            with st.expander("What's happening in this step?"):
                st.markdown("The system receives your raw query and logs the initial configuration, including the selected model, persona, and answer mode. This sets the stage for the entire process.")

        with st.container(border=True):
            st.markdown("#### 🗣️ Step 2: Query Tokenization")
            tokens = word_tokenize(query.lower())
            st.metric("Total Tokens", len(tokens))
            st.write("Tokens:", tokens)
            with st.expander("What's happening in this step?"):
                st.markdown("The system breaks your query down into individual words or sub-words called **tokens**. This is the first step in understanding the language.")
            
        with st.container(border=True):
            st.markdown("#### 🔑 Step 3: Keyword Extraction")
            stop_words = set(stopwords.words('english'))
            keywords = [w for w in tokens if w.isalpha() and w not in stop_words]
            st.metric("Keywords for Search", len(keywords))
            st.write("Identified Keywords:", keywords)
            with st.expander("What's happening in this step?"):
                st.markdown("Common words (like 'the', 'is', 'a'), known as **stopwords**, are filtered out to identify the most important **keywords**. These keywords are crucial for the fast, literal search.")

        if st.session_state.analyzer_mode != "General LLM":
            # (RAG-specific steps)
            with st.container(border=True):
                st.markdown("#### 🧠 Step 4: Query Expansion")
                expander = QueryExpander()
                expanded_queries = expander.expand_query(query)
                st.write("**Generated Search Queries:**", expanded_queries)
                with st.expander("What's happening in this step?"):
                    st.markdown("To improve the chances of finding relevant documents, the system generates several variations of your original query. This helps to match a wider range of text in the database.")
            
            retriever = HybridRetriever(embed_model, all_chunks)
            with st.container(border=True):
                st.markdown("#### 📚 Step 5: Keyword-Based Retrieval (TF-IDF)")
                key_results = retriever.retrieve_keyword(query, top_k=5)
                for idx, score in key_results: st.markdown(f"- `{all_chunks[idx][1]}` (Score: {score:.2f})")
                with st.expander("What's happening in this step?"):
                    st.markdown("The system performs a fast search for documents that contain the identified **keywords**. This is great for finding exact matches for specific terms, tags, or codes.")

            with st.container(border=True):
                st.markdown("#### 💡 Step 6: Semantic Retrieval (Vector Search)")
                sem_results = retriever.retrieve_semantic(query, faiss_index, top_k=5)
                for idx, dist in sem_results: st.markdown(f"- `{all_chunks[idx][1]}` (Score: {1-dist:.2f})")
                with st.expander("What's happening in this step?"):
                    st.markdown("The system converts the query into a vector (a list of numbers representing its meaning) and searches for documents with a similar **semantic meaning**, even if they don't use the exact same keywords. This finds conceptually related information.")

            with st.container(border=True):
                st.markdown("#### 🤝 Step 7: Hybrid Fusion & Ranking")
                initial_chunks_with_scores = retriever.hybrid_retrieve(query, faiss_index, top_k=12)
                initial_chunks = [(all_chunks[idx][0], all_chunks[idx][1]) for idx, score in initial_chunks_with_scores]
                for idx, score in initial_chunks_with_scores[:5]: st.markdown(f"- `{all_chunks[idx][1]}` (Score: {score:.2f})")
                with st.expander("What's happening in this step?"):
                    st.markdown("The results from the keyword and semantic searches are combined (**fused**) into a single, relevance-ranked list. This gives you the best of both worlds: exact matches and conceptual similarity.")

            with st.container(border=True):
                st.markdown("#### 🗑️ Step 8: De-duplication")
                seen_texts = set()
                deduplicated_chunks = []
                for text, source in initial_chunks:
                    if text not in seen_texts:
                        seen_texts.add(text)
                        deduplicated_chunks.append((text, source))
                st.metric("Chunks before De-duplication", len(initial_chunks), f"-{len(initial_chunks) - len(deduplicated_chunks)} duplicates")
                with st.expander("What's happening in this step?"):
                    st.markdown("The system removes any duplicate document chunks from the retrieved list to ensure the context is clean and efficient.")

            with st.container(border=True):
                st.markdown("#### ✨ Step 9: Relevance Reranking")
                reranker = DocumentReranker()
                reranked_chunks = reranker.rerank_documents(query, deduplicated_chunks, top_k=8)
                st.info("The cross-encoder reranked the chunks. Here is the new top 5:")
                for i, (text, source) in enumerate(reranked_chunks[:5]): st.markdown(f"**{i+1}. `{source}`**")
                with st.expander("What's happening in this step?"):
                    st.markdown("A more powerful AI model (a **cross-encoder**) re-evaluates the top results for contextual relevance to the original query. This is a crucial step for improving the quality of the final context by removing 'false positives'.")

            with st.container(border=True):
                st.markdown("#### ✂️ Step 10: Contextual Compression")
                compressor = ContextualCompressor()
                compressed_chunks = compressor.compress_context(query, reranked_chunks)
                original_size = sum(len(text) for text, _ in reranked_chunks)
                compressed_size = sum(len(text) for text, _ in compressed_chunks)
                c1, c2, c3 = st.columns(3)
                c1.metric("Original Size", f"{original_size} chars")
                c2.metric("Compressed Size", f"{compressed_size} chars")
                c3.metric("Reduction", f"{(1 - compressed_size / original_size) * 100 if original_size > 0 else 0:.1f}%")
                final_context = "\n\n".join([c[0] for c in compressed_chunks])
                with st.expander("What's happening in this step?"):
                    st.markdown("The system compresses the reranked chunks, keeping only the most relevant sentences. This ensures the final context is concise, focused, and fits within the model's limits.")

        with st.container(border=True):
            st.markdown("#### 🎭 Step 11: Persona Integration")
            persona_map = {"Default: Precise Industrial Assistant": "You are a helpful industrial assistant.", "Safety Officer": "You are a plant safety officer.", "Maintenance Specialist": "You are a senior maintenance engineer.", "Process Engineer": "You are a process engineer."}
            system_prompt = persona_map[st.session_state.analyzer_persona]
            st.info(f"The following System Prompt will be used:")
            st.write(f"> {system_prompt}")
            with st.expander("What's happening in this step?"):
                st.markdown("The selected **persona** is converted into a **system prompt**. This is a special instruction that guides the AI's personality, tone, and style of response (e.g., to be cautious like a safety officer or technical like an engineer).")

        with st.container(border=True):
            st.markdown("#### 📝 Step 12: Final Prompt Assembly")
            if st.session_state.analyzer_mode == "General LLM": final_prompt = f"{system_prompt}\n\n### Question\n{query}\n\n### Answer"
            else: final_prompt = f"""{system_prompt}\n\n### Context\n{final_context if 'final_context' in locals() else 'No context retrieved.'}\n\n### Question\n{query}\n\n### Answer"""
            with st.expander("Show Full Prompt Sent to LLM"): st.code(final_prompt, language="markdown")
            with st.expander("What's happening in this step?"):
                st.markdown("The system prompt, the compressed context (if any), and your original query are combined into a single, final prompt. This is the exact text that will be sent to the language model.")

        st.session_state["current_model_kind"] = st.session_state.analyzer_model
        kind, adapter, template = get_llm_adapter()
        
        with st.container(border=True):
            st.markdown("#### ⚙️ Step 13: Model & Parameter Loading")
            st.success(f"Adapter for **{kind.upper()}** model `{st.session_state.analyzer_model}` loaded successfully.")
            with st.expander("What's happening in this step?"):
                st.markdown("The selected AI model is loaded into memory, and the generation parameters (like temperature, max tokens, etc.) from the sidebar are prepared. These settings control the creativity and length of the AI's response.")
            st.json(st.session_state.get("gen_params", {"info": "Default parameters used."}))

        with st.container(border=True):
            st.markdown("#### 🧠 Step 14: Live Inference & Token Streaming")
            answer_placeholder = st.empty()
            answer_placeholder.markdown("`Waiting for model response...`")
            
            t_start = time.time(); first_token_time = None
            response_generator = adapter.generate(template=template, system_prompt="", user_prompt=final_prompt, stream=True)
            
            buffer = ""
            for chunk in response_generator:
                if first_token_time is None: first_token_time = time.time()
                buffer += chunk
                answer_placeholder.markdown(buffer + "▌")
            t_end = time.time()
            answer_placeholder.markdown(buffer)
            with st.expander("What's happening in this step?"):
                st.markdown("The final prompt is sent to the selected AI model, which generates the answer one **token** at a time. This is the core 'thinking' process of the AI, streamed back to you in real-time.")
        
        with st.container(border=True):
            st.markdown("#### 📊 Step 15: Answer Analysis")
            c1, c2, c3 = st.columns(3)
            time_to_first = (first_token_time - t_start) if first_token_time else 0
            total_time = t_end - t_start
            chars_per_sec = len(buffer) / total_time if total_time > 0 else 0
            c1.metric("Time to First Token", f"{time_to_first:.2f}s")
            c2.metric("Total Generation Time", f"{total_time:.2f}s")
            c3.metric("Characters per Second", f"{chars_per_sec:.1f}")
            with st.expander("What's happening in this step?"):
                st.markdown("The final, generated text is analyzed for key performance metrics, such as how quickly the answer started appearing and the overall generation speed.")
        
        st.session_state.run_analyzer = False