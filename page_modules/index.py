# pages/index.py
# Index management page

from PyPDF2 import PdfReader
import numpy as np
import streamlit as st
import os
from rag.document_processing import load_pdfs, enhanced_chunk_text
from rag.retrieval import create_embeddings, build_faiss_index
from utils.helpers import ensure_dir

def page_index():
    st.header("📚 Index & Knowledge Base")
    st.markdown("---")
    st.write("Put PDF files in a folder named **`Database`** (same directory as this app).")

    if st.button("🔄 Build/Refresh RAG Index"):
        with st.spinner("Indexing PDFs..."):
            pdf_folder = "Database"
            if not os.path.exists(pdf_folder):
                st.error(f"📁 Database folder '{pdf_folder}' not found. Please create it and add PDF files.")
                return
            docs_texts = load_pdfs(pdf_folder)
            if not docs_texts:
                st.error("No PDF files found in the Database folder.")
                return

            all_chunks = []
            for pdf_name, doc in docs_texts:
                all_chunks.extend(enhanced_chunk_text(doc, pdf_name, chunk_size=st.session_state.get("chunk_size",400), overlap=st.session_state.get("chunk_overlap",50)))

            embeddings, embed_model = create_embeddings(all_chunks)
            faiss_index = build_faiss_index(np.array(embeddings))
            st.session_state["faiss_index"] = faiss_index
            st.session_state["embed_model"] = embed_model
            st.session_state["all_chunks"] = all_chunks

            st.success(f"✅ Indexed {len(all_chunks)} chunks from {len(docs_texts)} PDFs.")

    
    with st.expander("📤 Upload Additional PDFs"):
        uploads = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
        if uploads and st.button("📚 Add to Index"):
            with st.spinner("Processing uploaded PDFs..."):
                temp_dir = "temp_uploads"
                ensure_dir(temp_dir)
                new_chunks = []
                for up in uploads:
                    tmp = os.path.join(temp_dir, up.name)
                    with open(tmp, "wb") as f:
                        f.write(up.getbuffer())
                    try:
                        pdf = PdfReader(tmp)
                        text = ""
                        for page in pdf.pages:
                            t = page.extract_text()
                            if t:
                                text += t + "\n"
                        new_chunks.extend(enhanced_chunk_text(text, up.name, chunk_size=st.session_state.get("chunk_size",400), overlap=st.session_state.get("chunk_overlap",50)))
                    except Exception as e:
                        st.error(f"Error processing {up.name}: {e}")
                    finally:
                        if os.path.exists(tmp):
                            os.remove(tmp)

                if new_chunks:
                    st.session_state["all_chunks"].extend(new_chunks)
                    embeddings, embed_model = create_embeddings(st.session_state["all_chunks"])
                    faiss_index = build_faiss_index(np.array(embeddings))
                    st.session_state["faiss_index"] = faiss_index
                    st.session_state["embed_model"] = embed_model
                    st.success(f"Added {len(new_chunks)} chunks from {len(uploads)} new PDFs.")