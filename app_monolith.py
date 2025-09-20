# app_monolith.py
# -*- coding: utf-8 -*-
"""
🏭 Plant AI Assistant – ctransformers • Auth • Pro UI • Interrupt
- Local GGUF via ctransformers (streaming)
- HF Transformers (Flan-T5)
- RAG + Chat + Menus + Analytics + Logs + Exports + Personas + Presets
- Access control with token gate (Parth123@#)
- Interrupt generation mid-stream (⏹ Stop)
"""

import os
import time
import json
import hashlib
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional

import streamlit as st
import numpy as np
import pandas as pd

# RAG deps
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# NLP
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

# HF Transformers (for Flan-T5)
from transformers import pipeline

# GGUF (Option B) – pure Python
from ctransformers import AutoModelForCausalLM

# For optional pre-downloads
from huggingface_hub import hf_hub_download


# ----------------------------
# NLTK data ensure
# ----------------------------
def _ensure_nltk():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
_ensure_nltk()


# ----------------------------
# UI Config & Styles
# ----------------------------
st.set_page_config(
    page_title="Plant AI Assistant",   # keep emoji out of tab title to avoid duplication
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* Headings */
.main-header {
  background: linear-gradient(90deg, #2E86AB, #A23B72, #F18F01);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  font-size: 2.5rem;
  font-weight: 900;
  text-align: center;
  margin: 0.25rem 0 0.25rem 0;
}
.sub-header {
  text-align:center; color:#444; font-weight:600; margin-bottom: 0.8rem;
}

/* Boxes & badges */
.answer-box { background-color: #f0f8ff; padding: 16px; border-radius: 10px; border-left: 5px solid #2E86AB; margin: 8px 0; }
.general-answer-box { background-color: #f5f5f5; padding: 16px; border-radius: 10px; border-left: 5px solid #A23B72; margin: 8px 0; }
.metrics-wrap { margin: 12px 0 16px 0; }
.badge { display:inline-block; padding:4px 10px; background:#EEF2FF; color:#334155; border:1px solid #CBD5E1; border-radius:999px; font-size:0.85rem; margin: 4px 10px 0 0; }

/* Centered warning under input */
.bottom-warning {
  text-align:center; color:#444; font-weight:600; padding: 10px 0 4px 0; margin-top: 6px;
}

/* Login card */
.login-card {
  max-width: 560px;
  margin: 6vh auto 0 auto;
  padding: 22px 26px;
 
}
.login-title { text-align:center; font-weight:800; font-size:1.6rem; margin-bottom: 6px; }
.login-sub  { text-align:center; color:#555; margin-bottom: 12px; }

/* Reduce top padding to avoid a blank strip on some themes */
.block-container { padding-top: 1.2rem; }

/* Sidebar bottom action group */
.sidebar-bottom {
  margin-top: 18px;
  padding-top: 12px;
  border-top: 1px solid #E5E7EB;
}
</style>
""", unsafe_allow_html=True)


# =====================================================
# ---------------------- AUTH -------------------------
# =====================================================
ACCESS_TOKEN = "Parth123@#"

def safe_rerun():
    """Use st.rerun() if available; otherwise ignore (older builds)."""
    try:
        st.rerun()
    except Exception:
        pass

def auth_gate() -> bool:
    """
    Shows a compact login card until a valid token is entered.
    Returns True if authenticated, False otherwise.
    """
    if "authed" not in st.session_state:
        st.session_state.authed = False

    if st.session_state.authed:
        return True

    # Minimal hint in sidebar
    with st.sidebar:
        st.markdown("### 🔐 Access Required")
        st.info("Enter security token to access the app.")

    # Header (page title uses industry emoji)
    st.markdown('<h1 class="main-header">🏭 Plant AI Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">🧑‍💻 Developed by <strong>Parth Shukla</strong> ✨</div>', unsafe_allow_html=True)

    # Login card
    st.markdown('<div class="login-card">', unsafe_allow_html=True)
    st.markdown('<div class="login-title">🔐 Secure Access</div>', unsafe_allow_html=True)
    st.markdown('<div class="login-sub">Please enter your access token to continue</div>', unsafe_allow_html=True)

    token = st.text_input(
        "Access Token",
        type="password",
        placeholder="Enter token (case sensitive)",
        label_visibility="collapsed"  # removes extra label row (no white box)
    )
    login = st.button("Enter", type="primary", use_container_width=True)
    if login:
        if token == ACCESS_TOKEN:
            st.session_state.authed = True
            st.success("Access granted ✅")
            safe_rerun()
        else:
            st.error("Invalid token. Access denied.")

    st.markdown("</div>", unsafe_allow_html=True)
    return False


if not auth_gate():
    st.stop()


# =====================================================
# --------------------  RAG CORE  ---------------------
# =====================================================
class QueryExpander:
    def __init__(self):
        from nltk.stem import WordNetLemmatizer
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    def expand_query(self, query: str) -> List[str]:
        expanded = [query]
        words = word_tokenize(query.lower())
        important = [w for w in words if w not in self.stop_words and w.isalpha()]
        if len(important) >= 2:
            expanded += [
                " ".join(reversed(important)),
                f"What is {query}?",
                f"How does {query} work?",
                f"Explain {query}"
            ]
        return expanded

class HybridRetriever:
    def __init__(self, semantic_model: SentenceTransformer, chunks: List[Tuple[str, str]]):
        self.semantic_model = semantic_model
        self.chunks = chunks
        self.tfidf_vectorizer = TfidfVectorizer(max_features=8000, stop_words='english', ngram_range=(1,2))
        texts = [c[0] for c in chunks]
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
    def retrieve_semantic(self, query: str, index, top_k=10):
        qvec = self.semantic_model.encode([query])
        D,I = index.search(qvec, top_k)
        return [(int(I[0][i]), float(D[0][i])) for i in range(len(I[0]))]
    def retrieve_keyword(self, query: str, top_k=10):
        q = self.tfidf_vectorizer.transform([query])
        sims = cosine_similarity(q, self.tfidf_matrix).flatten()
        top_idx = np.argsort(sims)[::-1][:top_k]
        return [(int(i), float(sims[i])) for i in top_idx]
    def hybrid_retrieve(self, query, index, top_k=15, alpha=0.7, use_hybrid=True):
        if use_hybrid:
            sem = self.retrieve_semantic(query, index, top_k)
            key = self.retrieve_keyword(query, top_k)
            sem_s = {idx: 1 - d for idx, d in sem}
            key_s = {idx: s for idx, s in key}
            all_idx = set(sem_s) | set(key_s)
            scores = {i: alpha*sem_s.get(i,0.0) + (1-alpha)*key_s.get(i,0.0) for i in all_idx}
            return sorted(scores.items(), key=lambda x:x[1], reverse=True)[:top_k]
        else:
            sem = self.retrieve_semantic(query, index, top_k)
            sem_s = {idx: 1 - d for idx, d in sem}
            return sorted(sem_s.items(), key=lambda x:x[1], reverse=True)[:top_k]

class DocumentReranker:
    def __init__(self):
        try:
            self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        except Exception:
            self.cross_encoder = None
    def rerank_documents(self, query, documents, top_k=5):
        if not self.cross_encoder or len(documents) <= top_k:
            return documents[:top_k]
        try:
            pairs = [[query, d[0]] for d in documents]
            scores = self.cross_encoder.predict(pairs)
            ds = list(zip(documents, scores))
            ds.sort(key=lambda x:x[1], reverse=True)
            return [d for d,_ in ds[:top_k]]
        except Exception:
            return documents[:top_k]

class ContextualCompressor:
    def __init__(self, max_context_length=2000):
        self.max_context_length = max_context_length
    def compress_context(self, query, documents):
        qwords = set(word_tokenize(query.lower()))
        out = []
        n = max(1, len(documents))
        per_doc = self.max_context_length // n
        for text, pdf in documents:
            sents = sent_tokenize(text)
            scored = []
            for s in sents:
                swords = set(word_tokenize(s.lower()))
                overlap = len(qwords & swords)
                scored.append((s, overlap))
            scored.sort(key=lambda x:x[1], reverse=True)
            buf = ""
            cur = 0
            for s,_ in scored:
                if cur + len(s) <= per_doc:
                    buf += s + " "
                    cur += len(s)
                else:
                    break
            if buf.strip():
                out.append((buf.strip(), pdf))
        return out


# ---------------------
# Helpers
# ---------------------
def load_pdfs(pdf_folder: str) -> List[Tuple[str, str]]:
    docs = []
    if not os.path.exists(pdf_folder):
        return docs
    for file in os.listdir(pdf_folder):
        if file.lower().endswith(".pdf"):
            path = os.path.join(pdf_folder, file)
            try:
                pdf = PdfReader(path)
                text = ""
                for page in pdf.pages:
                    t = page.extract_text()
                    if t:
                        text += t + "\n"
                docs.append((file, text))
            except Exception as e:
                st.warning(f"Error reading {file}: {e}")
    return docs

def pdf_meta_scan(pdf_folder: str) -> pd.DataFrame:
    rows = []
    if not os.path.exists(pdf_folder):
        return pd.DataFrame(columns=["file","ext","size_kb","pages","modified","path"])
    for file in os.listdir(pdf_folder):
        path = os.path.join(pdf_folder, file)
        if not os.path.isfile(path): continue
        ext = os.path.splitext(file)[1].lower()
        size_kb = os.path.getsize(path)/1024.0
        modified = datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M:%S")
        pages = None
        if ext == ".pdf":
            try:
                pdf = PdfReader(path)
                pages = len(pdf.pages)
            except Exception:
                pages = None
        rows.append({"file":file, "ext":ext.replace(".","").upper(),
                     "size_kb":size_kb, "pages":pages, "modified":modified, "path":path})
    return pd.DataFrame(rows).sort_values("file").reset_index(drop=True)

def enhanced_chunk_text(text: str, pdf_name: str, chunk_size=400, overlap=50) -> List[Tuple[str, str]]:
    sents = sent_tokenize(text)
    chunks = []
    cur = ""
    size = 0
    for s in sents:
        w = len(s.split())
        if size + w <= chunk_size:
            cur += s + " "; size += w
        else:
            if cur.strip(): chunks.append((cur.strip(), pdf_name))
            ov = " ".join(cur.split()[-overlap:]) if cur else ""
            cur = (ov + " " + s + " ").strip()
            size = len(cur.split())
    if cur.strip(): chunks.append((cur.strip(), pdf_name))
    return chunks

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
    expander = QueryExpander()
    expanded = expander.expand_query(query)

    retriever = HybridRetriever(embed_model, chunks)
    all_docs = []
    for q in expanded:
        res = retriever.hybrid_retrieve(q, index, top_k=top_k, alpha=alpha_hybrid, use_hybrid=use_hybrid)
        for idx,_ in res:
            if idx < len(chunks):
                all_docs.append(chunks[idx])

    seen = set(); uniq = []
    for d in all_docs:
        t = d[0]
        if t not in seen:
            seen.add(t); uniq.append(d)

    if use_reranking and len(uniq) > 5:
        uniq = DocumentReranker().rerank_documents(query, uniq, top_k=min(8, len(uniq)))
    else:
        uniq = uniq[:8]

    if use_compression and uniq:
        uniq = ContextualCompressor(max_context_length=compression_budget).compress_context(query, uniq)

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

    answer = llm_generate_fn(prompt, max_new_tokens=max_length)
    return answer.strip(), source_pdfs, len(uniq)


# =====================================================
# -----------------  LLM ADAPTERS  --------------------
# =====================================================
def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def md5_file(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

# ---- ctransformers choices (local file OR HF auto-download) ----
CTRANS_CHOICES: Dict[str, Dict[str, Any]] = {
    "🟡 Local Mistral (GGUF)": {
        "path": r"C:\Plant AI Assistant\models\mistral-7b-instruct-v0.2.Q8_0.gguf",
        "model_type": "mistral",
        "template": "mistral"
    },
    "🟡 Mistral-7B-Instruct (Q4_K_M, auto)": {
        "repo_id": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        "filename": "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "model_type": "mistral",
        "template": "mistral"
    },
    "🟡 Llama-2-7B-Chat (Q4_K_M, auto)": {
        "repo_id": "TheBloke/Llama-2-7B-Chat-GGUF",
        "filename": "llama-2-7b-chat.Q4_K_M.gguf",
        "model_type": "llama",
        "template": "llama2"
    },
    "🟡 Qwen2-7B-Instruct (Q4_K_M, auto)": {
        "repo_id": "bartowski/Qwen2-7B-Instruct-GGUF",
        "filename": "qwen2-7b-instruct.Q4_K_M.gguf",
        "model_type": "qwen2",  # fallback to "qwen" if needed
        "template": "chatml"
    },
}

# ---- HF text2text choices ----
HF_T2T_CHOICES = {
    "Flan-T5 Small (HF)": "google/flan-t5-small",
    "Flan-T5 Base (HF)": "google/flan-t5-base",
    "Flan-T5 Large (HF)": "google/flan-t5-large",
}

MODEL_FAMILY = list(CTRANS_CHOICES.keys()) + list(HF_T2T_CHOICES.keys())


def _ctrans_resolve_local_or_repo(spec: Dict[str, Any]) -> Tuple[str, str, bool]:
    if "path" in spec:
        p = spec["path"]
        if os.path.exists(p):
            base, model_file = os.path.split(p)
            return base, model_file, False
    return spec["repo_id"], spec["filename"], True


class CTransChatModel:
    def __init__(self, spec: Dict[str, Any], *, context_length: int = 4096, gpu_layers: int = 0, seed: Optional[int]=None,
                 mirostat_mode:int=0, mirostat_tau:float=5.0, mirostat_eta:float=0.1, stop_sequences: Optional[List[str]]=None):
        self.spec = spec
        self.context_length = context_length
        self.gpu_layers = gpu_layers
        self.model = None
        self.model_type = spec.get("model_type", "mistral")
        self.seed = seed
        self.mirostat_mode = mirostat_mode
        self.mirostat_tau = mirostat_tau
        self.mirostat_eta = mirostat_eta
        self.stop_sequences = stop_sequences or []

    def load(self):
        if self.model is None:
            base, model_file, is_repo = _ctrans_resolve_local_or_repo(self.spec)
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    base,
                    model_file=model_file,
                    model_type=self.model_type,
                    context_length=self.context_length,
                    gpu_layers=self.gpu_layers
                )
            except Exception as e:
                if self.model_type == "qwen2":
                    self.model = AutoModelForCausalLM.from_pretrained(
                        base, model_file=model_file, model_type="qwen",
                        context_length=self.context_length, gpu_layers=self.gpu_layers
                    )
                else:
                    raise e
        return self.model

    @staticmethod
    def _format_prompt(template: str, system_prompt: str, user_prompt: str) -> str:
        sys = (system_prompt or "You are a helpful, precise industrial assistant.").strip()
        up = (user_prompt or "").strip()
        if template in ("mistral", "llama2"):
            return f"<s>[INST] <<SYS>>\n{sys}\n<</SYS>>\n\n{up} [/INST]"
        elif template == "chatml":  # Qwen-style
            return f"""<|im_start|>system
{sys}<|im_end|>
<|im_start|>user
{up}<|im_end|>
<|im_start|>assistant
"""
        else:
            return f"{sys}\n\n{up}\n"

    def _build_kwargs(self, max_new_tokens, temperature, top_p, top_k, repeat_penalty, stream):
        kwargs = dict(
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(top_k),
            repetition_penalty=float(repeat_penalty),
            stream=bool(stream),
        )
        if self.stop_sequences:
            kwargs["stop"] = self.stop_sequences
        if self.seed is not None:
            kwargs["seed"] = int(self.seed)
        if self.mirostat_mode in (1,2):
            kwargs["mirostat"] = int(self.mirostat_mode)
            kwargs["mirostat_tau"] = float(self.mirostat_tau)
            kwargs["mirostat_eta"] = float(self.mirostat_eta)
        return kwargs

    def generate(self, template: str, system_prompt: str, user_prompt: str, *,
                 max_new_tokens=512, temperature=0.3, top_p=0.9, top_k=40,
                 repeat_penalty=1.2, stream=False):
        model = self.load()
        prompt = self._format_prompt(template, system_prompt, user_prompt)
        kwargs = self._build_kwargs(max_new_tokens, temperature, top_p, top_k, repeat_penalty, stream)

        try:
            if stream:
                for token in model(prompt, **kwargs):
                    yield token
            else:
                return model(prompt, **kwargs)
        except TypeError:
            # Fallback if some kwargs unsupported by the wheel
            for k in ["mirostat", "mirostat_tau", "mirostat_eta", "seed", "stop"]:
                kwargs.pop(k, None)
            if stream:
                for token in model(prompt, **kwargs):
                    yield token
            else:
                return model(prompt, **kwargs)


@st.cache_resource(show_spinner=False)
def get_hf_t2t_pipeline(model_id: str):
    return pipeline("text2text-generation", model=model_id)

def hf_t2t_generate(pipe, prompt: str, *, max_new_tokens=512, temperature=0.3, top_p=0.9):
    try:
        out = pipe(prompt, max_length=max_new_tokens, do_sample=True, temperature=temperature, top_p=top_p)
        txt = out[0]["generated_text"]
        if "Answer:" in txt:
            txt = txt.split("Answer:")[-1].strip()
        return txt
    except Exception as e:
        return f"Error generating response: {e}"


# =====================================================
# ----------------- SESSION STATE INIT ----------------
# =====================================================
def _init_state():
    defaults = {
        "logs": [],
        "analytics": {"total_queries":0, "avg_response_time":0.0, "retrieval_stats":[], "model_usage":{}},
        "faiss_index": None,
        "embed_model": None,
        "all_chunks": [],
        "chat": [],
        "current_model_kind": "🟡 Local Mistral (GGUF)",
        "persona_choice": "Default: Precise Industrial Assistant",
        "custom_system_prompt": "",
        "answer_mode_main": "Plant Specific (RAG)",
        # generation control
        "is_generating": False,
        "interrupt_generation": False,
        "gen_run_id": None,
        "presets": {}
    }
    for k,v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
_init_state()


# =====================================================
# --------------------- SIDEBAR -----------------------
# =====================================================
with st.sidebar:
    st.title("⚙️ Controls")

    # ---- Navigation (selectbox, no radio) ----
    nav = st.selectbox(
        "📂 Menu",
        ["Home", "Models", "Index", "Database Analysis", "Presets", "Logs", "Analytics", "Exports", "Developer Tools"],
        index=0
    )

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

    # ---- Generation/Runtime (sidebar) ----
    st.markdown("**📝 Generation & Runtime**")
    streaming = st.checkbox("Stream tokens", value=True, key="streaming")
    max_tokens = st.number_input("Max new tokens", 64, 4096, 512, 32, key="max_tokens")
    temperature = st.slider("Temperature", 0.0, 1.5, 0.3, 0.05, key="temperature")
    top_p = st.slider("Top-p", 0.05, 1.0, 0.9, 0.05, key="top_p")
    top_k = st.slider("Top-k", 1, 200, 40, 1, key="top_k")
    repeat_penalty = st.slider("Repetition penalty", 1.0, 2.0, 1.2, 0.05, key="repeat_penalty")

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
        safe_rerun()


# Final system prompt (applied below)
if st.session_state["persona_choice"] == "Custom...":
    system_prompt_final = st.session_state["custom_system_prompt"].strip() or "You are a helpful, precise industrial assistant."
else:
    system_prompt_final = {
        "Default: Precise Industrial Assistant": "You are a helpful, precise industrial assistant for plant operations.",
        "Safety Officer": "You are a plant safety officer. Emphasize safety, compliance, and risk mitigation.",
        "Maintenance Specialist": "You are a senior maintenance engineer. Focus on diagnostics, preventive maintenance, and reliability.",
        "Process Engineer": "You are a process engineer. Optimize parameters, explain control logic, and discuss trade-offs."
    }[st.session_state["persona_choice"]]


# =====================================================
# -------------------- TOP HEADER ---------------------
# =====================================================
st.markdown('<h1 class="main-header">🏭 Plant AI Assistant</h1>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">🧑‍💻 Developed by <strong>Parth Shukla</strong> ✨</div>', unsafe_allow_html=True)


# =====================================================
# --------------------- PAGES -------------------------
# =====================================================
def page_models():
    st.subheader("🧠 Model Manager")
    kind = st.session_state["current_model_kind"]

    if kind in CTRANS_CHOICES:
        spec = CTRANS_CHOICES[kind]
        st.write("**Selected Local GGUF (ctransformers):**", kind)
        if "path" in spec:
            p = spec["path"]
            st.write(f"Local path: `{p}`")
            if os.path.exists(p):
                st.success("Local model file found ✅")
                st.code(f"MD5: {md5_file(p)}", language="text")
            else:
                st.warning("Local model file not found. You can use an auto-download option below.")
        else:
            st.write(f"Will auto-download from HF: `{spec['repo_id']}` / `{spec['filename']}`")

        if st.button("📥 Ensure Auto-Download Model Present"):
            if "repo_id" in spec:
                with st.spinner("Downloading model from Hugging Face (first time only)..."):
                    try:
                        local_file = hf_hub_download(repo_id=spec["repo_id"], filename=spec["filename"])
                        st.success(f"Ready: {local_file}")
                        st.code(f"MD5: {md5_file(local_file)}", language="text")
                    except Exception as e:
                        st.error(f"Download failed: {e}")
            else:
                st.info("This selection uses a local file; no download needed.")
    else:
        st.write("**Selected HF Transformers model:**", kind)
        st.write("No extra steps required. Model will auto-download when used.")

    st.markdown("---")
    st.write("**Available Local GGUF Options (ctransformers):**")
    st.json(CTRANS_CHOICES)


def page_index():
    st.subheader("📚 Knowledge Base & Index")
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

    st.markdown("---")
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

def page_database_analysis():
    st.subheader("🗃️ Database Analysis")
    pdf_folder = "Database"
    df = pdf_meta_scan(pdf_folder)
    if df.empty:
        st.info("No files found in the `Database` folder. Please add PDFs and refresh.")
        return

    total_files = len(df)
    pdf_files = df[df["ext"] == "PDF"]
    total_pdfs = len(pdf_files)
    total_pages = int(pdf_files["pages"].fillna(0).sum())
    total_size_mb = df["size_kb"].sum() / 1024.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Files", total_files)
    c2.metric("PDF Files", total_pdfs)
    c3.metric("Total Pages (PDF)", total_pages)
    c4.metric("Total Size (MB)", f"{total_size_mb:.2f}")

    st.markdown("#### File Type Distribution")
    type_counts = df["ext"].value_counts().reset_index()
    type_counts.columns = ["Type", "Count"]
    st.bar_chart(type_counts.set_index("Type"))

    st.markdown("#### Top 15 Largest Files (MB)")
    df_largest = df.sort_values(by="size_kb", ascending=False).head(15).copy()
    df_largest["size_mb"] = df_largest["size_kb"] / 1024.0
    st.dataframe(
        df_largest[["file", "ext", "size_mb", "pages", "modified"]],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("#### All Files")
    st.dataframe(
        df[["file", "ext", "size_kb", "pages", "modified", "path"]],
        use_container_width=True,
        hide_index=True,
    )

    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download File Inventory (CSV)",
        data=csv_data,
        file_name="database_inventory.csv",
        mime="text/csv",
    )


def page_presets():
    st.subheader("🎚️ Presets (Save / Apply)")
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


def page_logs():
    st.subheader("🗒️ System Logs")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Refresh"):
            try:
                st.rerun()
            except Exception:
                pass
    with col2:
        if st.button("🗑️ Clear Logs"):
            st.session_state["logs"] = []
            st.success("Logs cleared!")

    if not st.session_state["logs"]:
        st.info("No logs available yet.")
        return

    for i, log in enumerate(reversed(st.session_state["logs"])):
        with st.expander(f"Query {len(st.session_state['logs']) - i}: {log['q'][:60]}..."):
            st.markdown(f"**⏰ Time:** {log['time']}")
            st.markdown(f"**❓ Question:** {log['q']}")
            st.markdown(f"**💡 Answer:**\n\n{log['a']}")
            if log['sources']:
                st.markdown(f"**📂 Sources:** {', '.join(log['sources'])}")
            st.markdown(f"**🧠 Model:** {log['model']} &nbsp;&nbsp; **⚙️ Mode:** {log['mode']}")
            st.markdown(f"**📄 Documents Retrieved:** {log.get('docs_retrieved', 'N/A')}")
            st.markdown(f"**⏱️ Response Time:** {log.get('response_time', 'N/A')} sec")


def page_analytics():
    st.subheader("📊 Performance Analytics")
    analytics = st.session_state["analytics"]
    if analytics["total_queries"] == 0:
        st.info("No analytics data available yet. Ask some questions first!")
        return
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total Queries", analytics["total_queries"])
    with c2:
        st.metric("Avg Response Time", f"{analytics['avg_response_time']:.2f}s")
    with c3:
        most_used = (
            max(analytics["model_usage"].items(), key=lambda x: x[1])
            if analytics["model_usage"]
            else ("N/A", 0)
        )
        st.metric("Most Used Model", most_used[0])

    if analytics["retrieval_stats"]:
        st.markdown("#### Retrieval Performance")
        avg_docs = np.mean([s["docs_retrieved"] for s in analytics["retrieval_stats"]])
        st.metric("Avg Documents Retrieved", f"{avg_docs:.1f}")


def page_exports():
    st.subheader("📦 Exports & Reports")
    if st.session_state["logs"]:
        if st.button("📄 Generate Chat History Report (Markdown)"):
            report = "# Plant AI Assistant – Chat History Report\n\n"
            report += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            report += f"Total Queries: {len(st.session_state['logs'])}\n\n"
            for i, log in enumerate(st.session_state["logs"], 1):
                report += f"## Query {i}\n"
                report += f"**Time:** {log['time']}\n\n"
                report += f"**Question:** {log['q']}\n\n"
                report += f"**Answer:**\n{log['a']}\n\n"
                if log['sources']:
                    report += f"**Sources:** {', '.join(log['sources'])}\n\n"
                report += f"**Model:** {log['model']}\n\n"
                report += f"**Response Time:** {log.get('response_time','N/A')}s\n\n"
            st.download_button(
                "⬇️ Download Report",
                data=report,
                file_name="plant_ai_report.md",
                mime="text/markdown",
            )
    else:
        st.info("No chat history to export yet.")

    analytics = st.session_state["analytics"]
    if analytics["total_queries"] > 0:
        summary = f"""**Session Analytics**
- Total Queries: {analytics['total_queries']}
- Average Response Time: {analytics['avg_response_time']:.2f}s
- Documents in Index: {len(st.session_state.get('all_chunks', []))}
"""
        st.markdown(summary)
    else:
        st.info("No analytics data available yet.")


def page_developer_tools():
    st.subheader("🧩 Developer Tools")
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


# =====================================================
# -------------- ADAPTER SELECTION HELPERS ------------
# =====================================================
@st.cache_resource(show_spinner=False)
def _get_ctrans_adapter(
    spec: Dict[str, Any],
    context_length: int,
    gpu_layers: int,
    seed: Optional[int],
    mirostat_mode: int,
    mirostat_tau: float,
    mirostat_eta: float,
    stop_sequences: List[str],
):
    return CTransChatModel(
        spec,
        context_length=context_length,
        gpu_layers=gpu_layers,
        seed=seed,
        mirostat_mode=mirostat_mode,
        mirostat_tau=mirostat_tau,
        mirostat_eta=mirostat_eta,
        stop_sequences=stop_sequences,
    )


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
        adapter = _get_ctrans_adapter(
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


# =====================================================
# --------------------- HOME / CHAT -------------------
# =====================================================
def _start_generation():
    # mark a new run id; used to visually simulate interrupts
    st.session_state["is_generating"] = True
    st.session_state["interrupt_generation"] = False
    st.session_state["gen_run_id"] = str(time.time())


def _request_interrupt():
    # set interrupt flag and rerun; old run will continue server-side, but UI reruns and stops appending
    st.session_state["interrupt_generation"] = True
    try:
        st.rerun()
    except Exception:
        pass


def home_page():
    # ---------- Quick Actions & User Guide ----------
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

    qcol1, qcol2, qcol3, qcol4 = st.columns(4)
    if qcol1.button("🛠️ Maintenance Guide", use_container_width=True):
        st.session_state["chat"].append({"role": "user", "content": "What are the routine maintenance procedures?"})
        try:
            st.rerun()
        except Exception:
            pass
    if qcol2.button("⚡ Safety Protocols", use_container_width=True):
        st.session_state["chat"].append({"role": "user", "content": "What are the key safety protocols and emergency procedures?"})
        try:
            st.rerun()
        except Exception:
            pass
    if qcol3.button("📈 KPIs & Parameters", use_container_width=True):
        st.session_state["chat"].append({"role": "user", "content": "What are the key performance indicators and operational parameters?"})
        try:
            st.rerun()
        except Exception:
            pass
    if qcol4.button("🧯 Troubleshooting", use_container_width=True):
        st.session_state["chat"].append({"role": "user", "content": "What are common issues and troubleshooting steps?"})
        try:
            st.rerun()
        except Exception:
            pass

    # ---------- Main Controls (3 dropdowns) ----------
    st.markdown("### 🎛️ Main Controls")
    m1, m2, m3 = st.columns(3)
    with m1:
        idx = MODEL_FAMILY.index(st.session_state["current_model_kind"]) if st.session_state["current_model_kind"] in MODEL_FAMILY else 0
        model_selected_main = st.selectbox("Model Selection", MODEL_FAMILY, index=idx, key="model_select_main")
        if model_selected_main != st.session_state["current_model_kind"]:
            st.session_state["current_model_kind"] = model_selected_main
    with m2:
        persona_idx = ["Default: Precise Industrial Assistant","Safety Officer","Maintenance Specialist","Process Engineer","Custom..."].index(st.session_state["persona_choice"])
        persona_main = st.selectbox("Persona / System Prompt", ["Default: Precise Industrial Assistant","Safety Officer","Maintenance Specialist","Process Engineer","Custom..."], index=persona_idx, key="persona_select_main")
        if persona_main != st.session_state["persona_choice"]:
            st.session_state["persona_choice"] = persona_main
    with m3:
        mode_options = ["Plant Specific (RAG)", "General LLM", "Hybrid"]
        mode_idx = mode_options.index(st.session_state["answer_mode_main"]) if st.session_state["answer_mode_main"] in mode_options else 0
        mode_main = st.selectbox("Answer Mode", mode_options, index=mode_idx, key="answer_mode_select_main")
        if mode_main != st.session_state["answer_mode_main"]:
            st.session_state["answer_mode_main"] = mode_main

    # ---------- Chat history ----------
    for m in st.session_state["chat"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # ---------- Chat input ----------
    user_question = st.chat_input("Ask your plant engineering question...")

    # Permanent warning under input (centered)
    st.markdown(
        '<div class="bottom-warning">⚠️ Always verify critical information with official plant documentation and qualified personnel.</div>',
        unsafe_allow_html=True,
    )

    if user_question:
        # render user immediately
        st.session_state["chat"].append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        # Start generation state
        _start_generation()

        # assistant placeholders
        with st.chat_message("assistant"):
            # Inline Stop button (simulated interrupt via rerun)
            st.info("You can stop the response mid-generation anytime.")
            stop_col = st.columns([1, 6, 1])[0]
            with stop_col:
                if st.button("⏹ Stop", use_container_width=True):
                    _request_interrupt()
                    return

            out_ph = st.empty()
            typing_ph = st.empty()

            # Thinking animation (pre-first token)
            think_frames = ["🤔 Thinking…", "🤔 Thinking… ·", "🤔 Thinking… ··", "🤔 Thinking… ···"]
            tf = 0
            typing_ph.markdown(think_frames[tf % len(think_frames)])

            # Adapter
            model_kind, adapter, template = get_llm_adapter()

            plant_answer = ""
            general_answer = ""
            source_pdfs = []
            docs_retrieved = 0
            start = time.time()

            def _direct(prompt_text: str, max_new_tokens=None):
                max_new_tokens = max_new_tokens or st.session_state.get("max_tokens", 512)
                if model_kind == "ctrans":
                    return adapter.generate(
                        template=template,
                        system_prompt=system_prompt_final,
                        user_prompt=prompt_text,
                        max_new_tokens=max_new_tokens,
                        temperature=st.session_state.get("temperature", 0.3),
                        top_p=st.session_state.get("top_p", 0.9),
                        top_k=st.session_state.get("top_k", 40),
                        repeat_penalty=st.session_state.get("repeat_penalty", 1.2),
                        stream=False,
                    )
                else:
                    return hf_t2t_generate(
                        adapter,
                        prompt_text,
                        max_new_tokens=max_new_tokens,
                        temperature=st.session_state.get("temperature", 0.3),
                        top_p=st.session_state.get("top_p", 0.9),
                    )

            # ensure RAG if needed
            use_rag_now = st.session_state["answer_mode_main"] in ["Plant Specific (RAG)", "Hybrid"] and st.session_state.get("use_rag", True)
            if use_rag_now:
                if st.session_state["faiss_index"] is None:
                    with st.spinner("📚 Building RAG index from ./Database ..."):
                        pdf_folder = "Database"
                        if not os.path.exists(pdf_folder):
                            st.error(f"📁 Database folder '{pdf_folder}' not found. Create it and add PDF files.")
                            st.session_state["is_generating"] = False
                            return
                        docs_texts = load_pdfs(pdf_folder)
                        if not docs_texts:
                            st.error("No PDF files found in the Database folder.")
                            st.session_state["is_generating"] = False
                            return
                        all_chunks = []
                        for pdf_name, doc in docs_texts:
                            all_chunks.extend(
                                enhanced_chunk_text(
                                    doc,
                                    pdf_name,
                                    chunk_size=st.session_state.get("chunk_size", 400),
                                    overlap=st.session_state.get("chunk_overlap", 50),
                                )
                            )
                        embeddings, embed_model = create_embeddings(all_chunks)
                        faiss_index = build_faiss_index(np.array(embeddings))
                        st.session_state["faiss_index"] = faiss_index
                        st.session_state["embed_model"] = embed_model
                        st.session_state["all_chunks"] = all_chunks

                faiss_index = st.session_state["faiss_index"]
                embed_model = st.session_state["embed_model"]
                all_chunks = st.session_state["all_chunks"]

                if model_kind == "ctrans" and st.session_state.get("streaming", True) and not st.session_state.get("interrupt_generation", False):
                    # streaming loop
                    anim = ["⏳ Generating…", "⏳ Generating… ·", "⏳ Generating… ··", "⏳ Generating… ···"]
                    ai = 0
                    stream_buf = ""

                    def _stream(prompt_text, max_new_tokens=None):
                        nonlocal stream_buf, ai
                        for tok in adapter.generate(
                            template=template,
                            system_prompt=system_prompt_final,
                            user_prompt=prompt_text,
                            max_new_tokens=max_new_tokens or st.session_state.get("max_tokens", 512),
                            temperature=st.session_state.get("temperature", 0.3),
                            top_p=st.session_state.get("top_p", 0.9),
                            top_k=st.session_state.get("top_k", 40),
                            repeat_penalty=st.session_state.get("repeat_penalty", 1.2),
                            stream=True,
                        ):
                            # If user pressed Stop (new rerun), this run won't see it; but UI will rerun and stop appending
                            stream_buf += tok
                            out_ph.markdown(stream_buf)
                            typing_ph.markdown(anim[ai % len(anim)])
                            ai += 1
                        typing_ph.empty()
                        return stream_buf

                    plant_answer, source_pdfs, docs_retrieved = advanced_retrieve_and_generate(
                        query=user_question,
                        embed_model=embed_model,
                        index=faiss_index,
                        chunks=all_chunks,
                        llm_generate_fn=_stream,
                        max_length=st.session_state.get("max_tokens", 512),
                        top_k=st.session_state.get("top_k_retrieval", 12),
                        use_reranking=st.session_state.get("use_reranking", True),
                        use_compression=st.session_state.get("use_compression", True),
                        use_hybrid=st.session_state.get("use_hybrid_search", True),
                        alpha_hybrid=st.session_state.get("alpha_hybrid", 0.7),
                        compression_budget=st.session_state.get("compression_budget", 2200),
                        system_prompt=system_prompt_final,
                    )
                else:
                    # non-stream or interrupted
                    plant_answer, source_pdfs, docs_retrieved = advanced_retrieve_and_generate(
                        query=user_question,
                        embed_model=embed_model,
                        index=faiss_index,
                        chunks=all_chunks,
                        llm_generate_fn=lambda p, max_new_tokens=None: _direct(p, max_new_tokens or st.session_state.get("max_tokens", 512)),
                        max_length=st.session_state.get("max_tokens", 512),
                        top_k=st.session_state.get("top_k_retrieval", 12),
                        use_reranking=st.session_state.get("use_reranking", True),
                        use_compression=st.session_state.get("use_compression", True),
                        use_hybrid=st.session_state.get("use_hybrid_search", True),
                        alpha_hybrid=st.session_state.get("alpha_hybrid", 0.7),
                        compression_budget=st.session_state.get("compression_budget", 2200),
                        system_prompt=system_prompt_final,
                    )
                    typing_ph.empty()
                    out_ph.markdown(f'<div class="answer-box">{plant_answer}</div>', unsafe_allow_html=True)

            # general/hybrid
            if st.session_state["answer_mode_main"] in ["General LLM", "Hybrid"] and not st.session_state.get("interrupt_generation", False):
                gp = f"""Provide a comprehensive technical answer to this plant engineering question using your general knowledge.
Focus on technical accuracy, safety considerations, and best practices.

Question: {user_question}

Answer:
"""
                if model_kind == "ctrans" and st.session_state.get("streaming", True):
                    gen_ph = st.empty()
                    anim = ["⏳ Generating…", "⏳ Generating… ·", "⏳ Generating… ··", "⏳ Generating… ···"]
                    ai = 0
                    gb = ""
                    for tok in adapter.generate(
                        template=template,
                        system_prompt=system_prompt_final,
                        user_prompt=gp,
                        max_new_tokens=st.session_state.get("max_tokens", 512),
                        temperature=st.session_state.get("temperature", 0.3),
                        top_p=st.session_state.get("top_p", 0.9),
                        top_k=st.session_state.get("top_k", 40),
                        repeat_penalty=st.session_state.get("repeat_penalty", 1.2),
                        stream=True,
                    ):
                        gb += tok
                        gen_ph.markdown(f'<div class="general-answer-box">{gb}</div>', unsafe_allow_html=True)
                        typing_ph.markdown(anim[ai % len(anim)])
                        ai += 1
                    typing_ph.empty()
                    general_answer = gb
                else:
                    if model_kind == "ctrans":
                        general_answer = adapter.generate(
                            template=template,
                            system_prompt=system_prompt_final,
                            user_prompt=gp,
                            max_new_tokens=st.session_state.get("max_tokens", 512),
                            temperature=st.session_state.get("temperature", 0.3),
                            top_p=st.session_state.get("top_p", 0.9),
                            top_k=st.session_state.get("top_k", 40),
                            repeat_penalty=st.session_state.get("repeat_penalty", 1.2),
                            stream=False,
                        )
                    else:
                        general_answer = hf_t2t_generate(
                            adapter,
                            gp,
                            max_new_tokens=st.session_state.get("max_tokens", 512),
                            temperature=st.session_state.get("temperature", 0.3),
                            top_p=st.session_state.get("top_p", 0.9),
                        )
                    st.markdown(f'<div class="general-answer-box">{general_answer}</div>', unsafe_allow_html=True)

            # ensure plant answer boxed after streaming
            if plant_answer and (model_kind == "ctrans" and st.session_state.get("streaming", True)):
                out_ph.markdown(f'<div class="answer-box">{plant_answer}</div>', unsafe_allow_html=True)

            # metrics and sources
            end = time.time()
            rt = round(end - start, 2)
            st.markdown('<div class="metrics-wrap">', unsafe_allow_html=True)
            st.markdown(f'<span class="badge">⏱️ Response generated in {rt}s</span>', unsafe_allow_html=True)
            if source_pdfs:
                st.markdown(f'<span class="badge">📂 Sources: {", ".join(source_pdfs)}</span>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # finalize state
            st.session_state["is_generating"] = False

            # log + chat history
            parts = []
            if plant_answer:
                parts.append(plant_answer)
            if general_answer and st.session_state["answer_mode_main"] in ["General LLM", "Hybrid"]:
                parts.append(general_answer)
            combined = "\n\n".join(parts).strip() or "No output generated. Try enabling RAG or adjusting parameters."
            st.session_state["chat"].append({"role": "assistant", "content": combined})
            st.session_state["logs"].append({
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "model": st.session_state["current_model_kind"],
                "mode": st.session_state["answer_mode_main"],
                "q": user_question,
                "a": combined,
                "sources": list(set(source_pdfs)) if source_pdfs else [],
                "docs_retrieved": docs_retrieved if use_rag_now else 0,
                "response_time": rt,
            })
            an = st.session_state["analytics"]
            an["total_queries"] += 1
            an["avg_response_time"] = ((an["avg_response_time"] * (an["total_queries"] - 1) + rt) / an["total_queries"])
            an["model_usage"][st.session_state["current_model_kind"]] = an["model_usage"].get(st.session_state["current_model_kind"], 0) + 1
            if use_rag_now:
                an["retrieval_stats"].append({"docs_retrieved": docs_retrieved, "response_time": rt})


# =====================================================
# ----------------- ROUTER / RENDER -------------------
# =====================================================
if nav == "Home":
    home_page()
elif nav == "Models":
    page_models()
elif nav == "Index":
    page_index()
elif nav == "Database Analysis":
    page_database_analysis()
elif nav == "Presets":
    page_presets()
elif nav == "Logs":
    page_logs()
elif nav == "Analytics":
    page_analytics()
elif nav == "Exports":
    page_exports()
elif nav == "Developer Tools":
    page_developer_tools()
