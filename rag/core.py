# rag/core.py
# RAG core functionality

import numpy as np
from typing import List, Tuple, Dict, Any
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
from nltk.tokenize import word_tokenize

class QueryExpander:
    def __init__(self):
        from nltk.stem import WordNetLemmatizer
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize
        
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
        from nltk.tokenize import word_tokenize, sent_tokenize
        
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