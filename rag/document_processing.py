# rag/document_processing.py
# Document processing functions

import os
from datetime import datetime
from typing import List, Tuple
from PyPDF2 import PdfReader
import pandas as pd
from nltk.tokenize import sent_tokenize

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
                print(f"Error reading {file}: {e}")
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