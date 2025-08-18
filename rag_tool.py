# rag_tool.py

import os
from typing import List, Optional
from functools import lru_cache

from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_core.tools import tool

STORE_DIR = os.getenv("RAG_STORE_DIR", "./rag_store")
EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "nomic-embed-text")

class _EmptyRetriever:
    def get_relevant_documents(self, query: str) -> List:
        return []

@lru_cache(maxsize=1)
def _cached_retriever(store_path: str, k: int):
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    if not os.path.isdir(store_path):
        return _EmptyRetriever()
    try:
        vs = FAISS.load_local(store_path, embeddings, allow_dangerous_deserialization=True)
        return vs.as_retriever(search_type="similarity", search_kwargs={"k": k})
    except Exception as e:
        print(f"[RAG] WARN: could not load FAISS index at {store_path}: {e}")
        return _EmptyRetriever()

def load_retriever(store_path: str = STORE_DIR, k: int = 4):
    # cached by (store_path, k)
    return _cached_retriever(store_path, k)

@tool("rag_search", return_direct=False)
def rag_tool(query: str, k: Optional[int] = 4) -> str:
    """
    Retrieve relevant context from local documents (txt/md/pdf).
    Returns short snippets with [file p.N] markers.
    """
    retriever = load_retriever(STORE_DIR, k or 4)
    docs = retriever.get_relevant_documents(query)
    if not docs:
        return "CONTEXT:\n(No relevant context found.)"
    lines = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source") or d.metadata.get("path") or f"doc-{i}"
        page = d.metadata.get("page")
        page_tag = f" p.{page}" if page is not None else ""
        text = (d.page_content or "").strip().replace("\n", " ")
        lines.append(f"- [{os.path.basename(src)}{page_tag}] {text[:500]}...")
    return "CONTEXT:\n" + "\n".join(lines)