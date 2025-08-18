# index_docs.py
import argparse, os
from langchain_community.document_loaders import (
    DirectoryLoader, TextLoader, PyPDFLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

def load_docs(docs_path: str):
    # TXT / MD
    txt_loader = DirectoryLoader(
        docs_path, glob="**/*.txt", loader_cls=TextLoader, show_progress=True
    )
    md_loader = DirectoryLoader(
        docs_path, glob="**/*.md", loader_cls=TextLoader, show_progress=True
    )
    # PDF (one Document per page, with page metadata)
    pdf_loader = DirectoryLoader(
        docs_path, glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True
    )
    docs = []
    for loader in (txt_loader, md_loader, pdf_loader):
        try:
            docs.extend(loader.load())
        except Exception as e:
            print(f"[WARN] loader failed: {e}")
    return docs

def build_index(docs_path: str, store_path: str, chunk_size: int, chunk_overlap: int):
    if not os.path.isdir(docs_path):
        raise SystemExit(f"[ERROR] Docs folder not found: {docs_path}")

    print(f"[RAG] Loading from: {docs_path}")
    documents = load_docs(docs_path)
    if not documents:
        raise SystemExit("[ERROR] No .txt/.md/.pdf files found.")

    # Optional: ensure useful metadata fields exist
    for d in documents:
        d.metadata.setdefault("source", d.metadata.get("file_path") or d.metadata.get("source") or "unknown")
        d.metadata.setdefault("page", d.metadata.get("page", None))

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)
    print(f"[RAG] Chunks: {len(chunks)}")

    if not chunks:
        print("[INFO] No extractable text found (PDF might be scanned or empty).")
        print("[INFO] Skipping FAISS build — no index will be created.")
        print("[HINT] Use OCR indexer or the vision tool to handle scanned PDFs.")
        return  # <-- Will not call FAISS.from_documents on empty data

    print("[RAG] Embeddings: nomic-embed-text (Ollama)")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    print("[RAG] Building FAISS index…")
    vs = FAISS.from_documents(chunks, embeddings)
    os.makedirs(store_path, exist_ok=True)
    vs.save_local(store_path)
    print(f"[RAG] Saved index to: {store_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="./docs")
    ap.add_argument("--store", default="./rag_store")
    ap.add_argument("--chunk-size", type=int, default=1000)
    ap.add_argument("--chunk-overlap", type=int, default=200)
    args = ap.parse_args()
    build_index(args.docs, args.store, args.chunk_size, args.chunk_overlap)
