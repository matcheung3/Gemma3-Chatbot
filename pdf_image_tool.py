# pdf_image_tool.py
import os
import io
import fitz  # PyMuPDF
from PIL import Image
from typing import List, Tuple

from langchain_core.tools import tool
from langchain_ollama import ChatOllama


# ---------- Config ----------
PAGE_IMG_DIR = "./page_images"                       # where rendered page PNGs go
VISION_MODEL = "PetrosStav/gemma3-tools:4b"         # your multimodal model
MAX_PAGES_PER_PDF = 3
RENDER_DPI = 200


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _abs_norm(path: str) -> str:
    """
    Return an absolute path usable by Ollama (no file:// prefix).
    Forward slashes are OK on Windows, but we don't add file://.
    """
    p = os.path.abspath(path)
    # Using forward slashes avoids some serialization edge cases on Windows
    return p.replace("\\", "/")


def _pdf_to_images(pdf_path: str, max_pages: int = MAX_PAGES_PER_PDF, dpi: int = RENDER_DPI) -> List[str]:
    """
    Render up to `max_pages` pages to PNG and return absolute file paths.
    Saved to ./page_images/<pdf_stem>/page-<n>.png
    """
    if not os.path.isfile(pdf_path):
        print(f"[VISION] WARN: PDF not found: {pdf_path}")
        return []

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"[VISION] WARN: Could not open PDF {pdf_path}: {e}")
        return []

    out_dir = os.path.join(PAGE_IMG_DIR, os.path.splitext(os.path.basename(pdf_path))[0])
    _ensure_dir(out_dir)

    paths = []
    pages = min(len(doc), max_pages)
    for i in range(pages):
        try:
            page = doc.load_page(i)
            pix = page.get_pixmap(dpi=dpi)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            fp = os.path.join(out_dir, f"page-{i+1}.png")
            img.save(fp, format="PNG")
            ap = _abs_norm(fp)
            if os.path.isfile(ap):
                print(f"[VISION] rendered: {ap}")
                paths.append(ap)
            else:
                print(f"[VISION] ERROR: did not find saved image at: {ap}")
        except Exception as e:
            print(f"[VISION] WARN: render/save failed for page {i+1} of {pdf_path}: {e}")

    return paths


def _build_mm_content(question: str, image_paths: List[str]) -> list:
    """
    Build LangChain multimodal content list with *local file paths* (no file://).
    """
    items = [{"type": "text", "text": question}]
    for p in image_paths:
        # IMPORTANT: pass plain absolute path; Ollama Python client will open() it.
        items.append({"type": "image_url", "image_url": p})
    return items


@tool("vision_pdf_search", return_direct=False)
def vision_pdf_search(query: str) -> str:
    """
    Multimodal PDF helper:
      - Finds PDFs under ./docs
      - Renders up to 3 pages to PNG
      - Asks the Gemma-3 model to answer based on those page images
    Returns a compact summary string derived from the images.
    """
    docs_root = "./docs"
    pdfs = []
    for root, _, files in os.walk(docs_root):
        for f in files:
            if f.lower().endswith(".pdf"):
                pdfs.append(os.path.join(root, f))

    if not pdfs:
        return "CONTEXT (vision): No PDFs found under ./docs."

    # Render first PDF (simplest); you can extend to choose by name or loop all
    pdf_path = pdfs[0]
    image_paths = _pdf_to_images(pdf_path, max_pages=MAX_PAGES_PER_PDF, dpi=RENDER_DPI)
    if not image_paths:
        return "CONTEXT (vision): Could not render any pages from PDFs."

    llm = ChatOllama(model=VISION_MODEL, temperature=0.2)
    content = _build_mm_content(
        question=(
            "You are given page images from a PDF. Answer the question by reading the images. "
            "If unsure, say so. Question: " + query
        ),
        image_paths=image_paths,
    )

    print(f"[VISION] sending {len(image_paths)} images to model")
    resp = llm.invoke([{"role": "user", "content": content}])
    answer = getattr(resp, "content", None) or str(resp)

    base = os.path.basename(pdf_path)
    pages = ", ".join([str(i + 1) for i in range(len(image_paths))])
    return f"CONTEXT (vision from {base} pages {pages}):\n{answer}"
