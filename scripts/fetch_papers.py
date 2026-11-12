"""Fetch and verify the 3 required arXiv papers at exact versions."""
import os
import urllib.request
from pathlib import Path
import fitz  # PyMuPDF

PAPERS = [
    {
        "id": "1706.03762v7",
        "title": "Attention Is All You Need",
        "filename": "1706.03762v7_attention_is_all_you_need.pdf",
        "url": "https://arxiv.org/pdf/1706.03762v7.pdf"
    },
    {
        "id": "2410.00037v1",
        "title": "Moshi: a speech-text foundation model for real-time dialogue",
        "filename": "2410.00037v1_moshi.pdf",
        "url": "https://arxiv.org/pdf/2410.00037v1.pdf"
    },
    {
        "id": "2507.07955v2",
        "title": "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling",
        "filename": "2507.07955v2_dynamic_chunking.pdf",
        "url": "https://arxiv.org/pdf/2507.07955v2.pdf"
    }
]

DATA_DIR = Path("data/papers")
EXTRACTED_DIR = Path("data/extracted")

def download_and_extract():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
    
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    for paper in PAPERS:
        pdf_path = DATA_DIR / paper["filename"]
        txt_path = EXTRACTED_DIR / f"{paper['id']}.txt"
        
        if not pdf_path.exists():
            print(f"Downloading {paper['title']} ({paper['id']})...")
            req = urllib.request.Request(paper["url"], headers=headers)
            with urllib.request.urlopen(req) as resp, open(pdf_path, "wb") as f:
                f.write(resp.read())
            print(f"  Saved to {pdf_path} ({pdf_path.stat().st_size} bytes)")
        else:
            print(f"Found existing {pdf_path}")
            
        # Extract text page by page with PyMuPDF
        doc = fitz.open(pdf_path)
        print(f"Extracting {len(doc)} pages from {paper['filename']}...")
        full_text = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            full_text.append(f"--- PAGE {page_num + 1} ---\n{text}")
            
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(full_text))
        print(f"  Extracted text to {txt_path} ({len(''.join(full_text))} chars)")

if __name__ == "__main__":
    download_and_extract()
