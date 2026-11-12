"""Retrieval engine: Text chunking, metadata attachment, BM25 indexing, and re-ranking."""
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
from rank_bm25 import BM25Okapi

EXTRACTED_DIR = Path("data/extracted")

PAPER_TITLES = {
    "1706.03762v7": "Attention Is All You Need",
    "2410.00037v1": "Moshi: a speech-text foundation model for real-time dialogue",
    "2507.07955v2": "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling"
}

def clean_text(text: str) -> str:
    """Normalize whitespace and fix hyphenated line breaks."""
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

def tokenize(text: str) -> List[str]:
    """Simple alphanumeric tokenizer for BM25."""
    return re.findall(r"\b\w+\b", text.lower())

class DocumentStore:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunks: List[Dict[str, Any]] = []
        self.bm25: BM25Okapi = None
        self._build_store()

    def _build_store(self):
        chunks = []
        for txt_file in sorted(EXTRACTED_DIR.glob("*.txt")):
            paper_id = txt_file.stem
            paper_title = PAPER_TITLES.get(paper_id, paper_id)
            content = txt_file.read_text(encoding="utf-8")
            
            # Split by pages
            pages = re.split(r"--- PAGE (\d+) ---", content)
            
            for i in range(1, len(pages), 2):
                page_num = int(pages[i])
                page_text = clean_text(pages[i + 1])
                
                # Sliding window chunking
                start = 0
                while start < len(page_text):
                    end = min(start + self.chunk_size, len(page_text))
                    chunk_str = page_text[start:end].strip()
                    if len(chunk_str) > 60:  # Ignore trivial tiny chunks
                        chunk_id = f"{paper_id}_p{page_num}_{start}"
                        chunks.append({
                            "chunk_id": chunk_id,
                            "paper_id": paper_id,
                            "paper_title": paper_title,
                            "page_num": page_num,
                            "text": chunk_str
                        })
                    start += self.chunk_size - self.chunk_overlap
                    if start >= len(page_text):
                        break

        self.chunks = chunks
        tokenized_corpus = [tokenize(c["text"]) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        rerank: bool = True
    ) -> List[Dict[str, Any]]:
        """BM25 retrieval with optional entity re-ranking."""
        if not self.chunks or self.bm25 is None:
            return []

        tokens = tokenize(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k * 3]

        candidates = []
        for idx in top_indices:
            score = scores[idx]
            if score <= 0.0:
                continue
            item = dict(self.chunks[idx])
            item["bm25_score"] = float(score)
            candidates.append(item)

        if not candidates:
            return []

        if rerank:
            # Re-rank by checking specific named entity and term coverage
            query_lower = query.lower()
            key_entities = ["moshi", "mimi", "transformer", "h-net", "attention", "dynamic chunking", "warmup", "bleu", "label smoothing"]
            matched_entities = [e for e in key_entities if e in query_lower]
            
            for c in candidates:
                c_text = c["text"].lower()
                bonus = sum(2.5 for e in matched_entities if e in c_text)
                c["final_score"] = c["bm25_score"] + bonus
            
            candidates.sort(key=lambda x: x["final_score"], reverse=True)

        return candidates[:top_k]

# Cache stores for the three predefined retrieval settings
_stores: Dict[str, DocumentStore] = {}

def get_store_for_setting(setting_name: str) -> DocumentStore:
    from relay.config import RETRIEVAL_SETTINGS
    cfg = RETRIEVAL_SETTINGS.get(setting_name, RETRIEVAL_SETTINGS["setting_2"])
    key = f"{cfg['chunk_size']}_{cfg['chunk_overlap']}"
    if key not in _stores:
        _stores[key] = DocumentStore(chunk_size=cfg["chunk_size"], chunk_overlap=cfg["chunk_overlap"])
    return _stores[key]

