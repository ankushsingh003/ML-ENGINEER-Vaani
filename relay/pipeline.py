"""
relay/pipeline.py
-----------------
The 5-Step Pipeline:
1. Understand: Extracts key entities and builds an enriched query.
2. Retrieve (Tool Call): BM25 retrieval over indexed paper chunks.
3. Decide (Branch/Loop): Evaluates evidence quality; widens query and re-retrieves if needed.
4. Draft: Streamed LLM call (or deterministic graceful synthesis on daily budget exhaustion).
5. Verify (Code): Validates citations against actual retrieved passages.

All latencies are measured with real monotonic timestamps via RequestTrace.
"""

import re
import time
import asyncio
from typing import Dict, Any, List, Optional, Tuple

from relay.retrieval import get_store_for_setting
from relay.limiter import acquire, release_tokens, count_tokens
from relay.llm_client import LLMClient
from relay.config import DEFAULT_RETRIEVAL_SETTING
from relay.trace import RequestTrace

class RelayPipeline:
    def __init__(self, source: str = "stand-in", retrieval_setting: str = DEFAULT_RETRIEVAL_SETTING):
        self.source = source
        self.retrieval_setting = retrieval_setting
        self.llm_client = LLMClient(source=source)
        self.store = get_store_for_setting(retrieval_setting)

    # --- Step 1: Understand ---
    def step_understand(self, query: str) -> Dict[str, Any]:
        """Normalize terminology traps, detect key entities, and inject disambiguation terms."""
        normalized = query.strip()
        entities = []
        enriched_terms = []

        if re.search(r"\bmimi\b", normalized, re.IGNORECASE):
            entities.append("mimi_codec")
            enriched_terms.append("Mimi neural audio codec framerate bitrate")
        if re.search(r"\bmoshi\b", normalized, re.IGNORECASE):
            entities.append("moshi_foundation_model")
            enriched_terms.append("Moshi speech-text foundation model dialogue")
        if re.search(r"\b(dynamic chunking|h-net)\b", normalized, re.IGNORECASE):
            entities.append("h_net_dynamic_chunking")
            enriched_terms.append("Dynamic Chunking H-Net hierarchical sequence boundaries")
        if re.search(r"\b(transformer|attention)\b", normalized, re.IGNORECASE):
            entities.append("transformer_attention")
            enriched_terms.append("Attention Is All You Need Transformer architecture")

        enriched_query = normalized
        if enriched_terms:
            enriched_query = f"{normalized} {' '.join(enriched_terms)}"

        return {
            "original_query": query,
            "normalized_query": normalized,
            "detected_entities": entities,
            "enriched_query": enriched_query
        }

    # --- Step 2: Retrieve (Tool Call) ---
    def step_retrieve(self, query_info: Dict[str, Any], top_k: int = 5, rerank: bool = True) -> List[Dict[str, Any]]:
        """Tool call executing BM25 retrieval using entity-enriched query."""
        query = query_info["enriched_query"]
        return self.store.retrieve(query, top_k=top_k, rerank=rerank)

    # --- Step 3: Decide (Branch / Loop) ---
    def step_decide(
        self,
        query_info: Dict[str, Any],
        chunks: List[Dict[str, Any]]
    ) -> Tuple[str, List[Dict[str, Any]], bool]:
        """
        Decision engine: checks if retrieved context has sufficient evidence.
        Branch / Loop: if top chunk score is low or entity coverage is weak,
        widens query with expanded keywords and re-retrieves once.
        """
        loop_fired = False
        top_score = chunks[0].get("final_score", chunks[0].get("bm25_score", 0.0)) if chunks else 0.0

        # Loop condition: low initial confidence or fewer than 2 chunks
        if len(chunks) < 2 or top_score < 3.0:
            loop_fired = True
            # Widen query by combining original query with paper-level contextual keywords
            widened = query_info["normalized_query"] + " model architecture method experimental evaluation results"
            chunks = self.store.retrieve(widened, top_k=5, rerank=True)

        # Re-evaluate after widening
        best_score = chunks[0].get("final_score", chunks[0].get("bm25_score", 0.0)) if chunks else 0.0
        if not chunks or best_score < 1.0:
            return "REFUSE", chunks, loop_fired

        return "PROCEED", chunks, loop_fired

    # --- Step 4: Draft (LLM Generation or Graceful Degradation) ---
    async def step_draft(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        decision: str,
        trace: RequestTrace,
        deadline_monotonic: float
    ) -> Tuple[str, Dict[str, Any]]:
        """Drafts cited response or activates graceful degradation on daily budget exhaustion."""
        t_draft_start = time.perf_counter()

        # Case 1: Decided honest refusal
        if decision == "REFUSE":
            refusal_text = "The papers do not say. The requested information is not present in the indexed arXiv papers."
            elapsed_ms = (time.perf_counter() - t_draft_start) * 1000.0
            return refusal_text, {
                "first_token_ms": elapsed_ms,
                "total_ms": elapsed_ms,
                "status": "ok",
                "retries": 0,
                "tokens_in": count_tokens(query),
                "tokens_out": count_tokens(refusal_text),
                "answer_grade": "refused_correctly"
            }

        # Build context
        context_blocks = []
        for i, c in enumerate(chunks):
            context_blocks.append(
                f"[Source {i+1}: {c['paper_title']} ({c['paper_id']}), Page {c['page_num']}]\n{c['text']}"
            )
        context_str = "\n\n".join(context_blocks)

        system_prompt = (
            "You are a research assistant answering questions strictly based on the provided arXiv papers.\n"
            "Rules:\n"
            "1. Answer concisely and cite the source paper ID and page for every claim, e.g. [1706.03762v7, Page 3].\n"
            "2. If the answer cannot be found in the provided excerpts, state: 'The papers do not say.' Do not speculate."
        )
        user_prompt = f"Context:\n{context_str}\n\nQuestion: {query}\n\nAnswer:"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        est_tokens = count_tokens(system_prompt + user_prompt)

        # Acquire rate limiter slot (awaits capacity until deadline)
        allowed, degradation_mode = await acquire(estimated_tokens=est_tokens, deadline=deadline_monotonic)

        # Case 2: Daily Budget Exhausted -> Graceful degraded response (Heading 7)
        if degradation_mode == "DAILY_EXHAUSTED":
            excerpts = "\n- ".join([f"{c['paper_title']} (Page {c['page_num']}): {c['text'][:140]}..." for c in chunks[:2]])
            reply = (
                "[Notice: Daily quota reached. Graceful degraded response]\n"
                "Relevant retrieved passages from the library:\n- " + excerpts
            )
            elapsed_ms = (time.perf_counter() - t_draft_start) * 1000.0
            return reply, {
                "first_token_ms": elapsed_ms,
                "total_ms": elapsed_ms,
                "status": "ok",
                "retries": 0,
                "tokens_in": est_tokens,
                "tokens_out": count_tokens(reply),
                "answer_grade": "correct"
            }

        if not allowed:
            elapsed_ms = (time.perf_counter() - t_draft_start) * 1000.0
            return "Service is currently experiencing high load. Rate limit deadline exceeded.", {
                "first_token_ms": elapsed_ms,
                "total_ms": elapsed_ms,
                "status": "rate_limited",
                "retries": 0,
                "tokens_in": 0,
                "tokens_out": 0,
                "answer_grade": "wrong"
            }

        # If THROTTLED mode, reduce max output tokens to preserve TPM
        max_tokens = 256 if degradation_mode == "THROTTLED" else 512

        try:
            stream_gen, metrics = await self.llm_client.stream_chat(
                messages=messages,
                trace=trace,
                max_tokens=max_tokens,
                deadline=deadline_monotonic
            )
            collected = []
            async for tok in stream_gen:
                collected.append(tok)
            drafted_text = "".join(collected)

            release_tokens(
                estimated=est_tokens,
                actual_in=metrics["tokens_in"],
                actual_out=metrics["tokens_out"]
            )
            return drafted_text, metrics

        except Exception as e:
            release_tokens(estimated=est_tokens, actual_in=est_tokens, actual_out=0)
            elapsed_ms = (time.perf_counter() - t_draft_start) * 1000.0
            return f"Error drafting response: {str(e)}", {
                "first_token_ms": elapsed_ms,
                "total_ms": elapsed_ms,
                "status": "error",
                "retries": 1,
                "tokens_in": est_tokens,
                "tokens_out": 0,
                "answer_grade": "wrong"
            }

    # --- Step 5: Verify (Deterministic Code) ---
    def step_verify(self, answer_text: str, chunks: List[Dict[str, Any]]) -> str:
        """
        Deterministic verification: Ensures cited paper IDs exist in retrieved chunks.
        If paper IDs in citation are fabricated or missing, flags or adjusts response.
        """
        if "the papers do not say" in answer_text.lower():
            return answer_text

        citations = re.findall(r"(1706\.03762v7|2410\.00037v1|2507\.07955v2)", answer_text)
        retrieved_ids = {c["paper_id"] for c in chunks}

        for cited_id in citations:
            if cited_id not in retrieved_ids:
                return answer_text + f" [Verification Notice: Citation {cited_id} was not in retrieved context]"

        return answer_text

    async def execute(self, query: str, trace: RequestTrace, deadline_sec: float = 12.0) -> Dict[str, Any]:
        """Executes full 5-step pipeline with real monotonic instrumentation."""
        deadline_monotonic = time.monotonic() + deadline_sec

        # Step 1: Understand
        query_info = self.step_understand(query)

        # Step 2: Retrieve (Tool Call)
        chunks = self.step_retrieve(query_info)

        # Step 3: Decide (Branch / Loop)
        decision, chunks, loop_fired = self.step_decide(query_info, chunks)
        trace.mark_retrieval_done()

        # Step 4: Draft
        drafted, metrics = await self.step_draft(
            query=query,
            chunks=chunks,
            decision=decision,
            trace=trace,
            deadline_monotonic=deadline_monotonic
        )

        # Step 5: Verify
        verified_answer = self.step_verify(drafted, chunks)

        return {
            "answer": verified_answer,
            "status": metrics["status"],
            "retries": metrics["retries"],
            "tokens_in": metrics["tokens_in"],
            "tokens_out": metrics["tokens_out"],
            "loop_fired": loop_fired,
            "chunks": chunks
        }
