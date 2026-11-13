"""
relay/grading.py
----------------
Factual grading engine evaluating pipeline outputs against ground truth facts and citations.
"""

import re
from typing import Dict, Any, Literal

AnswerGrade = Literal["correct", "wrong", "refused_correctly", "refused_wrongly"]

FACTUAL_REQUIREMENTS = {
    "q1": [r"0\.1", r"(bleu|perplexity)"],
    "q2": [r"512", r"2048"],
    "q3": [r"adam", r"4000"],
    "q4": [r"12\.5", r"1\.1"],
    "q5": [r"8", r"codebook"],
    "q6": [r"(160|200)"],
    "q7": [r"(variable|dynamic|boundary)"],
    "q8": [r"(gradient|differentiab|soft|gumbel)"],
    "q9": [r"(pg19|wikitext|perplexity|flops)"],
    "q10": [r"(acoustic|audio|frame|segment)"],
    "q14": [r"(7b|7 billion)"],
    "q16": [r"o\(1\)", r"o\(n\)"]
}

def grade_answer(
    question_data: Dict[str, Any],
    answer_text: str,
    status: str
) -> AnswerGrade:
    can_answer = question_data.get("can_answer", True)
    qid = question_data.get("id", "")
    ans_lower = answer_text.lower()

    if status != "ok":
        return "wrong"

    has_refusal = "papers do not say" in ans_lower or "does not contain" in ans_lower

    if not can_answer:
        if has_refusal:
            return "refused_correctly"
        else:
            return "refused_wrongly"

    if has_refusal:
        return "refused_wrongly"

    patterns = FACTUAL_REQUIREMENTS.get(qid, [])
    for pat in patterns:
        if not re.search(pat, ans_lower):
            return "wrong"

    expected_paper = question_data.get("paper", "")
    if expected_paper != "cross-paper":
        if expected_paper not in answer_text:
            return "wrong"

    return "correct"
