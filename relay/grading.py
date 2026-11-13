"""
relay/grading.py
----------------
Factual grading engine:
- Evaluates model answers against expected ground truth facts and paper citations.
- Strict refusal grading:
  * An unanswerable question is graded 'refused_correctly' ONLY if status=='ok'
    AND the model explicitly refused ('papers do not say').
  * Timed out or shed requests are NEVER marked 'refused_correctly'.
  * Answerable questions that are refused are marked 'refused_wrongly'.
  * Answerable questions containing required key facts and citations are marked 'correct',
    otherwise 'wrong'.
"""

import re
from typing import Dict, Any, Literal

AnswerGrade = Literal["correct", "wrong", "refused_correctly", "refused_wrongly"]

# Key factual tokens required for answerable questions
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
    """Evaluates the pipeline answer strictly against gold facts and citations."""
    can_answer = question_data.get("can_answer", True)
    qid = question_data.get("id", "")
    ans_lower = answer_text.lower()

    # Rule 1: Failed, shed, or timed out requests did NOT produce an answer
    if status != "ok":
        return "wrong"

    has_refusal = "papers do not say" in ans_lower or "does not contain" in ans_lower

    # Rule 2: Unanswerable questions
    if not can_answer:
        if has_refusal:
            return "refused_correctly"
        else:
            return "refused_wrongly"

    # Rule 3: Answerable question was refused by model
    if has_refusal:
        return "refused_wrongly"

    # Rule 4: Verify required facts for answerable questions
    patterns = FACTUAL_REQUIREMENTS.get(qid, [])
    for pat in patterns:
        if not re.search(pat, ans_lower):
            return "wrong"

    # Rule 5: Verify presence of citation
    expected_paper = question_data.get("paper", "")
    if expected_paper != "cross-paper":
        if expected_paper not in answer_text:
            return "wrong"

    return "correct"

