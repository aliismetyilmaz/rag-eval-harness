"""Scoring: deterministic checks first, an LLM judge only where judgment is needed."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from . import llm
from .corpus import Chunk

JUDGE_PROMPT = """You grade answers from a customer-support assistant against a reference answer.

Correctness, compared with the reference answer:
- correct: states the key facts of the reference and contradicts none of them.
- partial: some key facts right, but something important is missing or vague.
- incorrect: wrong, contradicts the reference, or relies on an outdated or deprecated fact.

Groundedness, compared only with the cited excerpts (ignore the reference here):
- fully: every factual claim is supported by the cited excerpts.
- partially: some claims are supported, at least one is not.
- not: the claims are not supported by the cited excerpts, or nothing was cited.

Simple arithmetic on supported numbers counts as supported."""


class Verdict(BaseModel):
    correctness: Literal["correct", "partial", "incorrect"]
    groundedness: Literal["fully", "partially", "not"]
    rationale: str


CORRECTNESS_SCORE = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}


def judge(
    question: dict, answer_text: str, cited: list[Chunk], model: str
) -> tuple[Verdict | None, llm.Call]:
    excerpts = "\n\n".join(f"[{c.id}]\n{c.text}" for c in cited) or "(no excerpts cited)"
    user = (
        f"Question: {question['question']}\n\n"
        f"Reference answer: {question['reference_answer']}\n\n"
        f"Assistant answer: {answer_text}\n\n"
        f"Cited excerpts:\n\n{excerpts}"
    )
    call = llm.ask(model=model, system=JUDGE_PROMPT, user=user, output_format=Verdict)
    return call.parsed, call


def score(
    question: dict,
    answer,
    sources: list[Chunk],
    verdict: Verdict | None,
) -> dict:
    """Turn one answer into metrics. `answer` is None if the model declined."""
    expected = set(question["expected_docs"])
    retrieved_docs = {c.doc_id for c in sources}
    result: dict = {
        "retrieval_recall": (
            len(expected & retrieved_docs) / len(expected)
            if question["answerable"] and expected and sources
            else None
        ),
    }
    if answer is None:
        return result | {"refused": True, "correctness": 0.0, "hallucinated": False}

    source_ids = {c.id: c for c in sources}
    cited = [source_ids[c] for c in answer.citations if c in source_ids]
    result |= {
        "refused": False,
        "abstention_correct": answer.answerable == question["answerable"],
        "citations_valid": len(cited) == len(answer.citations),
        "cited_expected_doc": (
            any(c.doc_id in expected for c in cited) if question["answerable"] else None
        ),
    }

    if not question["answerable"]:
        # The right behavior is to say the docs don't cover it. Answering anyway
        # is a hallucination by construction, so no judge call is needed.
        result["correctness"] = 1.0 if not answer.answerable else 0.0
        result["hallucinated"] = answer.answerable
    elif not answer.answerable:
        # Declined a question the docs do answer: wrong, but not a hallucination.
        result["correctness"] = 0.0
        result["hallucinated"] = False
    else:
        assert verdict is not None
        result["correctness"] = CORRECTNESS_SCORE[verdict.correctness]
        result["grounded"] = verdict.groundedness == "fully"
        result["hallucinated"] = verdict.groundedness == "not" or verdict.correctness == "incorrect"
    return result


def needs_judge(question: dict, answer) -> bool:
    return answer is not None and question["answerable"] and answer.answerable
