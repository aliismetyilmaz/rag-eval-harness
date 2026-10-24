"""Tests for everything that runs without an API key: data, chunking, retrieval, scoring."""

import json
from pathlib import Path

from rag_eval.corpus import chunk_by_section, chunk_fixed, load_docs
from rag_eval.grading import Verdict, score
from rag_eval.pipelines import Answer, build_pipelines
from rag_eval.retrieval import BM25

ROOT = Path(__file__).resolve().parent.parent
DOCS = load_docs()
QUESTIONS = [json.loads(line) for line in (ROOT / "data/questions.jsonl").read_text().splitlines()]


def test_questions_reference_real_docs():
    doc_ids = {d.id for d in DOCS}
    assert len({q["id"] for q in QUESTIONS}) == len(QUESTIONS)
    for q in QUESTIONS:
        assert set(q["expected_docs"]) <= doc_ids, q["id"]
        assert q["answerable"] == (q["category"] != "unanswerable")


def test_section_chunks_carry_title_and_date():
    chunks = chunk_by_section(DOCS)
    pricing = [c for c in chunks if c.doc_id == "plans-and-pricing"]
    assert len(pricing) >= 4
    for c in pricing:
        assert c.text.startswith("# Plans and Pricing\n_Last updated: 2026-01-15_")
    assert len({c.id for c in chunks}) == len(chunks)


def test_fixed_chunks_cover_every_word():
    chunks = chunk_fixed(DOCS, size=80, overlap=20)
    for doc in DOCS:
        doc_words = set(doc.text.split())
        chunk_words = {w for c in chunks if c.doc_id == doc.id for w in c.text.split()}
        assert doc_words == chunk_words, doc.id


def test_bm25_finds_obvious_matches():
    index = BM25(chunk_by_section(DOCS))
    assert index.search("HMAC signature webhook", k=1)[0].doc_id == "webhooks"
    assert index.search("SAML Okta single sign-on", k=1)[0].doc_id == "security-and-sso"


def test_pipelines_build():
    pipelines = build_pipelines()
    assert list(pipelines) == ["closed_book", "rag_fixed", "rag_section", "full_context"]
    assert pipelines["closed_book"].sources("anything") == []
    assert len(pipelines["rag_section"].sources("refund seat")) == 4


def _q(answerable=True, expected=("billing",)):
    return {"answerable": answerable, "expected_docs": list(expected)}


def test_answering_unanswerable_question_is_hallucination():
    sources = chunk_by_section(DOCS)[:2]
    ans = Answer(answerable=True, answer="Yes, we sign BAAs.", citations=[])
    m = score(_q(answerable=False, expected=()), ans, sources, None)
    assert m["correctness"] == 0.0 and m["hallucinated"] is True


def test_abstaining_on_unanswerable_question_is_correct():
    ans = Answer(answerable=False, answer="The docs don't cover that.", citations=[])
    m = score(_q(answerable=False, expected=()), ans, [], None)
    assert m["correctness"] == 1.0 and m["hallucinated"] is False and m["abstention_correct"]


def test_judged_answer_and_citation_checks():
    billing = [c for c in chunk_by_section(DOCS) if c.doc_id == "billing"]
    ans = Answer(answerable=True, answer="No refund.", citations=[billing[0].id, "made-up#9"])
    verdict = Verdict(correctness="correct", groundedness="fully", rationale="")
    m = score(_q(), ans, billing, verdict)
    assert m["correctness"] == 1.0
    assert m["citations_valid"] is False  # one citation doesn't exist
    assert m["cited_expected_doc"] is True
    assert m["retrieval_recall"] == 1.0


def test_refusal_scores_zero():
    m = score(_q(), None, [], None)
    assert m["refused"] and m["correctness"] == 0.0
