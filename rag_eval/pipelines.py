"""The four answering setups being compared, and the answer step itself."""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from . import llm
from .corpus import Chunk, chunk_by_section, chunk_fixed, load_docs
from .retrieval import BM25

TOP_K = 4

SYSTEM_PROMPT = """You are the support assistant for Relaykit, a customer-support help desk product.
Answer customer questions using only the documentation excerpts you are given. Each
excerpt starts with its ID in square brackets.

- If the excerpts contain the answer, set answerable to true, answer concisely, and list the
  IDs of every excerpt you relied on in citations.
- If the excerpts do not contain the specific answer, set answerable to false. In answer,
  say that the docs don't cover it, and mention any closely related facts the excerpts do
  contain. Never fill gaps with outside knowledge or guesses.
- If excerpts disagree, prefer the most recently dated source (a changelog entry overrides
  an older page) and say which source you followed. Deprecated or sunset features are not
  current answers."""

CLOSED_BOOK_PROMPT = """You are the support assistant for Relaykit, a customer-support help desk product.
Answer customer questions about Relaykit. If you don't know the answer, set answerable to
false and say so. Leave citations empty."""


class Answer(BaseModel):
    answerable: bool = Field(description="True only if the answer is supported by the excerpts.")
    answer: str
    citations: list[str] = Field(description="IDs of the excerpts the answer relies on.")


@dataclass
class Pipeline:
    name: str
    description: str
    chunks: list[Chunk] = field(default_factory=list)
    index: BM25 | None = None
    full_context: bool = False

    def sources(self, question: str) -> list[Chunk]:
        if self.index is not None:
            return self.index.search(question, TOP_K)
        if self.full_context:
            return self.chunks
        return []


def build_pipelines() -> dict[str, Pipeline]:
    docs = load_docs()
    section_chunks = chunk_by_section(docs)
    fixed_chunks = chunk_fixed(docs)
    pipelines = [
        Pipeline("closed_book", "No docs: the model answers from its own knowledge"),
        Pipeline(
            "rag_fixed",
            f"BM25 top-{TOP_K} over naive 80-word chunks",
            fixed_chunks,
            BM25(fixed_chunks),
        ),
        Pipeline(
            "rag_section",
            f"BM25 top-{TOP_K} over section chunks with doc title + date",
            section_chunks,
            BM25(section_chunks),
        ),
        Pipeline(
            "full_context",
            "Every doc in the prompt (cached), no retrieval",
            section_chunks,
            full_context=True,
        ),
    ]
    return {p.name: p for p in pipelines}


def format_sources(chunks: list[Chunk]) -> str:
    return "\n\n".join(f"[{c.id}]\n{c.text}" for c in chunks)


def answer(pipeline: Pipeline, question: str, model: str) -> tuple[list[Chunk], llm.Call]:
    sources = pipeline.sources(question)
    if pipeline.name == "closed_book":
        system: str | list[dict] = CLOSED_BOOK_PROMPT
        user = question
    elif pipeline.full_context:
        # The docs are identical for every question, so cache them in the system prompt.
        system = [
            {"type": "text", "text": SYSTEM_PROMPT},
            {
                "type": "text",
                "text": "Documentation excerpts:\n\n" + format_sources(sources),
                "cache_control": {"type": "ephemeral"},
            },
        ]
        user = f"Question: {question}"
    else:
        system = SYSTEM_PROMPT
        user = f"Documentation excerpts:\n\n{format_sources(sources)}\n\nQuestion: {question}"
    call = llm.ask(model=model, system=system, user=user, output_format=Answer)
    return sources, call
