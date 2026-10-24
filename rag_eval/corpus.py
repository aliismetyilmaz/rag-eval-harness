"""Load the docs corpus and split it into chunks.

Two chunking strategies are compared:
- "section": one chunk per "## " section, prefixed with the doc title and its
  "Last updated" line so the model can tell old docs from new ones.
- "fixed": naive sliding windows of N words with overlap and no doc header.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

CORPUS_DIR = Path(__file__).resolve().parent.parent / "corpus"


@dataclass(frozen=True)
class Doc:
    id: str
    title: str
    text: str


@dataclass(frozen=True)
class Chunk:
    id: str
    doc_id: str
    text: str


def load_docs(corpus_dir: Path = CORPUS_DIR) -> list[Doc]:
    docs = []
    for path in sorted(corpus_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8").strip()
        title = text.splitlines()[0].lstrip("# ").strip()
        docs.append(Doc(id=path.stem, title=title, text=text))
    return docs


def _header(doc: Doc) -> str:
    """Title plus the 'Last updated' line, if the doc has one."""
    match = re.search(r"^_Last updated: .*_$", doc.text, flags=re.MULTILINE)
    return f"# {doc.title}\n{match.group(0)}" if match else f"# {doc.title}"


def chunk_by_section(docs: list[Doc]) -> list[Chunk]:
    chunks = []
    for doc in docs:
        header = _header(doc)
        parts = re.split(r"^(?=## )", doc.text, flags=re.MULTILINE)
        # parts[0] is the intro before the first "## " heading.
        for n, part in enumerate(parts):
            body = part.strip()
            if n == 0:
                body = body.replace(header, "").replace(f"# {doc.title}", "").strip()
                if not body:
                    continue
            chunks.append(Chunk(id=f"{doc.id}#{n}", doc_id=doc.id, text=f"{header}\n\n{body}"))
    return chunks


def chunk_fixed(docs: list[Doc], size: int = 80, overlap: int = 20) -> list[Chunk]:
    chunks = []
    step = size - overlap
    for doc in docs:
        words = doc.text.split()
        for n, start in enumerate(range(0, max(len(words) - overlap, 1), step)):
            text = " ".join(words[start : start + size])
            chunks.append(Chunk(id=f"{doc.id}#w{n}", doc_id=doc.id, text=text))
    return chunks
