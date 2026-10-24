# RAG Eval Harness

**How do you know an AI support assistant is telling customers the truth?**

This project builds a docs-grounded support assistant four different ways and measures
each one on accuracy, hallucination, citation quality, cost, and latency. The goal is not
the chatbot. It's the evaluation system that tells a product team which version is safe
to ship and why.

## Problem

Teams ship AI assistants on top of their help docs, then find out from customers that
the bot:

- invents answers to questions the docs don't cover,
- quotes outdated pages (a deprecated API, an old plan limit),
- cites sources that don't actually support what it said.

"It looked good in the demo" is not a launch criterion. Product teams need a repeatable
scorecard they can run on every prompt, model, or retrieval change.

## Users

- **Product managers** deciding whether an assistant is ready to launch, and which
  tradeoff (quality, cost, latency) to make.
- **Engineers** changing prompts, chunking, or models, who need to know if they made it
  better or worse.
- **Support leads** who need to trust that the bot says "I don't know" instead of
  guessing.

## Success metrics

| Metric | What it answers | Launch bar (example) |
|---|---|---|
| **Answer accuracy** | Is the answer right? (LLM judge vs. a reference answer; partial = 0.5) | ≥ 90% |
| **Hallucination rate** | How often does it confidently say something wrong or unsupported? | ≤ 3% |
| **Abstained correctly** | Does it say "the docs don't cover this" exactly when it should? | ≥ 95% |
| **Cited the right doc** | Do its citations point at the doc that holds the answer? | ≥ 90% |
| **Retrieval recall** | Did search even find the right doc? Separates retrieval bugs from answer bugs. | ≥ 95% |
| **Cost / question** and **latency** | What does quality cost? | Depends on volume |

The launch bars are examples of what I'd propose to a team, not industry standards.

## What's being tested

**The knowledge base** ([`corpus/`](corpus/)): ten help-center pages for *Relaykit*, a
fictional help-desk SaaS. Fictional docs mean the right answers are known exactly, the
model can't answer from memory, and the docs can include deliberate traps:

- a **deprecated v1 API page** that contradicts the current v2 page,
- a **changelog** that overrides older pages (a raised plan limit, a new channel on Starter),
- pricing that is **intentionally unpublished** (Enterprise is "contact sales").

**The questions** ([`data/questions.jsonl`](data/questions.jsonl)): 32 questions in four
types: single-doc lookups, multi-doc reasoning, conflicting/outdated sources, and
questions the docs can't answer.

**The four setups:**

| Setup | How it answers |
|---|---|
| `closed_book` | No docs. Baseline showing what the model makes up on its own. |
| `rag_fixed` | Keyword search (BM25) over naive 80-word chunks, top 4 passed to the model. |
| `rag_section` | Same search, but chunks follow the doc's sections and carry the page title and "last updated" date. |
| `full_context` | No search. All docs go into the prompt, with prompt caching. |

## Architecture

```
questions.jsonl ──► pipeline ──► Claude (structured output: answerable, answer, citations)
                      │                         │
        corpus/ ──► chunker ──► BM25 search     ▼
                                          grading
                         ┌──────────────────┴──────────────────┐
                 deterministic checks                     LLM judge
           abstention, citation validity,          correctness vs. reference,
           right doc cited, retrieval recall      groundedness vs. cited text
                         └──────────────────┬──────────────────┘
                                            ▼
                               results/latest.md scorecard
```

Code lives in [`rag_eval/`](rag_eval/): `corpus.py` (loading and chunking),
`retrieval.py` (BM25), `pipelines.py` (the four setups and the answer prompt),
`grading.py` (scoring and judge), `run.py` (runs everything, writes the scorecard).

## Key decisions and tradeoffs

- **Deterministic checks first, LLM judge only where needed.** Whether the model
  abstained and whether its citations exist are checked in code: free, fast, and
  impossible to argue with. The judge is only called when a real answer has to be
  compared with a reference, which also cuts eval cost.
- **Answering an unanswerable question counts as a hallucination automatically.** For a
  support bot, a confident wrong answer is worse than "I don't know," so the scoring is
  built to make that visible rather than averaging it away.
- **Structured output instead of parsing prose.** The model returns
  `{answerable, answer, citations}` as validated JSON, so "did it abstain?" is a field,
  not a regex.
- **BM25 instead of embeddings.** Zero extra infrastructure, fully inspectable, and good
  enough to show how retrieval quality moves the other metrics. Embeddings are the next
  experiment, not a default.
- **Fictional corpus.** Trades realism for control: every answer is known, and the
  model can't lean on training data.
- **Long context as a real option.** With a small corpus, putting everything in the
  prompt (cached) may beat retrieval. The eval measures whether it does, and at what cost.

## Results

Run `python -m rag_eval.run` to produce the scorecard in
[`results/latest.md`](results/latest.md).

_Results will be added after the first full run._

## Run it yourself

Requires Python 3.10+ and an Anthropic API key.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Your key is read from the environment and is never stored in code.
export ANTHROPIC_API_KEY=...      # or copy .env.example to .env (git-ignored)

pytest                             # offline tests, no API key needed
python -m rag_eval.run --limit 3   # quick smoke test
python -m rag_eval.run             # full run: 32 questions x 4 setups
```

Options: `--pipelines rag_section,full_context` runs a subset; `--model` and
`--judge-model` (or the `RAG_MODEL` / `JUDGE_MODEL` env vars) swap models. The default
is `claude-opus-5-5` for both.

## What I'd do next

1. **Embedding or hybrid retrieval**, measured against BM25 on the same scorecard.
2. **Judge calibration**: hand-label a sample of answers and measure judge agreement
   before trusting judge scores for launch decisions.
3. **Cheaper models**: rerun with Sonnet and Haiku to find the cheapest setup that
   still clears the launch bars.
4. **CI gate**: run on every pull request and fail the build if hallucination rate
   regresses.
5. **Real traffic**: replace synthetic questions with anonymized support tickets.

