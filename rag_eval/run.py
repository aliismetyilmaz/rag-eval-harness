"""Run the eval: every question through every pipeline, then grade and summarize.

    python -m rag_eval.run                 # full run
    python -m rag_eval.run --limit 3       # quick smoke test
    python -m rag_eval.run --pipelines rag_section,full_context
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from . import grading, llm
from .pipelines import Pipeline, answer, build_pipelines

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS = ROOT / "data" / "questions.jsonl"
RESULTS = ROOT / "results"


def load_questions(limit: int | None) -> list[dict]:
    lines = QUESTIONS.read_text(encoding="utf-8").splitlines()
    questions = [json.loads(line) for line in lines if line.strip()]
    return questions[:limit] if limit else questions


def run_one(pipeline: Pipeline, question: dict, model: str, judge_model: str) -> dict:
    sources, call = answer(pipeline, question["question"], model)
    ans = call.parsed
    verdict, judge_call = None, None
    if grading.needs_judge(question, ans):
        cited_ids = set(ans.citations)
        cited = [c for c in sources if c.id in cited_ids]
        verdict, judge_call = grading.judge(question, ans.answer, cited, judge_model)
    metrics = grading.score(question, ans, sources, verdict)
    return {
        "pipeline": pipeline.name,
        "question_id": question["id"],
        "category": question["category"],
        "question": question["question"],
        "answer": ans.model_dump() if ans else None,
        "retrieved": [c.id for c in sources] if pipeline.index else [],
        "verdict": verdict.model_dump() if verdict else None,
        "metrics": metrics,
        "cost_usd": call.cost_usd,
        "judge_cost_usd": judge_call.cost_usd if judge_call else 0.0,
        "latency_s": call.latency_s,
        "cache_read_tokens": call.cache_read_tokens,
    }


def mean(values) -> float | None:
    values = [float(v) for v in values if v is not None]
    return statistics.mean(values) if values else None


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def summarize(rows: list[dict], pipelines: dict[str, Pipeline]) -> str:
    out = [
        "| Setup | Answer accuracy | Hallucination rate | Abstained correctly | "
        "Cited the right doc | Retrieval recall | Cost / question | Median latency |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, pipeline in pipelines.items():
        mine = [r for r in rows if r["pipeline"] == name]
        if not mine:
            continue
        m = [r["metrics"] for r in mine]
        out.append(
            f"| `{name}` | {pct(mean(x['correctness'] for x in m))} "
            f"| {pct(mean(x['hallucinated'] for x in m))} "
            f"| {pct(mean(x.get('abstention_correct') for x in m))} "
            f"| {pct(mean(x.get('cited_expected_doc') for x in m))} "
            f"| {pct(mean(x['retrieval_recall'] for x in m))} "
            f"| ${mean(r['cost_usd'] for r in mine):.4f} "
            f"| {statistics.median(r['latency_s'] for r in mine):.1f}s |"
        )

    out += ["", "**Answer accuracy by question type**", ""]
    categories = sorted({r["category"] for r in rows})
    out.append("| Setup | " + " | ".join(categories) + " |")
    out.append("|---|" + "---|" * len(categories))
    for name in pipelines:
        mine = [r for r in rows if r["pipeline"] == name]
        if not mine:
            continue
        cells = [
            pct(mean(r["metrics"]["correctness"] for r in mine if r["category"] == cat))
            for cat in categories
        ]
        out.append(f"| `{name}` | " + " | ".join(cells) + " |")

    refusals = sum(r["metrics"]["refused"] for r in rows)
    total = sum(r["cost_usd"] + r["judge_cost_usd"] for r in rows)
    out += [
        "",
        f"Total run cost: ${total:.2f} (answers + judge). Model refusals: {refusals}.",
        "",
        "Setups: " + "; ".join(f"`{p.name}` = {p.description}" for p in pipelines.values()),
    ]
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="only the first N questions")
    parser.add_argument("--pipelines", help="comma-separated subset of pipelines")
    parser.add_argument("--model", default=llm.default_model("RAG_MODEL"))
    parser.add_argument("--judge-model", default=llm.default_model("JUDGE_MODEL"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    pipelines = build_pipelines()
    if args.pipelines:
        wanted = args.pipelines.split(",")
        unknown = set(wanted) - set(pipelines)
        if unknown:
            parser.error(f"unknown pipelines: {', '.join(sorted(unknown))}")
        pipelines = {k: v for k, v in pipelines.items() if k in wanted}
    questions = load_questions(args.limit)

    jobs = [(p, q) for p in pipelines.values() for q in questions]
    print(f"Running {len(jobs)} answers with {args.model} (judge: {args.judge_model})...")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_one, p, q, args.model, args.judge_model) for p, q in jobs]
        rows = []
        for i, future in enumerate(futures, 1):
            rows.append(future.result())
            print(f"  {i}/{len(jobs)}", end="\r", file=sys.stderr)

    run_dir = RESULTS / datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir.mkdir(parents=True)
    with open(run_dir / "answers.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    summary = summarize(rows, pipelines)
    header = f"# Results\n\nModel: `{args.model}` · Judge: `{args.judge_model}` · Questions: {len(questions)}\n\n"
    (run_dir / "summary.md").write_text(header + summary + "\n", encoding="utf-8")
    (RESULTS / "latest.md").write_text(header + summary + "\n", encoding="utf-8")
    print("\n" + summary)
    print(f"\nSaved to {run_dir.relative_to(ROOT)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
