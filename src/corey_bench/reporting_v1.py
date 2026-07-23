from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

from .protocol import EvalSuite, load_protocol
from .runner import read_jsonl


def _review_scores(run_dir: Path) -> dict[str, float]:
    model_scores: dict[str, float] = {}
    human_scores: dict[str, float] = {}
    for row in read_jsonl(run_dir / "reviews.jsonl"):
        attempt_id = row.get("attempt_id")
        score = row.get("score")
        if attempt_id and isinstance(score, (int, float)):
            target = human_scores if row.get("reviewer_type", "human") == "human" else model_scores
            target[str(attempt_id)] = max(0.0, min(1.0, float(score)))
    return {**model_scores, **human_scores}


def _review_costs(run_dir: Path) -> dict[str, float]:
    costs: dict[str, float] = defaultdict(float)
    for row in read_jsonl(run_dir / "reviews.jsonl"):
        if row.get("reviewer_type") == "model" and row.get("attempt_id"):
            costs[str(row["attempt_id"])] += float(row.get("cost_usd") or row.get("usage", {}).get("cost") or 0)
    return costs


def _attempt_score(row: dict[str, Any], reviews: dict[str, float]) -> tuple[float | None, bool]:
    auto = row.get("grade", {}).get("score")
    human = reviews.get(str(row.get("attempt_id")))
    needs_human = bool(row.get("grade", {}).get("human_required"))
    if human is not None and isinstance(auto, (int, float)):
        return (float(auto) + human) / 2, True
    if human is not None:
        return human, True
    if needs_human:
        return None, False
    return (float(auto), True) if isinstance(auto, (int, float)) else (None, False)


def _aggregate(values: list[float], policy: str) -> float | None:
    if not values:
        return None
    if policy in {"min", "all_pass"}:
        return min(values)
    if policy == "max":
        return max(values)
    return mean(values)


def _job_key(row: dict[str, Any]) -> tuple[str, str, int] | None:
    try:
        return (
            str(row["eval_id"]),
            str(row.get("condition") or "weights-only"),
            int(row.get("repetition") or 1),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _suite_completion(
    manifest: dict[str, Any],
    results: list[dict[str, Any]],
    model: str,
    suite: EvalSuite,
) -> dict[str, Any]:
    required: set[tuple[str, str, int]] = set()
    for definition in suite.evals:
        conditions = list(definition.conditions)
        if definition.id in {"2.1", "2.2"}:
            conditions.append("agentic")
        for condition in conditions:
            for repetition in range(1, definition.repetitions + 1):
                required.add((definition.id, condition, repetition))

    scheduled = {
        key
        for row in manifest.get("jobs", [])
        if row.get("model") == model and (key := _job_key(row)) is not None
    }
    completed = {
        key
        for row in results
        if row.get("model") == model and (key := _job_key(row)) is not None
    }
    missing_scheduled = required - scheduled
    missing_completed = required - completed
    protocol_version = str(manifest.get("suite", {}).get("version") or "")
    protocol_current = protocol_version == suite.version
    # A prompt-only protocol revision does not erase a completed card. The
    # required current job matrix is the compatibility boundary: if a revision
    # adds an eval, treatment, or repetition, the missing key invalidates the
    # older run automatically.
    full_suite_scheduled = not missing_scheduled
    full_suite_complete = full_suite_scheduled and not missing_completed
    return {
        "protocol_version": protocol_version,
        "protocol_current": protocol_current,
        "required_attempts": len(required),
        "scheduled_required_attempts": len(required & scheduled),
        "completed_required_attempts": len(required & completed),
        "missing_test_count": len(missing_completed),
        "missing_tests": [
            f"{eval_id}:{condition}:r{repetition}"
            for eval_id, condition, repetition in sorted(missing_completed)
        ],
        "missing_eval_ids": sorted(
            {eval_id for eval_id, _condition, _repetition in missing_completed},
            key=lambda value: tuple(int(part) for part in value.split(".")),
        ),
        "full_suite_scheduled": full_suite_scheduled,
        "full_suite_complete": full_suite_complete,
    }


def build_report_data(run_dir: Path, suite: EvalSuite | None = None) -> dict[str, Any]:
    suite = suite or load_protocol()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    results = read_jsonl(run_dir / "results.jsonl")
    reviews = _review_scores(run_dir)
    review_costs = _review_costs(run_dir)
    models = manifest.get("models", manifest.get("config", {}).get("models", []))
    selected_ids = [item["id"] for item in manifest.get("evals", [])]
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        grouped[(row.get("model", ""), row.get("eval_id", ""), row.get("condition", "weights-only"))].append(row)

    model_rows = []
    for model in models:
        attempts = [row for row in results if row.get("model") == model]
        completion = _suite_completion(manifest, results, model, suite)
        eval_rows = []
        weighted_points = 0.0
        resolved_weight = 0.0
        for eval_id in selected_ids:
            definition = suite.get(eval_id)
            conditions = [condition for condition in definition.conditions if (model, eval_id, condition) in grouped]
            if (model, eval_id, "agentic") in grouped:
                conditions.append("agentic")
            if not conditions:
                conditions = ["weights-only"]
            for condition in conditions:
                rows = grouped.get((model, eval_id, condition), [])
                scores = []
                resolved_flags = []
                for row in rows:
                    score, resolved = _attempt_score(row, reviews)
                    if score is not None:
                        scores.append(score)
                    resolved_flags.append(resolved)
                score = _aggregate(scores, definition.aggregation)
                complete = bool(rows) and len(rows) == sum(
                    1 for job in manifest.get("jobs", [])
                    if job["model"] == model and job["eval_id"] == eval_id and job["condition"] == condition
                )
                resolved = complete and bool(resolved_flags) and all(resolved_flags)
                cost = sum(float(row.get("cost_usd") or row.get("usage", {}).get("cost") or 0) for row in rows)
                verdicts = [row.get("grade", {}).get("verdict") for row in rows if row.get("grade", {}).get("verdict")]
                eval_rows.append(
                    {
                        "eval_id": eval_id, "title": definition.title, "tier": definition.tier,
                        "condition": condition, "score": score, "resolved": resolved, "attempts": len(rows),
                        "cost_usd": cost, "verdict": verdicts[-1] if verdicts else "Not run",
                        "weight": definition.weight if condition == "weights-only" else 0,
                        "frontier": definition.tier == 7,
                    }
                )
                if condition == "weights-only" and definition.tier < 7 and definition.status != "alternate" and resolved and score is not None:
                    weighted_points += score * definition.weight
                    resolved_weight += definition.weight
        inference_cost = sum(float(row.get("cost_usd") or row.get("usage", {}).get("cost") or 0) for row in attempts)
        review_cost = sum(review_costs.get(str(row.get("attempt_id")), 0) for row in attempts)
        total_cost = inference_cost + review_cost
        latencies = [float(row.get("latency_seconds", 0)) for row in attempts if row.get("status") == "ok"]
        total_tokens = sum(int(row.get("usage", {}).get("total_tokens") or 0) for row in attempts)
        resolved_score = weighted_points if abs(resolved_weight - 100) < 1e-9 else None
        final_score = resolved_score if completion["full_suite_complete"] else None
        model_rows.append(
            {
                "model": model, "evals": eval_rows, "weighted_points": weighted_points,
                "resolved_weight": resolved_weight, "final_score": final_score,
                **completion,
                "rankable": final_score is not None,
                "total_cost": total_cost,
                "inference_cost": inference_cost, "review_cost": review_cost,
                "dollars_per_point": total_cost / final_score if final_score and final_score > 0 else None,
                "median_latency": median(latencies) if latencies else None, "total_tokens": total_tokens,
                "completed_attempts": len(attempts),
                "failed_attempts": sum(row.get("status") != "ok" for row in attempts),
            }
        )
    return {
        "run_id": manifest["run_id"], "created_at": manifest.get("created_at"),
        "suite": manifest.get("suite", {}), "models": model_rows,
        "expected_jobs": manifest.get("expected_jobs", len(manifest.get("jobs", []))),
        "completed_jobs": len(results),
    }


def _money(value: float | None) -> str:
    if value is None:
        return "—"
    if value < 0.01:
        return f"${value:.5f}"
    return f"${value:.2f}"


def report_markdown(data: dict[str, Any]) -> str:
    lines = [
        f"# Quinnferno — run {data['run_id']}", "",
        "## Headline metrics", "",
        "| Model | Suite score | Total suite cost | Dollars per point | Coverage |",
        "|---|---:|---:|---:|---:|",
    ]
    for model in data["models"]:
        score = (
            f"{model['final_score']:.1f}/100"
            if model["rankable"]
            else f"Not ranked ({model['weighted_points']:.1f}/{model['resolved_weight']:.0f} provisional)"
        )
        coverage = f"{model['completed_required_attempts']}/{model['required_attempts']} required attempts"
        lines.append(f"| `{model['model']}` | {score} | {_money(model['total_cost'])} | {_money(model['dollars_per_point'])} | {coverage} |")
    for model in data["models"]:
        lines.extend(["", f"## {model['model']}", "", "| Eval | Score | Cost/run | Verdict |", "|---|---:|---:|---|"])
        for row in model["evals"]:
            label = row["title"] + (f" [{row['condition']}]" if row["condition"] != "weights-only" else "")
            score = f"{row['score'] * 100:.1f}%" if row["score"] is not None else "Pending review"
            per_run = row["cost_usd"] / row["attempts"] if row["attempts"] else None
            lines.append(f"| {row['eval_id']} {label} | {score} | {_money(per_run)} | {row['verdict']} |")
    lines.extend(["", "Models are not ranked until every required current-protocol test attempt is recorded and every weighted human or deferred judgment is resolved. Tier 7 is reported separately and never contributes to /100.", ""])
    return "\n".join(lines)


def write_v1_reports(run_dir: Path, suite: EvalSuite | None = None) -> dict[str, Path]:
    data = build_report_data(run_dir, suite)
    json_path = run_dir / "report.json"
    markdown_path = run_dir / "report.md"
    json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown_path.write_text(report_markdown(data), encoding="utf-8")
    return {"json": json_path, "markdown": markdown_path}
