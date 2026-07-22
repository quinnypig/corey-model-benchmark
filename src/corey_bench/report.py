from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return records


def _money(value: float) -> str:
    if value == 0:
        return "$0"
    if value < 0.01:
        return f"${value:.5f}"
    return f"${value:.3f}"


def build_report(run_dir: Path) -> str:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    results = read_jsonl(run_dir / "results.jsonl")
    reviews = read_jsonl(run_dir / "reviews.jsonl")
    models = manifest["models"]
    successful = [row for row in results if row.get("status") == "ok"]

    lines = [
        f"# {manifest['suite']['name']} — run {manifest['run_id']}",
        "",
        f"Suite version: `{manifest['suite']['version']}`  ",
        f"Temperature: `{manifest['temperature']}`  ",
        f"Reasoning: `{manifest.get('reasoning', 'unspecified')}`  ",
        f"Repetitions: `{manifest['repetitions']}`  ",
        "",
        "Automated checks are guardrails, not an overall intelligence score. The human rubric is the primary comparison for nuanced work.",
        "",
        "## Summary",
        "",
        "| Model | Completed | Auto checks | Human score | Median latency | Tokens | Cost |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    review_by_model: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for review in reviews:
        weighted = [(float(item["score"]), float(item["weight"])) for item in review.get("scores", [])]
        if weighted:
            review_by_model[review["model"]].extend(weighted)

    for model in models:
        rows = [row for row in results if row["model"] == model]
        ok = [row for row in rows if row.get("status") == "ok"]
        auto = [row["automated"]["score"] for row in ok if row.get("automated", {}).get("score") is not None]
        weighted_reviews = review_by_model.get(model, [])
        human = (
            sum(score * weight for score, weight in weighted_reviews) / sum(weight for _, weight in weighted_reviews)
            if weighted_reviews
            else None
        )
        latencies = sorted(float(row["latency_seconds"]) for row in ok)
        latency = median(latencies) if latencies else None
        tokens = sum(int(row.get("usage", {}).get("total_tokens") or 0) for row in ok)
        cost = sum(float(row.get("usage", {}).get("cost") or 0) for row in ok)
        auto_text = f"{mean(auto) * 100:.1f}%" if auto else "—"
        lines.append(f"| `{model}` | {len(ok)}/{len(rows)} | {auto_text}")
        suffix = (
            f" | {human:.2f}/5" if human is not None else " | —"
        ) + (f" | {latency:.1f}s" if latency is not None else " | —") + f" | {tokens:,} | {_money(cost)} |"
        lines[-1] += suffix

    lines.extend(["", "## By case", ""])
    case_lookup = {case["id"]: case for case in manifest["cases"]}
    for case_id, case in case_lookup.items():
        lines.extend([f"### {case['title']} (`{case_id}`)", ""])
        for model in models:
            rows = [
                row for row in results
                if row["model"] == model and row["case_id"] == case_id
            ]
            if not rows:
                continue
            scores = [row["automated"]["score"] for row in rows if row.get("status") == "ok" and row["automated"]["score"] is not None]
            failures = sum(row.get("status") != "ok" for row in rows)
            auto_text = f"{mean(scores) * 100:.1f}% auto" if scores else "no auto score"
            if failures:
                auto_text += f", {failures} failed request(s)"
            lines.append(f"- `{model}`: {auto_text}")
        lines.append("")

    if not reviews:
        lines.extend(
            [
                "## Human review",
                "",
                "No reviews yet. Run `corey-bench review " + str(run_dir) + "` for blinded rubric scoring, then regenerate this report.",
                "",
            ]
        )
    guardrail_failures = []
    for row in successful:
        for check in row.get("automated", {}).get("checks", []):
            if not check["passed"]:
                guardrail_failures.append((row, check))
    if guardrail_failures:
        lines.extend(["## Automated failure ledger", ""])
        for row, check in guardrail_failures:
            lines.append(
                f"- `{row['model']}` / `{row['case_id']}` / repetition {row['repetition']}: {check['label']} ({check['observed']})"
            )
        lines.append("")
    errors = [row for row in results if row.get("status") != "ok"]
    if errors:
        lines.extend(["## Request errors", ""])
        for row in errors:
            lines.append(f"- `{row['model']}` / `{row['case_id']}`: {row.get('error', 'unknown error')}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_report(run_dir: Path) -> Path:
    output = run_dir / "report.md"
    output.write_text(build_report(run_dir), encoding="utf-8")
    return output
