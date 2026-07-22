from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .openrouter import OpenRouterClient, OpenRouterError
from .report import read_jsonl, write_report
from .scoring import score_response
from .suite import Case, SuiteError, load_suite


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_dotenv(path: Path = Path(".env")) -> None:
    """Load a small, conventional .env file without adding a runtime dependency."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key.replace("_", "a").isalnum() or key[:1].isdigit():
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def _select_cases(cases: list[Case], ids: list[str], categories: list[str]) -> list[Case]:
    selected = cases
    if ids:
        wanted = set(ids)
        selected = [case for case in selected if case.id in wanted]
        missing = wanted - {case.id for case in selected}
        if missing:
            raise ValueError("Unknown case id(s): " + ", ".join(sorted(missing)))
    if categories:
        wanted_categories = set(categories)
        selected = [case for case in selected if case.category in wanted_categories]
        missing = wanted_categories - {case.category for case in cases}
        if missing:
            raise ValueError("Unknown category/categories: " + ", ".join(sorted(missing)))
    if not selected:
        raise ValueError("No cases selected")
    return selected


def cmd_list(args: argparse.Namespace) -> int:
    suite = load_suite(args.suite)
    print(f"{suite.name} v{suite.version}: {suite.description}\n")
    for case in suite.cases:
        print(f"{case.id:28} {case.category:22} {case.title}")
    print(f"\n{len(suite.cases)} cases")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    suite = load_suite(args.suite)
    possible = sum(sum(check.points for check in case.checks) for case in suite.cases)
    rubrics = sum(len(case.rubric) for case in suite.cases)
    print(f"Valid: {suite.path} ({len(suite.cases)} cases, {possible:g} automated points, {rubrics} rubric criteria)")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    _load_dotenv()
    suite = load_suite(args.suite)
    cases = _select_cases(suite.cases, args.case, args.category)
    models = args.model or []
    if not models:
        raise ValueError("at least one --model is required; repeat --model to compare exact OpenRouter model IDs")
    if not args.dry_run and not os.environ.get("OPENROUTER_API_KEY"):
        raise ValueError("OPENROUTER_API_KEY is required (or use --dry-run)")

    client = None
    if not args.dry_run:
        client = OpenRouterClient(os.environ["OPENROUTER_API_KEY"], timeout=args.timeout, attempts=args.attempts)
        client.require_models_available(models)

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    run_dir = Path(args.output) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "run_id": run_id,
        "created_at": _utc_now(),
        "suite": {
            "name": suite.name,
            "version": suite.version,
            "description": suite.description,
            "source": str(suite.path),
        },
        "models": models,
        "temperature": args.temperature,
        "seed": args.seed,
        "reasoning": args.reasoning,
        "repetitions": args.repetitions,
        "cases": [
            {
                "id": case.id,
                "title": case.title,
                "category": case.category,
                "prompt": case.prompt,
                "system": case.system,
                "checks": [asdict(check) for check in case.checks],
                "rubric": [asdict(rubric) for rubric in case.rubric],
                "reference_notes": case.reference_notes,
            }
            for case in cases
        ],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    results_path = run_dir / "results.jsonl"

    jobs = [(model, case, repetition) for repetition in range(args.repetitions) for case in cases for model in models]
    print(f"Run {run_id}: {len(jobs)} requests across {len(models)} model(s) and {len(cases)} case(s)")
    if args.dry_run:
        for model, case, repetition in jobs:
            print(f"DRY RUN  {model}  {case.id}  repetition={repetition + 1}")
        write_report(run_dir)
        print(f"Wrote {run_dir}")
        return 0

    assert client is not None
    for index, (model, case, repetition) in enumerate(jobs, 1):
        print(f"[{index}/{len(jobs)}] {model} / {case.id} ...", flush=True)
        started_at = _utc_now()
        started = time.monotonic()
        record: dict[str, Any] = {
            "run_id": run_id,
            "case_id": case.id,
            "category": case.category,
            "model": model,
            "repetition": repetition + 1,
            "started_at": started_at,
        }
        try:
            completion = client.complete(
                model=model,
                system=case.system,
                prompt=case.prompt,
                max_tokens=case.max_tokens,
                temperature=args.temperature,
                seed=(args.seed + repetition) if args.seed is not None else None,
                reasoning=args.reasoning,
            )
            record.update(
                {
                    "status": "ok",
                    "response": completion.text,
                    "response_id": completion.response_id,
                    "provider": completion.provider,
                    "raw_model": completion.raw_model,
                    "usage": completion.usage,
                    "reasoning": completion.reasoning,
                    "finish_reason": completion.finish_reason,
                    "native_finish_reason": completion.native_finish_reason,
                    "automated": score_response(case.checks, completion.text),
                }
            )
        except OpenRouterError as exc:
            record.update({"status": "error", "error": str(exc), "response": "", "usage": {}})
        record["latency_seconds"] = round(time.monotonic() - started, 3)
        _append_jsonl(results_path, record)
        if record["status"] == "error":
            print(f"  ERROR: {record['error']}", file=sys.stderr)
        else:
            auto_score = record["automated"]["score"]
            score_text = f"{auto_score * 100:.1f}%" if auto_score is not None else "not scored"
            print(f"  completed in {record['latency_seconds']:.1f}s; automated guardrails: {score_text}")
    report_path = write_report(run_dir)
    print(f"Report: {report_path}")
    failures = sum(row.get("status") != "ok" for row in read_jsonl(results_path))
    return 1 if failures else 0


def _blind_id(run_id: str, model: str, case_id: str, repetition: int) -> str:
    digest = hashlib.sha256(f"{run_id}\0{model}\0{case_id}\0{repetition}".encode()).hexdigest()
    return digest[:8].upper()


def cmd_review(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    results = [row for row in read_jsonl(run_dir / "results.jsonl") if row.get("status") == "ok"]
    review_path = run_dir / "reviews.jsonl"
    reviewed = {
        (row["model"], row["case_id"], row["repetition"])
        for row in read_jsonl(review_path)
    }
    case_lookup = {case["id"]: case for case in manifest["cases"]}
    pending = [row for row in results if (row["model"], row["case_id"], row["repetition"]) not in reviewed]
    random.Random(manifest["run_id"]).shuffle(pending)
    if not pending:
        print("Nothing left to review.")
        write_report(run_dir)
        return 0

    print(f"{len(pending)} blinded responses to review. Enter 1–5, 's' to skip, or 'q' to stop.\n")
    for index, row in enumerate(pending, 1):
        case = case_lookup[row["case_id"]]
        blind = _blind_id(manifest["run_id"], row["model"], row["case_id"], row["repetition"])
        print("=" * 78)
        print(f"[{index}/{len(pending)}] {case['title']} — response {blind}")
        print("\nPROMPT\n" + case["prompt"])
        print("\nRESPONSE\n" + row["response"])
        print("\nRUBRIC")
        scores = []
        skipped = False
        for rubric in case.get("rubric", []):
            while True:
                answer = input(f"{rubric['name']} — {rubric['description']} [1-5/s/q]: ").strip().lower()
                if answer == "q":
                    write_report(run_dir)
                    print(f"Stopped. Progress saved in {review_path}")
                    return 0
                if answer == "s":
                    skipped = True
                    break
                if answer in {"1", "2", "3", "4", "5"}:
                    scores.append({**rubric, "score": int(answer)})
                    break
                print("Please enter 1, 2, 3, 4, 5, s, or q.")
            if skipped:
                break
        if skipped:
            continue
        notes = input("Optional notes: ").strip()
        _append_jsonl(
            review_path,
            {
                "reviewed_at": _utc_now(),
                "blind_id": blind,
                "model": row["model"],
                "case_id": row["case_id"],
                "repetition": row["repetition"],
                "scores": scores,
                "notes": notes,
            },
        )
    report_path = write_report(run_dir)
    print(f"Review complete. Report: {report_path}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    path = write_report(Path(args.run_dir))
    print(path)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="corey-bench", description="Run Corey Quinn's personal model benchmark")
    parser.add_argument("--suite", help="path to benchmark suite JSON")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="list benchmark cases")
    list_parser.set_defaults(func=cmd_list)

    validate_parser = subparsers.add_parser("validate", help="validate the suite")
    validate_parser.set_defaults(func=cmd_validate)

    run_parser = subparsers.add_parser("run", help="run models through OpenRouter")
    run_parser.add_argument("--model", action="append", help="exact OpenRouter model ID; required, repeat for comparisons")
    run_parser.add_argument("--case", action="append", default=[], help="only run this case ID; repeatable")
    run_parser.add_argument("--category", action="append", default=[], help="only run this category; repeatable")
    run_parser.add_argument("--repetitions", type=int, default=1)
    run_parser.add_argument("--temperature", type=float, default=0.1)
    run_parser.add_argument("--seed", type=int, default=8675309)
    run_parser.add_argument(
        "--reasoning",
        choices=["off", "on", "minimal", "low", "medium", "high", "xhigh"],
        default="on",
        help="OpenRouter reasoning configuration (default: on)",
    )
    run_parser.add_argument("--output", default="runs")
    run_parser.add_argument("--run-id")
    run_parser.add_argument("--timeout", type=float, default=300)
    run_parser.add_argument("--attempts", type=int, default=4)
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.set_defaults(func=cmd_run)

    review_parser = subparsers.add_parser("review", help="blindly score model outputs with the human rubric")
    review_parser.add_argument("run_dir")
    review_parser.set_defaults(func=cmd_review)

    report_parser = subparsers.add_parser("report", help="regenerate a run report")
    report_parser.add_argument("run_dir")
    report_parser.set_defaults(func=cmd_report)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if getattr(args, "repetitions", 1) < 1:
            raise ValueError("--repetitions must be at least 1")
        return int(args.func(args))
    except (OpenRouterError, SuiteError, ValueError, OSError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    sys.exit(main())
