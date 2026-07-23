from __future__ import annotations

from collections import defaultdict
from typing import Any

from .protocol import EvalSuite
from .reporting_v1 import build_report_data
from .runner import RunStore


TERMINAL_STATUSES = {
    "completed",
    "execution_errors",
    "completed_with_errors",  # legacy state, migrated on queue startup
    "cancelled",
    "budget_exhausted",
}


def build_model_cards(
    store: RunStore,
    suite: EvalSuite,
    catalog: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build a historical model index from canonical run receipts."""
    metadata = {row["id"]: row for row in (catalog or []) if isinstance(row.get("id"), str)}
    cards: dict[str, dict[str, Any]] = {}

    for state in store.recent():
        run_id = str(state.get("run_id") or "")
        if not run_id:
            continue
        try:
            report = build_report_data(store.run_dir(run_id), suite)
        except (OSError, ValueError, KeyError):
            continue
        for model_report in report["models"]:
            model_id = model_report["model"]
            card = cards.setdefault(
                model_id,
                {
                    "id": model_id,
                    "name": metadata.get(model_id, {}).get("name") or model_id,
                    "metadata": metadata.get(model_id, {}),
                    "suite_version": suite.version,
                    "runs": [],
                },
            )
            resolved_weight = float(model_report.get("resolved_weight") or 0)
            weighted_points = float(model_report.get("weighted_points") or 0)
            provisional_score = weighted_points / resolved_weight * 100 if resolved_weight else None
            run = {
                "run_id": run_id,
                "created_at": state.get("created_at") or report.get("created_at"),
                "status": state.get("status", "unknown"),
                "completed_jobs": state.get("completed_jobs", 0),
                "expected_jobs": state.get("expected_jobs", report.get("expected_jobs", 0)),
                "score_percent": model_report.get("final_score") if model_report.get("rankable") else None,
                "provisional_score_percent": provisional_score,
                "weighted_points": weighted_points,
                "resolved_weight": resolved_weight,
                "final_score": model_report.get("final_score"),
                "rankable": bool(model_report.get("rankable")),
                "full_suite_complete": bool(model_report.get("full_suite_complete")),
                "full_suite_scheduled": bool(model_report.get("full_suite_scheduled")),
                "protocol_current": bool(model_report.get("protocol_current")),
                "protocol_version": model_report.get("protocol_version"),
                "required_attempts": int(model_report.get("required_attempts") or 0),
                "scheduled_required_attempts": int(model_report.get("scheduled_required_attempts") or 0),
                "completed_required_attempts": int(model_report.get("completed_required_attempts") or 0),
                "missing_test_count": int(model_report.get("missing_test_count") or 0),
                "missing_tests": model_report.get("missing_tests", []),
                "missing_eval_ids": model_report.get("missing_eval_ids", []),
                "total_cost": float(model_report.get("total_cost") or 0),
                "dollars_per_point": model_report.get("dollars_per_point"),
                "median_latency": model_report.get("median_latency"),
                "total_tokens": int(model_report.get("total_tokens") or 0),
                "completed_attempts": int(model_report.get("completed_attempts") or 0),
                "failed_attempts": int(model_report.get("failed_attempts") or 0),
                "benchmark_failed_attempts": int(model_report.get("benchmark_failed_attempts") or 0),
                "execution_error_attempts": int(model_report.get("execution_error_attempts") or 0),
                "evals": model_report.get("evals", []),
                "terminal": state.get("status") in TERMINAL_STATUSES,
            }
            card["runs"].append(run)

    for card in cards.values():
        card["runs"].sort(key=lambda row: str(row.get("created_at") or ""), reverse=True)
        card["latest_run"] = card["runs"][0]
        # A complete current-protocol run always outranks partial evidence.
        card["representative_run"] = max(
            card["runs"],
            key=lambda row: (
                row["rankable"],
                row["full_suite_complete"],
                row["protocol_current"],
                row["completed_required_attempts"],
                row["resolved_weight"],
                row["completed_attempts"],
                str(row.get("created_at") or ""),
            ),
        )
        card["run_count"] = len(card["runs"])
        card["total_cost"] = sum(row["total_cost"] for row in card["runs"])
        card["total_attempts"] = sum(row["completed_attempts"] for row in card["runs"])
        card["failed_attempts"] = sum(row["failed_attempts"] for row in card["runs"])
        card["execution_error_attempts"] = sum(row["execution_error_attempts"] for row in card["runs"])
        card["total_tokens"] = sum(row["total_tokens"] for row in card["runs"])
        card["valid"] = card["representative_run"]["full_suite_complete"]
        card["rankable"] = card["representative_run"]["rankable"]
        card["eval_profile"] = _eval_profile(card["representative_run"]["evals"])

    return sorted(
        cards.values(),
        key=lambda card: (
            not card["rankable"],
            -(card["representative_run"]["score_percent"] or 0),
            -card["representative_run"]["completed_required_attempts"],
            card["name"].casefold(),
        ),
    )


def _eval_profile(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("eval_id") or "")].append(row)
    profile = []
    for eval_id, eval_rows in grouped.items():
        first = eval_rows[0]
        profile.append(
            {
                "eval_id": eval_id,
                "title": first.get("title"),
                "tier": first.get("tier"),
                "scores": {row.get("condition"): row.get("score") for row in eval_rows},
                "attempts": sum(int(row.get("attempts") or 0) for row in eval_rows),
                "cost_usd": sum(float(row.get("cost_usd") or 0) for row in eval_rows),
                "verdict": next(
                    (row.get("verdict") for row in eval_rows if row.get("condition") == "weights-only"),
                    first.get("verdict", "Not run"),
                ),
            }
        )
    return sorted(profile, key=lambda row: tuple(int(part) for part in row["eval_id"].split(".")))
