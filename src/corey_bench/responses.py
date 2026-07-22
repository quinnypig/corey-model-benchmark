from __future__ import annotations

from typing import Any

from .protocol import EvalSuite
from .runner import RunStore
from .runner import read_jsonl


def response_records(
    store: RunStore,
    suite: EvalSuite,
    *,
    model_id: str | None = None,
    eval_id: str | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for state in store.recent():
        run_id = state.get("run_id")
        try:
            manifest = store.manifest(run_id)
            evals = {row["id"]: row for row in manifest.get("evals", [])}
            results = store.results(run_id)
            reviews = _preferred_reviews(read_jsonl(store.run_dir(run_id) / "reviews.jsonl"))
        except (OSError, ValueError, KeyError):
            continue
        for result in reversed(results):
            if model_id and result.get("model") != model_id:
                continue
            if eval_id and result.get("eval_id") != eval_id:
                continue
            definition = evals.get(result.get("eval_id"), {})
            turns = result.get("turns") or []
            if turns:
                for turn in turns:
                    records.append(_record(run_id, state, result, definition, turn, reviews.get(str(result.get("attempt_id")))))
            else:
                responses = result.get("responses") or ([result.get("response")] if result.get("response") else [])
                for index, response in enumerate(responses, 1):
                    records.append(
                        _record(run_id, state, result, definition, {"index": index, "response": response, "messages": []}, reviews.get(str(result.get("attempt_id"))))
                    )
    return records


def _record(
    run_id: str,
    state: dict[str, Any],
    result: dict[str, Any],
    definition: dict[str, Any],
    turn: dict[str, Any],
    review: dict[str, Any] | None,
) -> dict[str, Any]:
    messages = turn.get("messages") if isinstance(turn.get("messages"), list) else []
    question = next(
        (str(message.get("content") or "") for message in reversed(messages) if message.get("role") == "user"),
        "",
    )
    if not question:
        prompts = definition.get("prompts") or []
        index = max(0, int(turn.get("index") or 1) - 1)
        question = str(prompts[index] if index < len(prompts) else (prompts[0] if prompts else ""))
    usage = turn.get("usage") if isinstance(turn.get("usage"), dict) else {}
    return {
        "run_id": run_id,
        "run_created_at": state.get("created_at"),
        "attempt_id": result.get("attempt_id"),
        "model": result.get("model"),
        "resolved_model": result.get("resolved_model") or turn.get("resolved_model"),
        "provider": turn.get("provider") or result.get("provider"),
        "eval_id": result.get("eval_id"),
        "eval_title": definition.get("title") or result.get("eval_id"),
        "tier": definition.get("tier"),
        "condition": result.get("condition"),
        "repetition": result.get("repetition"),
        "turn": turn.get("index", 1),
        "question": question,
        "response": turn.get("response", ""),
        "usage": usage,
        "cost_usd": float(usage.get("cost") or 0),
        "finish_reason": turn.get("finish_reason"),
        "grade": result.get("grade", {}),
        "review": review,
        "completed_at": result.get("completed_at"),
    }


def _preferred_reviews(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    model_reviews: dict[str, dict[str, Any]] = {}
    human_reviews: dict[str, dict[str, Any]] = {}
    for row in rows:
        attempt_id = str(row.get("attempt_id") or "")
        if not attempt_id or not isinstance(row.get("score"), (int, float)):
            continue
        target = human_reviews if row.get("reviewer_type", "human") == "human" else model_reviews
        target[attempt_id] = row
    return {**model_reviews, **human_reviews}
