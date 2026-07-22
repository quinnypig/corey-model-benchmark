from __future__ import annotations

import json
import re
from dataclasses import asdict
from typing import Any

from .suite import Check


def _evaluate(check: Check, response: str) -> tuple[bool, str]:
    folded = response.casefold()
    if check.type == "contains":
        passed = str(check.value).casefold() in folded
        return passed, f"contains {check.value!r}"
    if check.type == "contains_any":
        values = check.values or ([str(check.value)] if check.value is not None else [])
        passed = any(value.casefold() in folded for value in values)
        return passed, "contains any of " + ", ".join(repr(value) for value in values)
    if check.type == "not_contains":
        values = check.values or ([str(check.value)] if check.value is not None else [])
        passed = all(value.casefold() not in folded for value in values)
        return passed, "does not contain " + ", ".join(repr(value) for value in values)
    if check.type == "regex":
        flags = 0
        if "i" in check.flags:
            flags |= re.IGNORECASE
        if "m" in check.flags:
            flags |= re.MULTILINE
        if "s" in check.flags:
            flags |= re.DOTALL
        passed = re.search(str(check.value), response, flags) is not None
        return passed, f"matches /{check.value}/"
    if check.type in {"min_words", "max_words"}:
        word_count = len(re.findall(r"\b[\w'-]+\b", response))
        limit = int(check.value)
        if check.type == "min_words":
            return word_count >= limit, f"word count {word_count} >= {limit}"
        return word_count <= limit, f"word count {word_count} <= {limit}"
    if check.type == "json_valid":
        try:
            json.loads(response)
            return True, "is valid JSON"
        except json.JSONDecodeError as exc:
            return False, f"invalid JSON: {exc.msg}"
    if check.type == "json_keys_exact":
        try:
            value = json.loads(response)
        except json.JSONDecodeError as exc:
            return False, f"invalid JSON: {exc.msg}"
        expected = set(check.values)
        observed = set(value) if isinstance(value, dict) else set()
        return observed == expected, f"top-level keys {sorted(observed)} == {sorted(expected)}"
    if check.type in {"json_field_type", "json_field_equals"}:
        try:
            parsed = json.loads(response)
        except json.JSONDecodeError as exc:
            return False, f"invalid JSON: {exc.msg}"
        if not isinstance(parsed, dict) or check.value not in parsed:
            return False, f"missing top-level field {check.value!r}"
        observed = parsed[check.value]
        if check.type == "json_field_equals":
            return observed == check.expected, f"{check.value} is {observed!r}, expected {check.expected!r}"
        expected_types = {
            "array": list,
            "boolean": bool,
            "number": (int, float),
            "object": dict,
            "string": str,
        }
        expected_type = expected_types.get(str(check.expected))
        if expected_type is None:
            raise ValueError(f"Unknown JSON type: {check.expected}")
        passed = isinstance(observed, expected_type)
        if check.expected == "number" and isinstance(observed, bool):
            passed = False
        return passed, f"{check.value} has type {type(observed).__name__}, expected {check.expected}"
    raise ValueError(f"Unknown check type: {check.type}")


def score_response(checks: list[Check], response: str) -> dict[str, Any]:
    details = []
    earned = 0.0
    possible = sum(check.points for check in checks)
    for check in checks:
        passed, observed = _evaluate(check, response)
        if passed:
            earned += check.points
        details.append(
            {
                "label": check.label or observed,
                "type": check.type,
                "passed": passed,
                "points": check.points,
                "observed": observed,
                "definition": asdict(check),
            }
        )
    return {
        "earned": earned,
        "possible": possible,
        "score": (earned / possible) if possible else None,
        "checks": details,
    }


def failed_score(checks: list[Check], reason: str) -> dict[str, Any]:
    possible = sum(check.points for check in checks)
    return {
        "earned": 0.0,
        "possible": possible,
        "score": 0.0 if possible else None,
        "checks": [
            {
                "label": check.label or check.type,
                "type": check.type,
                "passed": False,
                "points": check.points,
                "observed": reason,
                "definition": asdict(check),
            }
            for check in checks
        ],
    }
