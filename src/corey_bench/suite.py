from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class SuiteError(ValueError):
    pass


@dataclass(frozen=True)
class Check:
    type: str
    points: float = 1.0
    value: Any = None
    values: list[str] = field(default_factory=list)
    label: str = ""
    flags: str = "i"
    expected: Any = None


@dataclass(frozen=True)
class Rubric:
    name: str
    description: str
    weight: float = 1.0


@dataclass(frozen=True)
class Case:
    id: str
    title: str
    category: str
    prompt: str
    system: str
    max_tokens: int
    tags: list[str]
    checks: list[Check]
    rubric: list[Rubric]
    reference_notes: str = ""


@dataclass(frozen=True)
class Suite:
    name: str
    version: str
    description: str
    cases: list[Case]
    path: Path


CHECK_TYPES = {
    "contains",
    "contains_any",
    "not_contains",
    "regex",
    "min_words",
    "max_words",
    "json_valid",
    "json_keys_exact",
    "json_field_type",
    "json_field_equals",
}


def default_suite_path() -> Path:
    candidate = Path(__file__).resolve().parents[2] / "benchmarks" / "corey_v0.json"
    if candidate.exists():
        return candidate
    return Path("benchmarks/corey_v0.json")


def load_suite(path: str | Path | None = None) -> Suite:
    suite_path = Path(path) if path else default_suite_path()
    try:
        raw = json.loads(suite_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SuiteError(f"Could not load suite {suite_path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise SuiteError(f"Invalid suite: top level must be a JSON object, got {type(raw).__name__}")
    errors: list[str] = []
    for key in ("name", "version", "description", "cases"):
        if key not in raw:
            errors.append(f"missing top-level field: {key}")

    cases: list[Case] = []
    seen: set[str] = set()
    for index, item in enumerate(raw.get("cases", [])):
        where = f"cases[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{where} must be an object")
            continue
        missing = [key for key in ("id", "title", "category", "prompt") if not item.get(key)]
        if missing:
            errors.append(f"{where} missing: {', '.join(missing)}")
            continue
        if item["id"] in seen:
            errors.append(f"duplicate case id: {item['id']}")
        seen.add(item["id"])

        checks: list[Check] = []
        for check_index, check in enumerate(item.get("checks", [])):
            if not isinstance(check, dict):
                errors.append(f"{where}.checks[{check_index}] must be an object")
                continue
            check_type = check.get("type")
            if check_type not in CHECK_TYPES:
                errors.append(f"{where}.checks[{check_index}] has unknown type {check_type!r}")
                continue
            if float(check.get("points", 1)) <= 0:
                errors.append(f"{where}.checks[{check_index}] points must be positive")
            checks.append(Check(**check))

        rubrics: list[Rubric] = []
        for rubric_index, rubric in enumerate(item.get("rubric", [])):
            if not isinstance(rubric, dict):
                errors.append(f"{where}.rubric[{rubric_index}] must be an object")
                continue
            if not rubric.get("name") or not rubric.get("description"):
                errors.append(f"{where}.rubric[{rubric_index}] needs name and description")
                continue
            if float(rubric.get("weight", 1)) <= 0:
                errors.append(f"{where}.rubric[{rubric_index}] weight must be positive")
            rubrics.append(Rubric(**rubric))
        if not rubrics:
            errors.append(f"{where} needs at least one human rubric criterion")

        cases.append(
            Case(
                id=item["id"],
                title=item["title"],
                category=item["category"],
                prompt=item["prompt"],
                system=item.get(
                    "system",
                    "Be accurate, direct, and useful. State uncertainty rather than inventing facts.",
                ),
                max_tokens=int(item.get("max_tokens", 1200)),
                tags=list(item.get("tags", [])),
                checks=checks,
                rubric=rubrics,
                reference_notes=item.get("reference_notes", ""),
            )
        )

    if not cases:
        errors.append("suite contains no valid cases")
    if errors:
        raise SuiteError("Invalid suite:\n- " + "\n- ".join(errors))
    return Suite(raw["name"], str(raw["version"]), raw["description"], cases, suite_path)
