from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROTOCOL_VERSION = "1.0"
EXECUTION_KINDS = {"single_turn", "multi_turn", "independent", "repeated", "deferred"}
AGGREGATORS = {"mean", "min", "max", "all_pass", "manual", "deferred"}
CONDITIONS = {"weights-only", "search-enabled", "agentic"}


class ProtocolError(ValueError):
    pass


@dataclass(frozen=True)
class EvalDefinition:
    id: str
    title: str
    tier: int
    category: str
    prompt: str
    weight: float
    grader: str
    aggregation: str = "mean"
    execution: str = "single_turn"
    status: str = "active"
    repetitions: int = 3
    conditions: tuple[str, ...] = ("weights-only",)
    max_tokens: int = 1200
    calibration: bool = False
    renderer: str | None = None
    human_review: bool = False
    rubric: tuple[dict[str, Any], ...] = ()
    followups: tuple[str, ...] = ()
    prompts: tuple[str, ...] = ()
    sample_count: int = 1
    variants: tuple[dict[str, Any], ...] = ()
    checks: tuple[dict[str, Any], ...] = ()
    notes: str = ""

    def variant(self, seed: int) -> dict[str, Any]:
        if not self.variants:
            return {"id": "canonical", "parameters": {}}
        return self.variants[random.Random(f"{seed}:{self.id}").randrange(len(self.variants))]

    def render(self, seed: int) -> "RenderedEval":
        variant = self.variant(seed)
        parameters = dict(variant.get("parameters", {}))
        try:
            prompt = self.prompt.format_map(_StrictFormat(parameters))
            prompts = tuple(item.format_map(_StrictFormat(parameters)) for item in self.prompts)
            followups = tuple(item.format_map(_StrictFormat(parameters)) for item in self.followups)
        except KeyError as exc:
            raise ProtocolError(f"{self.id}: missing variant parameter {exc.args[0]!r}") from exc
        if self.calibration:
            prompt = prompt.rstrip() + "\n\nState your confidence in your answer as a percentage on the final line as `Confidence: N%`."
            prompts = tuple(
                item.rstrip() + "\n\nState your confidence in your answer as a percentage on the final line as `Confidence: N%`."
                for item in prompts
            )
        messages = [{"role": "user", "content": prompt}]
        digest = hashlib.sha256(
            json.dumps(
                {"messages": messages, "prompts": prompts, "followups": followups},
                sort_keys=True,
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()
        return RenderedEval(
            definition=self,
            variant_id=str(variant.get("id", "canonical")),
            parameters=parameters,
            messages=messages,
            prompts=prompts,
            followups=followups,
            prompt_sha256=digest,
        )


class _StrictFormat(dict[str, Any]):
    def __missing__(self, key: str) -> Any:
        raise KeyError(key)


@dataclass(frozen=True)
class RenderedEval:
    definition: EvalDefinition
    variant_id: str
    parameters: dict[str, Any]
    messages: list[dict[str, str]]
    prompts: tuple[str, ...]
    followups: tuple[str, ...]
    prompt_sha256: str


@dataclass(frozen=True)
class EvalSuite:
    name: str
    version: str
    description: str
    evals: tuple[EvalDefinition, ...]
    path: Path

    @property
    def weighted_total(self) -> float:
        return sum(item.weight for item in self.evals if item.tier < 7 and item.status != "alternate")

    def get(self, eval_id: str) -> EvalDefinition:
        for item in self.evals:
            if item.id == eval_id:
                return item
        raise ProtocolError(f"Unknown eval id: {eval_id}")


def default_v1_path() -> Path:
    candidate = Path(__file__).resolve().parents[2] / "benchmarks" / "quinn_v1.json"
    if candidate.exists():
        return candidate
    packaged = Path(__file__).resolve().parent / "data" / "quinn_v1.json"
    if packaged.exists():
        return packaged
    return Path("benchmarks/quinn_v1.json")


def _as_tuple(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ProtocolError(f"Expected an array, got {type(value).__name__}")
    return tuple(value)


def load_protocol(path: str | Path | None = None) -> EvalSuite:
    suite_path = Path(path) if path else default_v1_path()
    try:
        raw = json.loads(suite_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Could not load protocol {suite_path}: {exc}") from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("evals"), list):
        raise ProtocolError("Protocol must be an object containing an evals array")

    errors: list[str] = []
    definitions: list[EvalDefinition] = []
    seen: set[str] = set()
    for index, item in enumerate(raw["evals"]):
        where = f"evals[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{where} must be an object")
            continue
        missing = [key for key in ("id", "title", "tier", "category", "prompt", "grader") if key not in item]
        if missing:
            errors.append(f"{where} missing {', '.join(missing)}")
            continue
        eval_id = str(item["id"])
        if not re.fullmatch(r"[1-7]\.\d+", eval_id):
            errors.append(f"{where} has invalid id {eval_id!r}")
        if eval_id in seen:
            errors.append(f"duplicate eval id {eval_id}")
        seen.add(eval_id)
        execution = str(item.get("execution", "single_turn"))
        aggregation = str(item.get("aggregation", "mean"))
        conditions = tuple(item.get("conditions", ["weights-only"]))
        if execution not in EXECUTION_KINDS:
            errors.append(f"{eval_id}: unknown execution {execution!r}")
        if aggregation not in AGGREGATORS:
            errors.append(f"{eval_id}: unknown aggregation {aggregation!r}")
        unknown_conditions = set(conditions) - CONDITIONS
        if unknown_conditions:
            errors.append(f"{eval_id}: unknown conditions {sorted(unknown_conditions)}")
        tier = int(item["tier"])
        weight = float(item.get("weight", 0))
        if not 1 <= tier <= 7:
            errors.append(f"{eval_id}: tier must be 1..7")
        if tier == 7 and weight != 0:
            errors.append(f"{eval_id}: Tier 7 weight must be zero")
        definitions.append(
            EvalDefinition(
                id=eval_id,
                title=str(item["title"]),
                tier=tier,
                category=str(item["category"]),
                prompt=str(item["prompt"]),
                weight=weight,
                grader=str(item["grader"]),
                aggregation=aggregation,
                execution=execution,
                status=str(item.get("status", "active")),
                repetitions=int(item.get("repetitions", 3)),
                conditions=conditions,
                max_tokens=int(item.get("max_tokens", 1200)),
                calibration=bool(item.get("calibration", False)),
                renderer=item.get("renderer"),
                human_review=bool(item.get("human_review", False)),
                rubric=_as_tuple(item.get("rubric")),
                followups=_as_tuple(item.get("followups")),
                prompts=_as_tuple(item.get("prompts")),
                sample_count=int(item.get("sample_count", 1)),
                variants=_as_tuple(item.get("variants")),
                checks=_as_tuple(item.get("checks")),
                notes=str(item.get("notes", "")),
            )
        )
    if errors:
        raise ProtocolError("Invalid protocol:\n- " + "\n- ".join(errors))
    suite = EvalSuite(
        name=str(raw.get("name", "Quinn Eval Suite")),
        version=str(raw.get("version", "1")),
        description=str(raw.get("description", "")),
        evals=tuple(definitions),
        path=suite_path,
    )
    if abs(suite.weighted_total - 100.0) > 1e-9:
        raise ProtocolError(f"Tier 1–6 active weights must total 100; got {suite.weighted_total:g}")
    return suite
