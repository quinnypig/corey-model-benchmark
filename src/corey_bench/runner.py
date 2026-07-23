from __future__ import annotations

import hashlib
import json
import os
import queue
import secrets
import threading
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .artifacts import ArtifactError, extract_code, salvage_svg_preview, validate_svg, write_svg_preview
from .graders import grade_attempt
from .judging import JudgeOutputError, judge_messages, parse_judge_output
from .openrouter import OpenRouterClient, OpenRouterError, OpenRouterPolicyError
from .protocol import EvalDefinition, EvalSuite, RenderedEval, load_protocol
from .telemetry import runtime_attributes, set_attributes, span, trim_memory


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: Path, record: dict[str, Any], lock: threading.Lock | None = None) -> None:
    serialized = json.dumps(record, ensure_ascii=False) + "\n"
    if lock:
        lock.acquire()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if lock:
            lock.release()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL {path}:{number}: {exc}") from exc
    return rows


def collapse_result_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the latest append-only receipt for each logical attempt."""
    latest: dict[str, dict[str, Any]] = {}
    anonymous: list[dict[str, Any]] = []
    for row in rows:
        attempt_id = str(row.get("attempt_id") or "")
        if attempt_id:
            latest[attempt_id] = row
        else:
            anonymous.append(row)
    return [*latest.values(), *anonymous]


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


@dataclass(frozen=True)
class RunConfig:
    models: list[str]
    eval_ids: list[str]
    conditions: list[str]
    repetitions: int | None = None
    temperature: float | None = 1.0
    reasoning: str = "provider-default"
    seed: int = 8675309
    max_budget_usd: float | None = 50.0
    estimated_cost_usd: float | None = None


@dataclass(frozen=True)
class Job:
    run_id: str
    model: str
    eval_id: str
    condition: str
    repetition: int
    attempt_id: str


@dataclass(frozen=True)
class ReviewJob:
    run_id: str
    attempt_id: str
    force: bool = False


class RunStore:
    def __init__(self, root: str | Path = "runs") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    def run_dir(self, run_id: str) -> Path:
        if not run_id or not all(char.isalnum() or char in "-_" for char in run_id):
            raise ValueError("Invalid run id")
        return self.root / run_id

    def lock(self, run_id: str) -> threading.Lock:
        with self._guard:
            return self._locks.setdefault(run_id, threading.Lock())

    def create(self, config: RunConfig, suite: EvalSuite) -> tuple[str, list[Job]]:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + secrets.token_hex(3)
        run_dir = self.run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=False)
        rendered = {eval_id: suite.get(eval_id).render(config.seed) for eval_id in config.eval_ids}
        jobs: list[Job] = []
        request_estimate = 0
        # Interleave models so a large run makes visible progress across the
        # whole roster instead of exhausting one provider before starting the next.
        for eval_id in config.eval_ids:
            definition = suite.get(eval_id)
            repetitions = config.repetitions or definition.repetitions
            for condition in config.conditions:
                if condition not in definition.conditions and condition != "agentic":
                    continue
                if condition == "agentic" and eval_id not in {"2.1", "2.2"}:
                    continue
                for repetition in range(1, repetitions + 1):
                    for model in config.models:
                        key = f"{run_id}\0{model}\0{eval_id}\0{condition}\0{repetition}"
                        attempt_id = hashlib.sha256(key.encode()).hexdigest()[:20]
                        jobs.append(Job(run_id, model, eval_id, condition, repetition, attempt_id))
                        request_estimate += condition_request_count(definition, rendered[eval_id], condition)
                        if definition.human_review:
                            request_estimate += 1
        manifest = {
            "protocol_version": suite.version,
            "run_id": run_id,
            "created_at": utc_now(),
            "suite": {"name": suite.name, "version": suite.version, "source": str(suite.path)},
            "config": asdict(config),
            "models": config.models,
            "expected_jobs": len(jobs),
            "estimated_requests": request_estimate,
            "evals": [definition_manifest(suite.get(eval_id), rendered[eval_id]) for eval_id in config.eval_ids],
            "jobs": [asdict(job) for job in jobs],
        }
        atomic_json(run_dir / "manifest.json", manifest)
        atomic_json(
            run_dir / "state.json",
            {
                "run_id": run_id, "status": "queued", "created_at": manifest["created_at"],
                "updated_at": manifest["created_at"], "expected_jobs": len(jobs), "completed_jobs": 0,
                "successful_jobs": 0, "failed_jobs": 0, "active_jobs": 0, "cancel_requested": False,
            },
        )
        return run_id, jobs

    def manifest(self, run_id: str) -> dict[str, Any]:
        return json.loads((self.run_dir(run_id) / "manifest.json").read_text(encoding="utf-8"))

    def state(self, run_id: str) -> dict[str, Any]:
        return json.loads((self.run_dir(run_id) / "state.json").read_text(encoding="utf-8"))

    def update_state(self, run_id: str, **changes: Any) -> dict[str, Any]:
        with self.lock(run_id):
            state = self.state(run_id)
            state.update(changes)
            state["updated_at"] = utc_now()
            atomic_json(self.run_dir(run_id) / "state.json", state)
            return state

    def mutate_state(self, run_id: str, fn: Any) -> dict[str, Any]:
        with self.lock(run_id):
            state = self.state(run_id)
            fn(state)
            state["updated_at"] = utc_now()
            atomic_json(self.run_dir(run_id) / "state.json", state)
            return state

    def append_result(self, run_id: str, result: dict[str, Any]) -> None:
        append_jsonl(self.run_dir(run_id) / "results.jsonl", result, self.lock(run_id))

    def results(self, run_id: str) -> list[dict[str, Any]]:
        with self.lock(run_id):
            return collapse_result_rows(read_jsonl(self.run_dir(run_id) / "results.jsonl"))

    def result_summary(self, run_id: str) -> tuple[set[str], float]:
        """Stream the large receipt ledger instead of materializing it in every worker."""
        attempts: set[str] = set()
        cost = 0.0
        path = self.run_dir(run_id) / "results.jsonl"
        with self.lock(run_id):
            if not path.exists():
                return attempts, cost
            with path.open(encoding="utf-8") as handle:
                for number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Invalid JSONL {path}:{number}: {exc}") from exc
                    if row.get("attempt_id"):
                        attempts.add(str(row["attempt_id"]))
                    cost += float(row.get("cost_usd") or row.get("usage", {}).get("cost") or 0)
        return attempts, cost

    def result(self, run_id: str, attempt_id: str) -> dict[str, Any] | None:
        path = self.run_dir(run_id) / "results.jsonl"
        matched = None
        with self.lock(run_id):
            if not path.exists():
                return None
            with path.open(encoding="utf-8") as handle:
                for number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Invalid JSONL {path}:{number}: {exc}") from exc
                    if row.get("attempt_id") == attempt_id:
                        matched = row
        return matched

    def reviewable_attempt_ids(self, run_id: str) -> list[str]:
        return [str(row["attempt_id"]) for row in self.results(run_id) if _can_model_review(row)]

    def recent(self) -> list[dict[str, Any]]:
        runs = []
        for path in sorted(self.root.glob("*/state.json"), reverse=True):
            try:
                state = json.loads(path.read_text(encoding="utf-8"))
                manifest_path = path.parent / "manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
                state["models"] = manifest.get("models", [])
                state["suite"] = manifest.get("suite", {}).get("name")
                runs.append(state)
            except (OSError, json.JSONDecodeError):
                continue
        return runs


def definition_manifest(definition: EvalDefinition, rendered: RenderedEval) -> dict[str, Any]:
    return {
        "id": definition.id, "title": definition.title, "tier": definition.tier,
        "category": definition.category, "weight": definition.weight, "status": definition.status,
        "grader": definition.grader, "aggregation": definition.aggregation, "execution": definition.execution,
        "human_review": definition.human_review, "rubric": list(definition.rubric),
        "variant_id": rendered.variant_id, "parameters": rendered.parameters,
        "messages": rendered.messages, "prompts": list(rendered.prompts), "followups": list(rendered.followups),
        "prompt_sha256": rendered.prompt_sha256, "renderer": definition.renderer,
    }


def request_count(definition: EvalDefinition, rendered: RenderedEval) -> int:
    if definition.execution == "independent":
        return len(rendered.prompts)
    if definition.execution == "repeated":
        return definition.sample_count
    if definition.execution == "multi_turn":
        return 1 + len(rendered.followups)
    return 1


def condition_request_count(definition: EvalDefinition, rendered: RenderedEval, condition: str) -> int:
    """Return the maximum provider calls made by one repetition of a condition."""
    if condition == "agentic":
        return 10
    return request_count(definition, rendered)


def suite_request_count(
    suite: EvalSuite,
    *,
    conditions: tuple[str, ...] = ("weights-only", "search-enabled", "agentic"),
    seed: int = 8675309,
    repetitions: int | None = None,
    include_reviews: bool = True,
) -> int:
    """Return the full-protocol maximum request count for one model."""
    total = 0
    for definition in suite.evals:
        rendered = definition.render(seed)
        for condition in conditions:
            allowed = condition in definition.conditions or (
                condition == "agentic" and definition.id in {"2.1", "2.2"}
            )
            if allowed:
                total += condition_request_count(definition, rendered, condition) * (repetitions or definition.repetitions)
                if include_reviews and definition.human_review:
                    total += repetitions or definition.repetitions
    return total


class RunQueue:
    SVG_ARTIFACT_VERSION = 3
    RESULT_SEMANTICS_VERSION = 2

    def __init__(
        self,
        api_key: str,
        *,
        store: RunStore | None = None,
        suite: EvalSuite | None = None,
        workers: int = 3,
        per_model_workers: int = 3,
        judge_model: str = "openai/gpt-5.6-luna-pro",
        judge_workers: int = 2,
        timeout: float = 300,
        attempts: int = 5,
        max_auto_recoveries: int = 3,
        recovery_window_seconds: int = 3600,
        report_interval_seconds: int = 30,
        max_review_attempts: int = 2,
    ) -> None:
        self.store = store or RunStore()
        self.suite = suite or load_protocol()
        self.client = OpenRouterClient(api_key, timeout=timeout, attempts=attempts)
        self.workers = max(1, min(int(workers), 24))
        self.per_model_workers = max(1, min(int(per_model_workers), self.workers))
        self.judge_model = judge_model
        self.judge_workers = max(0, min(int(judge_workers), 4))
        self.max_auto_recoveries = max(1, int(max_auto_recoveries))
        self.recovery_window_seconds = max(60, int(recovery_window_seconds))
        self.report_interval_seconds = max(0, int(report_interval_seconds))
        self.max_review_attempts = max(1, int(max_review_attempts))
        self._queue: queue.Queue[Job | None] = queue.Queue()
        self._review_queue: queue.Queue[ReviewJob | None] = queue.Queue()
        self._threads: list[threading.Thread] = []
        self._judge_threads: list[threading.Thread] = []
        self._model_limits: dict[str, threading.BoundedSemaphore] = {}
        self._model_limits_guard = threading.Lock()
        self._report_guard = threading.Lock()
        self._report_last: dict[str, float] = {}
        self._review_guard = threading.Lock()
        self._review_inflight: set[tuple[str, str]] = set()
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        with span(
            "queue.startup",
            {
                "queue.worker_count": self.workers,
                "queue.per_model_worker_count": self.per_model_workers,
                "queue.judge_worker_count": self.judge_workers,
                "queue.judge_model": self.judge_model,
                "queue.max_review_attempts": self.max_review_attempts,
                **runtime_attributes(),
            },
        ) as current:
            self._started = True
            migrated_results = self._migrate_result_semantics()
            rebuilt_artifacts = self._rebuild_svg_artifacts()
            for number in range(self.workers):
                thread = threading.Thread(target=self._worker, name=f"quinnferno-worker-{number+1}", daemon=True)
                thread.start()
                self._threads.append(thread)
            recovery = self.recover()
            for number in range(self.judge_workers):
                thread = threading.Thread(target=self._judge_worker, name=f"quinnferno-judge-{number+1}", daemon=True)
                thread.start()
                self._judge_threads.append(thread)
            recovered_reviews = self.recover_reviews()
            set_attributes(
                current,
                {
                    **recovery,
                    **migrated_results,
                    **rebuilt_artifacts,
                    "recovery.review_count": recovered_reviews,
                    "queue.depth": self._queue.qsize(),
                    "queue.review_depth": self._review_queue.qsize(),
                    **runtime_attributes(),
                },
            )

    def _migrate_result_semantics(self) -> dict[str, int]:
        """Reclassify legacy model outcomes and reconcile execution health."""
        migrated_runs = 0
        policy_outcomes = 0
        empty_outcomes = 0
        marker_name = f".result-semantics-v{self.RESULT_SEMANTICS_VERSION}.json"
        for state in self.store.recent():
            run_id = state["run_id"]
            run_dir = self.store.run_dir(run_id)
            marker = run_dir / marker_name
            if marker.exists():
                continue
            for row in self.store.results(run_id):
                if row.get("status") != "error":
                    continue
                diagnostic = str(row.get("error") or "")
                replacement = dict(row)
                if "content_filter" in diagnostic or "considered high risk" in diagnostic:
                    replacement.update(
                        {
                            "status": "blocked",
                            "outcome_type": "provider_policy_block",
                            "provider_error": diagnostic,
                            "grade": {
                                "score": 0.0,
                                "pass": False,
                                "verdict": "Provider safety filter blocked the benchmark prompt",
                                "human_required": False,
                            },
                        }
                    )
                    replacement.pop("error", None)
                    self.store.append_result(run_id, replacement)
                    policy_outcomes += 1
                elif "role 'assistant' must not be empty" in diagnostic:
                    replacement.update(
                        {
                            "status": "ok",
                            "outcome_type": "empty_response",
                            "diagnostic": diagnostic,
                            "response": "",
                            "responses": [""],
                            "grade": {
                                "score": 0.0,
                                "pass": False,
                                "verdict": "Model returned an empty response",
                                "human_required": False,
                            },
                        }
                    )
                    replacement.pop("error", None)
                    self.store.append_result(run_id, replacement)
                    empty_outcomes += 1
            self._reconcile_run_health(run_id)
            atomic_json(
                marker,
                {
                    "version": self.RESULT_SEMANTICS_VERSION,
                    "migrated_at": utc_now(),
                },
            )
            migrated_runs += 1
        return {
            "result.migrated_run_count": migrated_runs,
            "result.policy_outcome_count": policy_outcomes,
            "result.empty_outcome_count": empty_outcomes,
        }

    def _reconcile_run_health(self, run_id: str) -> dict[str, Any]:
        results = self.store.results(run_id)
        execution_errors = [row for row in results if row.get("status") == "error"]
        cancelled = [row for row in results if row.get("status") == "cancelled"]
        benchmark_failures = [
            row
            for row in results
            if row.get("status") not in {"error", "cancelled"}
            and row.get("grade", {}).get("pass") is False
        ]

        def reconcile(state: dict[str, Any]) -> None:
            state["completed_jobs"] = len(results)
            state["successful_jobs"] = len(results) - len(execution_errors) - len(cancelled)
            state["failed_jobs"] = len(execution_errors)
            state["execution_error_jobs"] = len(execution_errors)
            state["cancelled_jobs"] = len(cancelled)
            state["benchmark_failed_jobs"] = len(benchmark_failures)
            if len(results) >= int(state.get("expected_jobs") or 0):
                if state.get("status") == "budget_exhausted":
                    return
                if state.get("cancel_requested") or state.get("status") == "cancelled":
                    state["status"] = "cancelled"
                elif execution_errors:
                    state["status"] = "execution_errors"
                else:
                    state["status"] = "completed"

        return self.store.mutate_state(run_id, reconcile)

    def _rebuild_svg_artifacts(self) -> dict[str, int]:
        """Upgrade stored SVG previews without changing historical grades."""
        rebuilt_runs = 0
        rebuilt_artifacts = 0
        salvaged_artifacts = 0
        errors = 0
        marker_name = f".svg-artifacts-v{self.SVG_ARTIFACT_VERSION}.json"
        for state in self.store.recent():
            run_id = state["run_id"]
            run_dir = self.store.run_dir(run_id)
            results_path = run_dir / "results.jsonl"
            marker_path = run_dir / marker_name
            if marker_path.exists() or not results_path.exists():
                continue
            temporary = results_path.with_suffix(".svg-rebuild.tmp")
            changed = False
            try:
                with self.store.lock(run_id):
                    with results_path.open(encoding="utf-8") as source, temporary.open("w", encoding="utf-8") as target:
                        for number, line in enumerate(source, 1):
                            if not line.strip():
                                continue
                            row = json.loads(line)
                            eval_id = str(row.get("eval_id") or "")
                            definition = next((item for item in self.suite.evals if item.id == eval_id), None)
                            if row.get("status") == "ok" and definition and definition.renderer == "svg":
                                outputs = row.get("responses") or [row.get("response") or ""]
                                job = Job(
                                    run_id=run_id,
                                    model=str(row.get("model") or "unknown"),
                                    eval_id=eval_id,
                                    condition=str(row.get("condition") or "weights-only"),
                                    repetition=int(row.get("repetition") or 1),
                                    attempt_id=str(row["attempt_id"]),
                                )
                                try:
                                    row["artifacts"] = self._artifacts(
                                        job, definition, [str(value) for value in outputs], row.get("grade") or {},
                                    )
                                    rebuilt_artifacts += 1
                                    salvaged_artifacts += int(
                                        bool(row["artifacts"] and row["artifacts"][0].get("salvaged"))
                                    )
                                    changed = True
                                except Exception:
                                    errors += 1
                            target.write(json.dumps(row, ensure_ascii=False) + "\n")
                        target.flush()
                        os.fsync(target.fileno())
                    temporary.replace(results_path)
                    atomic_json(
                        marker_path,
                        {
                            "version": self.SVG_ARTIFACT_VERSION,
                            "rebuilt_at": utc_now(),
                            "changed": changed,
                        },
                    )
                rebuilt_runs += 1
            except (OSError, ValueError, json.JSONDecodeError):
                errors += 1
                if temporary.exists():
                    temporary.unlink()
        return {
            "artifact.rebuilt_run_count": rebuilt_runs,
            "artifact.rebuilt_count": rebuilt_artifacts,
            "artifact.salvaged_count": salvaged_artifacts,
            "artifact.rebuild_error_count": errors,
        }

    def submit(self, config: RunConfig) -> str:
        if len(config.models) > 10:
            raise ValueError("A run may contain at most 10 models")
        if not config.models or not config.eval_ids:
            raise ValueError("Select at least one model and one eval")
        with span(
            "run.submit",
            {
                "run.model_count": len(config.models),
                "run.eval_count": len(config.eval_ids),
                "run.conditions": config.conditions,
                "run.max_budget_usd": config.max_budget_usd,
            },
        ) as current:
            catalog = self.client.list_models()
            available = {item["id"]: item for item in catalog}
            missing = [model for model in config.models if model not in available]
            if missing:
                raise OpenRouterError("Requested model ID(s) are unavailable for this API key: " + ", ".join(missing))
            config = replace(config, estimated_cost_usd=estimate_cost(config, self.suite, available, self.judge_model if self.judge_workers else None))
            run_id, jobs = self.store.create(config, self.suite)
            for job in jobs:
                self._queue.put(job)
            set_attributes(
                current,
                {
                    "run.id": run_id,
                    "run.job_count": len(jobs),
                    "run.estimated_cost_usd": config.estimated_cost_usd,
                    "queue.depth": self._queue.qsize(),
                },
            )
            return run_id

    def recover(self) -> dict[str, int]:
        recovered_runs = 0
        recovered_jobs = 0
        paid_calls_at_risk = 0
        paused_runs = 0
        for state in self.store.recent():
            if state.get("status") not in {"queued", "running", "interrupted"}:
                continue
            run_id = state["run_id"]
            completed, _ = self.store.result_summary(run_id)
            manifest = self.store.manifest(run_id)
            retry_attempts = set(state.get("retry_attempts") or [])
            now = datetime.now(timezone.utc)
            cutoff = now.timestamp() - self.recovery_window_seconds
            history = []
            for value in state.get("recovery_history", []):
                try:
                    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
                except ValueError:
                    continue
                if parsed.timestamp() >= cutoff:
                    history.append(parsed.isoformat())
            previous_active = int(state.get("active_jobs") or 0)
            paid_calls_at_risk += previous_active
            if previous_active and len(history) >= self.max_auto_recoveries:
                self.store.update_state(
                    run_id,
                    status="recovery_paused",
                    active_jobs=0,
                    active_attempts={},
                    recovery_history=history,
                    recovery_paused_at=utc_now(),
                    recovery_reason=f"{len(history)} automatic recoveries inside {self.recovery_window_seconds} seconds",
                )
                paused_runs += 1
                continue
            if previous_active:
                history.append(now.isoformat())
            self.store.update_state(
                run_id,
                status="queued",
                active_jobs=0,
                active_attempts={},
                recovery_history=history,
                recovery_count=int(state.get("recovery_count") or 0) + 1,
                last_recovered_at=utc_now(),
            )
            added = 0
            for raw in manifest.get("jobs", []):
                if raw.get("attempt_id") not in completed or raw.get("attempt_id") in retry_attempts:
                    self._queue.put(Job(**raw))
                    added += 1
            recovered_runs += 1
            recovered_jobs += added
        return {
            "recovery.run_count": recovered_runs,
            "recovery.job_count": recovered_jobs,
            "recovery.paused_run_count": paused_runs,
            "recovery.paid_calls_at_risk": paid_calls_at_risk,
        }

    def recover_reviews(self) -> int:
        reviewed: set[tuple[str, str]] = set()
        queued = 0
        for state in self.store.recent():
            run_id = state["run_id"]
            for review in read_jsonl(self.store.run_dir(run_id) / "reviews.jsonl"):
                if review.get("reviewer_type") == "model":
                    reviewed.add((run_id, str(review.get("attempt_id"))))
            attempts = self._review_attempt_counts(run_id)
            for attempt_id in self.store.reviewable_attempt_ids(run_id):
                key = (run_id, attempt_id)
                if key in reviewed:
                    continue
                if attempts.get(attempt_id, 0) >= self.max_review_attempts:
                    append_jsonl(
                        self.store.run_dir(run_id) / "reviews.jsonl",
                        self._review_fuse_receipt(attempt_id, attempts[attempt_id]),
                        self.store.lock(run_id),
                    )
                    reviewed.add(key)
                else:
                    self._review_queue.put(ReviewJob(*key))
                    queued += 1
        return queued

    def resume(self, run_id: str) -> int:
        state = self.store.state(run_id)
        if state.get("status") != "recovery_paused":
            raise ValueError("Run is not paused by the recovery fuse")
        completed, _ = self.store.result_summary(run_id)
        manifest = self.store.manifest(run_id)
        jobs = [Job(**raw) for raw in manifest.get("jobs", []) if raw.get("attempt_id") not in completed]
        self.store.update_state(
            run_id,
            status="queued",
            active_jobs=0,
            active_attempts={},
            recovery_history=[],
            recovery_resumed_at=utc_now(),
        )
        for job in jobs:
            self._queue.put(job)
        with span("run.resume", {"run.id": run_id, "recovery.job_count": len(jobs), **runtime_attributes()}):
            pass
        return len(jobs)

    def retry_execution_errors(self, run_id: str) -> int:
        state = self.store.state(run_id)
        if state.get("status") not in {"execution_errors", "completed_with_errors"}:
            raise ValueError("Run has no terminal execution errors to retry")
        errors = [row for row in self.store.results(run_id) if row.get("status") == "error"]
        manifest = self.store.manifest(run_id)
        jobs_by_id = {str(raw.get("attempt_id")): Job(**raw) for raw in manifest.get("jobs", [])}
        jobs = [jobs_by_id[str(row.get("attempt_id"))] for row in errors if str(row.get("attempt_id")) in jobs_by_id]
        if not jobs:
            raise ValueError("No retryable execution-error receipts were found")

        retry_ids = {job.attempt_id for job in jobs}

        def prepare(value: dict[str, Any]) -> None:
            value["status"] = "queued"
            value["cancel_requested"] = False
            value["completed_jobs"] = max(0, int(value.get("completed_jobs") or 0) - len(jobs))
            value["failed_jobs"] = max(0, int(value.get("failed_jobs") or 0) - len(jobs))
            value["execution_error_jobs"] = value["failed_jobs"]
            value["retry_attempts"] = sorted(set(value.get("retry_attempts") or []) | retry_ids)
            value["last_error_retry_at"] = utc_now()

        self.store.mutate_state(run_id, prepare)
        for job in jobs:
            self._queue.put(job)
        with span(
            "run.retry_execution_errors",
            {"run.id": run_id, "retry.job_count": len(jobs), "queue.depth": self._queue.qsize()},
        ):
            pass
        return len(jobs)

    def retry_pending_reviews(self, run_id: str) -> int:
        self.store.state(run_id)  # validate the run before reading its ledgers
        reviews = read_jsonl(self.store.run_dir(run_id) / "reviews.jsonl")
        scored = {
            str(row.get("attempt_id"))
            for row in reviews
            if isinstance(row.get("score"), (int, float))
        }
        attempt_ids = [
            attempt_id
            for attempt_id in self.store.reviewable_attempt_ids(run_id)
            if attempt_id not in scored
        ]
        if not attempt_ids:
            raise ValueError("No unresolved model judgments were found")
        for attempt_id in attempt_ids:
            self._review_queue.put(ReviewJob(run_id, attempt_id, force=True))
        with span(
            "run.retry_pending_reviews",
            {
                "run.id": run_id,
                "review.retry_count": len(attempt_ids),
                "queue.review_depth": self._review_queue.qsize(),
            },
        ):
            pass
        return len(attempt_ids)

    def cancel(self, run_id: str) -> None:
        self.store.update_state(run_id, cancel_requested=True, status="cancelling")

    def _worker(self) -> None:
        while True:
            job = self._queue.get()
            if job is None:
                return
            try:
                with span(
                    "eval.job",
                    {
                        "run.id": job.run_id,
                        "eval.attempt_id": job.attempt_id,
                        "eval.id": job.eval_id,
                        "eval.condition": job.condition,
                        "eval.repetition": job.repetition,
                        "gen_ai.request.model": job.model,
                        "queue.depth": self._queue.qsize(),
                        **runtime_attributes(),
                    },
                ) as current:
                    state = self.store.state(job.run_id)
                    if state.get("cancel_requested"):
                        self._finish_skipped(job)
                        set_attributes(current, {"eval.status": "cancelled"})
                        continue
                    manifest = self.store.manifest(job.run_id)
                    budget = manifest.get("config", {}).get("max_budget_usd")
                    existing, spent = self.store.result_summary(job.run_id)
                    retry_attempts = set(state.get("retry_attempts") or [])
                    set_attributes(current, {"run.recorded_spend_usd": spent, "run.max_budget_usd": budget})
                    if budget is not None and spent >= float(budget):
                        self.store.update_state(job.run_id, cancel_requested=True, status="budget_exhausted")
                        self._finish_skipped(job)
                        set_attributes(current, {"eval.status": "budget_exhausted"})
                        continue
                    if job.attempt_id in existing and job.attempt_id not in retry_attempts:
                        set_attributes(current, {"eval.status": "duplicate_skipped"})
                        continue
                    self.store.mutate_state(job.run_id, lambda value: _start_job(value, job))
                    result = self._execute(job)
                    self.store.append_result(job.run_id, result)
                    final_state = self.store.mutate_state(
                        job.run_id,
                        lambda value: _finish_job(
                            value,
                            result.get("status") != "error",
                            job.attempt_id,
                            benchmark_failed=result.get("grade", {}).get("pass") is False,
                        ),
                    )
                    completed = final_state.get("status") in {"completed", "execution_errors", "cancelled"}
                    self._refresh_report(job.run_id, force=completed)
                    set_attributes(
                        current,
                        {
                            "eval.status": result.get("status"),
                            "eval.score": result.get("grade", {}).get("score"),
                            "eval.passed": result.get("grade", {}).get("pass"),
                            "eval.latency_seconds": result.get("latency_seconds"),
                            "gen_ai.response.model": result.get("resolved_model"),
                            "gen_ai.usage.input_tokens": result.get("usage", {}).get("prompt_tokens"),
                            "gen_ai.usage.output_tokens": result.get("usage", {}).get("completion_tokens"),
                            "gen_ai.usage.total_tokens": result.get("usage", {}).get("total_tokens"),
                            "quinnferno.cost_usd": result.get("cost_usd"),
                            "openrouter.provider": result.get("provider"),
                            "queue.review_depth": self._review_queue.qsize(),
                            **runtime_attributes(),
                        },
                    )
                    if completed:
                        with span("memory.trim", {"run.id": job.run_id}) as memory_span:
                            set_attributes(memory_span, trim_memory())
                    if _can_model_review(result):
                        self._review_queue.put(ReviewJob(job.run_id, job.attempt_id))
            except OpenRouterPolicyError as exc:
                try:
                    result = self._policy_outcome_result(job, exc)
                    self.store.append_result(job.run_id, result)
                    self.store.mutate_state(
                        job.run_id,
                        lambda value: _finish_job(
                            value,
                            True,
                            job.attempt_id,
                            benchmark_failed=True,
                        ),
                    )
                    self._refresh_report(job.run_id)
                except Exception:
                    pass
            except Exception as exc:  # worker containment boundary
                try:
                    result = self._error_result(job, exc)
                    self.store.append_result(job.run_id, result)
                    self.store.mutate_state(job.run_id, lambda value: _finish_job(value, False, job.attempt_id))
                    self._refresh_report(job.run_id)
                except Exception:
                    pass
            finally:
                self._queue.task_done()

    def _judge_worker(self) -> None:
        while True:
            job = self._review_queue.get()
            if job is None:
                return
            key = (job.run_id, job.attempt_id)
            with self._review_guard:
                if key in self._review_inflight:
                    self._review_queue.task_done()
                    continue
                self._review_inflight.add(key)
            try:
                with span(
                    "eval.review",
                    {
                        "run.id": job.run_id,
                        "eval.attempt_id": job.attempt_id,
                        "gen_ai.request.model": self.judge_model,
                        "queue.review_depth": self._review_queue.qsize(),
                        **runtime_attributes(),
                    },
                ) as current:
                    review = self._execute_review(job)
                    if review:
                        set_attributes(
                            current,
                            {
                                "review.status": review.get("status", "ok"),
                                "review.score": review.get("score"),
                                "quinnferno.cost_usd": review.get("cost_usd"),
                                "gen_ai.usage.total_tokens": review.get("usage", {}).get("total_tokens"),
                                **runtime_attributes(),
                            },
                        )
            except Exception as exc:
                append_jsonl(
                    self.store.run_dir(job.run_id) / "judge-errors.jsonl",
                    {"attempt_id": job.attempt_id, "judge_model": self.judge_model, "at": utc_now(), "error": f"{type(exc).__name__}: {exc}"},
                    self.store.lock(job.run_id),
                )
            finally:
                with self._review_guard:
                    self._review_inflight.discard(key)
                self._review_queue.task_done()

    def _execute_review(self, job: ReviewJob) -> dict[str, Any] | None:
        reviews_path = self.store.run_dir(job.run_id) / "reviews.jsonl"
        existing_reviews = [
            row
            for row in read_jsonl(reviews_path)
            if row.get("attempt_id") == job.attempt_id
        ]
        if any(isinstance(row.get("score"), (int, float)) for row in existing_reviews):
            return None
        if not job.force and any(
            row.get("attempt_id") == job.attempt_id and row.get("reviewer_type") == "model"
            for row in existing_reviews
        ):
            return None
        result = self.store.result(job.run_id, job.attempt_id)
        if not result or not _can_model_review(result):
            return None
        attempts = self._review_attempt_counts(job.run_id).get(job.attempt_id, 0)
        if attempts >= self.max_review_attempts and not job.force:
            receipt = self._review_fuse_receipt(job.attempt_id, attempts)
            append_jsonl(reviews_path, receipt, self.store.lock(job.run_id))
            return receipt
        manifest = self.store.manifest(job.run_id)
        budget = manifest.get("config", {}).get("max_budget_usd")
        _, inference_spend = self.store.result_summary(job.run_id)
        review_spend = sum(float(row.get("cost_usd") or 0) for row in read_jsonl(reviews_path) if row.get("reviewer_type") == "model")
        if budget is not None and inference_spend + review_spend >= float(budget):
            receipt = {
                "attempt_id": job.attempt_id, "reviewed_at": utc_now(), "reviewer_type": "model",
                "judge_model": self.judge_model, "status": "skipped_budget", "cost_usd": 0,
            }
            append_jsonl(
                reviews_path,
                receipt,
                self.store.lock(job.run_id),
            )
            return receipt
        eval_manifest = next(row for row in manifest.get("evals", []) if row.get("id") == result.get("eval_id"))
        append_jsonl(
            self.store.run_dir(job.run_id) / "judge-attempts.jsonl",
            {
                "attempt_id": job.attempt_id,
                "judge_model": self.judge_model,
                "attempt_number": attempts + 1,
                "manual_retry": job.force,
                "started_at": utc_now(),
            },
            self.store.lock(job.run_id),
        )
        with self._model_limit(self.judge_model):
            completion = self.client.complete_messages(
                model=self.judge_model,
                messages=judge_messages(eval_manifest=eval_manifest, result=result),
                max_tokens=1400,
                temperature=0,
                seed=8675309,
                reasoning="low",
                response_format={"type": "json_object"},
            )
        judge_output = completion.text or completion.reasoning or ""
        try:
            judged = parse_judge_output(judge_output)
        except JudgeOutputError as exc:
            receipt = {
                "attempt_id": job.attempt_id,
                "reviewed_at": utc_now(),
                "reviewer_type": "model",
                "judge_model": self.judge_model,
                "status": "invalid_judge_output",
                "error": str(exc),
                "judge_response": completion.text[:20_000],
                "judge_reasoning": (completion.reasoning or "")[:20_000],
                "usage": completion.usage,
                "cost_usd": float(completion.usage.get("cost") or 0),
                "response_id": completion.response_id,
            }
            append_jsonl(reviews_path, receipt, self.store.lock(job.run_id))
            return receipt
        receipt = {
            "attempt_id": job.attempt_id,
            "reviewed_at": utc_now(),
            "reviewer_type": "model",
            "judge_model": self.judge_model,
            "score": judged["score"],
            "verdict": judged["verdict"],
            "rationale": judged["rationale"],
            "rubric_scores": judged["rubric_scores"],
            "usage": completion.usage,
            "cost_usd": float(completion.usage.get("cost") or 0),
            "response_id": completion.response_id,
        }
        append_jsonl(
            reviews_path,
            receipt,
            self.store.lock(job.run_id),
        )
        self._refresh_report(job.run_id)
        return receipt

    def _review_attempt_counts(self, run_id: str) -> dict[str, int]:
        intent_counts: dict[str, int] = {}
        error_counts: dict[str, int] = {}
        with self.store.lock(run_id):
            attempts = read_jsonl(self.store.run_dir(run_id) / "judge-attempts.jsonl")
            errors = read_jsonl(self.store.run_dir(run_id) / "judge-errors.jsonl")
        for row in attempts:
            attempt_id = str(row.get("attempt_id") or "")
            if attempt_id:
                intent_counts[attempt_id] = intent_counts.get(attempt_id, 0) + 1
        for row in errors:
            attempt_id = str(row.get("attempt_id") or "")
            if attempt_id:
                error_counts[attempt_id] = error_counts.get(attempt_id, 0) + 1
        return {
            attempt_id: max(intent_counts.get(attempt_id, 0), error_counts.get(attempt_id, 0))
            for attempt_id in intent_counts.keys() | error_counts.keys()
        }

    def _review_fuse_receipt(self, attempt_id: str, attempts: int) -> dict[str, Any]:
        return {
            "attempt_id": attempt_id,
            "reviewed_at": utc_now(),
            "reviewer_type": "model",
            "judge_model": self.judge_model,
            "status": "skipped_recovery_fuse",
            "review_attempts": attempts,
            "cost_usd": 0,
        }

    def _finish_skipped(self, job: Job) -> None:
        result = self._error_result(job, RuntimeError("Run cancelled"), status="cancelled")
        self.store.append_result(job.run_id, result)
        self.store.mutate_state(job.run_id, lambda value: _finish_job(value, None, job.attempt_id))

    def _execute(self, job: Job) -> dict[str, Any]:
        manifest = self.store.manifest(job.run_id)
        config = manifest["config"]
        definition = self.suite.get(job.eval_id)
        rendered = definition.render(int(config["seed"]))
        started_at = utc_now()
        start = time.monotonic()
        outputs: list[str] = []
        turn_receipts: list[dict[str, Any]] = []
        total_usage: dict[str, float] = {}

        if job.condition == "agentic":
            messages = list(rendered.messages)
            for sample_index in range(1, 11):
                completion = self._complete(job, definition, messages, config, sample_index)
                outputs.append(completion.text)
                turn_receipts.append(completion_receipt(completion, list(messages), sample_index))
                merge_usage(total_usage, completion.usage)
                if not completion.text.strip():
                    break
                provisional = grade_attempt(definition, [completion.text], parameters=rendered.parameters)
                if provisional.get("pass"):
                    break
                failed = [name for name, passed in provisional.get("gates", {}).items() if not passed]
                messages.extend(
                    [
                        {"role": "assistant", "content": completion.text},
                        {
                            "role": "user",
                            "content": "The standardized harness found these failed gates: " + ", ".join(failed) + ". Return a complete revised single HTML file that fixes them. Output only HTML.",
                        },
                    ]
                )
        elif definition.execution == "independent":
            contexts = [[{"role": "user", "content": prompt}] for prompt in rendered.prompts]
            for sample_index, messages in enumerate(contexts, 1):
                completion = self._complete(job, definition, messages, config, sample_index)
                outputs.append(completion.text)
                turn_receipts.append(completion_receipt(completion, messages, sample_index))
                merge_usage(total_usage, completion.usage)
        elif definition.execution == "repeated":
            for sample_index in range(1, definition.sample_count + 1):
                completion = self._complete(job, definition, list(rendered.messages), config, sample_index, force_no_seed=True)
                outputs.append(completion.text)
                turn_receipts.append(completion_receipt(completion, rendered.messages, sample_index))
                merge_usage(total_usage, completion.usage)
        elif definition.execution == "multi_turn":
            messages = list(rendered.messages)
            for sample_index in range(1, len(rendered.followups) + 2):
                completion = self._complete(job, definition, messages, config, sample_index)
                outputs.append(completion.text)
                turn_receipts.append(completion_receipt(completion, list(messages), sample_index))
                merge_usage(total_usage, completion.usage)
                if not completion.text.strip():
                    break
                messages.append({"role": "assistant", "content": completion.text})
                if sample_index <= len(rendered.followups):
                    messages.append({"role": "user", "content": rendered.followups[sample_index - 1]})
        else:
            completion = self._complete(job, definition, list(rendered.messages), config, 1)
            outputs.append(completion.text)
            turn_receipts.append(completion_receipt(completion, rendered.messages, 1))
            merge_usage(total_usage, completion.usage)

        raw_dir = self.store.run_dir(job.run_id) / "raw" / job.attempt_id
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_paths = []
        for index, output in enumerate(outputs, 1):
            path = raw_dir / f"response-{index:03}.txt"
            path.write_text(output, encoding="utf-8")
            raw_paths.append(str(path.relative_to(self.store.run_dir(job.run_id))))

        grade = grade_attempt(definition, outputs, parameters=rendered.parameters)
        artifacts = self._artifacts(job, definition, outputs, grade)
        last = turn_receipts[-1] if turn_receipts else {}
        return {
            "attempt_id": job.attempt_id, "run_id": job.run_id, "status": "ok", "model": job.model,
            "resolved_model": last.get("resolved_model"), "provider": last.get("provider"),
            "eval_id": job.eval_id, "variant_id": rendered.variant_id, "condition": job.condition,
            "reasoning_mode": config["reasoning"], "temperature": config["temperature"],
            "repetition": job.repetition, "prompt_sha256": rendered.prompt_sha256,
            "started_at": started_at, "completed_at": utc_now(), "latency_seconds": round(time.monotonic() - start, 3),
            "usage": total_usage, "cost_usd": float(total_usage.get("cost", 0)), "turns": turn_receipts,
            "raw_response_paths": raw_paths, "response": outputs[-1] if outputs else "",
            "responses": outputs, "grade": grade, "artifacts": artifacts,
        }

    def _complete(
        self, job: Job, definition: EvalDefinition, messages: list[dict[str, Any]], config: dict[str, Any],
        sample_index: int, force_no_seed: bool = False,
    ) -> Any:
        seed = None if force_no_seed or config.get("seed") is None else int(config["seed"]) + job.repetition * 1000 + sample_index
        condition = "weights-only" if job.condition == "agentic" else job.condition
        with self._model_limit(job.model):
            return self.client.complete_messages(
                model=job.model, messages=messages, max_tokens=definition.max_tokens,
                temperature=config.get("temperature"), seed=seed, reasoning=config["reasoning"], condition=condition,
            )

    def _model_limit(self, model: str) -> threading.BoundedSemaphore:
        with self._model_limits_guard:
            return self._model_limits.setdefault(model, threading.BoundedSemaphore(self.per_model_workers))

    def _artifacts(self, job: Job, definition: EvalDefinition, outputs: list[str], grade: dict[str, Any]) -> list[dict[str, Any]]:
        if not definition.renderer or not outputs:
            return []
        directory = self.store.run_dir(job.run_id) / "artifacts"
        directory.mkdir(parents=True, exist_ok=True)
        if definition.renderer == "svg":
            try:
                safe_svg, _ = validate_svg(outputs[-1])
                artifact = write_svg_preview(safe_svg, directory, job.attempt_id)
                artifact["artifact_id"] = job.attempt_id
                artifact["valid"] = True
                if not artifact.get("preview"):
                    artifact["error"] = artifact.get("render_error") or "SVG renderer did not produce a preview"
                return [artifact]
            except ArtifactError as exc:
                try:
                    safe_svg, repair = salvage_svg_preview(outputs[-1])
                    artifact = write_svg_preview(safe_svg, directory, job.attempt_id)
                    artifact.update(
                        {
                            "artifact_id": job.attempt_id,
                            "valid": False,
                            "salvaged": True,
                            "discarded_bytes": repair["discarded_bytes"],
                            "error": str(exc),
                        }
                    )
                    return [artifact]
                except ArtifactError:
                    return [{"artifact_id": job.attempt_id, "kind": "svg", "valid": False, "error": str(exc)}]
        if definition.renderer == "html":
            path = directory / f"{job.attempt_id}.html.txt"
            path.write_text(extract_code(outputs[-1], "html"), encoding="utf-8")
            return [{"artifact_id": job.attempt_id, "kind": "html-source", "valid": bool(grade.get("pass")), "path": path.name}]
        return []

    def _error_result(self, job: Job, exc: Exception, status: str = "error") -> dict[str, Any]:
        return {
            "attempt_id": job.attempt_id, "run_id": job.run_id, "status": status, "model": job.model,
            "eval_id": job.eval_id, "condition": job.condition, "repetition": job.repetition,
            "completed_at": utc_now(), "error": f"{type(exc).__name__}: {exc}", "usage": {}, "cost_usd": 0,
            "grade": {"score": 0.0, "pass": False, "verdict": "Request failed", "human_required": False},
        }

    def _policy_outcome_result(self, job: Job, exc: OpenRouterPolicyError) -> dict[str, Any]:
        return {
            "attempt_id": job.attempt_id,
            "run_id": job.run_id,
            "status": "blocked",
            "outcome_type": "provider_policy_block",
            "model": job.model,
            "eval_id": job.eval_id,
            "condition": job.condition,
            "repetition": job.repetition,
            "completed_at": utc_now(),
            "provider_error": str(exc),
            "usage": {},
            "cost_usd": 0,
            "response": "",
            "responses": [],
            "grade": {
                "score": 0.0,
                "pass": False,
                "verdict": "Provider safety filter blocked the benchmark prompt",
                "human_required": False,
            },
        }

    def _refresh_report(self, run_id: str, *, force: bool = False) -> None:
        now = time.monotonic()
        with self._report_guard:
            previous = self._report_last.get(run_id, 0.0)
            if not force and now - previous < self.report_interval_seconds:
                return
            self._report_last[run_id] = now
        try:
            from .reporting_v1 import write_v1_reports

            with span("report.refresh", {"run.id": run_id, "report.forced": force, **runtime_attributes()}) as current:
                with self.store.lock(run_id):
                    paths = write_v1_reports(self.store.run_dir(run_id), self.suite)
                set_attributes(
                    current,
                    {
                        "report.json_bytes": paths["json"].stat().st_size,
                        "report.markdown_bytes": paths["markdown"].stat().st_size,
                        **runtime_attributes(),
                    },
                )
        except Exception:
            # Results are canonical. A report can always be rebuilt later.
            return


def completion_receipt(completion: Any, messages: list[dict[str, Any]], index: int) -> dict[str, Any]:
    return {
        "index": index, "messages": messages, "response": completion.text, "response_id": completion.response_id,
        "resolved_model": completion.raw_model, "provider": completion.provider, "usage": completion.usage,
        "finish_reason": completion.finish_reason, "native_finish_reason": completion.native_finish_reason,
        "annotations": completion.annotations, "request_attempts": completion.request_attempts,
    }


def merge_usage(total: dict[str, float], usage: dict[str, Any]) -> None:
    for key, value in usage.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            total[key] = total.get(key, 0) + value


def _start_job(state: dict[str, Any], job: Job) -> None:
    state["status"] = "running"
    state["started_at"] = state.get("started_at") or utc_now()
    state["active_jobs"] = int(state.get("active_jobs", 0)) + 1
    active = dict(state.get("active_attempts", {}))
    active[job.attempt_id] = {"model": job.model, "eval_id": job.eval_id, "condition": job.condition, "started_at": utc_now()}
    state["active_attempts"] = active


def _finish_job(
    state: dict[str, Any],
    execution_ok: bool | None,
    attempt_id: str,
    *,
    benchmark_failed: bool = False,
) -> None:
    state["active_jobs"] = max(0, int(state.get("active_jobs", 0)) - 1)
    state["completed_jobs"] = int(state.get("completed_jobs", 0)) + 1
    if execution_ok is True:
        state["successful_jobs"] = int(state.get("successful_jobs", 0)) + 1
    elif execution_ok is False:
        state["failed_jobs"] = int(state.get("failed_jobs", 0)) + 1
    else:
        state["cancelled_jobs"] = int(state.get("cancelled_jobs", 0)) + 1
    state["execution_error_jobs"] = int(state.get("failed_jobs", 0))
    if benchmark_failed:
        state["benchmark_failed_jobs"] = int(state.get("benchmark_failed_jobs", 0)) + 1
    active = dict(state.get("active_attempts", {}))
    active.pop(attempt_id, None)
    state["active_attempts"] = active
    retry_attempts = set(state.get("retry_attempts") or [])
    retry_attempts.discard(attempt_id)
    state["retry_attempts"] = sorted(retry_attempts)
    if state["completed_jobs"] >= state["expected_jobs"]:
        state["status"] = "completed" if not state.get("failed_jobs") else "execution_errors"
        if state.get("cancel_requested"):
            state["status"] = "cancelled"
        state["completed_at"] = utc_now()


def build_queue_from_env(store: RunStore | None = None) -> RunQueue:
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        raise ValueError("OPENROUTER_API_KEY is required")
    workers = int(os.environ.get("QUINNFERNO_WORKERS", "12"))
    per_model_workers = int(os.environ.get("QUINNFERNO_PER_MODEL_WORKERS", "3"))
    judge_model = os.environ.get("QUINNFERNO_JUDGE_MODEL", "openai/gpt-5.6-luna-pro")
    judge_workers = int(os.environ.get("QUINNFERNO_JUDGE_WORKERS", "2"))
    max_auto_recoveries = int(os.environ.get("QUINNFERNO_MAX_AUTO_RECOVERIES", "3"))
    recovery_window_seconds = int(os.environ.get("QUINNFERNO_RECOVERY_WINDOW_SECONDS", "3600"))
    report_interval_seconds = int(os.environ.get("QUINNFERNO_REPORT_INTERVAL_SECONDS", "30"))
    max_review_attempts = int(os.environ.get("QUINNFERNO_MAX_REVIEW_ATTEMPTS", "2"))
    return RunQueue(
        key,
        store=store,
        workers=workers,
        per_model_workers=per_model_workers,
        judge_model=judge_model,
        judge_workers=judge_workers,
        max_auto_recoveries=max_auto_recoveries,
        recovery_window_seconds=recovery_window_seconds,
        report_interval_seconds=report_interval_seconds,
        max_review_attempts=max_review_attempts,
    )


def _can_model_review(result: dict[str, Any]) -> bool:
    return (
        result.get("status") == "ok"
        and bool(result.get("grade", {}).get("human_required"))
        and bool(result.get("attempt_id"))
    )


def estimate_cost(
    config: RunConfig,
    suite: EvalSuite,
    catalog: dict[str, dict[str, Any]],
    judge_model: str | None = None,
) -> float:
    from .costing import estimate_run_cost

    return float(estimate_run_cost(config, suite, catalog, judge_model)["total"]["expected"])
