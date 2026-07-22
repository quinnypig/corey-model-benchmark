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

from .artifacts import ArtifactError, extract_code, validate_svg, write_svg_preview
from .graders import grade_attempt
from .openrouter import OpenRouterClient, OpenRouterError
from .protocol import EvalDefinition, EvalSuite, RenderedEval, load_protocol


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
        for model in config.models:
            for eval_id in config.eval_ids:
                definition = suite.get(eval_id)
                repetitions = config.repetitions or definition.repetitions
                for condition in config.conditions:
                    if condition not in definition.conditions and condition != "agentic":
                        continue
                    if condition == "agentic" and eval_id not in {"2.1", "2.2"}:
                        continue
                    for repetition in range(1, repetitions + 1):
                        key = f"{run_id}\0{model}\0{eval_id}\0{condition}\0{repetition}"
                        attempt_id = hashlib.sha256(key.encode()).hexdigest()[:20]
                        jobs.append(Job(run_id, model, eval_id, condition, repetition, attempt_id))
                        request_estimate += condition_request_count(definition, rendered[eval_id], condition)
        manifest = {
            "protocol_version": "1.0",
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
            return read_jsonl(self.run_dir(run_id) / "results.jsonl")

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
    return total


class RunQueue:
    def __init__(
        self,
        api_key: str,
        *,
        store: RunStore | None = None,
        suite: EvalSuite | None = None,
        workers: int = 3,
        timeout: float = 300,
        attempts: int = 5,
    ) -> None:
        self.store = store or RunStore()
        self.suite = suite or load_protocol()
        self.client = OpenRouterClient(api_key, timeout=timeout, attempts=attempts)
        self.workers = max(1, min(int(workers), 8))
        self._queue: queue.Queue[Job | None] = queue.Queue()
        self._threads: list[threading.Thread] = []
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        for number in range(self.workers):
            thread = threading.Thread(target=self._worker, name=f"quinnferno-worker-{number+1}", daemon=True)
            thread.start()
            self._threads.append(thread)
        self.recover()

    def submit(self, config: RunConfig) -> str:
        if len(config.models) > 10:
            raise ValueError("A run may contain at most 10 models")
        if not config.models or not config.eval_ids:
            raise ValueError("Select at least one model and one eval")
        catalog = self.client.list_models()
        available = {item["id"]: item for item in catalog}
        missing = [model for model in config.models if model not in available]
        if missing:
            raise OpenRouterError("Requested model ID(s) are unavailable for this API key: " + ", ".join(missing))
        config = replace(config, estimated_cost_usd=estimate_cost(config, self.suite, available))
        run_id, jobs = self.store.create(config, self.suite)
        for job in jobs:
            self._queue.put(job)
        return run_id

    def recover(self) -> None:
        for state in self.store.recent():
            if state.get("status") not in {"queued", "running", "interrupted"}:
                continue
            run_id = state["run_id"]
            completed = {row.get("attempt_id") for row in self.store.results(run_id)}
            manifest = self.store.manifest(run_id)
            self.store.update_state(run_id, status="queued", active_jobs=0)
            for raw in manifest.get("jobs", []):
                if raw.get("attempt_id") not in completed:
                    self._queue.put(Job(**raw))

    def cancel(self, run_id: str) -> None:
        self.store.update_state(run_id, cancel_requested=True, status="cancelling")

    def _worker(self) -> None:
        while True:
            job = self._queue.get()
            if job is None:
                return
            try:
                state = self.store.state(job.run_id)
                if state.get("cancel_requested"):
                    self._finish_skipped(job)
                    continue
                manifest = self.store.manifest(job.run_id)
                budget = manifest.get("config", {}).get("max_budget_usd")
                spent = sum(float(row.get("cost_usd") or 0) for row in self.store.results(job.run_id))
                if budget is not None and spent >= float(budget):
                    self.store.update_state(job.run_id, cancel_requested=True, status="budget_exhausted")
                    self._finish_skipped(job)
                    continue
                existing = {row.get("attempt_id") for row in self.store.results(job.run_id)}
                if job.attempt_id in existing:
                    continue
                self.store.mutate_state(job.run_id, lambda value: _start_job(value, job))
                result = self._execute(job)
                self.store.append_result(job.run_id, result)
                self.store.mutate_state(job.run_id, lambda value: _finish_job(value, result.get("status") == "ok", job.attempt_id))
                self._refresh_report(job.run_id)
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

    def _finish_skipped(self, job: Job) -> None:
        result = self._error_result(job, RuntimeError("Run cancelled"), status="cancelled")
        self.store.append_result(job.run_id, result)
        self.store.mutate_state(job.run_id, lambda value: _finish_job(value, False, job.attempt_id))

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
        return self.client.complete_messages(
            model=job.model, messages=messages, max_tokens=definition.max_tokens,
            temperature=config.get("temperature"), seed=seed, reasoning=config["reasoning"], condition=condition,
        )

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
                return [artifact]
            except ArtifactError as exc:
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

    def _refresh_report(self, run_id: str) -> None:
        try:
            from .reporting_v1 import write_v1_reports

            with self.store.lock(run_id):
                write_v1_reports(self.store.run_dir(run_id), self.suite)
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


def _finish_job(state: dict[str, Any], success: bool, attempt_id: str) -> None:
    state["active_jobs"] = max(0, int(state.get("active_jobs", 0)) - 1)
    state["completed_jobs"] = int(state.get("completed_jobs", 0)) + 1
    key = "successful_jobs" if success else "failed_jobs"
    state[key] = int(state.get(key, 0)) + 1
    active = dict(state.get("active_attempts", {}))
    active.pop(attempt_id, None)
    state["active_attempts"] = active
    if state["completed_jobs"] >= state["expected_jobs"]:
        state["status"] = "completed" if not state.get("failed_jobs") else "completed_with_errors"
        if state.get("cancel_requested"):
            state["status"] = "cancelled"
        state["completed_at"] = utc_now()


def build_queue_from_env(store: RunStore | None = None) -> RunQueue:
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        raise ValueError("OPENROUTER_API_KEY is required")
    workers = int(os.environ.get("QUINNFERNO_WORKERS", "3"))
    return RunQueue(key, store=store, workers=workers)


def estimate_cost(config: RunConfig, suite: EvalSuite, catalog: dict[str, dict[str, Any]]) -> float:
    total = 0.0
    for model in config.models:
        pricing = catalog.get(model, {}).get("pricing", {})
        try:
            prompt_rate = float(pricing.get("prompt") or 0)
            completion_rate = float(pricing.get("completion") or 0)
            request_rate = float(pricing.get("request") or 0)
        except (TypeError, ValueError):
            prompt_rate = completion_rate = request_rate = 0.0
        for eval_id in config.eval_ids:
            definition = suite.get(eval_id)
            rendered = definition.render(config.seed)
            repetitions = config.repetitions or definition.repetitions
            if definition.execution == "repeated":
                estimated_output = 3
            else:
                estimated_output = min(definition.max_tokens, 1800) * 0.55
            prompt_chars = len(definition.prompt) + sum(map(len, rendered.prompts)) + sum(map(len, rendered.followups))
            estimated_input = max(30, prompt_chars / 4 / max(1, request_count(definition, rendered)))
            for condition in config.conditions:
                allowed = condition in definition.conditions or (
                    condition == "agentic" and eval_id in {"2.1", "2.2"}
                )
                if not allowed:
                    continue
                calls = condition_request_count(definition, rendered, condition) * repetitions
                total += calls * (request_rate + estimated_input * prompt_rate + estimated_output * completion_rate)
                if condition == "search-enabled":
                    total += calls * 0.005
    return round(total, 4)
