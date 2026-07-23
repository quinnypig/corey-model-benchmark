from __future__ import annotations

import json
import hashlib
import io
import os
import secrets
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from flask import Flask, Response, abort, flash, jsonify, redirect, render_template, request, send_file, session, url_for

from .cli import _load_dotenv
from .costing import estimate_run_cost
from .model_cards import build_model_cards
from .openrouter import OpenRouterError
from .reporting_v1 import build_report_data, write_v1_reports
from .responses import response_records
from .runner import RunConfig, RunQueue, RunStore, append_jsonl, build_queue_from_env, suite_request_count
from .telemetry import instrument_flask, runtime_attributes


FULL_CONDITIONS = ["weights-only", "search-enabled", "agentic"]


class ModelCatalog:
    def __init__(self, run_queue: RunQueue, ttl: int = 300) -> None:
        self.queue = run_queue
        self.ttl = ttl
        self.loaded_at = 0.0
        self.models: list[dict[str, Any]] = []
        self.error: str | None = None

    def get(self, force: bool = False) -> list[dict[str, Any]]:
        if force or not self.models or time.monotonic() - self.loaded_at > self.ttl:
            try:
                models = self.queue.client.list_models()
                for model in models:
                    try:
                        created = float(model.get("created") or 0)
                        model["created_label"] = datetime.fromtimestamp(created, timezone.utc).strftime("%b %d, %Y") if created else ""
                    except (TypeError, ValueError, OSError):
                        model["created_label"] = ""
                    model["pricing_label"] = _pricing_label(model.get("pricing"))
                self.models = sorted(
                    models,
                    key=lambda item: (-_model_created(item), str(item.get("name", "")).casefold(), item["id"]),
                )
                self.error = None
                self.loaded_at = time.monotonic()
            except OpenRouterError as exc:
                self.error = str(exc)
        return self.models


def create_app(*, runs_root: str | Path | None = None, run_queue: RunQueue | None = None) -> Flask:
    _load_dotenv()
    app = Flask(__name__)
    app.config.update(
        SECRET_KEY=os.environ.get("QUINNFERNO_SECRET_KEY") or secrets.token_hex(32),
        MAX_CONTENT_LENGTH=1_000_000,
    )
    instrument_flask(app)
    root = Path(runs_root or os.environ.get("QUINNFERNO_RUNS", "runs")).resolve()
    store = run_queue.store if run_queue else RunStore(root)
    queue_manager = run_queue or build_queue_from_env(store)
    queue_manager.start()
    suite = queue_manager.suite
    catalog = ModelCatalog(queue_manager)
    app.extensions["quinnferno_queue"] = queue_manager
    app.extensions["quinnferno_store"] = store

    @app.before_request
    def csrf_setup() -> Response | None:
        session.setdefault("csrf", secrets.token_urlsafe(24))
        if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
            supplied = request.form.get("csrf") or request.headers.get("X-CSRF-Token")
            if not supplied or not secrets.compare_digest(str(supplied), str(session["csrf"])):
                return Response("CSRF validation failed", status=403)
        return None

    @app.context_processor
    def common() -> dict[str, Any]:
        return {"csrf_token": session.get("csrf"), "app_name": "Quinnferno"}

    @app.get("/")
    def index() -> str:
        models = catalog.get()
        return render_template(
            "index.html", suite=suite, models=models, catalog_error=catalog.error,
            recent=store.recent()[:20], requests_per_model=suite_request_count(suite),
            requests_per_forced_repetition=suite_request_count(suite, repetitions=1),
        )

    @app.get("/api/models")
    def api_models() -> Response:
        models = catalog.get(force=request.args.get("refresh") == "1")
        return jsonify({"data": models, "error": catalog.error})

    @app.get("/models")
    def model_index() -> str:
        cards = build_model_cards(store, suite, catalog.get())
        return render_template(
            "models.html",
            cards=cards,
            rankable_count=sum(card["rankable"] for card in cards),
            run_count=len({run["run_id"] for card in cards for run in card["runs"]}),
            attempt_count=sum(card["total_attempts"] for card in cards),
            total_cost=sum(card["total_cost"] for card in cards),
        )

    @app.get("/models/<path:model_id>")
    def model_detail(model_id: str) -> str:
        cards = build_model_cards(store, suite, catalog.get())
        card = next((item for item in cards if item["id"] == model_id), None)
        if card is None:
            abort(404)
        platypus_cards = []
        for run in card["runs"]:
            try:
                for row in store.results(run["run_id"]):
                    if row.get("model") == model_id and row.get("eval_id") == "1.2":
                        platypus_cards.append({"run_id": run["run_id"], **row})
            except (OSError, ValueError):
                continue
        return render_template("model.html", card=card, platypus_cards=platypus_cards[:6])

    @app.get("/models/<path:model_id>/responses.jsonl")
    def model_responses_jsonl(model_id: str) -> Response:
        _require_tested_model(store, suite, model_id)
        rows = response_records(store, suite, model_id=model_id)
        body = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        filename = _download_name(model_id) + "-responses.jsonl"
        return Response(
            body,
            mimetype="application/x-ndjson",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.get("/models/<path:model_id>/responses.zip")
    def model_responses_zip(model_id: str) -> Response:
        card = _require_tested_model(store, suite, model_id)
        rows = response_records(store, suite, model_id=model_id)
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            bundle.writestr("model-card.json", json.dumps(card, indent=2, ensure_ascii=False, default=str) + "\n")
            bundle.writestr("responses.jsonl", "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
            for row in rows:
                name = f"raw/{row['run_id']}/{row['attempt_id']}/response-{int(row['turn']):03}.txt"
                bundle.writestr(name, str(row.get("response") or ""))
        archive.seek(0)
        return send_file(
            archive,
            mimetype="application/zip",
            as_attachment=True,
            download_name=_download_name(model_id) + "-evidence.zip",
        )

    @app.get("/responses")
    def responses() -> str:
        model_id = request.args.get("model") or None
        eval_id = request.args.get("eval") or None
        all_rows = response_records(store, suite, model_id=model_id, eval_id=eval_id)
        try:
            page = max(1, int(request.args.get("page", "1")))
        except ValueError:
            page = 1
        per_page = 40
        start = (page - 1) * per_page
        return render_template(
            "responses.html",
            rows=all_rows[start:start + per_page], total=len(all_rows), page=page,
            pages=max(1, (len(all_rows) + per_page - 1) // per_page),
            selected_model=model_id, selected_eval=eval_id,
            models=build_model_cards(store, suite, catalog.get()), evals=suite.evals,
        )

    @app.post("/api/estimate")
    def api_estimate() -> Response:
        payload = request.get_json(silent=True) or {}
        models = list(dict.fromkeys(str(value).strip() for value in payload.get("models", []) if str(value).strip()))
        if not models or len(models) > 10:
            return jsonify({"error": "Choose between one and ten models"}), 400
        repetitions_raw = payload.get("repetitions")
        try:
            repetitions = int(repetitions_raw) if repetitions_raw not in (None, "") else None
            if repetitions is not None and not 1 <= repetitions <= 10:
                raise ValueError
            seed = int(payload.get("seed") or 8675309)
        except (TypeError, ValueError):
            return jsonify({"error": "Invalid repetitions or seed"}), 400
        config = RunConfig(
            models=models,
            eval_ids=[definition.id for definition in suite.evals],
            conditions=FULL_CONDITIONS,
            repetitions=repetitions,
            temperature=1.0,
            reasoning=str(payload.get("reasoning") or "provider-default"),
            seed=seed,
        )
        available = {model["id"]: model for model in catalog.get()}
        judge_model = getattr(queue_manager, "judge_model", None) if getattr(queue_manager, "judge_workers", 0) else None
        return jsonify(estimate_run_cost(config, suite, available, judge_model))

    @app.post("/runs")
    def create_run() -> Response:
        models = list(dict.fromkeys(value.strip() for value in request.form.getlist("models") if value.strip()))
        custom = [value.strip() for value in request.form.get("custom_models", "").splitlines() if value.strip()]
        models = list(dict.fromkeys(models + custom))
        eval_ids = [definition.id for definition in suite.evals]
        conditions = FULL_CONDITIONS
        repetitions_raw = request.form.get("repetitions", "").strip()
        temperature_raw = request.form.get("temperature", "1.0").strip()
        config = RunConfig(
            models=models, eval_ids=eval_ids, conditions=conditions,
            repetitions=int(repetitions_raw) if repetitions_raw else None,
            temperature=float(temperature_raw) if temperature_raw else None,
            reasoning=request.form.get("reasoning", "provider-default"),
            seed=int(request.form.get("seed", "8675309")),
            max_budget_usd=float(request.form.get("max_budget_usd", "50")) if request.form.get("max_budget_usd", "").strip() else None,
        )
        try:
            run_id = queue_manager.submit(config)
        except (ValueError, OpenRouterError) as exc:
            flash(str(exc), "error")
            return redirect(url_for("index"))
        return redirect(url_for("run_detail", run_id=run_id))

    @app.get("/runs/<run_id>")
    def run_detail(run_id: str) -> str:
        run_dir = _run_dir(store, run_id)
        state = store.state(run_id)
        manifest = store.manifest(run_id)
        results = store.results(run_id)
        report = build_report_data(run_dir, suite) if results else None
        diagnostics = _run_diagnostics(run_dir, results)
        return render_template(
            "run.html",
            state=state,
            manifest=manifest,
            results=results,
            report=report,
            diagnostics=diagnostics,
        )

    @app.get("/api/runs/<run_id>")
    def run_status(run_id: str) -> Response:
        _run_dir(store, run_id)
        state = store.state(run_id)
        state["progress"] = state["completed_jobs"] / state["expected_jobs"] if state.get("expected_jobs") else 0
        return jsonify(state)

    @app.post("/runs/<run_id>/cancel")
    def cancel_run(run_id: str) -> Response:
        _run_dir(store, run_id)
        queue_manager.cancel(run_id)
        return redirect(url_for("run_detail", run_id=run_id))

    @app.post("/runs/<run_id>/resume")
    def resume_run(run_id: str) -> Response:
        _run_dir(store, run_id)
        try:
            queue_manager.resume(run_id)
        except ValueError as exc:
            flash(str(exc), "error")
        return redirect(url_for("run_detail", run_id=run_id))

    @app.post("/runs/<run_id>/retry-errors")
    def retry_run_errors(run_id: str) -> Response:
        _run_dir(store, run_id)
        try:
            count = queue_manager.retry_execution_errors(run_id)
            flash(f"Queued {count} execution-error attempt{'s' if count != 1 else ''} for repair.")
        except ValueError as exc:
            flash(str(exc), "error")
        return redirect(url_for("run_detail", run_id=run_id))

    @app.post("/runs/<run_id>/retry-reviews")
    def retry_run_reviews(run_id: str) -> Response:
        _run_dir(store, run_id)
        try:
            count = queue_manager.retry_pending_reviews(run_id)
            flash(f"Queued {count} unresolved model judgment{'s' if count != 1 else ''} for repair.")
        except ValueError as exc:
            flash(str(exc), "error")
        return redirect(url_for("run_detail", run_id=run_id))

    @app.get("/runs/<run_id>/report.md")
    def markdown_report(run_id: str) -> Response:
        run_dir = _run_dir(store, run_id)
        paths = write_v1_reports(run_dir, suite)
        return send_file(paths["markdown"], mimetype="text/markdown", as_attachment=True, download_name=f"quinnferno-{run_id}.md")

    @app.get("/runs/<run_id>/report.json")
    def json_report(run_id: str) -> Response:
        run_dir = _run_dir(store, run_id)
        paths = write_v1_reports(run_dir, suite)
        return send_file(paths["json"], mimetype="application/json", as_attachment=True, download_name=f"quinnferno-{run_id}.json")

    @app.get("/runs/<run_id>/review")
    def review(run_id: str) -> str:
        run_dir = _run_dir(store, run_id)
        reviews = {
            row.get("attempt_id"): row
            for row in _read_reviews(run_dir / "reviews.jsonl")
            if row.get("reviewer_type", "human") == "human"
        }
        candidates = []
        for row in store.results(run_id):
            if row.get("status") == "ok" and row.get("grade", {}).get("human_required") and row.get("attempt_id") not in reviews:
                blinded = dict(row)
                blinded.pop("model", None)
                candidates.append(blinded)
        return render_template("review.html", run_id=run_id, candidates=candidates, remaining=len(candidates))

    @app.post("/runs/<run_id>/review")
    def submit_review(run_id: str) -> Response:
        run_dir = _run_dir(store, run_id)
        attempt_id = request.form.get("attempt_id", "")
        result = next((row for row in store.results(run_id) if row.get("attempt_id") == attempt_id), None)
        if not result:
            abort(404)
        score = max(0, min(100, int(request.form.get("score", "0")))) / 100
        append_jsonl(
            run_dir / "reviews.jsonl",
            {"attempt_id": attempt_id, "reviewed_at": _now(), "reviewer_type": "human", "score": score, "notes": request.form.get("notes", ""), "verdict": request.form.get("verdict", "")[:120]},
        )
        write_v1_reports(run_dir, suite)
        return redirect(url_for("review", run_id=run_id))

    @app.get("/runs/<run_id>/pairwise")
    def pairwise(run_id: str) -> str:
        run_dir = _run_dir(store, run_id)
        votes = {row.get("comparison_id") for row in _read_reviews(run_dir / "comparisons.jsonl")}
        comparisons = []
        results = [row for row in store.results(run_id) if row.get("status") == "ok" and row.get("eval_id") in {"1.1", "1.2", "2.1", "2.2", "4.1"}]
        by_eval: dict[str, list[dict[str, Any]]] = {}
        for row in results:
            by_eval.setdefault(row["eval_id"], []).append(row)
        for eval_id, rows in by_eval.items():
            for left_index, left in enumerate(rows):
                for right in rows[left_index + 1:]:
                    if left.get("model") == right.get("model"):
                        continue
                    comparison_id = "-".join(sorted([left["attempt_id"], right["attempt_id"]]))
                    if comparison_id not in votes:
                        shown_left, shown_right = left, right
                        if int(hashlib.sha256(comparison_id.encode()).hexdigest(), 16) % 2:
                            shown_left, shown_right = right, left
                        comparisons.append({"comparison_id": comparison_id, "eval_id": eval_id, "left": shown_left, "right": shown_right})
        return render_template("pairwise.html", run_id=run_id, comparison=comparisons[0] if comparisons else None, remaining=len(comparisons))

    @app.post("/runs/<run_id>/pairwise")
    def submit_pairwise(run_id: str) -> Response:
        run_dir = _run_dir(store, run_id)
        choice = request.form.get("choice")
        if choice not in {"left", "right", "tie"}:
            abort(400)
        append_jsonl(run_dir / "comparisons.jsonl", {"comparison_id": request.form.get("comparison_id"), "choice": choice, "judged_at": _now()})
        return redirect(url_for("pairwise", run_id=run_id))

    @app.get("/platypuses")
    def platypuses() -> str:
        cards = []
        for state in store.recent():
            run_id = state.get("run_id")
            try:
                for row in store.results(run_id):
                    if row.get("eval_id") == "1.2":
                        cards.append({"run_id": run_id, **row})
            except (OSError, ValueError):
                continue
        return render_template("gallery.html", cards=cards)

    @app.get("/runs/<run_id>/artifacts/<artifact_id>.png")
    def png_artifact(run_id: str, artifact_id: str) -> Response:
        run_dir = _run_dir(store, run_id)
        if not _safe_id(artifact_id):
            abort(404)
        path = run_dir / "artifacts" / f"{artifact_id}.png"
        if not path.exists():
            abort(404)
        response = send_file(path, mimetype="image/png", conditional=True)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Content-Security-Policy"] = "default-src 'none'"
        return response

    @app.get("/runs/<run_id>/raw/<attempt_id>/<int:number>")
    def raw_response(run_id: str, attempt_id: str, number: int) -> Response:
        run_dir = _run_dir(store, run_id)
        if not _safe_id(attempt_id) or number < 1 or number > 500:
            abort(404)
        path = run_dir / "raw" / attempt_id / f"response-{number:03}.txt"
        if not path.exists():
            abort(404)
        return send_file(path, mimetype="text/plain", as_attachment=True, download_name=f"{attempt_id}-{number}.txt")

    @app.get("/healthz")
    def health() -> Response:
        return jsonify({
            "status": "ok", "queue_depth": queue_manager._queue.qsize(),
            "review_queue_depth": queue_manager._review_queue.qsize() if hasattr(queue_manager, "_review_queue") else 0,
            "review_jobs_active": len(queue_manager._review_inflight) if hasattr(queue_manager, "_review_inflight") else 0,
            "workers": getattr(queue_manager, "workers", None),
            "per_model_workers": getattr(queue_manager, "per_model_workers", None),
            "runtime": runtime_attributes(),
        })

    @app.get("/readyz")
    def ready() -> Response:
        return jsonify({"status": "ready", "protocol": suite.version})

    return app


def _model_created(model: dict[str, Any]) -> float:
    try:
        return float(model.get("created") or 0)
    except (TypeError, ValueError):
        return 0


def _pricing_label(pricing: Any) -> str:
    if not isinstance(pricing, dict) or "prompt" not in pricing or "completion" not in pricing:
        return "pricing unavailable"
    try:
        prompt = float(pricing.get("prompt") or 0) * 1_000_000
        completion = float(pricing.get("completion") or 0) * 1_000_000
    except (TypeError, ValueError):
        return "pricing unavailable"
    if prompt < 0 or completion < 0:
        return "pricing unavailable"
    if prompt == 0 and completion == 0:
        return "free inference"
    return f"{_format_million_rate(prompt)} in / {_format_million_rate(completion)} out per 1M"


def _format_million_rate(value: float) -> str:
    if value >= 10:
        rendered = f"{value:,.0f}"
    elif value >= 1:
        rendered = f"{value:,.2f}".rstrip("0").rstrip(".")
    elif value >= 0.01:
        rendered = f"{value:.3f}".rstrip("0").rstrip(".")
    else:
        rendered = f"{value:.5f}".rstrip("0").rstrip(".")
    return f"${rendered}"


def _run_dir(store: RunStore, run_id: str) -> Path:
    try:
        path = store.run_dir(run_id)
    except ValueError:
        abort(404)
    if not path.is_dir():
        abort(404)
    return path


def _safe_id(value: str) -> bool:
    return bool(value) and all(char.isalnum() or char in "-_" for char in value)


def _download_name(model_id: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "-" for char in model_id).strip("-") or "model"


def _require_tested_model(store: RunStore, suite: Any, model_id: str) -> dict[str, Any]:
    card = next((item for item in build_model_cards(store, suite) if item["id"] == model_id), None)
    if card is None:
        abort(404)
    return card


def _read_reviews(path: Path) -> list[dict[str, Any]]:
    from .runner import read_jsonl

    return read_jsonl(path)


def _run_diagnostics(run_dir: Path, results: list[dict[str, Any]]) -> dict[str, Any]:
    execution_errors = [row for row in results if row.get("status") == "error"]
    test_failures = [
        row
        for row in results
        if row.get("status") not in {"error", "cancelled"}
        and row.get("grade", {}).get("pass") is False
    ]
    reviews = _read_reviews(run_dir / "reviews.jsonl")
    scored_reviews = {
        str(row.get("attempt_id"))
        for row in reviews
        if isinstance(row.get("score"), (int, float))
    }
    latest_reviews: dict[str, dict[str, Any]] = {}
    for row in reviews:
        if row.get("reviewer_type") == "model" and row.get("attempt_id"):
            latest_reviews[str(row["attempt_id"])] = row
    judge_error_counts: dict[str, int] = {}
    for row in _read_reviews(run_dir / "judge-errors.jsonl"):
        attempt_id = str(row.get("attempt_id") or "")
        if attempt_id:
            judge_error_counts[attempt_id] = judge_error_counts.get(attempt_id, 0) + 1
    pending_reviews = []
    for row in results:
        attempt_id = str(row.get("attempt_id") or "")
        if (
            row.get("status") == "ok"
            and row.get("grade", {}).get("human_required")
            and attempt_id not in scored_reviews
        ):
            latest = latest_reviews.get(attempt_id, {})
            pending_reviews.append(
                {
                    **row,
                    "review_status": latest.get("status") or "queued",
                    "review_error": latest.get("error"),
                    "judge_error_count": judge_error_counts.get(attempt_id, 0),
                }
            )
    return {
        "execution_errors": execution_errors,
        "test_failures": test_failures,
        "blocked_tests": [row for row in test_failures if row.get("status") == "blocked"],
        "pending_reviews": pending_reviews,
    }


def _now() -> str:
    from .runner import utc_now

    return utc_now()


def serve(host: str = "127.0.0.1", port: int = 8765, debug: bool = False) -> None:
    app = create_app()
    app.run(host=host, port=port, debug=debug, use_reloader=False, threaded=True)
