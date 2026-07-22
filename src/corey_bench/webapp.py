from __future__ import annotations

import json
import hashlib
import os
import secrets
import time
from pathlib import Path
from typing import Any

from flask import Flask, Response, abort, flash, jsonify, redirect, render_template, request, send_file, session, url_for

from .cli import _load_dotenv
from .openrouter import OpenRouterError
from .protocol import load_protocol
from .reporting_v1 import build_report_data, write_v1_reports
from .runner import RunConfig, RunQueue, RunStore, append_jsonl, build_queue_from_env


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
                self.models = sorted(self.queue.client.list_models(), key=lambda item: (str(item.get("name", "")).casefold(), item["id"]))
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
    root = Path(runs_root or os.environ.get("QUINNFERNO_RUNS", "runs"))
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
            recent=store.recent()[:20], default_cases={"1.2", "1.4", "3.2", "3.3", "5.7", "5.8", "6.4"},
        )

    @app.get("/api/models")
    def api_models() -> Response:
        models = catalog.get(force=request.args.get("refresh") == "1")
        return jsonify({"data": models, "error": catalog.error})

    @app.post("/runs")
    def create_run() -> Response:
        models = list(dict.fromkeys(value.strip() for value in request.form.getlist("models") if value.strip()))
        custom = [value.strip() for value in request.form.get("custom_models", "").splitlines() if value.strip()]
        models = list(dict.fromkeys(models + custom))
        eval_ids = list(dict.fromkeys(request.form.getlist("evals")))
        conditions = list(dict.fromkeys(request.form.getlist("conditions") or ["weights-only"]))
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
        return render_template("run.html", state=state, manifest=manifest, results=results, report=report)

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
        reviews = {row.get("attempt_id"): row for row in _read_reviews(run_dir / "reviews.jsonl")}
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
            {"attempt_id": attempt_id, "reviewed_at": _now(), "score": score, "notes": request.form.get("notes", ""), "verdict": request.form.get("verdict", "")[:120]},
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
        return jsonify({"status": "ok", "queue_depth": queue_manager._queue.qsize()})

    @app.get("/readyz")
    def ready() -> Response:
        return jsonify({"status": "ready", "protocol": suite.version})

    return app


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


def _read_reviews(path: Path) -> list[dict[str, Any]]:
    from .runner import read_jsonl

    return read_jsonl(path)


def _now() -> str:
    from .runner import utc_now

    return utc_now()


def serve(host: str = "127.0.0.1", port: int = 8765, debug: bool = False) -> None:
    app = create_app()
    app.run(host=host, port=port, debug=debug, use_reloader=False, threaded=True)
