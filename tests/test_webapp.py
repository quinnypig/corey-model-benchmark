import tempfile
import unittest
import os
from pathlib import Path

from corey_bench.protocol import load_protocol
from corey_bench.runner import RunConfig, RunStore
from corey_bench.webapp import _pricing_label, create_app


class FakeClient:
    def list_models(self):
        return [{"id": "example/model", "name": "Example Model", "context_length": 1000, "pricing": {}}]


class FakeQueue:
    def __init__(self, root):
        self.store = RunStore(root)
        self.suite = load_protocol()
        self.client = FakeClient()
        self._queue = type("Q", (), {"qsize": lambda self: 0})()

    def start(self):
        pass

    def submit(self, config):
        self.submitted = config
        return "full-suite-test"

    def retry_execution_errors(self, run_id):
        self.retried_run_id = run_id
        return 1

    def retry_pending_reviews(self, run_id):
        self.retried_reviews_run_id = run_id
        return 1


class WebAppTests(unittest.TestCase):
    def test_roster_pricing_labels_are_per_million(self):
        self.assertEqual(
            _pricing_label({"prompt": "0.0000001", "completion": "0.0000002"}),
            "$0.1 in / $0.2 out per 1M",
        )
        self.assertEqual(_pricing_label({"prompt": "0", "completion": "0"}), "free inference")
        self.assertEqual(_pricing_label({}), "pricing unavailable")

    def test_dashboard_health_and_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            app = create_app(run_queue=FakeQueue(Path(directory)))
            app.config["TESTING"] = True
            client = app.test_client()
            page = client.get("/")
            self.assertEqual(page.status_code, 200)
            self.assertIn(b"Quinnferno", page.data)
            self.assertIn(b"example/model", page.data)
            self.assertIn(b"Most recently added", page.data)
            self.assertIn(b"All 29 evals", page.data)
            self.assertNotIn(b'name="evals"', page.data)
            self.assertEqual(client.get("/healthz").status_code, 200)
            self.assertEqual(client.get("/runs/%2e%2e%2fetc").status_code, 404)

    def test_submission_always_queues_the_complete_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            queue = FakeQueue(Path(directory))
            app = create_app(run_queue=queue)
            app.config["TESTING"] = True
            client = app.test_client()
            client.get("/")
            with client.session_transaction() as browser_session:
                csrf = browser_session["csrf"]
            response = client.post("/runs", data={"csrf": csrf, "models": "example/model"})
            self.assertEqual(response.status_code, 302)
            self.assertEqual(len(queue.submitted.eval_ids), 29)
            self.assertEqual(queue.submitted.conditions, ["weights-only", "search-enabled", "agentic"])

    def test_cost_estimate_endpoint_uses_selected_models(self):
        with tempfile.TemporaryDirectory() as directory:
            app = create_app(run_queue=FakeQueue(Path(directory)))
            app.config["TESTING"] = True
            client = app.test_client()
            client.get("/")
            with client.session_transaction() as browser_session:
                csrf = browser_session["csrf"]
            response = client.post(
                "/api/estimate",
                json={"models": ["example/model"]},
                headers={"X-CSRF-Token": csrf},
            )
            self.assertEqual(response.status_code, 200)
            estimate = response.get_json()
            self.assertEqual(estimate["models"][0]["id"], "example/model")
            self.assertIn("expected", estimate["total"])

    def test_relative_runs_root_supports_report_download(self):
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                fake_queue = FakeQueue(Path("runs").resolve())
                run_id, jobs = fake_queue.store.create(
                    RunConfig(models=["example/model"], eval_ids=["3.2"], conditions=["weights-only"], repetitions=1),
                    fake_queue.suite,
                )
                fake_queue.store.append_result(
                    run_id,
                    {
                        "attempt_id": jobs[0].attempt_id, "status": "ok", "model": "example/model",
                        "eval_id": "3.2", "condition": "weights-only", "cost_usd": 0,
                        "latency_seconds": 1, "usage": {},
                        "response": "ALLOWED — same-account identity policy is sufficient.",
                        "responses": ["ALLOWED — same-account identity policy is sufficient."],
                        "turns": [{"index": 1, "messages": [{"role": "user", "content": "Is it allowed?"}], "response": "ALLOWED — same-account identity policy is sufficient.", "usage": {"cost": 0.001}}],
                        "grade": {"score": 1, "pass": True, "human_required": False, "verdict": "pass"},
                    },
                )
                app = create_app(run_queue=fake_queue)
                app.config["TESTING"] = True
                response = app.test_client().get(f"/runs/{run_id}/report.json")
                self.assertEqual(response.status_code, 200)
                self.assertIn(b'"models"', response.data)
                client = app.test_client()
                models_page = client.get("/models").data
                self.assertIn(b"Tested model roster", models_page)
                self.assertIn(b"not ranked", models_page)
                model_page = client.get("/models/example/model").data
                self.assertIn(b"Capability profile", model_page)
                self.assertIn(b"not yet a valid model card", model_page)
                self.assertIn(b"1 of 97 required attempts", model_page)
                self.assertIn(b"ALLOWED", client.get("/responses?model=example/model&eval=3.2").data)
                exported = client.get("/models/example/model/responses.jsonl")
                self.assertEqual(exported.status_code, 200)
                self.assertIn(b'"question": "Is it allowed?"', exported.data)
                archive = client.get("/models/example/model/responses.zip")
                self.assertEqual(archive.status_code, 200)
                self.assertEqual(archive.mimetype, "application/zip")
            finally:
                os.chdir(previous)

    def test_run_page_separates_execution_errors_from_failed_tests(self):
        with tempfile.TemporaryDirectory() as directory:
            queue = FakeQueue(Path(directory))
            run_id, jobs = queue.store.create(
                RunConfig(
                    models=["example/model"],
                    eval_ids=["3.2"],
                    conditions=["weights-only"],
                    repetitions=1,
                ),
                queue.suite,
            )
            queue.store.append_result(
                run_id,
                {
                    "attempt_id": jobs[0].attempt_id,
                    "status": "error",
                    "model": jobs[0].model,
                    "eval_id": jobs[0].eval_id,
                    "condition": jobs[0].condition,
                    "repetition": jobs[0].repetition,
                    "error": "upstream connection reset",
                    "grade": {"score": 0, "pass": False, "human_required": False},
                },
            )
            queue.store.update_state(
                run_id,
                status="execution_errors",
                completed_jobs=1,
                failed_jobs=1,
                execution_error_jobs=1,
            )
            app = create_app(run_queue=queue)
            app.config["TESTING"] = True
            client = app.test_client()

            page = client.get(f"/runs/{run_id}")
            self.assertEqual(page.status_code, 200)
            self.assertIn(b"1 execution error needs repair", page.data)
            self.assertIn(b"infrastructure/provider errors, not failed tests", page.data)
            self.assertIn(b"upstream connection reset", page.data)
            self.assertIn(b"no benchmark result until repaired", page.data)

            client.get("/")
            with client.session_transaction() as browser_session:
                csrf = browser_session["csrf"]
            response = client.post(
                f"/runs/{run_id}/retry-errors",
                data={"csrf": csrf},
            )
            self.assertEqual(response.status_code, 302)
            self.assertEqual(queue.retried_run_id, run_id)


if __name__ == "__main__":
    unittest.main()
