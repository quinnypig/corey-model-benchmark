import json
import tempfile
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from corey_bench.openrouter import Completion
from corey_bench.protocol import load_protocol
from corey_bench.runner import RunConfig, RunQueue, RunStore, collapse_result_rows, suite_request_count


class RunnerV1Tests(unittest.TestCase):
    def test_append_only_result_corrections_use_latest_receipt(self):
        rows = [
            {"attempt_id": "a", "status": "error"},
            {"attempt_id": "b", "status": "ok"},
            {"attempt_id": "a", "status": "ok"},
        ]
        self.assertEqual(
            collapse_result_rows(rows),
            [{"attempt_id": "a", "status": "ok"}, {"attempt_id": "b", "status": "ok"}],
        )

    def test_full_suite_request_upper_bound_includes_agentic_and_reviews(self):
        self.assertEqual(suite_request_count(load_protocol()), 573)

    def test_job_expansion_and_manifest_receipts(self):
        with tempfile.TemporaryDirectory() as directory:
            store = RunStore(directory)
            config = RunConfig(
                models=["example/a", "example/b"], eval_ids=["1.2", "5.8"],
                conditions=["weights-only"], repetitions=2,
            )
            run_id, jobs = store.create(config, load_protocol())
            self.assertEqual(len(jobs), 8)
            manifest = store.manifest(run_id)
            self.assertEqual(manifest["expected_jobs"], 8)
            self.assertEqual(manifest["estimated_requests"], 20)
            self.assertTrue(all(item["prompt_sha256"] for item in manifest["evals"]))
            self.assertEqual([job.model for job in jobs[:4]], ["example/a", "example/b", "example/a", "example/b"])

    def test_recovery_fuse_pauses_repeated_paid_work_and_manual_resume_clears_it(self):
        with tempfile.TemporaryDirectory() as directory:
            store = RunStore(directory)
            run_id, jobs = store.create(
                RunConfig(
                    models=["example/model"],
                    eval_ids=["3.2"],
                    conditions=["weights-only"],
                    repetitions=1,
                ),
                load_protocol(),
            )
            now = datetime.now(timezone.utc).isoformat()
            store.update_state(
                run_id,
                status="running",
                active_jobs=1,
                active_attempts={jobs[0].attempt_id: {"model": jobs[0].model}},
                recovery_history=[now, now, now],
            )
            queue = RunQueue(
                "test",
                store=store,
                judge_workers=0,
                max_auto_recoveries=3,
                recovery_window_seconds=3600,
            )

            recovery = queue.recover()

            self.assertEqual(recovery["recovery.paused_run_count"], 1)
            self.assertEqual(recovery["recovery.paid_calls_at_risk"], 1)
            self.assertEqual(store.state(run_id)["status"], "recovery_paused")
            self.assertEqual(queue._queue.qsize(), 0)

            self.assertEqual(queue.resume(run_id), 1)
            self.assertEqual(store.state(run_id)["status"], "queued")
            self.assertEqual(store.state(run_id)["recovery_history"], [])
            self.assertEqual(queue._queue.qsize(), 1)

    def test_svg_artifact_migration_salvages_preview_without_changing_grade(self):
        with tempfile.TemporaryDirectory() as directory:
            store = RunStore(directory)
            run_id, jobs = store.create(
                RunConfig(
                    models=["example/model"],
                    eval_ids=["1.2"],
                    conditions=["weights-only"],
                    repetitions=1,
                ),
                load_protocol(),
            )
            grade = {"score": 0.0, "pass": False, "verdict": "Truncated SVG"}
            store.append_result(
                run_id,
                {
                    "attempt_id": jobs[0].attempt_id,
                    "run_id": run_id,
                    "status": "ok",
                    "model": jobs[0].model,
                    "eval_id": "1.2",
                    "condition": "weights-only",
                    "repetition": 1,
                    "response": '<svg xmlns="http://www.w3.org/2000/svg"><g><circle r="10"/><path d="M',
                    "grade": grade,
                    "artifacts": [{"kind": "svg", "valid": False, "error": "old validator error"}],
                },
            )
            queue = RunQueue("test", store=store, judge_workers=0)
            with patch(
                "corey_bench.runner.write_svg_preview",
                return_value={"kind": "svg", "path": "a.svg", "preview": "a.png", "render_error": None},
            ):
                stats = queue._rebuild_svg_artifacts()

            row = json.loads((store.run_dir(run_id) / "results.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(stats["artifact.rebuilt_count"], 1)
            self.assertEqual(stats["artifact.salvaged_count"], 1)
            self.assertTrue(row["artifacts"][0]["salvaged"])
            self.assertFalse(row["artifacts"][0]["valid"])
            self.assertEqual(row["grade"], grade)
            self.assertTrue((store.run_dir(run_id) / ".svg-artifacts-v3.json").exists())
            self.assertEqual(queue._rebuild_svg_artifacts()["artifact.rebuilt_run_count"], 0)

    def test_legacy_model_outcomes_are_results_but_transport_error_is_retryable(self):
        with tempfile.TemporaryDirectory() as directory:
            store = RunStore(directory)
            run_id, jobs = store.create(
                RunConfig(
                    models=["example/model"],
                    eval_ids=["1.2"],
                    conditions=["weights-only"],
                    repetitions=3,
                ),
                load_protocol(),
            )
            errors = [
                'OpenRouter HTTP 400: {"error":{"type":"content_filter","message":"considered high risk"}}',
                "OpenRouter HTTP 400: role 'assistant' must not be empty",
                "OpenRouter request failed: IncompleteRead(3586 bytes read)",
            ]
            for job, error in zip(jobs, errors):
                store.append_result(
                    run_id,
                    {
                        "attempt_id": job.attempt_id,
                        "status": "error",
                        "model": job.model,
                        "eval_id": job.eval_id,
                        "condition": job.condition,
                        "repetition": job.repetition,
                        "error": error,
                        "grade": {"score": 0, "pass": False, "human_required": False},
                    },
                )
            store.update_state(
                run_id,
                status="completed_with_errors",
                completed_jobs=3,
                successful_jobs=0,
                failed_jobs=3,
            )
            queue = RunQueue("test", store=store, judge_workers=0)

            stats = queue._migrate_result_semantics()

            self.assertEqual(stats["result.policy_outcome_count"], 1)
            self.assertEqual(stats["result.empty_outcome_count"], 1)
            self.assertEqual(
                [row["status"] for row in store.results(run_id)],
                ["blocked", "ok", "error"],
            )
            state = store.state(run_id)
            self.assertEqual(state["status"], "execution_errors")
            self.assertEqual(state["execution_error_jobs"], 1)
            self.assertEqual(state["benchmark_failed_jobs"], 2)
            self.assertEqual(state["successful_jobs"], 2)

            self.assertEqual(queue.retry_execution_errors(run_id), 1)
            state = store.state(run_id)
            self.assertEqual(state["status"], "queued")
            self.assertEqual(state["completed_jobs"], 2)
            self.assertEqual(state["execution_error_jobs"], 0)
            self.assertEqual(queue._queue.qsize(), 1)

    def test_empty_multi_turn_response_is_graded_instead_of_replayed_as_assistant(self):
        with tempfile.TemporaryDirectory() as directory:
            store = RunStore(directory)
            run_id, jobs = store.create(
                RunConfig(
                    models=["example/model"],
                    eval_ids=["5.8"],
                    conditions=["weights-only"],
                    repetitions=1,
                ),
                load_protocol(),
            )
            queue = RunQueue("test", store=store, judge_workers=0)
            completion = Completion(
                text="",
                response_id="empty",
                provider="fixture",
                usage={"total_tokens": 1, "cost": 0.001},
                raw_model="example/model",
                reasoning=None,
                finish_reason="stop",
                native_finish_reason="stop",
                annotations=[],
            )
            with patch.object(queue.client, "complete_messages", return_value=completion) as complete:
                result = queue._execute(jobs[0])

            self.assertEqual(complete.call_count, 1)
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["responses"], [""])
            self.assertFalse(result["grade"]["pass"])

    def test_manual_review_retry_bypasses_recovery_fuse_but_not_a_score(self):
        with tempfile.TemporaryDirectory() as directory:
            store = RunStore(directory)
            run_id, jobs = store.create(
                RunConfig(
                    models=["example/model"],
                    eval_ids=["1.2"],
                    conditions=["weights-only"],
                    repetitions=1,
                ),
                load_protocol(),
            )
            store.append_result(
                run_id,
                {
                    "attempt_id": jobs[0].attempt_id,
                    "status": "ok",
                    "model": jobs[0].model,
                    "eval_id": jobs[0].eval_id,
                    "condition": jobs[0].condition,
                    "repetition": jobs[0].repetition,
                    "grade": {
                        "score": 1,
                        "pass": True,
                        "human_required": True,
                        "verdict": "fixture",
                    },
                },
            )
            reviews = store.run_dir(run_id) / "reviews.jsonl"
            reviews.write_text(
                json.dumps(
                    {
                        "attempt_id": jobs[0].attempt_id,
                        "reviewer_type": "model",
                        "status": "skipped_recovery_fuse",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            queue = RunQueue("test", store=store, judge_workers=0)

            self.assertEqual(queue.retry_pending_reviews(run_id), 1)
            queued = queue._review_queue.get_nowait()
            self.assertEqual(queued.attempt_id, jobs[0].attempt_id)
            self.assertTrue(queued.force)

            with reviews.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "attempt_id": jobs[0].attempt_id,
                            "reviewer_type": "model",
                            "score": 0.75,
                        }
                    )
                    + "\n"
                )
            with self.assertRaisesRegex(ValueError, "No unresolved"):
                queue.retry_pending_reviews(run_id)


if __name__ == "__main__":
    unittest.main()
