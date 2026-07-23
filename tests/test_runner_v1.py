import tempfile
import unittest
from datetime import datetime, timezone

from corey_bench.protocol import load_protocol
from corey_bench.runner import RunConfig, RunQueue, RunStore, suite_request_count


class RunnerV1Tests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
