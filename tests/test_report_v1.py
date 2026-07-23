import tempfile
import unittest

from corey_bench.protocol import load_protocol
from corey_bench.reporting_v1 import build_report_data, report_markdown
from corey_bench.runner import RunConfig, RunStore


class ReportV1Tests(unittest.TestCase):
    def test_all_pass_uses_worst_attempt_and_all_cost(self):
        with tempfile.TemporaryDirectory() as directory:
            store = RunStore(directory)
            run_id, jobs = store.create(
                RunConfig(models=["example/model"], eval_ids=["3.2"], conditions=["weights-only"], repetitions=3),
                load_protocol(),
            )
            for job, score in zip(jobs, [1.0, 0.8, 0.6]):
                store.append_result(
                    run_id,
                    {
                        "attempt_id": job.attempt_id, "status": "ok", "model": job.model,
                        "eval_id": job.eval_id, "condition": job.condition, "repetition": job.repetition,
                        "cost_usd": 0.01,
                        "latency_seconds": 1, "usage": {"total_tokens": 10},
                        "grade": {"score": score, "pass": score >= 0.8, "human_required": False, "verdict": "fixture"},
                    },
                )
            data = build_report_data(store.run_dir(run_id))
            model = data["models"][0]
            self.assertAlmostEqual(model["weighted_points"], 6.0)
            self.assertAlmostEqual(model["total_cost"], 0.03)
            self.assertIsNone(model["final_score"])
            self.assertFalse(model["rankable"])
            self.assertEqual(model["completed_required_attempts"], 3)
            self.assertEqual(model["required_attempts"], 97)
            self.assertIn("6.0/10 provisional", report_markdown(data))

    def test_only_a_complete_current_protocol_run_is_rankable(self):
        with tempfile.TemporaryDirectory() as directory:
            suite = load_protocol()
            store = RunStore(directory)
            run_id, jobs = store.create(
                RunConfig(
                    models=["example/model"],
                    eval_ids=[definition.id for definition in suite.evals],
                    conditions=["weights-only", "search-enabled", "agentic"],
                ),
                suite,
            )
            for job in jobs:
                store.append_result(
                    run_id,
                    {
                        "attempt_id": job.attempt_id,
                        "status": "ok",
                        "model": job.model,
                        "eval_id": job.eval_id,
                        "condition": job.condition,
                        "repetition": job.repetition,
                        "cost_usd": 0.01,
                        "latency_seconds": 1,
                        "usage": {"total_tokens": 10},
                        "grade": {
                            "score": 0.75,
                            "pass": True,
                            "human_required": False,
                            "verdict": "fixture",
                        },
                    },
                )

            model = build_report_data(store.run_dir(run_id), suite)["models"][0]

            self.assertEqual(len(jobs), 97)
            self.assertEqual(model["completed_required_attempts"], 97)
            self.assertEqual(model["missing_test_count"], 0)
            self.assertTrue(model["full_suite_complete"])
            self.assertTrue(model["rankable"])
            self.assertAlmostEqual(model["final_score"], 75.0)


if __name__ == "__main__":
    unittest.main()
