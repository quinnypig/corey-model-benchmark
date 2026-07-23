import tempfile
import unittest

from corey_bench.model_cards import build_model_cards, build_model_comparison
from corey_bench.protocol import load_protocol
from corey_bench.runner import RunConfig, RunStore


class ModelCardTests(unittest.TestCase):
    def test_partial_high_score_never_outranks_complete_suite(self):
        with tempfile.TemporaryDirectory() as directory:
            suite = load_protocol()
            store = RunStore(directory)
            partial_id, partial_jobs = store.create(
                RunConfig(
                    models=["example/partial"],
                    eval_ids=["3.2"],
                    conditions=["weights-only"],
                ),
                suite,
            )
            for job in partial_jobs:
                store.append_result(partial_id, self._result(job, 1.0))

            full_id, full_jobs = store.create(
                RunConfig(
                    models=["example/complete"],
                    eval_ids=[definition.id for definition in suite.evals],
                    conditions=["weights-only", "search-enabled", "agentic"],
                ),
                suite,
            )
            for job in full_jobs:
                store.append_result(full_id, self._result(job, 0.5))

            cards = build_model_cards(store, suite)

            self.assertEqual(cards[0]["id"], "example/complete")
            self.assertTrue(cards[0]["rankable"])
            self.assertEqual(cards[0]["representative_run"]["score_percent"], 50.0)
            partial = next(card for card in cards if card["id"] == "example/partial")
            self.assertFalse(partial["valid"])
            self.assertFalse(partial["rankable"])
            self.assertIsNone(partial["representative_run"]["score_percent"])
            self.assertEqual(partial["representative_run"]["provisional_score_percent"], 100.0)
            self.assertEqual(partial["representative_run"]["completed_required_attempts"], 3)

            comparison = build_model_comparison(cards[:1], suite)
            self.assertEqual(len(comparison["sections"]), 7)
            aws = next(section for section in comparison["sections"] if section["tier"] == 3)
            self.assertEqual(aws["title"], "AWS reasoning")
            iam = next(test for test in aws["tests"] if test["eval_id"] == "3.2")
            self.assertEqual(iam["cells"][0]["weights_score"], 0.5)

    @staticmethod
    def _result(job, score):
        return {
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
                "score": score,
                "pass": score >= 0.5,
                "human_required": False,
                "verdict": "fixture",
            },
        }


if __name__ == "__main__":
    unittest.main()
