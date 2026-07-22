import unittest

from corey_bench.costing import estimate_run_cost
from corey_bench.protocol import load_protocol
from corey_bench.runner import RunConfig


class CostEstimatorTests(unittest.TestCase):
    def test_full_suite_cost_is_a_tokenizer_aware_range(self):
        suite = load_protocol()
        config = RunConfig(
            models=["openai/gpt-4o"],
            eval_ids=[definition.id for definition in suite.evals],
            conditions=["weights-only", "search-enabled", "agentic"],
        )
        catalog = {
            "openai/gpt-4o": {
                "id": "openai/gpt-4o",
                "name": "OpenAI: GPT-4o",
                "architecture": {"tokenizer": "GPT"},
                "pricing": {"prompt": "0.0000025", "completion": "0.00001", "web_search": "0.01"},
            }
        }
        estimate = estimate_run_cost(config, suite, catalog)
        model = estimate["models"][0]
        self.assertEqual(model["tokenizer_family"], "GPT")
        self.assertEqual(model["encoding"], "o200k_base")
        self.assertLess(estimate["total"]["low"], estimate["total"]["expected"])
        self.assertLess(estimate["total"]["expected"], estimate["total"]["high"])
        self.assertGreater(model["tokens"]["expected"]["input"], 0)
        self.assertGreater(model["tokens"]["expected"]["output"], 0)

    def test_missing_catalog_models_are_reported(self):
        suite = load_protocol()
        config = RunConfig(
            models=["missing/model"], eval_ids=["1.2"], conditions=["weights-only"]
        )
        estimate = estimate_run_cost(config, suite, {})
        self.assertEqual(estimate["missing"], ["missing/model"])
        self.assertEqual(estimate["total"]["expected"], 0)

    def test_frontier_judge_cost_is_included(self):
        suite = load_protocol()
        config = RunConfig(models=["candidate/model"], eval_ids=["1.1"], conditions=["weights-only"])
        catalog = {
            "candidate/model": {"id": "candidate/model", "architecture": {"tokenizer": "GPT"}, "pricing": {"prompt": "0.000001", "completion": "0.000002"}},
            "judge/model": {"id": "judge/model", "architecture": {"tokenizer": "GPT"}, "pricing": {"prompt": "0.000001", "completion": "0.000006"}},
        }
        without_judge = estimate_run_cost(config, suite, catalog)
        with_judge = estimate_run_cost(config, suite, catalog, "judge/model")
        self.assertGreater(with_judge["total"]["expected"], without_judge["total"]["expected"])
        self.assertGreater(with_judge["models"][0]["review_cost"]["expected"], 0)


if __name__ == "__main__":
    unittest.main()
