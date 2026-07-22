import tempfile
import unittest
from pathlib import Path

from corey_bench.protocol import load_protocol
from corey_bench.runner import RunConfig, RunStore, suite_request_count


class RunnerV1Tests(unittest.TestCase):
    def test_full_suite_request_upper_bound_includes_agentic_turns(self):
        self.assertEqual(suite_request_count(load_protocol()), 528)

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
            self.assertEqual(manifest["estimated_requests"], 16)
            self.assertTrue(all(item["prompt_sha256"] for item in manifest["evals"]))


if __name__ == "__main__":
    unittest.main()
