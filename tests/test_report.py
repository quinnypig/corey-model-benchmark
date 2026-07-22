import json
import tempfile
import unittest
from pathlib import Path

from corey_bench.report import build_report


class ReportTests(unittest.TestCase):
    def test_report_combines_automated_and_human_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            manifest = {
                "run_id": "test-run",
                "suite": {"name": "Test suite", "version": "1"},
                "models": ["example/model"],
                "temperature": 0.1,
                "reasoning": "on",
                "repetitions": 1,
                "cases": [{"id": "case-1", "title": "Case one"}],
            }
            result = {
                "status": "ok",
                "model": "example/model",
                "case_id": "case-1",
                "repetition": 1,
                "latency_seconds": 2.5,
                "usage": {"total_tokens": 100, "cost": 0.001},
                "automated": {
                    "score": 0.5,
                    "checks": [
                        {"passed": False, "label": "important check", "observed": "missing"}
                    ],
                },
            }
            review = {
                "model": "example/model",
                "case_id": "case-1",
                "repetition": 1,
                "scores": [{"score": 4, "weight": 2}],
            }
            (run_dir / "manifest.json").write_text(json.dumps(manifest))
            (run_dir / "results.jsonl").write_text(json.dumps(result) + "\n")
            (run_dir / "reviews.jsonl").write_text(json.dumps(review) + "\n")

            report = build_report(run_dir)
            self.assertIn("50.0%", report)
            self.assertIn("4.00/5", report)
            self.assertIn("100", report)
            self.assertIn("Automated failure ledger", report)


if __name__ == "__main__":
    unittest.main()
