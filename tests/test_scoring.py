import unittest

from corey_bench.scoring import score_response
from corey_bench.suite import Check


class ScoringTests(unittest.TestCase):
    def test_text_checks_are_case_insensitive(self):
        result = score_response(
            [
                Check(type="contains", value="fargate"),
                Check(type="not_contains", values=["you are right"]),
                Check(type="max_words", value=4),
            ],
            "Fargate is priced differently.",
        )
        self.assertEqual(result["score"], 1.0)

    def test_json_checks_reject_markdown_fences_and_extra_keys(self):
        checks = [
            Check(type="json_valid"),
            Check(type="json_keys_exact", values=["answer"]),
        ]
        valid = score_response(checks, '{"answer": true}')
        fenced = score_response(checks, '```json\n{"answer": true}\n```')
        extra = score_response(checks, '{"answer": true, "bonus": 1}')
        self.assertEqual(valid["score"], 1.0)
        self.assertEqual(fenced["score"], 0.0)
        self.assertEqual(extra["score"], 0.5)

    def test_regex_flags(self):
        result = score_response(
            [Check(type="regex", value="^verdict:", flags="im")],
            "Intro\nVERDICT: false",
        )
        self.assertEqual(result["score"], 1.0)


if __name__ == "__main__":
    unittest.main()
