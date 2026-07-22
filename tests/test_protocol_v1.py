import unittest

from corey_bench.protocol import load_protocol


class ProtocolV1Tests(unittest.TestCase):
    def test_complete_suite_and_weights(self):
        suite = load_protocol()
        self.assertEqual(len(suite.evals), 29)
        self.assertEqual(suite.weighted_total, 100)
        self.assertTrue(all(item.weight == 0 for item in suite.evals if item.tier == 7))

    def test_render_is_deterministic_and_omits_system(self):
        definition = load_protocol().get("1.4")
        first = definition.render(42)
        second = definition.render(42)
        self.assertEqual(first.variant_id, second.variant_id)
        self.assertEqual(first.prompt_sha256, second.prompt_sha256)
        self.assertTrue(all(message["role"] != "system" for message in first.messages))
        self.assertTrue(all("Confidence:" in prompt for prompt in first.prompts))


if __name__ == "__main__":
    unittest.main()
