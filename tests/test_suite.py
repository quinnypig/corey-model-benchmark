import unittest

from corey_bench.suite import load_suite


class SuiteTests(unittest.TestCase):
    def test_default_suite_is_valid_and_unique(self):
        suite = load_suite()
        ids = [case.id for case in suite.cases]
        self.assertEqual(len(ids), 8)
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(case.rubric for case in suite.cases))
        self.assertTrue(all(case.checks for case in suite.cases))


if __name__ == "__main__":
    unittest.main()
