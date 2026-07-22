import tempfile
import unittest
from pathlib import Path

from corey_bench.protocol import load_protocol
from corey_bench.runner import RunStore
from corey_bench.webapp import create_app


class FakeClient:
    def list_models(self):
        return [{"id": "example/model", "name": "Example Model", "context_length": 1000, "pricing": {}}]


class FakeQueue:
    def __init__(self, root):
        self.store = RunStore(root)
        self.suite = load_protocol()
        self.client = FakeClient()
        self._queue = type("Q", (), {"qsize": lambda self: 0})()

    def start(self):
        pass


class WebAppTests(unittest.TestCase):
    def test_dashboard_health_and_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            app = create_app(run_queue=FakeQueue(Path(directory)))
            app.config["TESTING"] = True
            client = app.test_client()
            page = client.get("/")
            self.assertEqual(page.status_code, 200)
            self.assertIn(b"Quinnferno", page.data)
            self.assertIn(b"example/model", page.data)
            self.assertEqual(client.get("/healthz").status_code, 200)
            self.assertEqual(client.get("/runs/%2e%2e%2fetc").status_code, 404)


if __name__ == "__main__":
    unittest.main()
