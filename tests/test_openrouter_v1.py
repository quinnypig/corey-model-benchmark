import io
import json
import unittest
import urllib.error
from unittest.mock import patch

from corey_bench.openrouter import OpenRouterClient


class FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps(
            {
                "id": "r1", "model": "example/model-v1", "provider": "provider",
                "choices": [{"message": {"content": "answer"}, "finish_reason": "stop"}],
                "usage": {"total_tokens": 10, "cost": 0.01},
            }
        ).encode()


class OpenRouterV1Tests(unittest.TestCase):
    @patch("urllib.request.urlopen", return_value=FakeResponse())
    def test_weights_only_omits_system_and_tools(self, mocked_open):
        OpenRouterClient("key").complete_messages(
            model="example/model", messages=[{"role": "user", "content": "hi"}],
            max_tokens=20, temperature=None, seed=None, reasoning="provider-default",
        )
        payload = json.loads(mocked_open.call_args.args[0].data)
        self.assertEqual(payload["messages"], [{"role": "user", "content": "hi"}])
        self.assertNotIn("tools", payload)
        self.assertNotIn("temperature", payload)
        self.assertNotIn("seed", payload)
        self.assertNotIn("reasoning", payload)

    @patch("urllib.request.urlopen", return_value=FakeResponse())
    def test_search_condition_uses_openrouter_server_tool(self, mocked_open):
        OpenRouterClient("key").complete_messages(
            model="example/model", messages=[{"role": "user", "content": "hi"}],
            max_tokens=20, temperature=1, seed=2, reasoning="off", condition="search-enabled",
        )
        payload = json.loads(mocked_open.call_args.args[0].data)
        self.assertEqual(payload["tools"][0]["type"], "openrouter:web_search")

    @patch("time.sleep")
    @patch("urllib.request.urlopen")
    def test_429_honors_retry_after(self, mocked_open, mocked_sleep):
        error = urllib.error.HTTPError(
            "https://openrouter.ai", 429, "rate limited", {"Retry-After": "3"}, io.BytesIO(b"busy")
        )
        mocked_open.side_effect = [error, FakeResponse()]
        completion = OpenRouterClient("key", attempts=2).complete_messages(
            model="example/model", messages=[{"role": "user", "content": "hi"}],
            max_tokens=20, temperature=1, seed=None, reasoning="off",
        )
        self.assertEqual(completion.request_attempts, 2)
        self.assertGreaterEqual(mocked_sleep.call_args.args[0], 3)


if __name__ == "__main__":
    unittest.main()
