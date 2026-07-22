import io
import json
import unittest
import urllib.error
from unittest.mock import patch

from corey_bench.openrouter import PRIVACY_SETTINGS_URL, OpenRouterClient, OpenRouterError


class FakeResponse:
    def __init__(self, body=None):
        self.body = body if body is not None else {
            "id": "generation-1",
            "model": "poolside/laguna-s-2.1-20260720",
            "provider": "Example Provider",
            "choices": [
                {
                    "finish_reason": "stop",
                    "native_finish_reason": "stop",
                    "message": {"content": "The answer", "reasoning": "Checked it."},
                }
            ],
            "usage": {"total_tokens": 42, "cost": 0.00001},
        }

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps(self.body).encode()


class OpenRouterTests(unittest.TestCase):
    @patch(
        "urllib.request.urlopen",
        return_value=FakeResponse(
            {
                "data": [
                    {"id": "poolside/laguna-s-2.1"},
                    {"id": "poolside/laguna-s-2.1:free"},
                ]
            }
        ),
    )
    def test_model_preflight_uses_authenticated_user_catalog(self, mocked_open):
        client = OpenRouterClient("test-key")
        client.require_models_available(
            ["poolside/laguna-s-2.1:free", "poolside/laguna-s-2.1"]
        )

        request = mocked_open.call_args.args[0]
        self.assertEqual(request.full_url, "https://openrouter.ai/api/v1/models/user")
        self.assertEqual(request.get_header("Authorization"), "Bearer test-key")

    @patch(
        "urllib.request.urlopen",
        return_value=FakeResponse({"data": [{"id": "poolside/laguna-s-2.1"}]}),
    )
    def test_model_preflight_rejects_unavailable_exact_variant(self, _mocked_open):
        with self.assertRaisesRegex(OpenRouterError, "laguna-s-2.1:free") as raised:
            OpenRouterClient("test-key").require_models_available(
                ["poolside/laguna-s-2.1:free"]
            )
        self.assertIn(PRIVACY_SETTINGS_URL, str(raised.exception))
        self.assertIn("not interchangeable", str(raised.exception))

    @patch("urllib.request.urlopen", return_value=FakeResponse({"unexpected": []}))
    def test_model_preflight_rejects_malformed_response(self, _mocked_open):
        with self.assertRaisesRegex(OpenRouterError, "expected a data array"):
            OpenRouterClient("test-key").require_models_available(["example/model"])

    @patch("urllib.request.urlopen")
    def test_model_preflight_reports_http_error(self, mocked_open):
        mocked_open.side_effect = urllib.error.HTTPError(
            "https://openrouter.ai/api/v1/models/user",
            403,
            "Forbidden",
            {},
            io.BytesIO(b'{"error":"denied"}'),
        )
        with self.assertRaisesRegex(OpenRouterError, "HTTP 403"):
            OpenRouterClient("test-key").require_models_available(["example/model"])

    @patch("urllib.request.urlopen", return_value=FakeResponse())
    def test_request_settings_and_response_metadata(self, mocked_open):
        completion = OpenRouterClient("test-key").complete(
            model="poolside/laguna-s-2.1:free",
            system="Be useful.",
            prompt="Hello",
            max_tokens=100,
            temperature=0.1,
            seed=7,
            reasoning="on",
        )
        request = mocked_open.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(payload["reasoning"], {"enabled": True})
        self.assertEqual(payload["usage"], {"include": True})
        self.assertEqual(completion.text, "The answer")
        self.assertEqual(completion.raw_model, "poolside/laguna-s-2.1-20260720")
        self.assertEqual(completion.finish_reason, "stop")
        self.assertEqual(completion.usage["total_tokens"], 42)


if __name__ == "__main__":
    unittest.main()
