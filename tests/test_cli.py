import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from corey_bench.cli import _load_dotenv, build_parser, cmd_run
from corey_bench.openrouter import OpenRouterError


class CliTests(unittest.TestCase):
    def test_dotenv_loads_key_without_overwriting_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            dotenv = Path(directory) / ".env"
            dotenv.write_text("OPENROUTER_API_KEY=from-file\nEXISTING=file-value\n")
            with patch.dict(os.environ, {"EXISTING": "environment-value"}, clear=True):
                _load_dotenv(dotenv)
                self.assertEqual(os.environ["OPENROUTER_API_KEY"], "from-file")
                self.assertEqual(os.environ["EXISTING"], "environment-value")

    def test_live_run_requires_explicit_model_before_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "runs"
            args = build_parser().parse_args(
                ["run", "--case", "billing_premise_trap", "--output", str(output)]
            )
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(ValueError, "at least one --model"):
                    cmd_run(args)
            self.assertFalse(output.exists())

    @patch("corey_bench.cli.OpenRouterClient")
    def test_live_run_preflights_before_creating_output_or_completing(self, client_class):
        client_class.return_value.require_models_available.side_effect = OpenRouterError(
            "model unavailable"
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "runs"
            args = build_parser().parse_args(
                [
                    "run",
                    "--model",
                    "example/unavailable",
                    "--case",
                    "billing_premise_trap",
                    "--output",
                    str(output),
                ]
            )
            with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=True):
                with self.assertRaisesRegex(OpenRouterError, "model unavailable"):
                    cmd_run(args)
            client_class.return_value.require_models_available.assert_called_once_with(
                ["example/unavailable"]
            )
            client_class.return_value.complete.assert_not_called()
            self.assertFalse(output.exists())

    @patch("corey_bench.cli.OpenRouterClient")
    def test_dry_run_skips_credentials_and_model_preflight(self, client_class):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "runs"
            args = build_parser().parse_args(
                [
                    "run",
                    "--dry-run",
                    "--model",
                    "example/model",
                    "--case",
                    "billing_premise_trap",
                    "--output",
                    str(output),
                    "--run-id",
                    "dry-run-test",
                ]
            )
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(cmd_run(args), 0)
            client_class.assert_not_called()
            self.assertTrue((output / "dry-run-test" / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
