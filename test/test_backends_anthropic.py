"""Tests for the Anthropic API backend."""
# pylint: disable=missing-function-docstring

import urllib.error
import unittest
import warnings
from unittest import mock

from ai_labelling.backends.anthropic import (
    AnthropicBackend,
    AnthropicHTTPError,
    sanitise_headers_for_debug,
)
from ai_labelling.models import ModelSpec


class AnthropicBackendTests(unittest.TestCase):
    """Verify direct Anthropic API behaviour through mocked calls."""

    def test_headers_require_api_key(self):
        """Anthropic requests should fail fast without an API key."""

        backend = AnthropicBackend(name="Anthropic")
        with mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(
                RuntimeError,
                "ANTHROPIC_API_KEY is not set",
            ):
                backend.headers()

    def test_get_default_model_uses_first_model_entry(self):
        """Anthropic ``*`` should resolve to the first listed model."""

        backend = AnthropicBackend(name="Anthropic")
        with mock.patch.object(
            backend,
            "json_request",
            return_value={
                "data": [
                    {"id": "claude-sonnet-4-20250514"},
                    {"id": "claude-3-7-sonnet-20250219"},
                ]
            },
        ):
            result = backend.get_default_model()

        self.assertEqual(result, "claude-sonnet-4-20250514")

    def test_extract_json_accepts_fenced_payload(self):
        """Anthropic text responses may wrap JSON in markdown fences."""

        backend = AnthropicBackend(name="Anthropic")
        result = backend.extract_json(
            "```json\n{\"add_labels\": [\"bug\"], \"reason\": \"x\"}\n```"
        )
        self.assertEqual(result, {"add_labels": ["bug"], "reason": "x"})

    def test_json_request_wraps_http_error_text(self):
        """HTTP failures should be converted into readable runtime errors."""

        backend = AnthropicBackend(name="Anthropic")
        error = urllib.error.HTTPError(
            backend.api_url + "/v1/messages",
            401,
            "Unauthorized",
            hdrs=None,
            fp=None,
        )
        error.read = lambda: b"bad key"

        with mock.patch.dict(
            "os.environ",
            {"ANTHROPIC_API_KEY": "secret"},
            clear=True,
        ):
            with mock.patch("urllib.request.urlopen", side_effect=error):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ResourceWarning)
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "HTTP 401: bad key",
                    ):
                        backend.json_request(
                            "/v1/messages",
                            method="POST",
                            body={},
                        )
        error.close()

    def test_json_request_wraps_url_error(self):
        """Network failures (URLError) should surface as RuntimeErrors."""

        backend = AnthropicBackend(name="Anthropic")
        url_error = urllib.error.URLError("connection refused")

        with mock.patch.dict(
            "os.environ",
            {"ANTHROPIC_API_KEY": "secret"},
            clear=True,
        ):
            with mock.patch(
                "urllib.request.urlopen", side_effect=url_error
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "connection refused"
                ):
                    backend.json_request("/v1/models", method="GET")

    def test_get_default_model_rejects_non_dict_payload(self):
        """A non-object response from the models endpoint is an error."""

        backend = AnthropicBackend(name="Anthropic")
        with mock.patch.object(
            backend, "json_request", return_value=["bad"]
        ):
            with self.assertRaisesRegex(
                RuntimeError, "unexpected Anthropic models payload"
            ):
                backend.get_default_model()

    def test_get_default_model_rejects_empty_data_list(self):
        """An empty models list should raise rather than return nothing."""

        backend = AnthropicBackend(name="Anthropic")
        with mock.patch.object(
            backend, "json_request", return_value={"data": []}
        ):
            with self.assertRaisesRegex(
                RuntimeError, "models list was empty"
            ):
                backend.get_default_model()

    def test_get_default_model_rejects_invalid_model_entry(self):
        """Model entries without a string ``id`` field are rejected."""

        backend = AnthropicBackend(name="Anthropic")
        with mock.patch.object(
            backend,
            "json_request",
            return_value={"data": [{"no_id": True}]},
        ):
            with self.assertRaisesRegex(
                RuntimeError, "unexpected Anthropic model entry"
            ):
                backend.get_default_model()

    def test_extract_json_rejects_invalid_json(self):
        """Malformed JSON content should raise a RuntimeError."""

        backend = AnthropicBackend(name="Anthropic")
        with self.assertRaisesRegex(RuntimeError, "invalid JSON"):
            backend.extract_json("not json at all")

    def test_extract_json_rejects_non_dict_response(self):
        """A JSON array at the top level should raise a RuntimeError."""

        backend = AnthropicBackend(name="Anthropic")
        with self.assertRaisesRegex(RuntimeError, "non-object"):
            backend.extract_json("[1, 2, 3]")

    def test_sanitise_headers_redacts_api_key(self):
        """The x-api-key header value should be replaced with a sentinel."""

        result = sanitise_headers_for_debug(
            {"x-api-key": "secret", "anthropic-version": "2023-06-01"}
        )
        self.assertEqual(result["x-api-key"], "***REDACTED***")
        self.assertEqual(result["anthropic-version"], "2023-06-01")

    def test_sanitise_headers_passes_through_headers_without_key(self):
        """Headers that contain no API key should be returned unchanged."""

        headers = {"content-type": "application/json"}
        result = sanitise_headers_for_debug(headers)
        self.assertEqual(result, headers)
        self.assertIsNot(result, headers)

    def test_run_prompt_omits_output_config_when_effort_is_none(self):
        """No output_config should be sent when reasoning_effort is None."""

        backend = AnthropicBackend(name="Anthropic")
        captured_body = {}

        def capture(  # pylint: disable=unused-argument
            _path, *, method=None, body=None
        ):
            if body:
                captured_body.update(body)
            return {
                "content": [
                    {"type": "text", "text": '{"add_labels":[],"reason":""}'}
                ]
            }

        with mock.patch.object(backend, "json_request", side_effect=capture):
            backend.run_prompt(
                "Prompt",
                ModelSpec("anthropic", "claude-haiku-4-5-20251001", None),
                allow_label_removals=False,
            )

        self.assertNotIn("output_config", captured_body)

    def test_run_prompt_retries_without_effort_on_400(self):
        """HTTP 400 'effort not supported' should trigger a no-effort retry."""

        backend = AnthropicBackend(name="Anthropic")
        success = {
            "content": [
                {"type": "text", "text": '{"add_labels":[],"reason":"ok"}'}
            ]
        }
        call_bodies = []

        def side_effect(  # pylint: disable=unused-argument
            _path, *, method=None, body=None
        ):
            call_bodies.append(dict(body) if body else {})
            if len(call_bodies) == 1:
                raise AnthropicHTTPError(
                    400,
                    '{"error": {"message": '
                    '"This model does not support the effort parameter."}}',
                )
            return success

        with mock.patch.object(
            backend, "json_request", side_effect=side_effect
        ):
            backend.run_prompt(
                "Prompt",
                ModelSpec("anthropic", "claude-haiku-4-5-20251001", "low"),
                allow_label_removals=False,
            )

        self.assertEqual(len(call_bodies), 2)
        self.assertIn("output_config", call_bodies[0])
        self.assertNotIn("output_config", call_bodies[1])

    def test_provider_metadata_accessors(self):
        """Static metadata should match the module-level constants."""

        backend = AnthropicBackend(name="Anthropic")
        self.assertEqual(backend.provider_name(), "anthropic")
        self.assertEqual(
            backend.default_model_name(), "claude-haiku-4-5-20251001"
        )
        self.assertIn("low", backend.reasoning_levels())
        self.assertIn("max", backend.reasoning_levels())
        self.assertIn("Claude Platform API", backend.help_description())

    def test_anthropic_backend_uses_messages_api_and_default_model(self):
        """Anthropic backend should resolve ``*`` and parse text content."""

        backend = AnthropicBackend(name="Anthropic")

        with mock.patch.object(
            backend,
            "get_default_model",
            return_value="claude-sonnet-4-20250514",
        ) as default_mock:
            with mock.patch.object(
                backend,
                "json_request",
                return_value={
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "{\"add_labels\": [\"bug\"], "
                                "\"remove_labels\": [], "
                                "\"reason\": \"match\"}"
                            ),
                        }
                    ]
                },
            ) as request_mock:
                result = backend.run_prompt(
                    "Prompt body",
                    ModelSpec("anthropic", None, "low"),
                    allow_label_removals=True,
                )

        self.assertEqual(result["add_labels"], ["bug"])
        default_mock.assert_called_once_with()
        request_mock.assert_called_once_with(
            "/v1/messages",
            method="POST",
            body={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Prompt body"}],
                "output_config": {"effort": "low"},
            },
        )
