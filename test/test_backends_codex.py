"""Tests for the Codex CLI backend."""
# pylint: disable=missing-function-docstring

import unittest
from unittest import mock

from ai_labelling.backends.codex import CodexBackend
from ai_labelling.models import ModelSpec


class CodexBackendTests(unittest.TestCase):
    """Verify Codex prompt execution and schema wiring."""

    def test_codex_backend_logs_sanitised_prompt_in_debug_level_two(self):
        """``DEBUG=2`` should log one sanitised prompt and no response dump."""

        backend = CodexBackend(name="Codex", command="codex")
        completed = mock.Mock(stdout="", stderr="")
        fake_json = {
            "add_labels": ["bug"],
            "remove_labels": [],
            "reason": "match",
        }

        with mock.patch.dict("os.environ", {"DEBUG": "2"}, clear=True):
            with mock.patch(
                "ai_labelling.backends.codex.run",
                return_value=completed,
            ) as run_mock:
                with mock.patch(
                    "ai_labelling.backends.codex.json.load",
                    return_value=fake_json,
                ):
                    with mock.patch(
                        "ai_labelling.backends.codex.debug_log"
                    ) as debug_log_mock:
                        result = backend.run_prompt(
                            (
                                "Issue title:\nReal title\n\n"
                                "Main body text:\nReal body\n"
                            ),
                            ModelSpec("codex", None, "low"),
                            allow_label_removals=True,
                        )

        self.assertEqual(result, fake_json)
        run_mock.assert_called_once()
        debug_log_mock.assert_called_once_with(
            "Issue title:\n<ISSUE TITLE OMITTED>\n"
            "Main body text:\n<ISSUE BODY OMITTED>"
        )

    def test_codex_backend_schema_omits_remove_labels_when_disabled(self):
        """Schema should not request removal labels when the flag is off."""

        backend = CodexBackend(name="Codex", command="codex")
        completed = mock.Mock(stdout="", stderr="")
        captured = {}

        def capture_dump(payload, _file_obj):
            captured["schema"] = payload

        with mock.patch(
            "ai_labelling.backends.codex.run",
            return_value=completed,
        ):
            with mock.patch(
                "ai_labelling.backends.codex.json.dump",
                side_effect=capture_dump,
            ):
                with mock.patch(
                    "ai_labelling.backends.codex.json.load",
                    return_value={"add_labels": ["bug"], "reason": "match"},
                ):
                    result = backend.run_prompt(
                        "Prompt",
                        ModelSpec("codex", None, "low"),
                        allow_label_removals=False,
                    )

        self.assertEqual(result["add_labels"], ["bug"])
        self.assertNotIn("remove_labels", captured["schema"]["properties"])
        self.assertEqual(
            captured["schema"]["required"],
            ["add_labels", "reason"],
        )

    def test_codex_backend_omits_model_flag_for_provider_default_wildcard(
        self,
    ):
        """Wildcard models should rely on provider defaults."""

        backend = CodexBackend(name="Codex", command="codex")
        completed = mock.Mock(stdout="", stderr="")

        with mock.patch(
            "ai_labelling.backends.codex.run",
            return_value=completed,
        ) as run_mock:
            with mock.patch(
                "ai_labelling.backends.codex.json.load",
                return_value={"add_labels": [], "reason": ""},
            ):
                backend.run_prompt(
                    "Prompt",
                    ModelSpec("codex", None, "low"),
                    allow_label_removals=False,
                )

        argv = run_mock.call_args.args[0]
        self.assertNotIn("--model", argv)
