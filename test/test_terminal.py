"""Tests for terminal formatting and subprocess helpers."""
# pylint: disable=missing-function-docstring

import sys
import unittest
from unittest import mock

from ai_labelling import shell
from ai_labelling.terminal import (
    colourise,
    debug_log,
    format_prompt_for_debug,
    get_debug_level,
    sanitise_prompt_for_debug,
    supports_colour,
)


class TerminalTests(unittest.TestCase):
    """Verify debug, colour, and subprocess shell helpers."""

    def test_get_debug_level_follows_debug_environment_variable(self):
        """``DEBUG`` should map to numeric logging levels."""

        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertEqual(get_debug_level(), 0)
        with mock.patch.dict("os.environ", {"DEBUG": ""}, clear=True):
            self.assertEqual(get_debug_level(), 0)
        with mock.patch.dict("os.environ", {"DEBUG": "0"}, clear=True):
            self.assertEqual(get_debug_level(), 0)
        with mock.patch.dict("os.environ", {"DEBUG": "1"}, clear=True):
            self.assertEqual(get_debug_level(), 1)
        with mock.patch.dict("os.environ", {"DEBUG": "3"}, clear=True):
            self.assertEqual(get_debug_level(), 3)
        with mock.patch.dict("os.environ", {"DEBUG": "verbose"}, clear=True):
            self.assertEqual(get_debug_level(), 1)

    def test_supports_colour_honours_no_color_and_term_dumb(self):
        """Environment should disable ANSI colour when requested."""

        tty_stream = mock.Mock()
        tty_stream.isatty.return_value = True

        with mock.patch.dict("os.environ", {"NO_COLOR": "1"}, clear=True):
            self.assertFalse(supports_colour(tty_stream))
        with mock.patch.dict("os.environ", {"TERM": "dumb"}, clear=True):
            self.assertFalse(supports_colour(tty_stream))

    def test_colourise_returns_plain_text_without_tty_support(self):
        """Non-TTY streams should not receive ANSI escapes."""

        stream = mock.Mock()
        stream.isatty.return_value = False
        self.assertEqual(colourise("hello", "blue", stream=stream), "hello")

    def test_sanitise_prompt_for_debug_omits_runtime_data(self):
        """``DEBUG=2`` prompt views should hide issue-specific data blocks."""

        prompt = """You are labeling a GitHub issue.

Issue title:
Crash in foo

Existing labels:
[
  "bug"
]

Valid labels:
[
  {"name": "bug"}
]

Main body text:
Long body text.
"""

        result = sanitise_prompt_for_debug(prompt)
        self.assertIn("<ISSUE TITLE OMITTED>", result)
        self.assertIn("<LABELS OMITTED>", result)
        self.assertIn("<LABEL DEFINITIONS OMITTED>", result)
        self.assertIn("<ISSUE BODY OMITTED>", result)
        self.assertNotIn("Crash in foo", result)
        self.assertNotIn("Long body text.", result)

    def test_format_prompt_for_debug_returns_full_prompt_at_level_three(self):
        """``DEBUG>=3`` should expose the full prompt text once."""

        with mock.patch.dict("os.environ", {"DEBUG": "3"}, clear=True):
            self.assertEqual(
                format_prompt_for_debug("Full prompt"),
                "Full prompt",
            )

    def test_run_hides_subcommand_trace_without_debug(self):
        """Subprocess execution should stay quiet when ``DEBUG`` is off."""

        completed = mock.Mock(stdout="", stderr="")
        with mock.patch.dict("os.environ", {}, clear=True):
            with mock.patch.object(
                shell.subprocess,
                "run",
                return_value=completed,
            ) as run_mock:
                with mock.patch("builtins.print") as print_mock:
                    result = shell.run(("echo", "hello"))

        self.assertIs(result, completed)
        run_mock.assert_called_once()
        print_mock.assert_not_called()

    def test_run_logs_subcommand_trace_in_debug_mode(self):
        """Subprocess execution should emit trace output when debug is on."""

        completed = mock.Mock(stdout="", stderr="")
        with mock.patch.dict("os.environ", {"DEBUG": "1"}, clear=True):
            with mock.patch.object(
                shell.subprocess,
                "run",
                return_value=completed,
            ):
                with mock.patch("builtins.print") as print_mock:
                    shell.run(("echo", "hello world"))

        printed = [
            call.args[0]
            for call in print_mock.call_args_list
            if call.args
        ]
        self.assertEqual(printed, ["+ echo 'hello world'"])

    def test_debug_log_is_silent_when_debug_off(self):
        """debug_log should do nothing when DEBUG is unset."""

        with mock.patch.dict("os.environ", {}, clear=True):
            with mock.patch("builtins.print") as print_mock:
                debug_log("should not appear")
        print_mock.assert_not_called()

    def test_debug_log_writes_to_stderr_when_active(self):
        """debug_log should print to stderr when DEBUG>=1."""

        with mock.patch.dict("os.environ", {"DEBUG": "1"}, clear=True):
            with mock.patch("builtins.print") as print_mock:
                debug_log("trace line")
        print_mock.assert_called_once()
        self.assertIs(
            print_mock.call_args.kwargs.get("file"), sys.stderr
        )

    def test_colourise_applies_bold_on_tty(self):
        """Bold flag should prepend the bold escape on a TTY stream."""

        tty = mock.Mock()
        tty.isatty.return_value = True
        with mock.patch.dict(
            "os.environ", {"TERM": "xterm-256color"}, clear=True
        ):
            result = colourise("hi", "red", stream=tty, bold=True)
        self.assertIn("\033[1m", result)
        self.assertIn("\033[31m", result)
        self.assertIn("hi", result)

    def test_format_prompt_for_debug_returns_none_below_level_two(self):
        """Levels 0 and 1 should return None (no prompt logged)."""

        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertIsNone(format_prompt_for_debug("p"))
        with mock.patch.dict("os.environ", {"DEBUG": "1"}, clear=True):
            self.assertIsNone(format_prompt_for_debug("p"))

    def test_format_prompt_for_debug_returns_sanitised_at_level_two(self):
        """Level 2 should return the sanitised (redacted) prompt."""

        prompt = "Issue title:\nReal title\n\nMain body text:\nReal body\n"
        with mock.patch.dict("os.environ", {"DEBUG": "2"}, clear=True):
            result = format_prompt_for_debug(prompt)
        self.assertIsNotNone(result)
        self.assertIn("<ISSUE TITLE OMITTED>", result)
        self.assertNotIn("Real title", result)
