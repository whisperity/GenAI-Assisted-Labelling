"""Tests for terminal/colour/debug helpers."""
# pylint: disable=missing-function-docstring

import sys
import unittest
from unittest import mock

from ai_labelling.terminal import (
    colourise,
    debug_log,
    format_prompt_for_debug,
    get_debug_level,
    sanitise_prompt_for_debug,
    supports_colour,
)


class DebugLevelTests(unittest.TestCase):
    """Verify ``DEBUG`` env var translation."""

    def test_get_debug_level_unset_is_zero(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertEqual(get_debug_level(), 0)

    def test_get_debug_level_empty_is_zero(self):
        with mock.patch.dict("os.environ", {"DEBUG": ""}, clear=True):
            self.assertEqual(get_debug_level(), 0)

    def test_get_debug_level_zero_string_is_zero(self):
        with mock.patch.dict("os.environ", {"DEBUG": "0"}, clear=True):
            self.assertEqual(get_debug_level(), 0)

    def test_get_debug_level_one_is_one(self):
        with mock.patch.dict("os.environ", {"DEBUG": "1"}, clear=True):
            self.assertEqual(get_debug_level(), 1)

    def test_get_debug_level_three_is_three(self):
        with mock.patch.dict("os.environ", {"DEBUG": "3"}, clear=True):
            self.assertEqual(get_debug_level(), 3)

    def test_get_debug_level_non_numeric_is_one(self):
        with mock.patch.dict("os.environ", {"DEBUG": "verbose"}, clear=True):
            self.assertEqual(get_debug_level(), 1)


class ColourSupportTests(unittest.TestCase):
    """Verify colour-support detection across env states."""

    def test_supports_colour_no_color_disables(self):
        tty_stream = mock.Mock()
        tty_stream.isatty.return_value = True
        with mock.patch.dict("os.environ", {"NO_COLOR": "1"}, clear=True):
            self.assertFalse(supports_colour(tty_stream))

    def test_supports_colour_dumb_term_disables(self):
        tty_stream = mock.Mock()
        tty_stream.isatty.return_value = True
        with mock.patch.dict("os.environ", {"TERM": "dumb"}, clear=True):
            self.assertFalse(supports_colour(tty_stream))

    def test_supports_colour_non_tty_disables(self):
        stream = mock.Mock()
        stream.isatty.return_value = False
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertFalse(supports_colour(stream))

    def test_supports_colour_tty_enables(self):
        stream = mock.Mock()
        stream.isatty.return_value = True
        with mock.patch.dict(
            "os.environ", {"TERM": "xterm-256color"}, clear=True,
        ):
            self.assertTrue(supports_colour(stream))

    def test_supports_colour_handles_streams_without_isatty(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertFalse(supports_colour(object()))


class ColouriseTests(unittest.TestCase):
    """Verify text colourisation."""

    def test_returns_plain_text_without_tty(self):
        stream = mock.Mock()
        stream.isatty.return_value = False
        self.assertEqual(colourise("hello", "blue", stream=stream), "hello")

    def test_applies_bold_on_tty(self):
        tty = mock.Mock()
        tty.isatty.return_value = True
        with mock.patch.dict(
            "os.environ", {"TERM": "xterm-256color"}, clear=True,
        ):
            result = colourise("hi", "red", stream=tty, bold=True)
        self.assertIn("\033[1m", result)
        self.assertIn("\033[31m", result)
        self.assertIn("hi", result)


class DebugLogTests(unittest.TestCase):
    """Verify debug-log gating and stream selection."""

    def test_silent_when_debug_off(self):
        with mock.patch.dict("os.environ", {}, clear=True), \
                mock.patch("builtins.print") as print_mock:
            debug_log("should not appear")
        print_mock.assert_not_called()

    def test_writes_to_stderr_when_active(self):
        with mock.patch.dict(
            "os.environ", {"DEBUG": "1"}, clear=True,
        ), mock.patch("builtins.print") as print_mock:
            debug_log("trace line")
        print_mock.assert_called_once()
        self.assertIs(
            print_mock.call_args.kwargs.get("file"), sys.stderr
        )


class SanitisePromptTests(unittest.TestCase):
    """Verify the prompt-redaction logic for ``DEBUG=2`` output."""

    def test_omits_runtime_data(self):
        prompt = (
            "You are labeling a GitHub issue.\n\n"
            "Issue title:\nCrash in foo\n\n"
            "Existing labels:\n[\"bug\"]\n\n"
            "Valid labels:\n[{\"name\": \"bug\"}]\n\n"
            "Main body text:\nLong body text.\n"
        )
        result = sanitise_prompt_for_debug(prompt)
        self.assertIn("<ISSUE TITLE OMITTED>", result)
        self.assertIn("<LABELS OMITTED>", result)
        self.assertIn("<LABEL DEFINITIONS OMITTED>", result)
        self.assertIn("<ISSUE BODY OMITTED>", result)
        self.assertNotIn("Crash in foo", result)
        self.assertNotIn("Long body text.", result)

    def test_keeps_preamble_lines(self):
        prompt = "Preamble line.\n\nIssue title:\nReal title\n"
        result = sanitise_prompt_for_debug(prompt)
        self.assertIn("Preamble line.", result)
        self.assertNotIn("Real title", result)


class FormatPromptForDebugTests(unittest.TestCase):
    """Verify level-aware prompt formatting."""

    def test_returns_full_prompt_at_level_three(self):
        with mock.patch.dict("os.environ", {"DEBUG": "3"}, clear=True):
            self.assertEqual(
                format_prompt_for_debug("Full prompt"),
                "Full prompt",
            )

    def test_returns_none_below_level_two(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertIsNone(format_prompt_for_debug("p"))
        with mock.patch.dict("os.environ", {"DEBUG": "1"}, clear=True):
            self.assertIsNone(format_prompt_for_debug("p"))

    def test_returns_sanitised_at_level_two(self):
        prompt = "Issue title:\nReal title\n\nMain body text:\nReal body\n"
        with mock.patch.dict("os.environ", {"DEBUG": "2"}, clear=True):
            result = format_prompt_for_debug(prompt)
        self.assertIsNotNone(result)
        self.assertIn("<ISSUE TITLE OMITTED>", result)
        self.assertNotIn("Real title", result)
