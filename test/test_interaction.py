"""Tests for interactive prompt helpers."""
# pylint: disable=missing-function-docstring

import unittest
from unittest import mock

from ai_labelling.interaction import (
    print_prompt_help,
    prompt_confirmation,
    prompt_yes_no,
)
from ai_labelling.models import UserQuit


class PromptConfirmationTests(unittest.TestCase):
    """Verify the multi-choice item-level prompt."""

    def test_retries_until_non_empty_valid_answer(self):
        answers = iter(["", "maybe", "a"])
        result = prompt_confirmation(
            "Prompt: ",
            allow_apply_all=False,
            input_fn=lambda _: next(answers),
        )
        self.assertEqual(result, "A")

    def test_accepts_done_alias(self):
        result = prompt_confirmation(
            "Prompt: ",
            allow_apply_all=True,
            input_fn=lambda _: "d",
        )
        self.assertEqual(result, "D")

    def test_accepts_yes_no(self):
        for letter in ("y", "Y", "n", "N"):
            result = prompt_confirmation(
                "Prompt: ",
                allow_apply_all=False,
                input_fn=lambda _prompt, _l=letter: _l,
            )
            self.assertEqual(result, letter.upper())

    def test_raises_on_quit(self):
        with self.assertRaises(UserQuit):
            prompt_confirmation(
                "Prompt: ",
                allow_apply_all=False,
                input_fn=lambda _: "q",
            )

    def test_help_token_prints_help_and_retries(self):
        answers = iter(["?", "y"])
        with mock.patch(
            "ai_labelling.interaction.print_prompt_help"
        ) as help_mock:
            result = prompt_confirmation(
                "Prompt: ",
                allow_apply_all=True,
                input_fn=lambda _: next(answers),
            )
        help_mock.assert_called_once_with(True)
        self.assertEqual(result, "Y")


class PromptYesNoTests(unittest.TestCase):
    """Verify the yes/no retry prompt."""

    def test_uses_default_yes_on_empty_answer(self):
        self.assertTrue(
            prompt_yes_no(
                "Retry? ", default_yes=True, input_fn=lambda _: ""
            )
        )

    def test_uses_default_no_on_empty_answer(self):
        self.assertFalse(
            prompt_yes_no(
                "Retry? ", default_yes=False, input_fn=lambda _: ""
            )
        )

    def test_retries_until_valid_answer(self):
        answers = iter(["maybe", "n"])
        self.assertFalse(
            prompt_yes_no(
                "Retry? ",
                default_yes=True,
                input_fn=lambda _: next(answers),
            )
        )

    def test_accepts_long_yes(self):
        self.assertTrue(
            prompt_yes_no(
                "Retry? ", default_yes=False, input_fn=lambda _: "yes"
            )
        )

    def test_accepts_long_no(self):
        self.assertFalse(
            prompt_yes_no(
                "Retry? ", default_yes=True, input_fn=lambda _: "no"
            )
        )

    def test_renders_default_yes_suffix(self):
        captured = []
        prompt_yes_no(
            "Retry? ",
            default_yes=True,
            input_fn=lambda prompt: captured.append(prompt) or "y",
        )
        self.assertIn("[Y/n]", captured[0])

    def test_renders_default_no_suffix(self):
        captured = []
        prompt_yes_no(
            "Retry? ",
            default_yes=False,
            input_fn=lambda prompt: captured.append(prompt) or "n",
        )
        self.assertIn("[y/N]", captured[0])


class PromptHelpTests(unittest.TestCase):
    """Verify the help-text printer."""

    def test_supports_item_mode(self):
        with mock.patch("builtins.print") as print_mock:
            print_prompt_help(False)
        printed = "\n".join(
            call.args[0] for call in print_mock.call_args_list if call.args
        )
        self.assertIn("handle this item", printed)
        self.assertIn("stop prompting more items", printed)

    def test_supports_apply_mode(self):
        with mock.patch("builtins.print") as print_mock:
            print_prompt_help(True)
        printed = "\n".join(
            call.args[0] for call in print_mock.call_args_list if call.args
        )
        self.assertIn("use this and all remaining", printed)
        self.assertIn("remaining labels in this action", printed)

    def test_always_includes_quit_and_help(self):
        with mock.patch("builtins.print") as print_mock:
            print_prompt_help(False)
        printed = "\n".join(
            c.args[0] for c in print_mock.call_args_list if c.args
        )
        self.assertIn("q - quit", printed)
        self.assertIn("? - help", printed)
