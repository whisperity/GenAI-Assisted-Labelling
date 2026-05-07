"""Tests for subprocess and version helpers."""
# pylint: disable=missing-function-docstring

import unittest
from unittest import mock

from ai_labelling import shell


class RunTests(unittest.TestCase):
    """Cover subprocess invocation and debug-trace behaviour."""

    def test_run_returns_subprocess_result(self):
        completed = mock.Mock(stdout="ok", stderr="")
        with mock.patch.dict("os.environ", {}, clear=True), \
                mock.patch.object(
                    shell.subprocess, "run", return_value=completed
                ) as run_mock:
            result = shell.run(("echo", "hi"))
        self.assertIs(result, completed)
        run_mock.assert_called_once()

    def test_run_passes_input_text(self):
        completed = mock.Mock(stdout="", stderr="")
        with mock.patch.dict("os.environ", {}, clear=True), \
                mock.patch.object(
                    shell.subprocess, "run", return_value=completed
                ) as run_mock:
            shell.run(("cat",), input_text="payload")
        kwargs = run_mock.call_args.kwargs
        self.assertEqual(kwargs.get("input"), "payload")
        self.assertTrue(kwargs.get("text"))

    def test_run_check_propagates(self):
        completed = mock.Mock(stdout="", stderr="")
        with mock.patch.dict("os.environ", {}, clear=True), \
                mock.patch.object(
                    shell.subprocess, "run", return_value=completed
                ) as run_mock:
            shell.run(("echo",), check=False)
        self.assertFalse(run_mock.call_args.kwargs.get("check"))


class GetScriptVersionTests(unittest.TestCase):
    """Cover git-based script version detection."""

    def test_returns_short_sha_when_git_succeeds(self):
        completed = mock.Mock(returncode=0, stdout="abc1234\n", stderr="")
        with mock.patch("ai_labelling.shell.run", return_value=completed):
            self.assertEqual(shell.get_script_version(), "abc1234")

    def test_returns_unknown_on_non_zero_exit(self):
        completed = mock.Mock(returncode=128, stdout="", stderr="not a repo")
        with mock.patch("ai_labelling.shell.run", return_value=completed):
            self.assertEqual(shell.get_script_version(), "unknown")

    def test_returns_unknown_when_run_raises_oserror(self):
        with mock.patch(
            "ai_labelling.shell.run", side_effect=FileNotFoundError("git")
        ):
            self.assertEqual(shell.get_script_version(), "unknown")
