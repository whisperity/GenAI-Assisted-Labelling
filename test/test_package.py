"""Tests for package-level facade helpers and top-level orchestration."""
# pylint: disable=duplicate-code,missing-function-docstring

import argparse
import runpy
import subprocess
import unittest
from unittest import mock

from test.helpers import make_item

import ai_labelling
from ai_labelling.config import DEFAULT_DATE_CUTOFF
from ai_labelling.models import (
    LabelDefinition,
    LabelSuggestion,
    SuggestionResult,
    UserQuit,
)


class PackageFacadeTests(unittest.TestCase):
    """Verify facade wrappers and the end-to-end ``main`` flow."""

    def test_main_returns_zero_when_no_items_match(self):
        args = argparse.Namespace(
            repo="llvm/llvm-project",
            created=False,
            date=DEFAULT_DATE_CUTOFF,
            limit=None,
            allow_label_removals=False,
            include_closed=False,
            include_issues=True,
            include_open=True,
            include_prs=False,
            dry_run=False,
            force=False,
            model="codex:*:low",
        )

        with mock.patch.object(ai_labelling, "parse_args", return_value=args):
            with mock.patch.object(
                ai_labelling,
                "list_repo_labels",
                return_value=[LabelDefinition("bug", "")],
            ):
                with mock.patch.object(
                    ai_labelling,
                    "collect_items",
                    return_value=[],
                ):
                    with mock.patch("builtins.print"):
                        result = ai_labelling.main()

        self.assertEqual(result, 0)

    def test_main_runs_full_review_flow_when_items_exist(self):
        item = make_item(1, "One")
        args = argparse.Namespace(
            repo="llvm/llvm-project",
            created=False,
            date=DEFAULT_DATE_CUTOFF,
            limit=None,
            allow_label_removals=False,
            include_closed=False,
            include_issues=True,
            include_open=True,
            include_prs=False,
            dry_run=False,
            force=False,
            model="codex:*:low",
        )
        labels = [LabelDefinition("bug", "Bug report")]
        suggestion_results = [
            SuggestionResult(
                item=item,
                label_suggestion=LabelSuggestion(
                    add_labels=["bug"],
                    remove_labels=[],
                    reason="match",
                ),
            )
        ]

        with mock.patch.object(ai_labelling, "parse_args", return_value=args):
            with mock.patch.object(
                ai_labelling,
                "list_repo_labels",
                return_value=labels,
            ):
                with mock.patch.object(
                    ai_labelling,
                    "collect_items",
                    return_value=[item],
                ):
                    with mock.patch.object(
                        ai_labelling,
                        "print_match_summary",
                    ) as summary_mock:
                        with mock.patch.object(
                            ai_labelling,
                            "print_matching_items",
                        ) as matching_mock:
                            with mock.patch.object(
                                ai_labelling,
                                "select_items_to_handle",
                                return_value=[item],
                            ) as select_mock:
                                with mock.patch.object(
                                    ai_labelling,
                                    "run_ai_batch",
                                    return_value=suggestion_results,
                                ) as batch_mock:
                                    with mock.patch.object(
                                        ai_labelling,
                                        "review_and_apply_suggestions",
                                    ) as review_mock:
                                        with mock.patch("builtins.print"):
                                            result = ai_labelling.main()

        self.assertEqual(result, 0)
        summary_mock.assert_called_once_with([item])
        matching_mock.assert_called_once_with([item], "Matching items")
        select_mock.assert_called_once_with([item], False)
        batch_mock.assert_called_once_with(
            [item],
            labels,
            "codex:*:low",
            False,
            input_fn=input,
        )
        review_mock.assert_called_once_with(
            "llvm/llvm-project",
            suggestion_results,
            False,
            False,
            dry_run=False,
        )

    def test_main_returns_zero_when_no_items_are_selected(self):
        item = make_item(1, "One")
        args = argparse.Namespace(
            repo="llvm/llvm-project",
            created=False,
            date=DEFAULT_DATE_CUTOFF,
            limit=None,
            allow_label_removals=False,
            include_closed=False,
            include_issues=True,
            include_open=True,
            include_prs=False,
            dry_run=False,
            force=False,
            model="codex:*:low",
        )

        with mock.patch.object(ai_labelling, "parse_args", return_value=args):
            with mock.patch.object(
                ai_labelling,
                "list_repo_labels",
                return_value=[LabelDefinition("bug", "")],
            ):
                with mock.patch.object(
                    ai_labelling,
                    "collect_items",
                    return_value=[item],
                ):
                    with mock.patch.object(
                        ai_labelling,
                        "print_match_summary",
                    ):
                        with mock.patch.object(
                            ai_labelling,
                            "print_matching_items",
                        ):
                            with mock.patch.object(
                                ai_labelling,
                                "select_items_to_handle",
                                return_value=[],
                            ):
                                with mock.patch("builtins.print"):
                                    result = ai_labelling.main()

        self.assertEqual(result, 0)

    def test_entrypoint_turns_user_quit_into_exit_zero(self):
        with mock.patch.object(ai_labelling, "main", side_effect=UserQuit):
            with mock.patch("builtins.print"):
                with self.assertRaises(SystemExit) as exit_context:
                    runpy.run_path("ai-labelling", run_name="__main__")
        self.assertEqual(exit_context.exception.code, 0)

    def test_entrypoint_turns_runtime_error_into_exit_one(self):
        with mock.patch.object(
            ai_labelling,
            "main",
            side_effect=RuntimeError("boom"),
        ):
            with mock.patch("builtins.print"):
                with self.assertRaises(SystemExit) as exit_context:
                    runpy.run_path("ai-labelling", run_name="__main__")
        self.assertEqual(exit_context.exception.code, 1)

    def test_entrypoint_propagates_called_process_error_code(self):
        error = subprocess.CalledProcessError(
            7,
            ("gh", "api"),
            output="stdout text\n",
            stderr="stderr text\n",
        )
        with mock.patch.object(ai_labelling, "main", side_effect=error):
            with mock.patch("builtins.print"):
                with self.assertRaises(SystemExit) as exit_context:
                    runpy.run_path("ai-labelling", run_name="__main__")
        self.assertEqual(exit_context.exception.code, 7)
