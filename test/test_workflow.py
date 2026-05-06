"""Tests for workflow state and control-flow helpers."""
# pylint: disable=missing-function-docstring

import argparse
import unittest
from datetime import datetime, timezone
from unittest import mock

from test.helpers import make_item
from ai_labelling.config import DEFAULT_DATE_CUTOFF
from ai_labelling.models import (
    LabelDefinition,
    LabelSuggestion,
    SuggestionResult,
    UserQuit,
)
from ai_labelling.workflow import (
    LabellingWorkflow,
    normalise_label_list,
    normalise_label_suggestions,
    prompt_confirmation,
    prompt_yes_no,
)


class WorkflowTests(
    unittest.TestCase,
):  # pylint: disable=too-many-public-methods
    """Verify item-selection and apply-review control flow."""

    def setUp(self):
        self.workflow = LabellingWorkflow()

    def test_normalise_label_list_returns_empty_for_non_list_input(self):
        valid = {"bug": "bug", "docs": "docs"}
        self.assertEqual(normalise_label_list(None, valid), [])
        self.assertEqual(normalise_label_list("bug", valid), [])
        self.assertEqual(normalise_label_list(42, valid), [])

    def test_normalise_label_list_skips_non_string_entries(self):
        valid = {"bug": "bug"}
        self.assertEqual(
            normalise_label_list([1, None, "bug"], valid), ["bug"]
        )

    def test_drop_invalid_duplicate_existing_labels(self):
        suggestion = normalise_label_suggestions(
            {
                "add_labels": ["bug", "BUG", "unknown", "clang"],
                "remove_labels": ["bug", "docs"],
                "reason": "test",
            },
            [
                LabelDefinition("bug", ""),
                LabelDefinition("clang", ""),
                LabelDefinition("docs", ""),
            ],
            ["clang"],
        )
        self.assertEqual(suggestion.add_labels, ["bug"])
        self.assertEqual(suggestion.remove_labels, [])
        self.assertEqual(suggestion.reason, "test")

    def test_prompt_confirmation_retries_until_non_empty_valid_answer(self):
        answers = iter(["", "maybe", "a"])
        result = prompt_confirmation(
            "Prompt: ",
            allow_apply_all=False,
            input_fn=lambda _: next(answers),
        )
        self.assertEqual(result, "A")

    def test_prompt_confirmation_accepts_done_alias(self):
        result = prompt_confirmation(
            "Prompt: ",
            allow_apply_all=True,
            input_fn=lambda _: "d",
        )
        self.assertEqual(result, "D")

    def test_prompt_confirmation_raises_on_quit(self):
        with self.assertRaises(UserQuit):
            prompt_confirmation(
                "Prompt: ",
                allow_apply_all=False,
                input_fn=lambda _: "q",
            )

    def test_prompt_yes_no_uses_default_yes_on_empty_answer(self):
        self.assertTrue(
            prompt_yes_no("Retry? ", default_yes=True, input_fn=lambda _: "")
        )

    def test_prompt_yes_no_uses_default_no_on_empty_answer(self):
        self.assertFalse(
            prompt_yes_no("Retry? ", default_yes=False, input_fn=lambda _: "")
        )

    def test_prompt_yes_no_retries_until_valid_answer(self):
        answers = iter(["maybe", "n"])
        self.assertFalse(
            prompt_yes_no(
                "Retry? ",
                default_yes=True,
                input_fn=lambda _: next(answers),
            )
        )

    def test_select_items_to_handle_respects_yes_no_done(self):
        items = [
            make_item(1, "One"),
            make_item(2, "Two"),
            make_item(3, "Three"),
        ]
        answers = iter(["n", "y", "d"])

        with mock.patch("ai_labelling.workflow.print_item_details"):
            with mock.patch("builtins.print"):
                selected = self.workflow.select_items_to_handle(
                    items,
                    False,
                    input_fn=lambda _: next(answers),
                )

        self.assertEqual([item.number for item in selected], [2])

    def test_select_items_to_handle_all_shortcut_selects_remaining(self):
        items = [
            make_item(1, "One"),
            make_item(2, "Two"),
            make_item(3, "Three"),
        ]

        with mock.patch("ai_labelling.workflow.print_item_details"):
            with mock.patch("builtins.print"):
                selected = self.workflow.select_items_to_handle(
                    items,
                    False,
                    input_fn=lambda _: "a",
                )

        self.assertEqual([item.number for item in selected], [1, 2, 3])

    def test_review_and_apply_suggestions_respects_apply_prompts(self):
        suggestion_results = [
            SuggestionResult(
                item=make_item(1, "One"),
                label_suggestion=LabelSuggestion(
                    add_labels=["bug", "docs"],
                    remove_labels=["old", "stale"],
                    reason="reason one",
                ),
            )
        ]
        answers = iter(["a", "d"])

        with mock.patch.object(self.workflow, "print_summary"):
            with mock.patch.object(
                self.workflow,
                "add_labels_with_retry",
            ) as add_mock:
                with mock.patch.object(
                    self.workflow,
                    "remove_label_with_retry",
                ) as remove_mock:
                    self.workflow.review_and_apply_suggestions(
                        "llvm/llvm-project",
                        suggestion_results,
                        False,
                        True,
                        input_fn=lambda _: next(answers),
                    )

        self.assertEqual(add_mock.call_count, 2)
        self.assertEqual(remove_mock.call_count, 0)
        self.assertEqual(
            [call.args[2] for call in add_mock.call_args_list],
            [["bug"], ["docs"]],
        )

    def test_review_and_apply_suggestions_force_removes_labels_too(self):
        suggestion_results = [
            SuggestionResult(
                item=make_item(1, "One"),
                label_suggestion=LabelSuggestion(
                    add_labels=["bug"],
                    remove_labels=["old"],
                    reason="reason one",
                ),
            )
        ]

        with mock.patch.object(self.workflow, "print_summary"):
            with mock.patch.object(
                self.workflow,
                "add_labels_with_retry",
            ) as add_mock:
                with mock.patch.object(
                    self.workflow,
                    "remove_label_with_retry",
                ) as remove_mock:
                    self.workflow.review_and_apply_suggestions(
                        "llvm/llvm-project",
                        suggestion_results,
                        True,
                        True,
                    )

        add_mock.assert_called_once()
        remove_mock.assert_called_once_with(
            "llvm/llvm-project",
            suggestion_results[0].item,
            "old",
            input_fn=mock.ANY,
        )

    def test_run_ai_batch_returns_empty_for_empty_input(self):
        self.assertEqual(
            self.workflow.run_ai_batch([], [], "codex:*:low", False),
            [],
        )

    def test_run_ai_batch_uses_single_process_fast_path(self):
        item = make_item(1, "One")
        expected = SuggestionResult(
            item=item,
            label_suggestion=LabelSuggestion(
                add_labels=["bug"],
                remove_labels=[],
                reason="match",
            ),
        )

        with mock.patch("ai_labelling.workflow.os.cpu_count", return_value=1):
            with mock.patch.object(
                self.workflow,
                "build_suggestion_result_with_retry",
                return_value=expected,
            ) as build_mock:
                with mock.patch("builtins.print"):
                    result = self.workflow.run_ai_batch(
                        [item],
                        [LabelDefinition("bug", "Bug report")],
                        "codex:*:low",
                        False,
                        input_fn=lambda _: "n",
                    )

        self.assertEqual(result, [expected])
        build_mock.assert_called_once()

    def test_run_ai_batch_uses_process_pool_for_multiple_items(self):
        items = [make_item(1, "One"), make_item(2, "Two")]
        first_result = SuggestionResult(
            item=items[0],
            label_suggestion=LabelSuggestion(["bug"], [], "first"),
        )
        second_result = SuggestionResult(
            item=items[1],
            label_suggestion=LabelSuggestion(["docs"], [], "second"),
        )
        future_one = mock.Mock()
        future_one.result.return_value = first_result
        future_two = mock.Mock()
        future_two.result.return_value = second_result
        executor = mock.MagicMock()
        executor.__enter__.return_value = executor
        executor.submit.side_effect = [future_one, future_two]

        with mock.patch("ai_labelling.workflow.os.cpu_count", return_value=4):
            with mock.patch(
                "ai_labelling.workflow.concurrent.futures.ProcessPoolExecutor",
                return_value=executor,
            ):
                with mock.patch(
                    "ai_labelling.workflow.concurrent.futures.as_completed",
                    return_value=[future_two, future_one],
                ):
                    with mock.patch("builtins.print"):
                        result = self.workflow.run_ai_batch(
                            items,
                            [LabelDefinition("bug", "Bug report")],
                            "codex:*:low",
                            False,
                            input_fn=lambda _: "n",
                        )

        self.assertEqual(result, [first_result, second_result])
        self.assertEqual(executor.submit.call_count, 2)

    def test_build_suggestion_result_with_retry_retries_ai_failures(self):
        item = make_item(1, "One")
        expected = SuggestionResult(
            item=item,
            label_suggestion=LabelSuggestion(["bug"], [], "match"),
        )

        with mock.patch.object(
            self.workflow,
            "build_suggestion_result",
            side_effect=[RuntimeError("boom"), expected],
        ) as build_mock:
            with mock.patch(
                "ai_labelling.workflow.print_exception_diagnostics"
            ) as diag_mock:
                result = self.workflow.build_suggestion_result_with_retry(
                    item,
                    [LabelDefinition("bug", "Bug report")],
                    "codex:*:low",
                    False,
                    input_fn=lambda _: "y",
                )

        self.assertEqual(result, expected)
        self.assertEqual(build_mock.call_count, 2)
        diag_mock.assert_called_once()

    def test_build_suggestion_result_with_retry_skips_after_decline(self):
        item = make_item(1, "One")
        with mock.patch.object(
            self.workflow,
            "build_suggestion_result",
            side_effect=RuntimeError("boom"),
        ):
            with mock.patch(
                "ai_labelling.workflow.print_exception_diagnostics"
            ):
                result = self.workflow.build_suggestion_result_with_retry(
                    item,
                    [LabelDefinition("bug", "Bug report")],
                    "codex:*:low",
                    False,
                    input_fn=lambda _: "",
                )
        self.assertIsNone(result)

    def test_run_ai_batch_retries_failed_future_when_user_accepts(self):
        items = [make_item(1, "One"), make_item(2, "Two")]
        success = SuggestionResult(
            item=items[1],
            label_suggestion=LabelSuggestion(["docs"], [], "second"),
        )
        retry_result = SuggestionResult(
            item=items[0],
            label_suggestion=LabelSuggestion(["bug"], [], "retried"),
        )
        failed_future = mock.Mock()
        failed_future.result.side_effect = RuntimeError("boom")
        success_future = mock.Mock()
        success_future.result.return_value = success
        executor = mock.MagicMock()
        executor.__enter__.return_value = executor
        executor.submit.side_effect = [failed_future, success_future]

        with mock.patch("ai_labelling.workflow.os.cpu_count", return_value=4):
            with mock.patch(
                "ai_labelling.workflow.concurrent.futures.ProcessPoolExecutor",
                return_value=executor,
            ):
                with mock.patch(
                    "ai_labelling.workflow.concurrent.futures.as_completed",
                    return_value=[failed_future, success_future],
                ):
                    with mock.patch.object(
                        self.workflow,
                        "build_suggestion_result_with_retry",
                        return_value=retry_result,
                    ) as retry_mock:
                        with mock.patch(
                            "ai_labelling.workflow.print_exception_diagnostics"
                        ) as diag_mock:
                            with mock.patch("builtins.print"):
                                result = self.workflow.run_ai_batch(
                                    items,
                                    [LabelDefinition("bug", "Bug report")],
                                    "codex:*:low",
                                    False,
                                    input_fn=lambda _: "y",
                                )

        self.assertEqual(result, [retry_result, success])
        retry_mock.assert_called_once()
        diag_mock.assert_called_once()

    def test_add_labels_with_retry_uses_default_yes(self):
        item = make_item(1, "One")
        with mock.patch.object(
            self.workflow.github_client,
            "add_labels",
            side_effect=[RuntimeError("boom"), None],
        ) as add_mock:
            with mock.patch(
                "ai_labelling.workflow.print_exception_diagnostics"
            ) as diag_mock:
                self.workflow.add_labels_with_retry(
                    "llvm/llvm-project",
                    item,
                    ["bug"],
                    input_fn=lambda _: "",
                )

        self.assertEqual(add_mock.call_count, 2)
        diag_mock.assert_called_once()

    def test_remove_label_with_retry_uses_default_yes(self):
        item = make_item(1, "One")
        with mock.patch.object(
            self.workflow.github_client,
            "remove_label",
            side_effect=[RuntimeError("boom"), None],
        ) as remove_mock:
            with mock.patch(
                "ai_labelling.workflow.print_exception_diagnostics"
            ) as diag_mock:
                self.workflow.remove_label_with_retry(
                    "llvm/llvm-project",
                    item,
                    "bug",
                    input_fn=lambda _: "",
                )

        self.assertEqual(remove_mock.call_count, 2)
        diag_mock.assert_called_once()

    def test_warn_force_mode_waits_for_requested_delay(self):
        with mock.patch(
            "ai_labelling.workflow.time_module.sleep"
        ) as sleep_mock:
            with mock.patch("builtins.print"):
                self.workflow.warn_force_mode(False, 3)
        sleep_mock.assert_called_once_with(3)

    def test_collect_items_sorts_and_limits_created_results(self):
        newer = make_item(2, "Two")
        older = make_item(1, "One")
        newer.created_at = "2026-05-03T00:00:00Z"
        older.created_at = "2026-05-01T00:00:00Z"
        args = argparse.Namespace(
            created=True,
            date=DEFAULT_DATE_CUTOFF,
            include_closed=False,
            include_issues=True,
            include_open=True,
            include_prs=False,
            limit=1,
        )

        with mock.patch.object(
            self.workflow.github_client,
            "search_items",
            return_value=[older, newer],
        ):
            with mock.patch(
                "ai_labelling.workflow.default_cutoff",
                return_value=datetime(
                    2026,
                    5,
                    5,
                    0,
                    0,
                    tzinfo=timezone.utc,
                ),
            ):
                result = self.workflow.collect_items("llvm/llvm-project", args)

        self.assertEqual([item.number for item in result], [2])

    def test_collect_items_returns_empty_with_no_entity_types_enabled(self):
        args = argparse.Namespace(
            created=False,
            date=DEFAULT_DATE_CUTOFF,
            include_closed=False,
            include_issues=False,
            include_open=True,
            include_prs=False,
            limit=None,
        )

        with mock.patch.object(
            self.workflow.github_client,
            "search_items",
        ) as search_mock:
            result = self.workflow.collect_items("llvm/llvm-project", args)

        self.assertEqual(result, [])
        search_mock.assert_not_called()

    def test_collect_items_returns_empty_with_no_states_enabled(self):
        args = argparse.Namespace(
            created=False,
            date=DEFAULT_DATE_CUTOFF,
            include_closed=False,
            include_issues=True,
            include_open=False,
            include_prs=False,
            limit=None,
        )

        with mock.patch.object(
            self.workflow.github_client,
            "search_items",
        ) as search_mock:
            result = self.workflow.collect_items("llvm/llvm-project", args)

        self.assertEqual(result, [])
        search_mock.assert_not_called()

    def test_collect_items_limits_prs_by_remaining_budget(self):
        issue = make_item(1, "One")
        pr = make_item(2, "Two", kind="pr")

        def search_side_effect(_repo, kind, _options):
            return [issue] if kind == "issue" else [pr]

        args = argparse.Namespace(
            created=False,
            date=DEFAULT_DATE_CUTOFF,
            include_closed=False,
            include_issues=True,
            include_open=True,
            include_prs=True,
            limit=2,
        )

        with mock.patch.object(
            self.workflow.github_client,
            "search_items",
            side_effect=search_side_effect,
        ) as search_mock:
            with mock.patch(
                "ai_labelling.workflow.default_cutoff",
                return_value=datetime(2026, 5, 5, tzinfo=timezone.utc),
            ):
                result = self.workflow.collect_items(
                    "llvm/llvm-project", args
                )

        self.assertEqual(len(result), 2)
        self.assertEqual(search_mock.call_count, 2)
        # PR search must have received the remaining limit (2 - 1 = 1)
        pr_call_options = search_mock.call_args_list[1].args[2]
        self.assertEqual(pr_call_options.limit, 1)

    def test_print_summary_shows_additions_and_reason(self):
        item = make_item(1, "One", labels=["existing"])
        suggestion = LabelSuggestion(
            add_labels=["bug"],
            remove_labels=[],
            reason="Matches bug pattern.",
        )

        with mock.patch("ai_labelling.workflow.print_item_details"):
            with mock.patch("builtins.print") as print_mock:
                LabellingWorkflow.print_summary(
                    item,
                    suggestion,
                    force=False,
                    allow_label_removals=False,
                )

        printed = "\n".join(
            str(call.args[0])
            for call in print_mock.call_args_list
            if call.args
        )
        self.assertIn("would add", printed)
        self.assertIn("bug", printed)
        self.assertIn("Matches bug pattern.", printed)

    def test_print_summary_no_labels_shows_none_to_add(self):
        item = make_item(1, "One")
        suggestion = LabelSuggestion(
            add_labels=[], remove_labels=[], reason=""
        )

        with mock.patch("ai_labelling.workflow.print_item_details"):
            with mock.patch("builtins.print") as print_mock:
                LabellingWorkflow.print_summary(
                    item,
                    suggestion,
                    force=False,
                    allow_label_removals=False,
                )

        printed = "\n".join(
            str(call.args[0])
            for call in print_mock.call_args_list
            if call.args
        )
        self.assertIn("no new labels to add", printed)

    def test_print_summary_removals_disabled_shows_note(self):
        item = make_item(1, "One", labels=["old"])
        suggestion = LabelSuggestion(
            add_labels=[], remove_labels=["old"], reason=""
        )

        with mock.patch("ai_labelling.workflow.print_item_details"):
            with mock.patch("builtins.print") as print_mock:
                LabellingWorkflow.print_summary(
                    item,
                    suggestion,
                    force=False,
                    allow_label_removals=False,
                )

        printed = "\n".join(
            str(call.args[0])
            for call in print_mock.call_args_list
            if call.args
        )
        self.assertIn("removals are disabled", printed)

    def test_review_dry_run_skips_applying_labels(self):
        suggestion_results = [
            SuggestionResult(
                item=make_item(1, "One"),
                label_suggestion=LabelSuggestion(
                    add_labels=["bug"],
                    remove_labels=["old"],
                    reason="reason",
                ),
            )
        ]

        with mock.patch.object(self.workflow, "print_summary"):
            with mock.patch.object(
                self.workflow, "add_labels_with_retry"
            ) as add_mock:
                with mock.patch.object(
                    self.workflow, "remove_label_with_retry"
                ) as remove_mock:
                    with mock.patch("builtins.print"):
                        self.workflow.review_and_apply_suggestions(
                            "owner/repo",
                            suggestion_results,
                            False,
                            True,
                            dry_run=True,
                        )

        add_mock.assert_not_called()
        remove_mock.assert_not_called()

    def test_review_dry_run_prints_summary(self):
        suggestion_results = [
            SuggestionResult(
                item=make_item(1, "One"),
                label_suggestion=LabelSuggestion(
                    add_labels=["bug"],
                    remove_labels=[],
                    reason="",
                ),
            )
        ]

        with mock.patch.object(self.workflow, "print_summary"):
            with mock.patch(
                "ai_labelling.workflow.print_dry_run_summary"
            ) as dry_mock:
                self.workflow.review_and_apply_suggestions(
                    "owner/repo",
                    suggestion_results,
                    False,
                    False,
                    dry_run=True,
                )

        dry_mock.assert_called_once_with(suggestion_results, False)

    def test_print_summary_force_uses_present_tense(self):
        item = make_item(1, "One")
        suggestion = LabelSuggestion(
            add_labels=["bug"], remove_labels=["old"], reason=""
        )

        with mock.patch("ai_labelling.workflow.print_item_details"):
            with mock.patch("builtins.print") as print_mock:
                LabellingWorkflow.print_summary(
                    item,
                    suggestion,
                    force=True,
                    allow_label_removals=True,
                )

        printed = "\n".join(
            str(call.args[0])
            for call in print_mock.call_args_list
            if call.args
        )
        self.assertIn("adding", printed)
        self.assertIn("removing", printed)
