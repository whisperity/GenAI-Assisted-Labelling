"""Tests for the labelling workflow coordinator."""
# pylint: disable=missing-function-docstring,too-many-lines,duplicate-code

import argparse
import unittest
from datetime import datetime, timezone
from unittest import mock

from test.helpers import make_item
from ai_labelling.config import DEFAULT_DATE_CUTOFF
from ai_labelling.models import (
    ClosingPR,
    IssueTypeDefinition,
    LabelDefinition,
    LabelSuggestion,
    SuggestionResult,
    UserQuit,
)
from ai_labelling.workflow import LabellingWorkflow


class WorkflowTests(  # pylint: disable=too-many-public-methods
    unittest.TestCase,
):
    """Verify item-selection and apply-review control flow."""

    def setUp(self):
        self.workflow = LabellingWorkflow()

    def test_select_items_to_handle_force_returns_all(self):
        items = [make_item(1, "One"), make_item(2, "Two")]
        result = self.workflow.select_items_to_handle(items, True)
        self.assertEqual(result, items)

    def test_select_items_to_handle_respects_yes_no_done(self):
        items = [
            make_item(1, "One"),
            make_item(2, "Two"),
            make_item(3, "Three"),
        ]
        answers = iter(["n", "y", "d"])

        with mock.patch("ai_labelling.workflow.print_item_details"), \
                mock.patch("builtins.print"):
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

        with mock.patch("ai_labelling.workflow.print_item_details"), \
                mock.patch("builtins.print"):
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

        with mock.patch.object(self.workflow, "print_summary"), \
                mock.patch("ai_labelling.workflow.print_changes_summary"), \
                mock.patch.object(
                    self.workflow, "add_labels_with_retry"
                ) as add_mock, \
                mock.patch.object(
                    self.workflow, "remove_label_with_retry"
                ) as remove_mock:
            self.workflow.review_and_apply_suggestions(
                "llvm/llvm-project",
                suggestion_results,
                False,
                True,
                input_fn=lambda _: next(answers),
            )

        self.assertEqual(add_mock.call_count, 1)
        self.assertEqual(remove_mock.call_count, 0)
        self.assertEqual(add_mock.call_args.args[2], ["bug", "docs"])

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

        with mock.patch.object(self.workflow, "print_summary"), \
                mock.patch("ai_labelling.workflow.print_changes_summary"), \
                mock.patch.object(
                    self.workflow, "add_labels_with_retry"
                ) as add_mock, \
                mock.patch.object(
                    self.workflow, "remove_label_with_retry"
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

    def test_review_apply_label_change_bucket_skips_with_done(self):
        """``D`` answer must abort the bucket without further prompts."""

        suggestion_results = [
            SuggestionResult(
                item=make_item(1, "One"),
                label_suggestion=LabelSuggestion(
                    add_labels=["a", "b", "c"],
                    remove_labels=[],
                    reason="",
                ),
            )
        ]

        with mock.patch.object(self.workflow, "print_summary"), \
                mock.patch("ai_labelling.workflow.print_changes_summary"), \
                mock.patch.object(
                    self.workflow, "add_labels_with_retry"
                ) as add_mock:
            self.workflow.review_and_apply_suggestions(
                "owner/repo",
                suggestion_results,
                False,
                False,
                input_fn=lambda _: "d",
            )

        add_mock.assert_not_called()

    def test_review_apply_label_change_bucket_skips_with_no(self):
        """``N`` skips that label but continues prompting the next."""

        suggestion_results = [
            SuggestionResult(
                item=make_item(1, "One"),
                label_suggestion=LabelSuggestion(
                    add_labels=["a", "b"],
                    remove_labels=[],
                    reason="",
                ),
            )
        ]
        answers = iter(["n", "y"])

        with mock.patch.object(self.workflow, "print_summary"), \
                mock.patch("ai_labelling.workflow.print_changes_summary"), \
                mock.patch.object(
                    self.workflow, "add_labels_with_retry"
                ) as add_mock:
            self.workflow.review_and_apply_suggestions(
                "owner/repo",
                suggestion_results,
                False,
                False,
                input_fn=lambda _: next(answers),
            )

        self.assertEqual(add_mock.call_count, 1)
        self.assertEqual(add_mock.call_args.args[2], ["b"])

    def test_review_quit_during_collection_aborts_without_api_calls(self):
        """``q`` during collection raises UserQuit and fires no API calls."""

        item = make_item(1, "One")
        suggestion_results = [
            SuggestionResult(
                item=item,
                label_suggestion=LabelSuggestion(
                    add_labels=["bug", "docs"],
                    remove_labels=[],
                    reason="r",
                ),
            )
        ]
        # User accepts "bug", quits at "docs" — no API call should fire.
        answers = iter(["y", "q"])

        with mock.patch.object(self.workflow, "print_summary"), \
                mock.patch("ai_labelling.workflow.print_changes_summary"), \
                mock.patch.object(
                    self.workflow, "add_labels_with_retry"
                ) as add_mock, \
                mock.patch.object(
                    self.workflow, "remove_label_with_retry"
                ) as remove_mock, \
                mock.patch.object(
                    self.workflow, "set_issue_type_with_retry"
                ) as type_mock:
            with self.assertRaises(UserQuit):
                self.workflow.review_and_apply_suggestions(
                    "owner/repo",
                    suggestion_results,
                    False,
                    True,
                    input_fn=lambda _: next(answers),
                )

        add_mock.assert_not_called()
        remove_mock.assert_not_called()
        type_mock.assert_not_called()

    def test_review_batches_label_adds_into_single_api_call(self):
        """All accepted adds for one item go in one ``add_labels`` call."""

        suggestion_results = [
            SuggestionResult(
                item=make_item(1, "One"),
                label_suggestion=LabelSuggestion(
                    add_labels=["a", "b", "c"],
                    remove_labels=[],
                    reason="r",
                ),
            )
        ]

        with mock.patch.object(self.workflow, "print_summary"), \
                mock.patch("ai_labelling.workflow.print_changes_summary"), \
                mock.patch.object(
                    self.workflow, "add_labels_with_retry"
                ) as add_mock:
            self.workflow.review_and_apply_suggestions(
                "owner/repo",
                suggestion_results,
                True,
                False,
            )

        add_mock.assert_called_once()
        self.assertEqual(add_mock.call_args.args[2], ["a", "b", "c"])

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

        with mock.patch(
            "ai_labelling.workflow.os.cpu_count", return_value=1
        ), mock.patch.object(
            self.workflow,
            "build_suggestion_result_with_retry",
            return_value=expected,
        ) as build_mock, mock.patch("builtins.print"):
            result = self.workflow.run_ai_batch(
                [item],
                [LabelDefinition("bug", "Bug report")],
                "codex:*:low",
                False,
                input_fn=lambda _: "n",
            )

        self.assertEqual(result, [expected])
        build_mock.assert_called_once()

    def test_run_ai_batch_serial_skips_none_results(self):
        """Declined retry returns ``None``; the batch must drop those."""

        items = [make_item(1, "One"), make_item(2, "Two")]
        ok = SuggestionResult(
            item=items[1],
            label_suggestion=LabelSuggestion(["bug"], [], ""),
        )

        with mock.patch(
            "ai_labelling.workflow.os.cpu_count", return_value=1
        ), mock.patch.object(
            self.workflow,
            "build_suggestion_result_with_retry",
            side_effect=[None, ok],
        ), mock.patch("builtins.print"):
            result = self.workflow.run_ai_batch(
                items,
                [LabelDefinition("bug", "")],
                "codex:*:low",
                False,
                input_fn=lambda _: "n",
            )

        self.assertEqual(result, [ok])

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

        with mock.patch(
            "ai_labelling.workflow.os.cpu_count", return_value=4
        ), mock.patch(
            "ai_labelling.workflow.concurrent.futures.ProcessPoolExecutor",
            return_value=executor,
        ), mock.patch(
            "ai_labelling.workflow.concurrent.futures.as_completed",
            return_value=[future_two, future_one],
        ), mock.patch("builtins.print"):
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
        ) as build_mock, mock.patch(
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
        ), mock.patch(
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

    def test_build_suggestion_result_passes_through_backend(self):
        """build_suggestion_result must call backend.suggest_labels."""

        item = make_item(1, "One")
        with mock.patch(
            "ai_labelling.workflow.parse_model_spec",
            return_value=mock.Mock(provider="codex", model="m"),
        ), mock.patch(
            "ai_labelling.workflow.get_backend_for_provider"
        ) as backend_factory:
            backend = backend_factory.return_value
            backend.suggest_labels.return_value = {
                "add_labels": [],
                "remove_labels": [],
                "reason": "",
            }
            result = self.workflow.build_suggestion_result(
                item,
                [LabelDefinition("bug", "")],
                "codex:m",
                False,
            )

        self.assertEqual(result.item, item)
        self.assertEqual(result.model, "codex:m")
        backend.suggest_labels.assert_called_once()

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

        with mock.patch(
            "ai_labelling.workflow.os.cpu_count", return_value=4
        ), mock.patch(
            "ai_labelling.workflow.concurrent.futures.ProcessPoolExecutor",
            return_value=executor,
        ), mock.patch(
            "ai_labelling.workflow.concurrent.futures.as_completed",
            return_value=[failed_future, success_future],
        ), mock.patch.object(
            self.workflow,
            "build_suggestion_result_with_retry",
            return_value=retry_result,
        ) as retry_mock, mock.patch(
            "ai_labelling.workflow.print_exception_diagnostics"
        ) as diag_mock, mock.patch("builtins.print"):
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

    def test_run_ai_batch_drops_failed_when_user_declines(self):
        items = [make_item(1, "One"), make_item(2, "Two")]
        failed_future = mock.Mock()
        failed_future.result.side_effect = RuntimeError("boom")
        ok_future = mock.Mock()
        ok_future.result.return_value = SuggestionResult(
            item=items[1],
            label_suggestion=LabelSuggestion(["bug"], [], ""),
        )
        executor = mock.MagicMock()
        executor.__enter__.return_value = executor
        executor.submit.side_effect = [failed_future, ok_future]

        with mock.patch(
            "ai_labelling.workflow.os.cpu_count", return_value=4
        ), mock.patch(
            "ai_labelling.workflow.concurrent.futures.ProcessPoolExecutor",
            return_value=executor,
        ), mock.patch(
            "ai_labelling.workflow.concurrent.futures.as_completed",
            return_value=[failed_future, ok_future],
        ), mock.patch(
            "ai_labelling.workflow.print_exception_diagnostics"
        ), mock.patch("builtins.print"):
            result = self.workflow.run_ai_batch(
                items,
                [],
                "codex:*:low",
                False,
                input_fn=lambda _: "n",
            )

        self.assertEqual([r.item.number for r in result], [2])

    def test_add_labels_with_retry_uses_default_yes(self):
        item = make_item(1, "One")
        with mock.patch.object(
            self.workflow.github_client,
            "add_labels",
            side_effect=[RuntimeError("boom"), None],
        ) as add_mock, mock.patch(
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
        ) as remove_mock, mock.patch(
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

    def test_set_issue_type_with_retry_aborts_on_decline(self):
        item = make_item(1, "One")
        type_def = IssueTypeDefinition("Bug", "", 1)
        with mock.patch.object(
            self.workflow.github_client,
            "set_issue_type",
            side_effect=RuntimeError("boom"),
        ) as set_mock, mock.patch(
            "ai_labelling.workflow.print_exception_diagnostics"
        ):
            self.workflow.set_issue_type_with_retry(
                "owner/repo", item, type_def, input_fn=lambda _: "n",
            )
        self.assertEqual(set_mock.call_count, 1)

    def test_warn_force_mode_waits_for_requested_delay(self):
        with mock.patch(
            "ai_labelling.workflow.time_module.sleep"
        ) as sleep_mock, mock.patch("builtins.print"):
            self.workflow.warn_force_mode(False, 3)
        sleep_mock.assert_called_once_with(3)

    def test_warn_force_mode_dry_run_message(self):
        with mock.patch(
            "ai_labelling.workflow.time_module.sleep"
        ), mock.patch("builtins.print") as print_mock:
            self.workflow.warn_force_mode(True, 0)
        printed = " ".join(
            str(c.args[0]) for c in print_mock.call_args_list if c.args
        )
        self.assertNotIn("apply every suggested label", printed)

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
        ), mock.patch(
            "ai_labelling.workflow.default_cutoff",
            return_value=datetime(2026, 5, 5, tzinfo=timezone.utc),
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
            self.workflow.github_client, "search_items"
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
            self.workflow.github_client, "search_items"
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
        ) as search_mock, mock.patch(
            "ai_labelling.workflow.default_cutoff",
            return_value=datetime(2026, 5, 5, tzinfo=timezone.utc),
        ):
            result = self.workflow.collect_items(
                "llvm/llvm-project", args
            )

        self.assertEqual(len(result), 2)
        self.assertEqual(search_mock.call_count, 2)
        pr_call_options = search_mock.call_args_list[1].args[2]
        self.assertEqual(pr_call_options.limit, 1)

    def test_collect_items_skips_pr_search_when_budget_exhausted(self):
        issue = make_item(1, "One")
        args = argparse.Namespace(
            created=False,
            date=DEFAULT_DATE_CUTOFF,
            include_closed=False,
            include_issues=True,
            include_open=True,
            include_prs=True,
            limit=1,
        )

        with mock.patch.object(
            self.workflow.github_client,
            "search_items",
            return_value=[issue],
        ) as search_mock, mock.patch(
            "ai_labelling.workflow.default_cutoff",
            return_value=datetime(2026, 5, 5, tzinfo=timezone.utc),
        ):
            result = self.workflow.collect_items("owner/repo", args)

        self.assertEqual(len(result), 1)
        self.assertEqual(search_mock.call_count, 1)

    def test_collect_items_id_mode_bypasses_search(self):
        item = make_item(7, "Direct lookup")
        args = argparse.Namespace(id=7)

        with mock.patch.object(
            self.workflow.github_client,
            "get_item",
            return_value=item,
        ) as get_mock, mock.patch.object(
            self.workflow.github_client, "search_items"
        ) as search_mock:
            result = self.workflow.collect_items("llvm/llvm-project", args)

        get_mock.assert_called_once_with("llvm/llvm-project", 7)
        search_mock.assert_not_called()
        self.assertEqual(result, [item])

    def test_collect_items_id_none_falls_through_to_normal_search(self):
        args = argparse.Namespace(
            id=None,
            created=False,
            date=DEFAULT_DATE_CUTOFF,
            include_closed=False,
            include_issues=False,
            include_open=True,
            include_prs=False,
            limit=None,
        )

        with mock.patch.object(
            self.workflow.github_client, "search_items"
        ) as search_mock:
            result = self.workflow.collect_items("llvm/llvm-project", args)

        search_mock.assert_not_called()
        self.assertEqual(result, [])

    def test_print_summary_shows_additions_and_reason(self):
        item = make_item(1, "One", labels=["existing"])
        suggestion = LabelSuggestion(
            add_labels=["bug"],
            remove_labels=[],
            reason="Matches bug pattern.",
        )

        with mock.patch("ai_labelling.workflow.print_item_details"), \
                mock.patch("builtins.print") as print_mock:
            LabellingWorkflow.print_summary(item, suggestion)

        printed = "\n".join(
            str(call.args[0])
            for call in print_mock.call_args_list
            if call.args
        )
        self.assertIn("Suggested additions:", printed)
        self.assertIn("bug", printed)
        self.assertIn("Matches bug pattern.", printed)

    def test_print_summary_no_additions_omits_suggested_additions(self):
        item = make_item(1, "One")
        suggestion = LabelSuggestion(
            add_labels=[], remove_labels=[], reason=""
        )

        with mock.patch("ai_labelling.workflow.print_item_details"), \
                mock.patch("builtins.print") as print_mock:
            LabellingWorkflow.print_summary(item, suggestion)

        printed = "\n".join(
            str(call.args[0])
            for call in print_mock.call_args_list
            if call.args
        )
        self.assertNotIn("Suggested additions:", printed)

    def test_print_summary_shows_existing_issue_type(self):
        item = make_item(1, "One")
        item.issue_type = "Bug"
        with mock.patch("ai_labelling.workflow.print_item_details"), \
                mock.patch("builtins.print") as print_mock:
            LabellingWorkflow.print_summary(
                item, LabelSuggestion([], [], ""),
            )
        printed = "\n".join(
            str(c.args[0]) for c in print_mock.call_args_list if c.args
        )
        self.assertIn("Issue type:", printed)
        self.assertIn("Bug", printed)

    def test_print_summary_shows_removals_when_suggested(self):
        item = make_item(1, "One", labels=["old"])
        suggestion = LabelSuggestion(
            add_labels=[], remove_labels=["old"], reason=""
        )

        with mock.patch("ai_labelling.workflow.print_item_details"), \
                mock.patch("builtins.print") as print_mock:
            LabellingWorkflow.print_summary(item, suggestion)

        printed = "\n".join(
            str(call.args[0])
            for call in print_mock.call_args_list
            if call.args
        )
        self.assertIn("Suggested removals:", printed)
        self.assertIn("old", printed)

    def test_print_summary_shows_both_additions_and_removals(self):
        item = make_item(1, "One")
        suggestion = LabelSuggestion(
            add_labels=["bug"], remove_labels=["old"], reason=""
        )

        with mock.patch("ai_labelling.workflow.print_item_details"), \
                mock.patch("builtins.print") as print_mock:
            LabellingWorkflow.print_summary(item, suggestion)

        printed = "\n".join(
            str(call.args[0])
            for call in print_mock.call_args_list
            if call.args
        )
        self.assertIn("Suggested additions:", printed)
        self.assertIn("Suggested removals:", printed)

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

        with mock.patch.object(self.workflow, "print_summary"), \
                mock.patch("ai_labelling.workflow.print_changes_summary"), \
                mock.patch.object(
                    self.workflow, "add_labels_with_retry"
                ) as add_mock, \
                mock.patch.object(
                    self.workflow, "remove_label_with_retry"
                ) as remove_mock:
            self.workflow.review_and_apply_suggestions(
                "owner/repo",
                suggestion_results,
                False,
                True,
                dry_run=True,
            )

        add_mock.assert_not_called()
        remove_mock.assert_not_called()

    def test_review_dry_run_prints_summary_with_dry_run_flag(self):
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

        with mock.patch.object(self.workflow, "print_summary"), \
                mock.patch(
                    "ai_labelling.workflow.print_changes_summary"
                ) as summary_mock:
            self.workflow.review_and_apply_suggestions(
                "owner/repo",
                suggestion_results,
                False,
                False,
                dry_run=True,
            )

        summary_mock.assert_called_once_with(
            suggestion_results, False, dry_run=True
        )

    def test_comment_reason_posts_comment_after_applying_labels(self):
        item = make_item(1, "One")
        suggestion_results = [
            SuggestionResult(
                item=item,
                label_suggestion=LabelSuggestion(
                    add_labels=["bug"],
                    remove_labels=[],
                    reason="matches bug pattern",
                ),
                model="anthropic:claude-haiku-4-5-20251001",
            )
        ]

        with mock.patch.object(self.workflow, "print_summary"), \
                mock.patch("ai_labelling.workflow.print_changes_summary"), \
                mock.patch(
                    "ai_labelling.workflow.get_script_version",
                    return_value="abc1234",
                ), mock.patch.object(
                    self.workflow, "add_labels_with_retry"
                ), mock.patch.object(
                    self.workflow.github_client, "post_comment"
                ) as comment_mock:
            self.workflow.review_and_apply_suggestions(
                "owner/repo",
                suggestion_results,
                True,
                False,
                comment_reason=True,
            )

        comment_mock.assert_called_once()
        _repo, _item, body = comment_mock.call_args.args
        self.assertIn("abc1234", body)
        self.assertIn("claude-haiku-4-5-20251001", body)
        self.assertIn("matches bug pattern", body)

    def test_comment_reason_swallows_post_comment_failure(self):
        """A failed comment post must not abort the run."""

        item = make_item(1, "One")
        suggestion_results = [
            SuggestionResult(
                item=item,
                label_suggestion=LabelSuggestion(["bug"], [], ""),
            )
        ]
        with mock.patch.object(self.workflow, "print_summary"), \
                mock.patch("ai_labelling.workflow.print_changes_summary"), \
                mock.patch(
                    "ai_labelling.workflow.get_script_version",
                    return_value="ver",
                ), mock.patch.object(
                    self.workflow, "add_labels_with_retry"
                ), mock.patch.object(
                    self.workflow.github_client,
                    "post_comment",
                    side_effect=RuntimeError("boom"),
                ), mock.patch(
                    "ai_labelling.workflow.print_exception_diagnostics"
                ) as diag_mock:
            self.workflow.review_and_apply_suggestions(
                "owner/repo",
                suggestion_results,
                True,
                False,
                comment_reason=True,
            )
        diag_mock.assert_called_once()

    def test_review_prompts_issue_type_before_label_additions(self):
        item = make_item(1, "One")
        bug_type = IssueTypeDefinition("Bug", "A bug", 1)
        suggestion_results = [
            SuggestionResult(
                item=item,
                label_suggestion=LabelSuggestion(
                    add_labels=["docs"],
                    remove_labels=[],
                    reason="",
                    issue_type="Bug",
                ),
            )
        ]
        call_order = []

        def fake_set_type(*_args, **_kwargs):
            call_order.append("type")

        def fake_add(*_args, **_kwargs):
            call_order.append("add")

        with mock.patch.object(self.workflow, "print_summary"), \
                mock.patch("ai_labelling.workflow.print_changes_summary"), \
                mock.patch.object(
                    self.workflow,
                    "set_issue_type_with_retry",
                    side_effect=fake_set_type,
                ), mock.patch.object(
                    self.workflow,
                    "add_labels_with_retry",
                    side_effect=fake_add,
                ):
            self.workflow.review_and_apply_suggestions(
                "owner/repo",
                suggestion_results,
                True,
                False,
                valid_issue_types=[bug_type],
            )

        self.assertEqual(call_order, ["type", "add"])

    def test_review_skips_issue_type_for_prs(self):
        pr = make_item(1, "One", kind="pr")
        bug_type = IssueTypeDefinition("Bug", "A bug", 1)
        suggestion_results = [
            SuggestionResult(
                item=pr,
                label_suggestion=LabelSuggestion(
                    add_labels=[],
                    remove_labels=[],
                    reason="",
                    issue_type="Bug",
                ),
            )
        ]

        with mock.patch.object(self.workflow, "print_summary"), \
                mock.patch("ai_labelling.workflow.print_changes_summary"), \
                mock.patch.object(
                    self.workflow, "set_issue_type_with_retry"
                ) as type_mock:
            self.workflow.review_and_apply_suggestions(
                "owner/repo",
                suggestion_results,
                True,
                False,
                valid_issue_types=[bug_type],
            )

        type_mock.assert_not_called()

    def test_review_issue_type_skipped_when_not_in_valid_map(self):
        """Suggested issue types unknown to the repo must be ignored."""

        item = make_item(1, "One")
        suggestion_results = [
            SuggestionResult(
                item=item,
                label_suggestion=LabelSuggestion(
                    add_labels=[],
                    remove_labels=[],
                    reason="",
                    issue_type="Mystery",
                ),
            )
        ]
        with mock.patch.object(self.workflow, "print_summary"), \
                mock.patch("ai_labelling.workflow.print_changes_summary"), \
                mock.patch.object(
                    self.workflow, "set_issue_type_with_retry"
                ) as type_mock:
            self.workflow.review_and_apply_suggestions(
                "owner/repo",
                suggestion_results,
                True,
                False,
                valid_issue_types=[IssueTypeDefinition("Bug", "", 1)],
            )
        type_mock.assert_not_called()

    def test_review_issue_type_prompt_accepted(self):
        item = make_item(1, "One")
        bug_type = IssueTypeDefinition("Bug", "A bug", 1)
        suggestion_results = [
            SuggestionResult(
                item=item,
                label_suggestion=LabelSuggestion(
                    add_labels=[],
                    remove_labels=[],
                    reason="",
                    issue_type="Bug",
                ),
            )
        ]

        with mock.patch.object(self.workflow, "print_summary"), \
                mock.patch("ai_labelling.workflow.print_changes_summary"), \
                mock.patch.object(
                    self.workflow, "set_issue_type_with_retry"
                ) as type_mock:
            self.workflow.review_and_apply_suggestions(
                "owner/repo",
                suggestion_results,
                False,
                False,
                input_fn=lambda _: "y",
                valid_issue_types=[bug_type],
            )

        type_mock.assert_called_once()

    def test_review_issue_type_prompt_declined(self):
        item = make_item(1, "One")
        bug_type = IssueTypeDefinition("Bug", "A bug", 1)
        suggestion_results = [
            SuggestionResult(
                item=item,
                label_suggestion=LabelSuggestion(
                    add_labels=[],
                    remove_labels=[],
                    reason="",
                    issue_type="Bug",
                ),
            )
        ]

        with mock.patch.object(self.workflow, "print_summary"), \
                mock.patch("ai_labelling.workflow.print_changes_summary"), \
                mock.patch.object(
                    self.workflow, "set_issue_type_with_retry"
                ) as type_mock:
            self.workflow.review_and_apply_suggestions(
                "owner/repo",
                suggestion_results,
                False,
                False,
                input_fn=lambda _: "n",
                valid_issue_types=[bug_type],
            )

        type_mock.assert_not_called()

    def test_comment_reason_not_posted_when_dry_run(self):
        item = make_item(1, "One")
        suggestion_results = [
            SuggestionResult(
                item=item,
                label_suggestion=LabelSuggestion(
                    add_labels=["bug"],
                    remove_labels=[],
                    reason="",
                ),
            )
        ]

        with mock.patch.object(self.workflow, "print_summary"), \
                mock.patch("ai_labelling.workflow.print_changes_summary"), \
                mock.patch.object(
                    self.workflow.github_client, "post_comment"
                ) as comment_mock:
            self.workflow.review_and_apply_suggestions(
                "owner/repo",
                suggestion_results,
                True,
                False,
                dry_run=True,
                comment_reason=True,
            )

        comment_mock.assert_not_called()


class InteractiveModeTests(unittest.TestCase):
    """Verify the interactive number-entry loop in run_interactive_mode."""

    def setUp(self):
        self.workflow = LabellingWorkflow()

    def _run(self, inputs, *, dry_run=False, allow_removals=False,
             issue_types=()):
        """Helper: run interactive mode with a fixed input sequence."""

        answers = iter(inputs)
        with mock.patch("ai_labelling.workflow.print_changes_summary"), \
                mock.patch("ai_labelling.workflow.print_item_details"), \
                mock.patch("builtins.print"):
            self.workflow.run_interactive_mode(
                "owner/repo",
                [LabelDefinition("bug", "a bug")],
                "anthropic",
                allow_removals,
                dry_run=dry_run,
                valid_issue_types=list(issue_types),
                input_fn=lambda _: next(answers),
            )

    def test_empty_input_reprompts_without_exiting(self):
        answers = iter(["", "", "q"])
        call_count = [0]

        def counting_input(_):
            call_count[0] += 1
            return next(answers)

        with mock.patch("ai_labelling.workflow.print_changes_summary"), \
                mock.patch("builtins.print"):
            self.workflow.run_interactive_mode(
                "owner/repo",
                [],
                "anthropic",
                False,
                input_fn=counting_input,
            )

        self.assertGreaterEqual(call_count[0], 3)

    def test_q_exits_loop(self):
        with mock.patch("ai_labelling.workflow.print_changes_summary") as s, \
                mock.patch("builtins.print"):
            self.workflow.run_interactive_mode(
                "owner/repo",
                [],
                "anthropic",
                False,
                input_fn=lambda _: "q",
            )
        s.assert_called_once()

    def test_quit_exits_loop(self):
        with mock.patch("ai_labelling.workflow.print_changes_summary") as s, \
                mock.patch("builtins.print"):
            self.workflow.run_interactive_mode(
                "owner/repo",
                [],
                "anthropic",
                False,
                input_fn=lambda _: "quit",
            )
        s.assert_called_once()

    def test_invalid_number_prints_error_and_continues(self):
        answers = iter(["abc", "q"])
        with mock.patch("ai_labelling.workflow.print_changes_summary"), \
                mock.patch("builtins.print") as print_mock:
            self.workflow.run_interactive_mode(
                "owner/repo",
                [],
                "anthropic",
                False,
                input_fn=lambda _: next(answers),
            )
        msgs = [str(c) for c in print_mock.call_args_list]
        self.assertTrue(any("abc" in m for m in msgs))

    def test_negative_number_prints_error(self):
        answers = iter(["-1", "q"])
        with mock.patch("ai_labelling.workflow.print_changes_summary"), \
                mock.patch("builtins.print") as print_mock:
            self.workflow.run_interactive_mode(
                "owner/repo",
                [],
                "anthropic",
                False,
                input_fn=lambda _: next(answers),
            )
        msgs = [str(c) for c in print_mock.call_args_list]
        self.assertTrue(any("-1" in m for m in msgs))

    def test_fetch_error_continues_loop(self):
        answers = iter(["1", "q"])

        with mock.patch("ai_labelling.workflow.print_changes_summary"), \
                mock.patch("ai_labelling.workflow.print_item_details"), \
                mock.patch(
                    "ai_labelling.workflow.print_exception_diagnostics"
                ) as diag_mock, \
                mock.patch.object(
                    self.workflow.github_client,
                    "get_item",
                    side_effect=RuntimeError("not found"),
                ), \
                mock.patch("builtins.print"):
            self.workflow.run_interactive_mode(
                "owner/repo",
                [],
                "anthropic",
                False,
                input_fn=lambda _: next(answers),
            )

        diag_mock.assert_called_once()

    def test_n_skips_item_without_ai(self):
        item = make_item(1, "One")
        answers = iter(["1", "n", "q"])

        with mock.patch("ai_labelling.workflow.print_changes_summary"), \
                mock.patch("ai_labelling.workflow.print_item_details"), \
                mock.patch.object(
                    self.workflow.github_client, "get_item", return_value=item
                ), \
                mock.patch.object(
                    self.workflow,
                    "build_suggestion_result_with_retry",
                ) as ai_mock, \
                mock.patch("builtins.print"):
            self.workflow.run_interactive_mode(
                "owner/repo",
                [],
                "anthropic",
                False,
                input_fn=lambda _: next(answers),
            )

        ai_mock.assert_not_called()

    def test_userquit_at_handle_prompt_exits_loop(self):
        item = make_item(1, "One")
        answers = iter(["1", "q"])

        with mock.patch("ai_labelling.workflow.print_changes_summary") as s, \
                mock.patch("ai_labelling.workflow.print_item_details"), \
                mock.patch.object(
                    self.workflow.github_client, "get_item", return_value=item
                ), \
                mock.patch.object(
                    self.workflow, "build_suggestion_result_with_retry"
                ) as ai_mock, \
                mock.patch("builtins.print"):
            self.workflow.run_interactive_mode(
                "owner/repo",
                [],
                "anthropic",
                False,
                input_fn=lambda _: next(answers),
            )

        ai_mock.assert_not_called()
        s.assert_called_once()

    def test_full_pipeline_applies_label(self):
        item = make_item(1, "One")
        suggestion = LabelSuggestion(
            add_labels=["bug"], remove_labels=[], reason="looks like a bug"
        )
        result = SuggestionResult(item=item, label_suggestion=suggestion)
        answers = iter(["1", "y", "y", "q"])

        with mock.patch("ai_labelling.workflow.print_changes_summary"), \
                mock.patch("ai_labelling.workflow.print_item_details"), \
                mock.patch.object(self.workflow, "print_summary"), \
                mock.patch.object(
                    self.workflow.github_client, "get_item", return_value=item
                ), \
                mock.patch.object(
                    self.workflow,
                    "build_suggestion_result_with_retry",
                    return_value=result,
                ), \
                mock.patch.object(
                    self.workflow, "add_labels_with_retry"
                ) as add_mock, \
                mock.patch("builtins.print"):
            self.workflow.run_interactive_mode(
                "owner/repo",
                [LabelDefinition("bug", "a bug")],
                "anthropic",
                False,
                input_fn=lambda _: next(answers),
            )

        add_mock.assert_called_once()
        self.assertIn("bug", add_mock.call_args.args[2])

    def test_userquit_during_apply_exits_loop(self):
        item = make_item(1, "One")
        suggestion = LabelSuggestion(
            add_labels=["bug"], remove_labels=[], reason="r"
        )
        result = SuggestionResult(item=item, label_suggestion=suggestion)

        def inputs(prompt):
            if "Issue/PR" in prompt:
                return "1"
            if "with AI" in prompt:
                return "y"
            raise UserQuit  # q at the label-add prompt

        with mock.patch("ai_labelling.workflow.print_changes_summary") as s, \
                mock.patch("ai_labelling.workflow.print_item_details"), \
                mock.patch.object(self.workflow, "print_summary"), \
                mock.patch.object(
                    self.workflow.github_client, "get_item", return_value=item
                ), \
                mock.patch.object(
                    self.workflow,
                    "build_suggestion_result_with_retry",
                    return_value=result,
                ), \
                mock.patch.object(
                    self.workflow, "add_labels_with_retry"
                ) as add_mock, \
                mock.patch("builtins.print"):
            self.workflow.run_interactive_mode(
                "owner/repo",
                [LabelDefinition("bug", "a bug")],
                "anthropic",
                False,
                input_fn=inputs,
            )

        add_mock.assert_not_called()
        s.assert_called_once()

    def test_dry_run_skips_api_calls(self):
        item = make_item(1, "One")
        suggestion = LabelSuggestion(
            add_labels=["bug"], remove_labels=[], reason="r"
        )
        result = SuggestionResult(item=item, label_suggestion=suggestion)
        answers = iter(["1", "y", "q"])

        with mock.patch("ai_labelling.workflow.print_changes_summary") as s, \
                mock.patch("ai_labelling.workflow.print_item_details"), \
                mock.patch.object(self.workflow, "print_summary"), \
                mock.patch.object(
                    self.workflow.github_client, "get_item", return_value=item
                ), \
                mock.patch.object(
                    self.workflow,
                    "build_suggestion_result_with_retry",
                    return_value=result,
                ), \
                mock.patch.object(
                    self.workflow, "add_labels_with_retry"
                ) as add_mock, \
                mock.patch("builtins.print"):
            self.workflow.run_interactive_mode(
                "owner/repo",
                [LabelDefinition("bug", "a bug")],
                "anthropic",
                False,
                dry_run=True,
                input_fn=lambda _: next(answers),
            )

        add_mock.assert_not_called()
        summary_call_args = s.call_args
        summary_results = summary_call_args.args[0]
        self.assertEqual(len(summary_results), 1)


class AssignIssueToSolverTests(unittest.TestCase):
    """Verify the --assign-issue-to-solver decision and apply logic."""

    # pylint: disable=protected-access

    def setUp(self):
        self.workflow = LabellingWorkflow()

    def _make_result(self, state="closed"):
        item = make_item(1, "Bug", state=state, kind="issue")
        return SuggestionResult(
            item=item,
            label_suggestion=LabelSuggestion(
                add_labels=[], remove_labels=[], reason="r"
            ),
        )

    def test_find_closing_pr_skips_open_issues(self):
        item = make_item(1, "T", state="open", kind="issue")
        result = self.workflow._find_closing_pr("owner/repo", item)
        self.assertIsNone(result)

    def test_find_closing_pr_skips_prs(self):
        item = make_item(1, "T", state="closed", kind="pr")
        result = self.workflow._find_closing_pr("owner/repo", item)
        self.assertIsNone(result)

    def test_find_closing_pr_returns_client_result(self):
        item = make_item(1, "T", state="closed", kind="issue")
        with mock.patch.object(
            self.workflow.github_client,
            "get_closing_pr",
            return_value=ClosingPR(pr_number=42, author_login="dev"),
        ):
            result = self.workflow._find_closing_pr("owner/repo", item)
        self.assertEqual(result, ClosingPR(pr_number=42, author_login="dev"))

    def test_collect_assignee_returns_false_when_no_pr(self):
        result = self.workflow._collect_assignee_decision(
            make_item(1, "T"),
            closing_pr=None,
            force=False,
            input_fn=lambda _: "y",
        )
        self.assertFalse(result)

    def test_collect_assignee_force_returns_true(self):
        result = self.workflow._collect_assignee_decision(
            make_item(1, "T"),
            closing_pr=ClosingPR(pr_number=42, author_login="dev"),
            force=True,
            input_fn=lambda _: "n",
        )
        self.assertTrue(result)

    def test_collect_assignee_y_returns_true(self):
        result = self.workflow._collect_assignee_decision(
            make_item(1, "T"),
            closing_pr=ClosingPR(pr_number=42, author_login="dev"),
            force=False,
            input_fn=lambda _: "y",
        )
        self.assertTrue(result)

    def test_collect_assignee_n_returns_false(self):
        result = self.workflow._collect_assignee_decision(
            make_item(1, "T"),
            closing_pr=ClosingPR(pr_number=42, author_login="dev"),
            force=False,
            input_fn=lambda _: "n",
        )
        self.assertFalse(result)

    def test_collect_assignee_skips_when_already_assigned(self):
        item = make_item(1, "T", assignees=["dev"])
        result = self.workflow._collect_assignee_decision(
            item,
            closing_pr=ClosingPR(pr_number=42, author_login="dev"),
            force=False,
            input_fn=lambda _: (_ for _ in ()).throw(
                AssertionError("prompt must not fire")
            ),
        )
        self.assertFalse(result)

    def test_collect_assignee_skips_case_insensitive(self):
        item = make_item(1, "T", assignees=["Dev"])
        result = self.workflow._collect_assignee_decision(
            item,
            closing_pr=ClosingPR(pr_number=42, author_login="dev"),
            force=True,
            input_fn=lambda _: "y",
        )
        self.assertFalse(result)

    def test_apply_sets_assignee_when_enabled_and_accepted(self):
        suggestion_results = [self._make_result()]
        with mock.patch.object(self.workflow, "print_summary"), \
                mock.patch("ai_labelling.workflow.print_changes_summary"), \
                mock.patch.object(
                    self.workflow, "_find_closing_pr",
                    return_value=ClosingPR(pr_number=42, author_login="dev")
                ), \
                mock.patch.object(
                    self.workflow, "set_assignees_with_retry"
                ) as assign_mock:
            self.workflow.review_and_apply_suggestions(
                "owner/repo",
                suggestion_results,
                False,
                False,
                assign_issue_to_solver=True,
                input_fn=lambda _: "y",
            )
        assign_mock.assert_called_once()
        self.assertIn("dev", assign_mock.call_args.args[2])

    def test_apply_skips_assignee_when_user_declines(self):
        suggestion_results = [self._make_result()]
        with mock.patch.object(self.workflow, "print_summary"), \
                mock.patch("ai_labelling.workflow.print_changes_summary"), \
                mock.patch.object(
                    self.workflow, "_find_closing_pr",
                    return_value=ClosingPR(pr_number=42, author_login="dev")
                ), \
                mock.patch.object(
                    self.workflow, "set_assignees_with_retry"
                ) as assign_mock:
            self.workflow.review_and_apply_suggestions(
                "owner/repo",
                suggestion_results,
                False,
                False,
                assign_issue_to_solver=True,
                input_fn=lambda _: "n",
            )
        assign_mock.assert_not_called()

    def test_apply_no_assignee_when_flag_off(self):
        suggestion_results = [self._make_result()]
        with mock.patch.object(self.workflow, "print_summary"), \
                mock.patch("ai_labelling.workflow.print_changes_summary"), \
                mock.patch.object(
                    self.workflow.github_client, "get_closing_pr"
                ) as pr_mock, \
                mock.patch.object(
                    self.workflow, "set_assignees_with_retry"
                ) as assign_mock:
            self.workflow.review_and_apply_suggestions(
                "owner/repo", suggestion_results, True, False,
            )
        pr_mock.assert_not_called()
        assign_mock.assert_not_called()
