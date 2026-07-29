"""Workflow coordinator for label suggestion runs."""
# pylint: disable=too-many-lines

import argparse
import concurrent.futures
import os
import sys
import time as time_module
from typing import Callable, List, Optional, Sequence

from ai_labelling.args import default_cutoff, parse_model_spec
from ai_labelling.backends import get_backend_for_provider
from ai_labelling.comment import format_comment_body
from ai_labelling.github_client import parse_closing_issues
from ai_labelling.config import (
    DEFAULT_DATE_CUTOFF,
    FORCE_WARNING_DELAY_SECONDS,
)
from ai_labelling.formatting import (
    format_label_block,
    format_reason,
    print_changes_summary,
    print_exception_diagnostics,
    print_item_details,
)
from ai_labelling.github_client import GitHubClient
from ai_labelling.interaction import prompt_confirmation, prompt_yes_no
from ai_labelling.models import (
    AppliedChanges,
    ClosingPR,
    InputFn,
    IssueTypeDefinition,
    IssueTypeMap,
    LabelDefinition,
    LabelSuggestion,
    SearchOptions,
    SuggestionResult,
    UserQuit,
    WorkItem,
    parse_github_timestamp,
)
from ai_labelling.shell import get_script_version
from ai_labelling.terminal import colourise, get_debug_level

_Operation = Callable[[], object]
"""Zero-argument callable passed to ``_retry_until_success``."""


class LabellingWorkflow:
    """Stateful workflow coordinator for searching, AI suggestions, review."""

    def __init__(self, github_client: Optional[GitHubClient] = None):
        self.github_client = github_client or GitHubClient()

    # ---------------- AI suggestion generation ----------------

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def build_suggestion_result(
        self,
        item: WorkItem,
        valid_labels: Sequence[LabelDefinition],
        model: str,
        allow_label_removals: bool,
        valid_issue_types: Sequence[IssueTypeDefinition] = (),
    ) -> SuggestionResult:
        """Run the AI backend for one work item and normalise the result."""

        model_spec = parse_model_spec(model)
        backend = get_backend_for_provider(model_spec.provider)
        suggestion = backend.suggest_labels(
            item,
            valid_labels,
            model_spec,
            allow_label_removals,
            valid_issue_types=valid_issue_types,
        )
        model_display = (
            f"{model_spec.provider}:{model_spec.model}"
            if model_spec.model
            else model_spec.provider
        )
        return SuggestionResult(
            item=item,
            label_suggestion=LabelSuggestion.from_raw(
                suggestion,
                valid_labels,
                item.labels,
                valid_issue_types=valid_issue_types,
                current_issue_type=item.issue_type,
            ),
            model=model_display,
        )

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def build_suggestion_result_with_retry(
        self,
        item: WorkItem,
        valid_labels: Sequence[LabelDefinition],
        model: str,
        allow_label_removals: bool,
        input_fn: InputFn = input,
        valid_issue_types: Sequence[IssueTypeDefinition] = (),
    ) -> Optional[SuggestionResult]:
        """Run one AI suggestion, optionally retrying after failures."""

        return _retry_until_success(
            lambda: self.build_suggestion_result(
                item,
                valid_labels,
                model,
                allow_label_removals,
                valid_issue_types=valid_issue_types,
            ),
            context=f"AI suggestion for {item.kind.upper()} #{item.number}",
            retry_prompt=(
                "Retry AI suggestion generation for "
                f"{item.kind.upper()} #{item.number}? "
            ),
            default_yes=False,
            input_fn=input_fn,
        )

    # ---------------- Item-write retries ----------------

    def add_labels_with_retry(
        self,
        repo: str,
        item: WorkItem,
        labels: Sequence[str],
        input_fn: InputFn = input,
    ) -> None:
        """Apply labels, offering retries when the write step fails."""

        _retry_until_success(
            lambda: self.github_client.add_labels(repo, item, labels),
            context=f"Applying labels to {item.kind.upper()} #{item.number}",
            retry_prompt=(
                f"Retry applying labels for {item.kind.upper()} "
                f"#{item.number}? "
            ),
            default_yes=True,
            input_fn=input_fn,
        )

    def remove_label_with_retry(
        self,
        repo: str,
        item: WorkItem,
        label: str,
        input_fn: InputFn = input,
    ) -> None:
        """Remove one label, offering retries when the write step fails."""

        _retry_until_success(
            lambda: self.github_client.remove_label(repo, item, label),
            context=(
                f"Removing label {label!r} from "
                f"{item.kind.upper()} #{item.number}"
            ),
            retry_prompt=(
                f"Retry removing label {label!r} from "
                f"{item.kind.upper()} #{item.number}? "
            ),
            default_yes=True,
            input_fn=input_fn,
        )

    def set_issue_type_with_retry(
        self,
        repo: str,
        item: WorkItem,
        issue_type: IssueTypeDefinition,
        input_fn: InputFn = input,
    ) -> None:
        """Set the issue type, offering retries when the write step fails."""

        _retry_until_success(
            lambda: self.github_client.set_issue_type(repo, item, issue_type),
            context=(
                f"Setting issue type {issue_type.name!r} on #{item.number}"
            ),
            retry_prompt=(
                f"Retry setting issue type {issue_type.name!r} "
                f"on #{item.number}? "
            ),
            default_yes=True,
            input_fn=input_fn,
        )

    def set_assignees_with_retry(
        self,
        repo: str,
        item: WorkItem,
        assignees: List[str],
        input_fn: InputFn = input,
    ) -> None:
        """Replace assignees on an issue or PR, offering retries on failure."""

        kind_display = item.kind.upper()
        _retry_until_success(
            lambda: self.github_client.set_assignees(repo, item, assignees),
            context=f"Setting assignees on {kind_display} #{item.number}",
            retry_prompt=(
                f"Retry setting assignees on {kind_display} #{item.number}? "
            ),
            default_yes=True,
            input_fn=input_fn,
        )

    # ---------------- High-level orchestration ----------------

    @staticmethod
    def warn_force_mode(
        dry_run: bool,
        delay_seconds: int = FORCE_WARNING_DELAY_SECONDS,
    ) -> None:
        """Warn loudly before running in fully automatic force mode."""

        suffix = (
            "and apply every suggested label automatically."
            if not dry_run
            else "."
        )
        print(
            colourise(
                "SUPER DANGEROUS: --force will send every matching item to "
                "the AI" + suffix,
                "red",
                bold=True,
            )
        )
        print(
            colourise(
                "This may EXHAUST AI QUOTAS suddenly. Press Ctrl-C now and "
                "rerun without '--force' if you want the normal reviewed "
                "flow.",
                "red",
                bold=True,
            )
        )
        print(
            colourise(
                f"Waiting {delay_seconds} seconds before continuing...",
                "yellow",
                bold=True,
            )
        )
        time_module.sleep(delay_seconds)
        print()

    @staticmethod
    def select_items_to_handle(
        items: Sequence[WorkItem],
        force: bool,
        input_fn: InputFn = input,
    ) -> List[WorkItem]:
        """Select which items should be sent to the AI backend."""

        if force:
            return list(items)

        selected_items: List[WorkItem] = []
        for index, item in enumerate(items):
            print_item_details(item)
            answer = prompt_confirmation(
                (
                    f"Handle {item.kind.upper()} #{item.number} with AI? "
                    "[y/n/a/d/q/?] "
                ),
                allow_apply_all=False,
                input_fn=input_fn,
            )
            if answer == "N":
                print(
                    colourise(
                        f"Skipping {item.kind.upper()} #{item.number}.",
                        "magenta",
                        bold=True,
                    )
                )
                print()
                continue
            if answer == "D":
                print(
                    colourise(
                        "Stopping before remaining items.",
                        "yellow",
                        bold=True,
                    )
                )
                break
            if answer == "A":
                selected_items.append(item)
                selected_items.extend(items[index + 1:])
                break
            selected_items.append(item)
        return selected_items

    def run_ai_batch(
        self,
        items: Sequence[WorkItem],
        valid_labels: Sequence[LabelDefinition],
        model: str,
        allow_label_removals: bool,
        input_fn: InputFn = input,
        valid_issue_types: Sequence[IssueTypeDefinition] = (),
    ) -> List[SuggestionResult]:
        """Run AI suggestions for selected items in parallel."""

        if not items:
            return []

        worker_count = min(len(items), os.cpu_count() or 1)
        print(
            colourise("Running processes: ", "blue", bold=True)
            + str(worker_count)
        )
        print(
            colourise("AI processing items: ", "blue", bold=True)
            + str(len(items))
        )
        print()

        if worker_count == 1:
            return self._run_ai_serially(
                items,
                valid_labels,
                model,
                allow_label_removals,
                input_fn,
                valid_issue_types,
            )
        return self._run_ai_parallel(
            items,
            valid_labels,
            model,
            allow_label_removals,
            worker_count,
            input_fn,
            valid_issue_types,
        )

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def _run_ai_serially(
        self,
        items: Sequence[WorkItem],
        valid_labels: Sequence[LabelDefinition],
        model: str,
        allow_label_removals: bool,
        input_fn: InputFn,
        valid_issue_types: Sequence[IssueTypeDefinition],
    ) -> List[SuggestionResult]:
        """Run AI suggestions sequentially for the single-worker case."""

        results: List[SuggestionResult] = []
        for item in items:
            result = self.build_suggestion_result_with_retry(
                item,
                valid_labels,
                model,
                allow_label_removals,
                input_fn=input_fn,
                valid_issue_types=valid_issue_types,
            )
            if result is not None:
                results.append(result)
        return results

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals
    def _run_ai_parallel(
        self,
        items: Sequence[WorkItem],
        valid_labels: Sequence[LabelDefinition],
        model: str,
        allow_label_removals: bool,
        worker_count: int,
        input_fn: InputFn,
        valid_issue_types: Sequence[IssueTypeDefinition],
    ) -> List[SuggestionResult]:
        """Run AI suggestions across a process pool, retrying on failure."""

        results: List[Optional[SuggestionResult]] = [None] * len(items)
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=worker_count
        ) as executor:
            future_to_index = {
                executor.submit(
                    self.build_suggestion_result,
                    item,
                    list(valid_labels),
                    model,
                    allow_label_removals,
                    list(valid_issue_types),
                ): index
                for index, item in enumerate(items)
            }
            for future in concurrent.futures.as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                # pylint: disable-next=broad-exception-caught
                except Exception as exc:
                    print_exception_diagnostics(
                        exc,
                        (
                            "AI suggestion for "
                            f"{items[index].kind.upper()} "
                            f"#{items[index].number}"
                        ),
                    )
                    if prompt_yes_no(
                        (
                            "Retry AI suggestion generation for "
                            f"{items[index].kind.upper()} "
                            f"#{items[index].number}? "
                        ),
                        default_yes=False,
                        input_fn=input_fn,
                    ):
                        results[index] = (
                            self.build_suggestion_result_with_retry(
                                items[index],
                                valid_labels,
                                model,
                                allow_label_removals,
                                input_fn=input_fn,
                                valid_issue_types=valid_issue_types,
                            )
                        )
        return [result for result in results if result is not None]

    @staticmethod
    def print_summary(
        item: WorkItem,
        label_suggestion: LabelSuggestion,
    ) -> None:
        """Print the suggested label changes for one work item."""

        print_item_details(item)
        if item.issue_type:
            print(
                colourise("Issue type:", "blue", bold=True)
                + f" {item.issue_type}"
            )
        if label_suggestion.issue_type is not None:
            print(
                colourise("Suggested issue type:", "green", bold=True)
                + f" {label_suggestion.issue_type}"
            )
        print(colourise("Existing labels:", "blue", bold=True))
        print(format_label_block(item.labels))
        if label_suggestion.add_labels:
            print(colourise("Suggested additions:", "green", bold=True))
            print(format_label_block(label_suggestion.add_labels))
        if label_suggestion.remove_labels:
            print(colourise("Suggested removals:", "magenta", bold=True))
            print(format_label_block(label_suggestion.remove_labels))
        if label_suggestion.reason:
            print(colourise("Reason:", "blue", bold=True))
            print(format_reason(label_suggestion.reason))
        print()

    # ---------------- Review and apply ----------------

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def review_and_apply_suggestions(
        self,
        repo: str,
        suggestion_results: Sequence[SuggestionResult],
        force: bool,
        allow_label_removals: bool,
        input_fn: InputFn = input,
        *,
        dry_run: bool = False,
        comment_reason: bool = False,
        valid_issue_types: Sequence[IssueTypeDefinition] = (),
        assign_issue_to_solver: bool = False,
        unassign_pr_if_solving_issue: bool = False,
        assign_pr_if_not_solving_issue: bool = False,
    ) -> None:
        """Review AI suggestions and optionally apply label changes."""

        issue_type_map: IssueTypeMap = {
            t.name.casefold(): t for t in valid_issue_types
        }
        version = get_script_version() if comment_reason else ""
        summary_results: List[SuggestionResult] = []

        for result in suggestion_results:
            self.print_summary(result.item, result.label_suggestion)
            if dry_run:
                summary_results.append(result)
                continue

            applied = self._apply_one_result(
                repo,
                result,
                force=force,
                allow_label_removals=allow_label_removals,
                issue_type_map=issue_type_map,
                input_fn=input_fn,
                assign_issue_to_solver=assign_issue_to_solver,
                unassign_pr_if_solving_issue=unassign_pr_if_solving_issue,
                assign_pr_if_not_solving_issue=assign_pr_if_not_solving_issue,
            )
            if applied.has_any_changes():
                summary_results.append(
                    SuggestionResult(
                        item=result.item,
                        label_suggestion=LabelSuggestion(
                            applied.added_labels,
                            applied.removed_labels,
                            result.label_suggestion.reason,
                            issue_type=applied.issue_type,
                        ),
                        model=result.model,
                        applied_assignee=applied.assignee,
                        pr_unassigned=applied.pr_unassigned,
                        pr_assigned_author=applied.pr_assigned_author,
                    )
                )
            if comment_reason:
                self._post_comment_for_result(
                    repo,
                    result,
                    applied=applied,
                    version=version,
                    allow_label_removals=allow_label_removals,
                )

        print_changes_summary(
            summary_results, allow_label_removals, dry_run=dry_run
        )

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-branches
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def run_interactive_mode(
        self,
        repo: str,
        valid_labels: Sequence[LabelDefinition],
        model: str,
        allow_label_removals: bool,
        *,
        dry_run: bool = False,
        comment_reason: bool = False,
        valid_issue_types: Sequence[IssueTypeDefinition] = (),
        assign_issue_to_solver: bool = False,
        unassign_pr_if_solving_issue: bool = False,
        assign_pr_if_not_solving_issue: bool = False,
        input_fn: InputFn = input,
    ) -> None:
        """Interactively handle items by number until the user quits.

        Labels are queried once; the user enters one issue/PR number per
        iteration. All filter and '--id' options are ignored.
        """

        issue_type_map: IssueTypeMap = {
            t.name.casefold(): t for t in valid_issue_types
        }
        version = get_script_version() if comment_reason else ""
        summary_results: List[SuggestionResult] = []

        while True:
            try:
                raw = input_fn(
                    colourise(
                        "Issue/PR number ('q' to exit): ",
                        "cyan",
                        bold=True,
                    )
                ).strip()
            except EOFError:
                break
            if not raw:
                continue
            if raw.casefold() in {"q", "quit"}:
                break
            try:
                number = int(raw)
                if number <= 0:
                    raise ValueError("not positive")
            except ValueError:
                print(
                    colourise(f"Invalid number: {raw!r}", "red", bold=True)
                )
                continue

            try:
                item = self.github_client.get_item(repo, number)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                print_exception_diagnostics(exc, f"Fetching #{number}")
                continue

            print_item_details(item)
            try:
                answer = prompt_confirmation(
                    f"Handle {item.kind.upper()} #{item.number} with AI? "
                    "[y/n/q/?] ",
                    allow_apply_all=False,
                    input_fn=input_fn,
                )
            except UserQuit:
                break
            if answer == "N":
                print(
                    colourise(
                        f"Skipping {item.kind.upper()} #{item.number}.",
                        "magenta",
                        bold=True,
                    )
                )
                print()
                continue

            result = self.build_suggestion_result_with_retry(
                item,
                valid_labels,
                model,
                allow_label_removals,
                input_fn=input_fn,
                valid_issue_types=valid_issue_types,
            )
            if result is None:
                continue

            self.print_summary(result.item, result.label_suggestion)
            if dry_run:
                summary_results.append(result)
                continue

            try:
                applied = self._apply_one_result(
                    repo,
                    result,
                    force=False,
                    allow_label_removals=allow_label_removals,
                    issue_type_map=issue_type_map,
                    input_fn=input_fn,
                    assign_issue_to_solver=assign_issue_to_solver,
                    unassign_pr_if_solving_issue=unassign_pr_if_solving_issue,
                    assign_pr_if_not_solving_issue=(
                        assign_pr_if_not_solving_issue
                    ),
                )
            except UserQuit:
                break
            if applied.has_any_changes():
                summary_results.append(
                    SuggestionResult(
                        item=result.item,
                        label_suggestion=LabelSuggestion(
                            applied.added_labels,
                            applied.removed_labels,
                            result.label_suggestion.reason,
                            issue_type=applied.issue_type,
                        ),
                        model=result.model,
                        applied_assignee=applied.assignee,
                        pr_unassigned=applied.pr_unassigned,
                        pr_assigned_author=applied.pr_assigned_author,
                    )
                )
            if comment_reason:
                self._post_comment_for_result(
                    repo,
                    result,
                    applied=applied,
                    version=version,
                    allow_label_removals=allow_label_removals,
                )

        print_changes_summary(
            summary_results, allow_label_removals, dry_run=dry_run
        )

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals
    def _apply_one_result(
        self,
        repo: str,
        result: SuggestionResult,
        *,
        force: bool,
        allow_label_removals: bool,
        issue_type_map: IssueTypeMap,
        input_fn: InputFn,
        assign_issue_to_solver: bool = False,
        unassign_pr_if_solving_issue: bool = False,
        assign_pr_if_not_solving_issue: bool = False,
    ) -> AppliedChanges:
        """Collect and apply all decisions for one result.

        All decisions are collected up-front before any GitHub API call fires.
        A ``q`` at any prompt aborts the whole item with no partial changes.
        """

        closing_pr: Optional[ClosingPR] = (
            self._find_closing_pr(repo, result.item)
            if assign_issue_to_solver
            else None
        )
        accepted_assignee = self._collect_assignee_decision(
            result.item,
            closing_pr=closing_pr,
            force=force,
            input_fn=input_fn,
        )

        pr_closing_issues = (
            parse_closing_issues(result.item.body)
            if (
                unassign_pr_if_solving_issue or assign_pr_if_not_solving_issue
            ) and result.item.kind == "pr"
            else []
        )
        accepted_pr_unassign = self._collect_pr_unassign_decision(
            result.item,
            closing_issues=pr_closing_issues,
            force=force,
            input_fn=input_fn,
        )
        accepted_pr_assign = self._collect_pr_assign_decision(
            result.item,
            has_closing_issues=bool(pr_closing_issues),
            enabled=assign_pr_if_not_solving_issue,
            force=force,
            input_fn=input_fn,
        )

        accepted_type = self._collect_issue_type_decision(
            result,
            issue_type_map=issue_type_map,
            force=force,
            input_fn=input_fn,
        )
        accepted_adds = self._collect_label_decisions(
            result.item,
            result.label_suggestion.add_labels,
            removal=False,
            force=force,
            input_fn=input_fn,
        )
        accepted_removes: List[str] = []
        if allow_label_removals:
            accepted_removes = self._collect_label_decisions(
                result.item,
                result.label_suggestion.remove_labels,
                removal=True,
                force=force,
                input_fn=input_fn,
            )

        applied_assignee: Optional[str] = None
        if accepted_assignee and closing_pr is not None:
            self.set_assignees_with_retry(
                repo, result.item,
                [closing_pr.author_login],
                input_fn=input_fn,
            )
            applied_assignee = closing_pr.author_login

        applied_pr_unassigned = False
        if accepted_pr_unassign:
            self.set_assignees_with_retry(
                repo, result.item, [], input_fn=input_fn,
            )
            applied_pr_unassigned = True

        applied_pr_assigned_author: Optional[str] = None
        if accepted_pr_assign:
            self.set_assignees_with_retry(
                repo, result.item,
                [result.item.author_login],
                input_fn=input_fn,
            )
            applied_pr_assigned_author = result.item.author_login

        applied_issue_type: Optional[str] = None
        if accepted_type is not None:
            self.set_issue_type_with_retry(
                repo, result.item, accepted_type, input_fn=input_fn,
            )
            applied_issue_type = accepted_type.name

        if accepted_adds:
            self.add_labels_with_retry(
                repo, result.item, accepted_adds, input_fn=input_fn,
            )

        applied_removes: List[str] = []
        for label in accepted_removes:
            self.remove_label_with_retry(
                repo, result.item, label, input_fn=input_fn,
            )
            applied_removes.append(label)

        return AppliedChanges(
            added_labels=list(accepted_adds),
            removed_labels=applied_removes,
            issue_type=applied_issue_type,
            closing_pr=closing_pr,
            assignee=applied_assignee,
            pr_unassigned=applied_pr_unassigned,
            pr_assigned_author=applied_pr_assigned_author,
        )

    def _find_closing_pr(
        self,
        repo: str,
        item: WorkItem,
    ) -> Optional[ClosingPR]:
        """Return the PR that closed this issue, or ``None``."""

        if item.kind != "issue" or item.state.casefold() != "closed":
            return None
        try:
            return self.github_client.get_closing_pr(repo, item.number)
        except Exception:  # pylint: disable=broad-exception-caught
            return None

    def _collect_assignee_decision(
        self,
        item: WorkItem,
        *,
        closing_pr: Optional[ClosingPR],
        force: bool,
        input_fn: InputFn,
    ) -> bool:
        """Prompt for an assignee change; return True if accepted."""

        if closing_pr is None:
            return False
        existing = {a.casefold() for a in item.assignees}
        if closing_pr.author_login.casefold() in existing:
            return False
        if force:
            return True
        answer = prompt_confirmation(
            f"Set assignee of ISSUE #{item.number} to "
            f"@{closing_pr.author_login} "
            f"(author of PR #{closing_pr.pr_number} that closed it)? "
            "[y/n/q/?] ",
            allow_apply_all=False,
            input_fn=input_fn,
        )
        return answer != "N"

    def _collect_pr_unassign_decision(
        self,
        item: WorkItem,
        *,
        closing_issues: List[int],
        force: bool,
        input_fn: InputFn,
    ) -> bool:
        """Prompt to unassign PR closing issues; return True if accepted."""

        if item.kind != "pr" or not item.assignees or not closing_issues:
            return False
        if force:
            return True
        issue_word = "issue" if len(closing_issues) == 1 else "issues"
        issue_nums = ", ".join(f"#{n}" for n in closing_issues)
        assignee_part = ", ".join(f"@{a}" for a in item.assignees)
        answer = prompt_confirmation(
            f"Unassign PR #{item.number} (closes {issue_word} {issue_nums}) "
            f"from {assignee_part}? [y/n/q/?] ",
            allow_apply_all=False,
            input_fn=input_fn,
        )
        return answer != "N"

    def _collect_pr_assign_decision(
        self,
        item: WorkItem,
        *,
        has_closing_issues: bool,
        enabled: bool,
        force: bool,
        input_fn: InputFn,
    ) -> bool:
        """Prompt to assign a PR to its author when it closes no issues."""

        if not enabled or item.kind != "pr" or has_closing_issues:
            return False
        if item.author_login.casefold() in {
            a.casefold() for a in item.assignees
        }:
            return False
        if force:
            return True
        answer = prompt_confirmation(
            f"Assign PR #{item.number} to @{item.author_login} "
            f"(author, PR closes no issue)? [y/n/q/?] ",
            allow_apply_all=False,
            input_fn=input_fn,
        )
        return answer != "N"

    def _collect_issue_type_decision(
        self,
        result: SuggestionResult,
        *,
        issue_type_map: IssueTypeMap,
        force: bool,
        input_fn: InputFn,
    ) -> Optional[IssueTypeDefinition]:
        """Prompt for the issue-type decision; return the type-def or None."""

        suggested = result.label_suggestion.issue_type
        if result.item.kind != "issue" or suggested is None:
            return None
        type_def = issue_type_map.get(suggested.casefold())
        if type_def is None:
            return None

        if force:
            return type_def
        answer = prompt_confirmation(
            (
                f'SET issue type to "{suggested}" '
                f'for issue #{result.item.number}? [y/n/q/?] '
            ),
            allow_apply_all=False,
            input_fn=input_fn,
        )
        return type_def if answer == "Y" else None

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def _collect_label_decisions(
        self,
        item: WorkItem,
        labels: Sequence[str],
        *,
        removal: bool,
        force: bool,
        input_fn: InputFn,
    ) -> List[str]:
        """Prompt user for each label; return the accepted-label list."""

        if not labels:
            return []

        accepted: List[str] = []
        accept_remaining = force
        kind_display = "issue" if item.kind == "issue" else "PR"
        prompt_template = (
            '**REMOVE** the label "{label}" from {kind} #{number}? '
            "[y/n/a/d/q/?] "
            if removal
            else 'ADD the label "{label}" to {kind} #{number}? '
            "[y/n/a/d/q/?] "
        )

        for label in labels:
            if accept_remaining:
                accepted.append(label)
                continue
            answer = prompt_confirmation(
                prompt_template.format(
                    label=label,
                    kind=kind_display,
                    number=item.number,
                ),
                allow_apply_all=True,
                input_fn=input_fn,
            )
            if answer == "A":
                accept_remaining = True
                accepted.append(label)
            elif answer == "Y":
                accepted.append(label)
            elif answer == "D":
                break
            # "N" → skip this label, continue
        return accepted

    def _post_comment_for_result(
        self,
        repo: str,
        result: SuggestionResult,
        *,
        applied: AppliedChanges,
        version: str,
        allow_label_removals: bool,
    ) -> None:
        """Render and post the per-item summary comment, swallowing errors."""

        body = format_comment_body(
            result.label_suggestion,
            applied.added_labels,
            applied.removed_labels,
            result.model,
            version,
            allow_label_removals,
            applied_issue_type=applied.issue_type,
            closing_pr=applied.closing_pr,
            applied_assignee=applied.assignee,
            pr_unassigned=applied.pr_unassigned,
            pr_assigned_author=applied.pr_assigned_author,
        )
        if get_debug_level() >= 2:
            print(
                f"# Comment for {result.item.kind.upper()} "
                f"#{result.item.number}:\n{body}",
                file=sys.stderr,
            )
        try:
            self.github_client.post_comment(repo, result.item, body)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            print_exception_diagnostics(
                exc,
                (
                    f"Posting comment on {result.item.kind.upper()} "
                    f"#{result.item.number}"
                ),
            )

    # ---------------- Item collection ----------------

    def collect_items(
        self,
        repo: str,
        args: argparse.Namespace,
    ) -> List[WorkItem]:
        """Collect and sort matching issues and pull requests."""

        if getattr(args, "id", None) is not None:
            return [
                self.github_client.get_item(repo, number)
                for number in args.id
            ]

        if not args.include_issues and not args.include_prs:
            return []
        if not args.include_open and not args.include_closed:
            return []

        cutoff = (
            default_cutoff()
            if args.date is DEFAULT_DATE_CUTOFF
            else args.date
        )

        search_options = SearchOptions(
            include_open=args.include_open,
            include_closed=args.include_closed,
            created=args.created,
            cutoff=cutoff,
            limit=args.limit,
        )
        items: List[WorkItem] = []

        if args.include_issues:
            items.extend(
                self.github_client.search_items(
                    repo, "issue", search_options,
                )
            )
        if args.include_prs:
            remaining = (
                None if args.limit is None
                else max(args.limit - len(items), 0)
            )
            if remaining is None or remaining > 0:
                pr_options = SearchOptions(
                    include_open=args.include_open,
                    include_closed=args.include_closed,
                    created=args.created,
                    cutoff=cutoff,
                    limit=remaining,
                )
                items.extend(
                    self.github_client.search_items(
                        repo, "pr", pr_options,
                    )
                )

        timestamp_field = "created_at" if args.created else "updated_at"
        items.sort(
            key=lambda item: parse_github_timestamp(
                getattr(item, timestamp_field)
            ),
            reverse=True,
        )
        if args.limit is not None:
            return items[: args.limit]
        return items


def _retry_until_success(
    operation: _Operation,
    *,
    context: str,
    retry_prompt: str,
    default_yes: bool,
    input_fn: InputFn,
):
    """Run ``operation`` and prompt for retry on failure until success/decline.

    Returns the operation's last result (None when the user declines a retry
    after a failure).
    """

    while True:
        try:
            return operation()
        except Exception as exc:  # pylint: disable=broad-exception-caught
            print_exception_diagnostics(exc, context)
            if not prompt_yes_no(
                retry_prompt,
                default_yes=default_yes,
                input_fn=input_fn,
            ):
                return None
