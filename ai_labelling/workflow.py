"""Workflow and interaction logic for label suggestion runs."""

import argparse
import concurrent.futures
import os
import sys
import time as time_module
from typing import Callable, Dict, List, Optional, Sequence

from ai_labelling.args import default_cutoff, parse_model_spec
from ai_labelling.backends import get_backend_for_provider
from ai_labelling.config import (
    DEFAULT_DATE_CUTOFF,
    FORCE_WARNING_DELAY_SECONDS,
)
from ai_labelling.formatting import (
    format_comment_body,
    format_label_block,
    format_reason,
    print_changes_summary,
    print_exception_diagnostics,
    print_item_details,
    print_prompt_help,
)
from ai_labelling.shell import run as _shell_run
from ai_labelling.github_client import GitHubClient, parse_github_timestamp
from ai_labelling.models import (
    IssueTypeDefinition,
    LabelDefinition,
    LabelSuggestion,
    SearchOptions,
    SuggestionResult,
    UserQuit,
    WorkItem,
)
from ai_labelling.terminal import colourise, get_debug_level


def prompt_confirmation(
    prompt: str,
    *,
    allow_apply_all: bool,
    input_fn: Callable[[str], str] = input,
) -> str:
    """Prompt until the user answers with a supported git-style choice."""

    while True:
        answer = input_fn(colourise(prompt, "yellow", bold=True)).strip()
        if not answer:
            continue
        normalised = answer.upper()
        if normalised == "?":
            print_prompt_help(allow_apply_all)
            continue
        if normalised == "Q":
            raise UserQuit
        if normalised in {"Y", "N", "A", "D"}:
            return normalised


def prompt_yes_no(
    prompt: str,
    *,
    default_yes: bool,
    input_fn: Callable[[str], str] = input,
) -> bool:
    """Prompt for a yes/no retry decision with an optional default answer."""

    suffix = "[Y/n] " if default_yes else "[y/N] "
    while True:
        answer = input_fn(
            colourise(prompt + suffix, "yellow", bold=True)
        ).strip()
        if not answer:
            return default_yes
        normalised = answer.casefold()
        if normalised in {"y", "yes"}:
            return True
        if normalised in {"n", "no"}:
            return False


def normalise_label_list(
    value: object,
    valid_labels: Dict[str, str],
) -> List[str]:
    """Normalise a raw label list to unique canonical repository labels."""

    if not isinstance(value, list):
        return []

    labels: List[str] = []
    seen = set()
    for entry in value:
        if not isinstance(entry, str):
            continue
        canonical = valid_labels.get(entry.casefold())
        if canonical is None or canonical.casefold() in seen:
            continue
        seen.add(canonical.casefold())
        labels.append(canonical)
    return sorted(labels, key=str.casefold)


def _get_script_version() -> str:
    """Return a short git SHA for the script repository, or 'unknown'."""

    try:
        result = _shell_run(
            (
                "git",
                "-C",
                os.path.dirname(__file__),
                "rev-parse",
                "--short",
                "HEAD",
            ),
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:  # pylint: disable=broad-exception-caught
        pass
    return "unknown"


# pylint: disable=too-many-locals
def normalise_label_suggestions(
    suggestion: Dict[str, object],
    valid_labels: Sequence[LabelDefinition],
    existing_labels: Sequence[str],
    *,
    valid_issue_types: Sequence[IssueTypeDefinition] = (),
    current_issue_type: Optional[str] = None,
) -> LabelSuggestion:
    """Filter model output down to valid label additions and removals."""

    valid = {
        label.name.casefold(): label.name for label in valid_labels
    }
    existing = {label.casefold(): label for label in existing_labels}

    raw_add = suggestion.get("add_labels", [])
    raw_remove = suggestion.get("remove_labels", [])
    reason = str(suggestion.get("reason", "")).strip()

    labels_to_add = normalise_label_list(raw_add, valid)
    labels_to_remove = normalise_label_list(raw_remove, valid)

    labels_to_add = [
        label for label in labels_to_add if label.casefold() not in existing
    ]
    labels_to_remove = [
        label for label in labels_to_remove if label.casefold() in existing
    ]

    issue_type: Optional[str] = None
    if valid_issue_types:
        raw_type = suggestion.get("issue_type")
        if isinstance(raw_type, str) and raw_type.strip():
            valid_type_map = {
                t.name.casefold(): t.name for t in valid_issue_types
            }
            canonical = valid_type_map.get(raw_type.strip().casefold())
            current_cf = (current_issue_type or "").casefold()
            if canonical and canonical.casefold() != current_cf:
                issue_type = canonical

    return LabelSuggestion(
        labels_to_add, labels_to_remove, reason, issue_type=issue_type
    )


class LabellingWorkflow:
    """Stateful workflow coordinator for searching, AI suggestions, review."""

    def __init__(self, github_client: Optional[GitHubClient] = None):
        self.github_client = github_client or GitHubClient()

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
            label_suggestion=normalise_label_suggestions(
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
        input_fn: Callable[[str], str] = input,
        valid_issue_types: Sequence[IssueTypeDefinition] = (),
    ) -> Optional[SuggestionResult]:
        """Run one AI suggestion, optionally retrying after failures."""

        while True:
            try:
                return self.build_suggestion_result(
                    item,
                    valid_labels,
                    model,
                    allow_label_removals,
                    valid_issue_types=valid_issue_types,
                )
            except Exception as exc:  # pylint: disable=broad-exception-caught
                print_exception_diagnostics(
                    exc,
                    f"AI suggestion for {item.kind.upper()} #{item.number}",
                )
                should_retry = prompt_yes_no(
                    (
                        "Retry AI suggestion generation for "
                        f"{item.kind.upper()} #{item.number}? "
                    ),
                    default_yes=False,
                    input_fn=input_fn,
                )
                if not should_retry:
                    return None

    def add_labels_with_retry(
        self,
        repo: str,
        item: WorkItem,
        labels: Sequence[str],
        input_fn: Callable[[str], str] = input,
    ) -> None:
        """Apply labels, offering retries when the write step fails."""

        while True:
            try:
                self.github_client.add_labels(repo, item, labels)
                return
            except Exception as exc:  # pylint: disable=broad-exception-caught
                print_exception_diagnostics(
                    exc,
                    f"Applying labels to {item.kind.upper()} #{item.number}",
                )
                should_retry = prompt_yes_no(
                    (
                        f"Retry applying labels for {item.kind.upper()} "
                        f"#{item.number}? "
                    ),
                    default_yes=True,
                    input_fn=input_fn,
                )
                if not should_retry:
                    return

    def remove_label_with_retry(
        self,
        repo: str,
        item: WorkItem,
        label: str,
        input_fn: Callable[[str], str] = input,
    ) -> None:
        """Remove one label, offering retries when the write step fails."""

        while True:
            try:
                self.github_client.remove_label(repo, item, label)
                return
            except Exception as exc:  # pylint: disable=broad-exception-caught
                print_exception_diagnostics(
                    exc,
                    (
                        f"Removing label {label!r} from "
                        f"{item.kind.upper()} #{item.number}"
                    ),
                )
                should_retry = prompt_yes_no(
                    (
                        f"Retry removing label {label!r} from "
                        f"{item.kind.upper()} #{item.number}? "
                    ),
                    default_yes=True,
                    input_fn=input_fn,
                )
                if not should_retry:
                    return

    def set_issue_type_with_retry(
        self,
        repo: str,
        item: WorkItem,
        issue_type: IssueTypeDefinition,
        input_fn: Callable[[str], str] = input,
    ) -> None:
        """Set the issue type, offering retries when the write step fails."""

        while True:
            try:
                self.github_client.set_issue_type(repo, item, issue_type)
                return
            except Exception as exc:  # pylint: disable=broad-exception-caught
                print_exception_diagnostics(
                    exc,
                    (
                        f"Setting issue type {issue_type.name!r} on "
                        f"#{item.number}"
                    ),
                )
                should_retry = prompt_yes_no(
                    (
                        f"Retry setting issue type {issue_type.name!r} "
                        f"on #{item.number}? "
                    ),
                    default_yes=True,
                    input_fn=input_fn,
                )
                if not should_retry:
                    return

    @staticmethod
    def warn_force_mode(
        dry_run: bool,
        delay_seconds: int = FORCE_WARNING_DELAY_SECONDS,
    ) -> None:
        """Warn loudly before running in fully automatic force mode."""

        print(
            colourise(
                "SUPER DANGEROUS: --force will send every matching item to "
                "the AI" +
                ("and apply every suggested label automatically."
                 if not dry_run else "."),
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
        input_fn: Callable[[str], str] = input,
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
        input_fn: Callable[[str], str] = input,
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
            return [
                result
                for item in items
                for result in [
                    self.build_suggestion_result_with_retry(
                        item,
                        valid_labels,
                        model,
                        allow_label_removals,
                        input_fn=input_fn,
                        valid_issue_types=valid_issue_types,
                    )
                ]
                if result is not None
            ]

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
                # pylint: disable=broad-exception-caught
                except Exception as exc:
                    print_exception_diagnostics(
                        exc,
                        (
                            "AI suggestion for "
                            f"{items[index].kind.upper()} "
                            f"#{items[index].number}"
                        ),
                    )
                    should_retry = prompt_yes_no(
                        (
                            "Retry AI suggestion generation for "
                            f"{items[index].kind.upper()} "
                            f"#{items[index].number}? "
                        ),
                        default_yes=False,
                        input_fn=input_fn,
                    )
                    if should_retry:
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

    # pylint: disable=too-many-arguments,too-many-statements
    # pylint: disable=too-many-locals,too-many-positional-arguments
    def review_and_apply_suggestions(
        self,
        repo: str,
        suggestion_results: Sequence[SuggestionResult],
        force: bool,
        allow_label_removals: bool,
        input_fn: Callable[[str], str] = input,
        *,
        dry_run: bool = False,
        comment_reason: bool = False,
        valid_issue_types: Sequence[IssueTypeDefinition] = (),
    ) -> None:
        """Review AI suggestions and optionally apply label changes."""

        issue_type_map = {t.name.casefold(): t for t in valid_issue_types}

        def apply_label_change_bucket(
            item: WorkItem,
            labels: Sequence[str],
            *,
            removal: bool,
        ) -> List[str]:
            if not labels:
                return []

            applied: List[str] = []
            apply_remaining = force
            skip_remaining = False
            kind_display = "issue" if item.kind == "issue" else "PR"
            prompt_template = (
                '**REMOVE** the label "{label}" from {kind} #{number}? '
                "[y/n/a/d/q/?] "
                if removal
                else 'ADD the label "{label}" to {kind} #{number}? '
                "[y/n/a/d/q/?] "
            )

            for label in labels:
                if skip_remaining:
                    break

                should_apply = apply_remaining
                if not should_apply:
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
                        apply_remaining = True
                        should_apply = True
                    elif answer == "Y":
                        should_apply = True
                    elif answer == "D":
                        skip_remaining = True
                        continue
                    elif answer == "N":
                        continue

                if should_apply:
                    if removal:
                        self.remove_label_with_retry(
                            repo,
                            item,
                            label,
                            input_fn=input_fn,
                        )
                    else:
                        self.add_labels_with_retry(
                            repo,
                            item,
                            [label],
                            input_fn=input_fn,
                        )
                    applied.append(label)

            return applied

        version = _get_script_version() if comment_reason else ""
        summary_results: List[SuggestionResult] = []
        for result in suggestion_results:
            self.print_summary(result.item, result.label_suggestion)
            if dry_run:
                summary_results.append(result)
            else:
                applied_issue_type: Optional[str] = None
                suggested_type = result.label_suggestion.issue_type
                if (
                    result.item.kind == "issue"
                    and suggested_type is not None
                ):
                    type_def = issue_type_map.get(
                        suggested_type.casefold()
                    )
                    if type_def is not None:
                        should_set = force
                        if not should_set:
                            answer = prompt_confirmation(
                                (
                                    f'SET issue type to '
                                    f'"{suggested_type}" '
                                    f'for issue '
                                    f'#{result.item.number}? '
                                    "[y/n/q/?] "
                                ),
                                allow_apply_all=False,
                                input_fn=input_fn,
                            )
                            should_set = answer == "Y"
                        if should_set:
                            self.set_issue_type_with_retry(
                                repo,
                                result.item,
                                type_def,
                                input_fn=input_fn,
                            )
                            applied_issue_type = suggested_type

                applied_add = apply_label_change_bucket(
                    result.item,
                    result.label_suggestion.add_labels,
                    removal=False,
                )
                applied_remove: List[str] = []
                if allow_label_removals:
                    applied_remove = apply_label_change_bucket(
                        result.item,
                        result.label_suggestion.remove_labels,
                        removal=True,
                    )
                if applied_add or applied_remove or applied_issue_type:
                    summary_results.append(
                        SuggestionResult(
                            item=result.item,
                            label_suggestion=LabelSuggestion(
                                applied_add,
                                applied_remove,
                                result.label_suggestion.reason,
                                issue_type=applied_issue_type,
                            ),
                            model=result.model,
                        )
                    )
                if comment_reason:
                    body = format_comment_body(
                        result.label_suggestion,
                        applied_add,
                        applied_remove,
                        result.model,
                        version,
                        allow_label_removals,
                        applied_issue_type=applied_issue_type,
                    )
                    if get_debug_level() >= 2:
                        print(
                            f"# Comment for "
                            f"{result.item.kind.upper()} "
                            f"#{result.item.number}:\n{body}",
                            file=sys.stderr,
                        )
                    try:
                        self.github_client.post_comment(
                            repo, result.item, body
                        )
                    # pylint: disable=broad-exception-caught
                    except Exception as exc:
                        print_exception_diagnostics(
                            exc,
                            f"Posting comment on "
                            f"{result.item.kind.upper()} "
                            f"#{result.item.number}",
                        )

        print_changes_summary(
            summary_results, allow_label_removals, dry_run=dry_run
        )

    def collect_items(
        self,
        repo: str,
        args: argparse.Namespace,
    ) -> List[WorkItem]:
        """Collect and sort matching issues and pull requests."""

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
                    repo,
                    "issue",
                    search_options,
                )
            )
        if args.include_prs:
            remaining = (
                None if args.limit is None else max(args.limit - len(items), 0)
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
                        repo,
                        "pr",
                        pr_options,
                    )
                )

        if args.created:
            items.sort(
                key=lambda item: parse_github_timestamp(item.created_at),
                reverse=True,
            )
        else:
            items.sort(
                key=lambda item: parse_github_timestamp(item.updated_at),
                reverse=True,
            )
        if args.limit is not None:
            return items[: args.limit]
        return items
