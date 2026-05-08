"""Public package facade for the AI-assisted labelling workflow."""
# pylint: disable=duplicate-code

import argparse
import os
import subprocess
import sys
from typing import List, Sequence

from ai_labelling.args import parse_args
from ai_labelling.github_auth import resolve_github_token
from ai_labelling.formatting import (
    print_match_summary,
    print_matching_items,
)
from ai_labelling.github_client import GitHubClient
from ai_labelling.models import (
    InputFn,
    IssueTypeDefinition,
    LabelDefinition,
    SuggestionResult,
    UserQuit,
    WorkItem,
)
from ai_labelling.terminal import colourise
from ai_labelling.workflow import LabellingWorkflow


_GITHUB_CLIENT = GitHubClient()
_WORKFLOW = LabellingWorkflow(_GITHUB_CLIENT)


def detect_repo() -> str:
    """Infer the active GitHub repository from known git remotes."""

    return _GITHUB_CLIENT.detect_repo()


def list_repo_labels(repo: str) -> List[LabelDefinition]:
    """Fetch, deduplicate, and sort repository labels with descriptions."""

    return _GITHUB_CLIENT.list_repo_labels(repo)


def list_issue_types(repo: str) -> List[IssueTypeDefinition]:
    """Fetch issue types for the repository's organisation."""

    return _GITHUB_CLIENT.list_issue_types(repo)


def collect_items(repo: str, args: argparse.Namespace) -> List[WorkItem]:
    """Collect and sort matching issues and pull requests."""

    return _WORKFLOW.collect_items(repo, args)


def select_items_to_handle(
    items: Sequence[WorkItem],
    force: bool,
    input_fn: InputFn = input,
) -> List[WorkItem]:
    """Select which items should be sent to the AI backend."""

    return LabellingWorkflow.select_items_to_handle(
        items, force, input_fn=input_fn,
    )


# pylint: disable=too-many-arguments,too-many-positional-arguments
def run_ai_batch(
    items: Sequence[WorkItem],
    valid_labels: Sequence[LabelDefinition],
    model: str,
    allow_label_removals: bool,
    input_fn: InputFn = input,
    valid_issue_types: Sequence[IssueTypeDefinition] = (),
) -> List[SuggestionResult]:
    """Run AI suggestions for selected items in parallel."""

    return _WORKFLOW.run_ai_batch(
        items,
        valid_labels,
        model,
        allow_label_removals,
        input_fn=input_fn,
        valid_issue_types=valid_issue_types,
    )


# pylint: disable=too-many-arguments,too-many-positional-arguments
def review_and_apply_suggestions(
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
) -> None:
    """Review AI suggestions and optionally apply label changes."""

    _WORKFLOW.review_and_apply_suggestions(
        repo,
        suggestion_results,
        force,
        allow_label_removals,
        input_fn=input_fn,
        dry_run=dry_run,
        comment_reason=comment_reason,
        valid_issue_types=valid_issue_types,
        assign_issue_to_solver=assign_issue_to_solver,
    )


def main() -> int:
    """Run the end-to-end labelling workflow."""

    args = parse_args()
    repo = args.repo or detect_repo()

    token = resolve_github_token(repo)
    if token:
        os.environ["GH_TOKEN"] = token

    valid_labels = list_repo_labels(repo)
    if not valid_labels:
        raise RuntimeError(f"repository {repo} has no labels")

    valid_issue_types = list_issue_types(repo)

    if args.interactive:
        print(colourise("Repository: ", "blue", bold=True) + repo)
        print(
            colourise("Labels: ", "blue", bold=True) + str(len(valid_labels))
        )
        print()
        _WORKFLOW.run_interactive_mode(
            repo,
            valid_labels,
            args.model,
            args.allow_label_removals,
            dry_run=args.dry_run,
            comment_reason=args.comment_reason,
            valid_issue_types=valid_issue_types,
            assign_issue_to_solver=args.assign_issue_to_solver,
        )
        return 0

    items = collect_items(repo, args)
    if not items:
        print(
            colourise(
                "No matching issues or pull requests found.",
                "yellow",
                bold=True,
            )
        )
        return 0

    print(colourise("Repository: ", "blue", bold=True) + repo)
    print(colourise("Labels: ", "blue", bold=True) + str(len(valid_labels)))
    print()
    print_match_summary(items)
    print()
    print_matching_items(items, "Matching items")

    if args.force:
        LabellingWorkflow.warn_force_mode(args.dry_run)

    selected_items = select_items_to_handle(items, args.force)
    if not selected_items:
        print(
            colourise(
                "No items selected for AI handling.",
                "yellow",
                bold=True,
            )
        )
        return 0

    suggestion_results = run_ai_batch(
        selected_items,
        valid_labels,
        args.model,
        args.allow_label_removals,
        input_fn=input,
        valid_issue_types=valid_issue_types,
    )
    review_and_apply_suggestions(
        repo,
        suggestion_results,
        args.force,
        args.allow_label_removals,
        dry_run=args.dry_run,
        comment_reason=args.comment_reason,
        valid_issue_types=valid_issue_types,
        assign_issue_to_solver=args.assign_issue_to_solver,
    )
    return 0


def cli_main() -> None:
    """Console script entry point with exception handling."""

    try:
        raise SystemExit(main())
    except UserQuit:
        print(
            colourise(
                "Quit requested. Terminating immediately.",
                "yellow",
                bold=True,
            )
        )
        raise SystemExit(0) from None
    except subprocess.CalledProcessError as exc:
        if exc.stdout:
            print(
                exc.stdout,
                file=sys.stderr,
                end="" if exc.stdout.endswith("\n") else "\n",
            )
        if exc.stderr:
            print(
                exc.stderr,
                file=sys.stderr,
                end="" if exc.stderr.endswith("\n") else "\n",
            )
        raise SystemExit(exc.returncode) from exc
    except RuntimeError as exc:
        print(
            colourise(str(exc), "red", stream=sys.stderr, bold=True),
            file=sys.stderr,
        )
        raise SystemExit(1) from exc


__all__ = [
    "GitHubClient",
    "cli_main",
    "LabelDefinition",
    "LabellingWorkflow",
    "UserQuit",
    "WorkItem",
    "collect_items",
    "colourise",
    "detect_repo",
    "list_issue_types",
    "list_repo_labels",
    "main",
    "parse_args",
    "print_match_summary",
    "print_matching_items",
    "review_and_apply_suggestions",
    "run_ai_batch",
    "select_items_to_handle",
]
