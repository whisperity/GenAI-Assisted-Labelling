"""Public package facade for the AI-assisted labelling workflow."""
# pylint: disable=duplicate-code

from ai_labelling.args import parse_args
from ai_labelling.formatting import (
    print_match_summary,
    print_matching_items,
)
from ai_labelling.github_client import GitHubClient
from ai_labelling.models import (
    LabelDefinition,
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


def list_repo_labels(repo: str):
    """Fetch, deduplicate, and sort repository labels with descriptions."""

    return _GITHUB_CLIENT.list_repo_labels(repo)


def list_issue_types(repo: str):
    """Fetch issue types for the repository's organisation."""

    return _GITHUB_CLIENT.list_issue_types(repo)


def collect_items(repo: str, args):
    """Collect and sort matching issues and pull requests."""

    return _WORKFLOW.collect_items(repo, args)


def select_items_to_handle(items, force, input_fn=input):
    """Select which items should be sent to the AI backend."""

    return LabellingWorkflow.select_items_to_handle(
        items, force, input_fn=input_fn,
    )


# pylint: disable=too-many-arguments,too-many-positional-arguments
def run_ai_batch(
    items,
    valid_labels,
    model,
    allow_label_removals,
    input_fn=input,
    valid_issue_types=(),
):
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
    repo,
    suggestion_results,
    force,
    allow_label_removals,
    input_fn=input,
    *,
    dry_run=False,
    comment_reason=False,
    valid_issue_types=(),
):
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
    )


def main() -> int:
    """Run the end-to-end labelling workflow."""

    args = parse_args()
    repo = args.repo or detect_repo()

    valid_labels = list_repo_labels(repo)
    if not valid_labels:
        raise RuntimeError(f"repository {repo} has no labels")

    valid_issue_types = list_issue_types(repo)

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
    )
    return 0


__all__ = [
    "GitHubClient",
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
