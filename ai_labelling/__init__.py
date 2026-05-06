"""Public package facade for the AI-assisted labelling workflow."""
# pylint: disable=duplicate-code

from typing import Callable, Optional, Sequence

from ai_labelling.args import (
    build_argument_parser,
    build_help_epilog,
    default_cutoff,
    format_help_epilog_entry,
    parse_args,
    parse_cutoff,
    parse_model_spec,
    parse_repo_arg,
    positive_int,
)
from ai_labelling.backends import (
    AIBackend,
    ANTHROPIC_BACKEND,
    BACKENDS,
    BACKENDS_BY_PROVIDER,
    CODEX_BACKEND,
    AnthropicBackend,
    CodexBackend,
    get_backend_for_model,
    get_provider_default_model,
    get_provider_help_description,
    get_provider_reasoning_levels,
    get_supported_providers,
)
from ai_labelling.config import (
    ANSI_RESET,
    ANSI_STYLES,
    DEFAULT_DATE_CUTOFF,
    DEFAULT_MODEL_SPEC,
    FORCE_WARNING_DELAY_SECONDS,
    REPO_DETECTION_ORDER,
)
from ai_labelling.formatting import (
    bucketise_items,
    classify_preview_block,
    colourise_inline_markdown,
    describe_match_bucket,
    format_body_preview,
    format_body_preview_colourised,
    format_comment_body,
    format_display_timestamp,
    format_label_block,
    format_reason,
    non_empty_lines,
    parse_github_timestamp,
    parse_preview_blocks,
    print_changes_summary,
    print_exception_diagnostics,
    print_item_details,
    print_match_summary,
    print_matching_items,
    print_prompt_help,
    summarise_body,
    take_non_empty_lines,
    wrap_preserving_newlines,
)
from ai_labelling.github_client import (
    GitHubClient,
    work_item_from_search_result,
)
from ai_labelling.models import (
    IssueTypeDefinition,
    LabelDefinition,
    LabelSuggestion,
    ModelSpec,
    PreviewBlock,
    SearchOptions,
    SuggestionResult,
    UserQuit,
    WorkItem,
)
from ai_labelling.shell import run
from ai_labelling.terminal import (
    colourise,
    debug_log,
    format_prompt_for_debug,
    get_debug_level,
    sanitise_prompt_for_debug,
    supports_colour,
)
from ai_labelling.workflow import (
    LabellingWorkflow,
    normalise_label_list,
    normalise_label_suggestions,
    prompt_confirmation,
    prompt_yes_no,
)


_GITHUB_CLIENT = GitHubClient()
_WORKFLOW = LabellingWorkflow(_GITHUB_CLIENT)


def _call_workflow(method_name: str, *args, **kwargs):
    """Call one method on the shared workflow instance."""

    return getattr(_WORKFLOW, method_name)(*args, **kwargs)


def parse_github_repo_from_remote(remote_url: str) -> Optional[str]:
    """Extract an ``owner/repository`` pair from a supported remote."""

    return GitHubClient.parse_repo_from_remote(remote_url)


def detect_repo() -> str:
    """Infer the active GitHub repository from known git remotes."""

    return _GITHUB_CLIENT.detect_repo()


def gh_json(argv: Sequence[str]) -> object:
    """Run ``gh`` and decode its stdout as JSON."""

    return _GITHUB_CLIENT.json(argv)


def list_issue_types(repo: str) -> list[IssueTypeDefinition]:
    """Fetch issue types for the repository's organisation."""

    return _GITHUB_CLIENT.list_issue_types(repo)


def list_repo_labels(repo: str) -> list[LabelDefinition]:
    """Fetch, deduplicate, and sort repository labels with descriptions."""

    return _GITHUB_CLIENT.list_repo_labels(repo)


def format_search_date(timestamp):
    """Format a timestamp for GitHub search date qualifiers."""

    return _GITHUB_CLIENT.format_search_date(timestamp)


def build_search_query(
    repo: str,
    kind: str,
    search_options: SearchOptions,
) -> str:
    """Build the GitHub search query for issues or pull requests."""

    return _GITHUB_CLIENT.build_search_query(repo, kind, search_options)


def search_items(
    repo: str,
    kind: str,
    search_options: SearchOptions,
) -> list[WorkItem]:
    """Search GitHub for matching issues or pull requests."""

    return _GITHUB_CLIENT.search_items(repo, kind, search_options)


def add_labels(repo: str, item: WorkItem, labels: Sequence[str]) -> None:
    """Apply new labels to a GitHub issue or pull request."""

    _GITHUB_CLIENT.add_labels(repo, item, labels)


def remove_label(repo: str, item: WorkItem, label: str) -> None:
    """Remove one label from a GitHub issue or pull request."""

    _GITHUB_CLIENT.remove_label(repo, item, label)


def anthropic_default_model() -> str:
    """Return the backend-selected default Anthropic model ID."""

    return ANTHROPIC_BACKEND.get_default_model()


def anthropic_extract_json(text: str) -> dict[str, object]:
    """Parse one JSON object from an Anthropic text response."""

    return ANTHROPIC_BACKEND.extract_json(text)


def build_suggestion_result(
    item: WorkItem,
    valid_labels: Sequence[LabelDefinition],
    model: str,
    allow_label_removals: bool,
) -> SuggestionResult:
    """Run the AI backend for one work item and normalise the result."""

    return _call_workflow(
        "build_suggestion_result",
        item,
        valid_labels,
        model,
        allow_label_removals,
    )


def build_suggestion_result_with_retry(
    item: WorkItem,
    valid_labels: Sequence[LabelDefinition],
    model: str,
    allow_label_removals: bool,
    input_fn: Callable[[str], str] = input,
) -> Optional[SuggestionResult]:
    """Run one AI suggestion, optionally retrying after failures."""

    return _call_workflow(
        "build_suggestion_result_with_retry",
        item=item,
        valid_labels=valid_labels,
        model=model,
        allow_label_removals=allow_label_removals,
        input_fn=input_fn,
    )


def add_labels_with_retry(
    repo: str,
    item: WorkItem,
    labels: Sequence[str],
    input_fn: Callable[[str], str] = input,
) -> None:
    """Apply labels, offering retries when the write step fails."""

    _call_workflow(
        "add_labels_with_retry",
        repo,
        item,
        labels,
        input_fn=input_fn,
    )


def remove_label_with_retry(
    repo: str,
    item: WorkItem,
    label: str,
    input_fn: Callable[[str], str] = input,
) -> None:
    """Remove one label, offering retries when the write step fails."""

    _call_workflow(
        "remove_label_with_retry",
        repo,
        item,
        label,
        input_fn=input_fn,
    )


def warn_force_mode(
    dry_run: bool,
    delay_seconds: int = FORCE_WARNING_DELAY_SECONDS
) -> None:
    """Warn loudly before running in fully automatic force mode."""

    LabellingWorkflow.warn_force_mode(dry_run, delay_seconds)


def select_items_to_handle(
    items: Sequence[WorkItem],
    force: bool,
    input_fn: Callable[[str], str] = input,
) -> list[WorkItem]:
    """Select which items should be sent to the AI backend."""

    return LabellingWorkflow.select_items_to_handle(
        items,
        force,
        input_fn=input_fn,
    )


# pylint: disable=too-many-arguments,too-many-positional-arguments
def run_ai_batch(
    items: Sequence[WorkItem],
    valid_labels: Sequence[LabelDefinition],
    model: str,
    allow_label_removals: bool,
    input_fn: Callable[[str], str] = input,
    valid_issue_types: Sequence[IssueTypeDefinition] = (),
) -> list[SuggestionResult]:
    """Run AI suggestions for selected items in parallel."""

    return _call_workflow(
        "run_ai_batch",
        items,
        valid_labels,
        model,
        allow_label_removals,
        input_fn=input_fn,
        valid_issue_types=valid_issue_types,
    )


def print_summary(
    item: WorkItem,
    label_suggestion: LabelSuggestion,
) -> None:
    """Print the suggested label changes for one work item."""

    LabellingWorkflow.print_summary(item, label_suggestion)


# pylint: disable=too-many-arguments,too-many-positional-arguments
def review_and_apply_suggestions(
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

    _call_workflow(
        "review_and_apply_suggestions",
        repo,
        suggestion_results,
        force,
        allow_label_removals,
        input_fn=input_fn,
        dry_run=dry_run,
        comment_reason=comment_reason,
        valid_issue_types=valid_issue_types,
    )


def collect_items(repo: str, args) -> list[WorkItem]:
    """Collect and sort matching issues and pull requests."""

    return _call_workflow("collect_items", repo, args)


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
        warn_force_mode(args.dry_run)

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


__all__ = [name for name in globals() if not name.startswith("_")]
