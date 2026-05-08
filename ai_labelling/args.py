"""Argument parsing and model-selection helpers."""

import argparse
from datetime import date, datetime, time, timedelta, timezone
from typing import Optional

from ai_labelling.backends import (
    get_provider_default_model,
    get_provider_help_description,
    get_provider_reasoning_levels,
    get_supported_providers,
)
from ai_labelling.config import (
    DEFAULT_DATE_CUTOFF,
    DEFAULT_MODEL_SPEC,
)
from ai_labelling.models import ModelSpec
from ai_labelling.terminal import colourise


def format_help_epilog_entry(name: str, description: str) -> str:
    """Format one epilog entry with help-like ANSI highlighting."""

    return (
        "  "
        + colourise(name, "cyan", bold=True)
        + f": {description}"
    )


def build_help_epilog() -> str:
    """Build the extended ``--help`` footer for debug and AI provider docs."""

    providers = "\n".join(
        format_help_epilog_entry(
            provider,
            get_provider_help_description(provider),
        )
        for provider in get_supported_providers()
    )
    return (
        colourise("Debugging:", "magenta", bold=True)
        + "\n"
        + format_help_epilog_entry(
            "DEBUG unset, empty, or 0",
            "no debug output",
        )
        + "\n"
        + format_help_epilog_entry(
            "DEBUG=1",
            "show subprocess commands and request/response timing",
        )
        + "\n"
        + format_help_epilog_entry(
            "DEBUG=2",
            "also show JSON responses (pretty-printed) "
            "and sanitised AI prompts",
        )
        + "\n"
        + format_help_epilog_entry(
            "DEBUG=3 or greater",
            "also show full AI prompts",
        )
        + "\n\n"
        + colourise("AI providers:", "magenta", bold=True)
        + "\n"
        f"{providers}"
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the command-line parser used by the labelling workflow."""

    parser = argparse.ArgumentParser(
        description=(
            "Suggest labels for GitHub issues and pull requests using a "
            "generative AI large language model service from the title and "
            "description of the issue, and the list of labels obtained from "
            "the GitHub repository."
        ),
        epilog=build_help_epilog(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    github_group = parser.add_argument_group("GitHub options")
    filter_group = parser.add_argument_group("filter options")
    ai_group = parser.add_argument_group("artificial intelligence options")
    action_group = parser.add_argument_group("action options")

    parser.add_argument(
        "--interactive",
        action="store_true",
        help=(
            "Enter an interactive loop where issue or pull request numbers "
            "are entered one at a time. All filter options and '--id' are "
            "ignored in this mode. Labels are queried once for the "
            "repository. Enter 'q' or 'quit' to exit the loop."
        ),
    )

    issue_group = filter_group.add_mutually_exclusive_group()
    issue_group.add_argument(
        "--issues",
        dest="include_issues",
        action="store_true",
        help=(
            "Allow the handling of issues as candidates for labelling. "
            "(Default)"
        ),
    )
    issue_group.add_argument(
        "--no-issues",
        dest="include_issues",
        action="store_false",
        help=(
            "Do not allow the handling of issues as candidates for "
            "labelling."
        ),
    )

    prs_group = filter_group.add_mutually_exclusive_group()
    prs_group.add_argument(
        "--prs",
        dest="include_prs",
        action="store_true",
        help=(
            "Allow the handling of pull requests as candidates for "
            "labelling."
        ),
    )
    prs_group.add_argument(
        "--no-prs",
        dest="include_prs",
        action="store_false",
        help=(
            "Do not allow the handling of pull requests as candidates for "
            "labelling. (Default)"
        ),
    )

    filter_group.add_argument(
        "--date",
        type=parse_cutoff,
        help=(
            "Only consider items created ('--created') or updated on/after "
            "this ISO date or date-time, for example 2026-05-01 or "
            "2026-05-01T12:00:00+02:00. The default is the last 24 hours. "
            "Use 0 or 'all' to search without a date cutoff."
        ),
    )

    update_group = filter_group.add_mutually_exclusive_group()
    update_group.add_argument(
        "--created",
        dest="created",
        action="store_true",
        help="Filter issues and pull requests based on creation time.",
    )
    update_group.add_argument(
        "--updated",
        dest="created",
        action="store_false",
        help=(
            "Filter issues and pull requests based on last update time. "
            "(Default)"
        ),
    )

    open_group = filter_group.add_mutually_exclusive_group()
    open_group.add_argument(
        "--open",
        dest="include_open",
        action="store_true",
        help=(
            "Allow the handling of open issues and pull requests. "
            "(Default)"
        ),
    )
    open_group.add_argument(
        "--no-open",
        dest="include_open",
        action="store_false",
        help="Do not allow the handling of open issues and pull requests.",
    )

    closed_group = filter_group.add_mutually_exclusive_group()
    closed_group.add_argument(
        "--closed",
        dest="include_closed",
        action="store_true",
        help="Allow the handling of closed issues and pull requests.",
    )
    closed_group.add_argument(
        "--no-closed",
        dest="include_closed",
        action="store_false",
        help=(
            "Do not allow the handling of closed issues and pull requests. "
            "(Default)"
        ),
    )

    filter_group.add_argument(
        "--limit",
        type=positive_int,
        help="Optional maximum number of matching issues/PRs to process.",
    )

    action_group.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Run the full AI suggestion pipeline but **DO NOT** apply any "
            "label changes. Prints a summary of what would have changed. "
            "Combine with '--force' to suppress all item prompts too."
        ),
    )
    action_group.add_argument(
        "--force",
        action="store_true",
        help=(
            "Handle all items matching the filters, and apply all label "
            "change suggestions ***WITHOUT ANY ADDITIONAL CONFIRMATION*** "
            "(unless '--dry-run' is specified). Use with caution!"
        ),
    )
    action_group.add_argument(
        "--allow-label-removals",
        action="store_true",
        help=(
            "Allow AI suggestions to include label removals and permit this "
            "tool to remove labels. By default, only new labels are "
            "suggested."
        ),
    )
    action_group.add_argument(
        "--comment-reason",
        dest="comment_reason",
        action="store_true",
        help=(
            "After labelling each item, post a GitHub comment with the AI "
            "model used, its reasoning, and which label changes were accepted "
            "or rejected. Has no effect with '--dry-run'."
        ),
    )

    github_group.add_argument(
        "--repository",
        dest="repo",
        type=parse_repo_arg,
        metavar="OWNER/REPOSITORY",
        help=(
            "GitHub repository in <owner>/<repository> form. If omitted, "
            "automatically inferred from the current Git working directory, "
            "if available."
        ),
    )
    github_group.add_argument(
        "--id",
        type=positive_int,
        metavar="NUMBER",
        help=(
            "Fetch and handle exactly one issue or pull request by its "
            "number. The tool automatically detects whether the number "
            "refers to an issue or a pull request. **ALL** filter options "
            "are ignored in this mode."
        ),
    )

    ai_group.add_argument(
        "--model",
        default=DEFAULT_MODEL_SPEC,
        help=(
            "Provider/model to use in the form of 'PROVIDER', "
            "'PROVIDER:MODEL', or 'PROVIDER:MODEL:REASONING-LEVEL'. "
            f"Default is `{DEFAULT_MODEL_SPEC}`. Use `*` as MODEL to ask "
            "the provider for its current default. Omitted MODEL uses "
            "the provider hard-coded default. Omitted REASONING-LEVEL "
            "omits effort from the request entirely. "
            "See below for the supported providers, hard-coded defaults, "
            "and effort levels."
        ),
    )

    parser.set_defaults(
        include_issues=True,
        include_prs=False,
        include_open=True,
        include_closed=False,
        created=False,
        date=DEFAULT_DATE_CUTOFF,
    )
    return parser


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for repository search and labelling."""

    return build_argument_parser().parse_args()


def positive_int(value: str) -> int:
    """Parse a strictly positive integer argument value."""

    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_model_spec(value: str) -> ModelSpec:
    """Parse PROVIDER, PROVIDER:MODEL, or PROVIDER:MODEL:REASONING specs.

    Model selection:
    - PROVIDER: use provider's hard-coded default model
    - PROVIDER:* or PROVIDER:*:REASONING: use provider's dynamic default
    - PROVIDER:MODEL or PROVIDER:MODEL:REASONING: use specified model

    Reasoning effort is omitted from the request when not specified.
    """

    parts = value.split(":")
    if len(parts) not in (1, 2, 3):
        raise argparse.ArgumentTypeError(
            "model must be in PROVIDER, PROVIDER:MODEL, or "
            "PROVIDER:MODEL:REASONING form"
        )

    normalised_provider = parts[0].casefold()
    if normalised_provider not in get_supported_providers():
        raise argparse.ArgumentTypeError(f"{parts[0]} is not supported")

    # Determine model
    if len(parts) == 1:
        # No model specified: use provider's hard-coded default
        model = get_provider_default_model(normalised_provider)
    elif parts[1] == "*":
        # Explicit wildcard: use provider's dynamic default
        # (None triggers provider-specific logic)
        model = None
    else:
        # Explicit model name
        model = parts[1]
        if not model:
            raise argparse.ArgumentTypeError("model name must not be empty")

    # Reasoning effort: None means "omit from request" (provider decides)
    if len(parts) >= 3:
        reasoning_effort = parts[2]
        allowed_reasoning_levels = get_provider_reasoning_levels(
            normalised_provider
        )
        normalised_reasoning = reasoning_effort.casefold()
        if normalised_reasoning not in allowed_reasoning_levels:
            raise argparse.ArgumentTypeError(
                "unsupported reasoning effort for "
                f"{normalised_provider}: {reasoning_effort}"
            )
    else:
        normalised_reasoning = None

    return ModelSpec(
        provider=normalised_provider,
        model=model,
        reasoning_effort=normalised_reasoning,
    )


def parse_repo_arg(value: str) -> str:
    """Validate a repository argument in ``owner/repository`` form."""

    parts = value.split("/")
    if len(parts) != 2 or not all(parts):
        raise argparse.ArgumentTypeError(
            "repository must be in <owner>/<repository> form"
        )
    return value


def parse_cutoff(value: str) -> Optional[datetime]:
    """Parse an ISO date or date-time and normalise it to UTC."""

    if value.casefold() in {"0", "all"}:
        return None

    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        try:
            parsed_date = date.fromisoformat(value)
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"invalid ISO date or date-time: {value!r}"
            ) from exc
        parsed = datetime.combine(parsed_date, time.min)

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=datetime.now().astimezone().tzinfo)

    return parsed.astimezone(timezone.utc)


def default_cutoff(now: Optional[datetime] = None) -> datetime:
    """Return the rolling default cutoff of 24 hours before execution."""

    effective_now = now or datetime.now(timezone.utc)
    return effective_now.astimezone(timezone.utc) - timedelta(hours=24)
