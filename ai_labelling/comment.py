"""Markdown comment-body construction for labelling actions."""

from typing import List, Optional, Sequence

from ai_labelling.config import REPO_URL
from ai_labelling.models import LabelSuggestion


def _version_url(version: str) -> str:
    """Return the GitHub URL that the version string in a comment links to."""

    if version.startswith("v"):
        return f"{REPO_URL}/releases/tag/{version}"
    if version != "unknown":
        return f"{REPO_URL}/tree/{version}"
    return REPO_URL


def _format_label_lines(
    labels: Sequence[str],
    applied_cf: set,
) -> List[str]:
    """Render bullet-list lines for one suggested-label group."""

    lines: List[str] = []
    for lbl in sorted(labels, key=str.casefold):
        if lbl.casefold() in applied_cf:
            lines.append(f"  - `{lbl}`")
        else:
            lines.append(f"  - ~~`{lbl}`~~ (rejected by operator)")
    return lines


def _format_issue_type_line(
    suggested: str,
    applied: Optional[str],
) -> str:
    """Render the issue-type line, with strike-through when not accepted."""

    accepted = (
        applied is not None and applied.casefold() == suggested.casefold()
    )
    if accepted:
        return f"**Suggested issue type:** `{suggested}`"
    return (
        f"**Suggested issue type:** ~~`{suggested}`~~"
        " (rejected by operator)"
    )


# pylint: disable=too-many-arguments,too-many-positional-arguments
def format_comment_body(
    original_suggestion: LabelSuggestion,
    applied_add: Sequence[str],
    applied_remove: Sequence[str],
    model: str,
    version: str,
    allow_label_removals: bool,
    *,
    applied_issue_type: Optional[str] = None,
) -> str:
    """Build the Markdown comment body for a labelling action."""

    applied_add_cf = {lbl.casefold() for lbl in applied_add}
    applied_remove_cf = {lbl.casefold() for lbl in applied_remove}

    lines: List[str] = [
        f"## [**AI-assisted labelling**]({REPO_URL})",
        "",
        (f"script version [`{version}`]({_version_url(version)}), "
         f"using model: `{model}`"),
        "",
    ]

    if original_suggestion.reason.strip():
        lines += ["**Reasoning:**", ""]
        for line in original_suggestion.reason.strip().splitlines():
            lines.append(f"> {line}")
        lines.append("")

    if original_suggestion.issue_type is not None:
        lines.append(
            _format_issue_type_line(
                original_suggestion.issue_type, applied_issue_type,
            )
        )
        lines.append("")

    if original_suggestion.add_labels:
        lines.append("**Suggested additions:**")
        lines.extend(
            _format_label_lines(original_suggestion.add_labels, applied_add_cf)
        )
        lines.append("")

    if allow_label_removals and original_suggestion.remove_labels:
        lines.append("**Suggested removals:**")
        lines.extend(
            _format_label_lines(
                original_suggestion.remove_labels, applied_remove_cf,
            )
        )
        lines.append("")

    while lines and lines[-1] == "":
        lines.pop()
    lines.append("")

    return "\n".join(lines)
