"""Formatting and presentation helpers."""

import re
import sys
import textwrap
import traceback
from typing import Callable, Dict, List, Sequence

from ai_labelling.models import (
    PreviewBlock,
    SuggestionResult,
    WorkItem,
    parse_github_timestamp,
)
from ai_labelling.terminal import colourise

_RenderFn = Callable[[List[str], str], List[str]]
"""Type alias for body-block rendering functions used in preview builders."""

# Inline Markdown span patterns, matched in priority order so that ***
# is tested before ** which is tested before bare *.
_INLINE_MD_RE = re.compile(
    r"\*\*\*(.+?)\*\*\*"      # bold + italic  → group 1
    r"|\*\*(.+?)\*\*"         # bold           → group 2
    r"|\*(.+?)\*"             # italic *       → group 3
    r"|(?<!\w)_(.+?)_(?!\w)"  # italic _       → group 4
    r"|`([^`]+)`"             # inline code    → group 5
)


def format_display_timestamp(timestamp: str) -> str:
    """Format a GitHub timestamp for human-readable terminal output."""

    parsed = parse_github_timestamp(timestamp)
    local = parsed.astimezone()
    return local.strftime("%Y-%m-%d %H:%M:%S %Z")


def classify_preview_block(block_text: str) -> str:
    """Classify a body block for preview display and summary accounting."""

    content_lines = [
        line.strip() for line in block_text.splitlines() if line.strip()
    ]
    if not content_lines:
        return "empty"
    if block_text.lstrip().startswith("```"):
        return "code"
    if all(line.startswith(">") for line in content_lines):
        return "quote"
    if len(content_lines) == 1:
        line = content_lines[0]
        if line.startswith("`") and line.endswith("`") and len(line) >= 2:
            return "code"
    if all(line.startswith("#") for line in content_lines):
        return "heading"
    return "text"


def parse_preview_blocks(body: str) -> List[PreviewBlock]:
    """Split an issue body into markdown-aware blocks."""

    stripped_body = body.strip()
    if not stripped_body:
        return [PreviewBlock("text", "(no description)")]

    blocks: List[PreviewBlock] = []
    current_lines: List[str] = []
    in_code_block = False

    def flush_current_block() -> None:
        """Finalise the current block and append it to the parsed list."""

        if not current_lines:
            return
        block_text = "\n".join(current_lines).strip()
        current_lines.clear()
        if not block_text:
            return
        blocks.append(
            PreviewBlock(
                classify_preview_block(block_text),
                block_text,
            )
        )

    for line in stripped_body.splitlines():
        if line.startswith("```"):
            current_lines.append(line)
            in_code_block = not in_code_block
            if not in_code_block:
                flush_current_block()
            continue
        if in_code_block:
            current_lines.append(line)
            continue
        if not line.strip():
            flush_current_block()
            continue
        current_lines.append(line)

    flush_current_block()
    return blocks


def summarise_body(body: str) -> str:
    """Return a short, human-friendly preview of an issue body."""

    paragraph_parts = [
        " ".join(block.text.split())
        for block in parse_preview_blocks(body)
        if block.kind == "text"
    ]
    paragraph_text = " ".join(part for part in paragraph_parts if part)

    sentence_parts = [
        sentence.strip()
        for sentence in paragraph_text.split(".")
        if sentence.strip()
    ]

    if len(sentence_parts) >= 3:
        summary = ". ".join(sentence_parts[:3]) + "."
    else:
        summary = paragraph_text

    if len(summary) > 280:
        return summary[:277].rstrip() + "..."
    return summary


def wrap_preserving_newlines(text: str, width: int) -> List[str]:
    """Wrap long lines without collapsing or reflowing existing line breaks."""

    wrapped_lines: List[str] = []
    for source_line in text.splitlines():
        if not source_line.strip():
            wrapped_lines.append("")
            continue
        wrapped_lines.extend(
            textwrap.wrap(
                source_line,
                width=width,
                replace_whitespace=False,
                drop_whitespace=False,
            )
        )
    return wrapped_lines


def non_empty_lines(lines: Sequence[str]) -> List[str]:
    """Return non-empty lines from a wrapped preview block."""

    return [line for line in lines if line.strip()]


def take_non_empty_lines(lines: Sequence[str], limit: int) -> List[str]:
    """Take wrapped preview lines until the requested non-empty line limit."""

    selected_lines: List[str] = []
    non_empty_count = 0
    for line in lines:
        selected_lines.append(line)
        if line.strip():
            non_empty_count += 1
        if non_empty_count >= limit:
            break
    while selected_lines and selected_lines[-1] == "":
        selected_lines.pop()
    return selected_lines


def _identity_block_lines(
    lines: List[str], block_kind: str  # pylint: disable=unused-argument
) -> List[str]:
    """No-op render function: return lines unchanged."""

    return lines


def _build_preview(
    body: str,
    *,
    width: int,
    max_lines: int,
    render_fn: _RenderFn,
) -> str:
    """Core preview loop shared by plain and colourised preview builders."""

    blocks = parse_preview_blocks(body)
    wrapped_lines: List[str] = []
    counted_lines = 0

    for index, block in enumerate(blocks):
        block_lines = wrap_preserving_newlines(block.text, width)
        block_counts = block.kind == "text"
        if (
            block_counts
            and counted_lines + len(non_empty_lines(block_lines)) > max_lines
        ):
            remaining_lines = max_lines - counted_lines
            if remaining_lines > 0:
                partial_lines = take_non_empty_lines(
                    block_lines,
                    remaining_lines,
                )
                if partial_lines:
                    partial_lines[-1] = (
                        partial_lines[-1].rstrip(". ") + "..."
                    )
                wrapped_lines.extend(
                    render_fn(partial_lines, block.kind)
                )
            break

        wrapped_lines.extend(render_fn(block_lines, block.kind))
        if block_counts:
            counted_lines += len(non_empty_lines(block_lines))
            if counted_lines >= max_lines:
                break
        if index + 1 < len(blocks):
            wrapped_lines.append("")

    while wrapped_lines and wrapped_lines[-1] == "":
        wrapped_lines.pop()
    return "\n".join(wrapped_lines)


def format_body_preview(
    body: str,
    *,
    width: int = 72,
    max_lines: int = 8,
) -> str:
    """Wrap and truncate an issue body preview for terminal display."""

    return _build_preview(
        body,
        width=width,
        max_lines=max_lines,
        render_fn=_identity_block_lines,
    )


def colourise_inline_markdown(line: str) -> str:
    """Colourise inline Markdown spans and strip their formatting markers."""

    result: List[str] = []
    pos = 0
    for match in _INLINE_MD_RE.finditer(line):
        before = line[pos:match.start()]
        if before:
            result.append(colourise(before, "white"))
        groups = match.groups()
        if groups[0] is not None:        # ***bold+italic*** → red
            result.append(colourise(groups[0], "red", bold=True))
        elif groups[1] is not None:      # **bold**          → red
            result.append(colourise(groups[1], "red", bold=True))
        elif groups[2] is not None:      # *italic*          → yellow
            result.append(colourise(groups[2], "yellow"))
        elif groups[3] is not None:      # _italic_          → yellow
            result.append(colourise(groups[3], "yellow"))
        elif groups[4] is not None:      # `code`            → green
            result.append(colourise(groups[4], "green"))
        pos = match.end()
    tail = line[pos:]
    if tail:
        result.append(colourise(tail, "white"))
    return "".join(result)


def _render_text_block_lines(lines: List[str]) -> List[str]:
    """Render a "text" block with inline-markdown colour and setext detect."""

    result: List[str] = []
    i = 0
    while i < len(lines):
        current = lines[i]
        if i + 1 < len(lines):
            nxt = lines[i + 1].strip()
            if (
                nxt
                and len(nxt) >= 2
                and nxt[0] in "=-"
                and all(c == nxt[0] for c in nxt)
            ):
                result.append(colourise(current.strip(), "reverse"))
                i += 2
                continue
        if current.strip():
            result.append(colourise_inline_markdown(current))
        else:
            result.append(current)
        i += 1
    return result


def _render_code_block_lines(lines: List[str]) -> List[str]:
    """Render a fenced or standalone code block, stripping fences/backticks."""

    result: List[str] = []
    for line in lines:
        if line.lstrip().startswith("```"):
            continue
        stripped = line.strip()
        if (
            stripped.startswith("`")
            and stripped.endswith("`")
            and "`" not in stripped[1:-1]
        ):
            result.append(stripped[1:-1])
        else:
            result.append(line)
    return result


def _render_colourised_block_lines(
    lines: List[str],
    block_kind: str,
) -> List[str]:
    """Apply ANSI colour to a list of pre-wrapped lines for one block."""

    if block_kind == "heading":
        return [
            colourise(line.lstrip("#").strip(), "reverse")
            for line in lines
            if line.strip()
        ]
    if block_kind == "code":
        return _render_code_block_lines(lines)
    if block_kind == "text":
        return _render_text_block_lines(lines)
    return [
        colourise(line, "white") if line.strip() else line
        for line in lines
    ]


def format_body_preview_colourised(
    body: str,
    *,
    width: int = 72,
    max_lines: int = 8,
) -> str:
    """Wrap and truncate a body preview with ANSI Markdown colouring."""

    return _build_preview(
        body,
        width=width,
        max_lines=max_lines,
        render_fn=_render_colourised_block_lines,
    )


def format_label_block(labels: Sequence[str]) -> str:
    """Format labels as a bullet list for terminal-friendly display."""

    if not labels:
        return "  - (none)"
    return "\n".join(
        f"  - {label}" for label in sorted(labels, key=str.casefold)
    )


def format_reason(reason: str) -> str:
    """Wrap and indent the model's reason for easier reading."""

    if not reason.strip():
        return ""
    return textwrap.fill(
        reason.strip(),
        width=72,
        initial_indent="  ",
        subsequent_indent="  ",
    )


def print_item_details(item: WorkItem) -> None:
    """Print a human-friendly item header and truncated description preview."""

    print(
        colourise(item.title, "cyan", bold=True)
        + " "
        + colourise(f"(#{item.number})", "yellow", bold=True)
    )
    print(colourise("Status: ", "blue", bold=True) + item.state.capitalize())
    print(
        colourise("Created by: ", "blue", bold=True)
        + f"@{item.author_login}"
    )
    print(
        colourise("Created at: ", "blue", bold=True)
        + format_display_timestamp(item.created_at)
    )
    print(
        colourise("Last modified: ", "blue", bold=True)
        + format_display_timestamp(item.updated_at)
    )
    print(colourise("Existing labels:", "blue", bold=True))
    print(format_label_block(item.labels))
    if item.assignees:
        print(colourise("Assignees:", "blue", bold=True))
        for login in item.assignees:
            print(f"  - @{login}")
    else:
        print(colourise("Assignee:", "blue", bold=True) + " [none]")
    print()
    print(format_body_preview_colourised(item.body))
    print()


def print_matching_items(
    items: Sequence[WorkItem],
    heading: str,
) -> None:
    """Print the issues and pull requests selected by the current filters."""

    print(colourise(f"{heading}:", "blue", bold=True))
    for item in items:
        state_colour = "green" if item.state.casefold() == "open" else "red"
        print(
            "- "
            + item.title
            + " "
            + colourise(f"(#{item.number})", "yellow", bold=True)
            + " "
            + colourise(f"[{item.state}]", state_colour, bold=True)
        )
    print()


def print_exception_diagnostics(exc: Exception, context: str) -> None:
    """Print a labelled stack trace for a failed retryable operation."""

    print(
        colourise(f"{context} failed:", "red", stream=sys.stderr, bold=True),
        file=sys.stderr,
    )
    traceback.print_exception(exc, file=sys.stderr)


def describe_match_bucket(items: Sequence[WorkItem]) -> str:
    """Describe one homogeneous matched-item bucket for summary output."""

    if not items:
        return "items"
    state = items[0].state.casefold()
    kind_label = "issues" if items[0].kind == "issue" else "PRs"
    return f"{state} {kind_label}"


def bucketise_items(
    items: Sequence[WorkItem],
) -> Dict[str, List[WorkItem]]:
    """Group items by open/closed state and by issue-vs-PR kind."""

    buckets: Dict[str, List[WorkItem]] = {}
    for item in items:
        state = item.state.casefold()
        kind_label = "issues" if item.kind == "issue" else "PRs"
        key = f"{state} {kind_label}"
        buckets.setdefault(key, []).append(item)
    return buckets


def print_match_summary(items: Sequence[WorkItem]) -> None:
    """Print a compact matched-item count summary."""

    buckets = bucketise_items(items)
    if len(buckets) == 1:
        label, bucket_items = next(iter(buckets.items()))
        print(
            colourise("Matched ", "blue", bold=True)
            + f"{label}: {len(bucket_items)}"
        )
        return

    print(colourise("Matched items:", "blue", bold=True))
    for label, bucket_items in buckets.items():
        words = label.split(" ", maxsplit=1)
        print(f"  - {words[0].capitalize()} {words[1]}: {len(bucket_items)}")


def print_changes_summary(  # pylint: disable=too-many-locals
    suggestion_results: Sequence[SuggestionResult],
    allow_label_removals: bool,
    *,
    dry_run: bool = True,
) -> None:
    """Print label-change summary after a run; silent when nothing changed."""

    tag = colourise("[Assignee]", "cyan", bold=True)
    type_tag = colourise("[Type]", "cyan", bold=True)

    lines: List[str] = []
    for result in suggestion_results:
        item = result.item
        suggestion = result.label_suggestion
        has_assignee = result.applied_assignee is not None
        has_type = item.kind == "issue" and suggestion.issue_type is not None
        has_adds = bool(suggestion.add_labels)
        has_removes = allow_label_removals and bool(suggestion.remove_labels)
        if not any([has_assignee, has_type, has_adds, has_removes]):
            continue
        state_colour = "green" if item.state.casefold() == "open" else "red"
        kind_display = "issue" if item.kind == "issue" else "PR"
        lines.append(
            "- "
            + colourise(f"#{item.number}", "yellow", bold=True)
            + " "
            + item.title
            + " "
            + colourise(
                f"[{item.state} {kind_display}]", state_colour, bold=True
            )
        )
        if has_assignee:
            old_part = (
                ", ".join(f"@{a}" for a in item.assignees)
                or "[none]"
            )
            lines.append(
                f"  {tag} {old_part} -> @{result.applied_assignee}"
            )
        if has_type:
            old_type = item.issue_type or "[none]"
            lines.append(
                f"  {type_tag} {old_type} -> {suggestion.issue_type}"
            )
        for lbl in suggestion.add_labels:
            lines.append("  " + colourise(f"+ {lbl}", "green"))
        if allow_label_removals:
            for lbl in suggestion.remove_labels:
                lines.append("  " + colourise(f"- {lbl}", "magenta"))

    if not lines:
        return

    title = (
        "Label changes (not applied):"
        if dry_run
        else "Label changes:"
    )
    print(colourise(title, "blue", bold=True))
    for line in lines:
        print(line)
    print()
