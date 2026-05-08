"""Terminal formatting and debug helpers."""

import os
import sys
from typing import List, Optional

ANSI_RESET = "\033[0m"
ANSI_STYLES = {
    "blue": "\033[34m",
    "cyan": "\033[36m",
    "green": "\033[32m",
    "grey": "\033[90m",
    "magenta": "\033[35m",
    "red": "\033[31m",
    "reverse": "\033[7m",
    "white": "\033[97m",
    "yellow": "\033[33m",
    "bold": "\033[1m",
}
"""ANSI escape sequences used for dependency-free terminal colouring."""


def supports_colour(stream: object) -> bool:
    """Return whether ANSI colours should be emitted for a stream."""

    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    return bool(getattr(stream, "isatty", lambda: False)())


def colourise(
    text: str,
    colour: str,
    *,
    stream: object = sys.stdout,
    bold: bool = False,
) -> str:
    """Wrap text with ANSI colour escapes when supported."""

    if not supports_colour(stream):
        return text

    prefix = ANSI_STYLES[colour]
    if bold:
        prefix = ANSI_STYLES["bold"] + prefix
    return f"{prefix}{text}{ANSI_RESET}"


def get_debug_level() -> int:
    """Return the numeric debug level requested through ``DEBUG``."""

    debug_value = os.environ.get("DEBUG")
    if debug_value in (None, "", "0"):
        return 0
    try:
        return max(1, int(debug_value))
    except ValueError:
        return 1


def debug_log(
    body: str, *, colour: str = "magenta", min_level: int = 1
) -> None:
    """Print one debug line when ``DEBUG`` is at least ``min_level``."""

    if get_debug_level() < min_level:
        return
    print(
        colourise(body, colour, stream=sys.stderr, bold=True),
        file=sys.stderr,
    )


_PROMPT_REDACTIONS = {
    "Issue title:": "<ISSUE TITLE OMITTED>",
    "Existing labels:": "<LABELS OMITTED>",
    "Valid labels:": "<LABEL DEFINITIONS OMITTED>",
    "Main body text:": "<ISSUE BODY OMITTED>",
}


def sanitise_prompt_for_debug(prompt: str) -> str:
    """Replace runtime-heavy prompt sections with placeholders for DEBUG=2."""

    output: List[str] = []
    skipping = False
    for line in prompt.splitlines():
        replacement = _PROMPT_REDACTIONS.get(line)
        if replacement is not None:
            output.append(line)
            output.append(replacement)
            skipping = True
            continue
        if not skipping:
            output.append(line)
    return "\n".join(output)


def format_prompt_for_debug(prompt: str) -> Optional[str]:
    """Return the prompt view appropriate for the current debug level."""

    debug_level = get_debug_level()
    if debug_level < 2:
        return None
    if debug_level == 2:
        return sanitise_prompt_for_debug(prompt)
    return prompt
