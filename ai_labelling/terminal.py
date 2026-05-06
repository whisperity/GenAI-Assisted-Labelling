"""Terminal formatting and debug helpers."""

import os
import sys
from typing import List, Optional

from ai_labelling.config import ANSI_RESET, ANSI_STYLES


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


def debug_log(body: str, *, colour: str = "magenta") -> None:
    """Print one debug line when any non-zero ``DEBUG`` level is active."""

    if get_debug_level() < 1:
        return
    print(
        colourise(body, colour, stream=sys.stderr, bold=True),
        file=sys.stderr,
    )


def sanitise_prompt_for_debug(prompt: str) -> str:
    """Replace runtime-heavy prompt sections with placeholders for DEBUG=2."""

    replacements = {
        "Issue title:\n": "Issue title:\n<ISSUE TITLE OMITTED>\n",
        "Existing labels:\n": "Existing labels:\n<LABELS OMITTED>\n",
        "Valid labels:\n": "Valid labels:\n<LABEL DEFINITIONS OMITTED>\n",
        "Main body text:\n": "Main body text:\n<ISSUE BODY OMITTED>\n",
    }
    sanitised_lines: List[str] = []
    skip_mode: Optional[str] = None
    section_headers = tuple(replacements)
    for line in prompt.splitlines():
        if skip_mode is None and line + "\n" in replacements:
            header = line + "\n"
            sanitised_lines.append(line)
            sanitised_lines.append(replacements[header].splitlines()[1])
            skip_mode = header
            continue
        if skip_mode is not None:
            if any(line + "\n" == header for header in section_headers):
                header = line + "\n"
                sanitised_lines.append(line)
                sanitised_lines.append(replacements[header].splitlines()[1])
                skip_mode = header
                continue
            continue
        sanitised_lines.append(line)
    return "\n".join(sanitised_lines)


def format_prompt_for_debug(prompt: str) -> Optional[str]:
    """Return the prompt view appropriate for the current debug level."""

    debug_level = get_debug_level()
    if debug_level < 2:
        return None
    if debug_level == 2:
        return sanitise_prompt_for_debug(prompt)
    return prompt
