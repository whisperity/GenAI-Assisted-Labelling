"""Interactive prompt helpers for the labelling workflow."""

from ai_labelling.models import InputFn, UserQuit
from ai_labelling.terminal import colourise


def print_prompt_help(allow_apply_all: bool) -> None:
    """Print git-style help text for interactive single-character answers."""

    if allow_apply_all:
        print(colourise("y - yes, use this", "blue", bold=True))
        print(colourise("n - no, skip this", "blue", bold=True))
        print(
            colourise(
                "a - all, use this and all remaining in this action",
                "blue",
                bold=True,
            )
        )
        print(
            colourise(
                "d - done, skip this and remaining labels in this action",
                "blue",
                bold=True,
            )
        )
    else:
        print(colourise("y - yes, handle this item", "blue", bold=True))
        print(colourise("n - no, skip this item", "blue", bold=True))
        print(
            colourise(
                "a - all, handle all remaining items",
                "blue",
                bold=True,
            )
        )
        print(
            colourise(
                "d - done, stop prompting more items",
                "blue",
                bold=True,
            )
        )
    print(colourise("q - quit, terminate program now", "blue", bold=True))
    print(colourise("? - help, show this help", "blue", bold=True))
    print()


def prompt_confirmation(
    prompt: str,
    *,
    allow_apply_all: bool,
    input_fn: InputFn = input,
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
    input_fn: InputFn = input,
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
