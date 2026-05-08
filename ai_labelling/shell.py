"""Subprocess helpers and git-based version detection."""

import importlib.metadata
import os
import shlex
import subprocess
from typing import Optional, Sequence

from ai_labelling.terminal import debug_log


def run(
    argv: Sequence[str],
    *,
    check: bool = True,
    input_text: Optional[str] = None,
) -> subprocess.CompletedProcess:
    """Run a subprocess and capture its text output streams."""

    debug_log(
        f"+ {' '.join((shlex.quote(s) for s in argv))}",
        colour="cyan",
    )
    return subprocess.run(
        argv,
        check=check,
        input=input_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def get_script_version() -> str:
    """Return the installed package version, a git SHA, or ``unknown``.

    Priority:
    1. ``importlib.metadata`` version of the installed ``ai-labelling``
       package, prefixed with ``v`` (e.g. ``v1.0.0``).
    2. Short git SHA of the repository the source lives in.
    3. The literal string ``unknown``.
    """

    try:
        return "v" + importlib.metadata.version("ai-labelling")
    except importlib.metadata.PackageNotFoundError:
        pass
    try:
        result = run(
            (
                "git",
                "-C",
                os.path.dirname(os.path.abspath(__file__)),
                "rev-parse",
                "--short",
                "HEAD",
            ),
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except OSError:
        pass
    return "unknown"
