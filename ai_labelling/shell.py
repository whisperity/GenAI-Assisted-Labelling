"""Subprocess helpers."""

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
