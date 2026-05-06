"""Configuration constants for the labelling workflow."""

REPO_DETECTION_ORDER = ("upstream/push", "upstream", "origin")
"""Git remotes checked when inferring the current GitHub repository."""

DEFAULT_MODEL_SPEC = "codex:gpt-5.4-mini:low"
"""Default AI provider, model selector, and reasoning effort."""

FORCE_WARNING_DELAY_SECONDS = 15
"""Delay before running in force mode to give the user a chance to abort."""

DEFAULT_DATE_CUTOFF = object()
"""Sentinel meaning the script should use its default rolling cutoff."""

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
