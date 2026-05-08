"""Configuration constants for the labelling workflow."""

REPO_URL = "https://github.com/whisperity/GenAI-Assisted-Labelling"
"""Public GitHub URL for this tool's repository."""

REPO_DETECTION_ORDER = ("upstream/push", "upstream", "origin")
"""Git remotes checked when inferring the current GitHub repository."""

DEFAULT_MODEL_SPEC = "codex:gpt-5.4-mini:low"
"""Default AI provider, model selector, and reasoning effort."""

FORCE_WARNING_DELAY_SECONDS = 15
"""Delay before running in force mode to give the user a chance to abort."""

DEFAULT_DATE_CUTOFF = object()
"""Sentinel meaning the script should use its default rolling cutoff."""
