"""Codex CLI backend implementation."""

import json
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

from ai_labelling.models import ModelSpec
from ai_labelling.shell import run
from ai_labelling.terminal import debug_log, format_prompt_for_debug

from .base import AIBackend

PROVIDER_NAME = "codex"
"""Provider string used in ``PROVIDER:MODEL:REASONING`` specs."""

DEFAULT_MODEL_NAME = "gpt-5.4-mini"
"""Hard-coded Codex default model used when the user omits one."""

REASONING_LEVELS = ("low", "medium", "high", "xhigh")
"""Reasoning-effort levels accepted by the Codex CLI."""


@dataclass
class CodexBackend(AIBackend):
    """AI backend implementation that shells out to the ``codex`` CLI."""

    command: str = "codex"

    @staticmethod
    def provider_name() -> str:
        """Return the stable provider string for this backend."""

        return PROVIDER_NAME

    @staticmethod
    def default_model_name() -> str:
        """Return the provider's hard-coded default model name."""

        return DEFAULT_MODEL_NAME

    @staticmethod
    def reasoning_levels() -> Sequence[str]:
        """Return supported reasoning-effort levels for this backend."""

        return REASONING_LEVELS

    @classmethod
    def help_description(cls) -> str:
        """Return the provider-specific ``--help`` description text."""

        return (
            "the `codex` CLI "
            "("
            f"default model `{cls.default_model_name()}`, "
            "effort: "
            + ", ".join(cls.reasoning_levels())
            + ")"
        )

    def run_prompt(
        self,
        prompt: str,
        model_spec: ModelSpec,
        *,
        allow_label_removals: bool,
    ) -> object:
        """Run a prompt through the Codex CLI and parse its JSON output."""

        debug_prompt = format_prompt_for_debug(prompt)
        if debug_prompt is not None:
            debug_log(debug_prompt)

        properties = {
            "add_labels": {
                "type": "array",
                "items": {"type": "string"},
                "default": [],
            },
            "reason": {"type": "string"},
        }
        required = ["add_labels", "reason"]
        if allow_label_removals:
            properties["remove_labels"] = {
                "type": "array",
                "items": {"type": "string"},
                "default": [],
            }
            required.insert(1, "remove_labels")

        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": properties,
            "required": required,
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json"
        ) as schema_file:
            json.dump(schema, schema_file)
            schema_file.flush()

            with tempfile.NamedTemporaryFile(
                mode="w+", suffix=".json"
            ) as output_file:
                argv = [
                    self.command,
                    "exec",
                    "--sandbox",
                    "read-only",
                    "--skip-git-repo-check",
                    "--ephemeral",
                    "--output-schema",
                    schema_file.name,
                    "--output-last-message",
                    output_file.name,
                ]
                if model_spec.model is not None:
                    argv.extend(["--model", model_spec.model])
                if model_spec.reasoning_effort:
                    argv.extend(
                        [
                            "-c",
                            (
                                "model_reasoning_effort="
                                f"\"{model_spec.reasoning_effort}\""
                            ),
                        ]
                    )
                argv.append("-")

                start = time.monotonic()
                start_ts = datetime.now().astimezone()
                try:
                    run(tuple(argv), input_text=prompt)
                finally:
                    elapsed = time.monotonic() - start
                    ts_str = start_ts.strftime("%Y-%m-%d %H:%M:%S %z")
                    debug_log(
                        f"  ↳ {ts_str}  ({elapsed:.3f}s)",
                        colour="yellow",
                        min_level=1,
                    )
                output_file.seek(0)
                return json.load(output_file)
