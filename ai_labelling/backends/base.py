"""Abstract backend interface."""

import dataclasses
import json
from dataclasses import dataclass
from typing import Dict, Sequence

from ai_labelling.models import (
    IssueTypeDefinition,
    LabelDefinition,
    ModelSpec,
    WorkItem,
)


@dataclass
class AIBackend:
    """Backend interface for AI-driven label suggestions."""

    name: str

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def suggest_labels(
        self,
        item: WorkItem,
        valid_labels: Sequence[LabelDefinition],
        model_spec: ModelSpec,
        allow_label_removals: bool,
        valid_issue_types: Sequence[IssueTypeDefinition] = (),
    ) -> Dict[str, object]:
        """Ask the backend for label suggestions for a single work item."""

        result = self.run_prompt(
            self.build_prompt(
                item,
                valid_labels,
                allow_label_removals=allow_label_removals,
                valid_issue_types=valid_issue_types,
            ),
            model_spec,
            allow_label_removals=allow_label_removals,
        )
        if not isinstance(result, dict):
            raise RuntimeError(
                f"{self.name} returned a non-object response"
            )
        if not isinstance(result.get("add_labels"), list):
            raise RuntimeError(
                f"{self.name} returned a response without a valid "
                f"'add_labels' list: {result!r}"
            )
        if not isinstance(result.get("reason"), str):
            raise RuntimeError(
                f"{self.name} returned a response without a valid "
                f"'reason' string: {result!r}"
            )
        return result

    def build_prompt(
        self,
        item: WorkItem,
        valid_labels: Sequence[LabelDefinition],
        *,
        allow_label_removals: bool,
        valid_issue_types: Sequence[IssueTypeDefinition] = (),
    ) -> str:
        """Build the generic prompt for label-suggestion backends."""

        removal_lines = ""
        confidence_action = "adding"
        if allow_label_removals:
            removal_lines = (
                "- remove_labels: labels that appear unnecessary based only "
                "on the title and\n  body text\n"
            )
            confidence_action = "adding or removing"

        issue_type_lines = ""
        if item.kind == "issue" and valid_issue_types:
            current = item.issue_type or "none"
            types_json = json.dumps(
                [
                    {"name": t.name, "description": t.description}
                    for t in valid_issue_types
                ],
                indent=2,
            )
            issue_type_lines = (
                f"\nCurrent issue type: {current}"
                f"\n\nValid issue types:\n{types_json}"
                "\n\n- issue_type: name of the most fitting issue type from "
                "the valid list, or null if the current type is already "
                "correct or none fits\n"
            )

        return f"""You are labeling a GitHub {item.kind}.

Choose labels only from the provided valid label list.
Base your decision only on the title and main body text of the {item.kind}.
Do not use comments, code, linked changes, or any outside context.

Respond with ONLY a single JSON object. No prose, no markdown fences, no
explanation outside the JSON. Your entire response must be valid JSON.

The JSON object must contain exactly these fields:
- "add_labels": array of label name strings that should be present but
  are currently missing (use [] if none)
{removal_lines}{issue_type_lines}- "reason": required non-empty explanation

If the title and body do not support {confidence_action} labels confidently,
return empty lists but still include a non-empty "reason".

Example response shape:
{{"add_labels": [], "reason": "explanation here"}}

Issue title:
{item.title}

Existing labels:
{json.dumps(item.labels, indent=2)}

Valid labels:
{json.dumps([dataclasses.asdict(label) for label in valid_labels], indent=2)}

Main body text:
{item.body}
"""

    def run_prompt(
        self,
        prompt: str,
        model_spec: ModelSpec,
        *,
        allow_label_removals: bool,
    ) -> object:
        """Run a backend-specific prompt invocation."""

        raise NotImplementedError
