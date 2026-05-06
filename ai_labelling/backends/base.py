"""Abstract backend interface."""

import dataclasses
import json
from dataclasses import dataclass
from typing import Dict, Sequence

from ai_labelling.models import LabelDefinition, ModelSpec, WorkItem


@dataclass
class AIBackend:
    """Backend interface for AI-driven label suggestions."""

    name: str

    def suggest_labels(
        self,
        item: WorkItem,
        valid_labels: Sequence[LabelDefinition],
        model_spec: ModelSpec,
        allow_label_removals: bool,
    ) -> Dict[str, object]:
        """Ask the backend for label suggestions for a single work item."""

        result = self.run_prompt(
            self.build_prompt(
                item,
                valid_labels,
                allow_label_removals=allow_label_removals,
            ),
            model_spec,
            allow_label_removals=allow_label_removals,
        )
        if not isinstance(result, dict):
            raise RuntimeError(
                f"{self.name} returned a non-object response"
            )
        return result

    def build_prompt(
        self,
        item: WorkItem,
        valid_labels: Sequence[LabelDefinition],
        *,
        allow_label_removals: bool,
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

        return f"""You are labeling a GitHub {item.kind}.

Choose labels only from the provided valid label list.
Base your decision only on the title and main body text of the {item.kind}.
Do not use comments, code, linked changes, or any outside context.

Return JSON with:
- add_labels: labels that should be present but are currently missing
{removal_lines}- reason: short explanation

If the title and body do not support {confidence_action} labels confidently,
return empty lists.

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
