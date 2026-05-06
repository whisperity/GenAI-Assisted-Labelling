"""Tests for the abstract backend prompt contract."""
# pylint: disable=duplicate-code

import unittest

from test.helpers import make_item

from ai_labelling.backends.base import AIBackend
from ai_labelling.models import LabelDefinition, ModelSpec


class BackendBaseTests(unittest.TestCase):
    """Verify backend prompt construction and delegation behaviour."""

    def test_generic_backend_prompt_includes_context(self):
        """The shared backend prompt should include title, labels, and body."""

        backend = AIBackend(name="Test")
        item = make_item(1, "Test issue", labels=["bug"])
        prompt = backend.build_prompt(
            item,
            [
                LabelDefinition("bug", "Bug report"),
                LabelDefinition("docs", "Documentation"),
            ],
            allow_label_removals=True,
        )

        self.assertIn("You are labeling a GitHub issue.", prompt)
        self.assertIn("Issue title:\nTest issue", prompt)
        self.assertIn('"name": "bug"', prompt)
        self.assertIn('"description": "Documentation"', prompt)
        self.assertIn("Body text", prompt)
        self.assertIn(
            "Base your decision only on the title and main body text",
            prompt,
        )
        self.assertIn("remove_labels", prompt)

    def test_generic_backend_prompt_omits_removals_when_disabled(self):
        """Removal instructions should disappear when removals are disabled."""

        backend = AIBackend(name="Test")
        prompt = backend.build_prompt(
            make_item(1, "Test issue", labels=["bug"]),
            [LabelDefinition("bug", "Bug report")],
            allow_label_removals=False,
        )

        self.assertNotIn("remove_labels", prompt)
        self.assertIn("do not support adding labels confidently", prompt)

    def test_suggest_labels_delegates_to_backend_runner(self):
        """Backends should use ``run_prompt`` and return its object result."""

        class FakeBackend(AIBackend):  # pylint: disable=too-few-public-methods
            """Simple backend used to capture the prompt for assertions."""

            def __init__(self):
                super().__init__(name="Fake")
                self.prompt = None
                self.model = None
                self.allow_label_removals = None

            def run_prompt(
                self,
                prompt,
                model_spec,
                *,
                allow_label_removals,
            ):
                self.prompt = prompt
                self.model = model_spec
                self.allow_label_removals = allow_label_removals
                return {
                    "add_labels": ["bug"],
                    "remove_labels": [],
                    "reason": "match",
                }

        backend = FakeBackend()
        item = make_item(2, "Another issue")
        result = backend.suggest_labels(
            item,
            [LabelDefinition("bug", "Bug report")],
            ModelSpec("codex", "gpt-5.4-mini", "low"),
            False,
        )

        self.assertEqual(result["add_labels"], ["bug"])
        self.assertEqual(backend.model.provider, "codex")
        self.assertEqual(backend.model.model, "gpt-5.4-mini")
        self.assertFalse(backend.allow_label_removals)
        self.assertIn("Body text", backend.prompt)
