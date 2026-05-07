"""Tests for shared data structures and pure data helpers."""
# pylint: disable=missing-function-docstring

import unittest
from datetime import timezone

from ai_labelling.models import (
    IssueTypeDefinition,
    LabelDefinition,
    LabelSuggestion,
    parse_github_timestamp,
)


class TimestampTests(unittest.TestCase):
    """Verify GitHub-style timestamp parsing."""

    def test_returns_aware_datetime(self):
        parsed = parse_github_timestamp("2026-05-01T00:00:00Z")
        self.assertEqual(parsed.tzinfo, timezone.utc)

    def test_parses_offset_form(self):
        parsed = parse_github_timestamp("2026-05-01T00:00:00+02:00")
        self.assertIsNotNone(parsed.tzinfo)


class LabelSuggestionFromRawTests(unittest.TestCase):
    """Verify normalisation of raw LLM output into ``LabelSuggestion``."""

    def setUp(self):
        self.valid = [
            LabelDefinition("bug", ""),
            LabelDefinition("docs", ""),
            LabelDefinition("clang", ""),
        ]

    def test_drops_unknown_and_duplicates(self):
        suggestion = LabelSuggestion.from_raw(
            {
                "add_labels": ["bug", "BUG", "unknown", "clang"],
                "remove_labels": ["bug", "docs"],
                "reason": "test",
            },
            self.valid,
            ["clang"],
        )
        self.assertEqual(suggestion.add_labels, ["bug"])
        self.assertEqual(suggestion.remove_labels, [])
        self.assertEqual(suggestion.reason, "test")

    def test_skips_non_string_entries(self):
        suggestion = LabelSuggestion.from_raw(
            {"add_labels": [1, None, "bug"], "reason": ""},
            self.valid,
            [],
        )
        self.assertEqual(suggestion.add_labels, ["bug"])

    def test_non_list_add_labels_yields_empty(self):
        suggestion = LabelSuggestion.from_raw(
            {"add_labels": "bug", "reason": ""},
            self.valid,
            [],
        )
        self.assertEqual(suggestion.add_labels, [])

    def test_remove_only_existing_labels(self):
        suggestion = LabelSuggestion.from_raw(
            {"remove_labels": ["bug", "docs"], "reason": ""},
            self.valid,
            ["bug"],
        )
        self.assertEqual(suggestion.remove_labels, ["bug"])

    def test_empty_payload_returns_empty_suggestion(self):
        suggestion = LabelSuggestion.from_raw({}, self.valid, [])
        self.assertEqual(suggestion.add_labels, [])
        self.assertEqual(suggestion.remove_labels, [])
        self.assertEqual(suggestion.reason, "")
        self.assertIsNone(suggestion.issue_type)

    def test_reason_is_stripped_and_stringified(self):
        suggestion = LabelSuggestion.from_raw(
            {"reason": "  test reason  "}, self.valid, [],
        )
        self.assertEqual(suggestion.reason, "test reason")

    def test_extracts_valid_issue_type(self):
        types = [IssueTypeDefinition("Bug", "A bug", 1)]
        suggestion = LabelSuggestion.from_raw(
            {"issue_type": "bug"},
            self.valid,
            [],
            valid_issue_types=types,
        )
        self.assertEqual(suggestion.issue_type, "Bug")

    def test_ignores_same_issue_type(self):
        types = [IssueTypeDefinition("Bug", "A bug", 1)]
        suggestion = LabelSuggestion.from_raw(
            {"issue_type": "Bug"},
            self.valid,
            [],
            valid_issue_types=types,
            current_issue_type="Bug",
        )
        self.assertIsNone(suggestion.issue_type)

    def test_ignores_unknown_issue_type(self):
        types = [IssueTypeDefinition("Bug", "A bug", 1)]
        suggestion = LabelSuggestion.from_raw(
            {"issue_type": "Mystery"},
            self.valid,
            [],
            valid_issue_types=types,
        )
        self.assertIsNone(suggestion.issue_type)

    def test_ignores_blank_issue_type(self):
        types = [IssueTypeDefinition("Bug", "A bug", 1)]
        suggestion = LabelSuggestion.from_raw(
            {"issue_type": "   "},
            self.valid,
            [],
            valid_issue_types=types,
        )
        self.assertIsNone(suggestion.issue_type)

    def test_ignores_non_string_issue_type(self):
        types = [IssueTypeDefinition("Bug", "A bug", 1)]
        suggestion = LabelSuggestion.from_raw(
            {"issue_type": 42},
            self.valid,
            [],
            valid_issue_types=types,
        )
        self.assertIsNone(suggestion.issue_type)

    def test_no_valid_issue_types_means_none(self):
        suggestion = LabelSuggestion.from_raw(
            {"issue_type": "Bug"},
            self.valid,
            [],
            valid_issue_types=(),
        )
        self.assertIsNone(suggestion.issue_type)


class DataclassEqualityTests(unittest.TestCase):
    """Verify the dataclasses behave as plain value types."""

    def test_label_definition_equality(self):
        self.assertEqual(
            LabelDefinition("bug", "x"), LabelDefinition("bug", "x"),
        )

    def test_issue_type_definition_default_id_is_zero(self):
        self.assertEqual(IssueTypeDefinition("Bug", "").type_id, 0)
