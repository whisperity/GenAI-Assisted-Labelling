"""Tests for the markdown comment-body builder."""
# pylint: disable=missing-function-docstring

import unittest

from ai_labelling.comment import format_comment_body
from ai_labelling.models import LabelSuggestion


def _suggestion(add=None, remove=None, reason="", issue_type=None):
    return LabelSuggestion(
        add_labels=add or [],
        remove_labels=remove or [],
        reason=reason,
        issue_type=issue_type,
    )


class CommentBodyTests(  # pylint: disable=too-many-public-methods
    unittest.TestCase,
):
    """Verify the Markdown comment body generated for labelling actions."""

    def test_comment_contains_header_link(self):
        body = format_comment_body(
            _suggestion(add=["bug"]),
            applied_add=["bug"],
            applied_remove=[],
            model="anthropic:claude-haiku-4-5-20251001",
            version="abc1234",
            allow_label_removals=False,
        )
        self.assertIn("AI-assisted labelling", body)
        self.assertIn("GenAI-Assisted-Labelling", body)

    def test_comment_contains_version_and_model(self):
        body = format_comment_body(
            _suggestion(add=["bug"]),
            applied_add=["bug"],
            applied_remove=[],
            model="anthropic:claude-haiku-4-5-20251001",
            version="abc1234",
            allow_label_removals=False,
        )
        self.assertIn("`abc1234`", body)
        self.assertIn("`anthropic:claude-haiku-4-5-20251001`", body)

    def test_comment_quotes_reasoning(self):
        body = format_comment_body(
            _suggestion(add=["bug"], reason="Matches a crash pattern."),
            applied_add=["bug"],
            applied_remove=[],
            model="m",
            version="v",
            allow_label_removals=False,
        )
        self.assertIn("> Matches a crash pattern.", body)

    def test_comment_quotes_multiline_reasoning(self):
        body = format_comment_body(
            _suggestion(add=[], reason="Line one.\nLine two."),
            applied_add=[],
            applied_remove=[],
            model="m",
            version="v",
            allow_label_removals=False,
        )
        self.assertIn("> Line one.", body)
        self.assertIn("> Line two.", body)

    def test_comment_omits_reason_section_when_blank(self):
        body = format_comment_body(
            _suggestion(add=["bug"], reason="   "),
            applied_add=["bug"],
            applied_remove=[],
            model="m",
            version="v",
            allow_label_removals=False,
        )
        self.assertNotIn("**Reasoning:**", body)

    def test_comment_accepted_addition_no_strikethrough(self):
        body = format_comment_body(
            _suggestion(add=["bug"]),
            applied_add=["bug"],
            applied_remove=[],
            model="m",
            version="v",
            allow_label_removals=False,
        )
        self.assertIn("  - `bug`", body)
        self.assertNotIn("~~", body)

    def test_comment_rejected_addition_uses_strikethrough(self):
        body = format_comment_body(
            _suggestion(add=["bug", "docs"]),
            applied_add=["bug"],
            applied_remove=[],
            model="m",
            version="v",
            allow_label_removals=False,
        )
        self.assertIn("  - `bug`", body)
        self.assertIn("  - ~~`docs`~~ (rejected by operator)", body)

    def test_comment_removals_shown_when_allowed(self):
        body = format_comment_body(
            _suggestion(add=[], remove=["stale"]),
            applied_add=[],
            applied_remove=["stale"],
            model="m",
            version="v",
            allow_label_removals=True,
        )
        self.assertIn("Suggested removals:", body)
        self.assertIn("  - `stale`", body)

    def test_comment_removals_hidden_when_disallowed(self):
        body = format_comment_body(
            _suggestion(add=[], remove=["stale"]),
            applied_add=[],
            applied_remove=[],
            model="m",
            version="v",
            allow_label_removals=False,
        )
        self.assertNotIn("Suggested removals:", body)
        self.assertNotIn("stale", body)

    def test_comment_rejected_removal_uses_strikethrough(self):
        body = format_comment_body(
            _suggestion(add=[], remove=["stale", "old"]),
            applied_add=[],
            applied_remove=["old"],
            model="m",
            version="v",
            allow_label_removals=True,
        )
        self.assertIn("  - `old`", body)
        self.assertIn("  - ~~`stale`~~ (rejected by operator)", body)

    def test_comment_addition_labels_sorted_alphabetically(self):
        body = format_comment_body(
            _suggestion(add=["zzz", "aaa"]),
            applied_add=["zzz", "aaa"],
            applied_remove=[],
            model="m",
            version="v",
            allow_label_removals=False,
        )
        self.assertLess(body.index("aaa"), body.index("zzz"))

    def test_comment_starts_with_heading_and_ends_with_newline(self):
        body = format_comment_body(
            _suggestion(add=["bug"]),
            applied_add=["bug"],
            applied_remove=[],
            model="m",
            version="v",
            allow_label_removals=False,
        )
        self.assertTrue(body.startswith("## "))
        self.assertTrue(body.endswith("\n"))

    def test_comment_accepted_issue_type_not_struck(self):
        body = format_comment_body(
            _suggestion(add=[], issue_type="Bug"),
            applied_add=[],
            applied_remove=[],
            model="m",
            version="v",
            allow_label_removals=False,
            applied_issue_type="Bug",
        )
        self.assertIn("**Suggested issue type:** `Bug`", body)
        self.assertNotIn("rejected by operator", body)

    def test_comment_rejected_issue_type_struck(self):
        body = format_comment_body(
            _suggestion(add=[], issue_type="Bug"),
            applied_add=[],
            applied_remove=[],
            model="m",
            version="v",
            allow_label_removals=False,
            applied_issue_type=None,
        )
        self.assertIn("~~`Bug`~~", body)
        self.assertIn("(rejected by operator)", body)

    def test_comment_omits_issue_type_when_none(self):
        body = format_comment_body(
            _suggestion(add=["bug"]),
            applied_add=["bug"],
            applied_remove=[],
            model="m",
            version="v",
            allow_label_removals=False,
        )
        self.assertNotIn("**Suggested issue type:**", body)
