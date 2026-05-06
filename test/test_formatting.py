"""Tests for human-facing formatting helpers."""
# pylint: disable=missing-function-docstring

import sys
import unittest
from unittest import mock

from test.helpers import make_item
from ai_labelling.formatting import (
    colourise_inline_markdown,
    describe_match_bucket,
    format_body_preview,
    format_body_preview_colourised,
    format_display_timestamp,
    format_label_block,
    format_reason,
    print_dry_run_summary,
    print_exception_diagnostics,
    print_item_details,
    print_match_summary,
    print_prompt_help,
    summarise_body,
)
from ai_labelling.models import LabelSuggestion, SuggestionResult


class TimestampAndBucketTests(unittest.TestCase):
    """Cover timestamp formatting and bucket description helpers."""

    def test_format_display_timestamp_contains_date(self):
        """The formatted timestamp should at minimum contain the date."""

        result = format_display_timestamp("2026-05-01T12:30:00Z")
        # 12:30 UTC is far from midnight so the date is stable in any tz
        self.assertIn("2026-05-01", result)
        self.assertRegex(result, r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}")

    def test_describe_match_bucket_returns_items_for_empty_list(self):
        self.assertEqual(describe_match_bucket([]), "items")

    def test_describe_match_bucket_returns_state_and_kind(self):
        items = [make_item(1, "One")]
        self.assertEqual(describe_match_bucket(items), "open issues")

    def test_describe_match_bucket_uses_pr_label(self):
        items = [make_item(1, "One", kind="pr", state="closed")]
        self.assertEqual(describe_match_bucket(items), "closed PRs")


class PrintHelpersTests(unittest.TestCase):
    """Cover print_item_details and print_exception_diagnostics."""

    def test_print_item_details_outputs_title_and_author(self):
        item = make_item(42, "Fix the bug", labels=["bug"])
        with mock.patch("builtins.print") as print_mock:
            print_item_details(item)
        printed = "\n".join(
            str(c.args[0])
            for c in print_mock.call_args_list
            if c.args
        )
        self.assertIn("Fix the bug", printed)
        self.assertIn("#42", printed)
        self.assertIn("octocat", printed)

    def test_print_dry_run_summary_lists_add_labels(self):
        results = [
            SuggestionResult(
                item=make_item(7, "Fix crash", labels=[]),
                label_suggestion=LabelSuggestion(
                    add_labels=["bug", "crash"],
                    remove_labels=[],
                    reason="",
                ),
            ),
        ]
        with mock.patch("builtins.print") as print_mock:
            print_dry_run_summary(results, allow_label_removals=False)

        printed = "\n".join(
            str(c.args[0])
            for c in print_mock.call_args_list
            if c.args
        )
        self.assertIn("Fix crash", printed)
        self.assertIn("#7", printed)
        self.assertIn("+bug", printed)
        self.assertIn("+crash", printed)

    def test_print_dry_run_summary_shows_removals_when_allowed(self):
        results = [
            SuggestionResult(
                item=make_item(8, "Clean up"),
                label_suggestion=LabelSuggestion(
                    add_labels=[],
                    remove_labels=["stale"],
                    reason="",
                ),
            ),
        ]
        with mock.patch("builtins.print") as print_mock:
            print_dry_run_summary(results, allow_label_removals=True)

        printed = "\n".join(
            str(c.args[0])
            for c in print_mock.call_args_list
            if c.args
        )
        self.assertIn("-stale", printed)

    def test_print_dry_run_summary_hides_removals_when_disabled(self):
        results = [
            SuggestionResult(
                item=make_item(9, "Old issue"),
                label_suggestion=LabelSuggestion(
                    add_labels=[],
                    remove_labels=["stale"],
                    reason="",
                ),
            ),
        ]
        with mock.patch("builtins.print") as print_mock:
            print_dry_run_summary(results, allow_label_removals=False)

        printed = "\n".join(
            str(c.args[0])
            for c in print_mock.call_args_list
            if c.args
        )
        self.assertNotIn("-stale", printed)
        self.assertIn("no changes suggested", printed)

    def test_print_exception_diagnostics_writes_context_to_stderr(self):
        exc = RuntimeError("something failed")
        with mock.patch("builtins.print") as print_mock:
            with mock.patch("traceback.print_exception"):
                print_exception_diagnostics(exc, "AI suggestion")
        print_mock.assert_called_once()
        self.assertIs(
            print_mock.call_args.kwargs.get("file"), sys.stderr
        )
        self.assertIn("AI suggestion", str(print_mock.call_args.args[0]))


class FormattingTests(unittest.TestCase):
    """Verify human-facing item and reason formatting helpers."""

    def test_summarise_body_prefers_first_sentences(self):
        body = (
            "First sentence. Second sentence. Third sentence. "
            "Fourth sentence.\n\nLater paragraph."
        )
        self.assertEqual(
            summarise_body(body),
            "First sentence. Second sentence. Third sentence.",
        )

    def test_summarise_body_skips_markdown_heading_paragraph(self):
        body = "## Summary\n\nActual first paragraph. More detail follows."
        self.assertEqual(
            summarise_body(body),
            "Actual first paragraph. More detail follows.",
        )

    def test_format_body_preview_wraps_and_truncates(self):
        body = "\n\n".join(
            [
                "## Summary",
                (
                    "This is a shorter first line that still wraps a bit for "
                    "preview readability."
                ),
                (
                    "Second paragraph line one.\n"
                    "Second paragraph line two should stay on its own source "
                    "line before wrapping."
                ),
            ]
        )

        result = format_body_preview(body, width=40, max_lines=5)
        self.assertIn("## Summary", result)
        self.assertIn("This is a shorter first line", result)
        self.assertIn("Second paragraph line one.", result)

    def test_format_body_preview_shows_heading_without_counting_it(self):
        body = "## Summary\n\nFirst line.\nSecond line."
        result = format_body_preview(body, width=80, max_lines=1)
        self.assertIn("## Summary", result)
        self.assertIn("First line...", result)

    def test_format_body_preview_preserves_code_block_newlines(self):
        body = (
            "## Summary\n\n"
            "TSVC `s352` is a 5-wide unrolled dot product:\n\n"
            "```c\n"
            "dot = 0.;\n"
            "for (i = 0; i < LEN_1D; i += 5) dot = dot + a[i]*b[i];\n"
            "```"
        )
        result = format_body_preview(body, width=120, max_lines=8)
        self.assertIn("```c\ndot = 0.;", result)

    def test_format_body_preview_shows_code_block_without_counting_it(self):
        body = (
            "```c\n"
            "dot = 0.;\n"
            "for (i = 0; i < LEN_1D; i += 5) dot += a[i];\n"
            "```\n\n"
            "Intro line.\n\n"
            "Closing sentence."
        )
        result = format_body_preview(body, width=120, max_lines=1)
        self.assertIn("```c\ndot = 0.;", result)
        self.assertIn("Intro line.", result)
        self.assertNotIn("Closing sentence.", result)

    def test_format_body_preview_shows_quotes_without_counting_them(self):
        body = (
            "> Warning\n> Keep this in mind.\n\n"
            "Intro line.\n\nClosing sentence."
        )
        result = format_body_preview(body, width=120, max_lines=1)
        self.assertIn("> Warning\n> Keep this in mind.", result)
        self.assertIn("Intro line.", result)
        self.assertNotIn("Closing sentence.", result)

    def test_format_body_preview_shows_standalone_code_line(self):
        body = "`dot = 0.;`\n\nIntro line.\n\nClosing sentence."
        result = format_body_preview(body, width=120, max_lines=1)
        self.assertIn("`dot = 0.;`", result)
        self.assertIn("Intro line.", result)
        self.assertNotIn("Closing sentence.", result)

    def test_format_label_block_uses_bullets(self):
        self.assertEqual(
            format_label_block(["bug", "area:docs,api"]),
            "  - bug\n  - area:docs,api",
        )

    def test_format_reason_wraps_and_indents(self):
        result = format_reason(
            "This is a fairly long explanation that should wrap over "
            "multiple lines for readability in the terminal."
        )
        self.assertTrue(result.startswith("  This is"))
        self.assertIn("\n  ", result)

    def test_print_prompt_help_supports_item_mode(self):
        with mock.patch("builtins.print") as print_mock:
            print_prompt_help(False)

        printed = "\n".join(
            call.args[0] for call in print_mock.call_args_list if call.args
        )
        self.assertIn("handle this item", printed)
        self.assertIn("stop prompting more items", printed)

    def test_print_prompt_help_supports_apply_mode(self):
        with mock.patch("builtins.print") as print_mock:
            print_prompt_help(True)

        printed = "\n".join(
            call.args[0] for call in print_mock.call_args_list if call.args
        )
        self.assertIn("use this and all remaining", printed)
        self.assertIn("remaining labels in this action", printed)

    def test_print_match_summary_uses_single_line_for_one_bucket(self):
        item = make_item(1, "One")
        with mock.patch("builtins.print") as print_mock:
            print_match_summary([item])

        printed = [call.args[0] for call in print_mock.call_args_list]
        self.assertEqual(printed, ["Matched open issues: 1"])

    def test_print_match_summary_lists_multiple_buckets(self):
        open_issue = make_item(1, "Issue")
        closed_pr = make_item(2, "PR", kind="pr", state="closed")

        with mock.patch("builtins.print") as print_mock:
            print_match_summary([open_issue, closed_pr])

        printed = [call.args[0] for call in print_mock.call_args_list]
        self.assertEqual(
            printed,
            [
                "Matched items:",
                "  - Open issues: 1",
                "  - Closed PRs: 1",
            ],
        )


class InlineMarkdownColourisationTests(unittest.TestCase):
    """Verify Markdown stripping and colour assignment for inline spans."""

    # Tests run without a TTY so colourise() returns plain text.
    # We verify that formatting markers are stripped from the output.

    def test_bold_markers_stripped(self):
        self.assertEqual(
            colourise_inline_markdown("**bold** text"),
            "bold text",
        )

    def test_italic_star_markers_stripped(self):
        self.assertEqual(
            colourise_inline_markdown("*italic* text"),
            "italic text",
        )

    def test_italic_underscore_markers_stripped(self):
        self.assertEqual(
            colourise_inline_markdown("_italic_ text"),
            "italic text",
        )

    def test_bold_italic_markers_stripped(self):
        self.assertEqual(
            colourise_inline_markdown("***both*** text"),
            "both text",
        )

    def test_inline_code_backticks_stripped(self):
        self.assertEqual(
            colourise_inline_markdown("call `make` to build"),
            "call make to build",
        )

    def test_mixed_spans_all_stripped(self):
        self.assertEqual(
            colourise_inline_markdown("**bold** and *italic* text"),
            "bold and italic text",
        )

    def test_plain_line_unchanged(self):
        self.assertEqual(
            colourise_inline_markdown("plain text line"),
            "plain text line",
        )

    def test_empty_line_unchanged(self):
        self.assertEqual(colourise_inline_markdown(""), "")

    def test_tty_bold_uses_red(self):
        """Bold markers should produce red ANSI output on a TTY."""
        tty = mock.Mock()
        tty.isatty.return_value = True
        with mock.patch("ai_labelling.formatting.colourise") as col_mock:
            col_mock.side_effect = lambda t, c, **kw: f"[{c}:{t}]"
            result = colourise_inline_markdown("**bold** text")
        self.assertIn("[red:bold]", result)
        self.assertIn("[white: text]", result)

    def test_tty_italic_uses_yellow(self):
        """Italic markers should produce yellow ANSI output on a TTY."""
        with mock.patch("ai_labelling.formatting.colourise") as col_mock:
            col_mock.side_effect = lambda t, c, **kw: f"[{c}:{t}]"
            result = colourise_inline_markdown("*italic*")
        self.assertIn("[yellow:italic]", result)

    def test_tty_inline_code_uses_green(self):
        """Inline code backticks should produce green ANSI output on a TTY."""
        with mock.patch("ai_labelling.formatting.colourise") as col_mock:
            col_mock.side_effect = lambda t, c, **kw: f"[{c}:{t}]"
            result = colourise_inline_markdown("`make`")
        self.assertIn("[green:make]", result)


class ColourisedBodyPreviewTests(unittest.TestCase):
    """Verify colourised body preview strips markers and handles blocks."""

    def test_atx_heading_marker_stripped(self):
        result = format_body_preview_colourised(
            "## Summary\n\nSome text.", width=80, max_lines=4
        )
        self.assertNotIn("##", result)
        self.assertIn("Summary", result)
        self.assertIn("Some text.", result)

    def test_setext_heading_underline_stripped(self):
        result = format_body_preview_colourised(
            "My Heading\n==========\n\nBody text.", width=80, max_lines=4
        )
        self.assertNotIn("==========", result)
        self.assertIn("My Heading", result)

    def test_bold_markers_stripped_in_body(self):
        result = format_body_preview_colourised(
            "Contains **bold** text.", width=80, max_lines=4
        )
        self.assertNotIn("**", result)
        self.assertIn("bold", result)

    def test_fenced_code_fences_stripped(self):
        result = format_body_preview_colourised(
            "```c\nint x = 0;\n```", width=80, max_lines=4
        )
        self.assertNotIn("```", result)
        self.assertIn("int x = 0;", result)

    def test_standalone_code_backticks_stripped(self):
        result = format_body_preview_colourised(
            "`x = 0;`\n\nLine two.", width=80, max_lines=4
        )
        self.assertNotIn("`", result)
        self.assertIn("x = 0;", result)

    def test_inline_code_in_text_backticks_stripped(self):
        result = format_body_preview_colourised(
            "Run `make` to build.", width=80, max_lines=4
        )
        self.assertNotIn("`", result)
        self.assertIn("make", result)

    def test_tty_heading_uses_reverse_video(self):
        """ATX headings should be rendered in reverse video on a TTY."""
        with mock.patch("ai_labelling.formatting.colourise") as col_mock:
            col_mock.side_effect = lambda t, c, **kw: f"[{c}:{t}]"
            result = format_body_preview_colourised(
                "## Title\n\nText.", width=80, max_lines=4
            )
        self.assertIn("[reverse:Title]", result)

    def test_tty_setext_heading_uses_reverse_video(self):
        """Setext headings should be rendered in reverse video on a TTY."""
        with mock.patch("ai_labelling.formatting.colourise") as col_mock:
            col_mock.side_effect = lambda t, c, **kw: f"[{c}:{t}]"
            result = format_body_preview_colourised(
                "Title\n=====\n\nText.", width=80, max_lines=4
            )
        self.assertIn("[reverse:Title]", result)
