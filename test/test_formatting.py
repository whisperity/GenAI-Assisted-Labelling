"""Tests for human-facing formatting helpers."""
# pylint: disable=missing-function-docstring,duplicate-code

import sys
import unittest
from unittest import mock

from test.helpers import make_item
from ai_labelling.formatting import (
    bucketise_items,
    classify_preview_block,
    colourise_inline_markdown,
    describe_match_bucket,
    format_body_preview,
    format_body_preview_colourised,
    format_display_timestamp,
    format_label_block,
    format_reason,
    non_empty_lines,
    parse_preview_blocks,
    print_changes_summary,
    print_exception_diagnostics,
    print_item_details,
    print_match_summary,
    print_matching_items,
    summarise_body,
    take_non_empty_lines,
    wrap_preserving_newlines,
)
from ai_labelling.models import LabelSuggestion, SuggestionResult


class TimestampAndBucketTests(unittest.TestCase):
    """Cover timestamp formatting and bucket description helpers."""

    def test_format_display_timestamp_contains_date(self):
        result = format_display_timestamp("2026-05-01T12:30:00Z")
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

    def test_bucketise_items_groups_by_state_and_kind(self):
        items = [
            make_item(1, "A"),
            make_item(2, "B", state="closed"),
            make_item(3, "C", kind="pr"),
        ]
        result = bucketise_items(items)
        self.assertIn("open issues", result)
        self.assertIn("closed issues", result)
        self.assertIn("open PRs", result)


class PreviewBlockTests(unittest.TestCase):
    """Verify markdown block parsing and classification."""

    def test_classify_empty_returns_empty(self):
        self.assertEqual(classify_preview_block(""), "empty")

    def test_classify_quote_block(self):
        self.assertEqual(
            classify_preview_block("> warning\n> next"), "quote"
        )

    def test_classify_heading_block(self):
        self.assertEqual(classify_preview_block("# Title"), "heading")

    def test_classify_inline_code_line(self):
        self.assertEqual(classify_preview_block("`x = 1`"), "code")

    def test_classify_fenced_code_block(self):
        self.assertEqual(classify_preview_block("```\ncode\n```"), "code")

    def test_parse_preview_blocks_handles_empty_body(self):
        blocks = parse_preview_blocks("")
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].text, "(no description)")

    def test_parse_preview_blocks_separates_paragraphs(self):
        blocks = parse_preview_blocks("First.\n\nSecond.")
        self.assertEqual([b.text for b in blocks], ["First.", "Second."])


class WrapAndCountTests(unittest.TestCase):
    """Verify text-wrapping and line-counting helpers."""

    def test_wrap_preserving_newlines_keeps_blank_lines(self):
        result = wrap_preserving_newlines("a\n\nb", width=10)
        self.assertEqual(result, ["a", "", "b"])

    def test_wrap_preserving_newlines_wraps_long_line(self):
        result = wrap_preserving_newlines("a b c d e f", width=4)
        self.assertGreater(len(result), 1)

    def test_non_empty_lines_filters_blanks(self):
        self.assertEqual(non_empty_lines(["", "a", " ", "b"]), ["a", "b"])

    def test_take_non_empty_lines_stops_at_limit(self):
        result = take_non_empty_lines(["a", "", "b", "c"], 2)
        self.assertEqual(result, ["a", "", "b"])

    def test_take_non_empty_lines_strips_trailing_blanks(self):
        result = take_non_empty_lines(["a", ""], 5)
        self.assertEqual(result, ["a"])


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

    def test_print_matching_items_lists_titles(self):
        items = [
            make_item(1, "Open one"),
            make_item(2, "Closed two", state="closed"),
        ]
        with mock.patch("builtins.print") as print_mock:
            print_matching_items(items, "Heading")
        printed = "\n".join(
            str(c.args[0]) for c in print_mock.call_args_list if c.args
        )
        self.assertIn("Heading:", printed)
        self.assertIn("Open one", printed)
        self.assertIn("Closed two", printed)

    def test_print_changes_summary_dry_run_lists_add_labels(self):
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
            print_changes_summary(results, allow_label_removals=False)

        printed = "\n".join(
            str(c.args[0])
            for c in print_mock.call_args_list
            if c.args
        )
        self.assertIn("Label changes (not applied):", printed)
        self.assertIn("Fix crash", printed)
        self.assertIn("#7", printed)
        self.assertIn("+ bug", printed)
        self.assertIn("+ crash", printed)

    def test_print_changes_summary_non_dry_run_uses_label_changes_title(self):
        results = [
            SuggestionResult(
                item=make_item(7, "Fix crash", labels=[]),
                label_suggestion=LabelSuggestion(
                    add_labels=["bug"],
                    remove_labels=[],
                    reason="",
                ),
            ),
        ]
        with mock.patch("builtins.print") as print_mock:
            print_changes_summary(
                results, allow_label_removals=False, dry_run=False
            )

        printed = "\n".join(
            str(c.args[0])
            for c in print_mock.call_args_list
            if c.args
        )
        self.assertIn("Label changes:", printed)
        self.assertNotIn("not applied", printed)

    def test_print_changes_summary_shows_removals_when_allowed(self):
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
            print_changes_summary(results, allow_label_removals=True)

        printed = "\n".join(
            str(c.args[0])
            for c in print_mock.call_args_list
            if c.args
        )
        self.assertIn("- stale", printed)

    def test_print_changes_summary_hides_removals_when_disabled(self):
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
            print_changes_summary(results, allow_label_removals=False)

        printed = "\n".join(
            str(c.args[0])
            for c in print_mock.call_args_list
            if c.args
        )
        self.assertNotIn("stale", printed)

    def test_print_changes_summary_skips_items_with_no_changes(self):
        results = [
            SuggestionResult(
                item=make_item(9, "Old issue"),
                label_suggestion=LabelSuggestion(
                    add_labels=[],
                    remove_labels=[],
                    reason="",
                ),
            ),
        ]
        with mock.patch("builtins.print") as print_mock:
            print_changes_summary(results, allow_label_removals=True)

        printed = "\n".join(
            str(c.args[0])
            for c in print_mock.call_args_list
            if c.args
        )
        self.assertNotIn("Old issue", printed)

    def test_print_changes_summary_indents_item_and_labels(self):
        results = [
            SuggestionResult(
                item=make_item(5, "Add feature"),
                label_suggestion=LabelSuggestion(
                    add_labels=["enhancement"],
                    remove_labels=[],
                    reason="",
                ),
            ),
        ]
        with mock.patch("builtins.print") as print_mock:
            print_changes_summary(results, allow_label_removals=False)

        call_args = [
            str(c.args[0]) for c in print_mock.call_args_list if c.args
        ]
        item_line = next(s for s in call_args if "#5" in s)
        self.assertTrue(item_line.startswith("- "))

        label_line = next(s for s in call_args if "enhancement" in s)
        self.assertTrue(label_line.startswith("  "))

    def test_print_changes_summary_shows_issue_type(self):
        results = [
            SuggestionResult(
                item=make_item(1, "One"),
                label_suggestion=LabelSuggestion(
                    add_labels=[],
                    remove_labels=[],
                    reason="",
                    issue_type="Bug",
                ),
            )
        ]
        with mock.patch("builtins.print") as print_mock:
            print_changes_summary(results, allow_label_removals=False)
        printed = "\n".join(
            str(c.args[0]) for c in print_mock.call_args_list if c.args
        )
        self.assertIn("~ Bug", printed)

    def test_print_exception_diagnostics_writes_context_to_stderr(self):
        exc = RuntimeError("something failed")
        with mock.patch("builtins.print") as print_mock, \
                mock.patch("traceback.print_exception"):
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

    def test_summarise_body_truncates_long_text(self):
        body = "Long sentence " * 50
        result = summarise_body(body)
        self.assertLessEqual(len(result), 280)
        self.assertTrue(result.endswith("..."))

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

    def test_format_label_block_handles_empty_list(self):
        self.assertEqual(format_label_block([]), "  - (none)")

    def test_format_label_block_uses_bullets(self):
        self.assertEqual(
            format_label_block(["bug", "area:docs,api"]),
            "  - area:docs,api\n  - bug",
        )

    def test_format_reason_returns_empty_for_blank(self):
        self.assertEqual(format_reason("   "), "")

    def test_format_reason_wraps_and_indents(self):
        result = format_reason(
            "This is a fairly long explanation that should wrap over "
            "multiple lines for readability in the terminal."
        )
        self.assertTrue(result.startswith("  This is"))
        self.assertIn("\n  ", result)

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
        with mock.patch("ai_labelling.formatting.colourise") as col_mock:
            col_mock.side_effect = lambda t, c, **kw: f"[{c}:{t}]"
            result = colourise_inline_markdown("**bold** text")
        self.assertIn("[red:bold]", result)
        self.assertIn("[white: text]", result)

    def test_tty_italic_uses_yellow(self):
        with mock.patch("ai_labelling.formatting.colourise") as col_mock:
            col_mock.side_effect = lambda t, c, **kw: f"[{c}:{t}]"
            result = colourise_inline_markdown("*italic*")
        self.assertIn("[yellow:italic]", result)

    def test_tty_underscore_italic_uses_yellow(self):
        with mock.patch("ai_labelling.formatting.colourise") as col_mock:
            col_mock.side_effect = lambda t, c, **kw: f"[{c}:{t}]"
            result = colourise_inline_markdown("_italic_")
        self.assertIn("[yellow:italic]", result)

    def test_tty_bold_italic_uses_red(self):
        with mock.patch("ai_labelling.formatting.colourise") as col_mock:
            col_mock.side_effect = lambda t, c, **kw: f"[{c}:{t}]"
            result = colourise_inline_markdown("***both***")
        self.assertIn("[red:both]", result)

    def test_tty_inline_code_uses_green(self):
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

    def test_quote_block_rendered_with_marker(self):
        result = format_body_preview_colourised(
            "> warning", width=80, max_lines=4
        )
        self.assertIn(">", result)

    def test_tty_heading_uses_reverse_video(self):
        with mock.patch("ai_labelling.formatting.colourise") as col_mock:
            col_mock.side_effect = lambda t, c, **kw: f"[{c}:{t}]"
            result = format_body_preview_colourised(
                "## Title\n\nText.", width=80, max_lines=4
            )
        self.assertIn("[reverse:Title]", result)

    def test_tty_setext_heading_uses_reverse_video(self):
        with mock.patch("ai_labelling.formatting.colourise") as col_mock:
            col_mock.side_effect = lambda t, c, **kw: f"[{c}:{t}]"
            result = format_body_preview_colourised(
                "Title\n=====\n\nText.", width=80, max_lines=4
            )
        self.assertIn("[reverse:Title]", result)
