"""Tests for argument parsing and model/date helpers."""
# pylint: disable=missing-function-docstring

import argparse
import unittest
from unittest import mock
from datetime import datetime, timezone

from ai_labelling import format_search_date
from ai_labelling.args import (
    build_argument_parser,
    build_help_epilog,
    default_cutoff,
    parse_cutoff,
    parse_model_spec,
    parse_repo_arg,
    positive_int,
)


class DateHandlingTests(unittest.TestCase):
    """Check date parsing and formatting helpers."""

    def test_parse_cutoff_uses_local_midnight_for_plain_dates(self):
        """Plain dates should become timezone-aware datetimes."""

        parsed = parse_cutoff("2026-05-01")
        self.assertIsNotNone(parsed.tzinfo)

    def test_parse_cutoff_all_disables_date_filter(self):
        """``--date all`` and ``--date 0`` should remove the cutoff."""

        self.assertIsNone(parse_cutoff("all"))
        self.assertIsNone(parse_cutoff("0"))

    def test_default_cutoff_tracks_last_twenty_four_hours(self):
        """The implicit cutoff should be 24 hours before runtime."""

        now = datetime(2026, 5, 6, 12, 30, tzinfo=timezone.utc)
        self.assertEqual(
            default_cutoff(now),
            datetime(2026, 5, 5, 12, 30, tzinfo=timezone.utc),
        )

    def test_format_search_date_uses_utc_day(self):
        """GitHub search dates should use the UTC calendar day."""

        timestamp = datetime(2026, 5, 1, 23, 45, tzinfo=timezone.utc)
        self.assertEqual(format_search_date(timestamp), "2026-05-01")

    def test_parse_cutoff_rejects_non_iso_string(self):
        """Completely invalid date strings should raise an error."""

        with self.assertRaises(argparse.ArgumentTypeError):
            parse_cutoff("yesterday")

    def test_parse_cutoff_normalises_naive_datetime_to_utc(self):
        """Naive date-time strings should be treated as local and shifted."""

        result = parse_cutoff("2026-05-01T12:00:00")
        self.assertIsNotNone(result)
        self.assertEqual(result.tzinfo, timezone.utc)


class DryRunFlagTests(unittest.TestCase):
    """Check the --dry-run argument."""

    def test_dry_run_defaults_to_false(self):
        args = build_argument_parser().parse_args([])
        self.assertFalse(args.dry_run)

    def test_dry_run_flag_sets_true(self):
        args = build_argument_parser().parse_args(["--dry-run"])
        self.assertTrue(args.dry_run)

    def test_dry_run_and_force_are_independent(self):
        args = build_argument_parser().parse_args(["--dry-run", "--force"])
        self.assertTrue(args.dry_run)
        self.assertTrue(args.force)


class CommentReasonFlagTests(unittest.TestCase):
    """Check the --comment-reason argument."""

    def test_comment_reason_defaults_to_false(self):
        args = build_argument_parser().parse_args([])
        self.assertFalse(args.comment_reason)

    def test_comment_reason_flag_sets_true(self):
        args = build_argument_parser().parse_args(["--comment-reason"])
        self.assertTrue(args.comment_reason)


class PositiveIntAndRepoArgTests(unittest.TestCase):
    """Check helper argument-type validators."""

    def test_positive_int_accepts_positive_value(self):
        self.assertEqual(positive_int("5"), 5)

    def test_positive_int_rejects_zero(self):
        with self.assertRaisesRegex(
            argparse.ArgumentTypeError, "positive integer"
        ):
            positive_int("0")

    def test_positive_int_rejects_negative(self):
        with self.assertRaisesRegex(
            argparse.ArgumentTypeError, "positive integer"
        ):
            positive_int("-3")

    def test_parse_repo_arg_accepts_owner_slash_repo(self):
        self.assertEqual(
            parse_repo_arg("llvm/llvm-project"),
            "llvm/llvm-project",
        )

    def test_parse_repo_arg_rejects_missing_slash(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_repo_arg("noslash")

    def test_parse_repo_arg_rejects_empty_owner(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_repo_arg("/repo")

    def test_parse_repo_arg_rejects_empty_repo(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_repo_arg("owner/")


class ModelSpecTests(unittest.TestCase):
    """Check model and reasoning-effort parsing."""

    def test_parser_defaults_to_updated_filtering(self):
        """The CLI should default to update-time filtering."""

        args = build_argument_parser().parse_args([])
        self.assertFalse(args.created)
        self.assertEqual(args.model, "codex:gpt-5.4-mini:low")

    def test_parser_created_flag_overrides_updated_default(self):
        """``--created`` should switch the cutoff field used by ``--date``."""

        args = build_argument_parser().parse_args(["--created"])
        self.assertTrue(args.created)

    def test_help_mentions_updated_flag(self):
        """``--help`` should describe the explicit updated-time switch."""

        help_text = build_argument_parser().format_help()
        self.assertIn("--updated", help_text)
        self.assertIn("last update", help_text)

    def test_parse_model_spec_with_provider_and_reasoning_suffix(self):
        """Known provider/model/reasoning specs should parse cleanly."""

        result = parse_model_spec("codex:gpt-5.4:low")
        self.assertEqual(result.provider, "codex")
        self.assertEqual(result.model, "gpt-5.4")
        self.assertEqual(result.reasoning_effort, "low")

    def test_parse_model_spec_with_provider_default_model_wildcard(self):
        """The ``*`` model selector should defer to the provider default."""

        result = parse_model_spec("codex:*:low")
        self.assertEqual(result.provider, "codex")
        self.assertIsNone(result.model)
        self.assertEqual(result.reasoning_effort, "low")

    def test_parse_model_spec_accepts_anthropic_provider_only(self):
        """Anthropic provider-only should use hard-coded default model."""

        result = parse_model_spec("anthropic")
        self.assertEqual(result.provider, "anthropic")
        self.assertEqual(result.model, "claude-haiku-4-5-20251001")
        self.assertIsNone(result.reasoning_effort)

    def test_parse_model_spec_accepts_anthropic_with_wildcard(self):
        """Anthropic with wildcard should defer to API for dynamic default."""

        result = parse_model_spec("anthropic:*:low")
        self.assertEqual(result.provider, "anthropic")
        self.assertIsNone(result.model)
        self.assertEqual(result.reasoning_effort, "low")

    def test_parse_model_spec_accepts_anthropic_max_effort(self):
        """Anthropic should accept the extra ``max`` effort level."""

        result = parse_model_spec("anthropic:*:max")
        self.assertEqual(result.provider, "anthropic")
        self.assertEqual(result.reasoning_effort, "max")

    def test_parse_model_spec_rejects_codex_max_effort(self):
        """Codex should reject Anthropic-only effort levels."""

        with self.assertRaisesRegex(
            argparse.ArgumentTypeError,
            "unsupported reasoning effort for codex",
        ):
            parse_model_spec("codex:gpt-5.4:max")

    def test_parse_model_spec_rejects_unsupported_provider(self):
        """Unknown providers should fail with a clear error."""

        with self.assertRaisesRegex(
            argparse.ArgumentTypeError,
            "myAI is not supported",
        ):
            parse_model_spec("myAI:gpt-5.4:low")

    def test_parse_model_spec_accepts_provider_only(self):
        """Provider-only specs use provider's hard-coded default model."""

        result = parse_model_spec("codex")
        self.assertEqual(result.provider, "codex")
        self.assertEqual(result.model, "gpt-5.4-mini")
        self.assertIsNone(result.reasoning_effort)

    def test_parse_model_spec_accepts_provider_and_model(self):
        """Provider:model specs omit reasoning effort when not specified."""

        result = parse_model_spec("codex:gpt-5.4")
        self.assertEqual(result.provider, "codex")
        self.assertEqual(result.model, "gpt-5.4")
        self.assertIsNone(result.reasoning_effort)

    def test_parse_model_spec_rejects_invalid_provider_with_colon(self):
        """Invalid provider in two-part spec should fail cleanly."""

        with self.assertRaisesRegex(
            argparse.ArgumentTypeError,
            "is not supported",
        ):
            parse_model_spec("myai:model")

    def test_parse_model_spec_rejects_invalid_shape(self):
        """Model specs with too many colons should fail."""

        with self.assertRaisesRegex(
            argparse.ArgumentTypeError,
            "PROVIDER, PROVIDER:MODEL, or PROVIDER:MODEL:REASONING",
        ):
            parse_model_spec("codex:gpt-5.4:low:extra")

    def test_help_epilog_describes_debug_levels_and_ai_providers(self):
        """``--help`` should document debug levels and supported providers."""

        help_text = build_argument_parser().format_help()
        self.assertIn(
            "DEBUG=1: show executed subprocess command lines",
            help_text,
        )
        self.assertIn(
            "DEBUG=2: also show a sanitised AI prompt template",
            help_text,
        )
        self.assertIn(
            "DEBUG=3 or greater: also show the full AI prompt",
            help_text,
        )
        self.assertIn("AI providers:", help_text)
        self.assertIn("codex: the `codex` CLI", help_text)
        self.assertIn("gpt-5.4-mini", help_text)
        self.assertIn("anthropic:", help_text)
        self.assertIn("claude-haiku-4-5-20251001", help_text)
        self.assertIn("max", help_text)

    def test_help_epilog_uses_help_like_colours_on_tty(self):
        """TTY help output should colour section titles and epilog keys."""

        with mock.patch(
            "ai_labelling.args.colourise",
            side_effect=lambda *a, **k: a[0],
        ):
            epilog = build_help_epilog()

        self.assertIn("Debugging:", epilog)
        with mock.patch(
            "ai_labelling.args.colourise",
            side_effect=lambda *a, **k: a[0],
        ) as colourise_mock:
            build_help_epilog()
        used_colours = [
            call.args[1] for call in colourise_mock.call_args_list
        ]
        self.assertIn("magenta", used_colours)
        self.assertIn("cyan", used_colours)
