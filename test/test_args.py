"""Tests for argument parsing and model/date helpers."""
# pylint: disable=missing-function-docstring

import argparse
import io
import re
import unittest
from contextlib import redirect_stderr
from unittest import mock
from datetime import datetime, timezone

from ai_labelling.args import (
    build_argument_parser,
    build_help_epilog,
    default_cutoff,
    format_help_epilog_entry,
    parse_args,
    parse_cutoff,
    parse_model_spec,
    parse_repo_arg,
    positive_int,
)


class DateHandlingTests(unittest.TestCase):
    """Check date parsing and formatting helpers."""

    def test_parse_cutoff_uses_local_midnight_for_plain_dates(self):
        parsed = parse_cutoff("2026-05-01")
        self.assertIsNotNone(parsed.tzinfo)

    def test_parse_cutoff_all_disables_date_filter(self):
        self.assertIsNone(parse_cutoff("all"))
        self.assertIsNone(parse_cutoff("0"))

    def test_parse_cutoff_all_case_insensitive(self):
        self.assertIsNone(parse_cutoff("ALL"))

    def test_default_cutoff_tracks_last_twenty_four_hours(self):
        now = datetime(2026, 5, 6, 12, 30, tzinfo=timezone.utc)
        self.assertEqual(
            default_cutoff(now),
            datetime(2026, 5, 5, 12, 30, tzinfo=timezone.utc),
        )

    def test_default_cutoff_uses_now_when_omitted(self):
        with mock.patch(
            "ai_labelling.args.datetime"
        ) as datetime_mock:
            fake_now = datetime(2026, 5, 6, tzinfo=timezone.utc)
            datetime_mock.now.return_value = fake_now
            datetime_mock.fromisoformat = datetime.fromisoformat
            result = default_cutoff()
        self.assertEqual(
            result, datetime(2026, 5, 5, tzinfo=timezone.utc)
        )

    def test_parse_cutoff_rejects_non_iso_string(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_cutoff("yesterday")

    def test_parse_cutoff_normalises_naive_datetime_to_utc(self):
        result = parse_cutoff("2026-05-01T12:00:00")
        self.assertIsNotNone(result)
        self.assertEqual(result.tzinfo, timezone.utc)


class ParserDefaultsTests(unittest.TestCase):
    """Check overall default values produced by the argument parser."""

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

    def test_comment_reason_defaults_to_false(self):
        args = build_argument_parser().parse_args([])
        self.assertFalse(args.comment_reason)

    def test_comment_reason_flag_sets_true(self):
        args = build_argument_parser().parse_args(["--comment-reason"])
        self.assertTrue(args.comment_reason)

    def test_id_defaults_to_none(self):
        args = build_argument_parser().parse_args([])
        self.assertIsNone(args.id)

    def test_id_flag_accepts_positive_integer(self):
        args = build_argument_parser().parse_args(["--id", "42"])
        self.assertEqual(args.id, 42)

    def test_id_flag_rejects_zero(self):
        with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
            build_argument_parser().parse_args(["--id", "0"])

    def test_id_flag_rejects_negative(self):
        with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
            build_argument_parser().parse_args(["--id", "-1"])

    def test_help_mentions_id_flag(self):
        help_text = build_argument_parser().format_help()
        self.assertIn("--id", help_text)

    def test_parse_args_uses_argv(self):
        with mock.patch("sys.argv", ["ai-labelling", "--dry-run"]):
            args = parse_args()
        self.assertTrue(args.dry_run)


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

    def test_positive_int_rejects_non_numeric(self):
        with self.assertRaises(ValueError):
            positive_int("not-a-number")

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

    def test_parse_repo_arg_rejects_too_many_slashes(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_repo_arg("owner/repo/extra")


class ModelSpecTests(unittest.TestCase):
    """Check model and reasoning-effort parsing."""

    def test_parser_defaults_to_updated_filtering(self):
        args = build_argument_parser().parse_args([])
        self.assertFalse(args.created)
        self.assertEqual(args.model, "codex:gpt-5.4-mini:low")

    def test_parser_created_flag_overrides_updated_default(self):
        args = build_argument_parser().parse_args(["--created"])
        self.assertTrue(args.created)

    def test_help_mentions_updated_flag(self):
        help_text = build_argument_parser().format_help()
        self.assertIn("--updated", help_text)
        self.assertIn("last update", help_text)

    def test_parse_model_spec_with_provider_and_reasoning_suffix(self):
        result = parse_model_spec("codex:gpt-5.4:low")
        self.assertEqual(result.provider, "codex")
        self.assertEqual(result.model, "gpt-5.4")
        self.assertEqual(result.reasoning_effort, "low")

    def test_parse_model_spec_with_provider_default_model_wildcard(self):
        result = parse_model_spec("codex:*:low")
        self.assertEqual(result.provider, "codex")
        self.assertIsNone(result.model)
        self.assertEqual(result.reasoning_effort, "low")

    def test_parse_model_spec_accepts_anthropic_provider_only(self):
        result = parse_model_spec("anthropic")
        self.assertEqual(result.provider, "anthropic")
        self.assertEqual(result.model, "claude-haiku-4-5-20251001")
        self.assertIsNone(result.reasoning_effort)

    def test_parse_model_spec_accepts_anthropic_with_wildcard(self):
        result = parse_model_spec("anthropic:*:low")
        self.assertEqual(result.provider, "anthropic")
        self.assertIsNone(result.model)
        self.assertEqual(result.reasoning_effort, "low")

    def test_parse_model_spec_accepts_anthropic_max_effort(self):
        result = parse_model_spec("anthropic:*:max")
        self.assertEqual(result.provider, "anthropic")
        self.assertEqual(result.reasoning_effort, "max")

    def test_parse_model_spec_rejects_codex_max_effort(self):
        with self.assertRaisesRegex(
            argparse.ArgumentTypeError,
            "unsupported reasoning effort for codex",
        ):
            parse_model_spec("codex:gpt-5.4:max")

    def test_parse_model_spec_rejects_unsupported_provider(self):
        with self.assertRaisesRegex(
            argparse.ArgumentTypeError,
            "myAI is not supported",
        ):
            parse_model_spec("myAI:gpt-5.4:low")

    def test_parse_model_spec_accepts_provider_only(self):
        result = parse_model_spec("codex")
        self.assertEqual(result.provider, "codex")
        self.assertEqual(result.model, "gpt-5.4-mini")
        self.assertIsNone(result.reasoning_effort)

    def test_parse_model_spec_accepts_provider_and_model(self):
        result = parse_model_spec("codex:gpt-5.4")
        self.assertEqual(result.provider, "codex")
        self.assertEqual(result.model, "gpt-5.4")
        self.assertIsNone(result.reasoning_effort)

    def test_parse_model_spec_rejects_invalid_provider_with_colon(self):
        with self.assertRaisesRegex(
            argparse.ArgumentTypeError,
            "is not supported",
        ):
            parse_model_spec("myai:model")

    def test_parse_model_spec_rejects_invalid_shape(self):
        with self.assertRaisesRegex(
            argparse.ArgumentTypeError,
            "PROVIDER, PROVIDER:MODEL, or PROVIDER:MODEL:REASONING",
        ):
            parse_model_spec("codex:gpt-5.4:low:extra")

    def test_parse_model_spec_rejects_empty_model_name(self):
        with self.assertRaisesRegex(
            argparse.ArgumentTypeError, "model name must not be empty"
        ):
            parse_model_spec("codex:")


class HelpEpilogTests(unittest.TestCase):
    """Check the extra ``--help`` epilog content."""

    def test_help_epilog_describes_debug_levels_and_ai_providers(self):
        _ansi = re.compile(r"\x1b\[[0-9;]*m")
        help_text = _ansi.sub("", build_argument_parser().format_help())
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

    def test_format_help_epilog_entry_returns_indented_line(self):
        with mock.patch(
            "ai_labelling.args.colourise",
            side_effect=lambda text, *_a, **_kw: text,
        ):
            line = format_help_epilog_entry("KEY", "what it does")
        self.assertEqual(line, "  KEY: what it does")
