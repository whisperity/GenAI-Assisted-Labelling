#!/usr/bin/env python3
"""Unit tests for the ``ai-labelling`` script."""
# pylint: disable=too-many-lines

import argparse
import importlib.machinery
import importlib.util
import pathlib
import unittest
import unittest.mock
from datetime import datetime, timezone


def load_module():
    """Load the top-level ``ai-labelling`` script as an importable module."""

    # Tests live in ``test/``, so step up once to reach the script.
    script_path = (
        pathlib.Path(__file__).resolve().parent.parent / "ai-labelling"
    )
    loader = importlib.machinery.SourceFileLoader(
        "ai_labelling", str(script_path)
    )
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


ai_labelling = load_module()


class ParseGitHubRepoTests(unittest.TestCase):
    """Cover repository detection from supported remote URL formats."""

    def test_parse_ssh_remote(self):
        """Accept standard GitHub SSH remotes."""

        self.assertEqual(
            ai_labelling.parse_github_repo_from_remote(
                "git@github.com:llvm/llvm-project.git"
            ),
            "llvm/llvm-project",
        )

    def test_parse_short_ssh_remote(self):
        """Accept SSH-style remotes without the ``git@`` prefix."""

        self.assertEqual(
            ai_labelling.parse_github_repo_from_remote(
                "github.com:Whisperity/llvm-project.git"
            ),
            "Whisperity/llvm-project",
        )

    def test_parse_http_remote(self):
        """Accept HTTP GitHub remotes."""

        self.assertEqual(
            ai_labelling.parse_github_repo_from_remote(
                "http://github.com/llvm/llvm-project.git"
            ),
            "llvm/llvm-project",
        )

    def test_ignore_non_github_remote(self):
        """Reject remotes hosted outside GitHub."""

        self.assertIsNone(
            ai_labelling.parse_github_repo_from_remote(
                "https://gitlab.com/llvm/llvm-project.git"
            )
        )

    def test_detect_repo_falls_through_remotes_until_one_matches(self):
        """Repository detection should skip failing remotes and keep trying."""

        failures = unittest.mock.Mock(returncode=1, stdout="")
        success = unittest.mock.Mock(
            returncode=0,
            stdout="git@github.com:llvm/llvm-project.git\n",
        )

        with unittest.mock.patch.object(
            ai_labelling,
            "run",
            side_effect=[failures, success],
        ) as run_mock:
            repo = ai_labelling.detect_repo()

        self.assertEqual(repo, "llvm/llvm-project")
        self.assertEqual(run_mock.call_count, 2)


class DateHandlingTests(unittest.TestCase):
    """Check date parsing and formatting helpers."""

    def test_parse_cutoff_date_uses_local_midnight(self):
        """Plain dates should become timezone-aware datetimes."""

        parsed = ai_labelling.parse_cutoff("2026-05-01")
        self.assertIsNotNone(parsed.tzinfo)

    def test_parse_cutoff_all_disables_date_filter(self):
        """``--date all`` and ``--date 0`` should remove the cutoff."""

        self.assertIsNone(ai_labelling.parse_cutoff("all"))
        self.assertIsNone(ai_labelling.parse_cutoff("0"))

    def test_default_cutoff_tracks_last_twenty_four_hours(self):
        """The implicit cutoff should be 24 hours before runtime."""

        now = datetime(2026, 5, 6, 12, 30, tzinfo=timezone.utc)

        self.assertEqual(
            ai_labelling.default_cutoff(now),
            datetime(2026, 5, 5, 12, 30, tzinfo=timezone.utc),
        )

    def test_format_search_date_uses_utc_day(self):
        """GitHub search dates should use the UTC calendar day."""

        timestamp = datetime(2026, 5, 1, 23, 45, tzinfo=timezone.utc)
        self.assertEqual(
            ai_labelling.format_search_date(timestamp), "2026-05-01"
        )


class ModelSpecTests(unittest.TestCase):
    """Check model and reasoning-effort parsing."""

    def test_parser_defaults_to_updated_filtering(self):
        """The CLI should default to update-time filtering."""

        parser = ai_labelling.build_argument_parser()

        args = parser.parse_args([])

        self.assertFalse(args.created)
        self.assertEqual(args.model, "codex:gpt-5.4-mini:low")

    def test_parser_created_flag_overrides_updated_default(self):
        """``--created`` should switch the cutoff field used by ``--date``."""

        parser = ai_labelling.build_argument_parser()

        args = parser.parse_args(["--created"])

        self.assertTrue(args.created)

    def test_help_mentions_updated_flag(self):
        """``--help`` should describe the explicit updated-time switch."""

        help_text = ai_labelling.build_argument_parser().format_help()

        self.assertIn("--updated", help_text)
        self.assertIn("last update", help_text)

    def test_parse_model_spec_with_provider_and_reasoning_suffix(self):
        """Known provider/model/reasoning specs should parse cleanly."""

        result = ai_labelling.parse_model_spec("codex:gpt-5.4:low")

        self.assertEqual(result.provider, "codex")
        self.assertEqual(result.model, "gpt-5.4")
        self.assertEqual(result.reasoning_effort, "low")

    def test_parse_model_spec_with_provider_default_model_wildcard(self):
        """The ``*`` model selector should defer to the provider default."""

        result = ai_labelling.parse_model_spec("codex:*:low")

        self.assertEqual(result.provider, "codex")
        self.assertIsNone(result.model)
        self.assertEqual(result.reasoning_effort, "low")

    def test_parse_model_spec_accepts_anthropic_provider(self):
        """Anthropic should be accepted as a supported provider."""

        result = ai_labelling.parse_model_spec("anthropic:*:low")

        self.assertEqual(result.provider, "anthropic")
        self.assertIsNone(result.model)
        self.assertEqual(result.reasoning_effort, "low")

    def test_parse_model_spec_accepts_anthropic_max_effort(self):
        """Anthropic should accept the extra ``max`` effort level."""

        result = ai_labelling.parse_model_spec("anthropic:*:max")

        self.assertEqual(result.provider, "anthropic")
        self.assertEqual(result.reasoning_effort, "max")

    def test_parse_model_spec_rejects_codex_max_effort(self):
        """Codex should reject Anthropic-only effort levels."""

        with self.assertRaisesRegex(
            argparse.ArgumentTypeError,
            "unsupported reasoning effort for codex",
        ):
            ai_labelling.parse_model_spec("codex:gpt-5.4:max")

    def test_parse_model_spec_rejects_unsupported_provider(self):
        """Unknown providers should fail with a clear error."""

        with self.assertRaisesRegex(
            argparse.ArgumentTypeError,
            "myAI is not supported",
        ):
            ai_labelling.parse_model_spec("myAI:gpt-5.4:low")

    def test_parse_model_spec_rejects_invalid_shape(self):
        """Model specs must include provider, model, and reasoning."""

        with self.assertRaisesRegex(
            argparse.ArgumentTypeError,
            "PROVIDER:MODEL:REASONING",
        ):
            ai_labelling.parse_model_spec("gpt-5.4:low")

    def test_help_epilog_describes_debug_levels_and_ai_providers(self):
        """``--help`` should document debug levels and supported providers."""

        help_text = ai_labelling.build_argument_parser().format_help()

        self.assertIn(
            "DEBUG=1: show executed subprocess command lines", help_text
        )
        self.assertIn(
            "DEBUG=2: also show a sanitized AI prompt template", help_text
        )
        self.assertIn(
            "DEBUG=3 or greater: also show the full AI prompt", help_text
        )
        self.assertIn("AI providers:", help_text)
        self.assertIn("codex: the `codex` CLI", help_text)
        self.assertIn("gpt-5.4-mini", help_text)
        self.assertIn("anthropic:", help_text)
        self.assertIn("claude-haiku-4-5-20251001", help_text)
        self.assertIn("max", help_text)

    def test_help_epilog_uses_help_like_colors_on_tty(self):
        """TTY help output should color section titles and epilog keys."""

        with unittest.mock.patch.object(
            ai_labelling, "supports_color", return_value=True
        ):
            epilog = ai_labelling.build_help_epilog()

        self.assertIn(ai_labelling.ANSI_STYLES["magenta"], epilog)
        self.assertIn(ai_labelling.ANSI_STYLES["cyan"], epilog)


class QueryTests(unittest.TestCase):
    """Verify GitHub search query construction."""

    def test_build_search_query_open_updated(self):
        """Open-item searches should filter on update time by default."""

        cutoff = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)
        self.assertEqual(
            ai_labelling.build_search_query(
                "llvm/llvm-project",
                "issue",
                ai_labelling.SearchOptions(True, False, False, cutoff, None),
            ),
            "repo:llvm/llvm-project is:issue is:open updated:>=2026-05-01",
        )

    def test_build_search_query_closed_created(self):
        """Closed-only searches should use ``is:closed`` and creation time."""

        cutoff = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)
        self.assertEqual(
            ai_labelling.build_search_query(
                "llvm/llvm-project",
                "pr",
                ai_labelling.SearchOptions(False, True, True, cutoff, None),
            ),
            "repo:llvm/llvm-project is:pr is:closed created:>=2026-05-01",
        )

    def test_build_search_query_open_and_closed_created(self):
        """Including both states should omit any explicit state qualifier."""

        cutoff = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)
        self.assertEqual(
            ai_labelling.build_search_query(
                "llvm/llvm-project",
                "pr",
                ai_labelling.SearchOptions(True, True, True, cutoff, None),
            ),
            "repo:llvm/llvm-project is:pr created:>=2026-05-01",
        )

    def test_search_items_stops_at_limit(self):
        """Search should stop once the requested item limit is reached."""

        payload = {
            "items": [
                {
                    "number": 2,
                    "title": "Two",
                    "body": "Body two",
                    "state": "open",
                    "labels": [],
                    "html_url": "https://example.invalid/2",
                    "updated_at": "2026-05-02T00:00:00Z",
                    "created_at": "2026-05-02T00:00:00Z",
                    "user": {"login": "octocat"},
                },
                {
                    "number": 1,
                    "title": "One",
                    "body": "Body one",
                    "state": "open",
                    "labels": [],
                    "html_url": "https://example.invalid/1",
                    "updated_at": "2026-05-01T00:00:00Z",
                    "created_at": "2026-05-01T00:00:00Z",
                    "user": {"login": "octocat"},
                },
            ]
        }

        with unittest.mock.patch.object(
            ai_labelling,
            "gh_json",
            return_value=payload,
        ) as gh_mock:
            result = ai_labelling.search_items(
                "llvm/llvm-project",
                "issue",
                ai_labelling.SearchOptions(True, False, False, None, 1),
            )

        self.assertEqual([item.number for item in result], [2])
        gh_mock.assert_called_once()

    def test_search_items_stops_when_cutoff_is_crossed(self):
        """Older results should stop the paginated search immediately."""

        payload = {
            "items": [
                {
                    "number": 2,
                    "title": "Newer",
                    "body": "Body two",
                    "state": "open",
                    "labels": [],
                    "html_url": "https://example.invalid/2",
                    "updated_at": "2026-05-03T00:00:00Z",
                    "created_at": "2026-05-03T00:00:00Z",
                    "user": {"login": "octocat"},
                },
                {
                    "number": 1,
                    "title": "Older",
                    "body": "Body one",
                    "state": "open",
                    "labels": [],
                    "html_url": "https://example.invalid/1",
                    "updated_at": "2026-04-01T00:00:00Z",
                    "created_at": "2026-04-01T00:00:00Z",
                    "user": {"login": "octocat"},
                },
            ]
        }
        cutoff = datetime(2026, 5, 1, tzinfo=timezone.utc)

        with unittest.mock.patch.object(
            ai_labelling,
            "gh_json",
            return_value=payload,
        ):
            result = ai_labelling.search_items(
                "llvm/llvm-project",
                "issue",
                ai_labelling.SearchOptions(True, False, False, cutoff, None),
            )

        self.assertEqual([item.number for item in result], [2])


class BackendTests(unittest.TestCase):
    """Verify backend prompt construction and delegation behavior."""

    def test_generic_backend_prompt_includes_context(self):
        """The shared backend prompt should include title, labels, and body."""

        backend = ai_labelling.AIBackend(name="Test")
        item = ai_labelling.WorkItem(
            number=1,
            title="Test issue",
            body="Body text",
            state="open",
            labels=["bug"],
            html_url="https://example.invalid/1",
            updated_at="2026-05-01T00:00:00Z",
            created_at="2026-05-01T00:00:00Z",
            author_login="octocat",
            kind="issue",
        )

        prompt = backend.build_prompt(
            item,
            [
                ai_labelling.LabelDefinition("bug", "Bug report"),
                ai_labelling.LabelDefinition("docs", "Documentation"),
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

        backend = ai_labelling.AIBackend(name="Test")
        item = ai_labelling.WorkItem(
            number=1,
            title="Test issue",
            body="Body text",
            state="open",
            labels=["bug"],
            html_url="https://example.invalid/1",
            updated_at="2026-05-01T00:00:00Z",
            created_at="2026-05-01T00:00:00Z",
            author_login="octocat",
            kind="issue",
        )

        prompt = backend.build_prompt(
            item,
            [ai_labelling.LabelDefinition("bug", "Bug report")],
            allow_label_removals=False,
        )

        self.assertNotIn("remove_labels", prompt)
        self.assertIn("do not support adding labels confidently", prompt)

    def test_suggest_labels_delegates_to_backend_runner(self):
        """Backends should use ``run_prompt`` and return its object result."""

        class FakeBackend(  # pylint: disable=too-few-public-methods
            ai_labelling.AIBackend
        ):
            """Simple backend used to capture the prompt for assertions."""

            def __init__(self):
                """Initialize captured prompt state for backend assertions."""

                super().__init__(name="Fake")
                self.prompt = None
                self.model = None
                self.allow_label_removals = None

            def run_prompt(
                self,
                prompt,
                model,
                *,
                allow_label_removals,
            ):
                """Capture prompt/model and return a fixed label suggestion."""

                self.prompt = prompt
                self.model = model
                self.allow_label_removals = allow_label_removals
                return {
                    "add_labels": ["bug"],
                    "remove_labels": [],
                    "reason": "match",
                }

        backend = FakeBackend()
        item = ai_labelling.WorkItem(
            number=2,
            title="Another issue",
            body="Needs fix",
            state="open",
            labels=[],
            html_url="https://example.invalid/2",
            updated_at="2026-05-01T00:00:00Z",
            created_at="2026-05-01T00:00:00Z",
            author_login="octocat",
            kind="issue",
        )

        result = backend.suggest_labels(
            item,
            [ai_labelling.LabelDefinition("bug", "Bug report")],
            "gpt-test",
            False,
        )

        self.assertEqual(result["add_labels"], ["bug"])
        self.assertEqual(backend.model, "gpt-test")
        self.assertFalse(backend.allow_label_removals)
        self.assertIn("Needs fix", backend.prompt)

    def test_get_debug_level_follows_debug_environment_variable(self):
        """``DEBUG`` should map to numeric logging levels."""

        with unittest.mock.patch.dict("os.environ", {}, clear=True):
            self.assertEqual(ai_labelling.get_debug_level(), 0)
        with unittest.mock.patch.dict(
            "os.environ", {"DEBUG": ""}, clear=True
        ):
            self.assertEqual(ai_labelling.get_debug_level(), 0)
        with unittest.mock.patch.dict(
            "os.environ", {"DEBUG": "0"}, clear=True
        ):
            self.assertEqual(ai_labelling.get_debug_level(), 0)
        with unittest.mock.patch.dict(
            "os.environ", {"DEBUG": "1"}, clear=True
        ):
            self.assertEqual(ai_labelling.get_debug_level(), 1)
        with unittest.mock.patch.dict(
            "os.environ", {"DEBUG": "3"}, clear=True
        ):
            self.assertEqual(ai_labelling.get_debug_level(), 3)
        with unittest.mock.patch.dict(
            "os.environ", {"DEBUG": "verbose"}, clear=True
        ):
            self.assertEqual(ai_labelling.get_debug_level(), 1)

    def test_sanitize_prompt_for_debug_omits_runtime_data(self):
        """``DEBUG=2`` prompt views should hide issue-specific data blocks."""

        prompt = """You are labeling a GitHub issue.

Issue title:
Crash in foo

Existing labels:
[
  "bug"
]

Valid labels:
[
  {"name": "bug"}
]

Main body text:
Long body text.
"""

        result = ai_labelling.sanitize_prompt_for_debug(prompt)

        self.assertIn("<ISSUE TITLE OMITTED>", result)
        self.assertIn("<LABELS OMITTED>", result)
        self.assertIn("<LABEL DEFINITIONS OMITTED>", result)
        self.assertIn("<ISSUE BODY OMITTED>", result)
        self.assertNotIn("Crash in foo", result)
        self.assertNotIn("Long body text.", result)

    def test_codex_backend_logs_sanitized_prompt_in_debug_level_two(self):
        """``DEBUG=2`` should log one sanitized prompt and no response dump."""

        backend = ai_labelling.CodexBackend(name="Codex", command="codex")
        completed = unittest.mock.Mock(stdout="", stderr="")
        fake_json = {
            "add_labels": ["bug"],
            "remove_labels": [],
            "reason": "match",
        }

        with unittest.mock.patch.dict(
            "os.environ", {"DEBUG": "2"}, clear=True
        ):
            with unittest.mock.patch.object(
                ai_labelling, "run", return_value=completed
            ) as run_mock:
                with unittest.mock.patch.object(
                    ai_labelling.json,
                    "load",
                    return_value=fake_json,
                ):
                    with unittest.mock.patch.object(
                        ai_labelling, "debug_log"
                    ) as debug_log_mock:
                        result = backend.run_prompt(
                            "Issue title:\nReal title\n\nMain body text:\n"
                            "Real body\n",
                            "codex:*:low",
                            allow_label_removals=True,
                        )

        self.assertEqual(result, fake_json)
        run_mock.assert_called_once()
        debug_log_mock.assert_called_once_with(
            "Issue title:\n<ISSUE TITLE OMITTED>\nMain body text:\n"
            "<ISSUE BODY OMITTED>",
        )

    def test_codex_backend_schema_omits_remove_labels_when_disabled(self):
        """Schema should not request removal labels when the flag is off."""

        backend = ai_labelling.CodexBackend(name="Codex", command="codex")
        completed = unittest.mock.Mock(stdout="", stderr="")
        captured = {}

        def capture_dump(payload, _file_obj):
            captured["schema"] = payload

        with unittest.mock.patch.object(
            ai_labelling, "run", return_value=completed
        ):
            with unittest.mock.patch.object(
                ai_labelling.json,
                "dump",
                side_effect=capture_dump,
            ):
                with unittest.mock.patch.object(
                    ai_labelling.json,
                    "load",
                    return_value={
                        "add_labels": ["bug"],
                        "reason": "match",
                    },
                ):
                    result = backend.run_prompt(
                        "Prompt",
                        "codex:*:low",
                        allow_label_removals=False,
                    )

        self.assertEqual(result["add_labels"], ["bug"])
        self.assertNotIn(
            "remove_labels",
            captured["schema"]["properties"],
        )
        self.assertEqual(
            captured["schema"]["required"],
            ["add_labels", "reason"],
        )

    def test_anthropic_default_model_uses_first_model_entry(self):
        """Anthropic ``*`` should resolve to the first listed model."""

        with unittest.mock.patch.object(
            ai_labelling,
            "anthropic_json_request",
            return_value={
                "data": [
                    {"id": "claude-sonnet-4-20250514"},
                    {"id": "claude-3-7-sonnet-20250219"},
                ]
            },
        ):
            result = ai_labelling.anthropic_default_model()

        self.assertEqual(result, "claude-sonnet-4-20250514")

    def test_anthropic_extract_json_accepts_fenced_payload(self):
        """Anthropic text responses may wrap JSON in markdown fences."""

        result = ai_labelling.anthropic_extract_json(
            "```json\n{\"add_labels\": [\"bug\"], \"reason\": \"x\"}\n```"
        )

        self.assertEqual(
            result,
            {"add_labels": ["bug"], "reason": "x"},
        )

    def test_anthropic_backend_uses_messages_api_and_default_model(self):
        """Anthropic backend should resolve ``*`` and parse text content."""

        backend = ai_labelling.AnthropicBackend(name="Anthropic")

        with unittest.mock.patch.object(
            ai_labelling,
            "anthropic_default_model",
            return_value="claude-sonnet-4-20250514",
        ) as default_mock:
            with unittest.mock.patch.object(
                ai_labelling,
                "anthropic_json_request",
                return_value={
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "{\"add_labels\": [\"bug\"], "
                                "\"remove_labels\": [], "
                                "\"reason\": \"match\"}"
                            ),
                        }
                    ]
                },
            ) as request_mock:
                result = backend.run_prompt(
                    "Prompt body",
                    "anthropic:*:low",
                    allow_label_removals=True,
                )

        self.assertEqual(result["add_labels"], ["bug"])
        default_mock.assert_called_once_with()
        request_mock.assert_called_once_with(
            "/v1/messages",
            method="POST",
            body={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Prompt body"}],
                "output_config": {"effort": "low"},
            },
        )

    def test_format_prompt_for_debug_returns_full_prompt_at_level_three(self):
        """``DEBUG>=3`` should expose the full prompt text once."""

        with unittest.mock.patch.dict(
            "os.environ", {"DEBUG": "3"}, clear=True
        ):
            result = ai_labelling.format_prompt_for_debug("Full prompt")

        self.assertEqual(result, "Full prompt")

    def test_run_hides_subcommand_trace_without_debug(self):
        """Subprocess execution should stay quiet when ``DEBUG`` is off."""

        completed = unittest.mock.Mock(stdout="", stderr="")
        with unittest.mock.patch.dict("os.environ", {}, clear=True):
            with unittest.mock.patch.object(
                ai_labelling.subprocess,
                "run",
                return_value=completed,
            ) as run_mock:
                with unittest.mock.patch("builtins.print") as print_mock:
                    result = ai_labelling.run(("echo", "hello"))

        self.assertIs(result, completed)
        run_mock.assert_called_once()
        print_mock.assert_not_called()

    def test_run_logs_subcommand_trace_in_debug_mode(self):
        """Subprocess execution should emit trace output when debug is on."""

        completed = unittest.mock.Mock(stdout="", stderr="")
        with unittest.mock.patch.dict(
            "os.environ", {"DEBUG": "1"}, clear=True
        ):
            with unittest.mock.patch.object(
                ai_labelling.subprocess,
                "run",
                return_value=completed,
            ):
                with unittest.mock.patch("builtins.print") as print_mock:
                    ai_labelling.run(("echo", "hello world"))

        printed = [
            call.args[0] for call in print_mock.call_args_list if call.args
        ]
        self.assertEqual(printed, ["+ echo 'hello world'"])


class SearchResultTests(unittest.TestCase):
    """Check conversion of GitHub search payloads into work items."""

    def test_work_item_from_search_result_captures_title_and_author(self):
        """Search results should preserve title, author, and label metadata."""

        item = ai_labelling.work_item_from_search_result(
            {
                "number": 42,
                "title": "Example title",
                "body": "Body text",
                "state": "open",
                "labels": [{"name": "bug"}],
                "html_url": "https://example.invalid/42",
                "updated_at": "2026-05-01T00:00:00Z",
                "created_at": "2026-05-01T00:00:00Z",
                "user": {"login": "octocat"},
            },
            "issue",
        )

        self.assertEqual(item.title, "Example title")
        self.assertEqual(item.author_login, "octocat")
        self.assertEqual(item.labels, ["bug"])

    def test_list_repo_labels_keeps_description_metadata(self):
        """Repository label listing should preserve label descriptions."""

        payload = [[
            {"name": "bug", "description": "Bug report"},
            {"name": "docs", "description": "Documentation work"},
        ]]

        with unittest.mock.patch.object(
            ai_labelling,
            "gh_json",
            return_value=payload,
        ):
            labels = ai_labelling.list_repo_labels("llvm/llvm-project")

        self.assertEqual(
            labels,
            [
                ai_labelling.LabelDefinition("bug", "Bug report"),
                ai_labelling.LabelDefinition("docs", "Documentation work"),
            ],
        )

    def test_list_repo_labels_deduplicates_casefolded_names(self):
        """Later pages should replace earlier case-insensitive duplicates."""

        payload = [
            [{"name": "Bug", "description": "Older description"}],
            [{"name": "bug", "description": "Newer description"}],
        ]

        with unittest.mock.patch.object(
            ai_labelling,
            "gh_json",
            return_value=payload,
        ):
            labels = ai_labelling.list_repo_labels("llvm/llvm-project")

        self.assertEqual(
            labels,
            [ai_labelling.LabelDefinition("bug", "Newer description")],
        )

    def test_work_item_from_search_result_rejects_kind_mismatch(self):
        """Unexpected issue-vs-PR payload mismatches should fail loudly."""

        with self.assertRaises(RuntimeError):
            ai_labelling.work_item_from_search_result(
                {"number": 1, "pull_request": {}},
                "issue",
            )


class NormalizeSuggestionTests(unittest.TestCase):
    """Check normalization and filtering of Codex label suggestions."""

    def test_drop_invalid_duplicate_existing_labels(self):
        """Ignore invalid, duplicate, and already-present label additions."""

        suggestion = ai_labelling.normalize_label_suggestions(
            {
                "add_labels": ["bug", "BUG", "unknown", "clang"],
                "remove_labels": ["bug", "docs"],
                "reason": "test",
            },
            [
                ai_labelling.LabelDefinition("bug", ""),
                ai_labelling.LabelDefinition("clang", ""),
                ai_labelling.LabelDefinition("docs", ""),
            ],
            ["clang"],
        )
        self.assertEqual(suggestion.add_labels, ["bug"])
        self.assertEqual(suggestion.remove_labels, [])
        self.assertEqual(suggestion.reason, "test")


class FormattingTests(unittest.TestCase):
    """Verify human-facing item and reason formatting helpers."""

    def test_summarize_body_prefers_first_sentences(self):
        """Body previews should use opening sentences and ignore later text."""

        body = (
            "First sentence. Second sentence. Third sentence. "
            "Fourth sentence.\n\n"
            "Later paragraph."
        )

        result = ai_labelling.summarize_body(body)

        self.assertEqual(
            result,
            "First sentence. Second sentence. Third sentence.",
        )

    def test_summarize_body_skips_markdown_heading_paragraph(self):
        """Markdown heading blocks should not become the preview."""

        body = "## Summary\n\nActual first paragraph. More detail follows."

        result = ai_labelling.summarize_body(body)

        self.assertEqual(
            result,
            "Actual first paragraph. More detail follows.",
        )

    def test_format_body_preview_wraps_and_truncates(self):
        """Issue previews should preserve lines, wrap, and line-cap."""

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

        result = ai_labelling.format_body_preview(body, width=40, max_lines=5)

        self.assertIn("## Summary", result)
        self.assertIn("This is a shorter first line", result)
        self.assertIn("Second paragraph line one.", result)

    def test_format_body_preview_shows_heading_without_counting_it(self):
        """Headings should appear without consuming prose budget."""

        body = "## Summary\n\nFirst line.\nSecond line."

        result = ai_labelling.format_body_preview(body, width=80, max_lines=1)

        self.assertIn("## Summary", result)
        self.assertIn("First line...", result)

    def test_format_body_preview_preserves_code_block_newlines(self):
        """Code fences should keep original internal line breaks."""

        body = (
            "## Summary\n\n"
            "TSVC `s352` is a 5-wide unrolled dot product:\n\n"
            "```c\n"
            "dot = 0.;\n"
            "for (i = 0; i < LEN_1D; i += 5) dot = dot + a[i]*b[i];\n"
            "```"
        )

        result = ai_labelling.format_body_preview(body, width=120, max_lines=8)

        self.assertIn("```c\ndot = 0.;", result)

    def test_format_body_preview_shows_code_block_without_counting_it(self):
        """Code blocks should appear without consuming prose budget."""

        body = (
            "```c\n"
            "dot = 0.;\n"
            "for (i = 0; i < LEN_1D; i += 5) dot += a[i];\n"
            "```\n\n"
            "Intro line.\n\n"
            "Closing sentence."
        )

        result = ai_labelling.format_body_preview(body, width=120, max_lines=1)

        self.assertIn("```c\ndot = 0.;", result)
        self.assertIn("Intro line.", result)
        self.assertNotIn("Closing sentence.", result)

    def test_format_body_preview_shows_quotes_without_counting_them(self):
        """Quoted or admonition lines should stay visible but not count."""

        body = (
            "> Warning\n"
            "> Keep this in mind.\n\n"
            "Intro line.\n\n"
            "Closing sentence."
        )

        result = ai_labelling.format_body_preview(body, width=120, max_lines=1)

        self.assertIn("> Warning\n> Keep this in mind.", result)
        self.assertIn("Intro line.", result)
        self.assertNotIn("Closing sentence.", result)

    def test_format_body_preview_shows_standalone_code_line(self):
        """Standalone inline-code markdown should not count as prose."""

        body = (
            "`dot = 0.;`\n\n"
            "Intro line.\n\n"
            "Closing sentence."
        )

        result = ai_labelling.format_body_preview(body, width=120, max_lines=1)

        self.assertIn("`dot = 0.;`", result)
        self.assertIn("Intro line.", result)
        self.assertNotIn("Closing sentence.", result)

    def test_format_label_block_uses_bullets(self):
        """Labels should render one per line to avoid comma ambiguity."""

        result = ai_labelling.format_label_block(["bug", "area:docs,api"])

        self.assertEqual(result, "  - bug\n  - area:docs,api")

    def test_format_reason_wraps_and_indents(self):
        """Reason text should be indented for easier reading."""

        result = ai_labelling.format_reason(
            "This is a fairly long explanation that should wrap over "
            "multiple lines for readability in the terminal."
        )

        self.assertTrue(result.startswith("  This is"))
        self.assertIn("\n  ", result)


class ConfirmationTests(unittest.TestCase):
    """Verify user confirmation helpers for batch and interactive prompts."""

    def test_prompt_confirmation_retries_until_non_empty_valid_answer(self):
        """Blank or invalid replies should keep prompting."""

        answers = iter(["", "maybe", "a"])

        result = ai_labelling.prompt_confirmation(
            "Prompt: ",
            allow_apply_all=False,
            input_fn=lambda _: next(answers),
        )

        self.assertEqual(result, "A")

    def test_prompt_confirmation_accepts_done_alias(self):
        """The ``D`` shortcut should be accepted like git-style prompts."""

        result = ai_labelling.prompt_confirmation(
            "Prompt: ",
            allow_apply_all=True,
            input_fn=lambda _: "d",
        )

        self.assertEqual(result, "D")

    def test_prompt_confirmation_raises_on_quit(self):
        """The ``Q`` shortcut should terminate the program immediately."""

        with self.assertRaises(ai_labelling.UserQuit):
            ai_labelling.prompt_confirmation(
                "Prompt: ",
                allow_apply_all=False,
                input_fn=lambda _: "q",
            )

    def test_print_prompt_help_supports_item_mode(self):
        """Item prompt help should describe item-handling semantics."""

        with unittest.mock.patch("builtins.print") as print_mock:
            ai_labelling.print_prompt_help(False)

        printed = "\n".join(
            call.args[0] for call in print_mock.call_args_list if call.args
        )
        self.assertIn("handle this item", printed)
        self.assertIn("stop prompting more items", printed)

    def test_print_prompt_help_supports_apply_mode(self):
        """Suggestion prompt help should describe apply-all semantics."""

        with unittest.mock.patch("builtins.print") as print_mock:
            ai_labelling.print_prompt_help(True)

        printed = "\n".join(
            call.args[0] for call in print_mock.call_args_list if call.args
        )
        self.assertIn("use this and all remaining", printed)
        self.assertIn("remaining labels in this action", printed)

    def test_prompt_yes_no_uses_default_yes_on_empty_answer(self):
        """Blank retry replies should honor a true default answer."""

        result = ai_labelling.prompt_yes_no(
            "Retry? ",
            default_yes=True,
            input_fn=lambda _: "",
        )

        self.assertTrue(result)

    def test_prompt_yes_no_uses_default_no_on_empty_answer(self):
        """Blank retry replies should honor a false default answer."""

        result = ai_labelling.prompt_yes_no(
            "Retry? ",
            default_yes=False,
            input_fn=lambda _: "",
        )

        self.assertFalse(result)

    def test_prompt_yes_no_retries_until_valid_answer(self):
        """Invalid retry replies should keep prompting until yes or no."""

        answers = iter(["maybe", "n"])

        result = ai_labelling.prompt_yes_no(
            "Retry? ",
            default_yes=True,
            input_fn=lambda _: next(answers),
        )

        self.assertFalse(result)


class FlowTests(unittest.TestCase):  # pylint: disable=too-many-public-methods
    """Verify item-selection and apply-review control flow."""

    def make_item(self, number, title, *, kind="issue", state="open"):
        """Create a small work item fixture for flow tests."""

        return ai_labelling.WorkItem(
            number=number,
            title=title,
            body="Body text",
            state=state,
            labels=[],
            html_url=f"https://example.invalid/{number}",
            updated_at="2026-05-01T00:00:00Z",
            created_at="2026-05-01T00:00:00Z",
            author_login="octocat",
            kind=kind,
        )

    def test_select_items_to_handle_respects_yes_no_done(self):
        """Selection should gather accepted items and stop on done."""

        items = [
            self.make_item(1, "One"),
            self.make_item(2, "Two"),
            self.make_item(3, "Three"),
        ]
        answers = iter(["n", "y", "d"])

        with unittest.mock.patch.object(ai_labelling, "print_item_details"):
            selected = ai_labelling.select_items_to_handle(
                items,
                False,
                input_fn=lambda _: next(answers),
            )

        self.assertEqual([item.number for item in selected], [2])

    def test_select_items_to_handle_all_shortcut_selects_remaining(self):
        """The all shortcut should enqueue the current and remaining items."""

        items = [
            self.make_item(1, "One"),
            self.make_item(2, "Two"),
            self.make_item(3, "Three"),
        ]

        with unittest.mock.patch.object(ai_labelling, "print_item_details"):
            selected = ai_labelling.select_items_to_handle(
                items,
                False,
                input_fn=lambda _: "a",
            )

        self.assertEqual([item.number for item in selected], [1, 2, 3])

    def test_review_and_apply_suggestions_respects_apply_prompts(self):
        """Apply review should honor per-label prompts for one issue."""

        suggestion_results = [
            ai_labelling.SuggestionResult(
                item=self.make_item(1, "One"),
                label_suggestion=ai_labelling.LabelSuggestion(
                    add_labels=["bug", "docs"],
                    remove_labels=["old", "stale"],
                    reason="reason one",
                ),
            )
        ]
        answers = iter(["a", "d"])

        with unittest.mock.patch.object(ai_labelling, "print_summary"):
            with unittest.mock.patch.object(
                ai_labelling,
                "add_labels_with_retry",
            ) as add_mock:
                with unittest.mock.patch.object(
                    ai_labelling,
                    "remove_label_with_retry",
                ) as remove_mock:
                    ai_labelling.review_and_apply_suggestions(
                        "llvm/llvm-project",
                        suggestion_results,
                        False,
                        True,
                        input_fn=lambda _: next(answers),
                    )

        self.assertEqual(add_mock.call_count, 2)
        self.assertEqual(remove_mock.call_count, 0)
        self.assertEqual(
            [call.args[2] for call in add_mock.call_args_list],
            [["bug"], ["docs"]],
        )

    def test_review_and_apply_suggestions_force_removes_labels_too(self):
        """Force mode should apply removals automatically when enabled."""

        suggestion_results = [
            ai_labelling.SuggestionResult(
                item=self.make_item(1, "One"),
                label_suggestion=ai_labelling.LabelSuggestion(
                    add_labels=["bug"],
                    remove_labels=["old"],
                    reason="reason one",
                ),
            )
        ]

        with unittest.mock.patch.object(ai_labelling, "print_summary"):
            with unittest.mock.patch.object(
                ai_labelling,
                "add_labels_with_retry",
            ) as add_mock:
                with unittest.mock.patch.object(
                    ai_labelling,
                    "remove_label_with_retry",
                ) as remove_mock:
                    ai_labelling.review_and_apply_suggestions(
                        "llvm/llvm-project",
                        suggestion_results,
                        True,
                        True,
                    )

        add_mock.assert_called_once()
        remove_mock.assert_called_once_with(
            "llvm/llvm-project",
            suggestion_results[0].item,
            "old",
            input_fn=unittest.mock.ANY,
        )

    def test_run_ai_batch_returns_empty_for_empty_input(self):
        """Batch AI execution should skip executor setup for no items."""

        result = ai_labelling.run_ai_batch([], [], "codex:*:low", False)

        self.assertEqual(result, [])

    def test_run_ai_batch_uses_single_process_fast_path(self):
        """Single-worker batches should avoid the process pool."""

        item = self.make_item(1, "One")
        expected = ai_labelling.SuggestionResult(
            item=item,
            label_suggestion=ai_labelling.LabelSuggestion(
                add_labels=["bug"],
                remove_labels=[],
                reason="match",
            ),
        )

        with unittest.mock.patch.object(
            ai_labelling.os,
            "cpu_count",
            return_value=1,
        ):
            with unittest.mock.patch.object(
                ai_labelling,
                "build_suggestion_result_with_retry",
                return_value=expected,
            ) as build_mock:
                result = ai_labelling.run_ai_batch(
                    [item],
                    [ai_labelling.LabelDefinition("bug", "Bug report")],
                    "codex:*:low",
                    False,
                    input_fn=lambda _: "n",
                )

        self.assertEqual(result, [expected])
        build_mock.assert_called_once()

    def test_run_ai_batch_uses_process_pool_for_multiple_items(self):
        """Multi-item batches should preserve ordering across futures."""

        items = [self.make_item(1, "One"), self.make_item(2, "Two")]
        first_result = ai_labelling.SuggestionResult(
            item=items[0],
            label_suggestion=ai_labelling.LabelSuggestion(
                add_labels=["bug"],
                remove_labels=[],
                reason="first",
            ),
        )
        second_result = ai_labelling.SuggestionResult(
            item=items[1],
            label_suggestion=ai_labelling.LabelSuggestion(
                add_labels=["docs"],
                remove_labels=[],
                reason="second",
            ),
        )
        future_one = unittest.mock.Mock()
        future_one.result.return_value = first_result
        future_two = unittest.mock.Mock()
        future_two.result.return_value = second_result
        executor = unittest.mock.MagicMock()
        executor.__enter__.return_value = executor
        executor.submit.side_effect = [future_one, future_two]

        with unittest.mock.patch.object(
            ai_labelling.os,
            "cpu_count",
            return_value=4,
        ):
            with unittest.mock.patch.object(
                ai_labelling.concurrent.futures,
                "ProcessPoolExecutor",
                return_value=executor,
            ):
                with unittest.mock.patch.object(
                    ai_labelling.concurrent.futures,
                    "as_completed",
                    return_value=[future_two, future_one],
                ):
                    result = ai_labelling.run_ai_batch(
                        items,
                        [ai_labelling.LabelDefinition("bug", "Bug report")],
                        "codex:*:low",
                        False,
                        input_fn=lambda _: "n",
                    )

        self.assertEqual(result, [first_result, second_result])
        self.assertEqual(executor.submit.call_count, 2)

    def test_build_suggestion_result_with_retry_retries_ai_failures(self):
        """AI failures should print diagnostics and allow a manual retry."""

        item = self.make_item(1, "One")
        expected = ai_labelling.SuggestionResult(
            item=item,
            label_suggestion=ai_labelling.LabelSuggestion(
                add_labels=["bug"],
                remove_labels=[],
                reason="match",
            ),
        )

        with unittest.mock.patch.object(
            ai_labelling,
            "build_suggestion_result",
            side_effect=[RuntimeError("boom"), expected],
        ) as build_mock:
            with unittest.mock.patch.object(
                ai_labelling,
                "print_exception_diagnostics",
            ) as diag_mock:
                result = ai_labelling.build_suggestion_result_with_retry(
                    item,
                    [ai_labelling.LabelDefinition("bug", "Bug report")],
                    "codex:*:low",
                    False,
                    input_fn=lambda _: "y",
                )

        self.assertEqual(result, expected)
        self.assertEqual(build_mock.call_count, 2)
        diag_mock.assert_called_once()

    def test_build_suggestion_result_with_retry_skips_after_decline(self):
        """Declining an AI retry should skip that item without crashing."""

        item = self.make_item(1, "One")

        with unittest.mock.patch.object(
            ai_labelling,
            "build_suggestion_result",
            side_effect=RuntimeError("boom"),
        ):
            with unittest.mock.patch.object(
                ai_labelling,
                "print_exception_diagnostics",
            ):
                result = ai_labelling.build_suggestion_result_with_retry(
                    item,
                    [ai_labelling.LabelDefinition("bug", "Bug report")],
                    "codex:*:low",
                    False,
                    input_fn=lambda _: "",
                )

        self.assertIsNone(result)

    def test_run_ai_batch_retries_failed_future_when_user_accepts(self):
        """Parent-side future failures should support an interactive retry."""

        items = [self.make_item(1, "One"), self.make_item(2, "Two")]
        success = ai_labelling.SuggestionResult(
            item=items[1],
            label_suggestion=ai_labelling.LabelSuggestion(
                add_labels=["docs"],
                remove_labels=[],
                reason="second",
            ),
        )
        retry_result = ai_labelling.SuggestionResult(
            item=items[0],
            label_suggestion=ai_labelling.LabelSuggestion(
                add_labels=["bug"],
                remove_labels=[],
                reason="retried",
            ),
        )
        failed_future = unittest.mock.Mock()
        failed_future.result.side_effect = RuntimeError("boom")
        success_future = unittest.mock.Mock()
        success_future.result.return_value = success
        executor = unittest.mock.MagicMock()
        executor.__enter__.return_value = executor
        executor.submit.side_effect = [failed_future, success_future]

        with unittest.mock.patch.object(
            ai_labelling.os,
            "cpu_count",
            return_value=4,
        ):
            with unittest.mock.patch.object(
                ai_labelling.concurrent.futures,
                "ProcessPoolExecutor",
                return_value=executor,
            ):
                with unittest.mock.patch.object(
                    ai_labelling.concurrent.futures,
                    "as_completed",
                    return_value=[failed_future, success_future],
                ):
                    with unittest.mock.patch.object(
                        ai_labelling,
                        "build_suggestion_result_with_retry",
                        return_value=retry_result,
                    ) as retry_mock:
                        with unittest.mock.patch.object(
                            ai_labelling,
                            "print_exception_diagnostics",
                        ) as diag_mock:
                            result = ai_labelling.run_ai_batch(
                                items,
                                [
                                    ai_labelling.LabelDefinition(
                                        "bug",
                                        "Bug report",
                                    )
                                ],
                                "codex:*:low",
                                False,
                                input_fn=lambda _: "y",
                            )

        self.assertEqual(result, [retry_result, success])
        retry_mock.assert_called_once()
        diag_mock.assert_called_once()

    def test_add_labels_builds_expected_gh_command(self):
        """Label application should call GitHub with one field per label."""

        item = self.make_item(7, "Seven")
        completed = unittest.mock.Mock(stdout="", stderr="")

        with unittest.mock.patch.object(
            ai_labelling,
            "run",
            return_value=completed,
        ) as run_mock:
            ai_labelling.add_labels(
                "llvm/llvm-project",
                item,
                ["bug", "docs"],
            )

        run_mock.assert_called_once_with(
            (
                "gh",
                "api",
                "--method",
                "POST",
                "repos/llvm/llvm-project/issues/7/labels",
                "-f",
                "labels[]=bug",
                "-f",
                "labels[]=docs",
            )
        )

    def test_add_labels_with_retry_uses_default_yes(self):
        """Label-apply retries should default to yes on empty input."""

        item = self.make_item(1, "One")

        with unittest.mock.patch.object(
            ai_labelling,
            "add_labels",
            side_effect=[RuntimeError("boom"), None],
        ) as add_mock:
            with unittest.mock.patch.object(
                ai_labelling,
                "print_exception_diagnostics",
            ) as diag_mock:
                ai_labelling.add_labels_with_retry(
                    "llvm/llvm-project",
                    item,
                    ["bug"],
                    input_fn=lambda _: "",
                )

        self.assertEqual(add_mock.call_count, 2)
        diag_mock.assert_called_once()

    def test_remove_label_builds_expected_gh_command(self):
        """Label removal should call GitHub with the label in the path."""

        item = self.make_item(7, "Seven")
        completed = unittest.mock.Mock(stdout="", stderr="")

        with unittest.mock.patch.object(
            ai_labelling,
            "run",
            return_value=completed,
        ) as run_mock:
            ai_labelling.remove_label(
                "llvm/llvm-project",
                item,
                "bug",
            )

        run_mock.assert_called_once_with(
            (
                "gh",
                "api",
                "--method",
                "DELETE",
                "repos/llvm/llvm-project/issues/7/labels/bug",
            )
        )

    def test_remove_label_with_retry_uses_default_yes(self):
        """Removal retries should default to yes on empty input."""

        item = self.make_item(1, "One")

        with unittest.mock.patch.object(
            ai_labelling,
            "remove_label",
            side_effect=[RuntimeError("boom"), None],
        ) as remove_mock:
            with unittest.mock.patch.object(
                ai_labelling,
                "print_exception_diagnostics",
            ) as diag_mock:
                ai_labelling.remove_label_with_retry(
                    "llvm/llvm-project",
                    item,
                    "bug",
                    input_fn=lambda _: "",
                )

        self.assertEqual(remove_mock.call_count, 2)
        diag_mock.assert_called_once()

    def test_warn_force_mode_waits_for_requested_delay(self):
        """Force-mode warning should sleep for the configured delay."""

        with unittest.mock.patch.object(
            ai_labelling.time_module,
            "sleep",
        ) as sleep_mock:
            with unittest.mock.patch("builtins.print"):
                ai_labelling.warn_force_mode(3)

        sleep_mock.assert_called_once_with(3)

    def test_collect_items_sorts_and_limits_created_results(self):
        """Item collection should sort by creation time and respect limits."""

        newer = self.make_item(2, "Two")
        older = self.make_item(1, "One")
        newer.created_at = "2026-05-03T00:00:00Z"
        older.created_at = "2026-05-01T00:00:00Z"
        args = argparse.Namespace(
            created=True,
            date=ai_labelling.DEFAULT_DATE_CUTOFF,
            include_closed=False,
            include_issues=True,
            include_open=True,
            include_prs=False,
            limit=1,
        )

        with unittest.mock.patch.object(
            ai_labelling,
            "search_items",
            return_value=[older, newer],
        ):
            with unittest.mock.patch.object(
                ai_labelling,
                "default_cutoff",
                return_value=datetime(
                    2026, 5, 5, 0, 0, tzinfo=timezone.utc
                ),
            ):
                result = ai_labelling.collect_items(
                    "llvm/llvm-project", args
                )

        self.assertEqual([item.number for item in result], [2])

    def test_collect_items_returns_empty_with_no_entity_types_enabled(self):
        """Disabling both issues and PRs should skip all searches."""

        args = argparse.Namespace(
            created=False,
            date=ai_labelling.DEFAULT_DATE_CUTOFF,
            include_closed=False,
            include_issues=False,
            include_open=True,
            include_prs=False,
            limit=None,
        )

        with unittest.mock.patch.object(
            ai_labelling,
            "search_items",
        ) as search_mock:
            result = ai_labelling.collect_items("llvm/llvm-project", args)

        self.assertEqual(result, [])
        search_mock.assert_not_called()

    def test_collect_items_returns_empty_with_no_states_enabled(self):
        """Disabling both open and closed states should skip all searches."""

        args = argparse.Namespace(
            created=False,
            date=ai_labelling.DEFAULT_DATE_CUTOFF,
            include_closed=False,
            include_issues=True,
            include_open=False,
            include_prs=False,
            limit=None,
        )

        with unittest.mock.patch.object(
            ai_labelling,
            "search_items",
        ) as search_mock:
            result = ai_labelling.collect_items("llvm/llvm-project", args)

        self.assertEqual(result, [])
        search_mock.assert_not_called()

    def test_main_returns_zero_when_no_items_match(self):
        """Top-level flow should stop cleanly when filters match nothing."""

        args = argparse.Namespace(
            repo="llvm/llvm-project",
            created=False,
            date=ai_labelling.DEFAULT_DATE_CUTOFF,
            limit=None,
            allow_label_removals=False,
            include_closed=False,
            include_issues=True,
            include_open=True,
            include_prs=False,
            force=False,
            model="codex:*:low",
        )

        with unittest.mock.patch.object(
            ai_labelling,
            "parse_args",
            return_value=args,
        ):
            with unittest.mock.patch.object(
                ai_labelling,
                "list_repo_labels",
                return_value=[ai_labelling.LabelDefinition("bug", "")],
            ):
                with unittest.mock.patch.object(
                    ai_labelling,
                    "collect_items",
                    return_value=[],
                ):
                    result = ai_labelling.main()

        self.assertEqual(result, 0)

    def test_main_runs_full_review_flow_when_items_exist(self):
        """Top-level flow should orchestrate selection, AI, and review."""

        item = self.make_item(1, "One")
        args = argparse.Namespace(
            repo="llvm/llvm-project",
            created=False,
            date=ai_labelling.DEFAULT_DATE_CUTOFF,
            limit=None,
            allow_label_removals=False,
            include_closed=False,
            include_issues=True,
            include_open=True,
            include_prs=False,
            force=False,
            model="codex:*:low",
        )
        labels = [ai_labelling.LabelDefinition("bug", "Bug report")]
        suggestion_results = [
            ai_labelling.SuggestionResult(
                item=item,
                label_suggestion=ai_labelling.LabelSuggestion(
                    add_labels=["bug"],
                    remove_labels=[],
                    reason="match",
                ),
            )
        ]

        with unittest.mock.patch.object(
            ai_labelling,
            "parse_args",
            return_value=args,
        ):
            with unittest.mock.patch.object(
                ai_labelling,
                "list_repo_labels",
                return_value=labels,
            ):
                with unittest.mock.patch.object(
                    ai_labelling,
                    "collect_items",
                    return_value=[item],
                ):
                    with unittest.mock.patch.object(
                        ai_labelling,
                        "print_match_summary",
                    ) as summary_mock:
                        with unittest.mock.patch.object(
                            ai_labelling,
                            "print_matching_items",
                        ) as matching_mock:
                            with unittest.mock.patch.object(
                                ai_labelling,
                                "select_items_to_handle",
                                return_value=[item],
                            ) as select_mock:
                                with unittest.mock.patch.object(
                                    ai_labelling,
                                    "run_ai_batch",
                                    return_value=suggestion_results,
                                ) as batch_mock:
                                    with unittest.mock.patch.object(
                                        ai_labelling,
                                        "review_and_apply_suggestions",
                                    ) as review_mock:
                                        result = ai_labelling.main()

        self.assertEqual(result, 0)
        summary_mock.assert_called_once_with([item])
        matching_mock.assert_called_once_with([item], "Matching items")
        select_mock.assert_called_once_with([item], False)
        batch_mock.assert_called_once_with(
            [item],
            labels,
            "codex:*:low",
            False,
            input_fn=input,
        )
        review_mock.assert_called_once_with(
            "llvm/llvm-project",
            suggestion_results,
            False,
            False,
        )

    def test_main_returns_zero_when_no_items_are_selected(self):
        """Top-level flow should stop after selection if nothing is chosen."""

        item = self.make_item(1, "One")
        args = argparse.Namespace(
            repo="llvm/llvm-project",
            created=False,
            date=ai_labelling.DEFAULT_DATE_CUTOFF,
            limit=None,
            allow_label_removals=False,
            include_closed=False,
            include_issues=True,
            include_open=True,
            include_prs=False,
            force=False,
            model="codex:*:low",
        )

        with unittest.mock.patch.object(
            ai_labelling,
            "parse_args",
            return_value=args,
        ):
            with unittest.mock.patch.object(
                ai_labelling,
                "list_repo_labels",
                return_value=[ai_labelling.LabelDefinition("bug", "")],
            ):
                with unittest.mock.patch.object(
                    ai_labelling,
                    "collect_items",
                    return_value=[item],
                ):
                    with unittest.mock.patch.object(
                        ai_labelling,
                        "print_match_summary",
                    ):
                        with unittest.mock.patch.object(
                            ai_labelling,
                            "print_matching_items",
                        ):
                            with unittest.mock.patch.object(
                                ai_labelling,
                                "select_items_to_handle",
                                return_value=[],
                            ):
                                result = ai_labelling.main()

        self.assertEqual(result, 0)

    def test_print_match_summary_uses_single_line_for_one_bucket(self):
        """One matching category should print a single summary line."""

        item = self.make_item(1, "One")

        with unittest.mock.patch("builtins.print") as print_mock:
            ai_labelling.print_match_summary([item])

        printed = [call.args[0] for call in print_mock.call_args_list]
        self.assertEqual(printed, ["Matched open issues: 1"])

    def test_print_match_summary_lists_multiple_buckets(self):
        """Mixed item kinds should be shown as separate summary rows."""

        open_issue = self.make_item(1, "Issue")
        closed_pr = self.make_item(2, "PR", kind="pr", state="closed")

        with unittest.mock.patch("builtins.print") as print_mock:
            ai_labelling.print_match_summary([open_issue, closed_pr])

        printed = [call.args[0] for call in print_mock.call_args_list]
        self.assertEqual(
            printed,
            [
                "Matched items:",
                "  - Open issues: 1",
                "  - Closed PRs: 1",
            ],
        )


if __name__ == "__main__":
    unittest.main()
