"""Tests for GitHub CLI client helpers."""
# pylint: disable=missing-function-docstring

import sys
import unittest
from datetime import datetime, timezone
from unittest import mock

from test.helpers import make_item
from ai_labelling.github_client import (
    GitHubClient,
    _parse_gh_timing,
    parse_closing_issues,
    work_item_from_search_result,
)
from ai_labelling.models import ClosingPR
from ai_labelling.models import (
    IssueTypeDefinition,
    LabelDefinition,
    SearchOptions,
)


class GitHubClientTests(  # pylint: disable=too-many-public-methods
    unittest.TestCase,
):
    """Cover repository detection, queries, and label mutations."""

    def setUp(self):
        self.client = GitHubClient()

    def test_parse_ssh_remote(self):
        self.assertEqual(
            self.client.parse_repo_from_remote(
                "git@github.com:llvm/llvm-project.git"
            ),
            "llvm/llvm-project",
        )

    def test_parse_short_ssh_remote(self):
        self.assertEqual(
            self.client.parse_repo_from_remote(
                "github.com:Whisperity/llvm-project.git"
            ),
            "Whisperity/llvm-project",
        )

    def test_parse_http_remote(self):
        self.assertEqual(
            self.client.parse_repo_from_remote(
                "http://github.com/llvm/llvm-project.git"
            ),
            "llvm/llvm-project",
        )

    def test_ignore_non_github_remote(self):
        self.assertIsNone(
            self.client.parse_repo_from_remote(
                "https://gitlab.com/llvm/llvm-project.git"
            )
        )

    def test_detect_repo_falls_through_remotes_until_one_matches(self):
        failures = mock.Mock(returncode=1, stdout="")
        success = mock.Mock(
            returncode=0,
            stdout="git@github.com:llvm/llvm-project.git\n",
        )

        with mock.patch(
            "ai_labelling.github_client.run",
            side_effect=[failures, success],
        ) as run_mock:
            repo = self.client.detect_repo()

        self.assertEqual(repo, "llvm/llvm-project")
        self.assertEqual(run_mock.call_count, 2)

    def test_build_search_query_open_updated(self):
        cutoff = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)
        self.assertEqual(
            self.client.build_search_query(
                "llvm/llvm-project",
                "issue",
                SearchOptions(True, False, False, cutoff, None),
            ),
            "repo:llvm/llvm-project is:issue is:open updated:>=2026-05-01",
        )

    def test_build_search_query_closed_created(self):
        cutoff = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)
        self.assertEqual(
            self.client.build_search_query(
                "llvm/llvm-project",
                "pr",
                SearchOptions(False, True, True, cutoff, None),
            ),
            "repo:llvm/llvm-project is:pr is:closed created:>=2026-05-01",
        )

    def test_build_search_query_open_and_closed_created(self):
        cutoff = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)
        self.assertEqual(
            self.client.build_search_query(
                "llvm/llvm-project",
                "pr",
                SearchOptions(True, True, True, cutoff, None),
            ),
            "repo:llvm/llvm-project is:pr created:>=2026-05-01",
        )

    def test_search_items_stops_at_limit(self):
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

        with mock.patch.object(
            self.client,
            "json",
            return_value=payload,
        ) as gh_mock:
            result = self.client.search_items(
                "llvm/llvm-project",
                "issue",
                SearchOptions(True, False, False, None, 1),
            )

        self.assertEqual([item.number for item in result], [2])
        gh_mock.assert_called_once()

    def test_search_items_stops_when_cutoff_is_crossed(self):
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

        with mock.patch.object(self.client, "json", return_value=payload):
            result = self.client.search_items(
                "llvm/llvm-project",
                "issue",
                SearchOptions(True, False, False, cutoff, None),
            )

        self.assertEqual([item.number for item in result], [2])

    def test_work_item_from_search_result_captures_title_and_author(self):
        item = work_item_from_search_result(
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
        payload = [[
            {"name": "bug", "description": "Bug report"},
            {"name": "docs", "description": "Documentation work"},
        ]]

        with mock.patch.object(self.client, "json", return_value=payload):
            labels = self.client.list_repo_labels("llvm/llvm-project")

        self.assertEqual(
            labels,
            [
                LabelDefinition("bug", "Bug report"),
                LabelDefinition("docs", "Documentation work"),
            ],
        )

    def test_list_repo_labels_deduplicates_casefolded_names(self):
        payload = [
            [{"name": "Bug", "description": "Older description"}],
            [{"name": "bug", "description": "Newer description"}],
        ]

        with mock.patch.object(self.client, "json", return_value=payload):
            labels = self.client.list_repo_labels("llvm/llvm-project")

        self.assertEqual(labels, [LabelDefinition("bug", "Newer description")])

    def test_work_item_from_search_result_rejects_kind_mismatch(self):
        with self.assertRaises(RuntimeError):
            work_item_from_search_result(
                {"number": 1, "pull_request": {}},
                "issue",
            )

    def test_add_labels_builds_expected_gh_command(self):
        item = make_item(7, "Seven")
        completed = mock.Mock(stdout="", stderr="")

        with mock.patch(
            "ai_labelling.github_client.run",
            return_value=completed,
        ) as run_mock:
            self.client.add_labels("llvm/llvm-project", item, ["bug", "docs"])

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

    def test_remove_label_builds_expected_gh_command(self):
        item = make_item(7, "Seven")
        completed = mock.Mock(stdout="", stderr="")

        with mock.patch(
            "ai_labelling.github_client.run",
            return_value=completed,
        ) as run_mock:
            self.client.remove_label("llvm/llvm-project", item, "bug")

        run_mock.assert_called_once_with(
            (
                "gh",
                "api",
                "--method",
                "DELETE",
                "repos/llvm/llvm-project/issues/7/labels/bug",
            )
        )

    def test_post_comment_sends_json_body_via_stdin(self):
        item = make_item(7, "Seven")
        completed = mock.Mock(returncode=0, stdout="", stderr="")

        with mock.patch(
            "ai_labelling.github_client.run",
            return_value=completed,
        ) as run_mock:
            self.client.post_comment("llvm/llvm-project", item, "hello")

        run_mock.assert_called_once_with(
            (
                "gh",
                "api",
                "--method",
                "POST",
                "repos/llvm/llvm-project/issues/7/comments",
                "--input",
                "-",
            ),
            input_text='{"body": "hello"}',
            check=False,
        )

    def test_post_comment_warns_on_failure(self):
        item = make_item(7, "Seven")
        completed = mock.Mock(returncode=1, stdout="", stderr="api error")

        with mock.patch(
            "ai_labelling.github_client.run",
            return_value=completed,
        ):
            with mock.patch("builtins.print") as print_mock:
                self.client.post_comment("llvm/llvm-project", item, "hello")

        stderr_calls = [
            c for c in print_mock.call_args_list
            if c.kwargs.get("file") is sys.stderr
        ]
        self.assertGreater(len(stderr_calls), 0)

    def test_json_raises_runtime_error_on_invalid_json(self):
        """gh output that is not valid JSON should surface as RuntimeError."""

        completed = mock.Mock(stdout="not json", returncode=0)
        with mock.patch(
            "ai_labelling.github_client.run", return_value=completed
        ):
            with self.assertRaisesRegex(RuntimeError, "failed to parse JSON"):
                self.client.json(("api", "some/endpoint"))

    def test_detect_repo_raises_when_all_remotes_fail(self):
        """An exception should be raised if no remote resolves."""

        failure = mock.Mock(returncode=1, stdout="")
        with mock.patch(
            "ai_labelling.github_client.run", return_value=failure
        ):
            with self.assertRaisesRegex(RuntimeError, "could not infer"):
                self.client.detect_repo()

    def test_list_repo_labels_rejects_non_list_payload(self):
        """A non-list API response should raise RuntimeError."""

        with mock.patch.object(
            self.client, "json", return_value={"error": "oops"}
        ):
            with self.assertRaisesRegex(
                RuntimeError, "unexpected label payload"
            ):
                self.client.list_repo_labels("llvm/llvm-project")

    def test_list_repo_labels_rejects_non_list_page(self):
        """Each pagination page must itself be a list."""

        with mock.patch.object(
            self.client, "json", return_value=[{"not": "a list"}]
        ):
            with self.assertRaisesRegex(
                RuntimeError, "unexpected label page payload"
            ):
                self.client.list_repo_labels("llvm/llvm-project")

    def test_list_repo_labels_skips_entries_without_name(self):
        """Label entries that lack a ``name`` field are silently ignored."""

        payload = [[
            {"description": "no name here"},
            {"name": "bug", "description": ""},
        ]]
        with mock.patch.object(self.client, "json", return_value=payload):
            labels = self.client.list_repo_labels("llvm/llvm-project")
        self.assertEqual([lbl.name for lbl in labels], ["bug"])

    def test_search_items_rejects_non_dict_payload(self):
        """A non-object search response should raise RuntimeError."""

        with mock.patch.object(
            self.client, "json", return_value=["unexpected"]
        ):
            with self.assertRaisesRegex(
                RuntimeError, "unexpected search payload"
            ):
                self.client.search_items(
                    "llvm/llvm-project",
                    "issue",
                    SearchOptions(True, False, False, None, None),
                )

    def test_search_items_rejects_non_list_items(self):
        """An ``items`` field that is not a list should raise RuntimeError."""

        with mock.patch.object(
            self.client, "json", return_value={"items": "bad"}
        ):
            with self.assertRaisesRegex(
                RuntimeError, "unexpected items payload"
            ):
                self.client.search_items(
                    "llvm/llvm-project",
                    "issue",
                    SearchOptions(True, False, False, None, None),
                )

    def test_work_item_from_search_result_rejects_non_dict_entry(self):
        """A search result entry that is not a dict should raise."""

        with self.assertRaisesRegex(RuntimeError, "unexpected issue entry"):
            work_item_from_search_result("not a dict", "issue")

    def test_work_item_from_search_result_extracts_issue_type(self):
        item = work_item_from_search_result(
            {
                "number": 1,
                "title": "T",
                "body": "",
                "state": "open",
                "labels": [],
                "html_url": "",
                "updated_at": "2026-05-01T00:00:00Z",
                "created_at": "2026-05-01T00:00:00Z",
                "user": {"login": "octocat"},
                "type": {"id": 42, "name": "Bug"},
            },
            "issue",
        )
        self.assertEqual(item.issue_type, "Bug")

    def test_work_item_from_search_result_no_issue_type_is_none(self):
        item = work_item_from_search_result(
            {
                "number": 1,
                "title": "T",
                "body": "",
                "state": "open",
                "labels": [],
                "html_url": "",
                "updated_at": "2026-05-01T00:00:00Z",
                "created_at": "2026-05-01T00:00:00Z",
                "user": {"login": "octocat"},
            },
            "issue",
        )
        self.assertIsNone(item.issue_type)

    def test_list_issue_types_returns_empty_on_failure(self):
        failed = mock.Mock(returncode=1, stdout="", stderr="404")
        with mock.patch("ai_labelling.github_client.run", return_value=failed):
            result = self.client.list_issue_types("llvm/llvm-project")
        self.assertEqual(result, [])

    def test_list_issue_types_parses_org_response(self):
        payload = '[{"id": 1, "name": "Bug", "description": "A bug"}]'
        ok = mock.Mock(returncode=0, stdout=payload, stderr="")
        with mock.patch("ai_labelling.github_client.run", return_value=ok):
            result = self.client.list_issue_types("llvm/llvm-project")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].name, "Bug")
        self.assertEqual(result[0].description, "A bug")
        self.assertEqual(result[0].type_id, 1)

    def test_set_issue_type_uses_id_when_available(self):
        item = make_item(7, "Seven")
        it = IssueTypeDefinition(name="Bug", description="", type_id=42)
        completed = mock.Mock(returncode=0, stdout="", stderr="")
        with mock.patch(
            "ai_labelling.github_client.run", return_value=completed
        ) as run_mock:
            self.client.set_issue_type("llvm/llvm-project", item, it)
        run_mock.assert_called_once_with(
            (
                "gh", "api", "--method", "PATCH",
                "repos/llvm/llvm-project/issues/7",
                "--input", "-",
            ),
            input_text='{"type": {"id": 42}}',
        )

    def test_set_issue_type_falls_back_to_name_when_id_zero(self):
        item = make_item(7, "Seven")
        it = IssueTypeDefinition(name="Task", description="")
        completed = mock.Mock(returncode=0, stdout="", stderr="")
        with mock.patch(
            "ai_labelling.github_client.run", return_value=completed
        ) as run_mock:
            self.client.set_issue_type("llvm/llvm-project", item, it)
        _args, kwargs = run_mock.call_args
        self.assertIn('"name": "Task"', kwargs["input_text"])

    def test_get_item_returns_issue_when_no_pull_request_key(self):
        payload = {
            "number": 5,
            "title": "Crash on startup",
            "body": "body text",
            "state": "open",
            "labels": [],
            "html_url": "https://github.com/llvm/llvm-project/issues/5",
            "updated_at": "2026-05-01T00:00:00Z",
            "created_at": "2026-05-01T00:00:00Z",
            "user": {"login": "alice"},
        }
        with mock.patch.object(
            self.client, "json", return_value=payload
        ) as json_mock:
            item = self.client.get_item("llvm/llvm-project", 5)

        json_mock.assert_called_once_with(
            ("api", "repos/llvm/llvm-project/issues/5")
        )
        self.assertEqual(item.number, 5)
        self.assertEqual(item.kind, "issue")
        self.assertEqual(item.title, "Crash on startup")

    def test_get_item_returns_pr_when_pull_request_key_present(self):
        payload = {
            "number": 10,
            "title": "Fix crash",
            "body": "",
            "state": "open",
            "labels": [],
            "html_url": "https://github.com/llvm/llvm-project/pull/10",
            "updated_at": "2026-05-02T00:00:00Z",
            "created_at": "2026-05-02T00:00:00Z",
            "user": {"login": "bob"},
            "pull_request": {"merged_at": None},
        }
        with mock.patch.object(self.client, "json", return_value=payload):
            item = self.client.get_item("llvm/llvm-project", 10)

        self.assertEqual(item.number, 10)
        self.assertEqual(item.kind, "pr")

    def test_get_item_raises_on_non_dict_payload(self):
        with mock.patch.object(self.client, "json", return_value=None):
            with self.assertRaises(RuntimeError):
                self.client.get_item("llvm/llvm-project", 99)

    def test_format_search_date_uses_utc_day(self):
        timestamp = datetime(2026, 5, 1, 23, 45, tzinfo=timezone.utc)
        self.assertEqual(
            GitHubClient.format_search_date(timestamp), "2026-05-01"
        )

    def test_build_search_query_without_cutoff_omits_qualifier(self):
        query = self.client.build_search_query(
            "owner/repo",
            "issue",
            SearchOptions(True, False, False, None, None),
        )
        self.assertEqual(query, "repo:owner/repo is:issue is:open")

    def test_add_labels_no_op_when_no_labels(self):
        item = make_item(7, "Seven")
        with mock.patch(
            "ai_labelling.github_client.run"
        ) as run_mock:
            self.client.add_labels("owner/repo", item, [])
        run_mock.assert_not_called()

    def test_emit_completed_output_handles_empty_streams(self):
        item = make_item(7, "Seven")
        completed = mock.Mock(stdout="", stderr="")
        with mock.patch(
            "ai_labelling.github_client.run", return_value=completed
        ), mock.patch("builtins.print") as print_mock:
            self.client.add_labels("owner/repo", item, ["bug"])
        print_mock.assert_not_called()

    def test_search_items_paginates_until_short_page(self):
        full_page = {
            "items": [
                {
                    "number": i,
                    "title": str(i),
                    "body": "",
                    "state": "open",
                    "labels": [],
                    "html_url": "",
                    "updated_at": "2026-05-03T00:00:00Z",
                    "created_at": "2026-05-03T00:00:00Z",
                    "user": {"login": "octocat"},
                }
                for i in range(100)
            ]
        }
        partial_page = {
            "items": [
                {
                    "number": 200,
                    "title": "Last",
                    "body": "",
                    "state": "open",
                    "labels": [],
                    "html_url": "",
                    "updated_at": "2026-05-03T00:00:00Z",
                    "created_at": "2026-05-03T00:00:00Z",
                    "user": {"login": "octocat"},
                }
            ]
        }
        with mock.patch.object(
            self.client, "json", side_effect=[full_page, partial_page]
        ):
            result = self.client.search_items(
                "owner/repo",
                "issue",
                SearchOptions(True, False, False, None, None),
            )
        self.assertEqual(len(result), 101)

    def test_search_items_breaks_on_empty_page(self):
        with mock.patch.object(
            self.client, "json", return_value={"items": []}
        ):
            result = self.client.search_items(
                "owner/repo",
                "issue",
                SearchOptions(True, False, False, None, None),
            )
        self.assertEqual(result, [])

    def test_list_issue_types_handles_invalid_json(self):
        bad = mock.Mock(returncode=0, stdout="not json", stderr="")
        with mock.patch(
            "ai_labelling.github_client.run", return_value=bad,
        ):
            self.assertEqual(
                self.client.list_issue_types("owner/repo"), []
            )

    def test_list_issue_types_rejects_non_list_payload(self):
        bad = mock.Mock(returncode=0, stdout='{"unexpected": true}', stderr="")
        with mock.patch(
            "ai_labelling.github_client.run", return_value=bad,
        ):
            self.assertEqual(
                self.client.list_issue_types("owner/repo"), []
            )

    def test_list_issue_types_skips_invalid_entries(self):
        payload = '[{"id": 1, "name": "Bug"}, {"name": ""}, "wrong type"]'
        ok = mock.Mock(returncode=0, stdout=payload, stderr="")
        with mock.patch(
            "ai_labelling.github_client.run", return_value=ok,
        ):
            result = self.client.list_issue_types("owner/repo")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].name, "Bug")

    def test_parse_repo_from_remote_handles_https_form(self):
        self.assertEqual(
            GitHubClient.parse_repo_from_remote(
                "https://github.com/llvm/llvm-project"
            ),
            "llvm/llvm-project",
        )

    def test_parse_repo_from_remote_rejects_garbage(self):
        self.assertIsNone(
            GitHubClient.parse_repo_from_remote("garbage://x/y")
        )

    def test_parse_repo_from_remote_rejects_owner_only(self):
        self.assertIsNone(
            GitHubClient.parse_repo_from_remote("git@github.com:onlyowner")
        )


class ParseGhTimingTests(unittest.TestCase):
    """Verify extraction of gh request-metadata lines."""

    _TYPICAL = (
        "* Request at 2026-05-08 11:16:29.808334 +0200 CEST"
        " m=+0.042254626\n"
        "* Request to"
        " https://api.github.com/repos/owner/repo/issues/1/labels\n"
        "* Request took 5.522104083s\n"
    )

    def test_returns_timing_line_for_typical_gh_output(self):
        timing, errors = _parse_gh_timing(self._TYPICAL)
        self.assertIsNotNone(timing)
        self.assertEqual(errors, "")

    def test_timing_line_contains_date_and_offset(self):
        timing, _ = _parse_gh_timing(self._TYPICAL)
        self.assertIn("2026-05-08 11:16:29", timing)
        self.assertIn("+0200", timing)

    def test_timing_line_contains_rounded_duration(self):
        timing, _ = _parse_gh_timing(self._TYPICAL)
        self.assertIn("5.522s", timing)

    def test_request_to_line_not_in_timing_or_errors(self):
        timing, errors = _parse_gh_timing(self._TYPICAL)
        self.assertNotIn("api.github.com", timing or "")
        self.assertNotIn("api.github.com", errors)

    def test_real_error_lines_returned_as_errors(self):
        stderr = (
            "* Request at 2026-05-08 11:00:00 +0000\n"
            "* Request took 1.0s\n"
            "HTTP 422 Unprocessable Entity\n"
        )
        timing, errors = _parse_gh_timing(stderr)
        self.assertIsNotNone(timing)
        self.assertIn("HTTP 422", errors)

    def test_empty_stderr_returns_no_timing_no_errors(self):
        timing, errors = _parse_gh_timing("")
        self.assertIsNone(timing)
        self.assertEqual(errors, "")

    def test_only_error_lines_returns_no_timing(self):
        timing, errors = _parse_gh_timing("gh: error: 401 Unauthorized\n")
        self.assertIsNone(timing)
        self.assertIn("401", errors)

    def test_monotonic_suffix_stripped_from_timestamp(self):
        timing, _ = _parse_gh_timing(self._TYPICAL)
        self.assertNotIn("m=+", timing or "")


class AssigneesParsingTests(unittest.TestCase):
    """Verify assignee parsing in work_item_from_search_result."""

    def _entry(self, assignees):
        return {
            "number": 1, "title": "T", "body": "", "state": "open",
            "labels": [], "html_url": "", "updated_at": "2026-05-01T00:00:00Z",
            "created_at": "2026-05-01T00:00:00Z",
            "user": {"login": "author"}, "assignees": assignees,
        }

    def test_parses_single_assignee(self):
        item = work_item_from_search_result(
            self._entry([{"login": "alice"}]), "issue"
        )
        self.assertEqual(item.assignees, ["alice"])

    def test_parses_multiple_assignees(self):
        item = work_item_from_search_result(
            self._entry([{"login": "alice"}, {"login": "bob"}]), "issue"
        )
        self.assertIn("alice", item.assignees)
        self.assertIn("bob", item.assignees)

    def test_empty_assignees_gives_empty_list(self):
        item = work_item_from_search_result(self._entry([]), "issue")
        self.assertEqual(item.assignees, [])


class GetClosingPrTests(unittest.TestCase):
    """Verify get_closing_pr GraphQL parsing."""

    def setUp(self):
        self.client = GitHubClient()

    def _response(self, pr_number=None, login=None):
        nodes = []
        if pr_number is not None:
            nodes = [{"closer": {
                "number": pr_number,
                "author": {"login": login or "dev"},
            }}]
        return mock.Mock(
            returncode=0,
            stdout=__import__("json").dumps({
                "data": {"repository": {"issue": {
                    "timelineItems": {"nodes": nodes}
                }}}
            }),
            stderr="",
        )

    def test_returns_pr_number_and_author(self):
        with mock.patch(
            "ai_labelling.github_client.run",
            return_value=self._response(42, "dev"),
        ):
            result = self.client.get_closing_pr("owner/repo", 1)
        self.assertEqual(result, ClosingPR(pr_number=42, author_login="dev"))

    def test_returns_none_when_no_closed_event(self):
        with mock.patch(
            "ai_labelling.github_client.run",
            return_value=self._response(),
        ):
            result = self.client.get_closing_pr("owner/repo", 1)
        self.assertIsNone(result)

    def test_returns_none_on_gh_error(self):
        with mock.patch(
            "ai_labelling.github_client.run",
            return_value=mock.Mock(returncode=1, stdout="", stderr="err"),
        ):
            result = self.client.get_closing_pr("owner/repo", 1)
        self.assertIsNone(result)

    def test_closer_without_number_returns_none(self):
        payload = {"data": {"repository": {"issue": {
            "timelineItems": {"nodes": [{"closer": {"author": {}}}]}
        }}}}
        with mock.patch(
            "ai_labelling.github_client.run",
            return_value=mock.Mock(
                returncode=0,
                stdout=__import__("json").dumps(payload),
                stderr="",
            ),
        ):
            result = self.client.get_closing_pr("owner/repo", 1)
        self.assertIsNone(result)


class SetAssigneesTests(unittest.TestCase):
    """Verify set_assignees builds the correct gh command and body."""

    def setUp(self):
        self.client = GitHubClient()

    def test_patches_issue_with_assignees_list(self):
        item = make_item(7, "Seven")
        completed = mock.Mock(stdout="", stderr="")

        with mock.patch(
            "ai_labelling.github_client.run", return_value=completed
        ) as run_mock:
            self.client.set_assignees("owner/repo", item, ["alice"])

        call_args = run_mock.call_args
        argv = call_args.args[0]
        self.assertIn("PATCH", argv)
        self.assertIn("repos/owner/repo/issues/7", argv)
        body = __import__("json").loads(call_args.kwargs["input_text"])
        self.assertEqual(body["assignees"], ["alice"])


class ParseClosingIssuesTests(unittest.TestCase):
    """Verify GitHub closing-keyword extraction from PR bodies."""

    def test_closes_single_issue(self):
        self.assertEqual(parse_closing_issues("closes #42"), [42])

    def test_fixes_single_issue(self):
        self.assertEqual(parse_closing_issues("Fixes #7"), [7])

    def test_resolves_single_issue(self):
        self.assertEqual(parse_closing_issues("resolved #100"), [100])

    def test_close_without_d_or_s(self):
        self.assertEqual(parse_closing_issues("close #1"), [1])

    def test_fix_without_suffix(self):
        self.assertEqual(parse_closing_issues("fix #3"), [3])

    def test_resolve_without_suffix(self):
        self.assertEqual(parse_closing_issues("resolve #5"), [5])

    def test_fixed_past_tense(self):
        self.assertEqual(parse_closing_issues("fixed #9"), [9])

    def test_multiple_keywords(self):
        result = parse_closing_issues("Closes #1 and also fixes #2")
        self.assertEqual(result, [1, 2])

    def test_case_insensitive(self):
        self.assertEqual(parse_closing_issues("CLOSES #99"), [99])

    def test_no_keywords_returns_empty(self):
        self.assertEqual(
            parse_closing_issues("See also #5, related to #6"), []
        )

    def test_empty_body_returns_empty(self):
        self.assertEqual(parse_closing_issues(""), [])

    def test_keyword_in_multiline_body(self):
        body = "Some description.\n\ncloses #10\n\nMore text."
        self.assertEqual(parse_closing_issues(body), [10])
