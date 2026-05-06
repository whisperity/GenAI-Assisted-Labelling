"""Tests for GitHub CLI client helpers."""
# pylint: disable=missing-function-docstring

import sys
import unittest
from datetime import datetime, timezone
from unittest import mock

from test.helpers import make_item
from ai_labelling.github_client import (
    GitHubClient,
    parse_github_timestamp,
    work_item_from_search_result,
)
from ai_labelling.models import LabelDefinition, SearchOptions


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

    def test_parse_github_timestamp_returns_aware_datetime(self):
        parsed = parse_github_timestamp("2026-05-01T00:00:00Z")
        self.assertEqual(parsed.tzinfo, timezone.utc)

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
