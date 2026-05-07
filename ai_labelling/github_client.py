"""GitHub CLI-backed client helpers."""

import json
import subprocess
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence

from ai_labelling.config import REPO_DETECTION_ORDER
from ai_labelling.models import (
    IssueTypeDefinition,
    LabelDefinition,
    SearchOptions,
    WorkItem,
    parse_github_timestamp,
)
from ai_labelling.shell import run
from ai_labelling.terminal import colourise


def _emit_completed_output(completed: subprocess.CompletedProcess) -> None:
    """Forward a completed ``gh`` invocation's stdout/stderr to the user."""

    if completed.stdout and completed.stdout.strip():
        print(colourise(completed.stdout.strip(), "green", bold=True))
    if completed.stderr and completed.stderr.strip():
        print(
            colourise(
                completed.stderr.strip(),
                "yellow",
                stream=sys.stderr,
            ),
            file=sys.stderr,
        )


def work_item_from_search_result(
    entry: object,
    expected_kind: str,
) -> WorkItem:
    """Convert a GitHub search result payload into a ``WorkItem``."""

    if not isinstance(entry, dict):
        raise RuntimeError("unexpected issue entry payload from GitHub")

    labels = []
    for label in entry.get("labels", []):
        if isinstance(label, dict) and "name" in label:
            labels.append(str(label["name"]))

    kind = "pr" if "pull_request" in entry else "issue"
    if kind != expected_kind:
        raise RuntimeError(f"expected {expected_kind}, got {kind}")

    type_obj = entry.get("type")
    issue_type = (
        str(type_obj["name"])
        if isinstance(type_obj, dict) and "name" in type_obj
        else None
    )

    return WorkItem(
        number=int(entry["number"]),
        title=str(entry.get("title") or ""),
        body=str(entry.get("body") or ""),
        state=str(entry.get("state", "")),
        labels=sorted(labels, key=str.casefold),
        html_url=str(entry.get("html_url", "")),
        updated_at=str(entry.get("updated_at", "")),
        created_at=str(entry.get("created_at", "")),
        author_login=str(entry.get("user", {}).get("login", "")),
        kind=kind,
        issue_type=issue_type,
    )


class GitHubClient:
    """Lightweight wrapper around ``gh`` and git remote discovery."""

    def json(self, argv: Sequence[str]) -> object:
        """Run ``gh`` and decode its stdout as JSON."""

        completed = run(("gh",) + tuple(argv))
        try:
            return json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"failed to parse JSON from {' '.join(argv)}:\n"
                f"{completed.stdout}"
            ) from exc

    def detect_repo(self) -> str:
        """Infer the active GitHub repository from known git remotes."""

        for remote in REPO_DETECTION_ORDER:
            completed = run(
                ("git", "remote", "get-url", remote),
                check=False,
            )
            if completed.returncode != 0:
                continue
            repo = self.parse_repo_from_remote(completed.stdout.strip())
            if repo:
                return repo

        raise RuntimeError(
            "could not infer a GitHub repository from git remotes; "
            "pass --repository owner/name"
        )

    @staticmethod
    def parse_repo_from_remote(remote_url: str) -> Optional[str]:
        """Extract an ``owner/repository`` pair from a supported remote."""

        prefixes = (
            "git@github.com:",
            "github.com:",
            "https://github.com/",
            "http://github.com/",
        )
        normalised = remote_url
        for prefix in prefixes:
            if normalised.startswith(prefix):
                normalised = normalised[len(prefix):]
                break
        else:
            return None

        if normalised.endswith(".git"):
            normalised = normalised[:-4]

        parts = normalised.split("/")
        if len(parts) != 2 or not all(parts):
            return None
        return f"{parts[0]}/{parts[1]}"

    def list_repo_labels(self, repo: str) -> List[LabelDefinition]:
        """Fetch, deduplicate, and sort repository labels with descriptions."""

        labels: Dict[str, LabelDefinition] = {}
        payload = self.json(
            ("api", f"repos/{repo}/labels", "--paginate", "--slurp")
        )
        if not isinstance(payload, list):
            raise RuntimeError("unexpected label payload from GitHub")

        for page_payload in payload:
            if not isinstance(page_payload, list):
                raise RuntimeError("unexpected label page payload from GitHub")
            for entry in page_payload:
                if isinstance(entry, dict) and "name" in entry:
                    name = str(entry["name"])
                    labels[name.casefold()] = LabelDefinition(
                        name=name,
                        description=str(entry.get("description") or ""),
                    )

        return sorted(labels.values(), key=lambda label: label.name.casefold())

    @staticmethod
    def format_search_date(timestamp: datetime) -> str:
        """Format a timestamp for GitHub search date qualifiers."""

        return timestamp.astimezone(timezone.utc).strftime("%Y-%m-%d")

    def build_search_query(
        self,
        repo: str,
        kind: str,
        search_options: SearchOptions,
    ) -> str:
        """Build the GitHub search query for issues or pull requests."""

        terms = [f"repo:{repo}", f"is:{kind}"]
        if search_options.include_open and not search_options.include_closed:
            terms.append("is:open")
        elif search_options.include_closed and not search_options.include_open:
            terms.append("is:closed")
        if search_options.cutoff is not None:
            qualifier = "created" if search_options.created else "updated"
            terms.append(
                f"{qualifier}:>="
                f"{self.format_search_date(search_options.cutoff)}"
            )
        return " ".join(terms)

    def search_items(
        self,
        repo: str,
        kind: str,
        search_options: SearchOptions,
    ) -> List[WorkItem]:
        """Search GitHub for matching issues or pull requests."""

        query = self.build_search_query(repo, kind, search_options)
        page = 1
        per_page = 100
        items: List[WorkItem] = []
        sort_field = "created" if search_options.created else "updated"

        while True:
            payload = self.json(
                (
                    "api",
                    "search/issues",
                    "--method",
                    "GET",
                    "-f",
                    f"q={query}",
                    "-f",
                    f"per_page={per_page}",
                    "-f",
                    f"page={page}",
                    "-f",
                    f"sort={sort_field}",
                    "-f",
                    "order=desc",
                )
            )

            if not isinstance(payload, dict):
                raise RuntimeError("unexpected search payload from GitHub")
            raw_items = payload.get("items", [])
            if not isinstance(raw_items, list):
                raise RuntimeError("unexpected items payload from GitHub")
            if not raw_items:
                break

            for entry in raw_items:
                item = work_item_from_search_result(entry, kind)
                item_time = parse_github_timestamp(
                    item.created_at
                    if search_options.created
                    else item.updated_at
                )
                if (
                    search_options.cutoff is not None
                    and item_time < search_options.cutoff
                ):
                    return items

                items.append(item)
                if (
                    search_options.limit is not None
                    and len(items) >= search_options.limit
                ):
                    return items

            if len(raw_items) < per_page:
                break
            page += 1

        items.sort(
            key=lambda item: parse_github_timestamp(
                item.created_at if search_options.created else item.updated_at
            ),
            reverse=True,
        )
        return items

    def get_item(self, repo: str, number: int) -> WorkItem:
        """Fetch a single issue or pull request by number."""

        entry = self.json(("api", f"repos/{repo}/issues/{number}"))
        if not isinstance(entry, dict):
            raise RuntimeError(f"unexpected payload for #{number}")
        kind = "pr" if "pull_request" in entry else "issue"
        return work_item_from_search_result(entry, kind)

    def add_labels(
        self,
        repo: str,
        item: WorkItem,
        labels: Sequence[str],
    ) -> None:
        """Apply new labels through ``gh``."""

        if not labels:
            return

        argv: List[str] = [
            "gh",
            "api",
            "--method",
            "POST",
            f"repos/{repo}/issues/{item.number}/labels",
        ]
        for label in labels:
            argv.extend(["-f", f"labels[]={label}"])

        _emit_completed_output(run(tuple(argv)))

    def remove_label(self, repo: str, item: WorkItem, label: str) -> None:
        """Remove one label through ``gh``."""

        _emit_completed_output(
            run(
                (
                    "gh",
                    "api",
                    "--method",
                    "DELETE",
                    f"repos/{repo}/issues/{item.number}/labels/{label}",
                )
            )
        )

    def list_issue_types(self, repo: str) -> List[IssueTypeDefinition]:
        """Fetch issue types for the repository's organisation."""

        owner = repo.split("/")[0]
        completed = run(
            ("gh", "api", f"orgs/{owner}/issue-types"),
            check=False,
        )
        if completed.returncode != 0:
            return []
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError:
            return []
        if not isinstance(payload, list):
            return []
        result: List[IssueTypeDefinition] = []
        for entry in payload:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name")
            if not isinstance(name, str) or not name:
                continue
            description = str(entry.get("description") or "")
            type_id = int(entry.get("id") or 0)
            result.append(
                IssueTypeDefinition(
                    name=name, description=description, type_id=type_id
                )
            )
        return result

    def set_issue_type(
        self,
        repo: str,
        item: WorkItem,
        issue_type: IssueTypeDefinition,
    ) -> None:
        """Set the issue type on a GitHub issue."""

        if issue_type.type_id:
            body = json.dumps({"type": {"id": issue_type.type_id}})
        else:
            body = json.dumps({"type": {"name": issue_type.name}})
        _emit_completed_output(
            run(
                (
                    "gh",
                    "api",
                    "--method",
                    "PATCH",
                    f"repos/{repo}/issues/{item.number}",
                    "--input",
                    "-",
                ),
                input_text=body,
            )
        )

    def post_comment(self, repo: str, item: WorkItem, body: str) -> None:
        """Post a comment on an issue or pull request through ``gh``."""

        completed = run(
            (
                "gh",
                "api",
                "--method",
                "POST",
                f"repos/{repo}/issues/{item.number}/comments",
                "--input",
                "-",
            ),
            input_text=json.dumps({"body": body}),
            check=False,
        )
        if completed.returncode != 0:
            print(
                colourise(
                    f"Warning: failed to post comment on "
                    f"{item.kind.upper()} #{item.number}",
                    "yellow",
                    stream=sys.stderr,
                ),
                file=sys.stderr,
            )
            if completed.stderr and completed.stderr.strip():
                print(
                    colourise(
                        completed.stderr.strip(),
                        "yellow",
                        stream=sys.stderr,
                    ),
                    file=sys.stderr,
                )
