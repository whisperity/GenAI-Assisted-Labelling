"""Shared data structures and pure data helpers for the labelling workflow."""

import dataclasses
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Sequence


def parse_github_timestamp(timestamp: str) -> datetime:
    """Parse a GitHub API timestamp into a timezone-aware ``datetime``."""

    return datetime.fromisoformat(timestamp.replace("Z", "+00:00"))


@dataclass
class WorkItem:  # pylint: disable=too-many-instance-attributes
    """Normalised GitHub issue or pull request data used by the workflow."""

    number: int
    title: str
    body: str
    state: str
    labels: List[str]
    html_url: str
    updated_at: str
    created_at: str
    author_login: str
    kind: str
    issue_type: Optional[str] = None
    assignees: List[str] = dataclasses.field(default_factory=list)


@dataclass(frozen=True)
class ModelSpec:
    """Parsed model selection with an optional reasoning-effort override."""

    provider: str
    model: Optional[str] = None
    reasoning_effort: Optional[str] = None


@dataclass(frozen=True)
class SearchOptions:
    """Parameters controlling GitHub issue and pull-request searches."""

    include_open: bool
    include_closed: bool
    created: bool
    cutoff: Optional[datetime]
    limit: Optional[int]


@dataclass(frozen=True)
class LabelDefinition:
    """Repository label metadata available to the AI backend."""

    name: str
    description: str


@dataclass(frozen=True)
class IssueTypeDefinition:
    """Repository issue-type metadata available to the AI backend."""

    name: str
    description: str
    type_id: int = 0


@dataclass(frozen=True)
class PreviewBlock:
    """A markdown-aware body block used to build issue previews."""

    kind: str
    text: str


def _filter_known_labels(
    raw: object,
    valid_lookup: Dict[str, str],
) -> List[str]:
    """Filter raw label entries to a sorted, deduplicated canonical list."""

    if not isinstance(raw, list):
        return []

    chosen: List[str] = []
    seen = set()
    for entry in raw:
        if not isinstance(entry, str):
            continue
        canonical = valid_lookup.get(entry.casefold())
        if canonical is None or canonical.casefold() in seen:
            continue
        seen.add(canonical.casefold())
        chosen.append(canonical)
    return sorted(chosen, key=str.casefold)


@dataclass(frozen=True)
class LabelSuggestion:
    """Normalised label and issue-type suggestions produced by the AI."""

    add_labels: List[str]
    remove_labels: List[str]
    reason: str
    issue_type: Optional[str] = None

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    @classmethod
    def from_raw(
        cls,
        raw: Dict[str, object],
        valid_labels: Sequence["LabelDefinition"],
        existing_labels: Sequence[str],
        *,
        valid_issue_types: Sequence["IssueTypeDefinition"] = (),
        current_issue_type: Optional[str] = None,
    ) -> "LabelSuggestion":
        """Build a normalised suggestion from a raw model response dict."""

        valid_lookup = {lbl.name.casefold(): lbl.name for lbl in valid_labels}
        existing_lookup = {lbl.casefold() for lbl in existing_labels}

        add_labels = [
            lbl
            for lbl in _filter_known_labels(
                raw.get("add_labels", []), valid_lookup,
            )
            if lbl.casefold() not in existing_lookup
        ]
        remove_labels = [
            lbl
            for lbl in _filter_known_labels(
                raw.get("remove_labels", []), valid_lookup,
            )
            if lbl.casefold() in existing_lookup
        ]
        reason = str(raw.get("reason", "")).strip()
        issue_type = _select_issue_type(
            raw.get("issue_type"),
            valid_issue_types,
            current_issue_type,
        )
        return cls(add_labels, remove_labels, reason, issue_type=issue_type)


def _select_issue_type(
    raw_type: object,
    valid_issue_types: Sequence[IssueTypeDefinition],
    current_issue_type: Optional[str],
) -> Optional[str]:
    """Resolve a raw issue-type suggestion to a canonical name (or None)."""

    if not valid_issue_types:
        return None
    if not isinstance(raw_type, str) or not raw_type.strip():
        return None
    valid_lookup = {t.name.casefold(): t.name for t in valid_issue_types}
    canonical = valid_lookup.get(raw_type.strip().casefold())
    if canonical is None:
        return None
    if canonical.casefold() == (current_issue_type or "").casefold():
        return None
    return canonical


@dataclass(frozen=True)
class SuggestionResult:
    """A work item paired with its normalised AI label suggestion."""

    item: WorkItem
    label_suggestion: LabelSuggestion
    model: str = ""
    applied_assignee: Optional[str] = None


class UserQuit(Exception):
    """Raised when the user explicitly requests immediate termination."""
