"""Shared data structures for the labelling workflow."""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional


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
class LabelSuggestion:
    """Normalised label and issue-type suggestions produced by the AI."""

    add_labels: List[str]
    remove_labels: List[str]
    reason: str
    issue_type: Optional[str] = None


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


@dataclass(frozen=True)
class SuggestionResult:
    """A work item paired with its normalised AI label suggestion."""

    item: WorkItem
    label_suggestion: LabelSuggestion
    model: str = ""


class UserQuit(Exception):
    """Raised when the user explicitly requests immediate termination."""
