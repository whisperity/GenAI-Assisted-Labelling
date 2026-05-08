"""Shared test helpers for the ``ai_labelling`` package."""
# pylint: disable=too-many-arguments,too-many-positional-arguments

from ai_labelling.models import ItemKind, WorkItem


def make_item(
    number: int,
    title: str,
    *,
    kind: ItemKind = "issue",
    state: str = "open",
    labels=None,
    assignees=None,
) -> WorkItem:
    """Create a small ``WorkItem`` fixture for tests."""

    return WorkItem(
        number=number,
        title=title,
        body="Body text",
        state=state,
        labels=list(labels or []),
        assignees=list(assignees or []),
        html_url=f"https://example.invalid/{number}",
        updated_at="2026-05-01T00:00:00Z",
        created_at="2026-05-01T00:00:00Z",
        author_login="octocat",
        kind=kind,
    )
