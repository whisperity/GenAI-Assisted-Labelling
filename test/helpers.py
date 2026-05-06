"""Shared test helpers for the ``ai_labelling`` package."""

from ai_labelling.models import WorkItem


def make_item(number, title, *, kind="issue", state="open", labels=None):
    """Create a small ``WorkItem`` fixture for tests."""

    return WorkItem(
        number=number,
        title=title,
        body="Body text",
        state=state,
        labels=list(labels or []),
        html_url=f"https://example.invalid/{number}",
        updated_at="2026-05-01T00:00:00Z",
        created_at="2026-05-01T00:00:00Z",
        author_login="octocat",
        kind=kind,
    )
