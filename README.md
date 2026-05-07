# GenAI-Assisted-Labelling

**Generative AI assisted labelling of GitHub issues and pull requests.**

`ai-labelling` reads issues and pull requests from a GitHub repository, asks a
generative AI model to suggest the most appropriate labels (and, for issues,
the most appropriate issue type) based on the title and body, and offers an
interactive review-and-apply loop that lets you accept, skip, or batch-apply
the suggestions through the GitHub CLI.

## Contents

1. [Dependencies](#dependencies)
2. [Quick start](#quick-start)
3. [How it works](#how-it-works)
4. [Repository selection](#repository-selection)
5. [Filtering issues and pull requests](#filtering-issues-and-pull-requests)
6. [Targeting one specific item](#targeting-one-specific-item)
7. [Choosing an AI provider and model](#choosing-an-ai-provider-and-model)
8. [Reasoning effort](#reasoning-effort)
9. [Dry-run, force, and label removals](#dry-run-force-and-label-removals)
10. [Posting an audit comment](#posting-an-audit-comment)
11. [Interactive prompts](#interactive-prompts)
12. [Debugging output](#debugging-output)
13. [Common recipes](#common-recipes)
14. [Exit codes](#exit-codes)
15. [Development](#development)

## Dependencies

- Python 3.10 or newer.
- The GitHub CLI client, `gh`: <https://cli.github.com>. The current
  `gh auth login` session must have permission to read issues, pull requests,
  labels, and (for issue types) the parent organisation.
- Access to one of the supported generative AI services:
  - **Codex CLI** — a ChatGPT subscription with Codex
    (<https://chatgpt.com/codex>) access, plus the `codex` binary on `PATH`
    and signed in.
  - **Anthropic Claude Platform API** — an API key from
    <https://platform.claude.com> exported as `ANTHROPIC_API_KEY`.

No additional Python packages are required at runtime; `ai-labelling` only
relies on the standard library.

## Quick start

From inside a Git checkout of the repository you want to label:

```bash
$ /path/to/ai-labelling
```

Or against any repository:

```bash
$ ./ai-labelling --repository owner/name
```

The default behaviour is:

- Look at **open issues** (pull requests are skipped unless asked for).
- Filter by **last updated** in the last **24 hours**.
- Use the **Codex** backend with model `gpt-5.4-mini` and reasoning effort
  `low`.
- Suggest **additions only** — no labels are removed unless
  `--allow-label-removals` is used.
- Prompt for confirmation on **every item** and **every label**.

The tool prints a one-line summary of every accepted or rejected change at the
end of the run.

## How it works

```
┌─────────────────────────┐    ┌─────────────────┐    ┌────────────────────┐
│ git remote / --repo     │ ─▶ │ gh search       │ ─▶ │ Item selection     │
│ + filter flags          │    │ (issues / prs)  │    │ (y/n/a/d/q)        │
└─────────────────────────┘    └─────────────────┘    └────────────────────┘
                                                                │
                                                                ▼
┌─────────────────────────┐    ┌─────────────────┐    ┌────────────────────┐
│ Apply via gh            │ ◀─ │ Review prompts  │ ◀─ │ AI batch           │
│ (add / remove / type)   │    │ per label/type  │    └────────────────────┘
└─────────────────────────┘    └─────────────────┘
```

1. The repository's labels (and, for issues, the parent organisation's issue
   types) are fetched once via `gh api`.
2. The configured filters drive a `gh api search/issues` query that returns
   the matching issues and/or pull requests.
3. You confirm which items go to the AI.
4. Each selected item is sent to the configured AI backend in parallel
   (`ProcessPoolExecutor` with one worker per CPU). Each prompt contains only
   the item's **title**, **body**, and the **valid label list**, plus the
   issue type list when applicable.
5. The model returns a JSON object with `add_labels`, optional
   `remove_labels`, optional `issue_type`, and a short `reason`. Labels not in
   the valid list are dropped silently.
6. For each item you confirm every suggested change individually (or
   batch-accept with `a`). All decisions for the item are collected first.
7. Once collection for the item finishes, confirmed changes are applied
   through `gh api` in a batch (label additions go in one request);
   quitting mid-collection leaves the item untouched.

## Repository selection

`--repository OWNER/NAME`
:  Operate on a specific GitHub repository. Required when the current working
   directory is not a Git checkout of the target repository.

If `--repository` is omitted, `ai-labelling` inspects the Git remotes of the
current working directory in this order: `upstream/push`, `upstream`,
`origin`. The first remote that points to `github.com` wins.

```bash
# Auto-detect from the current Git checkout.
$ cd ~/work/llvm-project
$ ai-labelling

# Override the auto-detection.
$ ai-labelling --repository llvm/llvm-project
```

## Filtering issues and pull requests

By default the tool considers **open issues** updated in the last 24 hours.
Combine the flags below to widen or narrow the search.

| Flag                       | Effect                                                             |
|----------------------------|--------------------------------------------------------------------|
| `--issues` / `--no-issues` | Include or exclude issues. Default: include.                       |
| `--prs` / `--no-prs`       | Include or exclude pull requests. Default: exclude.                |
| `--open` / `--no-open`     | Include or exclude open items. Default: include.                   |
| `--closed` / `--no-closed` | Include or exclude closed items. Default: exclude.                 |
| `--updated` / `--created`  | Filter on last-update or creation timestamp. Default: `--updated`. |
| `--date DATE`              | Lower bound on the chosen timestamp. Default: 24 hours ago.        |
| `--limit N`                | Cap the number of matching items.                                  |

`--date` accepts an ISO `YYYY-MM-DD` date or a full `YYYY-MM-DDTHH:MM:SS`
date-time, optionally with a timezone offset. Naive values are interpreted in
the host's local timezone and normalised to UTC. Pass `0` or `all` to disable
the cutoff entirely.

```bash
# Open issues opened on or after 1 May 2026.
$ ai-labelling --created --date 2026-05-01

# Both issues and PRs touched in the last 24 hours.
$ ai-labelling --prs

# Closed issues only, no time cutoff, max 50 items.
$ ai-labelling --no-open --closed --date all --limit 50

# All open and closed PRs from a precise instant.
$ ai-labelling --no-issues --prs --closed --date 2026-05-01T12:00:00+02:00
```

Disabling everything in a category (e.g. `--no-issues --no-prs`) makes the
tool exit cleanly with an empty match.

## Targeting one specific item

`--id NUMBER`
:  Fetch one issue or pull request by its number and skip all filters. The
   kind (issue vs. PR) is detected automatically.

```bash
# Re-label issue or PR #4242 in the auto-detected repository.
$ ai-labelling --id 4242

# Same against another repo.
$ ai-labelling --repository llvm/llvm-project --id 4242
```

`--id` is the cleanest way to revisit a single item after the AI got something
wrong: every other filter flag is ignored.

## Choosing an AI provider and model

Use `--model` with one of three forms:

```
PROVIDER
PROVIDER:MODEL
PROVIDER:MODEL:REASONING-LEVEL
```

| Form                       | Meaning                                                                |
|----------------------------|------------------------------------------------------------------------|
| `PROVIDER`                 | Use the provider's hard-coded default model. No reasoning effort sent. |
| `PROVIDER:*`               | Ask the provider for its current default model (Anthropic only).       |
| `PROVIDER:MODEL`           | Use a specific model name.                                             |
| `PROVIDER:MODEL:REASONING` | As above, with an explicit reasoning effort.                           |
| `PROVIDER:*:REASONING`     | Provider-default model with an explicit effort.                        |

Supported providers and their hard-coded defaults:

| Provider          | Default model               | Reasoning levels                        |
|-------------------|-----------------------------|-----------------------------------------|
| `codex` (default) | `gpt-5.4-mini`              | `low`, `medium`, `high`, `xhigh`        |
| `anthropic`       | `claude-haiku-4-5-20251001` | `low`, `medium`, `high`, `xhigh`, `max` |

The default `--model` value is `codex:gpt-5.4-mini:low`.

```bash
# Codex with the hard-coded default model and no effort override.
$ ai-labelling --model codex

# Codex, named model, high effort.
$ ai-labelling --model codex:gpt-5.4:high

# Anthropic, hard-coded Haiku default.
$ ai-labelling --model anthropic

# Anthropic, latest model from /v1/models, max effort.
$ ai-labelling --model anthropic:*:max

# Anthropic with a specific named model.
$ ai-labelling --model anthropic:claude-sonnet-4-20250514:medium
```

The Codex backend shells out to `codex exec --sandbox read-only --ephemeral`
with a JSON output schema. The Anthropic backend posts directly to
`/v1/messages` and reads `ANTHROPIC_API_KEY` from the environment; it falls
back automatically and retries without the effort parameter if the model
rejects it with HTTP 400.

## Reasoning effort

Reasoning effort is **optional**. Omitting the third segment from the
`--model` value causes the tool to **not send any effort at all**, so the
provider applies its own default. Pass an explicit level only when you want
to override that default.

```bash
# Send no effort hint at all.
$ ai-labelling --model anthropic

# Force Anthropic's "max" effort.
$ ai-labelling --model anthropic:*:max

# Set Codex effort without changing the model.
$ ai-labelling --model codex:gpt-5.4-mini:high
```

## Dry-run, force, and label removals

| Flag                     | Effect                                                                                                                    |
|--------------------------|---------------------------------------------------------------------------------------------------------------------------|
| `--dry-run`              | Run the full pipeline but never call `gh` to mutate labels or types.                                                      |
| `--force`                | Skip the per-item and per-label confirmation prompts; apply everything the AI suggests.                                   |
| `--allow-label-removals` | Let the AI propose removals and let the tool execute them. Without this flag, removals are neither requested nor applied. |

`--dry-run` and `--force` combine: `--dry-run --force` quietly produces the
full change summary without any prompts and without touching GitHub. Useful
for cost estimation and prompt-tuning runs.

```bash
# What WOULD change in the last 24 h, no GitHub writes.
$ ai-labelling --dry-run

# Same, fully unattended (no prompts at all).
$ ai-labelling --dry-run --force

# Allow the AI to drop labels it considers stale.
$ ai-labelling --allow-label-removals

# Aggressive nightly run: full triage, removals allowed, no prompts.
$ ai-labelling --force --allow-label-removals
```

When `--force` is set without `--dry-run`, the tool prints a 15-second red
warning before continuing. Press `Ctrl-C` to abort.

## Posting an audit comment

`--comment-reason`
:  After applying labels and the issue type for each item, post a GitHub
   comment summarising the model used, the model's reasoning, and which
   suggestions were accepted or rejected. The comment is rendered in
   Markdown with strike-through for rejected suggestions. Has **no** effect
   under `--dry-run`.

```bash
$ ai-labelling --id 4242 --comment-reason
$ ai-labelling --force --allow-label-removals --comment-reason
```

The comment links back to this project's repository, includes the short Git
SHA of the running script (so reviewers can see exactly which version made
the call), and lists every original suggestion alongside whether the
operator accepted it.

## Interactive prompts

Two prompt styles are used. Both follow Git's interactive
`git add --patch`-style single-letter answers.

**Item selection** (after the matching list is printed):

```
Handle ISSUE #123 with AI? [y/n/a/d/q/?]
```

| Key | Meaning                                         |
|-----|-------------------------------------------------|
| `y` | Send this item to the AI.                       |
| `n` | Skip this item.                                 |
| `a` | Send this item and all remaining matched items. |
| `d` | Stop prompting; do not send any more items.     |
| `q` | Quit immediately.                               |
| `?` | Print the help legend.                          |

**Label / issue-type apply** (after the AI returns suggestions):

```
SET issue type to "Bug" for issue #123? [y/n/q/?]
ADD the label "performance" to issue #123? [y/n/a/d/q/?]
**REMOVE** the label "needs-triage" from issue #123? [y/n/a/d/q/?]
```

| Key | Meaning                                                              |
|-----|----------------------------------------------------------------------|
| `y` | Approve this change.                                                 |
| `n` | Skip this change.                                                    |
| `a` | Approve this change and all remaining changes in the current bucket. |
| `d` | Stop prompting for this bucket; skip the rest.                       |
| `q` | Quit immediately.                                                    |
| `?` | Print the help legend.                                               |

All decisions for one item (issue type, additions, removals) are collected
**before** any GitHub API call fires. Approved label additions are sent in
a single `POST /repos/{owner}/{repo}/issues/{number}/labels` request;
removals run as one `DELETE` per label (GitHub has no batch-remove
endpoint). This means a `q` (quit) part-way through prompting an item
aborts the whole item cleanly — no partial label state is ever written.

`q` raises `UserQuit` everywhere, terminating the run with exit code `0`.

## Debugging output

`ai-labelling` reads the `DEBUG` environment variable.

| `DEBUG`             | Output                                                                                      |
|---------------------|---------------------------------------------------------------------------------------------|
| unset / `0` / empty | None.                                                                                       |
| `1`                 | Trace each subprocess command line and each Anthropic HTTP request (with secrets redacted). |
| `2`                 | Plus a sanitised AI prompt template (issue title, body, and label lists redacted).          |
| `3` or higher       | Plus the full prompt as sent.                                                               |

```bash
$ DEBUG=1 ai-labelling --id 42
$ DEBUG=2 ai-labelling --dry-run
$ DEBUG=3 ai-labelling --model anthropic --id 42
```

Comments rendered with `--comment-reason` are also echoed to stderr at
`DEBUG=2` and above so you can preview them before they are posted.

## Common recipes

**Triage everything new since yesterday morning, interactively:**

```bash
$ ai-labelling --date "$(date -u -v-1d +%Y-%m-%dT09:00:00Z)"
```

**Mass-clean a backlog of stale labels:**

```bash
$ ai-labelling \
    --no-open --closed \
    --date all \
    --allow-label-removals \
    --limit 100
```

**Nightly cron, dry-run report only:**

```bash
$ ai-labelling --dry-run --force --prs --comment-reason  # comment-reason no-op under --dry-run
```

**Audit one specific PR with full reasoning posted as a comment:**

```bash
$ ai-labelling --id 9001 --no-issues --prs --comment-reason
```

**Switch provider for a single run without changing defaults:**

```bash
$ ai-labelling --model anthropic:*:high --id 42
```

**Use Anthropic against a non-default repository, no effort hint:**

```bash
$ ANTHROPIC_API_KEY=sk-ant-... \
    ai-labelling --repository owner/name --model anthropic
```

**See exactly what the prompt looks like before committing:**

```bash
$ DEBUG=2 ai-labelling --dry-run --id 42
```

## Exit codes

| Code | Meaning                                                                                                      |
|------|--------------------------------------------------------------------------------------------------------------|
| `0`  | Success, or user-requested quit (`q` in any prompt).                                                         |
| `1`  | A `RuntimeError` from the workflow (no labels in the repo, malformed GitHub payload, AI HTTP failure, etc.). |
| `N`  | The `gh` CLI exited with code `N`. The captured `stdout` and `stderr` are forwarded.                         |

## Development

The package layout mirrors a typical small CLI:

```
ai_labelling/
  args.py          — argparse setup, model spec parser
  comment.py       — Markdown audit-comment builder
  config.py        — non-ANSI constants and sentinels
  formatting.py    — preview/markdown rendering, summaries, item details
  github_client.py — gh wrapper: search, get, list labels, add/remove labels
  interaction.py   — prompt_confirmation / prompt_yes_no / help text
  models.py        — dataclasses + LabelSuggestion.from_raw, timestamp parser
  shell.py         — subprocess.run wrapper + git-based version helper
  terminal.py      — ANSI colours, debug log, prompt sanitiser
  workflow.py      — LabellingWorkflow coordinator
  backends/
    base.py        — AIBackend, prompt builder
    anthropic.py   — direct /v1/messages backend
    codex.py       — codex CLI backend
ai-labelling       — executable entrypoint
test/              — unit tests mirroring the package layout
```

Run the test suite with the standard library runner:

```bash
$ python3 -m unittest discover -s test
```

The tests are fully mocked — no API key, no network access, and no `gh`
binary are required to run them.

Lint with `pylint` and `pycodestyle`:

```bash
$ pylint ai_labelling/ test/
$ pycodestyle ai_labelling/ test/
```
