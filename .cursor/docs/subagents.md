# Subagents and delegation (playbook)

This project is small, but these habits keep automated work reliable.

## When to use readonly exploration

Use a **readonly** subagent or narrow search when you need to:

- Map where a symbol is defined or used across the tree.
- Compare patterns (`src/` vs `tests/`) before editing.
- Answer “how does X work?” without changing files.

Avoid write access until the goal and files to touch are clear.

## When to use shell-oriented work

Prefer a **shell**-style workflow when you need to:

- Run `uv sync --all-groups` after dependency edits.
- Run `uv run pytest` and iterate on failures.
- Verify CLI behavior (`uv run detect --help` or similar).

Always use the repository root as the working directory unless the task explicitly says otherwise.

## Splitting work

- **Parallel**: independent searches (e.g. “find all OpenCV usage” + “find all CLI options”) can run together.
- **Sequential**: run tests only after code changes are coherent; avoid overlapping edits on the same file from two agents.

## Handoff checklist

When one agent step hands off to another, include:

1. Goal in one sentence.
2. Paths already read or changed.
3. Command(s) run and whether they passed.
4. Open risks (e.g. “needs camera/images to validate end-to-end”).
