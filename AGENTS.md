# Agent guide (ME 235B — me235b)

This repository is a small Python package named **me235b** (ArUco-style detection using OpenCV, NumPy, Rich, Typer). Python **3.14+** is required (`requires-python` in `pyproject.toml`).

## Quick commands

Run from the repository root:

| Action | Command |
|--------|---------|
| Install deps (including dev) | `uv sync --all-groups` |
| Run tests | `uv run pytest` |
| CLI entry point (editable install) | `uv run detect --help` |

The package exposes the `detect` script via `[project.scripts]` in `pyproject.toml`.

## Layout

- `src/` — library and CLI (`cli.py`, `detector.py`, …); installed as the **`me235b`** package
- `tests/` — pytest tests
- `assets/` — optional images or fixtures for local experiments
- `main.py` — thin entry if used for class scaffolding
- `answers.md` — written lab answers (treat as student-owned content)

## Conventions for changes

- Prefer **small, focused diffs**; match existing style and imports.
- Run **`uv run pytest`** before finishing a task that touches behavior.
- Do not commit virtualenvs, `__pycache__`, `.pytest_cache`, or `*.egg-info/` (see `.gitignore`).

## Cursor-specific configuration

Project-local agent context lives under **`.cursor/`**:

| Path | Purpose |
|------|---------|
| `.cursor/rules/*.mdc` | Rules (scoped or always-on) for Cursor |
| `.cursor/commands/*.md` | Reusable slash commands in chat (type `/` in agent input) |
| `.cursor/docs/` | Extra maintainer notes (e.g. subagent playbook) |
| `.cursor/README.md` | Index of those files |

VS Code / Cursor **tasks** live in `.vscode/tasks.json` (labels like `pytest`, `uv: sync`).

**`.cursorignore`** (repo root) trims what Cursor indexes—virtualenvs, caches, and build metadata—without affecting Git.

## Subagents and delegation

For multi-step or wide exploration inside Cursor, see **`.cursor/docs/subagents.md`** for when to prefer readonly exploration vs shell execution.
