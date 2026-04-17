# Preflight before sharing code

1. Run `uv sync --all-groups` if `pyproject.toml` / `uv.lock` changed.
2. Run `uv run pytest` and ensure green.
3. Run `uv run detect --help` for a quick CLI smoke test.
4. Skim the diff: no accidental secrets (API keys, personal paths), no `.venv` or `__pycache__` files staged.

Produce a short summary suitable for a commit message or PR description: what changed and why.
