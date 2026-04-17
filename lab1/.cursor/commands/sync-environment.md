# Sync environment

Install and refresh dependencies (including dev) using uv:

```bash
uv sync --all-groups
```

Report any resolver errors. If `pyproject.toml` or `uv.lock` changed, remind the user to commit the lockfile when that matches course expectations.
