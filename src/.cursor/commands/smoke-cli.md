# Smoke-test CLI

Verify the Typer CLI is wired and imports succeed:

```bash
uv run detect --help
```

If this fails, trace the error (missing deps, bad import, script entry). Fix `lab1/cli.py`, `pyproject.toml` `[project.scripts]`, or packaging layout as needed.
