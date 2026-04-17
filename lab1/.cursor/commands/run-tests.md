# Run tests

From the repository root, run the full pytest suite with the project environment:

```bash
uv run pytest
```

If dependencies are missing or stale, run `uv sync --all-groups` first, then repeat pytest.

Summarize failures with file paths and suggested fixes; apply minimal code changes to make tests pass unless the user asked only for diagnosis.
