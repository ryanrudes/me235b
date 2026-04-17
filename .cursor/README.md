# Cursor agent configuration (this repo)

This folder is safe to commit: it contains **no secrets**, only prompts and rules for AI-assisted work on the **me235b** project (ME 235B).

## Contents

| Item | Description |
|------|-------------|
| `rules/` | `.mdc` rule files with YAML frontmatter (`alwaysApply` and/or `globs`). |
| `commands/` | Markdown bodies for project **slash commands** (Cursor loads them from here). |
| `docs/subagents.md` | Guidance on delegating work (explore vs shell, scope). |

Root-level **[AGENTS.md](../AGENTS.md)** is the primary handoff document for any coding agent (stack, layout, commands).

## Optional additions

- **MCP**: add `.cursor/mcp.json` locally if you use MCP servers; do not commit secrets.
- **`.cursorignore`**: patterns to trim indexing context (see repo root if present).
- **Hooks**: project hooks live in **`.cursor/hooks.json`** with scripts under **`.cursor/hooks/`** (see [Cursor hook docs](https://cursor.com/docs)). This repo does not ship active hooks so clones stay zero-config; add them when you need gating (for example `beforeShellExecution`) or post-edit automation (`afterFileEdit`).
