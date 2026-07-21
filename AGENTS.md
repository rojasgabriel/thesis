# Project Instructions

Behavioral-only analysis code for Churchland Lab thesis work.

- Start from the actual branch, working tree, notebooks, scripts, and tests.
- Read `README.md` before changing the data-access or package surface.
- Keep behavioral analysis here; electrophysiology work belongs in the
  `ephys` repository.
- Historical notebooks may use `djchurchland`; new reusable work should
  follow the active `labdata` and local plugin path in `pyproject.toml`.
- Preserve the current uncommitted migration unless the task explicitly owns it.
- Use Notion for current priorities and decisions, not as a copy of repository
  state.

Run the narrowest relevant checks, then `uvx ruff check .`,
`uvx ruff format --check .`, and `uv run pytest` when the full migration
surface is affected.
