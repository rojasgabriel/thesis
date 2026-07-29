# Project Instructions

Behavioral-only analysis code for Churchland Lab thesis work.

- Start from the actual branch, working tree, notebooks, scripts, and tests.
- Read `README.md` and `docs/MIGRATION.md` before changing the data-access or
  package surface.
- Keep behavioral analysis here; electrophysiology work belongs in the
  `ephys` repository.
- Maintained reusable work uses `labdata`, `labdata_plugin/`, and
  `src/behavior_analyses/`. Historical `djchurchland` notebooks live under
  `historical/djchurchland/`.
- Do not reintroduce `/Users/gabriel` paths or machine-local dependency paths.
- Use Notion for current priorities and decisions, not as a copy of repository
  state.

Run the narrowest relevant checks, then `uvx ruff check .`,
`uvx ruff format --check .`, and
`uv run python -m unittest discover -s tests -v` when the full migration
surface is affected.
