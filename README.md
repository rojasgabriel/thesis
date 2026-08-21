# thesis

Behavior and electrophysiology analysis code for Churchland Lab thesis work.

## Setup

```bash
uv sync
uvx ruff check .
uvx ruff format --check .
uvx ty check
uv run python -m unittest discover -s tests -v
```

Python 3.11 is the supported environment. `fit-psychometric` is vendored under
`third_party/fit_psychometric`, so the project does not need a sibling checkout.

## Layout

- `src/thesis/behavior/` — behavioral analysis and timing code
- `src/thesis/ephys/` — electrophysiology configuration and reusable helpers
- `scripts/` — maintained analyses, diagnostics, and interactive tools
- `notebooks/` — exploration grouped by behavior or ephys
- `labdata_plugin/schema.py` — the live `EventMapping` and `LocomotionPeaks` tables
- `tests/` — unit and schema-contract checks
- `ANALYSES.md` — maintained scientific entry points and interpretation limits

## Database safety

Importing `labdata_plugin` alone does not register any DataJoint tables. Import
`labdata_plugin.schema` only when the live plugin tables are needed.

The repository merge does not create, alter, populate, rename, or delete any
database table. The removed behavior schemas remain absent. `EventMapping` and
`LocomotionPeaks` keep their existing table definitions and data.

Behavior code prefers the installed Chipmunk LabData plugin. A local checkout
can be selected with `CHIPMUNK_PLUGIN_PATH` when needed.

See `docs/MIGRATION.md` for the repository merge record.
