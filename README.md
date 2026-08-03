# behavior_analyses

Behavioral-only analysis code for thesis work in the Churchland Lab.

## Setup and checks

```bash
uv sync
uvx ruff check .
uvx ruff format --check .
uv run python -m unittest discover -s tests -v
```

`fit-psychometric` is vendored under `third_party/fit_psychometric` (no sibling
checkout or machine-local path required). Optional Chipmunk plugin fallback:

```bash
export CHIPMUNK_PLUGIN_PATH=/path/to/labdata/plugins/chipmunk
```

## Layout

- `src/behavior_analyses/` — reusable analysis modules
- `scripts/analyses/` — maintained scripts and migration entry points
- `labdata_plugin/` — local analysis-schema plugin under development
- `tests/` — migration-focused tests
- `docs/MIGRATION.md` — inventory, portability notes, live-validation gates
- `behavioral_metrics/`, `psychometric_curves/`,
  `psychophysical_kernels/` — maintained labdata notebooks + helpers
- `archive/djchurchland/` — preserved pre-migration notebooks
- `archive/labdata_migration/` — preserved superseded migration smoke notebooks
- `notebooks/` — ingestion and exploratory work
- `oft/` — open-field analyses (archived notebooks relocated)

## Data access

Maintained analysis paths use `labdata` and the local plugin in
`labdata_plugin/`. Prefer `from chipmunk import Chipmunk` for trial-level
Chipmunk data. Archived `djchurchland` notebooks are under
`archive/djchurchland/` and are not active entry points.

## Migration CLIs

```bash
uv run python scripts/analyses/seed_behavior_analysis_set.py --help
uv run python scripts/analyses/migrate_behavior_analysis_schema.py --help
uv run python scripts/analyses/populate_behavior_tables.py --help
uv run python scripts/analyses/plot_psychometrics.py --help
```

Use `--dry-run` on seed/populate before any database writes.
