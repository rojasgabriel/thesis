# behavior_analyses

Behavioral-only analysis code for thesis work in the Churchland Lab.

## Setup and checks

```bash
uv sync
uvx ruff check .
uvx ruff format --check .
uv run pytest
```

## Layout

- `src/behavior_analyses/` — reusable analysis modules
- `scripts/analyses/` — maintained scripts and migration entry points
- `labdata_plugin/` — local analysis-schema plugin under development
- `tests/` — migration-focused tests
- `behavioral_metrics/`, `psychometric_curves/`,
  `psychophysical_kernels/`, and `oft/` — established analysis areas
- `notebooks/` — ingestion and exploratory work

## Data access

The active migration uses `labdata` with the local plugin configuration in
`pyproject.toml`. Older notebooks may still use `djchurchland`; inspect the
specific path before extending it.

The repository may contain uncommitted migration work. Git and the local files
are the source of truth for implementation state; Notion owns priorities and
cross-system decisions.
