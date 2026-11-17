# thesis

Behavior and electrophysiology analyses for my Churchland Lab thesis.

```bash
uv sync --locked
```

Requires configured LabData database access and the Chipmunk plugin.

- [GLM results](docs/glm.md) · [checks and methods](docs/glm_checks.md) · [decisions](docs/glm_decisions.md)
- [PETH notebook](notebooks/ephys/tutorials/tutorial_peth.ipynb)

## Run

```bash
uv run populate-session GRB064 20260916_132939         # dry run
uv run populate-session GRB064 20260916_132939 --apply # insert missing results
uv run psth
THESIS_CACHE_DIR=figures/cache uv run python -m thesis.ephys.analyses.glm.flash_movement
```

Generated figures go in `figures/`, GLM fits and results in `results/glm/`.
[Saved documentation figures](docs/figures/README.md) are tracked separately.

## Checks

```bash
uv sync --locked --group check
uv run ruff check .
uv run ruff format --check .
uv run python -m unittest discover -s tests -v
```
