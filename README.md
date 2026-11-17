# Movement and sensory processing in V1

Behavior and electrophysiology analyses for Gabriel Rojas-Bowe's thesis in the
Churchland Lab at UCLA. The main question is how sensory responses in mouse
visual cortex change around movement during a perceptual decision task.

This is an active research repository. It contains the session-preparation
pipeline, analysis code, exploratory notebooks, and the current scientific
argument. Most analyses require access to the lab's DataJoint database; raw data,
caches, and fitted models are not included.

## Start here

1. Follow the [setup guide](docs/setup.md) to install the environment and arrange
   database and data access.
2. Run the [PETH tutorial](notebooks/ephys/tutorials/tutorial_peth.ipynb): load one
   recording session and plot one unit's raster and average response.
3. Read the [paper story](docs/paper-story.md) for the scientific overview, then
   [the GLM analysis](docs/glm.md) for the model and current findings.

The tutorial uses **GRB006 / 20240821_121447**. Ask Gabriel for access to this
session and the Chipmunk plugin before starting. You can read the saved notebook
outputs and figures without database access.

## Install

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and Git, then:

```bash
git clone https://github.com/rojasgabriel/thesis.git
cd thesis
uv sync --locked
```

The repository pins Python 3.11.14 in `.python-version`. Installation does not
configure LabData, grant database access, or download recordings. Continue with
[setup and your first plot](docs/setup.md).

## Repository map

| Location | Purpose |
| --- | --- |
| `src/thesis/behavior/` | Behavioral analyses |
| `src/thesis/ephys/preprocessing/` | Clock alignment, session preparation, and quality checks |
| `src/thesis/ephys/analyses/` | Complete analysis workflows, including the GLM |
| `src/thesis/ephys/tools/` | Interactive inspection tools and job submission |
| `src/labdata_plugin/` | Project-specific DataJoint tables |
| `notebooks/ephys/` | Exploratory population analyses and the PETH tutorial |
| `docs/` | Setup, scientific interpretation, methods, and decision history |
| `tests/` | Offline regression tests using synthetic data and database test doubles |

## Workflows

| Goal | Start with |
| --- | --- |
| Make your first event-aligned plot | [PETH tutorial](notebooks/ephys/tutorials/tutorial_peth.ipynb) |
| Inspect units interactively | `uv run psth`, `uv run stimulus-response`, `uv run unit-stability` |
| Prepare an ingested, spike-sorted session | [Session preparation](docs/setup.md#prepare-a-new-ephys-session) |
| Fit the encoding model and reproduce its figures | [GLM run instructions](docs/glm_checks.md#run-and-inspect) |
| Understand current conclusions | [GLM findings](docs/glm.md) and [paper story](docs/paper-story.md) |
| Understand past analytical choices | [Decision history](docs/glm_decisions.md) |

The population [PCA](notebooks/ephys/population_geometry.ipynb) and
[dPCA](notebooks/ephys/demixed_pca.ipynb) notebooks are exploratory views of one
session. They describe their selection rules and limitations at the top.

The GLM documents include [saved figure snapshots](docs/figures/README.md).
Regenerating them requires the inputs and completed fits described in the methods;
CI does not reproduce scientific results.

## Contributing and checking changes

Keep reusable code under `src/thesis/` and notebooks exploratory. Describe the
scientific question and selection rules when changing an analysis. Run the real
analysis and inspect its figure when changing a plot.

To run the same checks as CI without lab access, install only the check group:

```bash
uv sync --locked --only-group check
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
PYTHONPATH=src uv run --no-sync python -m unittest discover -s tests -v
```

`--only-group check` replaces the active environment with the small check
installation. Run `uv sync --locked` to restore the analysis environment, or use
`uv sync --locked --group check` to install both together. The checks cover Python
lint, formatting, and offline regressions; database, hardware, and full analysis
validation require lab access.
