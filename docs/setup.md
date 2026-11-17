# Setup and your first plot

Start with one prepared session and the PETH tutorial. Session ingestion, spike
sorting, and GLM fitting are separate workflows.

## Install the environment

From the repository root:

```bash
uv sync --locked
```

Use Git and [uv](https://docs.astral.sh/uv/getting-started/installation/).
Python 3.11.14 is pinned in `.python-version`; uv can install it if needed.
The lockfile pins Python packages and Git dependency revisions. The full
installation includes PyTorch and Qt, so the initial download is substantial.

The development environment has been used on an Apple M2 MacBook Pro with
Python 3.11.14. The offline CI job runs on Linux. Full database and GUI workflows
are not covered by that job.

## Arrange LabData access

Before running the tutorial, ask Gabriel or the lab administrator for:

- A personal database account, the correct host, base database name, and project
  selection, plus any network access needed to reach them.
- The approved Chipmunk plugin and instructions for registering it with LabData.
  `labdata` is installed by this repository; the Chipmunk plugin is separate.
- Read access to GRB006 / 20240821_121447, including parsed task trials, sorted
  spikes, clock mappings, unit quality results, and stability parameter set 0.
- The lab's data location and any file access required by later analyses.

LabData 0.1.27 reads `~/labdata/user_preferences.json` by default. Its relevant
settings are `database`, `plugins`, `local_paths`, and `scratch_path`.
`LABDATA_PREFERENCES` can select a different existing preferences file.
Have the administrator configure the account and plugin for your machine;
copying another person's preferences can select the wrong project or credentials.

LabData also supports these environment overrides:

| Variable | Meaning |
| --- | --- |
| `LABDATA_DATABASE_HOST` | Database server |
| `LABDATA_DATABASE_USER` | Your database user |
| `LABDATA_DATABASE_PASS` | Your password, supplied privately |
| `LABDATA_DATABASE` | Base schema name |
| `LABDATA_DATABASE_PROJECT` | Project suffix and project-specific plugin selection |
| `LABDATA_OVERRIDE_DATAPATH` | Overrides both local data roots and scratch path |

Keep credentials and machine-specific preferences outside the repository. Use
read access for the tutorial; writing analysis tables is a separate operation.

## Open the PETH tutorial

In VS Code or another notebook editor, open
[`tutorial_peth.ipynb`](../notebooks/ephys/tutorials/tutorial_peth.ipynb) and select
the repository's `.venv/bin/python` kernel. `ipykernel` is already installed.
If your editor needs a named kernel, register one from the repository root:

```bash
uv run python -m ipykernel install --user --name thesis --display-name "Python (thesis)"
```

Restart the kernel, then run all cells in order. The notebook loads unit 20 from
GRB006 / 20240821_121447 using quality criterion 1 and stability parameter 0.
It aligns spikes to the first visual stimulus pulse in each task trial, then
shows a raster and a PETH with 10 ms bins from −100 to +150 ms.

Expect a nonempty trial count, spikes on trial rows, and a firing-rate curve
with a shaded standard error across trials. The saved example is a reference;
counts can change if upstream parsing or quality results change. The notebook
does not subtract a baseline or select units for visual responsiveness.

### If the first run fails

| Symptom | Check |
| --- | --- |
| `No module named thesis` or another installed package | Select the repository kernel and rerun `uv sync --locked`. |
| Connection error or an account prompt | Confirm database preferences, personal credentials, and network access. |
| `No module named chipmunk` | Confirm the approved plugin is installed and registered for the selected project. |
| Missing stability results or no passing units | Ask Gabriel to check session preparation and the requested filters. |
| Unit 20 is absent | Inspect `unit_table["unit_id"]`; select an available unit and record that change. |
| No trials or timestamp-alignment error | Check parsed task records and StreamSync; do not bypass the timing validation. |
| Missing file | Confirm LabData data roots and access to that exact session before downloading anything. |

## Data, caches, and outputs

The repository contains code and selected documentation figures. It does not
contain recordings, credentials, fitted models, or trial/unit caches.

- LabData locates session files through its configured data roots and database
  records. Check the lab's Hodgkin copy before requesting an S3 download.
- `THESIS_CACHE_DIR` enables local pickle caches for decorated trial/unit loaders.
  Without it, those loaders query the database on each call. Use only trusted
  caches; they retain the results from when they were created.
- Analysis figures normally go under `figures/`; GLM fits, designs, checkpoints,
  and result tables go under `results/glm/`. Both directories are ignored by Git.
- Documentation snapshots live under `docs/figures/`. They let readers see the
  results without running the analyses.

The GLM additionally needs frame times, a local cam0 video, prepared designs,
and completed fits. Start with the [GLM run instructions](glm_checks.md#run-and-inspect)
after the tutorial works. A successful tutorial does not establish GLM readiness.

## Prepare a new ephys session

This step is for an ingested, spike-sorted session that needs analysis tables.
It is not needed to read a session that is already prepared.

```bash
uv run populate-session GRB064 20260916_132939
```

This is a dry run: it checks clock mappings, unit metrics, quality counts,
stability results, and stimulus responses without creating schemas, tables, or
lookup rows. Inspect its report before applying changes with a suitably
privileged account:

```bash
uv run populate-session GRB064 20260916_132939 --apply
```

Apply inserts missing results for quality criteria 0–2 and parameter set 0,
then checks unit counts. It explicitly creates missing analysis tables and seeds
defaults; rerunning keeps existing results. Use `uv run insert_audio_events`
separately when measured OneBox audio must be recovered from raw files.

## Other analysis entry points

These require prepared sessions and their own inputs:

```bash
uv run double-peak-candidates GRB064 20260916_132939
uv run python -m thesis.ephys.analyses.double_peaks
uv run python -m thesis.ephys.analyses.category_decoding -a GRB006 -s 20240821_121447
uv run python -m thesis.ephys.analyses.rate_decoding -a GRB006 -s 20240821_121447
```

Outputs include `figures/double_peak/grb064_candidates.svg`,
`figures/category_decoding/category_decoding.svg`, and
`figures/rate_decoding/rate_decoding.svg`. The cross-session double-peak workflow
is a separate supplement, including a 30 ms control.
