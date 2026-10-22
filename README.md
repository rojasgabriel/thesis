# thesis

Behavior and electrophysiology analysis code for Churchland Lab thesis work.

[V1 GLM](docs/glm.md) is the encoding-model workflow.

## Requirements

- [uv](https://astral.sh/uv)

## Setup

```bash
git clone https://github.com/rojasgabriel/thesis.git
cd thesis
uv sync
```

## Prepare a new ephys session

After LabData ingestion and spike sorting, check the session, then populate its
clock mappings, unit metrics, quality counts, stability results, and stimulus
responses:

```bash
uv run populate-session GRB064 20260916_132939
uv run populate-session GRB064 20260916_132939 --apply
```

The first command is a dry run. The second inserts missing results for project
quality criteria 0-2 and parameter set 0, then checks their unit counts.
Dry-run imports do not create schemas, tables, or lookup rows. The apply
command creates missing analysis tables and seeds their defaults explicitly.
Rerunning it keeps existing results. Use
`uv run insert_audio_events` separately when measured OneBox audio must be
recovered from raw files.

## Ephys analysis figures

```bash
THESIS_CACHE_DIR=figures/cache nice -n 10 uv run python -m thesis.ephys.analyses.glm.flash_movement
uv run python -m thesis.ephys.analyses.stim_responses_by_locomotion
uv run double-peak-candidates GRB064 20260916_132939
uv run python -m thesis.ephys.analyses.double_peaks
uv run python -m thesis.ephys.analyses.category_decoding -a GRB006 -s 20240821_121447
uv run python -m thesis.ephys.analyses.rate_decoding -a GRB006 -s 20240821_121447
```

The first command builds the combined flash PSTHs and observed peak scatters,
then the observed/full-model/without-flash-by-exit comparison on identical
held-out trial windows and preceding-gap classes. It uses cached trials,
a saved response classification, and completed GLM fits. The main outputs are
`figures/glm/flash_observation.svg` and `figures/glm/flash_comparison.svg`;
`figures/glm/flash_model.svg` is the supporting kernel-only view.
[glm.md](docs/glm.md) gives the scientific argument and links to its checks.
The second command makes the separate working peak comparison at
`figures/locomotion/locomotion_peaks.svg`. The third shows
confirmed 15 ms double-peak candidates for one session and writes
`figures/double_peak/grb064_candidates.svg`. The `double_peaks` command makes the
separate cross-session double-peak supplement, including its 30 ms control.
The decoding commands decode the stimulus category, the choice, and the stimulus rate from
the population firing rates of one session, and write
`figures/category_decoding/category_decoding.svg` and
`figures/rate_decoding/rate_decoding.svg`.

## Tested on

- MacBook Pro M2
- macOS Golden Gate 27.0 (beta 6)
- Python 3.11.14
