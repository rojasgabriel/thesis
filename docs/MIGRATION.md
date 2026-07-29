# LabData migration notes

## Goal

Finish moving maintained behavior analyses from `djchurchland` to `labdata`,
keep machine-local paths out of the runtime, and validate against LabData before
merging [PR #8](https://github.com/rojasgabriel/behavior_analyses/pull/8).

## Inventory

| Surface | Classification | Notes |
| --- | --- | --- |
| `src/behavior_analyses/` | migrate (done) | Reusable learning / psychometric / kernel math |
| `labdata_plugin/` | migrate (done) | User-schema tables for session sets and fits |
| `scripts/analyses/` | migrate (done) | Seed / populate / plot CLIs |
| `psychometric_curves/utils.py` | migrate (done) | LabData/Chipmunk plotting helpers |
| `psychometric_curves/*.ipynb` (old) | archived | Moved under `archive/djchurchland/` |
| `behavioral_metrics/*.ipynb` (old) | archived | Moved under `archive/djchurchland/` |
| `psychophysical_kernels/*.ipynb` (old) | archived | Moved under `archive/djchurchland/` |
| `sess.ipynb` (old) | archived | Moved under `archive/djchurchland/root/` |
| `oft/` notebooks | archived | Open-field; not Chipmunk LabData path |
| `psychometric_curves/fit_psychometric.py` | preserve local copy | Upstream also vendored in `third_party/fit_psychometric` |
| `labdata2_testing/`, `notebooks/ingest_subjects.ipynb` | already labdata | Leave as-is |

## Portability

- `fit-psychometric` is vendored at `third_party/fit_psychometric` (upstream
  `jcouto/fit_psychometric@665d058`) so CI and local `uv sync` do not need a
  sibling checkout or `/Users/gabriel/...` path.
- Chipmunk access prefers `from chipmunk import Chipmunk`. Optional local
  fallback uses `CHIPMUNK_PLUGIN_PATH` or
  `tool.behavior_analyses.chipmunk_plugin_path` (empty by default).
- LabData 0.1.x requires DataJoint `<2`. DataJoint 0.14.9 is the latest
  compatible release and upstream pins `setuptools<82` because it still uses
  `pkg_resources`.

## Live LabData validation

Read-only checks completed on 2026-07-28:

- DataJoint 0.14.9 connected and exposed the expected Chipmunk
  `TrialParameters` and LabData `DecisionTask.TrialSet` fields.
- GRB006 had 263 LabData trial sets.
- The corrected psychometric query returned 422 choice trials for
  `20240826_113307`, all with finite boundary-centered intensities spanning
  -8 to +8 Hz.

Approved disposable write checks completed on 2026-07-28:

- Seeded `lab_tasks_479_smoke_20260728` with the GRB006
  `20240819_110829` visual trial set: one session, one trial set, and one
  subject/trial-set aggregate.
- Created the four computed analysis tables and populated one row in each:
  `LearningSessionMetrics`, `PsychometricSessionFit`,
  `PsychometricSubjectFit`, and `PsychophysicalKernel`.
- Verified 393 choice trials in the psychometric outputs and a 10-fold kernel
  fit over 393 trials (`score_mean = 0.844744`).

The repository's Python 3.10 environment has a damaged local SciPy binary, so
the successful populate ran from the same lockfile under Python 3.11.
