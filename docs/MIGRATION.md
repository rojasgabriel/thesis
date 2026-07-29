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
| `psychometric_curves/*.ipynb` (old) | historical | Moved under `historical/djchurchland/` |
| `behavioral_metrics/*.ipynb` (old) | historical | Moved under `historical/djchurchland/` |
| `psychophysical_kernels/*.ipynb` (old) | historical | Moved under `historical/djchurchland/` |
| `sess.ipynb` (old) | historical | Moved under `historical/djchurchland/root/` |
| `oft/` notebooks | historical | Open-field; not Chipmunk LabData path |
| `psychometric_curves/fit_psychometric.py` | preserve local copy | Upstream also vendored in `third_party/fit_psychometric` |
| `labdata2_testing/`, `notebooks/ingest_subjects.ipynb` | already labdata | Leave as-is |

## Portability

- `fit-psychometric` is vendored at `third_party/fit_psychometric` (upstream
  `jcouto/fit_psychometric@665d058`) so CI and local `uv sync` do not need a
  sibling checkout or `/Users/gabriel/...` path.
- Chipmunk access prefers `from chipmunk import Chipmunk`. Optional local
  fallback uses `CHIPMUNK_PLUGIN_PATH` or
  `tool.behavior_analyses.chipmunk_plugin_path` (empty by default).

## Live LabData validation

Read-only schema/query checks and disposable seed/populate require DataJoint
credentials and an explicit approval for writes. Without those, local unit
tests + CI are the automated gates; live smoke tests remain a manual step.
