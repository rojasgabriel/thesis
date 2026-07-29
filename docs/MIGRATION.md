# LabData migration notes

## Goal

Finish moving maintained behavior analyses from `djchurchland` to `labdata`,
keep machine-local paths out of the runtime, and validate against LabData before
merging [PR #8](https://github.com/rojasgabriel/behavior_analyses/pull/8).

## Inventory

| Surface | Classification | Notes |
| --- | --- | --- |
| `src/behavior_analyses/` | migrate (done) | Reusable learning / psychometric / kernel math |
| `labdata_plugin/` | migrate (local refactor done) | Skill-informed analysis-set/config/fit schema; live activation pending |
| `scripts/analyses/` | migrate (done) | Seed / populate / plot CLIs |
| `psychometric_curves/utils.py` | migrate (done) | LabData/Chipmunk plotting helpers |
| `psychometric_curves/*.ipynb` (old) | archived | Moved under `archive/djchurchland/` |
| `behavioral_metrics/*.ipynb` (old) | archived | Moved under `archive/djchurchland/` |
| `psychophysical_kernels/*.ipynb` (old) | archived | Moved under `archive/djchurchland/` |
| migration stress notebooks (old schema) | archived | Moved under `archive/labdata_migration/`; superseded by tested CLIs |
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

## Plugin schema design lock

This design follows the Notion skills **Design LabData Plugin Tables** v0.2
and **Plan Analysis** v0.5.

| Table | Tier | One row represents | Primary dependencies | Persisted facts |
| --- | --- | --- | --- | --- |
| `BehaviorAnalysisSet` | Manual | One curated analysis selection and its provenance | none | name, description, selection thresholds/version |
| `BehaviorAnalysisSet.TrialSet` | Part | One selected upstream task TrialSet | master + `DecisionTask.TrialSet` | optional inclusion reason |
| `PsychometricFitConfig` | Lookup | One versioned psychometric eligibility configuration | none | minimum choices/stimulus values, analysis version |
| `PsychometricSessionFit` | Computed | One fit for one upstream TrialSet and config | `DecisionTask.TrialSet` + config | status, fit sample size, curve/parameters/diagnostics |
| `PsychometricSubjectFit` | Computed | One pooled fit for one analysis set, subject, condition, and config | analysis set + `Subject` + config | status, fit sample size, curve/parameters/diagnostics |
| `PsychophysicalKernelFitConfig` | Lookup | One versioned pooled-kernel configuration | none | bins, CV folds, seed, calibration rate, regularization, version |
| `PsychophysicalKernel` | Computed | One pooled kernel for one analysis set, subject, condition, and config | analysis set + `Subject` + kernel config | status, fit sample size, weights, held-out scores, bias |

Keep:

- one manual selector containing only selection provenance and upstream TrialSet
  membership
- direct upstream keys and versioned fit configuration in computed primary keys
- explicit `fit` / `skipped` rows so eligible keys do not remain pending
- numerical outputs and sample sizes needed to reproduce and interpret plots

Drop or derive:

- `BehaviorSessionSet.Session` and `.SubjectTrialSet`; both project from selected
  TrialSets
- `LearningSessionMetrics`; learning curves read canonical counts/performance
  directly from `DecisionTask.TrialSet`
- duplicated `p_side`, `n_side`, `fit_params`, selection-owned kernel settings,
  and upstream trial payloads
- figures in database blobs; maintained plot code produces editable PDF/SVG
  outputs from numerical tables

Open live decision:

- the deployed tables use the old definitions and must be archived before the
  canonical class names can be activated. This requires a separately approved
  live migration.

## Live LabData validation

Read-only checks completed on 2026-07-28:

- DataJoint 0.14.9 connected and exposed the expected Chipmunk
  `TrialParameters` and LabData `DecisionTask.TrialSet` fields.
- GRB006 had 263 LabData trial sets.
- The corrected psychometric query returned 422 choice trials for
  `20240826_113307`, all with finite boundary-centered intensities spanning
  -8 to +8 Hz.
- The shared user schema already contains ephys-owned
  `PsychophysicalKernelParam` / `SessionPsychophysicalKernel`, so this plugin
  uses the collision-safe and method-specific
  `PsychophysicalKernelFitConfig`.
- The deployed behavior schema contains two selections: one disposable smoke
  set and `migration_stress_test` with 183 selected TrialSets. Across both
  selections it currently stores 184 duplicated learning rows, 128 session
  fits, three pooled subject fits, and three pooled kernels.

Approved disposable write checks completed on 2026-07-28:

- Seeded `lab_tasks_479_smoke_20260728` with the GRB006
  `20240819_110829` visual trial set: one session, one trial set, and one
  subject/trial-set aggregate.
- Created the old four computed analysis tables and populated one row in each:
  `LearningSessionMetrics`, `PsychometricSessionFit`,
  `PsychometricSubjectFit`, and `PsychophysicalKernel`.
- Verified 393 choice trials in the psychometric outputs and a 10-fold kernel
  fit over 393 trials (`score_mean = 0.844744`).

The repository's Python 3.10 environment has a damaged local SciPy binary, so
the successful populate ran from the same lockfile under Python 3.11.

## Approval-gated live migration

No live schema write is implied by importing the refactored plugin. Before
importing it against the shared database, the approved migration must:

1. archive the eight old behavior tables with dated, plugin-specific names
   (no drops);
2. activate the seven locked relations above and their two default config rows;
3. copy the two selection masters and 184 TrialSet membership rows;
4. copy compatible fitted rows under the default config while deduplicating
   session fits by their upstream TrialSet key;
5. dry-run pending keys, then populate only the two migrated analysis-set IDs;
6. verify counts/headings and render bounded diagnostic PDF figures.

Figures use a plain white canvas, neutral comparison titles, units, sample
sizes, frameless legends, and vector output. Once produced, raw figures,
commands, configuration IDs, and observation-first notes belong on the
LAB-TASKS-479 **Results** subpage in Notion.
