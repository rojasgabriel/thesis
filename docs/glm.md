# V1 GLM

Poisson encoding model of V1 spikes in the Chipmunk task, fitted on two
sessions: GRB006 `20240821_121447` (167 units) and GRB064 `20260916_132939`
(76 units). The model predicts the spike count of each unit in 1-ms bins from
12 regressors: two flash kernels by the gap from the flash before, two
kernels for the change of the flash response after the withdrawal (flash ×
move, and flash × move × gap), a fixation kernel (first flash to center
exit), a withdrawal kernel (0 to 100 ms), one column each for the initiation,
the choice, and the previous choice, 10 motion-energy PCs of the video, a
session drift, and a strictly past spike-history filter for each unit. All
terms add in log rate. Coefficients are conditional associations, not causal
effects.

This file has the model, its results, and its checks. The history of the
design (the earlier designs, and the test behind each change) is in
[glm_decisions.md](glm_decisions.md).

`prepare` reuses saved windows and video features by file presence. It reuses
the saved design when the metadata has the current task regressor names, the
current column count for each regressor, and the current penalty scale of the
fixation kernel.
It does not compare basis time ranges or spacing, and it does not check whether
the windows or video files changed. If you change a basis and keep its column
count, or if you rebuild the windows or video features, delete
`common_design_me.npy` and `common_design_me.json` before you run `prepare`.
Only one `glm` command can use a subject/session output directory
at a time.

## Data

- Units: quality 1, stability 0. No sensory-response filter. Exclude unit 212
  because of unstable out-of-fold spike-history extrapolation. Draw a fixed
  random sample of 20 from the remaining units for `--units sample`
  (`SAMPLE_SEED = 20260914`). `--units sample1`, `sample2`, ... each draw 20
  other units; no unit is in two samples.
- Trials: completed left/right trials without early withdrawal. Drop a trial
  that has no initiation, first measured flash, withdrawal, or response-entry
  time. The remaining trials must have these events in that order. If a trial
  does not, `prepare` stops with an error that lists its trial number; it does
  not drop the trial.
- Align to the first measured flash, not the Bpod stimulus command.
- Stimulus (`poissonStimulusTrain.m` of the task code): the train is 1 s long,
  in 25 bins of 40 ms (a 15 ms flash and a 25 ms minimum gap). A trial with
  rate N has N flashes: one always in the first bin, and N − 1 in random bins
  of the other 24. So only the first flash is at a fixed time; in an average
  over trials the responses to later flashes are spread evenly (11.9 flashes/s
  in the fixation, 11.0 after the exit). The train stops early only at the
  response entry (`extraStimDur` = 0). The first flash comes 10 to 100 ms
  after the center poke (`preStimDelay` in `Experiments/VisualTask.mat`, plus
  the output latency; 24 to 114 ms in the data). The go cue plays when the
  wait time (about 0.5 s) is reached. No sound plays at the center poke: the
  state machine (`VisualTaskSMA.m`) has a start tone there, but its length is
  the virtual-observer delay, which the settings fix at 0.
- Grid: DAMN 1 ms, pre=100 ms, post=2.54 s (2,639 centers). Keep the full
  window. Fitting **masks bins after that trial’s response entry**; it does not
  shorten the grid. Only flashes from initiation through response entry enter
  the design.
- Split whole trials randomly 60/20/20 (`SPLIT_SEED = 20260912`). Models fit on the
  training block. The validation block scores the model with and without the
  10 motion-energy PCs. Validation also picks a penalty for each unit and model,
  but no later step uses these penalties: each cross-validation fold picks its
  own. The third block is not scored on its own; held-out performance comes from
  cross-validation over whole trials.
- `fit` then cross-validates with 10 PCs: 10 folds over whole trials
  (`CV_SEED = 20260913`). Each trial belongs to one test fold. A fold with no
  test spikes for a unit is skipped; at least two scored folds are required.
  Reports per-unit mean and SEM across the scored folds. It finally refits each
  unit on every valid trial at the median fold penalty and writes
  `unit_*_final.json`. The kernel and model-design figures read that model. The
  prediction figure reads the saved fold models instead, so no plotted trial
  was in the training data of its model. The final model's score comes from the
  fold records. `unique` refits within those folds using their
  selected penalties.

The PC count is fixed at 10. PCA and common-design
scaling use the original training split, rather than being recomputed
inside each fold. These scores are not a fully nested estimate of the complete
selection and preprocessing procedure.

## Design

Common columns use a z-score on **valid** original training bins; that scaling
is fixed for selection, cross-validation, and the final refit. Self-history
scaling uses the training rows for selection, the non-test rows in each fold,
and all valid rows for the final refit. Intercept unpenalized. No go cue, outcome,
or punishment kernel.

| Group | Representation | Columns |
| --- | --- | ---: |
| Flash, 40 ms gap | A flash within 50 ms of another; 8 causal linearly spaced raised cosines, 0–150 ms | 8 |
| Flash, 80+ ms gap | Every other flash (the first flash also); same bases | 8 |
| Flash × move | Same flash bases, scaled by the centered period (a flash after the center exit against one in the fixation) | 8 |
| Flash × move × gap | Same flash bases, scaled by the centered period times the centered gap class | 8 |
| Withdrawal | 4 causal raised cosines, 0–100 ms after center exit | 4 |
| Initiation | 1 from center entry to the first flash, else 0; no temporal basis | 1 |
| Fixation | One kernel on the first flash, set to zero from the center exit on; 12 raised cosines with even spacing in log(t + 25 ms), 0–950 ms; no step | 12 |
| Choice (key `side_after_exit`) | +1 or −1 for the side of the response, from the center exit on | 1 |
| Motion energy | One contemporaneous value for each of the first 10 PCs; no temporal basis | 10 |
| Session drift | 4 raised cosines over the session, with the last one left out | 3 |
| Self-history | 10 log-spaced raised cosines at strictly past lags, 1–100 ms | 10 |

74 columns in 12 regressors. `basis_diagram.py` draws the basis functions
(`figures/glm/basis_functions.svg`).

Names in the figures: a flash in the fixation is "still" and a flash after the
center exit is "move" (task periods, not a measure of locomotion); the side of
the response after the exit is "choice"; the side of the last response before
the trial is "previous choice".

Design of 2026-10-13, fitted on all units of GRB006 (167 units, median
cross-validated D² 0.0726, IQR 0.046 to 0.105) and GRB064 (76 units, 0.0702,
IQR 0.045 to 0.100). Changes from the design of 2026-10-12: 2 gap classes in
place of 3, no center-entry kernel, the previous choice, and flash × move ×
gap. The fits of the earlier designs are in the folders next to the session
folder (`_three_gap_classes`, `_eleven_regressors`).

No flash × movement interaction, response-entry,
same-bin history, cross-unit coupling, or previous-trial
terms. Motion-energy coefficients are contemporaneous nuisance associations.
Pose tracking is not included.

## Video

NIDQ ch1 falling edges in decoded-frame order. Do not use container times.

640×512 grayscale → 80×64. Feature is `|frame[t] − frame[t−1]|` when decoded
indices are adjacent; the first frame of a gap is zeros. PCA on training
frames that have a real difference, 200 axes. Scores scaled on those training
frames. The model uses the first 10.

## Fit

`count_t ~ Poisson(exp(intercept + X_t β))` at 1 ms on unmasked bins.

DAMN supplies the 1 ms grid, the spike counts, and the event convolution, and it
clips each kernel at the trial-window edges. NumPy `np.interp` puts the
motion-energy scores on the grid; bins outside the camera frame times get zero.
Fits use sklearn `PoissonRegressor` (log link, unpenalized intercept, L2,
LBFGS). Each unit's design appends its standardized self-history regressor.

Per unit and model: fixed L2 grid from 1e−6 to 1e4 in powers of ten. Fit from
strong to weak penalties, with a cold retry after a failed warm start. Drop
penalties that still fail to converge and break loss ties toward the stronger
penalty. The grid does not expand automatically. Each cross-validation fold selects its
penalty using an inner validation split. The null is a constant rate from the
training rows for selection, or all non-test rows for a cross-validation fold.
`cross_validation_summary.json` gives the median and IQR over units of the
cross-validated D² (each unit's mean over its test folds). No output gives an
IQR for bits/spike; the summary figure shows histograms of D² and bits/spike
over units, each with its median. One session: no population p-value.

```bash
uv run python -m thesis.ephys.preprocessing.audit_camera_pulses \
    --output figures/glm/GRB006_20240821_121447/camera_pulses.json
uv run glm prepare
uv run glm fit --units all
```

Run the camera pulse audit one time before the first `prepare`. It reads the
NIDQ ch1 edges from the database, removes a short glitch pulse only when the
remaining falling-edge count then equals the decoded frame count, and writes
`frame_times.npy` next to its `--output` file. `prepare` stops with
`FileNotFoundError` if `frame_times.npy` is not in the session output directory.
The audit is written for GRB006 `20240821_121447` only, and it does not replace
existing output files.

`prepare` writes `stimulus_windows_me.npz`, `video_me_features.npz`, and
`common_design_me.npy` under `figures/glm/<subject>_<session>/`. Pass
`--subject`, `--session`, or `--root` to work on another dataset. These flags
change only the data source and the output directory. The unit-212 exclusion
(`EXCLUDED_UNIT_IDS` in `fit.py`) applies only to GRB006 `20240821_121447`;
change it before you fit another session. `prepare` also needs exactly one local cam0 back-view
(`BackStereoView`) video. `fit` writes
named summaries under `all_fit_me/`. Restartable per-unit records stay under
`all_fit_me/checkpoints/`. `unique` writes the
held-out prediction SVG there, because it shows the example unit and its
selection needs the `unique` results.
The example unit is unit 197 when it was fit (`EXAMPLE_UNIT_ID` in
`results.py`; 55 spikes/s and a clear flash response in GRB006
`20240821_121447`). For a fit without it, the example is the unit whose visual
flash regressor explains the most deviance per second (`example_unit_id`): unique
flash ΔD² × bits/spike ÷ D² × firing rate. Unique flash ΔD² alone is not used.
It is a fraction of the null deviance of the unit, so its maximum selects a unit
with few spikes, and the rasters of that unit are too sparse to compare.
`figure_data.json` records the id.
Use `--units sample` for a 20-unit smoke run; its checkpoints, summaries, and
figures stay separately in `sample_fit_me/`, and its example unit is the best
of those 20 units. Derived example arrays are saved as
`figure_data.npz` and `figure_data.json` in the fit directory. Interrupted
figure-data writes are replaced on retry under the session lock. Existing
figures at the session root are older exports and are not updated by these
commands.

Fit checkpoints are reused by file presence. If inputs or model settings
change, clear the affected checkpoints before rerunning.

The prediction figure shows the example unit. Each plotted trial uses its saved
out-of-fold model, so that model did not train on the trial. Predicted rasters
are Poisson draws from one-step-ahead expected counts conditioned on observed
spike history, not free-running simulations. The trace compares observed and
predicted firing rates in complete 10-ms bins without smoothing. It shows only
the bins where at least half of the plotted trials are still valid; the rasters
show every valid bin. Post-response
bins are excluded from rates, likelihood, and scores. Raster gaps after response
entry are shaded within the displayed interval.

The model-design figure uses one real test trial from the example unit. The trial is
nearest the unit's median pre-response spike count. Panel a shows every
modeled flash (the measured flashes from initiation through response entry),
withdrawal and the filter support of both, the first three continuous motion-energy PC
scores, and strictly past history support. Panel b
shows the exact fitted terms `X_g β_g` for visual, other task, all selected
motion-energy, and history columns. Their sum plus the intercept is checked
against the full linear predictor. Panel c compares the observed spike raster
with Poisson draws from the fitted conditional counts. This explanatory panel
uses the final model fit on all valid bins; held-out predictions stay in the
separate prediction figure. Post-response bins are not shown. The single kernel figure shows task and self-history traces
for the example unit.

## Unique explained deviance

Within each cross-validation fold, shuffle one regressor and refit at that fold's
full-model alpha. Shuffle only among valid bins, within trial. Motion-energy
PCs stay one regressor.

`unique ΔD² = complete D² − one-removed D²`
`maximal ΔD² = regressor-alone D² − all-shuffled D²`

`trial-specific ΔD² (motion energy) = complete D² − D² with motion energy from another trial`

The third value is a control for motion energy only. It replaces the regressor with
the motion energy of another trial at the same time in the trial. This keeps the
average time course of motion energy in a trial. If the trial-specific ΔD² is
near zero and the unique ΔD² is not, the regressor predicts spikes through the time
in the trial and not through the movement of that trial. `summary.json` gives it
as `video_trial_specific`. Session drift and the previous choice do not change
in a trial, so they are shuffled over trials.

All run inside the same 10 folds as `fit`, at the penalty that fold selected,
so they are comparable with the cross-validated deviance. Reported as the mean
across folds with its SEM. Following
[Oesch et al. 2026](https://doi.org/10.1038/s41467-026-70639-1) and
[Musall et al. 2019](https://www.nature.com/articles/s41593-019-0502-4).

```bash
uv run glm unique --units all
```

The same command writes `figure_data.npz` for the example unit in the fit
directory (`all_fit_me/`). `main_figure.py` draws the kernels, the example trial, the
unique and maximal ΔD², and the population summary from these files.

## The rule for a regressor

From 2026-10-12 (`inclusion.py`). The importance of a regressor is tested on
all units and on held-out trials, because 20 units are too few (the side term
and the flash × move term were not reliable on 20 units and are reliable
on 167). The parameterization of a term (bases, window, spacing) is still chosen on the 20
selection units. A regressor is in the model when it passes two tests, each
with a Holm correction for the number of regressors that are tested together:

1. Gain: the cross-validated D² is higher with the regressor than without it
   (or with it shuffled). The gain of a unit is the mean over its folds.
   One-sided Wilcoxon signed-rank test over units.
2. Units: more units reliably need the regressor than chance gives. A unit
   needs it when its fold gains are above zero (one-sided t-test over folds,
   p < 0.05), so a regressor with no effect gives 5% of the units. One-sided
   binomial test of the count against 5%. For 167 units this is 14 units
   (8.4%) for one regressor and 17 to 18 units (about 11%) with the correction
   for 7 to 12 regressors.

The fraction of units that are reliably worse with a regressor is reported as
the measured rate of false positives of the unit test: a real regressor cannot
make a unit predict worse except by noise. In the current model it is 0 to 3%
for every regressor, so 5% is a fair chance level. Limits: the folds of a unit
share training trials, and the units share the trials.

The parameterization of a term (its window, its bases, its classes) is
chosen by prediction on 80 units, and the collinearity of the resulting
design is then measured; see "Parameterization of each term in the final
design".

The tests say that a regressor is real for the population. They do not say
that it is worth its columns; that is a choice of parsimony. The tests are in
`inclusion.py`. `candidate_terms.py` applies them to a design with a new term
(all regressors of the design, by the shuffle of each), and `main_figure.py`
writes them for the model in use to `main_figure_stats.json` (all 12 regressors
pass both tests in the two sessions).

## Parameterization of each term in the final design, on 80 units

The six grids were run on 4 sets of 20 units (`--samples 4`, 80 units) with
the design of 12 regressors as the base (median validation D² 0.0800). Each
row is a design against the parameterization in use (paired over the 80 units, two-sided
Wilcoxon). The results of the earlier base designs are in
`<grid>_earlier_design` and `<grid>_eleven_regressors`; they give the same
choices.

| Grid | Design | Median change | Units better | p |
| --- | --- | ---: | ---: | ---: |
| Flash kernel length (in use: 150 ms) | 200 ms | +0.16% | 42 | 0.48 |
| | 250 ms | −0.66% | 30 | 0.18 |
| | 300 ms | −0.74% | 37 | 0.45 |
| Session drift (in use: 4 raised cosines) | none | −0.32% | 31 | 0.017 |
| | 8 | −0.08% | 37 | 0.81 |
| | 12 | −0.35% | 31 | 0.071 |
| | cubic spline | +0.21% | 45 | 0.14 |
| Gap classes (in use: 40, 80+) | 1 kernel | −1.37% | 24 | 0.0002 |
| | 40, 80 to 120, 160+ | −0.20% | 35 | 0.21 |
| | 40, 80, 120+ | −0.25% | 33 | 0.46 |
| | 40, 80, 120, 160+ | −0.38% | 33 | 0.15 |
| Fixation term (in use: log-spaced kernel) | none | −3.36% | 10 | 1e-10 |
| | step from the center entry | −1.74% | 18 | 3e-7 |
| | step from the first flash | −1.86% | 19 | 2e-7 |
| | kernel, linear spacing | −0.59% | 21 | 9e-5 |
| | kernel, 10 bases | −0.16% | 32 | 0.10 |
| | kernel, no initiation column | 0.00% | 41 | 0.50 |
| Withdrawal kernel (in use: 0 to 100 ms) | none | −0.78% | 25 | 0.0006 |
| | ±100 ms around the exit | +0.05% | 43 | 0.60 |
| | ±300 ms around the exit | +0.58% | 51 | 0.010 |
| Motion-energy PCs (in use: 10) | none | −0.71% | 26 | 0.0009 |
| | 25 | −0.14% | 37 | 0.76 |
| | 50 | −0.36% | 35 | 0.30 |
| | 100 | −1.87% | 26 | 0.001 |
| | 200 | −5.66% | 12 | 2e-10 |

- The design is made in two steps.
  1. The parameterization of each term is chosen by prediction: another
     parameterization replaces the one in use when its validation D² is
     reliably higher on these units (paired Wilcoxon).
  2. The collinearity of the regressors of the resulting design is measured
     with the published diagnostics (see "Collinearity of the regressors"):
     the generalized variance inflation factor (Fox and Monette 1992; limit
     5) and the condition indices (Belsley, Kuh, and Welsch 1980; limit 30).
     A regressor that is collinear with another must go, because its weights
     cannot be read.
- Step 1: no other parameterization predicts reliably better, with one
  exception. The withdrawal kernel from 300 ms before to 300 ms after the
  exit predicts better than the kernel of 0 to 100 ms in use (+0.58%, 51 of
  80 units, p = 0.010). DECISION (Gabriel, 2026-10-13): the model keeps the
  kernel of 0 to 100 ms, because the ±300 ms kernel fails parameter recovery
  of the fixation kernel in both animals (`parameter_recovery.py`, 3
  simulations for each unit):
  - With the ±300 ms kernel, the published diagnostics do not show
    collinearity (GVIF^(1/p) 1.53 for the withdrawal kernel and 1.45 for the
    fixation kernel; largest condition index about 6). Parameter recovery
    shows the problem that they do not.
  - The fixation level (200 to 450 ms) is recovered worse. GRB006: median
    error 0.129 against 0.080 with the kernel in use, larger in 139 of 167
    units, p = 7e-18; recovered over true 0.55 against 0.76. GRB064: 0.119
    against 0.090, larger in 54 of 76 units, p = 9e-5; 0.73 against 0.84.
  - The recovery of the flash × move effect does not change (p from 0.06 to
    0.76 in the two animals and the two gap classes).
- Step 2, for the design in use, in GRB006 and in the held-out animal GRB064
  (no choice of step 1 used GRB064):
  - 2a, collinearity: no regressor is above the limits in either animal
    (largest GVIF^(1/p) 2.45; largest condition index 7.7), so no regressor
    goes.
  - 2b, parameter recovery of every regressor (`parameter_recovery.py`):
    spikes are simulated from the fitted model of each unit, the model is fit
    again to them with the same penalty, and each recovered kernel (or weight
    vector, for motion energy and drift) is compared with the true one. The
    median agreement is 0.68 to 0.85 in GRB006 and 0.68 to 0.96 in GRB064;
    the lowest is session drift in both. The median size of the recovered
    kernel over the true size is 0.84 to 1.07, except spike history (0.71 in
    GRB006 and 0.85 in GRB064, the effect of the penalty). The level of the
    fixation kernel is recovered at 0.76 (GRB006) and 0.84 (GRB064) of its
    size: the leak of "Leak of the fixation level".
- Motion energy: 10 PCs are needed (none is worse), and more PCs do not help;
  100 and 200 are worse.
- The design with one flash kernel has no flash × move × gap term: it is the
  product of the period and the class of the shortest gaps.
- The initiation column gives no gain on these 80 units. Its importance is
  tested on all units (16% need it in GRB006, 24% in GRB064).

## Results

Model in use: the design of 2026-10-13 (12 regressors, 74 columns), one L2
penalty for all terms, selected for each unit. Artifacts are in
`figures/glm/<subject>_<session>/`, which git does not track.

| | GRB006 20240821_121447 | GRB064 20260916_132939 |
| --- | ---: | ---: |
| Units, trials | 167, 294 | 76, 169 |
| Video | back view | bottom view |
| Median cross-validated D² (IQR) | 0.073 (0.046 to 0.105) | 0.070 (0.045 to 0.100) |
| Median bits/spike | 0.56 | 0.46 |
| Regressors that pass the two tests of the rule | 12 of 12 | 12 of 12 |

| Regressor | Columns | GRB006: units that need it | units worse | median unique ΔD² | GRB064: units that need it | units worse | median unique ΔD² |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Spike history | 10 | 92% | 2.4% | 0.0257 | 92% | 1.3% | 0.0267 |
| Fixation | 12 | 78% | 0.0% | 0.0032 | 91% | 0.0% | 0.0054 |
| Flash, 80+ ms gap | 8 | 77% | 0.0% | 0.0031 | 61% | 0.0% | 0.0015 |
| Flash, 40 ms gap | 8 | 74% | 0.0% | 0.0029 | 50% | 0.0% | 0.0012 |
| Motion energy | 10 | 57% | 0.0% | 0.0015 | 45% | 0.0% | 0.0011 |
| Withdrawal | 4 | 51% | 0.0% | 0.0008 | 47% | 0.0% | 0.0008 |
| Flash × move | 8 | 41% | 0.0% | 0.0008 | 38% | 1.3% | 0.0007 |
| Choice | 1 | 35% | 2.4% | 0.0004 | 45% | 0.0% | 0.0006 |
| Session drift | 3 | 28% | 2.4% | 0.0004 | 14% | 1.3% | 0.0002 |
| Flash × move × gap | 8 | 26% | 0.6% | 0.0004 | 24% | 0.0% | 0.0005 |
| Previous choice | 1 | 19% | 0.6% | 0.0002 | 16% | 1.3% | 0.0001 |
| Initiation | 1 | 16% | 3.6% | 0.0001 | 24% | 1.3% | 0.0002 |

Chance gives 18 units or fewer of 167 (11%) and 11 or fewer of 76 (14%) for 12
regressors. The previous choice and session drift pass only just in GRB064
(12 and 11 units).

Findings, with the section that has the details:

- **The rate is lower in the fixation, and the effect ends at the
  withdrawal.** Raw rate, GRB006: 1.11 before the first flash, 0.64 in the
  last 200 ms of the fixation, 1.44 from 100 to 400 ms after the withdrawal
  (each unit over its mean rate). Higher after the withdrawal in 148 of 167
  units (p = 1e-21). The fixation kernel is too small in size (parameter
  recovery gets back 59% of a simulated level in GRB006 and 79% in GRB064),
  so the size of the effect is taken from the raw rates. ("What causes the
  fixation effect"; "Leak of the fixation level".)
- **The flash response has two parameterizations by the gap from the flash before.** One
  kernel for all flashes is worse than two gap classes (80 units, −1.4%,
  p = 0.0002); three or four classes are not better. ("Parameterization of each term in
  the final design".)
- **The raw response to a flash is higher in move, and most of that is the
  rate level.** Raw change in log rate +0.59 (GRB006) and +0.19 (GRB064).
  The response to the flash alone, with the leak taken out, changes by +0.15
  in GRB006 and −0.13 in GRB064 at the 80+ ms gap: the two animals have
  opposite signs. With a weaker penalty on the fixation kernel (sensitivity
  analysis) the values are +0.09 and −0.16. ("Flash response when still and
  when the animal moves"; "Penalty: one for all terms".)
- **The previous choice changes the rate before the withdrawal** in 19% and
  16% of the units. On raw rates about half of the units differ by it, so the
  one column holds a part of the effect. ("Previous choice".)
- **Checks.** Collinearity is low (largest GVIF^(1/p) 2.45, largest condition
  index 7.7). The kernels are stable in split halves, with two limits: flash
  × move × gap (agreement 0.61 and 0.57) and the fixation kernel after 700 ms
  in GRB006. The 87 units that took no part in the choices of parameterization do not
  differ from the 80 selection units in D² (median 0.066 against 0.082,
  p = 0.23) or in the unique ΔD² of a regressor (p from 0.10 to 0.96; session
  drift 0.03, not corrected for 12 tests). Each parameterization of a term holds on 80
  units.
- **Known limits of the model.**
  - Calibration (GRB006): observed over predicted rate is 0.81 to 0.93 from
    the 10th to the 65th percentile of the predicted rate, and 1.05 to 1.12
    from the 80th to the 95th. The model predicts too much at low and middle
    rates and too little at high rates. The cause is not known.
  - Residual structure (in-sample ΔD² of a term that is not in the model,
    against shuffled labels): time in trial 0.0009 against 0.0006 (p = 2e-15),
    choice in windows after the withdrawal 0.0007 against 0.0003 (p = 3e-17),
    flash rate 0.0008 against 0.0006 (p = 2e-8), session drift as a spline
    0.0007 against 0.0007 (p = 0.04). Each is small next to the D² of the
    model.
  - GRB064: the model gives 1.17 times the observed response to the first
    flash after the withdrawal.
  - "Move" is a task period. No term of the model measures locomotion; a
    flash kernel scaled by motion energy passes in GRB006 and fails in GRB064
    (different camera views).
- Still open: the early-withdrawal flashes of GRB006 (the database row is not
  deleted yet), and the link of the model to the decoding analyses.
## Flash response when still and when the animal moves

`flash_by_period.py` (`flash_by_period.svg`, both sessions) asks if the higher
rate after a flash when the animal has left the center port
(`stim_responses_by_locomotion.py`) is a change of the response to the flash.
"Still" is a flash in the fixation and "move" is a flash after the center
exit. These are task periods, not a measure of locomotion.

- Raw response (panel a): in each trial, the last flash in the fixation and
  the first flash after the exit; the mean rate 40 to 100 ms after the flash,
  with no baseline subtraction.
- Flash kernel (panel b): the response of the flash kernel of the model for a
  flash at a gap of 80 ms or more in each period. The fixation kernel, the
  withdrawal, the choice, motion energy, and spike history are taken out.

| Measure | GRB006 | GRB064 |
| --- | ---: | ---: |
| Raw response, still and move (median, spikes/s) | 6.1 and 12.7 | 7.8 and 14.0 |
| Raw change, log(move / still), median | +0.59 (91 of 112 units higher, p = 3e-13) | +0.19 (31 of 56, p = 0.017) |
| Flash kernel at the 80+ ms gap, move minus still, median | +0.24 (132 of 167 units higher, p = 3e-14) | −0.05 (31 of 76, p = 0.058) |
| Flash kernel at the 40 ms gap, move minus still, median | +0.06 (98 of 167, p = 0.002) | −0.01 (37 of 76, p = 0.52) |

- Values of the design with 12 regressors (the 40 ms gap is in
  `flash_kernels.svg`, panels i and l). With flash × move × gap in the model, the
  change at the 80+ ms gap in GRB006 is larger than the design with one
  flash × move kernel gave (+0.08), because that kernel was an average over
  the two gap classes.
- GRB006: the response to the flash is higher in move, most at the 80+ ms
  gap. It is +0.24 of the raw +0.59; the rest is the change of the rate level
  at the exit (the end of the fixation effect).
- GRB064: the raw response is higher in move, and the response to the flash
  alone does not change. So the two animals do not agree on a larger flash
  response in move.
- The raw data show the same difference, with no model (`matched_flashes.py`,
  flashes after a long gap, matched for time and gap): after the exit, the
  response goes up more than the baseline in GRB006 (+0.18 in log, p = 3e-4)
  and much less than the baseline in GRB064 (−0.75, 8 of 65 units above 0,
  p = 1e-9). So the opposite sign is in the data, not only in the model. No
  direction of the flash × move effect is reported until a third animal is
  fit.
- Check of the model: the final refit predicts the raw change in GRB006
  (median +0.60; predicted over observed response, summed over units, 0.99
  still and 0.99 move). In GRB064 it predicts too large a change (+0.57
  against +0.19; 0.94 still and 1.17 move). With the design of 11 regressors
  the error was in the first flash after the exit (median 34 ms after it; the
  ratio was 0.95 for the later flashes). The gap term does not remove it. The
  cause is not tested.
- Units with a raw response below 1 spike/s in a period are left out of panel
  a (55 and 20 units).

## What causes the fixation effect

### The fixation effect in both animals

`hold_effect.py`, 2026-10-14. Each test is within one animal. GRB064 took no
part in the design choices.

| Measure | GRB006 (167 units) | GRB064 (76 units) |
| --- | ---: | ---: |
| Normalized rate before the first flash, last 200 ms of the fixation, 100 to 400 ms after the exit | 1.11, 0.64, 1.44 | 1.13, 0.64, 1.44 |
| Fixation level of the descriptive model, median Δ log rate (units below 0) | −0.23 (86%) | −0.34 (84%) |
| Level after the exit, median Δ log rate | +0.13 | +0.23 |
| Units higher after the exit than in fixation | 148 of 167, p = 1e-21 | 65 of 76, p = 3e-10 |
| Fixation kernel fit last (lower bound), median (units below 0) | −0.18 (90%), p = 1e-26 | −0.20 (89%), p = 1e-11 |
| Stillness: center port over side port at the lowest motion energy | 0.78, p = 3e-7 | 0.84, p = 0.31 |
| Response after the exit, more over less motion energy | 0.97, p = 0.54 | 0.98, p = 0.91 |
| Fixation level against the unique ΔD² of the flash (Spearman) | ρ = −0.20, p = 0.01 | ρ = +0.18, p = 0.12 |

- The effect replicates in GRB064. The raw rates are almost the same in the
  two animals, and the model gives a level below 0 in most units of each.
- In GRB064 the rate falls sooner after the first flash (at about 50 ms; at
  about 90 ms in GRB006).
- Two controls do not replicate in GRB064:
  - Stillness, lowest motion-energy group only: the center-port rate is not
    reliably lower than the side-port rate (0.84, p = 0.31). But over all
    groups, the side-port rate goes up with motion energy and the center-port
    rate does not, in both animals (slope higher at the side port in 125 of
    167 units, p = 8e-11, and 58 of 76 units, p = 5e-6;
    `port_vs_stillness.py`). In GRB064 the center-port rate is lower than the
    side-port rate in the three higher groups (p = 0.001 to 7e-6). So the low
    rate in the port is not only the rate of a still animal. The camera view is
    different (bottom view), so motion energy is not the same measure in the
    two animals.
  - Which units: the relation of the fixation level to the flash response
    (GRB006) is not in GRB064.
- Panel e: 1 unit of GRB006 and 3 of GRB064 have no two-step estimate (the
  fit with no penalty does not converge, because the unit has very few spikes
  in fixation; values from −9 to −24 738). They are outside the axis
  (−2.5 to 1.6) and in the rank test; the median does not change without
  them. Panel d hides 0 and 1 units, panel h 1 and 3. All are in the tests.

### Tests of 2026-10-11 (GRB006)

Tests of 2026-10-11 with no model (`matched_flashes.py`, 167 units, results in
`matched_flashes.json`; panels g and h of the fixation-effect figure). The
measures are the rate around a flash (−20 to 20 ms, the baseline) and after it
(30 to 120 ms, the response). The first flash of a trial is not used. The
units share the trials, so a test over units does not show that a result
holds for other trials.

| Explanation | What it predicts | What the data show |
| --- | --- | --- |
| Slow adaptation to the train | The rate depends on the time in the train and the flashes, not on the exit | The effect ends in about 100 ms at the exit while the flashes continue, a kernel that goes on after the exit predicts worse, and the flash rate of the trial has no effect. Not tested at a matched time: see below |
| Movement after the exit | A larger response with more movement | No: among flashes after the exit, the upper half of motion energy (2.2 times the lower half) gives the same baseline and response (ratio 0.97 and 0.97, p = 0.09 and 0.5) |
| Stillness in the port | The same low rate at a side port with the same motion energy, and the same rise with motion energy | No, in both animals: the side-port rate goes up with motion energy and the center-port rate does not (p = 8e-11 and 5e-6). At the lowest motion energy alone: 0.78 of the side-port rate in GRB006, 0.84 (p = 0.31) in GRB064 |
| A change of the gain of the whole unit (state) | The same factor for all units and for baseline and response | Not fully, and the two animals differ. Units with a larger flash response are suppressed more in GRB006 (Spearman ρ = −0.20, p = 0.01; the earlier design gave −0.41), not in GRB064 (ρ = +0.18, p = 0.12). For flashes after a long gap, the response changes more than the baseline in GRB006 (+0.18 in log, p = 0.0003) and much less in GRB064 (−0.75, 8 of 65 units above 0, p = 1e-9); for short gaps, no difference in either (0.00 and −0.03) |
| The view of the stimulus from the center port | A step at the exit that needs the stimulus, in units that respond to it, and a common effect of the side of the turn | Agrees with the fast step at the exit and the relation to the flash response. Against it: the rate goes up at the exit by the same amount when the stimulus stops there (GRB064), the side of the response has no common direction, and the rate is not lower from the center entry to the first flash |

- The matched-time test is not possible in this session. The fixation has
  almost the same length in all trials (1st to 95th percentile: 488 to
  708 ms; the animal leaves about 50 ms after the go cue). Only 21 flashes in
  the fixation are at a time from the first flash where there are also flashes
  after the exit. So the time in the train, the go cue, and the exit cannot be
  separated here by their times. A session with wait times that differ, with
  early withdrawals, or with the stimulus shown outside the port would separate
  them.
- The change at the exit is fast: flashes in the first 100 ms after the exit
  have a baseline 1.46 times lower than later flashes, and a response only
  1.08 times lower.
- The fixation level has no relation to the unique ΔD² of motion energy, fast
  adaptation, or spike history, or to depth.
- Rig facts (from Gabriel, 2026-10-11): the stimulus is an LED panel above
  the ports, centered on the center port, so it is in the upper visual field
  when the animal is in the center port. The flash is full-field and the arena
  is dark, so the light is everywhere, but the source moves in the visual
  field when the head turns (to the left of the field for a turn to the
  right). The probe is in posterior left V1 (right visual field), aimed at the
  upper field and the binocular zone by surgical coordinates, with no
  retinotopy. There is no measure of the head or the eye. A second camera
  gives a view from the right side, at an angle.
- Side of the response (`matched_flashes.py`): the view predicts one direction
  over units, because the source moves to the right visual field for a turn to
  the left. The flash response after the exit has no common direction (right
  over left 1.04, p = 0.2; baseline 1.12, p = 0.02). Single units differ: 46%
  of the units have a response more than 1.5 times larger for one side (35 to
  the left, 41 to the right). So the side matters for single units, but not in
  the direction that the position of the source predicts for left V1.
- The exit with no stimulus (`exit_without_stimulus.py` and its figure
  `exit_without_stimulus.svg`, GRB064
  20260916_132939, 78 units, no model). In an early-withdrawal trial the animal
  leaves the center port before the go cue and the flashes stop at the exit
  (78 trials, exit 403 ms after the first flash; 169 completed trials, exit at
  582 ms). The rate goes up at the exit by the same amount in both: 2.17 times
  in completed trials (67 of 77 units, p = 4e-10) and 2.12 times in early
  withdrawals (62 of 74 units, p = 2e-8). The rate after the exit is the same
  with and without the stimulus (early over completed 0.98, p = 1.0). So:
  - The effect is in a second animal.
  - The rise at the exit does not need the stimulus, so it is not a better
    view of the stimulus.
  - The rise is at the exit and not at a time in the train (403 against
    582 ms), so it is not slow adaptation.
  - The punishment of an early withdrawal is a 2 s tone with no light (from
    Gabriel), so the arena is dark after an early exit. The tone is the one
    difference left: the rate after the exit is the same with the tone and no
    stimulus as with the stimulus and no tone.
- In GRB006 the trial table has no flashes for the 209 early-withdrawal
  trials, so this test is not possible there. This is a gap in the saved
  events, not in the task: the stimulus events of GRB006 in the database
  (`nidq`, `ai0`) are flash onsets that were written from the analog channel
  in a past repair, and they have 295 trains, one for each completed trial.
  The early-withdrawal trials had a stimulus (the animal was in the port for
  about 510 ms before the punishment). To get their flashes, the onsets must
  be extracted again from the raw analog channel; that code is not in this
  repository. In GRB064 the fixation of the
  completed trials has almost the same length as in GRB006 (1st to 95th
  percentile: 455 to 747 ms), so the matched-time test is not possible there
  either; the early withdrawals are what separates the exit from the time in
  the train.
- What starts the suppression (2026-10-12, observed population rate in 10 ms
  bins, no model; scratch run, not in the repository). The time from the
  center entry to the first flash differs between trials (26 to 99 ms in
  GRB006), so the trials were cut in two halves by that time (medians 36 and
  73 ms).
  - GRB006 (168 units, 295 trials): the drop is locked to the first flash. It
    comes 85 ms after the first flash in both halves (difference 0 ms, 95%
    interval −10 to 0). From the center entry it comes at 125 and 155 ms
    (difference +30 ms, interval 30 to 50), which is the difference in the
    time of the first flash.
  - GRB064 (78 units, 239 trials): not clear. The rate goes down slowly from
    about 200 ms before the first flash, so this measure of the time of the
    drop does not work there. The drop is the same in early-withdrawal and
    completed trials (rate from 120 to 250 ms over the rate before: 0.50 and
    0.51).
- The side preference of single units does not need the stimulus (2026-10-12,
  both sessions, no model; scratch run). Side index of a unit: (right − left)
  / (right + left) of its rate. After the exit, about half of the units have
  an index above 0.2 in size (51% in GRB006, 50% in GRB064; split-half
  Spearman ρ = 0.82 and 0.77). The index with the stimulus on (100 ms after
  the exit to the last flash) agrees with the index with the stimulus off
  (150 ms after the last flash to the response entry, in the dark): ρ = 0.51
  in GRB006 and 0.57 in GRB064, against a limit of about 0.7 and 0.8 from the
  split-half values. In the fixation the index is small and has no relation to
  the index after the exit (ρ = −0.05 and −0.11). So the side preference is a
  signal of the movement or of the place of the animal, not of the view of the
  stimulus. Limit: the window with the stimulus off is later in the movement.
- The most probable explanation with these data, in both animals: the rate
  is low while the animal holds in the center port with the stimulus on, and
  it goes up when the animal leaves, with or without the stimulus. It does not
  follow motion energy after the exit. It is not only stillness: at a side
  port the rate goes up with motion energy, and in the center port it does not
  (slope test, p = 8e-11 and 5e-6). So it is a state of the trial (holding in
  the port during the stimulus) that ends with the exit. Slow adaptation, the
  view of the stimulus, and a gain that grows with movement are less
  probable.
- Only in GRB006: the suppression starts about 80 ms after the first flash and
  not at the center entry, and it is larger in units with a larger flash
  response. GRB064 shows neither (the rate falls before 50 ms, and ρ = +0.18
  for the flash response). So "units with a visual response" is not part of
  the explanation. Not known: what starts it, and what the side preference of
  single units after the exit is.

## Collinearity of the regressors

Question: are the regressors so correlated that the weights cannot be read?
`collinearity.py` gives the usual statistics for the design of each unit (74
columns: task, motion energy, and the spike history of the unit; the valid
bins of all trials), in `collinearity.json`. Panels g to i of
`design_optimization.svg` draw them for GRB006.

- GVIF^(1/p): the generalized variance inflation factor of a regressor with p
  columns (Fox and Monette, 1992), on the scale of a VIF. It is the factor by
  which the collinearity with the other regressors multiplies the variance of
  its weights. Limits in use: 5 (some concern) and 10 (a problem).
- Condition index (Belsley, Kuh, and Welsch, 1980): 10 to 30 is a moderate
  dependency, above 30 a strong one.

| Regressor | GVIF^(1/p), GRB006: median (max over units) | GRB064 |
| --- | ---: | ---: |
| Choice | 2.15 (2.45) | 1.33 (1.72) |
| Flash, 80+ ms gap | 1.45 (1.51) | 1.58 (1.62) |
| Flash × move | 1.44 (1.47) | 1.45 (1.47) |
| Flash, 40 ms gap | 1.30 (1.34) | 1.39 (1.41) |
| Initiation | 1.26 (1.28) | 1.16 (1.22) |
| Flash × move × gap | 1.25 (1.26) | 1.26 (1.27) |
| Fixation | 1.24 (1.25) | 1.26 (1.28) |
| Withdrawal | 1.21 (1.24) | 1.17 (1.19) |
| Previous choice | 1.19 (1.22) | 1.12 (1.19) |
| Motion energy | 1.17 (1.21) | 1.08 (1.10) |
| Session drift | 1.02 (1.06) | 1.06 (1.07) |
| Spike history | 1.01 (1.11) | 1.02 (1.09) |

- Values of the design with 12 regressors (74 columns). No regressor is near
  the limit of 5 in a unit of the two sessions. The largest value is the
  choice in GRB006 (2.45).
- The largest condition index of a unit is 5.8 in the median (6.6 at most) in
  GRB006 and 6.7 (7.7 at most) in GRB064. No unit has an index above 10, so
  there is no dependency to decompose.
- The VIF of a single column is higher in two regressors, from columns of the
  same regressor: 4.0 for a basis function of the flash kernel of the 40 ms
  gap in GRB064, and 4.0 to 5.2 for session drift. That is the overlap of
  the basis functions of one kernel. It makes one weight of a basis function
  uncertain, not the kernel, which is their sum.
- The first canonical correlation with all other columns is highest for the
  fixation kernel (0.86 and 0.83), flash × move (0.83 and 0.79), and motion
  energy (0.81 and 0.53): one weighted sum of the columns of each is shared
  with the other regressors. The other dimensions are not, so the GVIF stays
  low.
- These are statistics of an ordinary fit. The L2 penalty makes the weights
  more stable than they say, and the split-half test below measures the
  weights directly.
- For two regressors the unique ΔD² is above the maximal ΔD² in most units
  (flash, 40 ms gap: 102 of 167; previous choice: 94 of 167). The cause is
  not strong collinearity: alone, with all other terms shuffled, the term
  must fit a response that is on top of the other responses.

## Are the kernels stable?

Question: can the weights be read, or do correlated regressors make them
change from one set of trials to another? `kernel_reliability.py` fits each
unit on the even and on the odd trials (no shared trial) and compares the
kernels (`kernel_reliability.svg`). Agreement is `1 - |a - b|² / (|a|² + |b|²)`:
1 for equal kernels, 0 for unrelated kernels. Design with 12 regressors; each
half has 147 trials in GRB006 and 84 or 85 in GRB064. The values are for the
units that need the regressor (all units in brackets).

| Regressor | Median agreement, GRB006 | GRB064 | Response over units (Spearman), GRB006 | GRB064 |
| --- | ---: | ---: | ---: | ---: |
| Spike history | 0.97 (0.97) | 0.96 (0.96) | 0.97 | 0.96 |
| Withdrawal | 0.90 (0.75) | 0.82 (0.74) | 0.93 | 0.91 |
| Flash, 80+ ms gap | 0.89 (0.83) | 0.86 (0.73) | 0.94 | 0.95 |
| Flash, 40 ms gap | 0.88 (0.80) | 0.73 (0.64) | 0.93 | 0.94 |
| Flash × move | 0.80 (0.72) | 0.68 (0.63) | 0.93 | 0.85 |
| Flash × move × gap | 0.61 (0.45) | 0.57 (0.36) | 0.62 | 0.39 |
| Fixation | 0.59 (0.51) | 0.83 (0.82) | 0.65 | 0.91 |
| Choice | one weight | | 0.93 | 0.97 |
| Previous choice | one weight | | 0.92 | 0.97 |
| Initiation | one weight | | 0.92 | 0.90 |

- The flash kernels of the two gap classes, flash × move, the withdrawal
  kernel, and spike history are stable: a fit on half of the trials gives the
  same kernel.
- Flash × move × gap is the least stable kernel in the two sessions. The term
  passes the rule over units, but the kernel of one unit is not precise: read
  it over units, not unit by unit.
- The fixation kernel is stable up to 700 ms. Median agreement by time from
  the first flash, GRB006: 0.84 (0 to 150 ms), 0.91 (150 to 450), 0.91 (450
  to 700), 0.26 (700 to 950). GRB064, where the fixation is longer: 0.87,
  0.93, 0.90, 0.73. In GRB006 the animal leaves before 708 ms in 95% of the
  trials, so few trials give data for the last bases. Do not read the
  fixation kernel after 700 ms in GRB006.
- A half has half of the data, so the final refit is more stable than this
  test shows.

## Leak of the fixation level into the flash × move kernels

The synthetic check found it first: with the flash × move kernels in the
model, a planted drop of the rate in the fixation (−0.58, no change of the
flash response) is recovered as −0.20 to −0.41 in the fixation kernel, and the
flash × move kernel becomes positive (+0.11 to +0.13) with nothing planted.
A sum of flash kernels over a flash train is close to a level, so the fit
gives a part of a level change to the flash × move kernels. The usual
collinearity statistics do not show it (the synthetic session has the
statistics of GRB006: condition index 5.2, canonical correlations 0.83 and
0.86), and the split-half test cannot show it, because the two halves have
the same bias. Three checks of `synthetic_check.py` on the fixation kernel
fail for this reason.

`leak_test.py` measures the leak in the real design of each unit
(`leak_test.svg`, `leak_test.json`): spikes are simulated from the final
refit of the unit with the two flash × move kernels set to 0, and the full
model is fit to them (3 simulations for each unit).

| Flash kernel, move minus still (median, Δ log rate) | GRB006 | GRB064 |
| --- | ---: | ---: |
| 80+ ms gap: real | +0.24 | −0.05 |
| 80+ ms gap: leak (nothing simulated) | +0.08 (132 of 167 units above 0, p = 5e-16) | +0.06 (54 of 76, p = 2e-6) |
| 80+ ms gap: real minus leak | +0.15 (117 of 167 above 0, p = 4e-9) | −0.13 (22 of 76 above 0, p = 0.0002) |
| 40 ms gap: real | +0.06 | −0.01 |
| 40 ms gap: leak | +0.03 (p = 2e-6) | +0.04 (p = 3e-5) |
| 40 ms gap: real minus leak | +0.04 (p = 0.069) | −0.04 (p = 0.042) |
| Fixation level of the refit over the simulated level, median | 0.55 | 0.78 |

- The leak is real and positive in the two sessions, and larger in units with
  a lower fixation level (Spearman −0.37 and −0.32).
- With the leak taken out, the flash response at the 80+ ms gap is higher in
  move in GRB006 (+0.15) and lower in move in GRB064 (−0.13). The two animals
  have opposite signs.
- The fixation kernel keeps about 55% (GRB006) and 78% (GRB064) of a simulated
  level, so the fitted fixation level is too small in size. The rest is in the
  flash × move kernels and in the shrinkage of the L2 penalty.
- So the size of the fixation effect and of the flash × move effect cannot be
  read from the weights alone; use the values with the leak taken out, and the
  model-free measures of `hold_effect.py` for the size of the fixation effect.

## Penalty: one for all terms (the model in use), and the sensitivity analysis

Decision of 2026-10-13: the model in use has one L2 penalty for all terms,
selected for each unit. That is the common practice for single-neuron task
GLMs (Park et al. 2014: one ridge penalty, evidence optimization; Musall et
al. 2019: ridge, one penalty for each pixel; Steinmetz et al. 2019: elastic
net with fixed values). With it:

- The leak test is the parameter-recovery check of the model (as in
  Rabinowitz et al. 2015, who simulated known modulators), and the flash ×
  move effect is reported with the leak taken out: +0.15 in GRB006 and −0.13
  in GRB064 at the 80+ ms gap.
- The size of the fixation effect is taken from the raw rates
  (`hold_effect.py`), because the fixation kernel of this model is too small.
- The fit with a weaker penalty on the fixation kernel is the sensitivity
  analysis (`HOLD_PENALTY_SCALE = 10`; fits in `_weaker_fixation_penalty`,
  with no unique-deviance step). A separate treatment of a slow baseline term
  has precedent (a drift term with its own prior in Rabinowitz et al. 2015; a
  drift term in Liska et al. 2024; one penalty for each group of regressors
  in banded ridge, Nunez-Elizalde et al. 2019), but no task GLM that was read
  leaves a state term with a weaker penalty. The conclusions are the same
  with it: +0.09 in GRB006 and −0.16 in GRB064.
- For a term of interest that shares variance with another term, Musall et
  al. (orthogonalization) and Steinmetz et al. (a kernel fit to the residuals
  of the model without it) give the shared part to the other terms. That is
  an alternative to the leak correction and is not run here.

In the tables of the next section, "same penalty" is the model in use and
"weaker" is the sensitivity analysis.

## A weaker penalty on the fixation kernel removes the leak

The leak comes from the L2 penalty. With the same penalty on all terms, the
fit makes the fixation kernel too small, and the flash × move kernels take the
rest of the level: in the simulations, the fixation kernel keeps the full
level at a penalty of 1e-5, 80% at 1e-4, and 39% at 1e-3.

Test on simulated spikes of every unit (the true flash × move effect is 0;
a scratch test, not a module):

| Way to fit the fixation term | Leak, GRB006 | GRB064 | Level kept, GRB006 | GRB064 |
| --- | ---: | ---: | ---: | ---: |
| Same penalty on all terms | +0.086 | +0.058 | 63% | 78% |
| One shared time course for all units, one weight for each unit | +0.093 | +0.058 | 56% | 77% |
| Weaker penalty on all terms (1e-5) | +0.073 | +0.049 | 83% | 95% |
| Penalty 100 times weaker on the fixation kernel only | +0.030 | −0.003 | 101% | 103% |

- A shared time course does not remove the leak: its weight has the same
  penalty.
- `HOLD_PENALTY_SCALE = 10` (the default is 1) makes the columns of the fixation kernel 10 times
  larger than a z-score, so the penalty on its weights is 100 times weaker.
  The saved scale of the columns has the factor in it, so the kernels are
  read as before. Factors of 30 and 100 stop the convergence of the fit.
- Prediction on real data is the same: on 80 units the same penalty is not
  different from the weaker one (`hold_grid`, −0.04%, 38 of 80 units better,
  p = 0.75), and on all units the cross-validated D² does not change (GRB006:
  92 of 167 units better, p = 0.52; GRB064: 46 of 76, p = 0.063).

Both sessions were fit again with the weaker penalty (design of 2026-10-13,
12 regressors; these fits are in `_weaker_fixation_penalty`), and
`leak_test.py` was run on them (`leak_test_weaker_fixation_penalty.json`):

| Median over units, Δ log rate | GRB006, same penalty | GRB006, weaker | GRB064, same penalty | GRB064, weaker |
| --- | ---: | ---: | ---: | ---: |
| Fixation level (kernel, 200 to 450 ms) | −0.24 | −0.54 | −0.46 | −0.71 |
| Simulated fixation level that the refit keeps | 59% | 89% | 79% | 102% |
| Flash kernel, move minus still, 80+ ms gap | +0.24 | +0.13 | −0.05 | −0.14 |
| Leak at the 80+ ms gap | +0.08 | +0.02 (p = 2e-7) | +0.06 | 0.00 (p = 0.82) |
| Real minus leak, 80+ ms gap | +0.15 | +0.09 (p = 3e-5) | −0.13 | −0.16 (p = 3e-7) |
| Flash kernel, move minus still, 40 ms gap | +0.06 | 0.00 | −0.01 | −0.04 |
| Real minus leak, 40 ms gap | +0.04 | −0.01 (p = 0.53) | −0.04 | −0.06 (p = 0.003) |

- The fixation level of the kernel is now −0.54 in GRB006. The raw rate in
  the fixation is 0.56 of the rate before the first flash (log −0.58), so the
  kernel now agrees with the model-free measure.
- The leak is gone in GRB064 and small in GRB006 (+0.02).
- The flash response at the 80+ ms gap is a little higher in move in GRB006
  (+0.09 with the leak taken out) and lower in move in GRB064 (−0.16). The two
  animals have opposite signs. At the 40 ms gap there is no change in GRB006
  and a small decrease in GRB064.
- The numbers of the other sections (flash kernels by period, kernel
  stability, collinearity, unique ΔD²) are from the model in use, with the
  same penalty on all terms.

## Performance in the measures of other papers

`performance.py` (`performance.svg`, `performance.json`): each trial is
predicted by the fold that did not train on it, with the task and
motion-energy terms and no spike history, and compared with the observed
spikes. Median over units of the Pearson correlation (r² in brackets):

| | GRB006 (167 units) | GRB064 (76 units) |
| --- | ---: | ---: |
| Single trials, 1-ms bins | 0.04 | 0.05 |
| Single trials, 10-ms bins | 0.12 (0.015) | 0.16 (0.024) |
| Single trials, 33-ms bins | 0.20 (0.04) | 0.27 (0.07) |
| Single trials, 100-ms bins | 0.27 (0.07) | 0.42 (0.17) |
| Mean over held-out trials, 20-ms bins | 0.88 (0.77) | 0.87 (0.76) |

- The model predicts the mean time course of a unit well and single trials
  weakly. With the spike-history term the prediction of a wide bin has the
  spikes of that bin in it, so it is not used here.
- Values from other papers, with different signals, bins, and predictors (no
  direct comparison): Musall et al. 2019, two-photon neurons in V1 at 30 Hz,
  cvR² 31% with many video and movement regressors; Akella et al. 2025,
  Neuropixels units of the visual cortex with rates smoothed at 50 ms, cvR²
  27 ± 14% with stimulus, behavior, and the activity of other areas and LFP
  (stimulus alone 20%); Parker et al. 2022, V1 units of freely moving mice in
  50-ms bins smoothed over 2 s, mean correlation 0.28 for the visual model.

## Figures

| Figure | Module | Content |
| --- | --- | --- |
| Main | `main_figure.py` | one trial, the kernels, and the prediction of the example unit (GRB006); the population and what each regressor explains, both sessions |
| Flash kernels | `flash_kernels.py` | the flash kernel by gap class and by period (still, move): median kernels of the units that fire more and less after a flash, maps of all units, paired scatters; both sessions |
| Flash by gap | `flash_by_gap.py` | observed and predicted rate around flashes at each gap, still and move; both sessions |
| Still against move | `flash_by_period.py` | raw response and flash kernel in the two periods, both sessions |
| Fixation effect | `hold_effect.py` | two example units (GRB006), the lower rate in the fixation, its end at the exit, and its controls; both sessions |
| Design optimization (supplement) | `design_optimization.py`, `collinearity.py`, `parameter_recovery.py` | step 1: the six design grids on 80 units of GRB006, each with its test of prediction; step 2, in GRB006 and the held-out GRB064: the collinearity of the design in use (variance inflation, condition indices, shared direction of each pair of regressors) and the parameter recovery of each regressor |
| Basis functions (supplement) | `basis_diagram.py` | the basis functions of each kernel |
| Checks of the model (supplement) | `model_checks.py` | single trials, held-out units (and the held-out animal GRB064), terms that are not in the model, calibration (both sessions), recovery of planted kernels |
| Kernel stability (supplement) | `kernel_reliability.py` | the kernels from two halves of the trials; both sessions |
| Leak test (supplement) | `leak_test.py` | the flash × move effect that the model makes when none is simulated, against the real effect |
| Flash against motion energy | `kernel_comparison.py` | the weights of the two terms over units; descriptive |

`main_figure.py`, `hold_effect.py`, `design_optimization.py`, and
`model_checks.py` write their caption next to the figure
(`<figure>_caption.md`), with the numbers of that run. The captions of the
other figures:

- **Flash kernels (`flash_kernels.svg`).** The flash kernel of the model by
  the gap from the flash before and by the period, in GRB006 (a, b, e, g–i)
  and GRB064 (c, d, f, j–l); the units of the two animals are not pooled.
  "Still" is a flash in the fixation and "move" is a flash after the
  withdrawal from the center port (task periods; locomotion is not measured).
  **a, c**, The median kernel over the units that fire more after a flash
  (red): left, a flash when still at a gap of 40 ms and of 80 ms or more;
  right, a flash at a gap of 80 ms or more when still and in move (dashed),
  with the interquartile range. **b, d**, The same for the units that fire
  less after a flash (blue). The kind of a unit is from its mean kernel
  response when still (above 0.2 or below −0.2 in Δ log rate), so the sign of
  the curves is by selection. **e, f**, The kernel of every unit in the four
  conditions, with one row for each unit and the same order in all maps of an
  animal (by the mean response when still); the color scale stops at ±1.
  **g–l**, The response (mean of the kernel from 40 to 100 ms) of each unit in
  two conditions; error bars are the SD over the 10 cross-validation folds.
  **g, j**, The 80+ ms gap against the 40 ms gap, when still. **h, k**, Move
  against still at the 80+ ms gap. **i, l**, Move against still at the 40 ms
  gap. In g to l the color is the kind of the unit from conditions that are
  not in the pair, and gray is a unit of no kind. The model gives a part of
  the fixation effect to the flash × move kernels, so the move-against-still
  differences are too high by about 0.08 (GRB006) and 0.06 (GRB064) at the
  80+ ms gap (`leak_test.py`).
- **Still against move (`flash_by_period.svg`).** One measure on three rates
  of the same flashes and units: the peak of the trial-mean rate (10-ms bins)
  from 30 to 120 ms after the last flash in the fixation (still) and the first
  flash after the withdrawal (move), minus the still rate from −40 to 0 ms, as
  in `stim_responses_by_locomotion.py`. Units: the "excited" units of
  StimulusResponse that the GLM fits (23 in GRB006, 15 in GRB064). Circles
  GRB006, triangles GRB064; small markers are units, large markers the median
  with its bootstrap 95% interval. **a**, Observed peaks. **b**, Peaks of the
  model rate (final refit, in-sample). **c**, Move minus still in the observed
  rate, the model rate, and the model rate with the flash × move terms set to
  0. The peak is higher in move in both animals (median +12.7 and +24.6
  spikes/s; 23 of 23 and 13 of 15 units). The model predicts most of it
  (+8.9 and +20.6), and the flash × move terms carry +2.6 (p = 0.01) and
  +1.3 spikes/s (p = 0.8). So in both animals most of the change is the rate
  change at the exit, not a larger response to the flash.
- **Flash by gap (`flash_by_gap.svg`).** Observed (black) and predicted
  (purple) rate around flashes, for the units that fire more and that fire
  less after a flash, in GRB006 (a, b) and GRB064 (c, d); the number of units
  of each kind is in the title of the block. One row for each gap from the flash before (40, 80,
  120, 160, and 200 ms or more), with the time from the flash before; the
  dashed line is the flash before and the solid line is the flash. Left
  column: still. Right column: move. The rate of a unit is divided by its
  mean rate around these flashes; lines are the mean over units.
- **Basis functions (`basis_functions.svg`).** Each kernel is a weighted sum
  of these functions, each drawn with a peak of 1. **a**, The flash kernels
  (the two gap classes, flash × move, and flash × move × gap): 8 raised
  cosines over 150 ms. **b**, Fixation: 12 raised cosines with even spacing
  in log time over 950 ms; after the last one the kernel keeps its last value
  until the withdrawal. **c**, Withdrawal: 4 over 100 ms. **d**, Spike
  history: 10 with log spacing over the 100 ms before the bin. **e**, Session
  drift: 4 over the session; the dotted one is not a column, because the
  model has an intercept. The initiation, the choice, the previous choice,
  and each motion-energy PC are one column each.
- **Kernel stability (`kernel_reliability.svg`).** Each unit is fit on the
  even and on the odd trials. **a**, The kernels of the example unit (GRB006)
  from the two halves. **b**, The agreement of the two kernels of each unit,
  1 − |a − b|² / (|a|² + |b|²), for each kernel regressor, GRB006 on the left
  and GRB064 on the right (color: units that need the regressor; gray: other
  units; line: the median of the units that need it). **c, d**, The response
  of each unit in one half against the other half, in GRB006 (c) and GRB064
  (d).
- **Leak test (`leak_test.svg`).** For each unit, spikes are
  simulated from its fitted model with the flash × move kernels set to 0, and
  the full model is fit to them. One row for each session. **a**, The real
  flash × move effect of each unit (move minus still, 80+ ms gap) against the
  effect that the model gives for the simulated spikes (the leak; the true
  value is 0); error bars are the SD over 3 simulations. **b**, The same at
  the 40 ms gap. **c**, The leak against the fixation level of the unit. The
  leak is +0.08 in GRB006 and +0.06 in GRB064 at the 80+ ms gap (median).

`glm fit` writes no figure. `glm unique` writes `figure_data.npz` for the
example unit.

## Synthetic check

Run this after you change the design, fit, or unique-deviance code:

```bash
uv run python -m thesis.ephys.analyses.glm.synthetic_check
```

It needs no database. It plants known flash, fixation, withdrawal,
motion-energy, and history effects in 80 simulated trials with the flash train
of the task. Then it runs `prepare`, `fit`, and `unique` on them and checks that
the effects come back. It compares 2 and 5 motion-energy PCs and takes about 2
minutes. It raises `AssertionError` with the failed checks. The checks of the
size of the fixation kernel report the known bias of the model and do not
stop the command (see "Leak of the fixation level"). Pass `--out DIR` (a new
directory) to keep the artifacts; `synthetic_recovery.json` there has the
planted and recovered kernels, which `model_checks.py` draws.

## Kernel comparison

```bash
uv run python -m thesis.ephys.analyses.glm.kernel_comparison
```

This script makes no new fits. It reads one fit set (`--units`, `all` by
default) in the session output directory and writes `kernel_comparison.svg` and
`kernel_comparison.json`. Panel a compares the size of the flash weights (the kernels of the two gap
classes together) and of the motion-energy weights (the norm of the
coefficients) for every unit. Panels b and c show the flash kernel at a gap of
80 ms or more and the motion-energy weight map of three example units: one that the flash predicts,
one that motion energy predicts, and one that both predict. Motion energy has no
temporal kernel, so its panel is a map over the video frame. The module
docstring gives the selection rule.

## Related methods

Same family as [Truccolo et al. 2005](https://pubmed.ncbi.nlm.nih.gov/15356183/)
and [Pillow et al. 2008](https://sites.stat.columbia.edu/liam/research/pubs/pillow-nature-08.pdf),
with a self-history regressor but no coupling between units.
[Talluri et al. 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10620084/)
(video SVD, ridge, trial CV),
[Stringer et al. 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6525101/)
(video as broad behavior). Basis counts and windows remain assumptions.
