# V1 GLM

Poisson encoding model of V1 spikes in the Chipmunk task, fitted on two
sessions: GRB006 `20240821_121447` (167 units) and GRB064 `20260916_132939`
(76 units). The model predicts the spike count of each unit in 1-ms bins from
11 regressors (design of 2026-10-16): two flash kernels by the gap from the
flash before, a flash × after exit kernel (the change of the flash response after
the center exit), a fixation kernel (first flash to center exit), one
pre-stimulus column (center entry to first flash), a response-period kernel
and a turn-direction kernel (center exit to response entry), the previous
choice, 10 motion-energy PCs of the video, a session drift, and a strictly
past spike-history filter for each unit. All terms add in log rate.
Coefficients are conditional associations, not causal effects.

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
- Grid: DAMN 1 ms, pre=500 ms, post=2.1 s (2.6 s). The window starts
  before the center entry in most trials, so the baseline is the rate before the
  entry (to 2026-10-15: pre=100 ms, post=2.54 s). Keep the full
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
| Pre-stimulus (key `fixation`) | 1 from center entry to the first flash, else 0; no temporal basis | 1 |
| Fixation (key `hold`) | One kernel on the first flash, set to zero from the center exit on; 10 raised cosines with even spacing in log(t + 50 ms), 0–700 ms; after 700 ms the last value is kept to the exit; no step | 10 |
| Response period | One kernel on the center exit, to the response entry (bins after it are masked); 8 raised cosines with even spacing in log(t + 25 ms), 0–1.3 s; after 1.3 s the last cosine keeps its level (`exit_basis`) | 8 |
| Turn direction | The same bases, times +1 or −1 for the side of the response | 8 |
| Previous choice (key `previous_side`) | +1 or −1 for the side of the last response before the trial, from the start of the window to the center exit | 1 |
| Motion energy | One contemporaneous value for each of the first 10 PCs; no temporal basis | 10 |
| Session drift | 4 raised cosines over the session, with the last one left out | 3 |
| Self-history | 10 log-spaced raised cosines at strictly past lags, 1–100 ms | 10 |

75 columns in 11 regressors. `basis_diagram.py` draws the basis functions
(`figures/glm/basis_functions.svg`); panel a of `main_figure.svg` draws the
design of one trial.

- The ends of the two exit kernels come from the behavior, not from a grid:
  the exit comes by 0.7 s after the first flash in 95% of the trials of both
  mice, and the response entry by 1.3 s after the exit. A kernel past these
  times has few trials and is not stable in split halves (the fixation kernel
  of 0.95 s had an agreement of 0.26 from 700 to 950 ms in GRB006).
- Response period and turn direction split the time after the exit into a part
  common to both sides and a part of opposite sign for left and right. The
  right-turn kernel is response period + turn direction, the left-turn kernel
  response period − turn direction.

Names in the figures: a flash in the fixation is "in port" and a flash after the
center exit is "after exit" (task periods, not a measure of locomotion); the side of
the response after the exit is "choice"; the side of the last response before
the trial is "previous choice".

Design of 2026-10-16, fitted on all units of GRB006 (167 units, median
cross-validated D² 0.072, IQR 0.044 to 0.103) and GRB064 (76 units, 0.074,
IQR 0.049 to 0.114). It is the design of 2026-10-15 with a coarser fixation
kernel: 10 cosines with a 50 ms stretch in place of 12 with 25 ms, because it
predicts better on the design of 2026-10-15 (+0.12%, 50 of 80 units,
p = 0.0047) and the kernels are recovered as well (`parameter_recovery
--grid hold_grid`; the fixation level is recovered at 0.67 and 0.85 against
0.66 and 0.84). The results change little: D² 0.0716 → 0.0717 and 0.0744 →
0.0744; all regressors still pass. The fits with the 12-cosine kernel are in
`figures/glm/_before_coarse_kernel_2026_10_16/`.

Changes of 2026-10-15 from the design of 2026-10-13 (decided with
Gabriel, 2026-10-15; the test behind each is in `candidate_terms.py` and in
[glm_decisions.md](glm_decisions.md)):

- Turn direction (8 bases, to the response entry) replaces the choice column
  (`side_after_exit`). The two had a first canonical correlation of 0.9.
- The response period replaces the withdrawal kernel (0 to 100 ms).
- Flash × move × gap is out (split halves 0.61 and 0.57; not reliable after
  the leak correction).
- The fixation kernel ends at 0.7 s, not 0.95 s.
- The window starts 0.5 s before the first flash, not 0.1 s.
- Time in trial is out: it had a canonical correlation of 1.00 with the
  fixation kernel. Flash × motion energy stays out (fails in GRB064, where the
  camera view is different). Accumulated evidence and the previous outcome
  fail the rule in both mice.

The fits of the design of 2026-10-13 are in
`figures/glm/_before_design_2026_10_15/`.

No go cue, outcome, response-entry, same-bin history, cross-unit coupling, or
previous-outcome terms. Motion-energy coefficients are contemporaneous
nuisance associations. Pose tracking is not included.

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
as `video_trial_specific`.

Regressors that do not follow the time in the trial are shuffled in another
way (`SHUFFLE_RULE` "trial variables over trials, sides by sign, 2026-10-16"):

- Session drift and the previous outcome do not change in a trial, so their
  rows are shuffled over trials.
- A side on a time course (previous choice, turn direction) keeps its own rows
  and takes the sign of another trial. A shuffle of the rows over trials moves
  the exit-aligned time course to another trial, where the exit comes at
  another time. That made the shuffle remove the response period also: in a
  synthetic unit with only flash responses, turn direction had a unique ΔD² of
  0.19. With the sign shuffle it is 0. The real results change little (turn
  direction is needed by 49% and 63% of the units).

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
and the flash × after exit term were not reliable on 20 units and are reliable
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
writes them for the model in use to `main_figure_stats.json` (all 11 regressors
pass both tests in the two sessions).

## Parameterization of each term in the final design, on 80 units

The seven grids were run on 4 sets of 20 units of GRB006 (`--samples 4`, 80
units) with the design in use (2026-10-16) as the base
(`design_optimization.svg`). Each row is a form against the form in use
(paired over the 80 units, two-sided Wilcoxon). The grids of the earlier base
designs are in `figures/glm/_before_design_2026_10_15/`,
`figures/glm/_before_coarse_kernel_2026_10_16/`, and the session folder
(`<grid>_earlier_design`, `<grid>_eleven_regressors`).

| Grid | Form | Median change | Units better | p |
| --- | --- | ---: | ---: | ---: |
| Gap classes (in use: 40, 80+) | 1 kernel | −1.24% | 28 | 9e-5 |
| | 1 kernel + a kernel for a flash up to 50 ms before | −0.03% | 39 | 0.56 |
| | 40, 80 to 120, 160+ | −0.24% | 36 | 0.33 |
| | 40, 80, 120+ | −0.32% | 30 | 0.12 |
| | 40, 80, 120, 160+ | −0.55% | 30 | 0.026 |
| Flash kernel length (in use: 150 ms) | 200 ms | +0.16% | 44 | 0.35 |
| | 250 ms | −0.03% | 40 | 0.53 |
| | 300 ms | −0.59% | 35 | 0.16 |
| Fixation term (in use: 10 cosines, log(t + 50 ms)) | none | −3.38% | 8 | 1e-12 |
| | step from the center entry | −1.20% | 16 | 1e-8 |
| | step from the first flash | −1.35% | 19 | 2e-8 |
| | linear spacing | −0.31% | 27 | 0.024 |
| | 12 cosines, log(t + 25 ms) (in use until 2026-10-16) | −0.12% | 30 | 0.0047 |
| | no pre-stimulus column | −0.07% | 29 | 0.0073 |
| | penalty 100 times weaker on the kernel | +0.02% | 42 | 0.91 |
| End of the fixation kernel (in use: 0.7 s) | 0.5 s | −0.06% | 35 | 0.12 |
| | 0.95 s | −0.10% | 36 | 0.34 |
| Exit basis (in use: 8 cosines to 1.3 s) | withdrawal kernel 0 to 100 ms in place of the response period | −0.93% | 17 | 4e-7 |
| | to 1.0 s | −0.08% | 38 | 0.35 |
| | to 1.6 s | −0.04% | 38 | 0.56 |
| | 6 cosines | −0.18% | 28 | 0.095 |
| | 10 cosines | −0.05% | 37 | 0.22 |
| Motion-energy PCs (in use: 10) | none | −0.17% | 32 | 0.16 |
| | 25 | −0.43% | 29 | 0.016 |
| | 50 | −1.01% | 25 | 0.0007 |
| | 100 | −3.27% | 20 | 9e-7 |
| | 200 | −6.03% | 7 | 7e-12 |
| Session drift (in use: 4 raised cosines) | none | −0.42% | 29 | 0.031 |
| | 8 | +0.04% | 42 | 0.92 |
| | 12 | −0.56% | 26 | 0.005 |
| | cubic spline | +0.02% | 40 | 0.63 |

- The design is made in two steps.
  1. The form of each term is chosen by prediction: another form replaces the
     one in use when its validation D² is reliably higher on these units
     (paired Wilcoxon) and the kernels are recovered as well.
  2. The collinearity of the regressors of the resulting design is measured
     with the published diagnostics (see "Collinearity of the regressors"):
     the generalized variance inflation factor (Fox and Monette 1992; limit
     5) and the condition indices (Belsley, Kuh, and Welsch 1980; limit 30).
     A regressor that is collinear with another must go, because its weights
     cannot be read.
- Step 1: no other form predicts reliably better than the forms in use.
  - The one change of 2026-10-16 came from this rule: on the design of
    2026-10-15 the coarse fixation kernel predicted better than the
    12-cosine kernel then in use, and it passed parameter recovery (the error
    of the fixation level and of flash × after exit did not grow: p from 0.67 to
    0.94). Now the 12-cosine kernel is the worse one (−0.12%, p = 0.0047).
  - "1 kernel + a kernel for a flash up to 50 ms before" has the same columns
    as the two classes in another form, so only the penalty differs; it
    predicts the same (p = 0.56).
  - The response period predicts better than the withdrawal kernel that it
    replaced (+0.9%, 63 of 80 units).
  - Without motion energy the model is not reliably worse on these 80 units
    (p = 0.16), and without session drift it is (p = 0.031); both pass the
    inclusion rule on all units, which is the test of whether a regressor is
    needed (see "The rule for a regressor").
  - Kernels that start before the exit (the withdrawal grid, on the design of
    2026-10-13 only, which had the withdrawal kernel): a kernel from 300 ms
    before to 300 ms after the exit predicted better (+0.58%, p = 0.010) but
    failed parameter recovery of the fixation level (recovered at 0.55 and
    0.73 against 0.76 and 0.84), so the terms at the exit start at the exit.
- Step 2, for the design in use, in GRB006 and in the held-out animal GRB064
  (no choice of step 1 used GRB064); panels a to e of `model_trust.svg`:
  - 2a, collinearity: no regressor is above the limits in either animal
    (largest GVIF^(1/p) 1.63; largest condition index 8.4), so no regressor
    goes.
  - 2b, parameter recovery of every regressor (`parameter_recovery.py`):
    spikes are simulated from the fitted model of each unit, the model is fit
    again to them with the same penalty, and each recovered kernel (or weight
    vector, for motion energy and drift) is compared with the true one. The
    median agreement is 0.68 to 0.85 in GRB006 (spike history 0.66) and 0.73
    to 0.97 in GRB064. The median size of the recovered kernel over the true
    size is 0.82 to 1.13, except spike history in GRB006 (0.53; see "Spike
    history and the firing rate"). The level of the fixation kernel is
    recovered at 0.70 (GRB006) and 0.80 (GRB064) of its size: the leak of
    "Leak of the fixation level". So the fitted fixation level is a lower
    bound.

## Results

Model in use: the design of 2026-10-16 (11 regressors, 75 columns), one L2
penalty for all terms, selected for each unit. Artifacts are in
`figures/glm/<subject>_<session>/`, which git does not track.

| | GRB006 20240821_121447 | GRB064 20260916_132939 |
| --- | ---: | ---: |
| Units, trials | 167, 294 | 76, 169 |
| Video | back view | bottom view |
| Median cross-validated D² (IQR) | 0.072 (0.044 to 0.103) | 0.074 (0.049 to 0.114) |
| Regressors that pass the two tests of the rule | 11 of 11 | 11 of 11 |

| Regressor | Columns | GRB006: units that need it | units worse | median unique ΔD² | GRB064: units that need it | units worse | median unique ΔD² |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Spike history | 10 | 96% | 1.2% | 0.0307 | 97% | 0.0% | 0.0292 |
| Fixation | 10 | 78% | 0.0% | 0.0028 | 91% | 0.0% | 0.0039 |
| Response period | 8 | 78% | 0.0% | 0.0018 | 84% | 0.0% | 0.0027 |
| Flash, 80+ ms gap | 8 | 75% | 0.6% | 0.0022 | 58% | 0.0% | 0.0011 |
| Flash, 40 ms gap | 8 | 72% | 0.0% | 0.0020 | 50% | 0.0% | 0.0007 |
| Turn direction | 8 | 50% | 1.2% | 0.0009 | 64% | 0.0% | 0.0018 |
| Flash × move | 8 | 43% | 0.6% | 0.0006 | 43% | 0.0% | 0.0008 |
| Motion energy | 10 | 42% | 1.2% | 0.0006 | 45% | 1.3% | 0.0006 |
| Session drift | 3 | 28% | 1.8% | 0.0003 | 14% | 0.0% | 0.0002 |
| Previous choice | 1 | 20% | 1.8% | 0.0002 | 21% | 1.3% | 0.0001 |
| Pre-stimulus | 1 | 19% | 0.6% | 0.0001 | 26% | 0.0% | 0.0001 |

Chance gives 17 units or fewer of 167 (10%) and 10 or fewer of 76 (13%) for 11
regressors. Session drift passes only just in GRB064 (11 units).

Findings, with the section that has the details:

- **The rate is lower in the fixation, and the effect ends at the
  exit.** Raw rate (each unit over its mean rate), GRB006: 1.04 before the
  first flash, 0.62 in the last 200 ms of the fixation, 1.37 from 100 to 400
  ms after the exit; higher after the exit in 148 of 167 units (p = 7e-22).
  GRB064: 0.98, 0.62, 1.32; 63 of 76 (p = 2e-10). The fixation kernel is too
  small in size (parameter recovery gets back 70% of a simulated level in
  GRB006 and 80% in GRB064), so the size of the effect is taken from the raw
  rates. ("What causes the fixation effect"; "Leak of the fixation level".)
- **The flash response has two forms by the gap from the flash before.** One
  kernel for all flashes is worse than two gap classes (80 units, −1.2%,
  p = 9e-5); three or four classes are not better.
  ("Parameterization of each term in the final design".)
- **The flash × after exit part of the flash peak is not more than the leak.** In
  the matched peak measure (excited units), real minus leak is +1.0 spikes/s
  in GRB006 (23 units, p = 0.20) and −2.3 in GRB064 (15 units, p = 0.25);
  pooled +0.80 [−4.9, 1.6]. In the kernel measure (all units, Δ log rate),
  real minus leak is +0.03 in GRB006 (p = 0.03) and −0.16 in GRB064
  (p = 2e-5): the two animals have opposite signs. ("Leak of the fixation
  level".)
- **The time after the exit has a part common to both sides and a part by
  side.** The response period is needed by 78% and 84% of the units, turn
  direction by 50% and 64%.
- **The previous choice changes the rate before the exit** in 20% and 21% of
  the units. On raw rates about half of the units differ by it, so the one
  column holds a part of the effect.
- **Checks** (`model_trust.svg`). Collinearity is low (largest GVIF^(1/p)
  1.63, largest condition index 8.4). Every kernel is stable in split halves
  (median agreement 0.69 or more in the units that need it). Parameter
  recovery is good, with two known biases: the fixation level (a lower
  bound), and spike history in units below 5 spikes/s.
- **Known limits of the model.**
  - The fitted fixation level is a lower bound: recovered at 0.70 and 0.80 of
    its size. The rest goes to the response period and to flash × after exit.
  - Spike history is read only in units at 5 spikes/s or more (78 of 167 and
    50 of 76 units). In them, recovery agreement is 0.99 and size 0.91 to
    0.92; below, agreement is 0.24 and 0.44.
  - Calibration (GRB006, design of 2026-10-13): observed over predicted rate
    is 0.81 to 0.93 from the 10th to the 65th percentile of the predicted
    rate, and 1.05 to 1.12 from the 80th to the 95th. The model predicts too
    much at low and middle rates and too little at high rates. The cause is
    not known.
  - "Move" is a task period. No term of the model measures locomotion; a
    flash kernel scaled by motion energy passes in GRB006 and fails in GRB064
    (different camera views).
  - Two mice, one session each.
- Still open: the side index with turn direction, and the link of the model
  to locomotion and the decoding analyses.

## What causes the fixation effect

### The fixation effect in both animals

`hold_effect.py`, design of 2026-10-16 (GLM units; window from 0.5 s before the first flash). Each test is within one animal. GRB064 took no
part in the design choices.

| Measure | GRB006 (167 units) | GRB064 (76 units) |
| --- | ---: | ---: |
| Normalized rate before the first flash, last 200 ms of the fixation, 100 to 400 ms after the exit | 1.04, 0.62, 1.37 | 0.98, 0.62, 1.32 |
| Fixation level of the descriptive model, median Δ log rate (units below 0) | −0.32 (90%) | −0.46 (84%) |
| Level after the exit, median Δ log rate | +0.08 | +0.10 |
| Units higher after the exit than in fixation | 148 of 167, p = 7e-22 | 63 of 76, p = 2e-10 |
| Fixation kernel fit last (lower bound), median (units below 0) | −0.24 (90%), p = 6e-27 | −0.22 (84%), p = 4e-10 |
| Stillness: center port over side port at the lowest motion energy | 0.78, p = 3e-7 | 0.84, p = 0.32 |
| Response after the exit, more over less motion energy | 0.97, p = 0.54 | 1.00, p = 0.67 |
| Fixation level against the unique ΔD² of the flash (Spearman; design of 2026-10-13) | ρ = −0.20, p = 0.01 | ρ = +0.10, p = 0.41 |

- The effect replicates in GRB064. The raw rates are almost the same in the
  two animals, and the model gives a level below 0 in most units of each.
- The fitted levels are lower bounds. In the main model, a simulated fixation
  level is recovered at 0.70 (GRB006) and 0.80 (GRB064) of its size
  (`parameter_recovery.py`); the rest goes to the response period and to
  flash × after exit.
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

### What starts and ends the effect

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
- GRB006 (2026-10-16, after `reextract_flash_onsets.py` added the 1170
  flashes of the early-withdrawal trials to the database; 167 units, 207
  early withdrawals with an exit 445 ms after the first flash, 294 completed
  trials with the exit at 553 ms; `figures/exit_without_stimulus/GRB006_20240821_121447/`).
  The result replicates: the rate after the exit is the same with and without
  the stimulus (early over completed 1.01, 89 of 167 units higher, p = 0.57),
  and it goes up at the exit in both (2.45 times in completed trials, 146 of
  167 units, p = 6e-22; 1.86 times in early withdrawals, 137 of 166, p = 2e-18).
  The rise is smaller in early withdrawals because the rate before the exit is
  higher (early over completed 1.13, 143 of 166 units, p = 3e-18): the
  fixation is shorter (445 against 553 ms), and in GRB006 the drop starts
  about 90 ms after the first flash and goes on through the fixation. In
  GRB064 the rate before the exit is the same in both (1.00, p = 0.92).
  The flashes of the completed trials did not change with the new row (3520
  onsets, the same as before), so the GLM fits are not affected.
- In GRB064 the fixation of the
  completed trials has almost the same length as in GRB006 (1st to 95th
  percentile: 455 to 747 ms), so the matched-time test is not possible there
  either; the early withdrawals are what separates the exit from the time in
  the train.
- Does the drop start before the stimulus? (2026-10-15,
  `suppression_onset.py`, no model; panel e of `hold_effect.svg`.) The time
  from the center entry to the first flash differs between trials (5th to
  95th percentile 26 to 99 ms in GRB006, 31 to 106 ms in GRB064). In the
  trials with a long delay (the upper half; median 73 and 83 ms), the rate of
  each unit from 20 ms after the entry to the first flash (the port with no
  stimulus; median window 53 and 63 ms) is compared with its rate 250 to
  100 ms before the entry, paired over units. For scale, the rate 180 to
  290 ms after the first flash over the same rate before the entry is the full
  drop.

  | Mouse | Units, long-delay trials | Entry to first flash, over before entry | 180–290 ms after the first flash, over before entry |
  | --- | --- | --- | --- |
  | GRB006 | 166, 148 | 0.95 (lower in 101 units, p = 0.002) | 0.59 |
  | GRB064 | 75, 120 | 0.70 (lower in 58 units, p = 4e-8) | 0.47 |

  On a log scale, about 10% of the full drop comes before the first flash in
  GRB006 and about half in GRB064. So the drop does not need the stimulus to
  start, and in GRB064 a large part of it comes with the entry. Two-sided
  Wilcoxon signed-rank tests of the log ratio; the units share the trials.
  The units are those of the GLM (without `fit.EXCLUDED_UNITS`), as in every
  analysis that is compared with it; one unit of each mouse has no spike
  before the entry and no ratio. Before this rule, GRB064 unit 414 (excluded
  from the GLM: 1 spike in all 239 trials) gave a large false peak in the
  normalized population rate.
  Limit: the population rate also goes down slowly in the 100 ms before the
  entry, while the animal comes to the port, so a part of the drop may follow
  the approach and not the entry. The trial table has no measure of the
  approach that could separate them.
  - This replaces the time of the drop (2026-10-12): the first time after
    40 ms at which the smoothed rate is below the middle of its levels before
    and late. It gave 0 ms (GRB006) and −30 ms (GRB064) for the shift of the
    long against the short delays from the entry. In GRB064 the rate was
    already below that middle level before 40 ms, so the measure gave the
    start of its search and not a time of the drop. The −30 ms is not a
    result.
- The side preference of single units does not need the stimulus (2026-10-15,
  `side_preference.py`, both sessions, no model, with the corrected response
  entries of GRB064; for the turn-direction work, not in `hold_effect.svg`). Side index of a unit:
  (right − left) / (right + left) of its rate. With the stimulus on after the
  exit (100 ms after the exit to 100 ms after the last flash), about half of
  the units have an index above 0.2 in size (50% of 167 units in GRB006, 53%
  of 76 in GRB064; split-half Spearman ρ = 0.86 and 0.85). The index with the
  stimulus on agrees with the index with the stimulus off (150 ms after the
  last flash to the response entry, in the dark; split-half ρ = 0.59 and
  0.79): ρ = 0.51 in GRB006 and 0.57 in GRB064, against a limit of about 0.71
  and 0.82 from the split-half values (the geometric mean of the two). In the
  fixation the index is small and has no relation to the index after the exit
  (ρ = −0.07 and −0.08). The units are those of the GLM. So the side preference is a
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
- Only in GRB006: most of the suppression comes after the first flash (about
  90% on a log scale), and it is larger in units with a larger flash response.
  In GRB064 about half of the drop comes between the entry and the first
  flash, and ρ = +0.10 for the flash response. So "units with a visual
  response" is not part of the explanation, and the drop starts with the
  entry into the port (or the approach to it) in both mice, with a part that
  follows the first flash. Not known: what starts it, and what the side
  preference of single units after the exit is.

The tests of 2026-10-11 on what causes the fixation effect (GRB006; matched
flashes, the flash × fixation term, the view, the latency) are in
[glm_decisions.md](glm_decisions.md), "Tests of 2026-10-11".

## Collinearity of the regressors

Question: are the regressors so correlated that the weights cannot be read?
`collinearity.py` gives the usual statistics for the design of each unit (77
columns: task, motion energy, and the spike history of the unit; the valid
bins of all trials), in `collinearity.json`. Panels a to c of
`model_trust.svg` draw them for both mice.

- GVIF^(1/p): the generalized variance inflation factor of a regressor with p
  columns (Fox and Monette, 1992), on the scale of a VIF. It is the factor by
  which the collinearity with the other regressors multiplies the variance of
  its weights. Limits in use: 5 (some concern) and 10 (a problem).
- Condition index (Belsley, Kuh, and Welsch, 1980): 10 to 30 is a moderate
  dependency, above 30 a strong one.

| Regressor | GVIF^(1/p), GRB006: median (max over units) | GRB064 |
| --- | ---: | ---: |
| Response period | 1.56 (1.60) | 1.37 (1.43) |
| Flash, 80+ ms gap | 1.50 (1.55) | 1.60 (1.63) |
| Motion energy | 1.34 (1.35) | 1.11 (1.12) |
| Flash, 40 ms gap | 1.33 (1.36) | 1.42 (1.43) |
| Turn direction | 1.32 (1.38) | 1.14 (1.20) |
| Fixation | 1.27 (1.28) | 1.30 (1.34) |
| Previous choice | 1.25 (1.29) | 1.12 (1.19) |
| Flash × move | 1.18 (1.21) | 1.21 (1.21) |
| Pre-stimulus | 1.12 (1.13) | 1.16 (1.20) |
| Session drift | 1.02 (1.05) | 1.05 (1.06) |
| Spike history | 1.01 (1.12) | 1.03 (1.10) |

- No regressor is near the limit of 5 in a unit of the two sessions. With
  the design of 2026-10-13 the largest value was the choice column (2.45);
  turn direction, its replacement, is at 1.38.
- The largest condition index of a unit is 6.1 in the median (6.8 at most) in
  GRB006 and 7.5 (8.4 at most) in GRB064. No unit has an index above 10, so
  there is no dependency to decompose.
- The VIF of a single column is highest in session drift (3.9 and 5.7) and in
  one basis function of the flash kernel of the 40 ms gap in GRB064 (4.1).
  That is the overlap of the basis functions of one kernel. It makes one
  weight of a basis function uncertain, not the kernel, which is their sum.
- The first canonical correlation with all other columns is highest for
  motion energy (0.88 and 0.59), the response period (0.87 and 0.81), and the
  fixation kernel (0.86 and 0.85). The largest shared directions are between
  motion energy and turn direction (0.73, GRB006), and between the response
  period and flash × after exit (0.70 and 0.68). One weighted sum of the columns of
  each is shared with the other regressors; the other dimensions are not, so
  the GVIF stays low.
- These are statistics of an ordinary fit. The L2 penalty makes the weights
  more stable than they say, and the split-half test below measures the
  weights directly. A level that two terms can share (the fixation level and
  the flash × after exit kernels) does not show in these statistics; parameter
  recovery and the leak test find it.

## Are the kernels stable?

Question: can the weights be read, or do correlated regressors make them
change from one set of trials to another? `kernel_reliability.py` fits each
unit on the even and on the odd trials (no shared trial) and compares the
kernels (panel f of `model_trust.svg`). Agreement is
`1 - |a - b|² / (|a|² + |b|²)`: 1 for equal kernels, 0 for unrelated kernels.
Each half has 147 trials in GRB006 and 84 or 85 in GRB064. The values are for
the units that need the regressor (all units in brackets).

| Regressor | Median agreement, GRB006 | GRB064 | Response over units (Spearman), GRB006 | GRB064 |
| --- | ---: | ---: | ---: | ---: |
| Spike history | 0.97 (0.97) | 0.97 (0.97) | 0.97 | 0.96 |
| Flash, 80+ ms gap | 0.90 (0.84) | 0.86 (0.82) | 0.95 | 0.96 |
| Fixation | 0.89 (0.86) | 0.94 (0.92) | 0.94 | 0.97 |
| Flash, 40 ms gap | 0.87 (0.77) | 0.80 (0.68) | 0.91 | 0.91 |
| Response period | 0.84 (0.80) | 0.80 (0.77) | 0.86 | 0.81 |
| Flash × move | 0.78 (0.67) | 0.83 (0.72) | 0.91 | 0.73 |
| Turn direction | 0.69 (0.55) | 0.78 (0.66) | 0.78 | 0.84 |
| Pre-stimulus | one weight | | 0.87 | 0.88 |
| Previous choice | one weight | | 0.83 | 0.91 |

- Every kernel is stable: a fit on half of the trials gives the same kernel
  (median agreement 0.69 or more in the units that need it).
- The fixation kernel is now stable to its end: agreement 0.89 before 700 ms
  and 0.89 after it in GRB006 (0.94 and 0.91 in GRB064). With the kernel to
  0.95 s it was 0.26 from 700 to 950 ms in GRB006, because few trials reach
  that time.
- Turn direction is the least stable kernel (0.69 and 0.78); read it over
  units.
- A half has half of the data, so the final refit is more stable than this
  test shows.

## Spike history and the firing rate

Parameter recovery of the spike-history kernel depends on the rate of the
unit (`model_trust.py`, `HISTORY_MIN_RATE` = 5 spikes/s):

| | Units ≥ 5 spikes/s | Recovery agreement, ≥ 5 / below | Recovered size, ≥ 5 | Split halves, ≥ 5 / below |
| --- | ---: | ---: | ---: | ---: |
| GRB006 | 78 of 167 | 0.99 / 0.24 | 0.91 | 0.99 / 0.95 |
| GRB064 | 50 of 76 | 0.99 / 0.44 | 0.92 | 0.98 / 0.92 |

- In units at 5 spikes/s or more, the history kernel is reliable.
- In slower units, few spikes are close enough together to fit it. Both
  halves give the same kernel, but the penalty makes it small, so a simulated
  kernel comes back too small. Read the history kernel only in the faster
  units. The history term is still needed by 96% of the units for prediction.

## Leak of the fixation level into the flash × after exit kernels

The synthetic check found it first (design of 2026-10-13): a planted drop of
the rate in the fixation (−0.58, no change of the flash response) is
recovered as −0.20 to −0.41 in the fixation kernel, and the flash × after exit
kernel becomes positive (+0.11 to +0.13) with nothing planted. A sum of flash
kernels over a flash train is close to a level, so the fit gives a part of a
level change to the flash × after exit kernels. The usual collinearity statistics
do not show it, and the split-half test cannot show it, because the two
halves have the same bias.

`leak_test.py` measures the leak in the real design of each unit
(`leak_test.json`; panels g to i of `model_trust.svg`): spikes are simulated
from the final refit of the unit with the flash × after exit kernel set to 0, and
the full model is fit to them (3 simulations for each unit).

| Flash × after exit, real minus leak | GRB006 | GRB064 |
| --- | ---: | ---: |
| Matched peak, excited units (spikes/s) | +1.0 (23 units, p = 0.20) | −2.3 (15 units, p = 0.25) |
| Pooled, hierarchical bootstrap | +0.80 [−4.9, 1.6] | |
| Kernel, after exit minus in port, all units (Δ log rate): real | +0.05 | −0.14 |
| Kernel: leak (nothing simulated) | +0.02 (109 of 167 above 0, p = 3e-7) | +0.01 (43 of 76, p = 0.3) |
| Kernel: real minus leak | +0.03 (95 of 167 above 0, p = 0.03) | −0.16 (18 of 76 above 0, p = 2e-5) |
| Fixation level of the refit over the simulated level, median | 0.62 | 0.80 |

- With one flash × after exit kernel the two gap classes have the same kernel
  difference, so the table has one kernel row for both.
- The leak is positive in GRB006 (p = 3e-7) and not reliable in GRB064
  (p = 0.3); it is larger in units with a lower fixation level (Spearman
  −0.16 and −0.11). With the 12-cosine fixation kernel it was +0.03 in both
  (p = 1e-7 and 0.003).
- In the matched peak measure of every flash comparison, the flash × after exit part
  is not more than the leak in either mouse. In the kernel measure of all
  units, the response at the exit is a little higher in GRB006 and lower in
  GRB064: the two animals have opposite signs.
- The fixation kernel keeps about 62% (GRB006) and 80% (GRB064) of a simulated
  level (0.70 and 0.80 in `parameter_recovery.py`, which simulates all terms),
  so the fitted fixation level is too small in size. The rest is in the
  response period, in the flash × after exit kernel, and in the shrinkage of the L2
  penalty.
- So the size of the fixation effect and of the flash × after exit effect cannot be
  read from the weights alone; use the values with the leak taken out, and the
  model-free measures of `hold_effect.py` for the size of the fixation effect.

## Penalty

One L2 penalty for all terms, selected for each unit (the common practice).
The sensitivity analysis with a weaker penalty on the fixation kernel (design
of 2026-10-13) is in [glm_decisions.md](glm_decisions.md), "Penalty".

## Figures

| Figure | Module | Content |
| --- | --- | --- |
| Main | `main_figure.py` | the design of one trial, one trial, the kernels, and the prediction of the example unit (GRB006); the population and what each regressor explains, the units of both mice pooled |
| Flash kernels | `flash_kernels.py` | the flash kernel by gap class and by period (in port, after exit): median kernels of the units that fire more and less after a flash, maps of all units, paired scatters; both sessions |
| Flash by gap | `flash_by_gap.py` | observed and predicted rate around flashes at each gap, in the port and after the exit; both sessions |
| Fixation effect | `hold_effect.py` | two example units (GRB006), the lower rate in the fixation, its onset, its end at the exit, and its controls; both sessions |
| Design choices (supplement) | `design_optimization.py` | the eight design grids on 80 units of GRB006, each with its test of prediction |
| Can the weights be read? (supplement) | `model_trust.py` | collinearity, parameter recovery, split halves, and the leak, for the design in use; both sessions |
| Basis functions (supplement) | `basis_diagram.py` | the basis functions of each kernel |
| Checks of the model (supplement) | `model_checks.py` | single trials, held-out units (and the held-out animal GRB064), terms that are not in the model, calibration (both sessions), recovery of planted kernels |

`model_trust.py` reads the results of `collinearity.py`,
`parameter_recovery.py`, `kernel_reliability.py`, and `leak_test.py`; these
four only fit and write their results.

`main_figure.py`, `hold_effect.py`, `design_optimization.py`,
`model_trust.py`, and `model_checks.py` write their caption next to the figure
(`<figure>_caption.md`), with the numbers of that run. The captions of the
other figures:

- **Flash kernels (`flash_kernels.svg`).** The flash kernel of the model by
  the gap from the flash before and by the period, in GRB006 (a, b, e, g–i)
  and GRB064 (c, d, f, j–l); the units of the two animals are not pooled.
  "In port" is a flash in the fixation and "after exit" is a flash after the
  withdrawal from the center port (task periods; locomotion is not measured).
  **a, c**, The median kernel over the units that fire more after a flash
  (red): left, a flash in the port at a gap of 40 ms and of 80 ms or more;
  right, a flash at a gap of 80 ms or more in the port and in after exit (dashed),
  with the interquartile range. **b, d**, The same for the units that fire
  less after a flash (blue). The kind of a unit is from its mean kernel
  response in the port (above 0.2 or below −0.2 in Δ log rate), so the sign of
  the curves is by selection. **e, f**, The kernel of every unit in the four
  conditions, with one row for each unit and the same order in all maps of an
  animal (by the mean response in the port); the color scale stops at ±1.
  **g–l**, The response (mean of the kernel from 40 to 100 ms) of each unit in
  two conditions; error bars are the SD over the 10 cross-validation folds.
  **g, j**, The 80+ ms gap against the 40 ms gap, in the port. **h, k**, Move
  against still at the 80+ ms gap. **i, l**, Move against still at the 40 ms
  gap. In g to l the color is the kind of the unit from conditions that are
  not in the pair, and gray is a unit of no kind. The model gives a part of
  the fixation effect to the flash × after exit kernel, so the move-against-still
  differences are too high by about 0.02 (GRB006) and 0.01 (GRB064)
  (`leak_test.py`).
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
  (the two gap classes and flash × after exit): 8 raised cosines over 150 ms.
  **b**, Fixation: 10 raised cosines with even spacing in log time over 700
  ms; after the last one the kernel keeps its last value until the center
  exit. **c**, Response period and turn direction: 8 raised cosines with even
  spacing in log time over 1.3 s from the center exit; after the last one the
  kernel keeps its last value until the response entry. **d**, Spike history:
  10 with log spacing over the 100 ms before the bin. **e**, Session drift: 4
  over the session; the dotted one is not a column, because the model has an
  intercept. The pre-stimulus column, the previous choice, and each
  motion-energy PC are one column each.

`glm fit` writes no figure. `glm unique` writes `figure_data.npz` for the
example unit.

## Synthetic check

Run this after you change the design, fit, or unique-deviance code:

```bash
uv run python -m thesis.ephys.analyses.glm.synthetic_check
```

It needs no database. It plants known flash, fixation, response-period,
motion-energy, and history effects in 80 simulated trials with the flash train
of the task. Then it runs `prepare`, `fit`, and `unique` on them and checks that
the effects come back. It compares 2 and 5 motion-energy PCs and takes about 2
minutes. It raises `AssertionError` with the failed checks. The checks of the
size of the fixation kernel report the known bias of the model and do not
stop the command (see "Leak of the fixation level"). Pass `--out DIR` (a new
directory) to keep the artifacts; `synthetic_recovery.json` there has the
planted and recovered kernels, which `model_checks.py` draws.

## Related methods

Same family as [Truccolo et al. 2005](https://pubmed.ncbi.nlm.nih.gov/15356183/)
and [Pillow et al. 2008](https://sites.stat.columbia.edu/liam/research/pubs/pillow-nature-08.pdf),
with a self-history regressor but no coupling between units.
[Talluri et al. 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10620084/)
(video SVD, ridge, trial CV),
[Stringer et al. 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6525101/)
(video as broad behavior). Basis counts and windows remain assumptions.
