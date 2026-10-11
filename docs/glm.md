# V1 GLM

Poisson encoding model for GRB006 session `20240821_121447`. Predict V1 spikes
from six regressors: a common flash kernel, the flash kernel of a flash that
follows another within 50 ms (the recent-flash term), a kernel for the
stimulus during the hold (first flash to center exit), a causal
post-withdrawal kernel (0 to 100 ms), 10 additive camera motion-energy PCs, and a strictly
past self-history filter for each unit. All terms add in log rate. An earlier
full design had four more regressors; [Design justification](#design-justification)
gives the fit that removed them.
Coefficients are conditional associations, not causal effects.

`prepare` reuses saved windows and video features by file presence. It reuses
the saved design when the metadata has the current task regressor names, the
current column count for each regressor, and the current adaptation parameters.
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
and all valid rows for the final refit. Intercept unpenalized. No session-time
drift, go cue, outcome, or punishment kernel.

| Group | Representation | Columns |
| --- | --- | ---: |
| Common flash | Every modeled flash; 8 causal linearly spaced raised cosines, 0–150 ms | 8 |
| Recent flash | Same flash bases, scaled by 1 for a flash that follows another within 50 ms and 0 otherwise, centered | 8 |
| Withdrawal | 4 causal raised cosines, 0–100 ms after center exit | 4 |
| Stimulus during hold | One kernel on the first flash, set to zero from the center exit on; 10 raised cosines with even spacing in log(t + 50 ms), 0–950 ms; no step | 10 |
| Motion energy | One contemporaneous value for each of the first 10 PCs; no temporal basis | 10 |
| Self-history | 10 log-spaced raised cosines at strictly past lags, 1–100 ms | 10 |

No flash × movement interaction, initiation, response-entry, response-side,
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
as `video_trial_specific`.

All run inside the same 10 folds as `fit`, at the penalty that fold selected,
so they are comparable with the cross-validated deviance. Reported as the mean
across folds with its SEM. Following
[Oesch et al. 2026](https://doi.org/10.1038/s41467-026-70639-1) and
[Musall et al. 2019](https://www.nature.com/articles/s41593-019-0502-4).

```bash
uv run glm unique --units all
```

The same command writes `figure_data.npz` and `predicted_spike_trains.svg`
for the example unit in the selected fit directory (`all_fit_me/` or
`sample_fit_me/`). `main_figure.py` draws the kernels, the example trial, the
unique and maximal ΔD², and the population summary from these files.

## Kernel comparison

```bash
uv run python -m thesis.ephys.analyses.glm.kernel_comparison
```

This script makes no new fits. It reads one fit set (`--units`, `all` by
default) in the session output directory and writes `kernel_comparison.svg` and
`kernel_comparison.json`. Panel a compares the size of the visual flash weights
and of the motion-energy weights (the norm of the regressor's coefficients) for
every unit. Panels b and c show the common visual flash kernel and the
motion-energy weight map of three example units: one that the flash predicts,
one that motion energy predicts, and one that both predict. Motion energy has no
temporal kernel, so its panel is a map over the video frame. The module
docstring gives the selection rule.

## Synthetic check

Run this after you change the design, fit, or unique-deviance code:

```bash
uv run python -m thesis.ephys.analyses.glm.synthetic_check
```

It needs no database. It plants known flash, state, adaptation, flash ×
motion-energy, withdrawal, motion-energy, and history effects in 80 simulated
trials. Then it runs
`prepare`, `fit`, and `unique` on them and checks that the effects come back. It
compares 2 and 5 motion-energy PCs instead of 10–200 and takes about 1 minute.
It raises `AssertionError` with the failed checks. Pass `--out DIR` (a new
directory) to keep the artifacts and look at the figures.

## Related methods

Same family as [Truccolo et al. 2005](https://pubmed.ncbi.nlm.nih.gov/15356183/)
and [Pillow et al. 2008](https://sites.stat.columbia.edu/liam/research/pubs/pillow-nature-08.pdf),
with a self-history regressor but no coupling between units.
[Talluri et al. 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10620084/)
(video SVD, ridge, trial CV),
[Stringer et al. 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6525101/)
(video as broad behavior). Basis counts and windows remain assumptions.

## Design justification

The full design had nine regressors: the five above and flash post − pre
withdrawal (6 columns), flash × motion energy (6), initiation (4), and response
entry (6), with 25 motion-energy PCs selected on validation. Its fit of
2026-10-10 (167 units, 294 trials) is in
`figures/glm/GRB006_20240821_121447_full_design/all_fit_me/`, which git does not
track. Do not render that folder again with the current code; the code for that
design is at commit `fcd34fa`.

- Motion-energy PCs: mean validation D² was 0.0733, 0.0738, 0.0736, 0.0723, and
  0.0690 for 10, 25, 50, 100, and 200 PCs. The curve is flat from 10 to 50, so
  the count is now fixed at 10.
- Full model: median cross-validated D² 0.071 (IQR 0.042 to 0.101), median
  0.51 bits/spike.

| Regressor | Columns | Median unique ΔD² | Share of median full-model D² | Units with unique ΔD² > 0 | Kept |
| --- | ---: | ---: | ---: | ---: | --- |
| Spike history | 10 | 0.0253 | 35.9% | 91% | yes |
| Visual flash | 6 | 0.0044 | 6.2% | 80% | yes |
| Withdrawal | 9 | 0.0029 | 4.1% | 75% | yes |
| Motion energy | 25 | 0.0019 | 2.6% | 56% | yes, 10 PCs |
| Flash adaptation | 6 | 0.0016 | 2.3% | 66% | yes |
| Flash post − pre withdrawal | 6 | 0.0009 | 1.2% | 46% | no |
| Flash × motion energy | 6 | 0.0005 | 0.7% | 30% | no |
| Response entry | 6 | 0.0003 | 0.4% | 17% | no |
| Initiation | 4 | 0.0001 | 0.1% | 18% | no |

The fifth column is the fraction of units with a one-sided t-test of the fold
values against zero at p < 0.05; a regressor with no effect gives about 5%. The
folds share training trials, so this test is somewhat too easy to pass.

- Initiation and response entry explain almost nothing.
- The two flash × movement terms are weak: together they add a median ΔD² of
  0.0014 (both shuffled together), about 2% of the full model. The sum of their
  single values is 0.0015, so they do not replace each other. Their joint
  maximal ΔD² is 0.0094, so the other regressors carry about 86% of what the
  two can explain. The simple model has no interaction term: movement is
  additive. A one-column gain test is planned as a separate comparison.
- Motion energy is trial-specific. With the regressor taken from another trial, the
  median loss is 0.0023 (IQR 0.0009 to 0.0046), and 89% of the units lose
  deviance. This is not smaller than the unique ΔD² from the shuffle within a
  trial.
- Lags were not added for motion energy. A check on this fit (removed from the
  code on 2026-10-10) found a correlation of about 0.011 between the spike count
  and PC 1 within ±100 ms, and at most 0.0013 for the residual of the final
  refit with each of PCs 1 to 10.

### Withdrawal and the time in the center port

Tests of 2026-10-10 on the 20-unit sample (medians, 10-fold cross-validated;
the residual tests are in-sample):

| Model | Full D² | Withdrawal ΔD² | Motion energy ΔD² | Motion energy, trial-specific ΔD² |
| --- | ---: | ---: | ---: | ---: |
| Withdrawal −300 to +300 ms, 9 bases | 0.0862 | 0.0065 | 0.0022 | 0.0023 |
| Withdrawal 0 to 100 ms, 4 bases | 0.0805 | 0.0012 | 0.0033 | 0.0023 |
| Withdrawal 0 to 100 ms and time from onset | 0.0918 | 0.0005 | 0.0016 | 0.0028 |

- The wide withdrawal kernel held a slow time course and not a withdrawal
  response. With only the short kernel, motion energy absorbs part of that time
  course: its unique ΔD² is above its trial-specific ΔD².
- The onset kernel (median unique ΔD² 0.0065) is low while the animal holds in
  the center port, about −0.4 in log rate from 200 to 450 ms, and goes back to
  zero after the go cue (487 ± 5 ms after the first flash) and the withdrawal
  (median 553 ms). The label gives the clock, not a cause: a change of the
  visual input in the port, stillness, and a slow visual suppression are not
  separated.
- The residual time course of the model without the onset kernel did not depend
  on flash rate after 300 ms (extra ΔD² 0.0008, shuffled labels 0.0009) and did
  not depend on the response side before withdrawal (3 of 20 units).
- After withdrawal, the residuals depend on the response side in about a third
  of the units, from 100 to 500 ms, with a total ΔD² of about 0.0004. A side
  contrast of the withdrawal kernel (0.0002) and a response-entry kernel were
  not added: the signal is not in 0 to 100 ms and is not locked to entry.
- The suppression is specific to the center port and is not stillness
  (`uv run python -m thesis.ephys.analyses.glm.port_vs_stillness`, raw rates,
  167 units). In the frames with the least motion energy, the rate in the
  center port is 0.78 of the rate at the side port (lower in 68% of units,
  p = 3e-7); in the center port the rate does not go up with motion energy; and
  the units that movement excites are not the suppressed ones (ρ = −0.02). The
  side ports have the same geometry, but the stimulus plays only while the
  animal is in the center port. The stimulus in view and the task state are not
  separated.
- One in-port state column replaces the 8-column onset kernel. On the 20 units
  it has a median unique ΔD² of 0.0025 and a negative weight in 17 units
  (median rate × 0.75). The full D² is 0.0011 lower than with the onset kernel
  (paired median) and 0.0040 higher than with no slow term; motion energy stays
  below its trial-specific ΔD² (0.0016 against 0.0025). With both in the model
  the onset kernel still adds 0.0025, so about 20% of the slow structure follows
  the stimulus clock; it was left out to keep one column with a tested meaning.
- `uv run python -m thesis.ephys.analyses.glm.residual_checks --units all`
  repeats the residual checks of this section (session drift, time in trial,
  flash rate, response side) on a fit set and writes `residual_checks.json` in
  its fit directory; it makes no figure. Use `--root` to run it on another design.
- Session drift is small (median ΔD² 0.0015, shuffled trial order 0.0009), so
  there is no drift term.
- Flash kernel (`flash_kernel_grid`, 20 units, validation D²): 8 bases against
  6 gives +1.6% (16 of 20 units, p = 0.0004); 10 against 8 gives nothing; a
  250 ms window against 150 ms gives +0.8% (9 of 20, p = 0.84). The recent-flash
  term adds 5.4% (16 of 20, p = 0.008), also with a 250 or 350 ms kernel.
- Recent-flash term (`adaptation_grid`): a synaptic-depression resource in its
  place predicts worse with a slow recovery, and its validation D² goes up to
  the level of the recent-flash term as the recovery gets faster than the
  shortest flash interval (40 ms). The best depression pair is +0.2% (10 of 20
  units, p = 0.60). So the data give no support for a slow depletion over the
  train, and the model uses the simpler term.
- `uv run python -m thesis.ephys.analyses.glm.design_optimization` draws these
  tests in one figure (`design_optimization.svg`) and writes its caption.
- The suppression follows the time in the port, not the clock (raw rates, all
  units). In the 40 completed trials with a withdrawal 150 ms or more after the
  go cue, the rate from 50 to 150 ms after the cue is the hold rate (median
  ratio 1.00), and 0.55 of the rate of the 103 fast trials in the same window.
  In the 9 early-withdrawal trials with a hold of 150 to 280 ms, the rate after
  the exit is 1.87 times the rate of completed trials at the same time from
  onset. So the go cue does not end the suppression; the exit does.

### Stimulus during hold

Tests of 2026-10-10 and 2026-10-11 on the 20-unit sample. The in-port column
of the section above became one kernel, from the first flash to the center
exit (`hold` in `dmat.py`).

- Why: with the in-port column, the model reached 50% of the first-flash peak
  of the units with a clear peak, and observed / predicted was 1.34 just before
  the first flash and 0.75 at 150 ms. The drop starts about 80 ms after the
  first flash in trials with a short and with a long delay from center entry,
  so it follows the first flash, not the entry. It ends at the center exit,
  while the flashes go on to the response entry. So the term is on only while
  the stimulus is on and the animal holds: "stimulus during hold".
- Form (`hold_grid`, validation D², against the kernel in use): no term
  −0.0065 (median; p = 4e-5), step from center entry −0.0021 (p = 0.0006), step
  from the first flash −0.0010 (p = 0.0004). A step from the first flash with a
  150 ms kernel on the first flash is equal (p = 0.76), and so is the kernel
  with linear spacing (p = 0.73; in 10-fold cross-validation the log spacing is
  0.6% better, 14 of 20 units, p = 0.033). The kernel was selected because it
  is one regressor and its flat level after the drop is then a result (median
  gain 0.6 from 300 to 550 ms; below 0.9 in 17 of 20 units, above 1.1 in 1).
  It reaches 88% of the first-flash peak.
- Other forms that were tested and removed from the code: a step with a delay
  of 50 to 150 ms (best at 80 ms, +2.9% against the in-port column, but with no
  meaning of its own); a flash kernel of 300 to 800 ms with no hold term (2 to
  5% worse); a flash-recovery term, 1 − exp(−gap / τ) or log(gap), with or
  without the recent-flash term (0 to +0.5% alone).
- The window (950 ms) must be longer than the longest hold, because no step
  goes with the kernel; `build_task_design` stops if it is not (longest hold
  here: 941 ms). After about 650 ms fewer than 10% of the trials give data.
- The suppression ends at the center exit, so it is not adaptation to the
  flashes alone (20 units, tests removed from the code). A kernel that goes on
  to the response entry predicts worse (−1.6%, better in 4 of 20 units,
  p = 0.008). With a column that is 1 from the center exit to the response
  entry, the gain relative to the time before the first flash is 0.70 in the
  hold (150 to 500 ms) and 1.17 after the exit (higher after the exit in 18 of
  20 units, p = 0.001); that column does not reliably improve D² (+1.4%,
  p = 0.23) and is not in the model.
- `uv run python -m thesis.ephys.analyses.glm.hold_effect --units all` draws
  the results figure for this effect (`hold_effect.svg`) on all 167 units. It
  fits a descriptive model with a kernel from the center exit to the response
  entry in place of the withdrawal kernel. The raw rate is 0.64 of the unit
  mean in the last 200 ms of the hold and 1.44 from 100 to 400 ms after the
  exit (1.11 before the first flash). The model gain is 0.73 in the hold (below
  1 in 90% of the units) and 1.20 after the exit (below 1 in 19%); it is higher
  after the exit in 144 of 167 units (p = 5e-22). On the 20 units, that longer
  kernel does not predict reliably better than the 0 to 100 ms withdrawal
  kernel (+1.0%, p = 0.20), so the main design keeps the withdrawal kernel.
- The kernel multiplies the rate (it adds to the log rate). It has no
  interaction with the flash kernel. In raw rates, the ratio of the flash
  response to the rate before the flash is the same in the hold and after the
  exit (1.27 and 1.26 for flashes 40 ms after another; 2.4 and 2.2 after a gap
  of 240 ms or more; population means, 20 units).
- Two residuals of this design, both left out of the model:
  - A response to the center entry. In the 96 trials with the first flash
    66 ms or more after the entry, the rate goes up 20 to 30 ms after the entry
    (observed 1.23 to 1.31, model 1.03 to 1.09). It is large in 7 of 20 units
    and not reliable over units (median observed / model 1.04, p = 0.28). A
    causal entry kernel (0 to 100 ms) gives +0.5% (p = 0.81). The first 40 ms
    of the hold kernel take part of it. Open question: it is not known if a
    light or a sound changes at the center poke.
  - The response to a flash after a pause has three phases. After the first
    flash, observed / model is 0.93 at 130 to 210 ms (p = 0.008), 1.11 at 230
    to 290 ms (p = 0.017), and 0.91 at 310 to 450 ms (p = 0.0001); the rebound
    is also there in the 44 trials with no flash from 100 to 260 ms. A later
    flash after a gap of 240 ms or more has a larger response than the model
    (1.35 against 1.09, p = 0.044) and a deeper dip at 100 to 160 ms (0.59
    against 0.82, p < 0.001). A flash kernel of 200 to 300 ms (+0.6 to +0.8%,
    p > 0.3) and a recovery term (+0.3 to +0.6%, p = 0.2) do not predict
    reliably better, and with the 300 ms kernel the dip stays.

### Design optimization figure

`design_optimization.svg` is the one figure for these choices. Each of its
tests is the final design with one thing changed, scored as D² on the
validation trials against the constant-rate model, on the 20-unit sample. Where
two options do not differ reliably (paired Wilcoxon signed-rank test over
units), the model uses the simpler one. Its caption and the exact test values
are written next to it.

```bash
uv run python -m thesis.ephys.analyses.glm.flash_kernel_grid
uv run python -m thesis.ephys.analyses.glm.adaptation_grid
uv run python -m thesis.ephys.analyses.glm.hold_grid
uv run python -m thesis.ephys.analyses.glm.withdrawal_grid
uv run python -m thesis.ephys.analyses.glm.pc_grid
uv run python -m thesis.ephys.analyses.glm.port_vs_stillness
uv run python -m thesis.ephys.analyses.glm.residual_checks --units sample
uv run python -m thesis.ephys.analyses.glm.design_optimization --units sample
```

The first five modules fit and score designs and make no figure. Results of
2026-10-11, with the hold kernel in the design:

- Flash kernel: 6 bases are worse than 8 (p = 0.0003); 10 and 12 are not
  better (p = 0.67 and 0.19). 100 ms is worse than 150 ms (p = 0.005); 200 and
  250 ms are not better (p = 0.37 and 0.93).
- Recent-flash term: with the hold kernel, the model without the term is no
  longer reliably worse (median −0.0012, mean −0.0033, lower in 13 of 20 units,
  p = 0.07; it was p = 0.008 with the in-port column). The term stays in the
  design for now; this is an open decision. A depression resource with a fast
  recovery is equal to it (p = 0.60).
- Withdrawal: no kernel is worse than the 0 to 100 ms kernel (p = 0.001); the
  ±100 and ±300 ms kernels are not better (p = 0.35 and 0.50).
- Motion-energy PCs: no motion energy is worse than 10 PCs (p = 0.006); 25 and
  50 are not better (p = 0.60 and 0.50); 200 is worse (p = 0.003).

The fit artifacts of the design with the in-port column are in
`figures/glm/GRB006_20240821_121447_in_port_step/`.

## Results

Full fit of the final design, 2026-10-11: 167 units, 294 trials, 10
motion-energy PCs. Artifacts are in
`figures/glm/GRB006_20240821_121447/all_fit_me/`, which git does not track.

- Full model: median cross-validated D² 0.071, median 0.54 bits/spike.
- Against the design with the in-port column (0.068, 0.50 bits/spike), D² is
  0.0020 higher per unit (paired median; higher in 136 of 167 units,
  p = 2e-18).
- Example unit 197: the model reaches 77% of the first-flash peak of the
  observed rate (52% with the in-port column). Its predicted rate before the
  first flash is too high (46 against 33 spikes/s).
- Hold kernel: the median gain from 150 to 500 ms is 0.61, and it is below 1 in
  89% of the units.

| Regressor | Columns | Median unique ΔD² | Units with unique ΔD² > 0 |
| --- | ---: | ---: | ---: |
| Spike history | 10 | 0.0266 | 92% |
| Stimulus during hold | 10 | 0.0065 | 88% |
| Visual flash | 8 | 0.0052 | 84% |
| Motion energy | 10 | 0.0019 | 65% |
| Withdrawal | 4 | 0.0016 | 61% |
| Flash adaptation (recent flash) | 8 | 0.0015 | 57% |

- Motion energy is trial-specific: with the regressor taken from another trial,
  the median loss is 0.0021 (IQR 0.0008 to 0.0041), the same as with the
  shuffle within a trial (p = 0.66).
- Held-out units (`heldout_units.py`): the 147 units that took no part in the
  design choices have a median D² of 0.070, against 0.094 for the 20 selection
  units (Mann-Whitney p = 0.45); no unique ΔD² differs between the two groups
  (p from 0.39 to 0.97).
- Residual checks on all units (in-sample ΔD² of a term that is not in the
  model, and with shuffled labels): session drift 0.0013 (0.0007), response
  side after withdrawal 0.0013 (0.0003), time in trial 0.0009 (0.0007; it was
  0.0025 with the in-port column), flash rate 0.0009 (0.0006), response side
  before withdrawal 0.0002 (0.0002).
- `uv run python -m thesis.ephys.analyses.glm.main_figure` draws the main
  figure (`main_figure.svg`) for the example unit and all units.
