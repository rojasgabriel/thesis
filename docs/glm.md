# V1 GLM

Poisson encoding model for GRB006 session `20240821_121447`. Predict V1 spikes
from seven regressors: a common flash kernel, the flash kernel of a flash that
follows another within 50 ms (the recent-flash term), a column for the
initiation (center entry to first flash), a kernel for the
stimulus during the fixation (first flash to center exit), a causal
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
| Flash × fixation | Same flash bases, on the flashes after the center exit | 8 |
| Withdrawal | 4 causal raised cosines, 0–100 ms after center exit | 4 |
| Initiation | 1 from center entry to the first flash, else 0; no temporal basis | 1 |
| Fixation | One kernel on the first flash, set to zero from the center exit on; 12 raised cosines with even spacing in log(t + 25 ms), 0–950 ms; no step | 12 |
| Choice (key `side_after_exit`) | +1 or −1 for the side of the response, from the center exit on | 1 |
| Motion energy | One contemporaneous value for each of the first 10 PCs; no temporal basis | 10 |
| Session drift | 4 raised cosines over the session, with the last one left out | 3 |
| Self-history | 10 log-spaced raised cosines at strictly past lags, 1–100 ms | 10 |

65 columns. `basis_diagram.py` draws the basis functions
(`figures/glm/basis_functions.svg`).

Design changes on 2026-10-12, not fitted on all units yet: 2 gap classes in
place of 3 (see "gap classes"), and no center-entry kernel (9% of the units
needed it; the initiation column and the fixation kernel cover
the same time).

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

It needs no database. It plants known flash, adaptation, fixation, withdrawal,
motion-energy, and history effects in 80 simulated trials. Then it runs
`prepare`, `fit`, and `unique` on them and checks that the effects come back. It
compares 2 and 5 motion-energy PCs instead of 10–200 and takes about 1 minute.
It raises `AssertionError` with the failed checks. Pass `--out DIR` (a new
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
  side ports have the same geometry, but the stimulus is off when the animal
  is at a side port (the train stops at the response entry). The stimulus in view and the task state are not
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
  go cue, the rate from 50 to 150 ms after the cue is the fixation rate (median
  ratio 1.00), and 0.55 of the rate of the 103 fast trials in the same window.
  In the 9 early-withdrawal trials with a fixation of 150 to 280 ms, the rate after
  the exit is 1.87 times the rate of completed trials at the same time from
  onset. So the go cue does not end the suppression; the exit does.

### Fixation

Tests of 2026-10-10 and 2026-10-11 on the 20-unit sample, with the first form
of the kernel (10 bases, 50 ms stretch); the next section has the final form. The in-port column
of the section above became one kernel, from the first flash to the center
exit (`hold` in `dmat.py`).

- Why: with the in-port column, the model reached 50% of the first-flash peak
  of the units with a clear peak, and observed / predicted was 1.34 just before
  the first flash and 0.75 at 150 ms. The drop starts about 80 ms after the
  first flash in trials with a short and with a long delay from center entry,
  so it follows the first flash, not the entry. It ends at the center exit,
  while the flashes go on (the train is 1 s long and stops early only at the
  response entry). So the term is on only while
  the stimulus is on and the animal holds: "fixation".
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
  meaning of its own); a flash kernel of 300 to 800 ms with no fixation term (2 to
  5% worse); a flash-recovery term, 1 − exp(−gap / τ) or log(gap), with or
  without the recent-flash term (0 to +0.5% alone).
- The window is 950 ms, and no step goes with the kernel. The longest
  fixation of GRB006 is 941 ms. In a trial with a longer fixation, the kernel
  keeps its last value up to the center exit (3 of 169 trials in GRB064, up to
  1,166 ms). After about 650 ms fewer than 10% of the trials give data.
- The suppression ends at the center exit, so it is not adaptation to the
  flashes alone (20 units, tests removed from the code). A kernel that goes on
  to the response entry predicts worse (−1.6%, better in 4 of 20 units,
  p = 0.008). With a column that is 1 from the center exit to the response
  entry, the gain relative to the time before the first flash is 0.70 in the
  fixation (150 to 500 ms) and 1.17 after the exit (higher after the exit in 18 of
  20 units, p = 0.001); that column does not reliably improve D² (+1.4%,
  p = 0.23) and is not in the model.
- `uv run python -m thesis.ephys.analyses.glm.hold_effect --units all` draws
  the results figure for this effect (`hold_effect.svg`) on all 167 units. It
  fits a descriptive model with a kernel from the center exit to the response
  entry in place of the withdrawal kernel. The raw rate is 0.64 of the unit
  mean in the last 200 ms of the fixation and 1.44 from 100 to 400 ms after the
  exit (1.11 before the first flash). The model gain is 0.73 in the fixation (below
  1 in 90% of the units) and 1.20 after the exit (below 1 in 19%); it is higher
  after the exit in 144 of 167 units (p = 5e-22). On the 20 units, that longer
  kernel does not predict reliably better than the 0 to 100 ms withdrawal
  kernel (+1.0%, p = 0.20), so the main design keeps the withdrawal kernel.
- The kernel multiplies the rate (it adds to the log rate). It has no
  interaction with the flash kernel. In raw rates, the ratio of the flash
  response to the rate before the flash is the same in the fixation and after the
  exit (1.27 and 1.26 for flashes 40 ms after another; 2.4 and 2.2 after a gap
  of 240 ms or more; population means, 20 units).
- Two residuals of this design, both left out of the model:
  - A response to the center entry. In the 96 trials with the first flash
    66 ms or more after the entry, the rate goes up 20 to 30 ms after the entry
    (observed 1.23 to 1.31, model 1.03 to 1.09). It is large in 7 of 20 units
    and not reliable over units (median observed / model 1.04, p = 0.28). A
    causal entry kernel (0 to 100 ms) gives +0.5% (p = 0.81). The first 40 ms
    of the fixation kernel take part of it. No sound plays at the center poke (the
    go cue, when the wait time is reached, is the only task sound of a
    completed trial), so touch and the head movement into the port are the
    candidates.
  - The response to a flash after a pause has three phases. After the first
    flash, observed / model is 0.93 at 130 to 210 ms (p = 0.008), 1.11 at 230
    to 290 ms (p = 0.017), and 0.91 at 310 to 450 ms (p = 0.0001); the rebound
    is also there in the 44 trials with no flash from 100 to 260 ms. A later
    flash after a gap of 240 ms or more has a larger response than the model
    (1.35 against 1.09, p = 0.044) and a deeper dip at 100 to 160 ms (0.59
    against 0.82, p < 0.001). A flash kernel of 200 to 300 ms (+0.6 to +0.8%,
    p > 0.3) and a recovery term (+0.3 to +0.6%, p = 0.2) do not predict
    reliably better, and with the 300 ms kernel the dip stays.

### Flash × fixation interaction

Test of 2026-10-11 (`flash_fixation_test.py`, results in
`flash_fixation_test.json`). The fixation kernel adds one value to the log
rate, so it lowers the baseline and the flash response by the same factor. The
test adds a second flash kernel for a flash after the center exit against a
flash in the fixation (`FLASH_AFTER_EXIT_TERM` in `dmat.py`, off in the model).
The first flash of a trial is not in the contrast. The value of a unit is the
mean of that kernel from 30 to 120 ms (Δ log rate, after the exit minus in the
fixation).

- 20 selection units: the term does not predict reliably better (validation D²
  +0.0003, better in 11 of 20 units, p = 1.0; value +0.01, p = 0.23). By the
  rule of the simpler option it is not in the model.
- All 167 units: the term predicts a little better. Cross-validated D² +0.0004
  (0.6% of the D² of the model; Wilcoxon p = 2e-8; validation D² better in 104
  of 167 units, p = 0.004). With the same 10 folds, 28% of the units predict
  reliably better with the term and 4% worse (a term with no effect gives 5%;
  initiation 23%, fast adaptation 60%). The 147 held-out units alone give the
  same result (29%, p = 3e-8).
- The value has a median of +0.06 (above zero in 106 of 167 units, p = 3e-5):
  the flash response after the exit is about 6% higher than the common factor
  gives. The fixation kernel is −0.31, so most of the fixation effect is the
  common factor.
- The 46 units that need the term are of two kinds. 19 have a positive kernel
  with a peak at 45 ms (median +0.67): a larger early response after the exit.
  27 have a negative kernel with a minimum near 90 ms (median −0.33): a smaller
  late response after the exit. This looks like an earlier response after the
  exit, but the observed responses do not show it: the peak latency is the
  same in the fixation and after the exit (72 and 72 ms, 112 units, p = 0.44;
  see "What causes the fixation effect"). So these are two groups of units,
  not one change in latency.
- So the fixation is mostly a gain change of the unit, with a small part that
  is specific to the flash response in about a quarter of the units. This part
  can be slow adaptation in the fixation or an effect of the movement after
  the exit; the test does not separate them.
- The old design had this contrast with no fixation kernel (unique ΔD² 0.0009,
  46% of units, not kept).

### The rule for a regressor

From 2026-10-12 (`inclusion.py`). The importance of a regressor is tested on
all units and on held-out trials, because 20 units are too few (the side term
and the flash × fixation term were not reliable on 20 units and are reliable
on 167). The form of a term (bases, window, spacing) is still chosen on the 20
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

The tests say that a regressor is real for the population. They do not say
that it is worth its columns; that is a choice of parsimony. The tests are in
`term_test.py` (one switch, the same folds), `candidate_terms.py` (all
regressors of a design, by the shuffle of each), and `main_figure_stats.json`
(the regressors of the model; all seven pass both tests).

### Candidate terms, all together

Test of 2026-10-12 (`candidate_terms.py`, all 167 units, results in
`candidate_terms.json`). One design: the 7 regressors of the model plus four
candidates (side after the exit, flash × fixation, flash recovery, and a
center-entry kernel), with the unique-deviance step. So each regressor is
tested with all the others in the design. Holm correction for 11 regressors;
the smallest count above chance is 17 units.

| Regressor | Columns | Units that need it | Held-out units | Units worse | Median unique ΔD² |
| --- | ---: | ---: | ---: | ---: | ---: |
| Spike history | 10 | 90% | 90% | 2.4% | 0.0257 |
| Visual flash | 8 | 78% | 77% | 0.0% | 0.0037 |
| Fixation | 12 | 71% | 72% | 0.0% | 0.0027 |
| Motion energy | 10 | 56% | 58% | 0.0% | 0.0015 |
| Withdrawal | 4 | 54% | 54% | 0.0% | 0.0008 |
| Flash × fixation (candidate) | 8 | 41% | 44% | 0.0% | 0.0009 |
| Fast adaptation | 8 | 40% | 37% | 1.2% | 0.0005 |
| Flash recovery (candidate) | 8 | 38% | 39% | 0.6% | 0.0006 |
| Side after the exit (candidate) | 1 | 32% | 33% | 1.8% | 0.0003 |
| Initiation | 1 | 13% | 12% | 1.8% | 0.0001 |
| Center entry (candidate) | 4 | 11% | 12% | 0.6% | 0.0001 |

- Every regressor passes both tests of the rule (largest Holm p: 0.0007, for
  the unit count of the center-entry kernel, 19 of 167 units).
- The median cross-validated D² goes from 0.0716 to 0.0744 (+3.9%).
- The regressors of the model are needed by fewer units than before (fast
  adaptation 60% to 40%, fixation kernel 87% to 71%, initiation 23% to 13%),
  because the candidates explain a part of what they explained. The unique
  ΔD² of a regressor is what no other regressor can explain.
- The flash × fixation term is needed by more units here (41%) than in its
  own test (28%).

### Side of the response after the exit

Test of 2026-10-12 (`side_term_test.py`, 20 selection units, results in
`side_term_test.json`). About half of the units fire more for a response to one
side after the center exit, with or without the stimulus (`side_preference.py`),
and the model has no term for the side. The test adds one column, +1 for a
response to the right port and −1 for the left port from the center exit on
(`SIDE_COLUMN` in `dmat.py`, off in the model).

- The term does not predict reliably better on the 20 units: validation D²
  +0.0001 (+0.2%), better in 11 of 20 units, Wilcoxon p = 0.25. By the rule of
  the simpler option it is not in the model.
- With the same 10 folds, 30% of the units predict reliably better with the
  term and 5% worse (cross-validated D² +0.0005, p = 0.09).
- The weights are not small: the median size is 0.28 in log rate (right
  against left, a factor of 1.3), with no common sign (11 of 20 above zero).
- All 167 units: the term predicts reliably better. Cross-validated D²
  +0.0005 in the median (0.6% of the D² of the model) and +0.0013 in the mean
  (1.8%); Wilcoxon p = 4e-14; validation D² better in 115 of 167 units
  (p = 1e-7). With the same 10 folds, 31% of the units predict reliably better
  with the term and 0.6% worse. The 147 held-out units alone give the same
  result (31%, p = 1e-13).
- On all units the weight has a common sign: the rate is higher for a response
  to the right in 120 of 167 units (median +0.16 in log rate). The 52 units
  that need the term have a median size of 0.49 (a factor of 1.6); 38 prefer
  the right and 14 the left.
- 16 units need this term and the flash × fixation term; 82 of 167 units need
  one or the other.
- Rule from 2026-10-12 (Gabriel): the importance of a regressor is tested on
  all units, because 20 units are too few (this term and the flash × fixation
  term were not reliable on 20 units and are reliable on 167). The form of a
  term (bases, window, spacing) is still chosen on the 20 selection units. By
  this rule the fast-adaptation term is no exception (60% of units), and this
  term and the flash × fixation term have a case to be in the model. Not
  decided.

### Previous choice, and the design with 11 regressors

Question: does the side from which the animal comes to the center port change
the rate? The side is the side of the last response before the trial
(`trials.last_side`: 205 trials from the left, 88 from the right; the same side
as the response of the trial in 53%).

- Raw rates, no model (a quick test, not a module): the rate differs between
  the two sides (Mann-Whitney, p < 0.05) in 58% of the units in the 300 ms
  before the center entry, 35% from the entry to the first flash, 60% in the
  150 ms after the first flash, and 34% from 200 to 450 ms. With the response
  3 or 5 trials back, or of the next trial, it is 1 to 9%, so it is not drift.
- Term: one column, +1 (right) or −1 (left) from the start of the window to the
  center exit (`PREVIOUS_SIDE_COLUMN`). `unique_dev.py` shuffles it over trials.
- Test on all 167 units (`candidate_terms.py`, `candidate_terms_previous_side.json`),
  with the design of 2026-10-12 (2 gap classes, no center-entry kernel, 66
  columns). Every regressor passes both tests of the rule (Holm p < 0.001).

| Regressor | Columns | Units that need it | Units worse | Median unique ΔD² |
| --- | ---: | ---: | ---: | ---: |
| Spike history | 10 | 92% | 2.4% | 0.0261 |
| Fixation | 12 | 77% | 0.0% | 0.0039 |
| Flash, 80+ ms gap | 8 | 75% | 0.0% | 0.0034 |
| Flash, 40 ms gap | 8 | 75% | 0.0% | 0.0028 |
| Motion energy | 10 | 57% | 0.0% | 0.0014 |
| Withdrawal | 4 | 53% | 0.0% | 0.0008 |
| Flash × fixation | 8 | 42% | 1.8% | 0.0008 |
| Choice | 1 | 32% | 1.2% | 0.0003 |
| Session drift | 3 | 28% | 2.4% | 0.0004 |
| Previous choice | 1 | 20% | 0.6% | 0.0002 |
| Initiation | 1 | 18% | 1.8% | 0.0001 |

- Median cross-validated D² 0.0734 (IQR 0.046 to 0.104). Against the design
  with 3 gap classes and the center-entry kernel (77 columns), same folds:
  median change +0.0002 (+0.3%), 105 of 167 units better, p = 0.0003.
- Collinearity of this design (task columns of the common design, z-scored;
  a quick test, not a module): the variance inflation factor of a column is at
  most 3.6 (2.9 for the flash kernel of the 40 ms gap; the usual limit is 5 to
  10), and the condition number of the correlation matrix is 28. The largest
  correlation between columns of two regressors is 0.40 (flash, 80+ ms gap,
  and fixation: both start at the first flash) and 0.38 (the two flash
  kernels). For two regressors the unique ΔD² is above the maximal ΔD² in most
  units (flash, 40 ms gap: 102 of 167; previous choice: 94 of 167). The cause
  is not strong collinearity: alone, with all other terms shuffled, the term
  must fit a response that is on top of the other responses.
- The previous side is in the model, but the column holds less than the raw
  rates show (20% of units against about half). One offset for the full time
  before the exit cannot follow an effect that is larger before the entry and
  in the first flash response. See the test below.
- Flash kernel by previous side (`FLASH_PREVIOUS_SIDE_TERM`, 8 columns: the
  flash bases scaled by the centered previous side, for the flashes in the
  fixation; `candidate_terms_flash_previous_side.json`, all 167 units, the 11
  regressors plus this term). The term passes both tests of the rule: 24 units
  need it (14%; chance gives 18 or fewer for 12 regressors), 1.2% are worse,
  median unique ΔD² 0.00015. But the model does not predict better with it:
  cross-validated D² against the design with 11 regressors, same folds, median
  change −0.0001 (−0.1%), 67 of 167 units better, p = 0.09. The previous-side
  column goes from 20% to 12% of the units with the term in the design, so the
  two terms hold part of the same effect. The term is off in the design.

### Flash kernels by gap class

Tests of 2026-10-12 (`gap_class_grid.py`, 20 selection units). The response to
a flash depends on the gap from the flash before. The first form of this was
two terms on a common flash kernel: fast adaptation (a flash within 50 ms) and
flash recovery (the log of the gap). They are correlated (−0.86 over 3,208
flashes), so the fit can move weight from one to the other and their kernels
cannot be read one by one. The flash × fixation term is not correlated with
them (−0.01 and 0.03). The design now has one flash kernel for each gap class
and no common kernel (`FLASH_GAP_EDGES_S` in `dmat.py`); the first flash of a
trial is in the class of the longest gaps. Each kernel is the response at that
gap, and the classes do not overlap.

| Design | Median validation D² | Against 3 classes (40, 80 to 120, 160+) |
| --- | ---: | --- |
| 1 kernel, no gap term | 0.0825 | −1.9%, p = 0.11 |
| Common kernel, fast adaptation, flash recovery | 0.0837 | −0.0%, p = 0.65 |
| 2 classes: 40, 80+ | 0.0828 | −0.2%, p = 0.78 |
| 3 classes: 40, 80 to 120, 160+ | 0.0839 | the design in use |
| 3 classes: 40, 80, 120+ | 0.0831 | +0.1%, p = 0.73 |
| 4 classes: 40, 80, 120, 160+ | 0.0842 | −0.2%, p = 0.33 |

- The classes lose nothing against the earlier terms (p = 0.65).
- 2 classes are better than 1 kernel (14 of 20 units, p = 0.024). 3 classes
  are not reliably better than 2 on 20 units (p = 0.78), and 4 not better
  than 3.
- The design has 3 classes, because the observed responses differ between
  80, 120, 160, and 200 ms (`flash_by_gap.py`), and 20 units are too few to
  show the need for a regressor. The third class is tested on all units with
  the same folds (`term_test.py --switch FLASH_GAP_EDGES_S --value "(0.05,)"`).

### Gap classes on 80 units

From 2026-10-12 the form of a term is chosen on 4 sets of 20 units (80 units,
no unit in two sets; `--samples 4` in the grid modules), which leaves 87 units
held out. One set of 20 had too little power. The gap-class grid on 80 units:

| Comparison | Median change in validation D² | Units better | p |
| --- | ---: | ---: | ---: |
| 2 classes (40, 80+) against 1 kernel | +1.46% | 59 of 80 | 1e-6 |
| 3 classes (40, 80 to 120, 160+) against 2 | −0.17% | 33 of 80 | 0.054 |
| 3 classes (40, 80, 120+) against 2 | −0.36% | 31 of 80 | 0.19 |
| 4 classes against 2 | −0.44% | 30 of 80 | 0.022 |
| Common kernel with the two gap terms against 2 classes | −0.06% | 38 of 80 | 1.0 |

- A step at 40 ms is necessary. More classes are not better, and 4 classes
  are reliably worse than 2.
- 2 classes are equal to the earlier gap terms, with 8 columns in place of 24.
- The test of 3 classes against 2 on all 167 units with the same folds gives
  the same answer (`term_test_flash_gap_edges_s_0_05.json`: median change
  −0.00002, 6.6% of units better, 7.2% worse, p = 0.49).
- So the form that the rule gives is 2 flash kernels: a flash 40 ms after
  another, and every other flash. The design has this form from 2026-10-12
  (`FLASH_GAP_EDGES_S = (0.05,)`).

### Length of the flash kernels, with kernels by gap class

Test of 2026-10-12 (`flash_length_grid.py`, 20 selection units). The observed
response to a flash is about 200 ms long (`flash_by_gap.py`), and the flash
kernels are 150 ms long. With one kernel for each gap class, a longer kernel
could hold the dip after a flash at a long gap. The raised cosines keep their
spacing, so a longer kernel has more of them.

| Length | Raised cosines | Median validation D² | Against 150 ms |
| --- | ---: | ---: | --- |
| 150 ms | 8 | 0.0839 | the design in use |
| 200 ms | 11 | 0.0859 | −0.1%, better in 10 of 20 units, p = 0.96 |
| 250 ms | 13 | 0.0847 | −0.3%, better in 8 of 20 units, p = 0.70 |
| 300 ms | 16 | 0.0866 | −0.5%, better in 9 of 20 units, p = 0.41 |

A longer kernel does not predict better (Friedman p = 0.90), so the kernels
stay at 150 ms. The median of each design is higher for the longer kernels, but
the paired differences are not: the units that gain and the units that lose
are about equal in number. So the dip from 150 to 190 ms after a flash at a
long gap is in the data and not in the model, and it is too small a part of
the bins to change the prediction.

### Session drift

Tests of 2026-10-12. A term for a slow change of the rate over the session
(`SESSION_DRIFT` in `dmat.py`, off in the model): smooth bumps over the time
of the trial in the session, constant inside a trial. It is not a variable of
the task; its use is to keep a slow change out of the other kernels. The
within-trial shuffle of the unique-deviance step cannot test it, so
`term_test.py` compares the design with and without it on the same folds.

- All 167 units (`term_test_session_drift.json`, cubic spline with 7 columns):
  cross-validated D² +0.0002 in the median (0.2% of the model) and +0.0009 in
  the mean (1.2%), Wilcoxon p = 0.0009. 13% of the units predict reliably
  better with the term and 5% worse; validation D² is not better (84 of 167
  units, p = 0.33). The gain comes from few units: 11 gain more than 0.005.
- Form (`drift_grid.py`, 20 selection units): 4, 8, and 12 raised cosines and
  the cubic spline do not differ reliably (8 against 4, p = 0.96; 12 against
  8, p = 0.78; spline against 8, p = 0.19). So the form is the simplest one:
  4 raised cosines (3 columns; the last one is left out because the model has
  an intercept). Raised cosines, and not the spline, so that the model has one
  kind of basis.

### What causes the fixation effect

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
| Stillness in the port | The same low rate at a side port with the same motion energy | No: 0.78 of the rate at a side port |
| A change of the gain of the whole unit (state) | The same factor for all units and for baseline and response | Not fully: units with a larger flash response are suppressed more (Spearman ρ = −0.41, p = 5e-8; −0.36 with rate and D² held), and the response changes more than the baseline for flashes after a long gap (+0.18 in log, p = 0.0003; 0.00 for short gaps) |
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
- The most probable explanation with these data: the rate is low while the
  animal holds in the center port with the stimulus on, and it goes up when
  the animal leaves, with or without the stimulus. The suppression starts
  about 80 ms after the first flash and not at the center entry. It is larger
  in units with a larger flash response. It does not follow motion energy
  after the exit, and it is not stillness. So it is a state of the trial
  (holding in the port during the stimulus) that lowers the rate of units
  with a visual response, and that ends with the exit. Slow adaptation, the
  view of the stimulus, and a gain that grows with movement are less
  probable. Not known: what starts it at the first flash, and what the side
  preference of single units after the exit is.

### Fixation kernel: spacing, resolution, and the initiation column

Tests of 2026-10-11. They changed the fixation kernel from 10 bases with a 50 ms
stretch to 12 bases with a 25 ms stretch, and added the initiation column.

- Log against linear spacing. The reason for log spacing is in the task, not
  in the fit: the rate changes in the first 100 ms and is almost flat after
  that; every trial has the first 490 ms of the fixation but only 5% have 708 ms;
  and the kernel is locked to the first flash, so only early times are aligned
  over trials. A full fit with 10 linear bases had the same population D²
  (0.070 against 0.071), but for unit 197 it reached about 60% of the
  first-flash peak and its kernel was uneven after 500 ms. With the final
  bases, linear spacing is reliably worse (p = 0.017).
- Start at the first flash against the center entry. A kernel from the center
  entry is equal in D² (p = 0.45 to 0.73) and it does stay near zero until the
  flash, but it blurs the drop: the flash comes 24 to 114 ms after the entry,
  so the model predicts too much rate from 80 to 160 ms after the flash
  (observed / predicted 0.79 against 0.93). The kernel stays on the first
  flash.
- Resolution. The model missed the first-flash peak (62% for unit 197) because
  the log bases near it were at 47, 86, and 139 ms. In single-unit refits of
  unit 197 and the 6 other units with the clearest peak, 14 bases, 12 bases
  with a 25 ms stretch, and a separate first-flash kernel all reached about
  80% in 10-ms bins. 12 bases with a 25 ms stretch (bases at 0, 10, 24, 43,
  70, 107, 159 ms, then wider) adds the fewest columns.
- Initiation column: 1 from the center entry to the first flash, where no other
  term is on. It makes the rate in that time correct for single units (more
  than 10% off in 15 of 20 units without it, in 3 of 20 with it). Alone it does
  not improve validation D² reliably (+1.0%, p = 0.25), its weight has no
  common sign (median +0.09, negative in 9 of 20 units), and on all units its
  unique ΔD² is 0.0001 (reliable in 23% of units). It is in the model because
  it corrects the prediction before the stimulus, not because it adds
  prediction. Together with the finer kernel it is reliably better than the
  design before (+1.35% validation D², p = 0.007).
- The output function and the penalty were checked and not changed. On
  held-out bins the model is too high at low and middle predicted rates
  (observed / predicted 0.83 to 0.93) and too low from the 80th to the 95th
  percentile (1.08 to 1.12). One gain on the linear predictor is 0.99
  (median), two gains (history, other terms) are 0.99 and 0.96, and a cubic
  output function does not improve held-out D² (p = 0.62); none changes the
  peak. So the kernels are not shrunk by the penalty and exp is adequate; the
  cause of the calibration curve is not known.

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
2026-10-11, with the fixation kernel in the design:

- Flash kernel: 6 bases are worse than 8 (p = 0.0005); 10 and 12 are not
  better (p = 0.73 and 0.14). 100 ms is worse than 150 ms (p = 0.017); 200 and
  250 ms are not better (p = 0.37 and 0.81).
- Recent-flash term: the model without the term is not reliably worse (median
  −0.0009, lower in 13 of 20 units, p = 0.11). This result moved with the form
  of the fixation kernel (p = 0.008 with the in-port column, 0.07, 0.048, and 0.11
  with three forms of the kernel), so it is at the limit. The term stays in
  the design, as an exception to the rule of the simpler option, for two
  reasons. On all units its unique ΔD² is reliable in 60% of the units, the
  same level as the withdrawal kernel. And with an explicit adaptation term in
  the model, the fixation kernel is still necessary, which is the evidence that the
  fixation effect is more than adaptation.
- Names: the recent-flash term is "fast adaptation" in the figures, because it
  is active only for a flash within 50 ms of another. The time from the center
  entry to the first flash is the "initiation", and the time from the first
  flash to the center exit is the "fixation"; its kernel is "stimulus during
  fixation", which says when it is active and not what causes it. The keys in
  the code and in the saved fits did not change: `fixation` is the initiation
  column and `hold` is the fixation kernel. Slow adaptation, if present, is
  inside the fixation kernel and cannot be separated from a state effect there:
  the two go together from the first flash to the center exit. The term is not
  named for adaptation because the effect ends at the center exit while the
  flashes continue, does not depend on the flash rate of the trial, and lowers
  the baseline and the flash response by the same factor.
- Fixation: no term, a step from the center entry, and a step from
  the first flash are worse than the kernel (p = 2e-5, 0.002, 0.001). The step
  with a first-flash kernel is equal (p = 0.67); the linear kernel is worse
  (p = 0.017); the coarse kernel (10 bases, 50 ms stretch) and the design
  without the initiation column are not reliably worse alone (p = 0.16, 0.43).
- Withdrawal: no kernel is worse than the 0 to 100 ms kernel (p = 0.002); the
  ±100 and ±300 ms kernels are not better (p = 0.28 and 0.47).
- Motion-energy PCs: no motion energy is worse than 10 PCs (p = 0.006); 25 and
  50 are not better (p = 0.62 and 0.65); 200 is worse (p = 0.001).

The figure has one row for each group of regressors: the raised cosines of its
kernel, drawn from `dmat.py`, then the tests of its choices.

Tests that this file marks as removed from the code were run with scratch
scripts. Their per-unit scores are in `flash_lobe_test/` in the session
directory (not tracked by git); they cannot be run again from the repository.

Earlier fits are kept next to the session directory: `_in_port_step` (the step
design), `_log10_fit` (10 bases, 50 ms stretch, no initiation column), and
`_linear_hold_fit` (10 linear bases).

### Figures

| Figure | Module | Content |
| --- | --- | --- |
| Main | `main_figure.py` | one trial, the kernels, prediction, and what explains the spikes |
| Fixation effect | `hold_effect.py` | two example units, the suppression during the fixation, its end at the exit, and its controls |
| Building the model (supplement) | `design_optimization.py` | the bases and the tests of each design choice |
| Checking the model (supplement) | `model_checks.py` | single trials, held-out units, terms that are not in the model, calibration, recovery of planted kernels |
| Flash against motion energy | `kernel_comparison.py` | weights of the two terms over units; descriptive |

Each figure module writes its caption next to the figure (`<figure>_caption.md`), with the numbers of that run.

`glm fit` writes no figure. `glm unique` writes `figure_data.npz` for the
example unit.

## Results

Full fit of the final design, 2026-10-11: 167 units, 294 trials, 10
motion-energy PCs. Artifacts are in
`figures/glm/GRB006_20240821_121447/all_fit_me/`, which git does not track.

- Full model: median cross-validated D² 0.072, median 0.54 bits/spike.
- Against the design with 10 fixation bases and no initiation column, D² is 0.0004
  higher per unit (paired median; higher in 118 of 167 units, p = 3e-8).
- Example unit 197: the model reaches the first-flash peak on held-out trials
  (about 105 against 105 spikes/s in 20-ms bins; 52% with the in-port column).
  Its predicted rate before the center entry is still too high (47 against 33
  spikes/s).

| Regressor | Columns | Median unique ΔD² | Units with unique ΔD² > 0 |
| --- | ---: | ---: | ---: |
| Spike history | 10 | 0.0271 | 92% |
| Fixation | 12 | 0.0063 | 87% |
| Visual flash | 8 | 0.0050 | 83% |
| Motion energy | 10 | 0.0021 | 64% |
| Withdrawal | 4 | 0.0014 | 62% |
| Fast adaptation (recent flash) | 8 | 0.0016 | 60% |
| Initiation | 1 | 0.0001 | 23% |

- Motion energy is trial-specific: with the regressor taken from another trial,
  the median loss is 0.0021, the same as with the shuffle within a trial.
- Held-out units (panels d and e of `model_checks.svg`, values in
  `model_checks_stats.json`):
  the 147 units that took no part in the design choices do not differ from the
  20 selection units in D² (p = 0.43) or in any unique ΔD² (p from 0.18 to
  0.86).
- Residual checks on all units (in-sample ΔD² of a term that is not in the
  model, and with shuffled labels): session drift 0.0013 (0.0007), response
  side after withdrawal 0.0013 (0.0003), flash rate 0.0009 (0.0006), time in
  trial 0.0008 (0.0007), response side before withdrawal 0.0002 (0.0002).
- The rate in the 100 ms before the first flash is more than 10% off in 42% of
  the units (median observed / predicted 1.01). Part of that time is before
  the center entry, which no term covers.
- Calibration (panel g of `model_checks.svg`; held-out 1-ms bins of each unit,
  sorted by predicted rate, in 20 groups): observed / predicted is 0.83 to
  0.90 from the 5th to the 50th percentile (above 1 in 19 to 35% of units),
  1.04 to 1.14 from the 80th to the 95th (above 1 in 66 to 81%), and 0.99 in
  the highest 5%. The model predicts too much at low and middle rates and too
  little at high rates. The finer fixation kernel did not change this, and the
  cause is not known (see the tests of a gain and of the output function
  above, which were run on the earlier fit).
- The low rate in the fixation is not the average of scattered flash responses.
  The flash terms of the model, averaged over trials, give 1.16 of the rate
  before the first flash in the fixation and 1.16 after the exit; the observed
  rate is 0.56 and 1.25 (below the flash terms in the fixation in 150 of 167
  units, p = 6e-23).
- Fixation effect (`hold_effect.svg`, descriptive model with a kernel after the
  center exit): Δ log rate −0.31 in the fixation and +0.19 after the exit, higher
  after the exit in 144 of 167 units. Two-step control: −0.48 with all terms
  fit together, −0.33 with the fixation kernel fit last (below 0 in 90% of units).
  Example units: 36 (the lowest fixation level among units with 5 to 20 spikes/s)
  and 576 (the highest fixation level among units with 2 spikes/s or more).
- `uv run python -m thesis.ephys.analyses.glm.main_figure` draws the main
  figure (`main_figure.svg`) for the example unit and all units.
