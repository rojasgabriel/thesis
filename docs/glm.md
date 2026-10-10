# V1 GLM

Poisson encoding model for GRB006 session `20240821_121447`. Predict V1 spikes
from a common flash kernel, a pre/post-withdrawal contrast, a flash-adaptation
term, a flash × motion-energy interaction, an initiation kernel truncated at
the first flash, peri-withdrawal
movement, response entry, and additive camera motion-energy PCs. Each unit
also has a strictly past self-history filter.
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
  training block. The validation block chooses the motion-energy PC count. To
  score each PC count, validation also picks a penalty for each unit and model,
  but no later step uses these penalties: each cross-validation fold picks its
  own. The third block is not scored on its own; held-out performance comes from
  cross-validation over whole trials.
- `fit` then cross-validates at that PC count: 10 folds over whole trials
  (`CV_SEED = 20260913`). Each trial belongs to one test fold. A fold with no
  test spikes for a unit is skipped; at least two scored folds are required.
  Reports per-unit mean and SEM across the scored folds. It finally refits each
  unit on every valid trial at the median fold penalty and writes
  `unit_*_final.json`. The kernel and model-design figures read that model. The
  prediction figure reads the saved fold models instead, so no plotted trial
  was in the training data of its model. The final model's score comes from the
  fold records. `unique` refits within those folds using their
  selected penalties.

The PC count is selected once before cross-validation. PCA and common-design
scaling also use the original training split, rather than being recomputed
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
| Common flash | Every modeled flash; 6 causal linearly spaced raised cosines, 0–150 ms | 6 |
| Flash state contrast | Same flash bases, scaled −0.5 before withdrawal and +0.5 at or after withdrawal | 6 |
| Flash adaptation | Same flash bases, scaled by centered synaptic resource; each flash uses 50% of the resource, which recovers with a 200-ms time constant | 6 |
| Flash × motion energy | Same flash bases, scaled by the total motion energy of the video frame at the flash (mean over pixels), centered and in SD over all modeled flashes | 6 |
| Initiation | 4 causal raised cosines, 0–90 ms, zeroed at and after the first flash | 4 |
| Withdrawal | 9 raised cosines, −300–+300 ms | 9 |
| Response entry | 6 bases before the event, −300 to −1 ms; the entry bin gets no value | 6 |
| Motion energy | One contemporaneous value per selected PC; no temporal basis | 1 / PC |
| Self-history | 10 log-spaced raised cosines at strictly past lags, 1–100 ms | 10 |

No response-side, same-bin history, cross-unit coupling, or previous-trial
terms. Motion-energy coefficients are contemporaneous nuisance associations.
Pose tracking is not included.

## Video

NIDQ ch1 falling edges in decoded-frame order. Do not use container times.

640×512 grayscale → 80×64. Feature is `|frame[t] − frame[t−1]|` when decoded
indices are adjacent; the first frame of a gap is zeros. PCA on training
frames that have a real difference, 200 axes. Scores scaled on those training
frames. Validation prefixes: 10, 25, 50, 100, 200.

## Fit

`count_t ~ Poisson(exp(intercept + X_t β))` at 1 ms on unmasked bins.

DAMN supplies the 1 ms grid, the spike counts, and the event convolution, and it
clips each kernel at the trial-window edges. NumPy `np.interp` puts the
motion-energy scores on the grid; bins outside the camera frame times get zero.
The code, not DAMN, sets the initiation kernel to zero at and after the first
flash.
Fits use sklearn `PoissonRegressor` (log link, unpenalized intercept, L2,
LBFGS). Each unit's design appends its standardized self-history block.

Per unit and model: fixed L2 grid from 1e−6 to 1e4 in powers of ten. Fit from
strong to weak penalties, with a cold retry after a failed warm start. Drop
penalties that still fail to converge and break loss ties toward the stronger
penalty. The grid does not expand automatically. Choose the motion-energy PC
count by mean per-unit validation D². Each cross-validation fold selects its
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
`all_fit_me/checkpoints/`. The same command writes the population summary
and the bits-versus-deviance scatter in `all_fit_me/`. `unique` writes the
kernel, model-design, and held-out prediction SVGs there, because these three
figures show the example unit and its selection needs the `unique` results.
The example unit is the unit whose visual flash block explains the most
deviance per second (`example_unit_id` in `results.py`): unique flash ΔD² ×
bits/spike ÷ D² × firing rate. Unique flash ΔD² alone is not used. It is a
fraction of the null deviance of the unit, so its maximum selects a unit with
few spikes, and the rasters of that unit are too sparse to compare.
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
the other task
events and their filter support, the first three continuous motion-energy PC
scores, and strictly past history support. Panel b
shows the exact fitted terms `X_g β_g` for visual, other task, all selected
motion-energy, and history columns. Their sum plus the intercept is checked
against the full linear predictor. Panel c compares the observed spike raster
with Poisson draws from the fitted conditional counts. This explanatory panel
uses the final model fit on all valid bins; held-out predictions stay in the
separate prediction figure. Post-response bins are not shown. The single kernel figure shows task and self-history traces
for the example unit.

## Unique explained deviance

Within each cross-validation fold, shuffle one block and refit at that fold's
full-model alpha. Shuffle only among valid bins, within trial. Motion-energy
PCs stay one block.

`unique ΔD² = complete D² − one-removed D²`
`maximal ΔD² = block-alone D² − all-shuffled D²`

`trial-specific ΔD² (motion energy) = complete D² − D² with motion energy from another trial`

The third value is a control for motion energy only. It replaces the block with
the motion energy of another trial at the same time in the trial. This keeps the
average time course of motion energy in a trial. If the trial-specific ΔD² is
near zero and the unique ΔD² is not, the block predicts spikes through the time
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

The same command writes `deviance_explained.svg` and the three example-unit
figures in the selected fit directory (`all_fit_me/` or `sample_fit_me/`).
`deviance_explained.svg` has five panels: (a) full-model D², (b) maximal and
(c) unique ΔD² for each block, as box plots without unit points or outlier
markers; (d) the median unique ΔD² of each block against its column count, on
logarithmic axes; (e) the fraction of units for which the block gives unique
ΔD² > 0 (one-sided t-test of the fold values against zero, p < 0.05), with a
dotted line at the chance level of 0.05.

## Regressor check

Use this to find blocks that add design columns but no held-out deviance. It
identifies these blocks; it does not remove them. Fit
and score some samples that share no unit, then compare them:

```bash
uv run glm fit --units sample1 && uv run glm unique --units sample1
uv run python -m thesis.ephys.analyses.glm.regressor_check
```

The script makes no new fits. It reads the `unique` records of each completed
`sample*_fit_me/` directory and writes `regressor_check.svg` and
`regressor_check.json` in the session output directory. The figure has the same
five panels as `deviance_explained.svg`, for the units of all samples. Panels d
and e also show the spread over the samples. Each sample selects its own
motion-energy PC count, so the motion-energy block can have a different column
count in each sample.

## Kernel comparison

```bash
uv run python -m thesis.ephys.analyses.glm.kernel_comparison
```

This script also makes no new fits and reads all completed sample fits. It
writes `kernel_comparison.svg` and `kernel_comparison.json`. Panel a compares
the size of the visual flash weights and of the motion-energy weights (the norm
of the block's coefficients) for every unit. Panels b and c show the common
visual flash kernel and the motion-energy weight map of three example units: one
that the flash predicts, one that motion energy predicts, and one that both
predict. Motion energy has no temporal kernel, so its panel is a map over the
video frame. The module docstring gives the selection rule.

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
with a self-history block but no coupling between units.
[Talluri et al. 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10620084/)
(video SVD, ridge, trial CV),
[Stringer et al. 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6525101/)
(video as broad behavior). Basis counts and windows remain assumptions.

## Results

Not yet run for this design. Do not treat `all_fit/` numbers as current.
