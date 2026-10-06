# V1 GLM

Poisson encoding model for GRB006 session `20240821_121447`. Predict V1 spikes
from a common flash kernel, a pre/post-withdrawal contrast, a flash-adaptation
term, an initiation kernel truncated at the first flash, peri-withdrawal
movement, response entry, and additive camera motion-energy PCs. Each unit
also has a strictly past self-history filter.
Coefficients are conditional associations, not causal effects.

`prepare` reuses saved windows and video features by file presence. It checks
the saved design metadata against the current task bases and adaptation
parameters. Only one `glm` command can use a subject/session output directory
at a time.

## Data

- Units: quality 1, stability 0. No sensory-response filter. Exclude unit 212
  because of unstable out-of-fold spike-history extrapolation. Draw a fixed
  random sample of 20 from the remaining units for `--units sample`
  (`SAMPLE_SEED = 20260914`).
- Trials: completed left/right. Require ordered initiation, first measured
  flash, withdrawal, and response entry.
- Align to the first measured flash, not the Bpod stimulus command.
- Grid: DAMN 1 ms, pre=100 ms, post=2.54 s (2,639 centers). Keep the full
  window. Fitting **masks bins after that trial’s response entry**; it does not
  shorten the grid. Only flashes from initiation through response entry enter
  the design.
- Split whole trials randomly 60/20/20 (`SPLIT_SEED = 20260912`). Train fits and
  validation chooses the motion-energy PC count and each unit's penalty. The
  third block is not scored on its own; held-out performance comes from
  cross-validation over whole trials.
- `fit` then cross-validates at that PC count: 10 folds over whole trials
  (`CV_SEED = 20260913`). Each trial belongs to one test fold. A fold with no
  test spikes for a unit is skipped; at least two scored folds are required.
  Reports per-unit mean and SEM across the scored folds. It finally refits each
  unit on every valid trial at the median fold penalty, writing `unit_*_final.json`:
  that model carries the coefficients the example figures read. Its score
  comes from the fold records. `unique` refits within those folds using their
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
| Initiation | 4 causal raised cosines, 0–90 ms, zeroed at and after the first flash | 4 |
| Withdrawal | 9 raised cosines, −300–+300 ms | 9 |
| Response entry | 6 bases, −300–0 ms | 6 |
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

DAMN supplies the grid, convolution, resampling, and trial-edge truncation.
Fits use sklearn `PoissonRegressor` (log link, unpenalized intercept, L2,
LBFGS). Each unit's design appends its standardized self-history block.

Per unit and model: fixed L2 grid from 1e−6 to 1e4 in powers of ten. Fit from
strong to weak penalties, with a cold retry after a failed warm start. Drop
penalties that still fail to converge and break loss ties toward the stronger
penalty. The grid does not expand automatically. Choose the motion-energy PC
count by mean per-unit validation D². Each cross-validation fold selects its
penalty using an inner validation split. The null is a constant rate from the
training rows for selection, or all non-test rows for a cross-validation fold.
Report D² and bits/spike as median (IQR) over units. One session: no population
p-value.

```bash
uv run glm prepare
uv run glm fit --units all
```

`prepare` writes `stimulus_windows_me.npz`, `video_me_features.npz`, and
`common_design_me.npy` under `figures/glm/<subject>_<session>/`. Pass
`--subject`, `--session`, or `--root` to work on another dataset. `fit` writes
named summaries under `all_fit_me/`. Restartable per-unit records stay under
`all_fit_me/checkpoints/`. The same command writes the population summary,
bits-versus-deviance scatter, and, when unit 197 was fit, the kernel,
model-design, and held-out prediction SVGs in `all_fit_me/`.
Use `--units sample` for a 20-unit smoke run; its checkpoints, summaries, and
figures stay separately in `sample_fit_me/`. A sample that omits unit 197 skips
those three example panels. Derived example arrays are saved as
`figure_data.npz` and `figure_data.json` in the fit directory. Interrupted
figure-data writes are replaced on retry under the session lock. Existing
figures at the session root are older exports and are not updated by these
commands.

Fit checkpoints are reused by file presence. If inputs or model settings
change, clear the affected checkpoints before rerunning.

The prediction figure shows unit 197. Each plotted trial uses its saved
out-of-fold model, so that model did not train on the trial. Predicted rasters
are Poisson draws from one-step-ahead expected counts conditioned on observed
spike history, not free-running simulations. The trace compares observed and
predicted firing rates in complete 10-ms bins without smoothing. Post-response
bins are excluded from rates, likelihood, and scores. Raster gaps after response
entry are shaded within the displayed interval.

The model-design figure uses one real test trial from unit 197. The trial is
nearest the unit's median pre-response spike count. Panel a shows every
measured flash, the other task
events and their filter support, the first three continuous motion-energy PC
scores, and strictly past history support. Panel b
shows the exact fitted terms `X_g β_g` for visual, other task, all selected
motion-energy, and history columns. Their sum plus the intercept is checked
against the full linear predictor. Panel c compares the observed spike raster
with Poisson draws from the fitted conditional counts. This explanatory panel
uses the final model fit on all valid bins; held-out predictions stay in the
separate prediction figure. Post-response bins are not shown. The single kernel figure shows task and self-history traces
for unit 197.

## Unique explained deviance

Within each cross-validation fold, shuffle one block and refit at that fold's
full-model alpha. Shuffle only among valid bins, within trial. Motion-energy
PCs stay one block.

`unique ΔD² = complete D² − one-removed D²`
`maximal ΔD² = block-alone D² − all-shuffled D²`

Both run inside the same 10 folds as `fit`, at the penalty that fold selected,
so they are comparable with the cross-validated deviance. Reported as the mean
across folds with its SEM. Following
[Oesch et al. 2026](https://doi.org/10.1038/s41467-026-70639-1) and
[Musall et al. 2019](https://www.nature.com/articles/s41593-019-0502-4).

```bash
uv run glm unique --units all
```

The same command writes `deviance_explained.svg` in the selected fit directory
(`all_fit_me/` or `sample_fit_me/`), alongside its other figures.

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
