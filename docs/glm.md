# V1 GLM

Poisson encoding model for GRB006 session `20240821_121447`. Predict V1 spikes
from flashes, a center-poke kernel truncated at the first flash, peri-exit
movement, pre-response choice side, and additive camera motion-energy PCs.
Each unit also has a strictly past self-history filter. Coefficients are
conditional associations, not causal effects.

`prepare` reuses a completed artifact when that file exists. Only one `glm`
command can use a subject/session output directory at a time.

## Data

- Units: quality 1, stability 0. No sensory-response filter. 168 eligible
  units; a fixed random sample of 20 for `--units sample`
  (`SAMPLE_SEED = 20260914`).
- Trials: completed left/right, no early withdrawal. Require ordered center
  entry, first measured flash, center exit, response entry.
- Align to the first measured flash, not the Bpod stimulus command.
- Grid: DAMN 1 ms, pre=100 ms, post=2.54 s (2,639 centers). Keep the full
  window. Fitting **masks bins after that trial’s response entry**; it does not
  shorten the grid. 10.5% of responses are after 1.5 s; no flashes occur after
  response.
- Split whole trials randomly 60/20/20 (`SPLIT_SEED = 20260912`). Train fits and
  validation chooses the motion-energy PC count and each unit's penalty. The
  third block is not scored on its own; held-out performance comes from
  cross-validation, which scores every trial.
- `fit` then cross-validates at that PC count: 10 folds over whole trials
  (`CV_SEED = 20260913`), so every trial is scored once and all 294 contribute.
  Reports per-unit mean and SEM across folds. It finally refits each unit on
  every valid trial at the median fold penalty, writing `unit_*_final.json`:
  that model carries the coefficients the figures and `unique` read, and its
  held-out score is the cross-validated deviance rather than a second split.
- 294 completed trials, 168 units, 3502 flashes.

## Design

Training-row z-score on **valid** training bins; that scaling is frozen for
validation and test. Intercept unpenalized. No session-time drift, go cue,
outcome, or punishment kernel.

| Group | Representation | Columns |
| --- | --- | ---: |
| Visual flashes | Every measured flash; 6 causal raised cosines, 0–150 ms | 6 |
| Center poke | 4 causal raised cosines, 0–90 ms, zeroed at and after the first flash | 4 |
| Center exit | 9 raised cosines, −300–+300 ms | 9 |
| Response entry | 6 bases, −300–0 ms; side-independent | 6 |
| Response side | Left=−1, right=+1; 6 bases, −300–0 ms (pre-entry only). Displayed kernel is right−left = 2× fitted | 6 |
| Motion energy | 3 raised cosines per PC, peaks at −200, 0, +200 ms | 3 / PC |
| Self-history | 10 log-spaced raised cosines at strictly past lags, 1–100 ms | 10 |

No same-bin history, cross-unit coupling, previous-trial, early-withdrawal, or
interaction terms. Video filters are acausal nuisance associations. Pose
tracking and flash×movement state are later, not this pass.

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

Per unit and model: L2 path from 1e−5 to 1e3, extend ×10 if a boundary wins,
require convergence, break ties toward the stronger penalty. Choose
motion-energy PC count by mean per-unit validation D². Null is a constant rate
from training rows (train+validation rows for the held-out trial split). Report
D² and bits/spike as median (IQR) over units. One session: no population
p-value.

```bash
uv run glm prepare
uv run glm fit --units all
```

`prepare` writes `stimulus_windows_me.npz`, `video_me_features.npz`, and
`common_design_me.npy` under `figures/glm/<subject>_<session>/`. Pass
`--subject`, `--session`, or `--root` to work on another dataset. `fit` writes
named summaries under `all_fit_me/`. Restartable per-unit records stay under
`all_fit_me/checkpoints/`; final figures stay under `all_fit_me/figures/`.
Use `--units sample` for a 20-unit smoke run.

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

## Figures

After the all-unit fit and unique-deviance analysis:

```bash
uv run glm figures --units all [--output-dir DIR] [--format {pdf,png,both}]
```

The prediction figure shows three distinct units: the best full-model fit, the
largest unique visual-flash contribution, and the largest unique contribution
from another task or motion-energy block. Each plotted trial uses its saved
out-of-fold model, so that model did not train on the trial. Predicted rasters
are Poisson draws from one-step-ahead rates conditioned on the observed spike
history, not free-running simulations. Post-response bins remain on the plotted
grid but are excluded from likelihood and scores.

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
