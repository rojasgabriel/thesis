# V1 GLM

Poisson encoding model for GRB006 session `20240821_121447`. Predict V1 spikes
from flashes, a center-poke kernel truncated at the first flash, peri-exit
movement, pre-response choice side, and additive camera motion-energy PCs.
Coefficients are conditional associations, not causal effects.

`PoissonGLM` owns the `_me` artifact names. `prepare` skips a step if that file
already exists. Keep the previous sklearn run (`stimulus_windows.npz`,
`video_features.npz`, `common_design.npy`, `all_fit/`) untouched.

## Data

- Units: quality 1, stability 0. No sensory-response filter. 168 eligible
  units; 12 depth-spaced units for `--units test`.
- Trials: completed left/right, no early withdrawal. Require ordered center
  entry, first measured flash, center exit, response entry.
- Align to the first measured flash, not the Bpod stimulus command.
- Grid: DAMN 1 ms, pre=100 ms, post=2.54 s (2,639 centers). Keep the full
  window. Fitting **masks bins after that trial’s response entry**; it does not
  shorten the grid. 10.5% of responses are after 1.5 s; no flashes occur after
  response.
- Split whole trials randomly 60/20/20 (`SPLIT_SEED = 20260912`). `--units test`
  scores train/validation only. After settings are fixed, `fit --units all`
  refits every eligible unit on train+validation and scores the held-out trial
  split once.
- 294 completed trials, 168 units, 3502 flashes.

## Design

Training-row z-score on **valid** training bins; that scaling is frozen for
validation and test. Intercept unpenalized. No spike history, session-time
drift, go cue, outcome, punishment, or common response-entry kernel.

| Group | Representation | Columns |
| --- | --- | ---: |
| Visual flashes | Every measured flash; 6 causal raised cosines, 0–150 ms | 6 |
| Center poke | 6 causal raised cosines, 0–150 ms, zeroed at and after the first flash | 6 |
| Center exit | 9 raised cosines, −300–+300 ms | 9 |
| Response side | Left=−1, right=+1; 6 bases, −300–0 ms (pre-entry only). Displayed kernel is right−left = 2× fitted | 6 |
| Motion energy | 3 raised cosines per PC, peaks at −200, 0, +200 ms | 3 / PC |

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
LBFGS). History is omitted so a later DAMN population fit can share one X.

Per unit and model: L2 path from 1e−3 to 1e3, extend ×10 if a boundary wins,
require convergence, break ties toward the stronger penalty. Choose
motion-energy PC count by mean per-unit validation D². Null is a constant rate
from training rows (train+validation rows for the held-out trial split). Report
D² and bits/spike as median (IQR) over units. One session: no population
p-value.

```bash
uv run python -m thesis.ephys.analyses.glm prepare
uv run python -m thesis.ephys.analyses.glm fit --units test
```

`prepare` writes `stimulus_windows_me.npz`, `video_me_features.npz`, and
`common_design_me.npy` under `figures/glm/`. `--units test` writes
`test_fit_me/` (resumable), no held-out trial metric. `fit --units all` writes
`all_fit_me/` only after the model is accepted.

## Attribution

Shuffle one block, refit, reselect α on validation, refit train+validation,
score the held-out trial split. Shuffle only among valid bins, within trial.
Motion-energy PCs stay one block.

`unique test D² = complete D² − shuffled D²`

```bash
uv run python -m thesis.ephys.analyses.glm attribute --units test
```

## Figures

After the all-unit held-out fit:

```bash
uv run python -m thesis.ephys.analyses.glm figures --units all
```

Rasters are independent Poisson draws from the covariate-conditioned rate (no
history simulation). Post-response bins remain on the plotted grid but are
excluded from likelihood and scores.

## Related methods

Same family as [Truccolo et al. 2005](https://pubmed.ncbi.nlm.nih.gov/15356183/)
and [Pillow et al. 2008](https://sites.stat.columbia.edu/liam/research/pubs/pillow-nature-08.pdf),
without a self-history block in this revision.
[Talluri et al. 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10620084/)
(video SVD, ridge, trial CV),
[Stringer et al. 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6525101/)
(video as broad behavior). Basis counts and windows remain assumptions.

## Results

Not yet run for this design. Do not treat `all_fit/` numbers as current.
