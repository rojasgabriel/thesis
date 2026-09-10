# V1 GLM: specification and workflow

## Scientific question

For GRB006 session `20240821_121447`, fit a Poisson encoding model that predicts
V1 spikes from sensory, task, behavioral, and spike-history variables. The
primary result is how well the complete model predicts held-out spikes. Video
is one behavioral regressor group used to capture movement-related variance;
it is not the scientific focus and this is not a video-only GLM. Validation
over camera-PC counts selects the size of that nuisance block.

The model estimates conditional associations. It does not establish that a
regressor causes a neural response.

## Trials, units, and time grid

- Use quality criterion 1 and stability parameter 0. Do not select units by
  sensory response.
- Use completed left/right trials without early withdrawal. Completed trials
  must have ordered center entry, first measured flash, center exit, and
  response entry. An invalid order raises an error.
- Align each trial to its first measured visual flash, not the Bpod stimulus
  command.
- Use DAMN's native 1 ms grid with pre=100 ms and post=2.54 s. It contains 2,639
  centers from -99 ms through +2.539 s. The corresponding edges are -99.5 ms
  and +2.5395 s.
- Keep the full window. Do not warp time, join trial gaps, smooth spike counts,
  subtract a baseline, or binarize counts.
- Split whole trials chronologically into 176 training, 59 validation, and 59
  test trials. Do not let one trial enter more than one split.
- Use the 12 fixed depth-spaced units for the pilot. The pilot uses only
  training and validation responses. It does not score test responses. After
  the pipeline passes, select settings with all eligible units, refit on
  training plus validation, and score the test set once.

The prepared file contains 294 trials and 775,866 response bins. Across all 168
eligible units, only 239 unit-bins contain two spikes. These bins are 0.00018%
of the 130,345,488 unit-bins, so the 1 ms Poisson model is a close discrete
approximation to a point-process model without deleting valid spikes.

```bash
uv run python -m thesis.ephys.preprocessing.prepare_v1_glm \
  --frame-times figures/v1_glm/frame_times.npy \
  --output figures/v1_glm/stimulus_windows.npz
```

The command refuses to overwrite an existing output.

## Design matrix

All nonconstant columns are centered and scaled with training rows only. The
same fixed scaling is applied to validation and test rows. The intercept is not
penalized.

| Group | Representation | Columns |
| --- | --- | ---: |
| Measured visual flashes | One event for each measured flash; six linearly spaced causal raised cosines from 0 to 300 ms | 6 |
| Center entry | Six linearly spaced causal raised cosines from 0 to 300 ms | 6 |
| Go-cue command | Bpod command converted to NIDQ time; six linearly spaced causal raised cosines from 0 to 300 ms | 6 |
| Center exit | Nine linearly spaced raised cosines from -300 to +300 ms | 9 |
| Common response entry | Nine linearly spaced raised cosines from -300 to +300 ms | 9 |
| Response side | Left=-1 and right=+1 at response entry, with the same nine peri-response bases | 9 |
| Eventual outcome | Error=-1 and rewarded=+1 at response entry; six bases restricted to -300 through -1 ms | 6 |
| Wrong-punishment command | Bpod command converted to NIDQ time; six linearly spaced causal raised cosines from 0 to 300 ms | 6 |
| Slow drift | Linear and quadratic absolute session time | 2 |
| Self-history | Ten log-spaced raised cosines over strictly past lags from 1 through 100 ms | 10 per unit |
| Raw video | For each retained PC, three linearly spaced raised cosines with peaks at -200, 0, and +200 ms | 3 per PC |

The task block has 57 columns. It has full training rank, a scaled condition
number of 5.64, and maximum absolute pairwise column correlation of 0.619. The
linear and quadratic session-time terms are common nuisance variables. They are
needed because firing rates change across the chronological split for a small
set of units even when spike amplitudes pass the stability filter.

The zero-camera validation candidate has 69 penalized columns: 57 task, 2
drift, and 10 self-history. The complete-model candidates have 99, 144, 219,
369, or 669 penalized columns for 10, 25, 50, 100, or 200 video PCs. Each model
also has one unpenalized intercept.

Six causal bases over 300 ms give a smooth event filter at about 60 ms peak
spacing. Nine peri-event bases cover motor and response associations over a
longer 600 ms range. The log-spaced history basis has much finer support near a
spike, where refractoriness changes quickly, and coarser support later. The
three video bases allow a smooth movement-related association before and after
the spike. Because the video filter is acausal, it is a nuisance association
and must not be read as a causal encoding filter.

There is no same-bin history, cross-unit coupling, previous-trial term,
early-withdrawal term, or interaction term in the current model. The additive
model is the first interpretable prediction model. More complex nonlinear or
coupled models require evidence from held-out failures before they are added.

## Video preparation and timing

The camera mapping uses NIDQ channel 1 falling edges in decoded-frame order.
One one-sample internal pulse is removed because this gives exactly 191,006
pulses for 191,006 decoded frames. Container presentation times are not used.
The largest adjacent change in the falling-edge residual relative to the stored
camera clock is 0.143 ms.

The raw grayscale video is area-downsampled from 640 by 512 to 80 by 64. PCA is
fit to training frames only. It retains 200 axes, which explain 96.20% of the
training-frame variance. Component scores are scaled with training frames only.
Validation compares nested prefixes of 10, 25, 50, 100, and 200 axes.

Raw-frame PCs can contain posture, movement, illumination, pupil, and other
visible changes. They are not identical to frame-difference motion-energy
features. This broad content is acceptable here because the stated purpose is
to absorb visible behavioral variance, but the result must be labeled as a raw
video contribution rather than a pure movement contribution.

```bash
uv run python -m thesis.ephys.preprocessing.audit_camera_pulses \
  --output figures/v1_glm/frame_mapping.json

uv run python -m thesis.ephys.preprocessing.video_svd \
  figures/v1_glm/stimulus_windows.npz \
  --output figures/v1_glm/video_features.npz
```

## Poisson model and fitting

For equal 1 ms bins, the model is

`count_t ~ Poisson(exp(intercept + X_t beta))`.

The exponential is the canonical log-link nonlinearity. Equal bin duration
means that no separate exposure offset is needed. The observed spike history
helps model refractoriness and other conditional spike dependence, but it does
not prove that the remaining conditional count variance is exactly Poisson.

DAMN remains the source of the accepted time grid, event convolution,
continuous resampling, and trial-edge truncation. Its current fitter clips the
linear predictor to [-8, 8]. At 1 ms, the lower bound forces predicted rates to
at least 0.335 Hz. One eligible unit has a training-window rate of 0.149 Hz, so
that fit would be biased and can start in a zero-gradient region. For this
reason, fitting uses scikit-learn's `PoissonRegressor`, which uses the same
canonical log link, an unpenalized intercept, an L2 penalty, and LBFGS without
that lower clip.

Select the L2 penalty separately for each unit and each model. Start with seven
values from 1e-3 through 1e3 and extend by factors of ten when a boundary value
wins. Require optimizer convergence. If validation losses are equal within
numerical precision, keep the stronger penalty. Select one video-PC count for
the population by mean per-unit validation deviance explained.

Validation metrics compare predictions with a constant count learned from the
training rows. After settings are fixed, keep the training-fit PCA axes and all
training-fit scaling fixed. Refit coefficients on training plus validation.
Final test metrics compare against a constant count learned from those same
training-plus-validation rows. Report Poisson deviance explained and
bits/spike. For this one session, show every unit and report the median and
interquartile range without a population p-value.

```bash
uv run python -m thesis.ephys.analyses.v1_glm prepare-design
uv run python -m thesis.ephys.analyses.v1_glm fit --units pilot
```

The first command saves the training-scaled common matrix to
`figures/v1_glm/common_design.npy`. The pilot command saves one validation
record per unit so an interrupted fit can resume. It does not calculate a test
metric. Use `fit --units all` only after the pilot passes and the model is
accepted.

## Audio-event interpretation

Measured OBX audio events are absent in this session. The loader extracts the
go-cue, wrong-punishment, and early-withdrawal sound commands from Bpod and
converts them to NIDQ time with Bpod StreamSync. These regressors mark commands,
not measured acoustic onset.

The wrong-punishment command occurs within about 5 ms of response entry on
error trials. The pre-response outcome contrast and post-command punishment
kernel make the matrix identifiable, but they do not create an independent
no-sound control. A fitted punishment coefficient is therefore a conditional
error-trial association, not proof of an auditory response. Early-withdrawal
trials require a separate analysis with their own Bpod sound command because
they do not share the completed-trial event sequence.

## Method check against spiking-GLM work

The main choices follow common spiking-GLM practice:

- Truccolo et al. describe point-process GLMs that combine extrinsic stimulus
  and behavior covariates with a neuron's own history in a log conditional
  intensity model: [J Neurophysiology, 2005](https://pubmed.ncbi.nlm.nih.gov/15356183/).
- Pillow et al. use an exponential conditional intensity, penalized likelihood,
  held-out log likelihood, and raised-cosine temporal filters. Their self-history
  filter uses 10 log-spaced bases to keep fine resolution near the spike and
  coarser resolution later: [Nature, 2008](https://sites.stat.columbia.edu/liam/research/pubs/pillow-nature-08.pdf).
- Talluri et al. include video SVD regressors with temporal filters, ridge
  regularization, whole-trial cross-validation, standardized predictors, and a
  smooth session-drift block in a visual-cortex encoding model:
  [Nature Neuroscience, 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10620084/).
- Stringer et al. show why high-dimensional video features can capture broad
  behavioral variance in mouse cortex, while also making clear that such
  features are behavioral predictors rather than a single named movement:
  [Science, 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6525101/).

These papers support the model family and safeguards. They do not determine a
unique basis count or time range. Those values remain explicit assumptions and
must be checked with validation performance, fitted kernel shape, residuals,
rate calibration, and sensitivity analyses.

## First full fit

The full fit completed for all 168 eligible units. Validation selected 25 raw
camera PCs. The optimum is shallow between 10 and 50 PCs, so 25 is a selected
prediction setting rather than a biologically meaningful dimensionality.

| Camera PCs | Mean validation deviance explained |
| ---: | ---: |
| none | 0.06668 |
| 10 | 0.06953 |
| **25** | **0.07024** |
| 50 | 0.06925 |
| 100 | 0.06555 |
| 200 | 0.06019 |

On the 59 untouched test trials, the complete model gave the following
performance across 168 units:

| Test metric | Median (IQR) |
| --- | ---: |
| Deviance explained | 0.05961 (0.03120 to 0.08835) |
| Bits/spike | 0.42199 (0.21586 to 0.76841) |

All 1,008 validation penalty paths selected an interior penalty after automatic
grid extension. All saved coefficients and metrics are finite and have the
expected dimensions. The test files were written only after every validation
file. Summed across units, the complete model predicted 189,044 spikes against
191,538 observed spikes.

The chronological test split also exposes limits. Four complete-model units
scored below the constant-rate null. Seven complete models produced an expected
count above one in at least one 1 ms bin. The Poisson likelihood permits these
tail values, but they warn against treating every fitted unit as a calibrated
spike simulator.

This result measures prediction from the complete set of regressors. It does
not separate the predictive contributions of visual, audio, other task,
history, and video groups. Group ablations would answer a different question;
individual coefficients are not a substitute because the regressors are
correlated.

Results are in `figures/v1_glm/all_fit/`: `summary.json` contains the gated fit
summary, `unit_results.csv` contains one row per unit, and `summary.png` and
`summary.pdf` show camera-PC selection, the complete model's held-out metric
distributions, and spike-count calibration. This is one session, so units are
displayed as observations without a population p-value.

## Predicted and observed spike trains

The prediction figure adapts the direct raster comparison in Pillow et al.
without treating these data as repeated presentations of one stimulus. It uses
all 59 chronological test trials, aligned to the first measured flash. These
trials have different later flashes, choices, outcomes, sounds, and movements.

The displayed unit is selected by a fixed, inspectable rule: its full-model
test deviance explained is nearest the 168-unit median. This selects unit 197
(`D^2=0.05951`, population median `0.05961`) without selecting for visual
appearance or unusually high performance. The choice is for post-fit display
only and does not change the reported test estimates.

Panel a shows the real spikes. Panel b shows model-predicted spike trains. They
are generated recursively: each generated spike becomes part of the model's
recent spike-history input for later bins. The external task, audio, drift, and
video covariates stay fixed. The history is seeded from real spikes before the
displayed window, and no rate clipping is applied.

Panel c shows the one-step prediction used for test scoring. For each 1 ms bin,
this prediction uses the real spikes from the preceding 100 ms rather than
earlier simulated spikes. It therefore shows the model's conditional expected
rate, not a generated spike train. The 20 ms Gaussian smoothing is for display
only.

```bash
uv run python -m thesis.ephys.analyses.v1_glm_prediction
```

The command writes both summary formats and `predicted_spike_trains.png` and
`predicted_spike_trains.pdf` in `figures/v1_glm/all_fit/`. The recursive draw is
an offline covariate-conditioned simulation, not a causal online forecast,
because some video and task filters use future information. A single simulated
raster illustrates model behavior; held-out deviance and bits/spike remain the
quantitative evaluation.

## Current state

The camera mapping, trial grid, raw-video PCA, task matrix, drift terms, history
basis, Poisson fitter, validation path, test gate, pilot, and full fit are
complete. Focused tests are in `tests/`:

```bash
uv run python -m unittest discover -s tests -v
```
