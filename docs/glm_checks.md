# V1 GLM: validation and supporting results

- **Question:** which checks support the comparison in [the main analysis](glm.md), and what remains uncertain?
- **Why:** predictive accuracy, repeatability, recovery, and control comparisons answer different questions. They do not all establish that a fitted flash effect is independent of locomotion.
- **Finding:** prediction is useful, but known-weight recovery and zero-effect simulations show limits on reading the weights. Raw video does not remove those limits.
- **Next:** use these checks to qualify the three main figures. Side preference is a check of the turn term, not part of the main sensory-enhancement argument.
- **Scope:** GRB006 / GRB064 order; one session per mouse. Current design: 2026-10-17; corrected controls checked: 2026-10-19. The held-out rate comparison of the main analysis is leak-corrected in its own measure (2026-10-21); the supporting kernel-peak scatter is not.

## Does the fit support interpretation of the flash component?

### What do the isolated flash-kernel peaks show?

- **Question:** how do the flash-only kernel peaks differ between stationary and movement periods at the same preceding gap?
- **Why:** these show the fitted flash component directly, while the main comparison measures its contribution to peak rates on the actual trials.
- **Finding:** mean kernel-peak changes in the flash-excited group are **+0.224 / +0.046 Δ log rate** at long gaps and **+0.072 / −0.060** at short gaps. Open circles show the mean of all **167 / 76** eligible units; filled circles show the same **23 / 15 flash-excited units** in both gap panels. These are final-fit kernel peaks; bars show SEM across units on both axes. Per-unit spread across saved fold fits remains in the report but is not drawn.
- **Next:** use the main held-out rate comparison to connect this component to the observed enhancement. Kernel peaks cannot be directly subtracted from peak rates, and their bias needs checking with this exact measure.

![Supporting flash-kernel peak comparison at matched long and short gaps](../figures/glm/flash_model.svg)

<a id="flash-exit-bias"></a>

### Can the fit create an apparent flash × exit effect?

- **Question:** can a fitted flash × exit effect appear when it is absent from simulated spikes?
- **Why:** a repeated flash kernel can resemble a sustained rate change. The penalty can assign background changes to flash terms.
- **Finding:** in the measure of the main comparison, the leak is small: **+0.04 / +0.03** normalized at long gaps in the excited units, against a held-out contribution of **+0.67 / −0.17**; corrected **+0.63 / −0.20**, pooled **+0.11** (95% interval −0.48 to 0.21). In the older matched-rate measure (different pairs, spikes/s), corrected peak changes are **+1.99 / −0.38 spikes/s** in **23 / 15** excited units (p = 0.13 / 0.28); pooled **+0.76**, 95% interval **−5.35 to 3.30**. Corrected kernel-mean effects across all units have opposite signs (**+0.039 / −0.153 Δ log rate**). Known fixation levels recover at only **0.75 / 0.84** of their size, despite high repeatability. These results do not establish an unchanged sensory response.
- **Next:** treat the new kernel-peak scatter as descriptive until that exact measure is checked for bias. The earlier matched-rate correction cannot simply be subtracted from its kernel peaks.

![Collinearity, known-weight recovery, repeatability, and flash × exit leak correction](../figures/glm/model_trust.svg)

### Is the low rate during fixation explained by visual input or stillness?

- **Question:** does low firing follow fixation, the flash train, or low movement?
- **Why:** these occur together in completed trials. Rates alone cannot identify which accounts for the change at exit.
- **Finding:** normalized firing is **0.62 / 0.61** late in fixation and **1.37 / 1.32** after exit. Firing can fall before the first flash and rises at early withdrawal when flashes stop. Rate rises with motion energy at the side port, but not at the center port. These controls support a fixation-linked change; they do not isolate its cause. The lowest-motion comparison is inconclusive in GRB064, and early withdrawal also adds a punishment tone.
- **Next:** retain a task-period rate component when estimating flash responses. Panels a–f show the rate change and its timing; g–j check fitting order, movement, and which units show it.

![Fixation, exit, onset, and movement controls](../figures/glm/hold_effect.svg)

### Does the model reproduce the observed flash-response time courses across gaps?

- **Question:** does the model capture the traces, beyond the peak summaries in the main comparison?
- **Why:** a peak can agree while response timing or shape is wrong. Gap-resolved traces check this part of the flash model.
- **Finding:** observed and fitted matched peaks are higher after exit across the gap groups in both mice. The traces compare timing and shape within gap and task period (black: observed; purple: predicted). They include background and preceding-flash responses and use the final refit, so this is an in-sample check, not an isolated sensory-gain estimate.
- **Next:** use this check to assess the flash-history model; use the simulation results above to qualify an exit-effect interpretation.

![Observed and predicted flash-response time courses by gap and task period](../figures/glm/flash_by_gap.svg)

### Which flash response does the model miss after the exit?

- **Question:** the full model predicts less long-gap enhancement than observed. Which flashes does it miss, and does the missing part need a flash?
- **Why:** an observed peak is the maximum of a noisy mean, so it is larger than the peak of a smooth predicted rate even when the model is correct. Spikes drawn from the held-out model rate have the same bias, so observed minus that "model + noise" peak is the miss. A miss that is there with no flash is an exit transient; a miss only with a flash is a change of the flash response.
- **Finding:** in the 38 flash-excited units, the miss is near 0 for flashes in the port and for the 2nd and later flashes after the exit. It is **+0.93 / +0.65** (p = 0.006 / 0.007) for the first flash after the exit, about 50 ms after the exit. In trials with no flash at that time, the model predicts the rate (miss **−0.64 / +0.13**). The miss with a flash minus with no flash is **+1.46 / +0.26** (p < 0.001 / 0.073); pooled **+0.73**, 95% interval **0.28 to 1.11**. The observed first response peaks earlier and is sharper than the model response (panel b).
- **Next:** this is a characterization, not a model term. The flash × after-exit kernel has one shape for all flashes after the exit, so the model shows where a fixed flash kernel stops being correct: a short, earlier and larger response at movement onset. A term built from this miss would be chosen on the data that found it ([decision log](glm_decisions.md#first-flash-after-the-exit-candidate-2026-10-21)). A term for it must have its form set in advance (for example, flash gain as a function of time since the exit, or of movement) and be tested on data that did not choose it. There are only 27 / 21 no-flash trials, and their pseudo flash comes later in the stimulus train.

![Model miss by flash rank after the exit, and with and without a flash at the same time after the exit](../figures/glm/first_flash_after_exit.svg)

## Do the conclusions depend on the video model?

### Does raw video improve held-out spike prediction?

- **Question:** do raw pixels contain predictive information beyond motion energy?
- **Why:** if they add information, we must check whether the model assigns that information to task terms when raw video is absent.
- **Finding:** adding 10 raw-video PCs to 10 motion-energy PCs improves median paired D² by **+0.00046 / +0.00109** across all **243** eligible units and the same ten folds. Prediction improves in **116/167 / 61/76** units. Pooled gain is **0.00069 D²**, 95% interval **0.00035–0.00151**. Coefficients and penalties are re-estimated; video preprocessing is not fully nested.
- **Next:** better prediction does not establish more accurate task weights. Check their levels, repeatability, and recovery below.

![Paired held-out prediction scores and gains from added raw video](../figures/glm/raw_video_full/raw_video_test.svg)

### Does raw video change the estimated size of fixation suppression?

- **Question:** how much do the signed task levels shift when raw-video terms enter the model?
- **Why:** similar kernel shapes can still have different sizes. The fixation result depends on both.
- **Finding:** median fixation levels become less negative: **−0.382 → −0.294 / −0.516 → −0.396 Δ log rate**. Pooled paired change is **+0.062**, 95% interval **0.051–0.089**. Fixation and response-period changes pass Holm correction; signed turn and flash × exit changes do not. Bars show fold SD on both axes, not independent confidence intervals.
- **Next:** the magnitude is sensitive to video modeling. Check whether adding video improves repeatability or resolves estimation bias before choosing between the weight estimates.

![Paired signed task-kernel levels with and without raw video](../figures/glm/raw_video_full/raw_video_kernel_levels.svg)

### Does better prediction give more repeatable task weights?

- **Question:** do kernels fitted to alternate trials agree more after adding raw video?
- **Why:** extra predictors can improve spike prediction while making the allocation among task terms less stable.
- **Finding:** among all eligible units, fixation agreement falls **0.872 → 0.851 / 0.936 → 0.900**; both changes pass correction for six tests per mouse (p = 1.2 × 10⁻¹⁰ / 5.5 × 10⁻⁵). Turn agreement rises **0.552 → 0.610 / 0.664 → 0.706**; only GRB006 passes correction. This all-unit selection differs from the needed-term selection in the main model checks.
- **Next:** prediction and repeatability do not pick one model uniformly. Simulations with known weights test the remaining bias.

![Paired split-half task-kernel agreement with and without raw video](../figures/glm/raw_video_full/split_half.svg)

### Does raw video resolve the known recovery and leak problems?

- **Question:** does adding raw video improve recovery of known fixation weights or settle the corrected flash × exit effect?
- **Why:** repeatable weights can still be biased. Recovery and zero-effect simulations address that limitation directly.
- **Finding:** in the fixed **20-unit pilot per mouse**, fixation-level recovery changes **0.763 → 0.753 / 0.842 → 0.759**. Largest condition indices rise **8.07 → 15.02 / 9.78 → 11.63**, below the chosen limit of 30. Corrected flash × exit kernel medians remain opposite: **+0.102 / −0.164 Δ log rate**. Each model has its own simulated truth and spikes. Only **three / five** pilot units are classified as excited, so their matched peaks cannot settle the population result.
- **Next:** retain raw video as a sensitivity control. It does not resolve fixation-weight bias; the small pilot cannot resolve the sensory question.

![Raw-video pilot collinearity, recovery, and leak checks](../figures/glm/raw_video_trust.svg)

## Check: does the turn term take the side preference?

### Does removing the fitted turn contribution remove the predicted preference?

- **Question:** how much of the observed left–right firing preference is carried by the fitted turn term?
- **Why:** activity after exit can differ by response side as well as task period. That side-linked component is a separate question from flash-response gain. The turn term was added because the raw GRB006 rates showed this preference, and a side-signed term after the exit is built to fit it. So a close agreement is expected: this checks the fit, and it is not an independent finding.
- **Finding:** observed versus predicted side-index agreement falls from **ρ = 0.990 / 0.992** to **0.521 / 0.790** when the fitted turn contribution is removed. Median absolute error rises from **0.009 → 0.146 / 0.006 → 0.154**. Keeping turn lowers error in **161/167 / 74/76 units**; pooled paired error reduction is **0.122**, 95% interval **0.105–0.147**. This removes one contribution without refitting; other terms and observed spike history remain.
- **Next:** test whether raw-video components absorb this contribution. The current comparison shows fitted allocation, not unique information or a causal effect of turning.

![Observed side preference with the full prediction and without the fitted turn contribution](../figures/glm/side_preference.svg)

### Does the fitted turn contribution survive adding raw video?

- **Question:** does the side-preference result depend on omitting raw-video components?
- **Why:** motion energy captures frame changes. Raw pixels can carry additional position, posture, and illumination signals that share information with response side.
- **Finding:** with 10 added raw-video components, full-model agreement is **ρ = 0.988 / 0.992** and falls to **0.348 / 0.505** without turn. Median error rises from **0.010 → 0.152 / 0.009 → 0.187**; pooled paired error reduction is **0.128**, 95% interval **0.104–0.170**. The fitted turn contribution remains large, although its allocation among shared signals changes.
- **Next:** the fitted term takes much of the preference under both video models, as it was built to. Linking it to choice decoding remains open. This is a check of the turn term; it does not settle the sensory-response question.

![Side preference after adding raw video, with and without the turn contribution](../figures/glm/raw_video_full/side_preference.svg)

## Methods and reproduction


### Reading the comparisons

- Fixation/exit kernels: median fixation levels are −0.35 / −0.50 Δ log rate; 89% / 82% are negative. Fitting fixation last gives −0.26 / −0.22. Five extreme finite two-step estimates are outside the plot but remain in the statistics.
- Onset control: long-delay pre-flash rate / pre-entry rate is 0.94 / 0.70 (p = 0.017 / 5 × 10⁻⁷). Zero ratios remain; tests use ratio minus 1. Approach and entry are unresolved.
- Early withdrawal: after-exit rate, early/completed, is 1.01 / 0.98. GRB064 has a separate 78-unit selection for this model-free control.
- Fixation versus flash importance: ρ = −0.084 / −0.194 (p = 0.28 / 0.093); there is no clear correlation.
- Main-model checks: largest group GVIF^(1/p) is 3.4 / 3.9; largest condition index is 8.4 / 9.9. Split-half agreement among units that need each term is 0.72 / 0.74 for turn and 0.90 / 0.94 for fixation. Agreement is 1 − ‖a − b‖² / (‖a‖² + ‖b‖²). Read spike-history kernels only in units firing at least 5 spikes/s.
- Leak correction: three simulations per unit, with the raw flash × exit regressor at zero and the in-port baseline held fixed. Corrected kernel p = 0.0068 / 1.0 × 10⁻⁶. Panels g–i use the measure of the main comparison (peak change over the mean rate of the unit; held-out real part) and show −3 to 4: two GRB006 units are outside g and two outside h; none in GRB064. All values enter tests and intervals.
- Side index: (right rate − left rate) / (right rate + left rate), from 100 ms after exit to response entry; require at least 50 ms per trial. Positive means right preference. Trial-bootstrap intervals are shown on both axes, conditional on saved predictions.

### Data

- Units: quality criteria 1, stability 0; no sensory-response filter for the GLM.
- Exclude GRB006 unit 212 for unstable out-of-fold history extrapolation; GRB064 units 414 and 437 have no validation spikes.
- Trials: completed left/right responses, without early withdrawal; require initiation, measured first flash, exit, and response entry in that order.
- Missing required events exclude a trial; invalid event order stops preparation.
- Align to the **first measured flash**. Use −0.5 to +2.1 s at 1 ms; mask bins after response entry.
- The train has 25 slots of 40 ms, starts with a flash, and can stop at response entry.
- Event-kernel baseline: before center entry. Scoring null: constant rate from the corresponding fit trials.
- Sampling unit: neural unit; side-index intervals resample trials. Units share trials.
- Population summaries pool units; bootstrap intervals resample mice, then units. Tests are within mouse.

### Design

- Count model: Poisson(exp(intercept + Xβ)); all terms add in log rate.
- **Flash:** 8 causal raised cosines, 0–150 ms; base response after a long gap, in port.
- **Flash × short gap:** the same 8 bases for gaps ≤50 ms.
- **Flash × after exit:** the same 8 bases on flashes after exit; zero in port.
- **Pre-stimulus:** 1 column from center entry to first flash; code key “fixation.”
- **Fixation:** 10 log-spaced cosines, log(t + 50 ms), to 700 ms; hold the last value until exit; code key “hold.”
- **Response period:** 8 log-spaced cosines, log(t + 25 ms), from exit to 1.3 s; hold the last value until response entry.
- **Turn direction:** those same 8 exit bases × response side, −1 left / +1 right.
- **Previous choice:** 1 signed column through center exit; zero for unknown side.
- **Motion energy:** 10 contemporaneous PC scores.
- **Session drift:** 3 columns from 4 raised cosines; omit one because the model has an intercept.
- **Self-history:** 10 log-spaced bases at strictly past lags, 1–100 ms.
- Right-response kernel = response period + turn; left-response kernel = response period − turn.
- No go-cue, outcome, punishment, response-entry, same-bin history, cross-unit coupling, or pose terms.
- **Question:** what time courses can the model represent?
- **Why:** kernel timing and duration are constrained by the chosen bases.
- **Finding:** the diagram shows the causal flash, fixation, exit, and history bases used by the current model. It describes model capacity, not a measured neural response.
- **Next:** judge the fitted kernels within these constraints; the design comparison below checks whether changing them improves prediction.

![Temporal basis functions for the current GLM](../figures/glm/basis_functions.svg)

### Video and fit

- Use synchronized camera edges in decoded-frame order; do not use container times.
- Resize grayscale frames to 80 × 64. Motion energy = absolute difference of adjacent decoded frames; zero at gaps.
- Fit 200 PCA axes on original training frames; use the first 10. Raw-video control uses grayscale pixels and adds 10 PCs.
- Scale common columns on valid original training bins; scale history separately on each fit's training rows.
- Initial whole-trial split: 60/20/20, seed 20260912. Validation compares with/without motion energy.
- Ten whole-trial CV folds, seed 20260913; each fold selects its own penalty using inner validation.
- One L2 penalty per unit/model, from 10⁻⁶ to 10⁴ in powers of ten; intercept unpenalized.
- Fit stronger penalties first, retry failed warm starts, and break loss ties toward the stronger penalty.
- Final refit: all valid trials, using the median fold penalty.
- Held-out scores and predictions come from saved fold models; predictions condition on observed spike history.
- A fold with no test spikes is skipped; require at least two scored folds. Report mean and SEM over scored folds.
- Common PCA and scaling are fixed across folds, so these scores do not validate the whole preprocessing procedure.

### Unique explained deviance and inclusion

- Unique ΔD² = full-model D² − D² with one block shuffled.
- Maximal ΔD² = D² with only that block intact − all-shuffled D².
- Refit comparators within the same ten trial folds, at each fold's selected penalty.
- Shuffle valid bins within trial; keep a block's columns together.
- Drift swaps between trials. Side terms take another trial's actual side code while keeping their own timing.
- Shuffle side terms in **raw columns before centering/scaling**; retain unknown codes and zero support.
- Motion-energy control swaps the video of another trial at the same within-trial time.
- Inclusion requires positive population gain (one-sided Wilcoxon) and more units with reliable gain than the nominal 5% chance rate (binomial test).
- A unit “needs” a term when its fold gains pass a one-sided t-test at p < 0.05. Apply Holm correction across regressors to the population tests.
- All 11 terms pass in both mice. Turn is needed by **83/167 / 45/76** units; previous choice by **31/167 / 16/76**.
- Shared training trials and shared neural trials make these tests approximate.

### Design choices

- **Question:** does a more detailed temporal design improve prediction enough to retain it?
- **Why:** extra bases or gap classes can add flexibility without useful predictive gain.
- **Finding:** paired comparisons on four fixed 20-unit GRB006 samples retain 150-ms flash kernels, a short-gap modifier, the held 700-ms fixation kernel, and the exit-aligned response period. Finer gap classes, more video PCs, and longer exit windows give no clear improvement. Recovery and collinearity also inform the choice. GRB064 takes no part in the choice of forms. It did take part in the choice of terms: the candidate tests of 2026-10-15 used both mice, and flash × motion energy stayed out because it failed in GRB064. So GRB064 is held out for the forms, not for the set of terms; only a third mouse can test the set of terms.
- **Next:** keep this design fixed for the comparisons above. Earlier choices are in [the decision history](glm_decisions.md).

![Design-choice grids on 80 GRB006 units](../figures/glm/GRB006_20240821_121447/design_optimization.svg)

### Run and inspect

- Source: [GLM modules](../src/thesis/ephys/analyses/glm/).
- Use cached trial/unit tables and low-priority fits:

~~~bash
export THESIS_CACHE_DIR=figures/cache
nice -n 10 uv run glm prepare
nice -n 10 uv run glm fit --units all --workers 2
nice -n 10 uv run glm unique --units all --workers 2
~~~

- Defaults are GRB006. Repeat with --subject GRB064 --session 20260916_132939.
- Before first prepare, supply frame_times.npy and one local cam0 video. The camera-pulse audit helper is specific to GRB006.
- Sample mode uses 20 fixed units, seed 20260914; sample1, sample2, … use disjoint groups.
- Outputs: `figures/glm/<subject>_<session>/`. Restart records: `all_fit_me/checkpoints/`.
- Preserve affected designs/checkpoints before changing inputs. Design reuse checks do not detect every basis or input change.
- Figures read completed fits and control arrays. The main document uses flash_movement (observations and matched held-out rate predictions) and main_figure; the other commands below draw supporting views:

~~~bash
THESIS_CACHE_DIR=figures/cache nice -n 10 uv run python -m thesis.ephys.analyses.glm.flash_movement
uv run python -m thesis.ephys.analyses.glm.main_figure
uv run python -m thesis.ephys.analyses.glm.hold_effect
uv run python -m thesis.ephys.analyses.glm.flash_kernels  # unit kinds for flash_by_gap; no figure
uv run python -m thesis.ephys.analyses.glm.flash_by_gap
THESIS_CACHE_DIR=figures/cache nice -n 10 uv run python -m thesis.ephys.analyses.glm.first_flash_after_exit
uv run python -m thesis.ephys.analyses.glm.model_trust
~~~

- Control arrays come from collinearity, parameter_recovery, kernel_reliability, and leak_test; hold_effect also reads the model-free onset, exit, and motion controls.
- Raw-video comparison and side preference:

~~~bash
nice -n 10 uv run python -m thesis.ephys.analyses.glm.raw_video_full --workers 2
nice -n 10 uv run python -m thesis.ephys.analyses.glm.raw_video_trust --workers 2
nice -n 10 uv run python -m thesis.ephys.analyses.glm.side_preference
nice -n 10 uv run python -m thesis.ephys.analyses.glm.side_preference --raw-video
~~~

- Run the synthetic check after design, fit, or shuffle changes; use a new output directory:

~~~bash
nice -n 10 uv run python -m thesis.ephys.analyses.glm.synthetic_check --out <new-directory>
~~~

- JSON reports beside the figures retain exact statistics and unit IDs; generated caption files give panel details.
- Earlier designs, retired tests, and penalty checks: [glm_decisions.md](glm_decisions.md).

### Related methods

- [Truccolo et al. 2005](https://pubmed.ncbi.nlm.nih.gov/15356183/) and [Pillow et al. 2008](https://sites.stat.columbia.edu/liam/research/pubs/pillow-nature-08.pdf): spiking GLMs with self-history.
- [Talluri et al. 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10620084/) and [Stringer et al. 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6525101/): video-based behavioral predictors.

### What information do the raw-video components contain in each mouse?

- **Question:** can the extra predictive information be assigned to movement alone?
- **Why:** otherwise a raw-video control could be read as a specific locomotion control.
- **Finding:** the first ten components explain **74.3% / 40.3%** of training-frame variance. The maps below show the GRB006 back view, then the GRB064 bottom view. Components mix posture, position, and illumination; brightness correlations reach **|ρ| = 0.64 / 0.49**. The two views are different measures.
- **Next:** interpret added video as a broad nuisance control. Identifying locomotion, posture, or illumination requires a more specific measurement; there is no useful pupil measurement here.

![GRB006 raw-video spatial components, back view](../figures/glm/GRB006_20240821_121447/raw_video_test/raw_components.svg)

![GRB064 raw-video spatial components, bottom view](../figures/glm/GRB064_20260916_132939/raw_video_test/raw_components.svg)

### Which checks remain incomplete?

- Calibration and weaker-fixation-penalty checks are not fully updated for the current design. Their older composite is omitted from the current figure sequence.
- The synthetic pipeline check passes. It does not resolve the known fixation-weight bias or validate the new kernel-peak measure against leak.
- Two mice with one session each do not establish replication across animals. There is no useful pupil measurement, and video does not isolate locomotion, posture, or illumination.
