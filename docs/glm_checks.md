# V1 GLM: validation and supporting results

- **Question:** which checks support the comparison in [the main analysis](glm.md), and what remains uncertain?
- **Why:** predictive accuracy, repeatability, recovery, and control comparisons answer different questions. They do not all establish that a fitted flash effect is independent of locomotion.
- **Finding:** prediction is useful, but known-weight recovery and zero-effect simulations show limits on reading the weights. Raw video does not remove those limits.
- **Next:** use these checks to qualify the three main figures. Side preference is a check of the turn term, not part of the main sensory-enhancement argument.
- **Scope:** GRB006 / GRB064 order; one session per mouse. Model of 2026-10-24, with flash recovery at τ = 50 ms (`dmat.FLASH_RECOVERY`). The raw-video control and the design grids ran on the design of 2026-10-17, without flash recovery. The held-out rate comparison of the main analysis is leak-corrected in its own measure; the supporting kernel-peak scatter is not.

## Does the fit support interpretation of the flash component?

### What do the isolated flash-kernel peaks show?

- **Question:** how do the flash-only kernel peaks differ between stationary and movement periods at the same preceding gap?
- **Why:** these show the fitted flash component directly, while the main comparison measures its contribution to peak rates on the actual trials.
- **Finding:** mean kernel-peak changes in the flash-excited group are **+0.144 / +0.034 Δ log rate** at long gaps and **+0.072 / −0.066** at short gaps. Open circles show the mean of all **172 / 79** eligible units; filled circles show the same **30 / 15 flash-excited units** in both gap panels. These are final-fit kernel peaks, with the recovery term at the gap of each condition; bars show SEM across units on both axes.
- **Next:** use the main held-out rate comparison to connect this component to the observed enhancement. Kernel peaks cannot be directly subtracted from peak rates, and their bias needs checking with this exact measure.

![Supporting flash-kernel peak comparison at matched long and short gaps](../figures/glm/flash_model.svg)

<a id="flash-exit-bias"></a>

### Do the flash weights agree with the StimulusResponse table?

- **Question:** the table calls a unit flash-excited from its raw response to the first flash of each trial (30–120 ms against −40–0 ms). Do the units it calls excited have positive flash weights in the model?
- **Why:** a sanity check of the model (Joao Couto's suggestion). Both measures answer "does this unit respond to the flash?" by different methods.
- **Finding:** units: the 172 / 79 GLM units. Data: the log rate ratio, recomputed from the spikes; it equals the stored `rate_delta` within 0.0001 spikes/s.
  - **(a) Flash weights** (the flash kernel at the first flash, 30–120 ms): they separate the table's excited units from the other units (AUC **0.90** [0.84, 0.95] / **0.98** [0.94, 1.00]); Spearman ρ **0.62 / 0.81**. Only 3 of the 45 excited units (GRB006 units 7 and 198, GRB064 unit 16) have weights ≤ 0. The weights are smaller than the data response: the model gives a part of the first-flash response, mostly the drop in rate, to the fixation kernel, which starts at the same flash.
  - **(b) All terms locked to the first flash** (flash kernel + fixation kernel − pre-stimulus step) fall near the identity line: ρ **0.87 / 0.94**, AUC **0.93 / 0.94**.
  - So the model reproduces the first-flash response, and the flash weights carry its sign. The size is shared with the fixation kernel (the known leak; [penalty](#does-the-flash--exit-result-depend-on-the-penalty-of-the-fixation-kernel)).
  - The table calls **88 / 36** units suppressed at the first flash, which is the fixation drop. With a 95% fold interval as the rule, the model calls about half of the table's excited units "not responsive" (15 of 30 / 7 of 15). The rule is stricter, not opposite: no excited unit is suppressed by the flash weights.
- **Next:** none for the model.

![Model flash response against the data first-flash response, by StimulusResponse class](../figures/glm/flash_weight_check.svg)

### Can the fit create an apparent flash × exit effect?

- **Question:** can a fitted flash × exit effect appear when it is absent from simulated spikes?
- **Why:** a repeated flash kernel can resemble a sustained rate change. The penalty can assign background changes to flash terms.
- **Finding (pairs of 2026-10-25, [result 1](glm.md)):** in the measure of the main comparison, the leak is **−0.55 / +0.02** normalized at long gaps in the excited units, against a held-out contribution of **+0.31 / −0.11**; corrected **+0.86 / −0.13** (p = 0.10 / 0.30), pooled **+0.07** (95% interval −0.26 to 0.27). In GRB006 the mean depends on a few units (median corrected +0.17). Corrected kernel-mean effects across all units have opposite signs (**+0.04 / −0.14 Δ log rate**, p = 0.017 / < 0.001). Known fixation levels recover at only **0.68 / 0.76** of their size, despite high repeatability. These results do not establish an unchanged sensory response.
- **Gaps of the pairs (2026-10-28):** the two flashes of a pair are not matched for their gap. At long gaps the median gap is 120 / 120 ms (GRB006) and 80 / 120 ms (GRB064), in port / after exit; the mean recovery (1 − exp(−gap / 50 ms)) is 0.88 / 0.91 and 0.87 / 0.91 (`flash_movement.json`, `pair_gaps`). The recovery differs by only 0.03–0.04 between the periods, so adaptation is not a large part of the after-exit increase. At 40 ms gaps the gaps are the same.
- **Log ratio (pairs of 2026-10-25):** the same comparison as log(after exit / in port) of the full-model peak minus the same without flash × exit (`leak_test.json`, "main comparison (log ratio)"). It does not depend on the normalization of a unit. A unit with a model peak ≤ 0 in the held-out or the final fit has no log ratio and is left out; its leak is the mean over the simulations with a log ratio, and is missing if fewer than half have one (since 2026-10-28; before, one simulation without a log ratio left the unit out). Excited units at long gaps left out: 4 of 30 / 2 of 15. In the excited units at long gaps, the held-out part is **+0.15 / −0.04** and the leak **+0.12 / +0.01**; corrected **+0.03 / −0.05** (×1.03 / ×0.95; p = 0.86 / 0.45; medians −0.07 / −0.17), pooled median **−0.08** (95% interval −0.21 to 0.00, p = 0.12). So the large GRB006 mean of the normalized measure comes from a few units; on the log scale neither mouse has a flash × exit effect at long gaps. At 40 ms gaps the corrected part is below zero: excited units pooled **−0.31** (−0.44 to −0.15, p = 0.004), all units pooled −0.14 (−0.29 to −0.07, p = 0.001), with 35 of 172 / 16 of 79 units left out; at long gaps over all units it is pooled −0.04 (−0.12 to 0.05).
- **Next:** treat the new kernel-peak scatter as descriptive until that exact measure is checked for bias. The earlier matched-rate correction cannot simply be subtracted from its kernel peaks.

![Collinearity, known-weight recovery, repeatability, and flash × exit leak correction](../figures/glm/model_trust.svg)

### Is the low rate during fixation explained by visual input or stillness?

- **Question:** does low firing follow fixation, the flash train, or low movement?
- **Why:** these occur together in completed trials. Rates alone cannot identify which accounts for the change at exit.
- **Finding:** normalized firing is **0.62 / 0.64** late in fixation and **1.37 / 1.28** after exit. Firing can fall before the first flash and rises at early withdrawal when flashes stop. Rate rises with motion energy at the side port, but not at the center port. These controls support a fixation-linked change; they do not isolate its cause. The lowest-motion comparison is inconclusive in GRB064, and early withdrawal also adds a punishment tone.
- **Next:** retain a task-period rate component when estimating flash responses. Panels a–f show the rate change and its timing; g–j check fitting order, movement, and which units show it.

![Fixation, exit, onset, and movement controls](../figures/glm/hold_effect.svg)

### Does the model reproduce the observed flash-response time courses across gaps?

- **Question:** does the model capture the traces, beyond the peak summaries in the main comparison?
- **Why:** a peak can agree while response timing or shape is wrong. Gap-resolved traces check this part of the flash model.
- **Finding:** observed and fitted matched peaks are higher after exit across the gap groups in both mice. The traces compare timing and shape within gap and task period (black: observed; wine: predicted). They include background and preceding-flash responses and use the final refit, so this is an in-sample check, not an isolated sensory-gain estimate.
- **Next:** use this check to assess the flash-history model; use the simulation results above to qualify an exit-effect interpretation.

![Observed and predicted flash-response time courses by gap and task period](../figures/glm/flash_by_gap.svg)

### Which flash response does the model miss after the exit?

- **Question:** the full model predicts less long-gap enhancement than observed. Which flashes does it miss, and does the missing part need a flash?
- **Why:** an observed peak is the maximum of a noisy mean, so it is larger than the peak of a smooth predicted rate even when the model is correct. Spikes drawn from the held-out model rate have the same bias, so observed minus that "model + noise" peak is the miss. A miss that is there with no flash is an exit transient; a miss only with a flash is a change of the flash response.
- **Finding:** in the 45 flash-excited units (model with flash recovery), the miss is near 0 for flashes in the port and for the 2nd and later flashes after the exit. It is **+0.77 / +0.58** (p = 0.001 / 0.008) for the first flash after the exit, about 50 ms after the exit. In trials with no flash at that time, the model predicts the rate (miss **−0.65 / +0.20**). The miss with a flash minus with no flash is **+1.56 / +0.12** (p < 0.001 / 0.08); pooled **+0.70**, 95% interval **0.15 to 1.14**. The observed first response is sharper than the model response (panel b).
- **Next:** the gap and the state account for part of this; the next section separates them. A term for the first flash after the exit was tested and not adopted, because it was made from this miss ([decision log](glm_decisions.md#first-flash-after-the-exit-candidate-2026-10-21)). There are only 27 / 21 no-flash trials, and their pseudo flash comes later in the stimulus train.

![Model miss by flash rank after the exit, and with and without a flash at the same time after the exit](../figures/glm/first_flash_after_exit.svg)

### How fast do flash responses recover from adaptation?

- **Question:** the response to a flash depends on the gap from the flash before. How long does it take to recover?
- **Why:** most flashes come 40 or 80 ms after another, so most responses are adapted. A model with one kernel for all gaps above 50 ms misses the rare long-gap flashes, and the first flash after the exit usually has a long gap.
- **Finding:** in-port flash peaks of GRB006 rise with the gap and level off by about 200 ms, at the level of the first flash of a trial. Response = A (1 − c exp(−gap / τ)), with c free (30 excited units, without the 40-ms group, where the flash before overlaps the measure), gives **τ = 54 ms** (95% 29 to 91): the curve is 0 up to about **68 ms** (τ ln c), rises fast, and is recovered by about 200 ms. GRB064 follows the same curve with τ fixed, and on its own gives 67 ms (21 to 141). Until 2026-10-24 c had a bound of 1.5 (with no recorded reason), which gave τ = 118 to 121 ms and a worse fit (χ² 11.1 against 4.0 on 4 df).
- **The τ of the model:** in the model, c goes into each unit's recovery weight, and only τ sets the shape of the term. On the 80 grid units (`tau_grid.py`, rule fixed before the run), τ = 40, 60, 80 and 160 ms predict the same as 120 ms (median change of D² below 0.0001, Holm p ≥ 0.72). So prediction does not choose τ, and the model uses the τ of the fit with c free, rounded to 10 ms: **τ = 50 ms** (from 2026-10-24; 120 ms before). A second run with 50 ms as the reference (2026-10-28, the same 80 units) agrees: τ = 40, 60, 80, 120 and 160 ms predict the same as 50 ms (median change of D² below 0.0001, 31 to 48 of 80 units better, Holm p ≥ 0.43), so τ stays at 50 ms. With it the term passes the inclusion rule in both mice (28% / 30% of units); with 120 ms it passed in 25% / 28% and added **+0.0036 / +0.0032 bits per spike** in the flash-excited units of the inclusion test (2026-10-23, before the gain fix: 18 of 23, 13 of 15 better).
- **Next:** the term takes about a third to a half of the miss at long gaps. It shares a direction with the start of the fixation kernel (first canonical correlation 0.72 / 0.70, `model_trust.svg` panel c): the first flash of each trial is fully recovered and starts the fixation kernel.

![In-port flash peak against the gap from the flash before, with the recovery fit](../figures/glm/flash_recovery.svg)

### What changes the flash response at the exit?

- **Question:** how much of the larger flash response after the exit is a gain on all activity, how much is the gap from the flash before, and what is left?
- **Why:** with a log link, a term that raises the rate after the exit multiplies every response. If only that gain acts, the flash peak grows by the same factor as the rate before the flash.
- **Finding:** grouped by period, rank and gap (45 flash-excited units, pooled, hierarchical bootstrap): later flashes after the exit grow by the gain of the rate before them (peak ×**2.17** against ×**1.80** at 50–100 ms; ×**1.79** against ×**1.85** at 100–200 ms). The first flash after the exit grows more than that gain at 50–100 and 100–200 ms gaps (peak ×**4.83** and ×**3.01**, against ×**1.39** and ×**1.24**), but close to it at 200+ ms (×**2.01** against ×**1.85**), where the in-port response has recovered. Its peak latency is not well measured (35 to 75 ms, 95% intervals 35 to 115 ms in each gap group). The model, with recovery, still misses it at 100+ ms gaps (**+0.50** and **+0.61**).
- **Flash-suppressed units (2026-10-25):** the same test on the units whose flash weights are below 0 (`flash_weight_check.py`, 13 / 21; `exit_flash_responses.py --group flash-suppressed`, the dip below the rate before the flash in place of the peak). Divided by their own rate before the flash, their in-port PSTHs stay near 1: these units show no clear flash dip in these flashes, so the test has little to measure. The dip growth over the rate growth is 0.87 (0.62–1.10) and 1.58 (1.17–2.10) for later flashes after 50–100 and 100–200 ms gaps, and 0.50, 1.50 and 1.24 (intervals include 1) for the first flash after the exit: no consistent pattern. The StimulusResponse table calls 88 / 36 units suppressed, but it classifies from the first flash of each trial, which also starts the fixation drop. Per flash (in-port flashes at least 80 ms after the flash before, Wilcoxon over flashes, Benjamini–Hochberg 0.05 in each mouse), 18 / 3 units are suppressed and 26 / 24 excited; those 21 suppressed units, not the 13 / 21 of the flash weights, are the group for this question.
- **Next:** this fits a reset of adaptation at the exit (for example, the panel falls on a new part of the visual field when the head moves) times the state gain; the rig has no head tracking to test it. With these, the only flash-specific change at the exit is the first flash; the flash × exit term is not consistent between the mice.

![Flash peak, latency, gain test and model miss by period, rank and gap](../figures/glm/exit_flash_responses.svg)

<a id="penalty-sensitivity"></a>

### Does the flash × exit result depend on the penalty of the fixation kernel?

- **Question:** with one L2 penalty for all terms, the fixation kernel is too small and part of the rate step at the exit can go to the flash × after-exit kernel. Do the results change when the penalty on the fixation kernel alone is 100 times weaker?
- **Why:** the leak test simulates from the fit itself, so it has the same small fixation kernel and can underestimate this. A fit with the fixation kernel free to take its level is a direct check.
- **Finding:** both mice were fit again with `HOLD_PENALTY_SCALE = 10` (`penalty_sensitivity.py`, model of 2026-10-24, flash pairs of 2026-10-25). Prediction does not change (median D² 0.0730 / 0.0743 against 0.0724 / 0.0744). The fixation level of the kernel becomes **−0.69 / −0.64** (from −0.37 / −0.48), and a refit keeps **0.90 / 0.98** of a simulated level (from 0.68 / 0.76). The full-model enhancement of result 3 is the same (1.70 / 0.79), and without flash × exit it is **1.62** / 0.88 (from 1.39 / 0.90). The leak-corrected flash × exit part is **+0.83** (p = 0.42) / −0.16 (p = 0.33) in the excited units (medians +0.04 / −0.15), pooled +0.00 (−0.30 to 0.13), and +0.32 (p = 0.011) / −0.02 (p = 0.22) over all units.
- **Next:** with the fixation kernel free, the raw GRB006 flash × exit part falls (+0.31 → +0.08) but the leak-corrected part hardly changes (+0.86 → +0.83). In the excited units neither mouse has a reliable effect; over all units the GRB006 part stays positive (+0.41 → +0.32, p = 0.011) and GRB064 has none, so the mice still disagree. A penalty for each band, chosen by prediction with a rule fixed in advance (banded ridge, `band_grid.py`, 80 grid units of GRB006, design of 2026-10-17), keeps one penalty for all terms: no band factor predicts better, and a weaker penalty on the whole task band predicts the same (Holm p = 1). So prediction does not choose the size of the task kernels, and this fit stays a check ([decision log](glm_decisions.md#banded-ridge-2026-10-22-and-2026-10-23)).

### Is the predicted rate calibrated over its range?

- **Question:** is the held-out predicted rate correct where it is low and where it is high?
- **Why:** a model can predict well on average and still compress its range, which would make the size of every effect too small.
- **Finding:** held-out 1-ms bins of each unit sorted into 20 groups by predicted rate (`calibration.py`). From the 5th to the 70th percentile the model is too high (observed / predicted **0.84 to 0.98**); from the 75th to the 95th it is too low (**1.03 to 1.13**). Both mice have the same curve, and the fit with the weaker fixation penalty has the same curve.
- **Next:** the model compresses its range by about 10%, so its effect sizes are a little too small; this does not depend on the penalty of the fixation kernel. The cause is not known: one gain on the linear predictor and a cubic output function were tested earlier and did not help ([decision log](glm_decisions.md)).

![Observed over predicted spike count by predicted-rate percentile, with the model in use and the weaker fixation penalty](../figures/glm/calibration.svg)

## Do the conclusions depend on the video model?

These fits are of the design of 2026-10-17, without flash recovery.

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
- **Finding:** observed versus predicted side-index agreement falls from **ρ = 0.989 / 0.989** to **0.540 / 0.735** when the fitted turn contribution is removed. Median absolute error rises from **0.009 → 0.138 / 0.007 → 0.156**. Keeping turn lowers error in **167/172 / 76/79 units**; pooled paired error reduction is **0.119**, 95% interval **0.102–0.147**. This removes one contribution without refitting; other terms and observed spike history remain.
- **Next:** test whether raw-video components absorb this contribution. The current comparison shows fitted allocation, not unique information or a causal effect of turning.

![Observed side preference with the full prediction and without the fitted turn contribution](../figures/glm/side_preference.svg)

### Does the fitted turn contribution survive adding raw video?

- **Question:** does the side-preference result depend on omitting raw-video components?
- **Why:** motion energy captures frame changes. Raw pixels can carry additional position, posture, and illumination signals that share information with response side.
- **Finding:** with 10 added raw-video components, full-model agreement is **ρ = 0.988 / 0.992** and falls to **0.348 / 0.505** without turn. Median error rises from **0.010 → 0.152 / 0.009 → 0.187**; pooled paired error reduction is **0.128**, 95% interval **0.104–0.170**. The fitted turn contribution remains large, although its allocation among shared signals changes.
- **Next:** the fitted term takes much of the preference under both video models, as it was built to. Linking it to choice decoding remains open. This is a check of the turn term; it does not settle the sensory-response question.

![Side preference after adding raw video, with and without the turn contribution](../figures/glm/raw_video_full/side_preference.svg)

## Link to the decoders (2026-10-29)

`glm/decoding_without_terms.py`: a decoder on the rates 300 ms before and after the center exit, from the observed spikes, with the fitted flash part divided out, and with all other model terms divided out (each trial changed by the fold model that held it out). GLM sessions and units (172 / 79); trials balanced for category × choice; chance is the 2.5–97.5% range of a mean AUC over shuffles.

### Is the category signal around the exit visual?

- **Question:** the category is clearest just after the exit. Does it come from the flash responses, or from task and movement?
- **Finding:** without the flash terms the category AUC falls by **0.09 / 0.21** after the exit (GRB064 to 0.58, inside chance) and by **0.09 / 0.15** in the port; without all other terms it changes by **≤ 0.05**. Single units and multi-units (636 / 509) add **+0.03 / +0.04** after the exit, so the GLM units carry almost all of the decoder's category signal there (in the port of GRB064, +0.10).
- **Limits:** spike history stays in all versions; the model misses most of the first flash after the exit, so the GRB006 value after the removal (0.75) is an upper limit of a non-visual signal; "visual" means what the GLM gives to the flash terms.

![Category decoding without model terms](../figures/glm/category_without_terms.svg)

### Is the choice signal after the exit a flash response moved in the visual field?

- **Question:** after the exit the head turns, so the LED panel lies in another part of the visual field for each side. Is the choice signal (AUC 1.00 / 0.97 after the exit) a flash response that differs with the side?
- **Model term:** flash × side (`dmat.FLASH_SIDE`, a candidate, not in the model): the after-exit flash kernel times the side of the response. In `candidate_terms.py` (all units, the design in use plus this term) it **passes the inclusion rule in both mice**: 56 / 172 and 25 / 79 units need it (18 / 11 needed above chance), median held-out gain **+0.0032 / +0.0032 bits per spike** (median unique ΔD² **0.00042 / 0.00046**), Holm p 2e-18 / 7e-7, 32% / 32% of the units outside the selection units. The turn term keeps most of its share (unique ΔD² 0.00095 → 0.00073 / 0.00187 → 0.00171).
- **Shape:** in the units that need it, the kernel has the shape of a flash response, not of a side offset: signed mean +0.03 / −0.03 at 0–20 ms, **+0.15 / +0.20 at 60–80 ms**, +0.03 / −0.02 at 120–150 ms. So part of the side signal after the exit is locked to the flashes.
- **Decoding:** it cannot separate the two side terms. Without the flash terms (with flash × side) the choice AUC stays **1.00 / 0.97**; without the other terms (the turn term among them) it is **0.90 / 0.80** with flash × side in the model, against 0.66 / 0.60 without it. With flashes every 50–250 ms, a flash-locked side kernel summed over a 300 ms window is close to a constant side signal, so either term carries the choice.
- **Limits:** a flash response that differs with the side can come from the new position of the panel in the visual field, or from a side-specific gain of the flash response; the data have no head or eye position to tell them apart. Flash × side is not in the model; adding it changes the category results by ≤ 0.01.

![Choice decoding without model terms](../figures/glm/choice_without_terms.svg)

## Methods and reproduction


### Reading the comparisons

- Fixation/exit kernels: median fixation levels are −0.35 / −0.50 Δ log rate; 89% / 82% are negative. Fitting fixation last gives −0.26 / −0.22. Five extreme finite two-step estimates are outside the plot but remain in the statistics.
- Onset control: long-delay pre-flash rate / pre-entry rate is 0.94 / 0.70 (p = 0.017 / 5 × 10⁻⁷). Zero ratios remain; tests use ratio minus 1. Approach and entry are unresolved.
- Early withdrawal: after-exit rate, early/completed, is 1.01 / 0.98. GRB064 has a separate 78-unit selection for this model-free control.
- Fixation versus flash importance (unique bits per spike of the flash terms, with recovery): ρ = −0.03 / −0.18 (p = 0.70 / 0.11); there is no clear correlation.
- Main-model checks: largest group GVIF^(1/p) is 4.2 (limit 5); largest condition index is 10.5 (limit 30). Split-half agreement among units that need each term is 0.72 / 0.77 for turn, 0.89 / 0.94 for fixation and 0.67 / 0.70 for flash recovery. Agreement is 1 − ‖a − b‖² / (‖a‖² + ‖b‖²). Read spike-history kernels only in units firing at least 5 spikes/s.
- Leak correction: three simulations per unit, with the raw flash × exit regressor at zero and the in-port baseline held fixed. Corrected kernel p = 0.039 / < 0.001. Panels g–i use the measure of the main comparison (peak change over the mean rate of the unit; held-out real part). All values enter tests and intervals.
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
- **Flash × recovery:** the same 8 bases, scaled by 1 − exp(−gap / 50 ms) minus its mean over the long-gap flashes of the session, for gaps above 50 ms (the first flash of a trial is fully recovered); zero for short gaps. The design metadata stores the center (`flash_recovery_center`).
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

- Unique ΔD² = full-model D² − D² with one block shuffled. It is reported in held-out bits per spike: in a fold, ΔD² times the full model's bits/D² of that fold (`unique_dev.add_bits_per_spike`; the factor depends only on the data). The inclusion decisions are the same in either measure.
- Maximal ΔD² = D² with only that block intact − all-shuffled D².
- Refit comparators within the same ten trial folds, at each fold's selected penalty.
- Shuffle valid bins within trial; keep a block's columns together.
- Drift swaps between trials. Side terms take another trial's actual side code while keeping their own timing.
- Shuffle side terms in **raw columns before centering/scaling**; retain unknown codes and zero support.
- Motion-energy control swaps the video of another trial at the same within-trial time.
- Inclusion requires positive population gain (one-sided Wilcoxon) and more units with reliable gain than the nominal 5% chance rate (binomial test).
- A unit “needs” a term when its fold gains pass a one-sided t-test at p < 0.05. Apply Holm correction across regressors to the population tests.
- All 12 terms pass in both mice. Turn is needed by **95/172 / 48/79** units; previous choice by **35/172 / 16/79**; flash recovery by **49/172 / 24/79**.
- Shared training trials and shared neural trials make these tests approximate.

### Design choices

- **Question:** does a more detailed temporal design improve prediction enough to retain it?
- **Why:** extra bases or gap classes can add flexibility without useful predictive gain.
- **Finding:** paired comparisons on four fixed 20-unit GRB006 samples retain 150-ms flash kernels, a short-gap modifier, the held 700-ms fixation kernel, and the exit-aligned response period. Finer gap classes, more video PCs (also 200 PCs with a 100 or 1000 times stronger video penalty; [decisions](glm_decisions.md#200-motion-energy-pcs-with-a-video-penalty-2026-10-24)), and longer exit windows give no clear improvement. Recovery and collinearity also inform the choice. GRB064 takes no part in the choice of forms. It did take part in the choice of terms: the candidate tests of 2026-10-15 used both mice, and flash × motion energy stayed out because it failed in GRB064. So GRB064 is held out for the forms, not for the set of terms; only a third mouse can test the set of terms.
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
- Outputs: data (fits, designs, caches, result tables) in `results/glm/<subject>_<session>/` and `results/glm/`; figures and captions at the same paths under `figures/glm/`. Restart records: `all_fit_me/checkpoints/`. The trial and unit cache stays in `figures/cache` (`THESIS_CACHE_DIR`).
- Preserve affected designs/checkpoints before changing inputs. Design reuse checks do not detect every basis or input change.
- Figures read completed fits and control arrays. The main document uses flash_movement (observations and matched held-out rate predictions) and main_figure; the other commands below draw supporting views:

~~~bash
THESIS_CACHE_DIR=figures/cache nice -n 10 uv run python -m thesis.ephys.analyses.glm.flash_movement
uv run python -m thesis.ephys.analyses.glm.main_figure
uv run python -m thesis.ephys.analyses.glm.hold_effect
uv run python -m thesis.ephys.analyses.glm.flash_kernels  # unit kinds for flash_by_gap; no figure
uv run python -m thesis.ephys.analyses.glm.flash_by_gap
THESIS_CACHE_DIR=figures/cache uv run python -m thesis.ephys.analyses.glm.flash_weight_check
THESIS_CACHE_DIR=figures/cache nice -n 10 uv run python -m thesis.ephys.analyses.glm.first_flash_after_exit
THESIS_CACHE_DIR=figures/cache nice -n 10 uv run python -m thesis.ephys.analyses.glm.penalty_sensitivity  # refits both mice
THESIS_CACHE_DIR=figures/cache nice -n 10 uv run python -m thesis.ephys.analyses.glm.calibration
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

- The synthetic pipeline check passes. It does not resolve the known fixation-weight bias or validate the new kernel-peak measure against leak.
- Two mice with one session each do not establish replication across animals. There is no useful pupil measurement, and video does not isolate locomotion, posture, or illumination.
