# Movement and sensory responses in V1

- **Question:** how does movement affect sensory responses in V1?
- **Thread:** establish the apparent enhancement → model the overlapping signals → compare the flash-related component between stationary and movement periods.
- **Scope:** these are task periods before and after center exit, not a direct locomotion measurement. GRB006 / GRB064 values below use that order. The matched held-out comparison uses normalized linear scatters as of 2026-10-21. Model validation, video controls, side preference, and full methods are in [the supplement](glm_checks.md).

## 1. Are flash responses larger during the movement period?

- **Question:** do flash-aligned firing rates and peaks increase after the mouse leaves the center port?
- **Why:** the PSTHs establish the observation; the paired scatter tests whether it extends across units. They belong in one figure.
- **Finding:** the matched comparison includes **23 / 15** stable flash-excited units. At long gaps, **119 / 56** pairs give mean normalized peak increases of **2.63 / 1.34**. At short gaps, **49 / 41** pairs give **0.010 / 0.015**, with median zero in both mice. Panels a–b show PSTHs pooled over these pairs; c–d show normalized long-gap and short-gap scatters. Both peaks use the stationary pre-flash baseline within their gap class, so a background-rate change remains.
- **Next:** this apparent enhancement could reflect a changed flash response, another signal that changes firing, or both. Build an encoding model to estimate these contributions together.

![Observed flash responses: population PSTHs and paired stationary/movement peak scatters for long and short preceding gaps](../figures/glm/flash_observation.svg)

## 2. Can an encoding model account for the overlapping signals?

- **Question:** can flash history, task events, response side, video, and spike history account for the firing?
- **Why:** these signals overlap during movement. A flash-aligned average cannot separate their fitted contributions.
- **Finding:** the Poisson GLM has **11 regressors, 75 columns, and 1-ms bins**. Median held-out D² is **0.072 / 0.075** relative to constant-rate prediction. All 11 terms pass both inclusion tests. The model contains a common flash kernel, a short-gap modifier, and a flash × after-exit modifier, alongside task and video terms.
- **Next:** compare observed and held-out predicted peaks on the same flashes, then remove flash × exit while keeping the other fitted contributions fixed. This connects the fitted term to the measured enhancement; predictive accuracy alone does not establish unique separation.

![Encoding-model design, example prediction, fitted kernels, and population prediction scores](../figures/glm/main_figure.svg)

## 3. Is the enhancement independent of locomotion?

- **Question:** how much of the peak enhancement is carried by the fitted flash × exit term, after accounting for the other modeled signals?
- **Why:** compare observed peaks, held-out full-model peaks, and peaks after removing only that term. All three columns use the same units, paired flashes, complete windows, baseline rule, peak calculation, and observed normalization rate per unit. Rows separate long and short preceding gaps.
- **Finding:** at long gaps, mean normalized enhancement is **2.63 / 1.34 observed**, **1.89 / 0.95 in the full model**, and **1.23 / 1.12 without flash × exit**. Most of the full-model enhancement remains after removal. The term contributes **+0.67 / −0.17** on average; its contribution is not consistently positive across mice. The full model also underpredicts the observed long-gap enhancement. The fit makes little of this term from nothing: in simulations with no flash × exit effect, the same measure gives **+0.04 / +0.03**. With that leak removed, the excited-unit contribution is **+0.63 / −0.20** (p = 0.32 / 0.33); pooled median **+0.11**, 95% interval **−0.48 to 0.21**. At short gaps it is **+0.00 / −0.07**. Over all units, the long-gap contribution is positive in GRB006 (median +0.19, p < 0.001) and not in GRB064 (−0.04, p = 0.55).
- **Next:** the leak does not explain the term, but its sign differs between the mice; a third mouse is needed. The prediction error is not spread over the movement period: the model misses only the [first flash after the exit](glm_checks.md#which-flash-response-does-the-model-miss-after-the-exit), in both mice, and only when a flash is there. The comparison measures the allocation to one fitted term; it does not establish a causal effect or independence from locomotion. Other signals, including observed spike history, remain fixed and can retain the enhancement.

![Matched flash peaks: observed, held-out full model, and the same prediction without flash × exit; rows show long and short preceding gaps](../figures/glm/flash_comparison.svg)

## How to read the comparison

- All three columns use 10-ms bins, a stationary baseline from −40 to 0 ms, and separately selected peaks from 30–120 ms. Each source uses its own stationary baseline and subtracts that same baseline from both task periods. The scatter divides these peaks by each unit's **mean observed rate over the full valid model interval of the same paired trials**, pooling spike counts and valid seconds across gap groups. The same denominator scales every source and condition. PSTHs remain in spikes/s; unchanged absolute peaks and spikes/s summaries remain in the report.
- Flashes are aligned to the nearest bin on the model's native 1-ms grid. Only completed GLM trials with both full response windows inside its valid interval enter any column. This drops one long-gap GRB006 pair from the previous raw comparison. Native-grid alignment also causes small changes from the earlier flash-centered histograms.
- The model columns use saved whole-trial held-out fits. Spike history is scaled using each fold's fitting rows. Removing the raw flash × exit contribution preserves the column-centering offset; no coefficients are refitted.
- Red is GRB006; blue is GRB064. Open circles show the mean of all **167 / 76** eligible units; filled circles show the mean of the saved **23 / 15 flash-excited units**. Both use alpha 0.8 and **SEM across units on both axes**. The population includes the flash-excited units. No individual units are drawn. PSTHs and the numerical findings above still use only the flash-excited group. Scatter means summarize normalized peaks, so units contribute relative to their own mean rates.
- Per-unit uncertainty remains in the report: paired-trial bootstrap SD, using the same draws across columns and recomputing the observed full-trial normalization rate in every draw. These estimates are conditional on the saved models. The displayed group SEM bars describe variation across units; they do not propagate trial-sampling or model-fitting uncertainty, and the two nested groups are not independent.
- Rate scatters use common **linear axes**, with limits set by the group means and full SEM bars. Every unit enters its group mean, including zero and negative peaks. A normalized value of 1 means a baseline-subtracted peak equal to that unit's pooled mean observed full-trial rate. The full-trial mean stays positive for all units and bootstrap draws; a flash-window mean did not.
- Both comparisons use the model's gap split: **≤50 ms** (nominal 40 ms) versus **>50 ms** (nominal 80+ ms). Retain the last in-port and first after-exit 15-ms flash only when both have the same gap class. Gaps use the immediately preceding measured pulse of any width.
- Exclusions: **117 / 60** mixed-gap pairs, **0 / 2** pairs without a preceding pulse, and **1 / 0** pairs with an incomplete model window. Trial IDs, measured gaps, unit peaks, and fitted-term contributions are saved in the figure report.
- Raw and predicted peaks include background activity and responses to nearby flashes. The [isolated-kernel scatters](glm_checks.md#what-do-the-isolated-flash-kernel-peaks-show) are supporting views on a different scale. The leak correction (`leak_test.py`) uses this exact measure: pairs, windows, baseline, peak and denominator. The leak comes from fits to all trials of simulated spikes. Real minus leak is given for the held-out real part and for the final refit; they agree (excited units, long gaps: +0.81 / −0.23 from the final refit).
- The log-link model adds terms in log rate, so non-flash terms can scale firing without changing the flash kernel. It does not compare separate additive-rate and multiplicative-rate model families.
- Run the main and supporting flash figures from saved fits and cached trials:

~~~bash
THESIS_CACHE_DIR=figures/cache nice -n 10 uv run python -m thesis.ephys.analyses.glm.flash_movement
~~~

- [Validation, controls, related results, and full methods](glm_checks.md) · [Earlier design decisions](glm_decisions.md)
