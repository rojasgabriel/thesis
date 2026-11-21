# V1 GLM

Question: does V1 represent the flash and the movement of the mouse
separately, and do the two representations interact?

Entries before 2026-11-21 are reconstructed. The sources are the git log,
the decision history `docs/glm_decisions.md` (removed in 12371d8; permalink
https://github.com/rojasgabriel/thesis/blob/61b3af413d791575204f09d9bc0465b5438a920e/docs/glm_decisions.md),
the Notion page "Modeling V1 spikes with a Poisson GLM"
(https://app.notion.com/p/415ecf086b7c81f4aa5bdc1de417108b), and the notes of the AI agents.
The dates are the dates of the commit or the note, not always of the first
thought. "AI" is an AI agent (Claude); the model name was not recorded.
"unknown" means that the sources do not name who decided.
"Gabriel (rule of 2026-10-10)" means that the selection rule of that date made
the choice.

## 2025-05-16 · question · Gabriel
What: Fit a Poisson GLM to the V1 units of GRB006.
Evidence: 588dc387

## 2026-09-14 · decision · unknown
What: Spike history goes back into the model. The flash design is one common
kernel plus two contrasts (before/after the withdrawal, first/later flash).
Why: The model without history was a first draft, not a final design.
Evidence: commit fe27b9d; agent note "v1-glm-redesign-spec"

## 2026-10-09 · question · Gabriel
What: Use the GLM to separate the visual and the movement representations and
to test if they interact.
Why: This result must justify the model design.

## 2026-10-10 · decision · Gabriel
What: Test the form of a term (bases, window, spacing) on a fixed sample of
20 units (seed 20260914). Each test changes one thing in the final design. If
two options do not differ (paired Wilcoxon), keep the simpler one.
Why: To prevent overfitting the design to the units. The other units are a
held-out test.

## 2026-10-10 · finding · AI
What: First full design (9 regressors, 167 units, 294 trials). Median D² 0.071,
0.51 bits/spike. Spike history, flash, withdrawal, motion energy and flash
adaptation are kept. Initiation, response entry, flash × motion energy and
flash post − pre withdrawal are removed (unique ΔD² ≤ 0.0009).
Evidence: code at fcd34fa; glm_decisions.md, "The first full design"

## 2026-10-10 · decision · Gabriel (rule of 2026-10-10)
What: 10 motion-energy PCs.
Why: Validation D² is flat from 10 to 50 PCs (0.0733 to 0.0736).

## 2026-10-11 · finding · AI
What: Flash × move on all 167 units: 28 % of units need it, D² +0.0004
(p = 2e-8). On the 20-unit sample it has p = 1.0. The flash response after
the exit is about 6 % higher than the common fixation factor gives.
Evidence: flash_fixation_test.json; glm_decisions.md, "Flash × move interaction"

## 2026-10-12 · correction · Gabriel
What: Corrects 2026-10-10. Test the importance of a regressor on all units.
The 20-unit sample stays only for the form of a term.
Why: 20 units are too few (flash × move: p = 1.0 on 20, p = 2e-8 on 167).

## 2026-10-12 · decision · Gabriel
What: Choose the form of a term on 4 sets of 20 units (80 units, `--samples
4`). 87 units stay held out.
Why: One set of 20 had too little power.

## 2026-10-12 · decision · Gabriel (rule of 2026-10-10)
What: Two flash kernels: a flash within 50 ms of the flash before, and every
other flash.
Why: 2 classes beat 1 kernel (+1.46 %, 59 of 80 units, p = 1e-6). 3 or 4
classes are not better.
Evidence: gap_class_grid.py; glm_decisions.md, "Gap classes on 80 units"

## 2026-10-12 · decision · Gabriel (rule of 2026-10-10)
What: Session drift term with 4 raised cosines.
Why: It keeps slow change out of the other kernels; 4, 8, 12 cosines and the
spline do not differ.
Evidence: drift_grid.py

## 2026-10-13 · finding · AI
What: The model gives part of the fixation level to the flash × move kernels
(leak). A shared population time course does not remove the leak. A fixation
penalty 100 times weaker removes it with the same prediction.
Why: The leak biases the kernels that we want to read. Collinearity
statistics and split halves did not show it. A simulation from the fit with
the effect set to 0 showed it.
Evidence: leak_test.py

## 2026-10-13 · decision · Gabriel
What: One L2 penalty for all terms. Report flash × move after the leak
correction. Keep the weaker-penalty fit as a sensitivity analysis.
Why: The weaker penalty is not standard practice.

## 2026-10-15 · decision · Gabriel
What: Candidate test, both mice. Turn direction (exit to response entry, 8
log cosines to 1.3 s) replaces the side term. The response period replaces the
withdrawal kernel. Time in trial is out. Flash × move × gap is out. The
fixation kernel ends at 0.7 s. The window starts 0.5 s before the first flash.
Why: Time in trial has canonical r = 1.00 with the fixation kernel. Turn
direction shares 0.9 with the side term. The fixation kernel after 0.7 s has
split-half agreement 0.26 (1–3 % of trials reach it).
Evidence: candidate_terms.py; glm_decisions.md, "Design of 2026-10-15"

## 2026-10-15 · dead end · AI
What: Flash × motion energy stays out.
Why: It passes in GRB006 (32 % of units) and fails in GRB064 (1 %). The camera
views differ (back in GRB006, bottom in GRB064).

## 2026-10-16 · correction · AI
What: The over-trials shuffle of unique ΔD² moved exit-aligned time courses
between trials. A synthetic flash-only unit got turn ΔD² 0.19. Fixed: a side
term keeps its rows and takes the sign of another trial. Turn direction now
passes in 49 % and 63 % of units.
Evidence: unique_dev.py (SIGN_OVER_TRIALS)

## 2026-10-16 · decision · Gabriel
What: Coarse fixation kernel (10 cosines, 50 ms stretch).
Why: +0.12 % over 12 cosines, 50 of 80 units, p = 0.0047; it passes parameter
recovery.

## 2026-10-17 · decision · unknown
What: Flash terms as a base kernel plus modifiers (flash just before, after
exit). Previous outcome is tested again and fails again.
Why: The same columns, easier to read.

## 2026-10-21 · dead end · Gabriel
What: The first-flash-after-exit term is not in the model. The result is
reported as a characterization.
Why: The term was made from the residual that it removes, on the same data. A
new term must have its form set in advance and a test on data that did not
choose it.
Evidence: first_flash_after_exit.py; removed code in 46d3d7d, 0f4e510

## 2026-10-21 · correction · Gabriel
What: Corrects 2026-10-12. GRB064 is held out only for the form of the terms,
not for the set of terms. The 2026-10-15 candidate test used both mice.
Why: Only a third mouse tests the set of terms.

## 2026-10-23 · finding · AI
What: Banded ridge (one penalty per band). No band factor predicts better, so
every band keeps 1. The task band is not identified by prediction.
Evidence: band_grid.py, band_grid/comparison.json

## 2026-10-23 · decision · Gabriel
What: Flash recovery term (gap > 50 ms, r = 1 − exp(−gap/τ)) is in the model.
Report gains in held-out bits per spike.
Why: In-port responses grow 2.6–4.3× from a 50–100 ms gap to a 200+ ms gap.
The term passes in both mice (42 of 167, 22 of 76 units).
Evidence: flash_recovery.py; PR #68

## 2026-10-24 · correction · Gabriel
What: Unit metrics used amplifier gain ×80, not ×100. Stable units 168 → 173
(GRB006), 78 → 81 (GRB064); flash-excited 23 → 30, 15 → 15. All fits add the
new units. Conclusions do not change.
Evidence: results/glm/_before_gain_fix_2026_10_24/

## 2026-10-24 · decision · Gabriel
What: Recovery τ = 50 ms.
Why: The curve fit had c at a bound of 1.5 with no recorded reason. With c
free, τ = 54 ms (29–91). Prediction does not choose τ (tau_grid, Holm
p ≥ 0.72), so τ comes from the curve, rounded.
Evidence: tau_grid.py; results/glm/_before_tau_50_2026_10_24/

## 2026-10-24 · dead end · AI
What: 200 motion-energy PCs. With the shared penalty they are worse (−0.0045,
p = 1e-10). With a video penalty ×100 they equal 10 PCs. Keep 10.
Evidence: pc_penalty_grid.py

## 2026-10-29 · decision · Gabriel
What: Flash × side (after-exit flash basis × response side) is in the model.
The weaker-penalty sensitivity fit is removed.
Why: After the exit the head turns, so the LED panel is in a different part of
the visual field for each side. 56 of 172 and 25 of 79 units need the term.
The leak test corrects the leak directly.
Evidence: 8e35876; results/glm/_before_flash_side_2026_10_29/

## 2026-10-29 · decision · Gabriel
What: The model is frozen. New terms are tested only on new sessions. The log
ratio is the main measure of the flash × exit result (chosen after both
measures were seen).
Why: Flash × side was chosen after the decoding was seen. With two sessions,
more terms risk fitting the analysis to the data.

## 2026-10-29 · decision · Gabriel
What: The GLM stays the main analysis. Model-free tests come before it in the
story.

## 2026-10-29 · finding · AI
What: Power test with planted effects: 80 % power at +0.21 (GRB006) and +0.28
(GRB064) log ratio.
Evidence: power_test.py, 7ac7378d

## 2026-10-31 · dead end · AI
What: Trial-rate term (stimulus strength, first flash to exit) on GRB064
20261008: 3 of 115 units need it (chance 14). The per-flash terms explain the
fast − slow difference.
Evidence: code removed in 710bbbc; parent f834b39 has it

## 2026-11-01 · decision · Gabriel
What: GRB064 uses session 20261008, not 20260916.
Why: 9 rates against 4, and behavior closer to expert.
Evidence: b6590ef; results/glm/_before_grb064_switch_2026_11_01/

## 2026-11-04 · finding · AI
What: With 20261008, the observed GRB064 increase after the exit is weaker
(0.17, was 1.05). Leak-corrected flash × exit, held out: ×0.77, p = 0.021.
Pooled over mice ×0.82 [0.64, 1.04], p = 0.063. Neither mouse shows a
flash-specific increase.

## 2026-11-04 · decision · Gabriel
What: Keep 20261008 and report both mice as they are.
Why: The mice differ and more data will come.

## 2026-11-09 · correction · Gabriel
What: The held-out measure takes its leak from held-out folds.
Why: With no planted effect the held-out measure had a bias of −0.04 in
GRB006. Result moves to −0.05 / −0.31 (p = 0.71 / 0.011). Conclusions do not
change.
Evidence: 167de520

## 2026-11-13 · decision · Gabriel
What: Every flash peak (observed, model, kernel) is the mean over ±20 ms
around the largest bin in 30–120 ms (`peaks.window_peak`).
Why: Anne asked for a window, not one bin. One measure in every
analysis.
Evidence: 53ae518a, 5f6bd2d1; PR 92

## 2026-11-14 · correction · AI
What: Corrects the leak and power runs that started after the 2026-11-13
decision. Those runs are superseded. The
branch did not have the 1e-6 float fix, and the in-port window held the
after-exit flash at the 40 ms gap. The in-port window now ends at 90 ms.
Evidence: b5fd897

## 2026-11-14 · finding · AI
What: With the window peak, gap ≥ 80 ms: pooled null (p 0.61 final, 0.70 held
out); GRB006 up, GRB064 down. 40 ms gap: pooled log decrease (p 0.001 final,
0.033 held out).

## 2026-11-15 · decision · Gabriel
What: Add GRB061 20261113 as the third mouse (provisional).
Why: Best behavior and highest wait of the candidate sessions.
Evidence: e64d76e1

## 2026-11-15 · finding · AI
What: The flash kernel dip of flash-suppressed units is deeper after the exit
(pooled −0.11 Δ log rate, 78 % of 187 units, every mouse). The excited peak
does not change (62 units).
Evidence: suppression_gain.py; PR #105; Notion https://app.notion.com/p/41aecf086b7c8186b065d0f7223bedcb

## 2026-11-16 · finding · AI
What: The trial mix does not explain the mouse difference. Matched rate × side
trials give the same values (GRB006 +1.05, GRB064 +0.07).
Evidence: trial_balance.py; PR #99

## 2026-11-16 · finding · AI
What: Three mice (GRB061 provisional), corrected held-out long-gap log ratio:
+0.10 / −0.35 / −0.19. Pooled −0.07 [−0.43, +0.23], p = 0.695. The short-gap
pooled result is no longer significant (−0.21, p = 0.092).
Evidence: Notion https://app.notion.com/p/415ecf086b7c81f4aa5bdc1de417108b

## 2026-11-17 · decision · Gabriel
What: Keep the 100 ms spike history in the main model.
Why: A 20 ms history improves calibration and has fewer runaway simulations,
but loses 0.07 bits/spike. The flash × exit conclusion is the same with 20 ms,
with no history, and with history set to 0 at prediction time.
Evidence: code not in main (separate worktree); Notion https://app.notion.com/p/415ecf086b7c81f4aa5bdc1de417108b

## 2026-11-18 · decision · Gabriel
What: Keep the 500 Hz cap on the null simulations of the bias correction.
Why: The population conclusion holds from 250 to 1,000 Hz.
Evidence: code not in main (separate worktree); Notion https://app.notion.com/p/415ecf086b7c81f4aa5bdc1de417108b

## 2026-11-18 · decision · Gabriel
What: Keep the Poisson likelihood, the shared penalty and the causal exit
kernel. Report the misfits as methods limitations.
Why: Clustering and refractoriness reject the conditional Poisson model, but
the effects are small (fixation residual −4 %; GRB064 pre-exit ramp changes the
flash windows by 1–4 %). A fix brings back decided problems.
Evidence: poisson_gof.py, residual_timing.py; PR #103

## 2026-11-20 · decision · Gabriel
What: The main fit runs in the `GLMFit` Computed table, so the table holds the
code commit and the outputs.
Evidence: PRs 107–109

## 2026-11-20 · correction · AI
What: Corrects 2026-11-15. In GRB006 the kernel dip is deeper after the exit, but the recorded dip
and the held-out prediction are both shallower. The model fits the data; the
deeper kernel comes from how the model divides the rate rise at the exit
between terms. "Deeper dip" is a statement about the kernel, not the response.
Evidence: Notion https://app.notion.com/p/41aecf086b7c8186b065d0f7223bedcb

## 2026-11-21 · finding · AI
What: 5 GRB064 sessions, 100 tracked units: after left turns the recorded dip
is deeper (−0.18 [−0.25, −0.12]); the GLM predicts about a quarter of it
(−0.04). No single term hides it. The predicted side difference is nearly all
flash × side and is about 4× too small.
Why: A candidate cause is shrinkage of the flash × side kernel. Not tested; it
needs a refit.
Evidence: Notion https://app.notion.com/p/41aecf086b7c8186b065d0f7223bedcb

## 2026-11-21 · decision · Gabriel
What: Each analysis keeps a timeline file. This file starts it for the GLM.
Evidence: AGENTS.md, "Analysis timelines"
