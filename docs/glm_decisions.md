# V1 GLM: history of the design decisions

This file has the earlier designs of the V1 GLM and the test behind each
change, in the order in which they were made (2026-10-10 to 2026-10-17).
Read [glm.md](glm.md) for the current scientific argument and main figures.
See [glm_checks.md](glm_checks.md) for validation, supporting figures, methods, and run commands.
This file is the design history.

Some sections name scripts that are no longer in the repository, because the
final model does not use them: `side_term_test.py`, `flash_fixation_test.py`,
`term_test.py`, `adaptation_grid.py`, and `flash_kernel_grid.py`. They are in
the git history before commit `cce48d9`. The terms that were tested and are
not in the model (the center-entry kernel, fast adaptation, flash recovery,
flash × previous choice, flash × motion energy) are also there.

Names changed over time: "hold" is the fixation, "stimulus during fixation" is
the fixation kernel, "side after exit" is the choice, "previous side" is the
previous choice, "in fixation" and "after exit" are "still" and "move", and
"flash × fixation" is "flash × move".

## The first full design and what was removed from it

The full design had nine regressors: the five above and flash post − pre
withdrawal (6 columns), flash × motion energy (6), initiation (4), and response
entry (6), with 25 motion-energy PCs selected on validation. Its fit of
2026-10-10 (167 units, 294 trials) is in
`GRB006_20240821_121447_full_design/all_fit_me/` of the archive of `figures/glm/` (the layout before 2026-10-23)
(moved to an external disk on 2026-10-21; git does not track it). Do not render that folder again with the current code; the code for that
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

### Flash × move interaction

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

### Candidate terms, all together

Test of 2026-10-12 (`candidate_terms.py`, all 167 units, results in
`candidate_terms.json`). One design: the 7 regressors of the model plus four
candidates (side after the exit, flash × move, flash recovery, and a
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
| Flash × move (candidate) | 8 | 41% | 44% | 0.0% | 0.0009 |
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
- The flash × move term is needed by more units here (41%) than in its
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
- 16 units need this term and the flash × move term; 82 of 167 units need
  one or the other.
- Rule from 2026-10-12 (Gabriel): the importance of a regressor is tested on
  all units, because 20 units are too few (this term and the flash × move
  term were not reliable on 20 units and are reliable on 167). The form of a
  term (bases, window, spacing) is still chosen on the 20 selection units. By
  this rule the fast-adaptation term is no exception (60% of units), and this
  term and the flash × move term have a case to be in the model. Not
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
| Flash × move | 8 | 42% | 1.8% | 0.0008 |
| Choice | 1 | 32% | 1.2% | 0.0003 |
| Session drift | 3 | 28% | 2.4% | 0.0004 |
| Previous choice | 1 | 20% | 0.6% | 0.0002 |
| Initiation | 1 | 18% | 1.8% | 0.0001 |

- Median cross-validated D² 0.0734 (IQR 0.046 to 0.104). Against the design
  with 3 gap classes and the center-entry kernel (77 columns), same folds:
  median change +0.0002 (+0.3%), 105 of 167 units better, p = 0.0003.
- Collinearity of this design: see "Collinearity of the regressors".
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

### Candidates for the flash response when the animal moves

Two terms were added together to the final design and tested on all units of
the two sessions (`candidate_terms.py`,
`candidate_terms_flash_move_candidates.json`, 13 regressors, 82 columns).

- Flash × move × gap (`FLASH_MOVE_GAP_TERM`, 8 columns): the flash bases
  scaled by the centered period times the centered gap class. With it, the
  change from still to move can differ between the 40 ms and the 80+ ms gap.
- Flash × motion energy (`FLASH_MOTION_TERM`, 8 columns): the flash bases
  scaled by the z-score of the total motion energy of the video at the flash
  (correlation with the period over flashes: 0.57 in GRB006). It is a measure
  of movement; the flash × move term has only the task period.

| Term | GRB006: units that need it | GRB064 | Rule |
| --- | ---: | ---: | --- |
| Flash × move × gap | 43 of 167 (26%), 0.0% worse, median unique ΔD² 0.0004 | 18 of 76 (24%), 1.3% worse, 0.0005 | passes in the two sessions |
| Flash × motion energy | 54 of 167 (32%), 0.6% worse, 0.0005 | 1 of 76 (1%), 1.3% worse, 0.0001 | passes in GRB006, fails in GRB064 |

- Whole model with the two terms against the final design, same folds:
  GRB006 +0.5% in the median (110 of 167 units better, p = 2e-5); GRB064
  −0.4% (32 of 76 better, p = 0.020).
- Flash × move × gap: the change of the flash response from still to move
  differs between the gap classes in about a quarter of the units of each
  animal. A fit with this term alone is the next test
  (`candidate_terms_flash_move_gap.json`).
- Flash × move × gap alone, added to the final design (12 regressors, 74
  columns; `candidate_terms_flash_move_gap.json`): it passes the two tests of
  the rule in the two sessions. GRB006: 44 of 167 units need it (26%), 0.6%
  worse, median unique ΔD² 0.0004; whole model +0.17% (96 of 167 units better,
  p = 0.011). GRB064: 18 of 76 (24%), 0% worse, 0.0005; whole model +0.19% (43
  of 76 better, p = 0.21). The gain is small and has the same size in the two
  animals. The term is in the design from 2026-10-13; with it,
  the flash kernel has a value for each of the four conditions (still and
  move, at the two gap classes).
- Flash × motion energy: not the same in the two animals. In GRB064 no unit
  needs it. The video is a different view in the two sessions (the back
  stereo view in GRB006, the bottom view in GRB064), so the total motion
  energy is not the same measure. The term stays out.
- The other regressors keep their level with the two terms in the design.

### Flash kernels by gap class

Tests of 2026-10-12 (`gap_class_grid.py`, 20 selection units). The response to
a flash depends on the gap from the flash before. The first form of this was
two terms on a common flash kernel: fast adaptation (a flash within 50 ms) and
flash recovery (the log of the gap). They are correlated (−0.86 over 3,208
flashes), so the fit can move weight from one to the other and their kernels
cannot be read one by one. The flash × move term is not correlated with
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

### Design of 2026-10-15: turn direction, response period, shorter kernels

Candidate test of 2026-10-15 (`candidate_terms.py`, all candidates added to
the design of 2026-10-13 together, both mice): time in trial (52% and 67% of
the units) and turn direction (37% and 42%) pass the rule; accumulated
evidence (4% and 1%) and the previous outcome (6% and 8%) fail. In a design
with the two that pass, time in trial had a first canonical correlation of
1.00 with the fixation kernel (GVIF^(1/p) 9.9 and 9.5, condition index 92),
and turn direction one of 0.91 and 0.87 with the side term.

Decisions (Gabriel, 2026-10-15):

- The fixation kernel stays the term that reads the fixation effect; no new
  time term may overlap it. Time in trial is out.
- Turn direction replaces the side term. Its kernel runs from the center exit
  to the response entry: 8 cosines with log spacing to 1.3 s (the 95th
  percentile of exit to entry in both mice), and the last cosine keeps its
  level to the entry.
- The response period (the same basis, no sign) replaces the withdrawal
  kernel, so the time after the exit has a part common to both sides.
- Flash × move × gap is out: split halves 0.61 and 0.57, and its effect was
  not reliable after the leak correction.
- The fixation kernel ends at 0.7 s (the 95th percentile of the fixation
  length in both mice), not 0.95 s. The kernel after 700 ms had a split-half
  agreement of 0.26 in GRB006, because 1 to 3% of the trials reach it.
- The window starts 0.5 s before the first flash, so the baseline is before
  the center entry. The initiation column is named "pre-stimulus".
- Flash × motion energy stays out (fails in GRB064, where the camera view is
  different).

Shuffle fix of 2026-10-16 (`unique_dev.py`): the over-trials shuffle of the
unique ΔD² moved the exit-aligned time course of a side term to another trial,
so the shuffle also removed the response period. In a synthetic unit with
only flash responses, turn direction had a unique ΔD² of 0.19. A side on a
time course (previous choice, turn direction) now keeps its own rows and
takes the sign of another trial; session drift and the previous outcome are
still shuffled over trials. With the fix, turn direction is needed by 49% and
63% of the units.

Result: all 11 regressors pass in both mice; median D² 0.072 and 0.074. The
fits of the design of 2026-10-13 are in `_before_design_2026_10_15/` of the
archive of `figures/glm/` (the layout before 2026-10-23).

### Coarse fixation kernel, 2026-10-16

All seven design grids were run again on the design of 2026-10-15
(`design_optimization.py`). One form predicted better than the form in use:
the fixation kernel with 10 cosines and a 50 ms stretch, against 12 cosines
with 25 ms (+0.12%, 50 of 80 units, p = 0.0047; on the design of 2026-10-13
it was not better, p = 0.10). Parameter recovery with it
(`parameter_recovery --grid hold_grid`) recovers every measure as well: the
error of the fixation level and of flash × move does not grow (p from 0.67 to
0.94), and the fixation level is recovered at 0.67 and 0.85 against 0.66 and
0.84. By the rule of the design, the coarse kernel replaces the 12-cosine
kernel (decided with Gabriel, 2026-10-16). Both mice were fit again, with
every check and every grid on the new design:

- D² 0.0716 → 0.0717 (GRB006) and 0.0744 → 0.0744 (GRB064); all 11
  regressors still pass; the units that need each regressor change by 0 to 4
  points.
- Fixation level recovered at 0.70 and 0.80; split halves of the fixation
  kernel 0.89 and 0.94; largest GVIF^(1/p) 1.63, condition index 8.4.
- On the new design no other form predicts better, and the 12-cosine kernel
  is now the worse one (−0.12%, p = 0.0047).
- The grid also has a new form of the gap classes: one flash kernel plus a
  kernel for a flash up to 50 ms before (`FLASH_JUST_BEFORE_TERM`). It spans
  the same columns as the two classes and predicts the same (p = 0.56).
- The exit basis got its own log stretch (`EXIT_LOG_STRETCH_S`, 25 ms): it
  shared the stretch of the fixation kernel, so a grid of the fixation kernel
  changed it too.

The fits with the 12-cosine kernel are in `_before_coarse_kernel_2026_10_16/` of
the archive of `figures/glm/` (the layout before 2026-10-23).

### Flash terms as a base and modifiers, 2026-10-17

Decided with Gabriel, for a model that reads in words. The flash part of the
design is now one flash kernel (the response in the port after a long gap),
a kernel for a flash up to 50 ms after another (`FLASH_JUST_BEFORE_TERM`),
and the flash × after-exit kernel coded 0 in the port and 1 after the exit.
Before, the design had one kernel for each of two gap classes, and the
after-exit values were centered over the flashes, which made the common
kernel a mean of the two periods. The columns span the same space in both
forms, so the prediction is the same (on the design of 2026-10-16: −0.03%,
p = 0.56); only the L2 penalty differs: it now pulls the change after a
short gap and the change after the exit toward zero. The labels "still" and
"move" became "in port" and "after exit", because "move" reads as a measure
of locomotion, and the rig has none.

### Previous outcome, second test, 2026-10-17

`candidate_terms.py` with the previous outcome alone, on the design of
2026-10-17, with the window from 0.5 s before the first flash (it now
reaches before the center entry, where an outcome effect would most likely
be). It fails in both mice again: 9 of 167 units need it in GRB006 (chance
18) and 4 of 76 in GRB064 (chance 11); gain p = 0.11 and 0.28 after Holm. It
stays out.

### Cleanup, 2026-10-17

Three rounds of a complexity audit. Deleted: `residual_checks.py`,
`kernel_comparison.py`, `flash_by_period.py` (its result is panels g–i of
`model_trust.svg`), `performance.py`, `withdrawal_grid.py`, the flash-onset
repair tool (the GRB006 row is written), the descriptive model of the
fixation figure (it now reads the fixation and response-period kernels of
the main fit), and the dead design terms (time in trial, accumulated
evidence, the side column, flash × move × gap). The comparisons of
`matched_flashes.py` that no figure reads are gone too. All of them are in
the git history.

### First flash after the exit, candidate, 2026-10-21

The model missed only the response to the first flash after the exit
(`first_flash_after_exit.py`). A third flash kernel, 1 for the first flash at
or after the exit (the flash basis), was added to the design in use and fit
on all units (`candidate_terms.py`, fits in `<session>_first_flash_after_exit`).

- It passes both inclusion tests in both mice: 59 of 167 and 30 of 76 units
  need it (35% and 39%; Holm p < 0.001). All other regressors still pass.
  Median D² is 0.0719 and 0.0748 (0.072 and 0.075 without it).
- The miss at the first flash after the exit falls from +0.93 / +0.65 to
  +0.63 / +0.38 (p = 0.003 / 0.022); the other flashes stay near 0. With a flash
  minus with no flash: +1.13 / −0.19, pooled +0.41 (−0.26 to 0.92).
- The fitted first response is almost as high as the observed one, but it
  peaks about 10 ms later (about 45 against 35 ms). The flash basis has 8
  cosines over 150 ms, about 21 ms apart.
- Form (`first_flash_grid.py`, 80 grid units of GRB006, validation D²
  against 150 ms with 8 cosines): 150 ms with 12 cosines −0.00002 (p = 0.76),
  with 16 −0.00004 (p = 0.68), 100 ms with 8 +0.00004 (p = 0.62); no term
  −0.0002 (p = 0.072). No form predicts reliably better, so the kernel keeps
  the flash basis. A finer basis does not make the earlier peak worth its
  columns.
- Not in the model (Gabriel, 2026-10-21). The term was made from the
  residual that it removes, on the same data, so passing the inclusion rule
  does not show that it is real; a rank indicator ("first") also mixes the
  rank of the flash with its time after the exit (0 to 250 ms), and a
  modifier cannot shift latency. The result is reported as a
  characterization (`first_flash_after_exit.py`). The term and the grid were
  removed from the code; they are in the git history (`46d3d7d`, `0f4e510`).
  A term for this must have its form set in advance and be tested on data
  that did not choose it.

### Banded ridge, 2026-10-22 and 2026-10-23

The weaker fixation penalty changes the GRB006 flash × after exit result
(`penalty_sensitivity.py`), and its factor of 10 was chosen by hand. Decided
with Gabriel: choose a penalty for each band of regressors by a fixed rule
instead (banded ridge; Nunez-Elizalde, Huth and Gallant 2019; Dupré la Tour et
al. 2022).

- Bands (`dmat.BAND_SCALES`, `dmat.penalty_band`): flash (the sensory terms),
  task (fixation, pre-stimulus, previous choice, response period, turn
  direction), nuisance (session drift and motion energy), and spike history as
  the reference band at 1. Only the ratios matter, because the overall penalty
  is chosen for each unit; a factor f makes the penalty of its band f² times
  weaker. All 1 is the model in use (its design is unchanged).
- Bands by concept and few, because one penalty for each regressor would give
  the fit more freedom to move shared variance between correlated terms.
- Rule, fixed before the run (`band_grid.py`): one band at a time at ×1/3, ×3
  and ×10, on the 80 grid units of GRB006 (GRB064 held out of this choice); a
  band keeps 1 unless a factor predicts reliably better (Holm p < 0.05 over the
  9 designs, median change above 0). Then the chosen factors together, and only
  then a refit of both mice.
- Prediction was flat for the fixation penalty, so the grid may keep 1 for
  every band; that result would show by a fixed rule that the data do not
  choose the penalties.
- Result (2026-10-23, `band_grid/comparison.json` in the GRB006 session
  directory, 80 units): no factor predicts reliably better. Three predict
  reliably worse: flash ×1/3 (median −0.0010, 23 of 80 units better, Holm
  p = 7e-5), flash ×10 (−0.0001, 34 of 80, p = 0.017) and task ×1/3 (−0.0006,
  27 of 80, p = 0.003). Task ×3 and ×10 and every nuisance factor do not
  differ (Holm p = 1). So every band keeps 1: one penalty for all terms is the
  choice of the rule, step 2 and a refit are not needed, and the model does
  not change.
- The task band is not identified by prediction: a penalty 100 times weaker
  on the whole band predicts the same (p = 1). So the size of the task
  kernels, as of the fixation kernel alone, depends on a penalty that the
  data do not choose; the weaker-fixation-penalty fit stays the sensitivity
  analysis.

### Flash recovery in the model, 2026-10-23

The model missed long-gap flashes in the port too (`exit_flash_responses.py`):
the design coded every gap above 50 ms the same, but in-port responses grow
2.6 to 4.3 times from a 50–100 ms gap to a 200+ ms gap. The old log-gap term
(2026-10-12) had gone out in the choice of the form, because the grid sees
little gain from rare long-gap flashes, and it correlated −0.86 with fast
adaptation.

- Form set before the test: the flash basis scaled by r = 1 − exp(−gap / τ),
  centered, for gaps above 50 ms; τ = 120 ms from the in-port flashes of
  GRB006 (`flash_recovery.py`: 121 ms, 95% 88 to 157; GRB064 alone 92 ms).
  Correlation 0.00 with the short-gap modifier.
- Inclusion (`candidate_terms.py`, all units): 42 of 167 and 22 of 76 units
  need it (Holm p = 2e-16 and 7e-9); all other terms still pass. Bits per
  spike: median +0.0006 per unit; in the flash-excited units +0.0036 / +0.0032
  (18 of 23, 13 of 15 better).
- Adopted (Gabriel, 2026-10-23): the candidate fit is the model; the earlier
  fits are in `results/glm/_before_flash_recovery_2026_10_23/`. All checks
  and figures were run again; the raw-video control and the design grids are
  still of the design of 2026-10-17.
- Effects: result 3 full model 1.89 / 0.95 → 1.98 / 1.21 (observed 2.63 /
  1.34); leak-corrected flash × exit +0.63 / −0.20 → +0.54 / +0.06; the miss
  at the first flash after the exit falls at 200+ ms gaps but stays at
  100–200 ms. Largest GVIF^(1/p) 4.2; the term shares a direction with the
  start of the fixation kernel (first canonical correlation 0.72 / 0.70).
- Reporting changed to held-out bits per spike (Gabriel, 2026-10-23); the
  inclusion decisions are the same as with D².

## Moved from glm.md (2026-10-17)

These sections are on the designs of 2026-10-11 to 2026-10-13. They are kept
here as the history of the tests; the model in use is in glm.md.

#### Tests of 2026-10-11 (GRB006)

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
| Stillness in the port | The same low rate at a side port with the same motion energy, and the same rise with motion energy | No, in both animals: the side-port rate goes up with motion energy and the center-port rate does not (p = 8e-11 and 5e-6). At the lowest motion energy alone: 0.78 of the side-port rate in GRB006, 0.84 (p = 0.31) in GRB064 |
| A change of the gain of the whole unit (state) | The same factor for all units and for baseline and response | Not fully, and the two animals differ. Units with a larger flash response are suppressed more in GRB006 (Spearman ρ = −0.20, p = 0.01; the earlier design gave −0.41), not in GRB064 (ρ = +0.10, p = 0.41). For flashes after a long gap, the response changes more than the baseline in GRB006 (+0.18 in log, p = 0.0003) and much less in GRB064 (−0.75, 8 of 65 units above 0, p = 1e-9); for short gaps, no difference in either (0.00 and −0.03) |
| The view of the stimulus from the center port | A step at the exit that needs the stimulus, in units that respond to it, and a common effect of the side of the turn | Agrees with the fast step at the exit and the relation to the flash response. Against it: the rate goes up at the exit by the same amount when the stimulus stops there (GRB064), the side of the response has no common direction, and the rate already falls from the center entry to the first flash, with no stimulus (0.95 of the rate before the entry in GRB006, 0.70 in GRB064) |

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
### Penalty: one for all terms, and the sensitivity analysis

The numbers of this section and the next are for the design of 2026-10-13
(12 regressors, with flash × move × gap). The weaker-penalty fit was not run
again on the design of 2026-10-15.

Decision of 2026-10-13: the model in use has one L2 penalty for all terms,
selected for each unit. That is the common practice for single-neuron task
GLMs (Park et al. 2014: one ridge penalty, evidence optimization; Musall et
al. 2019: ridge, one penalty for each pixel; Steinmetz et al. 2019: elastic
net with fixed values). With it:

- The leak test is the parameter-recovery check of the model (as in
  Rabinowitz et al. 2015, who simulated known modulators), and the flash ×
  move effect is reported with the leak taken out (see "Leak of the fixation
  level").
- The size of the fixation effect is taken from the raw rates
  (`hold_effect.py`), because the fixation kernel of this model is too small.
- The fit with a weaker penalty on the fixation kernel is the sensitivity
  analysis (`HOLD_PENALTY_SCALE = 10`; fits in `_weaker_fixation_penalty`,
  with no unique-deviance step). A separate treatment of a slow baseline term
  has precedent (a drift term with its own prior in Rabinowitz et al. 2015; a
  drift term in Liska et al. 2024; one penalty for each group of regressors
  in banded ridge, Nunez-Elizalde et al. 2019), but no task GLM that was read
  leaves a state term with a weaker penalty. The conclusions are the same
  with it: +0.09 in GRB006 and −0.16 in GRB064.
- For a term of interest that shares variance with another term, Musall et
  al. (orthogonalization) and Steinmetz et al. (a kernel fit to the residuals
  of the model without it) give the shared part to the other terms. That is
  an alternative to the leak correction and is not run here.

In the tables of the next section, "same penalty" is the model in use and
"weaker" is the sensitivity analysis.

### A weaker penalty on the fixation kernel removes the leak

The leak comes from the L2 penalty. With the same penalty on all terms, the
fit makes the fixation kernel too small, and the flash × move kernels take the
rest of the level: in the simulations, the fixation kernel keeps the full
level at a penalty of 1e-5, 80% at 1e-4, and 39% at 1e-3.

Test on simulated spikes of every unit (the true flash × move effect is 0;
a scratch test, not a module):

| Way to fit the fixation term | Leak, GRB006 | GRB064 | Level kept, GRB006 | GRB064 |
| --- | ---: | ---: | ---: | ---: |
| Same penalty on all terms | +0.086 | +0.058 | 63% | 78% |
| One shared time course for all units, one weight for each unit | +0.093 | +0.058 | 56% | 77% |
| Weaker penalty on all terms (1e-5) | +0.073 | +0.049 | 83% | 95% |
| Penalty 100 times weaker on the fixation kernel only | +0.030 | −0.003 | 101% | 103% |

- A shared time course does not remove the leak: its weight has the same
  penalty.
- `HOLD_PENALTY_SCALE = 10` (the default is 1) makes the columns of the fixation kernel 10 times
  larger than a z-score, so the penalty on its weights is 100 times weaker.
  The saved scale of the columns has the factor in it, so the kernels are
  read as before. Factors of 30 and 100 stop the convergence of the fit.
- Prediction on real data is the same: on 80 units the same penalty is not
  different from the weaker one (`hold_grid`, −0.04%, 38 of 80 units better,
  p = 0.75), and on all units the cross-validated D² does not change (GRB006:
  92 of 167 units better, p = 0.52; GRB064: 46 of 76, p = 0.063).

Both sessions were fit again with the weaker penalty (design of 2026-10-13,
12 regressors; these fits are in `_weaker_fixation_penalty`), and
`leak_test.py` was run on them (`leak_test_weaker_fixation_penalty.json`):

| Median over units, Δ log rate | GRB006, same penalty | GRB006, weaker | GRB064, same penalty | GRB064, weaker |
| --- | ---: | ---: | ---: | ---: |
| Fixation level (kernel, 200 to 450 ms) | −0.24 | −0.54 | −0.46 | −0.71 |
| Simulated fixation level that the refit keeps | 59% | 89% | 79% | 102% |
| Flash kernel, move minus still, 80+ ms gap | +0.24 | +0.13 | −0.05 | −0.14 |
| Leak at the 80+ ms gap | +0.08 | +0.02 (p = 2e-7) | +0.06 | 0.00 (p = 0.82) |
| Real minus leak, 80+ ms gap | +0.15 | +0.09 (p = 3e-5) | −0.13 | −0.16 (p = 3e-7) |
| Flash kernel, move minus still, 40 ms gap | +0.06 | 0.00 | −0.01 | −0.04 |
| Real minus leak, 40 ms gap | +0.04 | −0.01 (p = 0.53) | −0.04 | −0.06 (p = 0.003) |

- The fixation level of the kernel is now −0.54 in GRB006. The raw rate in
  the fixation is 0.56 of the rate before the first flash (log −0.58), so the
  kernel now agrees with the model-free measure.
- The leak is gone in GRB064 and small in GRB006 (+0.02).
- The flash response at the 80+ ms gap is a little higher in move in GRB006
  (+0.09 with the leak taken out) and lower in move in GRB064 (−0.16). The two
  animals have opposite signs. At the 40 ms gap there is no change in GRB006
  and a small decrease in GRB064.
- The numbers of the other sections (flash kernels by period, kernel
  stability, collinearity, unique ΔD²) are from the model in use, with the
  same penalty on all terms.
