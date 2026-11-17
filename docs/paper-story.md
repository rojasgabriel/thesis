# Paper story

Working synthesis reconciled on 2026-11-16 against the [saved results](figures/README.md).
The model design is frozen from 2026-10-29. The main flash comparison now includes
three mice; most mechanistic and decoding controls still cover the original two.
This is a working scientific interpretation, not a locked claim.

## Question

How does leaving the center port, from evidence sampling to choice execution,
change V1 responses to visual evidence? How much of that change is carried by
the fitted flash × exit term after accounting for other modeled signals?

“In port” and “after exit” are task periods, not measured locomotion. Choice,
side, elapsed time, flash order, and adaptation change around exit. Without eye
or head tracking, the analyses cannot isolate a causal movement effect.

## What the data show

1. **The observed change differs across sessions.** At long gaps, mean normalized
   peak changes are **+1.01 / +0.10 / −0.22** for GRB006 / GRB064 / GRB061, using
   **30 / 22 / 10** saved flash-excited units. GRB006 has a clear increase; the
   other sessions do not show the same pattern. GRB061 is provisional: its
   required wait is 0.37 s, below the session-selection criterion of 0.5 s.
2. **The model describes overlapping signals.** The saved overview shows 287
   units from GRB006 and GRB064, with pooled held-out median **0.50 bits/spike**
   and **D² = 0.075**. All 13 regressor groups pass inclusion tests in those
   original mice. That overview is not a three-mouse model-validation result.
3. **There is no established common positive flash × exit contribution.** The
   corrected held-out long-gap log-ratio medians are **+0.10 / −0.35 / −0.19**,
   with within-mouse p = **0.152 / 0.0103 / 0.945**. The three-mouse pooled median
   is **−0.07**, 95% **−0.43 to +0.23**, p = **0.695**. At short gaps the pooled
   median is **−0.21**, 95% **−0.62 to +0.05**, p = **0.092**. The earlier
   significant two-mouse short-gap pooled result does not describe this cohort.
   The ratio excludes nonpositive or otherwise unusable peaks; the short-gap
   GRB061 estimate retains only four units.
4. **Sensitivity is uneven.** Held-out long-gap 80% power is reached at planted
   log ratios of about **+0.15 / +0.14 / +0.42**. The ten-unit GRB061 analysis
   has much lower power. A nonsignificant effect does not establish no change,
   and the pooled interval still permits a meaningful positive effect.
5. **The original two-mouse controls suggest several contributors.** Background
   gain, adaptation, and response side help explain GRB006 and GRB064. The
   per-unit gain control cannot distinguish an unchanged flash response from
   a shared gain in GRB006, while GRB064 shows a smaller flash response after
   exit. These controls have not established the same explanation in GRB061.
6. **Movement during the wait is a separate observational comparison.** In-port
   flash-response slopes are **−29% / −28% / −45% per SD** of motion energy,
   p = **0.00027 / 0.00010 / 0.084**. The provisional mouse is uncertain, and
   the older time-in-wait adjustments cover only GRB006 and GRB064. Movement
   and other state changes remain possible explanations of this association.
7. **The first-flash miss is heterogeneous too.** The three-mouse pooled
   flash-minus-no-flash model miss is **+0.25**, 95% **−0.06 to +0.45**,
   p = **0.109**. The positive two-mouse pooled interval should not be used to
   describe the current three-mouse figure. Category and choice decoding,
   side, raw-video, and depth results remain supporting two-mouse controls.

## Current claim

V1 flash-aligned responses change around center-port exit, with different
patterns across the three recorded sessions. The saved fitted-term comparison
does not establish a shared positive flash × exit effect. It also does not
establish equivalence or a lack of causal movement modulation. The provisional
third session and the limited number of mice restrict generalization.

## Figure arc

1. Task, timing, chronic V1 recording, and the transition from center port to choice.
2. [Observation](figures/glm/flash_observation.svg): three mice, with PSTHs and
   nested all-unit/flash-excited group means; no individual-unit scatter points.
3. [Encoding model](figures/glm/main_figure.svg): the original two-mouse overview.
4. [Matched comparison](figures/glm/flash_comparison.svg): observed and held-out
   predictions in three mice, followed by [corrected results](glm.md#correct-for-an-apparent-effect-created-by-fitting)
   and [power](figures/glm/power_test.svg). The displayed rate peaks themselves
   are not leak-corrected.
5. [In-port movement](figures/glm/flash_movement_in_port.svg) and
   [first-flash model miss](figures/glm/first_flash_after_exit.svg): three mice.
6. [Supporting controls](glm_checks.md): explicitly identify which still cover
   only GRB006 and GRB064 before drawing on gain, adaptation, side, or decoding.

The GLM remains the main analysis. Double peaks, population geometry, depth,
and audio-event recovery are supporting analyses or methods.

## Next steps

- Add sessions that meet the behavioral criteria, with more than one session
  per mouse; keep the provisional GRB061 result distinct.
- Extend model validation and the relevant supporting controls to the third
  mouse before treating the extension as a full replication.
- Inspect individual-unit distributions alongside group means before making
  claims about how broadly a change is shared within a session.
- Test terms suggested by these data, such as in-port movement × flash and
  improved high-rate summation, on new sessions rather than selecting them here.
- Obtain more matched error trials to test choice signals before exit, and use
  eye/head measurements or an experimental control to separate visual-field
  changes from state and movement.
