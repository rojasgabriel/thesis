# Paper story

Working synthesis as of 2026-10-30, from the frozen GLM of 2026-10-29 on two
mice (GRB006 / GRB064, one session each). It is not a locked plan or claim.
Numbers and methods are in [glm.md](glm.md) and [glm_checks.md](glm_checks.md).

## Question

How does leaving the center port, from stationary evidence sampling to choice
execution, change V1 responses to the visual evidence, and how much of that
change is movement?

"In port" and "after the exit" are task periods, not measured locomotion.
Choice, side, elapsed time, flash order, and adaptation all change at the
exit. There is no head or eye tracking. That is a limitation of the study.

## What the data show

1. **Observation.** After the exit, flash responses are larger in GRB006:
   26 of 30 flash-excited units at long gaps (normalized peak +1.01). In
   GRB064 (20261008) the increase is small: 14 of 22 units, +0.10. At short
   gaps, most of the increase is a higher rate before the flash.
2. **Model.** A Poisson GLM with 13 terms (flash history with gap, recovery,
   exit, and side modifiers; task events; video; spike history) predicts
   held-out spikes at 0.55 / 0.47 bits per spike. All terms pass the
   inclusion tests.
3. **The flash × exit term carries almost none of the increase.** After the
   leak correction, the held-out log ratio is +0.10 / −0.35 in medians
   (GRB006 p = 0.15; GRB064 p = 0.01, below zero; pooled median −0.06,
   95% −0.46 to +0.20). The held-out test has 80% power at +0.15 / +0.14.
   GRB006 shows no change and GRB064 a decrease. Because the mice differ, the
   pooled interval does not exclude a positive effect of that size.
4. **What the increase is instead:**
   - **Gain on all activity.** The rate before the flash goes up (GRB006:
     g_b 1.84). Per unit, with the trial noise included, the GRB006 flash
     response cannot be told apart from the same gain or from no change (g_f
     2.05); in GRB064 it is smaller after the exit than in the port (g_f
     0.27; g_f − 1 from −1.22 to −0.25).
   - **Adaptation reset.** The first flash after the exit is a little larger
     than this gain predicts (×1.8 at 100–200 ms gaps, 1.03–2.9), and the
     model misses only a small part of it.
   - **Side.** The flash response differs with the response side
     (flash × side). The direction is opposite in the two mice. Without eye or
     head position, a visual-field shift and a side-specific gain look the
     same.
5. **Body movement does not make the flash response larger.** In the port,
   more motion energy goes with a smaller flash response (−29% / −28% per
   SD; 22 of 27 / 17 of 21 units). After the exit, the response does not
   change with motion energy. This can be movement or a state that goes with
   it.
6. **Decoding.**
   - **Category:** the category signal around the exit is mostly visual.
     Without the flash terms, the AUC falls by 0.09 / 0.16. The GRB006
     leftover comes from a known miss of the flash model at high flash rates.
   - **Choice:** before the exit, a decoder trained on correct trials leans
     toward the stimulus on error trials (AUC 0.41 / 0.38, p = 0.20 / 0.17),
     but that measure mixes the rates. At the same flash rate, the activity
     before the exit goes with the choice in GRB064 (choice probability 0.62,
     p = 0.03; GRB006 0.57, p = 0.16). The large choice signal appears with
     the movement.
7. **Depth.** No movement-related kernel changes with depth.

## Current claim

The larger flash responses after the exit are mostly a state change at the
exit: a gain on all activity, a reset of adaptation, and a side-dependent
part. The data do not show a flash-specific effect of how much the body
moves.

## Competing explanations, updated

- **Additive gain / state change at the exit:** supported. One gain on all
  activity explains most of the increase.
- **Flash-specific modulation by movement:** not supported. In the port the
  relation goes in the other direction.
- **Visual change with the turn:** possible for the side part only. This
  needs eye or head position, or retinotopy, which the data do not have.

## Figure arc

1. Task, timing, chronic V1 recording, center port → choice.
2. Observation: flash responses after the exit vs in port
   (`figures/glm/flash_observation.svg`).
3. Encoding model: design, prediction, kernels (`figures/glm/main_figure.svg`).
4. Comparison: observed, model, and model without flash × exit
   (`figures/glm/flash_comparison.svg`), with power and the per-unit gain.
5. What changes at the exit: gain, adaptation reset, flash × side, and
   in-port movement (`figures/glm/flash_movement_in_port.svg`).
6. Decoding: category and choice without model terms, choice by outcome.

The GLM is the main analysis. Model-free tests are support before the GLM.

## Not the spine

Double peaks, PCA/dPCA descriptions, depth, audio-event recovery. These are
fine as controls or methods.

## Next steps

- **New sessions or a third mouse.** The model is frozen, so only new data
  can test flash × exit, the direction of flash × side, and candidate terms.
- **Candidate terms for new sessions:** in-port movement × flash, and better
  summation at high flash rates.
- **More error trials** to test the choice signal before the exit at a fixed
  stimulus in both mice.
- **Brightness protocol** (brightnesschipmunk draft PR #1): rig test and more
  brightness levels.
