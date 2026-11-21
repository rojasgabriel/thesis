# Suppression onset

Question: does the drop of the V1 rate in the center port start at the
first flash or at the port entry?

Entries before 2026-11-21 are reconstructed from the git log, the module
docstring, saved result files and agent notes. See
`ephys/analyses/glm/timeline.md` for the conventions.

## 2026-10-10 · finding · AI
What: In the GLM residuals, the drop starts about 80 ms after the first flash
in trials with short and long entry delays, so it follows the flash.
Evidence: glm_decisions.md, "Fixation" (removed in 12371d8)

## 2026-10-21 · decision · unknown
What: A model-free test: in long-delay trials, rate from the entry to the
first flash ÷ rate before the entry, paired over units.
Evidence: 0ca525e6 (PR #56)

## 2026-10-11 · correction · AI
What: Corrects 2026-10-10. With no model, the rate already falls from the
center entry to the first flash, with no stimulus: 0.95 (GRB006) and 0.70
(GRB064 20260916) of the rate before the entry.
Evidence: glm_decisions.md, "Tests of 2026-10-11" (removed in 12371d8)

## 2026-11-16 · finding · AI
What: Last numbers of the docs (GRB064 20261008): long-delay rate before the
first flash ÷ rate before the entry is 0.95 / 0.88 (p = 0.02 / 0.03). The
approach and the entry are not separated.
Evidence: docs/glm_checks.md (removed in 12371d8)
