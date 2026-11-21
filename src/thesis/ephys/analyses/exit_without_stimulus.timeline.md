# Exit without stimulus

Question: does the rise of the rate at the center exit need the stimulus?

Entries before 2026-11-21 are reconstructed from the git log, the module
docstring, saved result files and agent notes. See
`ephys/analyses/glm/timeline.md` for the conventions.

## 2026-10-11 · finding · AI
What: In GRB064 the rate rises at the exit by the same amount when the
stimulus stops there. This argues against the view of the stimulus as the
cause of the fixation effect.
Evidence: glm_decisions.md, "Tests of 2026-10-11" (removed in 12371d8)

## 2026-10-21 · decision · unknown
What: A model-free module: completed against early-withdrawal trials,
aligned to the first exit after the first flash, paired over units.
Limit: an early withdrawal has a 2 s punishment tone.
Evidence: 0ca525e6 (PR #56)

## 2026-11-16 · finding · AI
What: Last numbers of the docs (GRB064 20261008). After-exit rate, early ÷ completed: 1.00 / 1.08 (p = 0.53 / 0.02).
The rise does not need the stimulus.
Evidence: docs/glm_checks.md (removed in 12371d8)
