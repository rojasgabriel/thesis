# Psychometric curves

Question: do the mice use the flash rate to choose a side?

Entries before 2026-11-21 are reconstructed from the git log, the module
docstring, saved result files and agent notes. See
`ephys/analyses/glm/timeline.md` for the conventions.

## 2026-10-24 · question · unknown
What: Psychometric curves for the talk.
Evidence: 6c222a73

## 2026-10-25 · decision · Gabriel
What: x axis in stimulus strength; fits stay in Hz.
Evidence: 7e39a3d8

## 2026-10-31 · decision · unknown
What: Sessions: visual and visual + audio sessions with an ephys recording
and at least 6 rates, no performance rule; visual and multisensory trials
pooled. `--recorded` gives the recorded session only.
Why: GRB006 has few visual-only sessions.
Evidence: 91d0cd51, 38f1175e, 0623707d

## 2026-10-31 · finding · AI
What: Accuracy on non-boundary trials: GRB006 0.78, GRB064 0.69. Slope
0.26 / 0.16 per Hz.
Evidence: results/behavior/psychometric.json (saved 2026-10-31)
