# Rate decoding

Question: can the stimulus rate be read from V1 population activity?

Entries before 2026-11-21 are reconstructed from the git log, the module
docstring, saved result files and agent notes. See
`ephys/analyses/glm/timeline.md` for the conventions.

## 2026-08-24 · question · Gabriel
What: Ridge regression of the stimulus rate, against a decoder trained on
shuffled rates.
Evidence: f0f71318

## 2026-10-24 · decision · Gabriel
What: Single units and multi-units by default.
Evidence: b8cab6cf

## 2026-10-25 · decision · Gabriel
What: Axes show stimulus strength, (rate − 12) / 8: 0 at the boundary, ±1 at
4 / 20 flashes/s. The decoder and the results stay in Hz.
Why: The sign and size of the evidence matter for the task, not the rate.
Evidence: 7e39a3d8
