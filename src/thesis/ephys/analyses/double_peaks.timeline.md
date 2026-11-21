# Double peaks

Question: do double-peaked flash responses come from the pulse onset and
offset?

Entries before 2026-11-21 are reconstructed from the git log, the module
docstring, saved result files and agent notes. See
`ephys/analyses/glm/timeline.md` for the conventions.

## 2024-07-19 · finding · Gabriel
What: Flash events are aligned; response peaks are near 40 ms.
Evidence: 185a3912

## 2026-04-13 · decision · unknown
What: Two-section figure: prevalence (GRB006 unit 77, GRB058 examples), then
the offset hypothesis (15 ms and 30 ms pulses on the same axes).
Evidence: 691543cd

## 2026-08-22 · decision · Gabriel
What: Separate the discovery sessions (GRB006 20240821, GRB058 20260224)
from the pulse-width control (GRB058 20260312).
Evidence: 83d118f9

## 2026-10-30 · decision · unknown
What: Double peaks are a control or methods item, not the main story.
Evidence: docs/paper-story.md, "Not the spine" (removed in 12371d8)
