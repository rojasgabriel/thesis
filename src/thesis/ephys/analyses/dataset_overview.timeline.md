# Dataset overview

Question: what do single trials, single units and the population look like
around the flashes and the task events, before any model?

Entries before 2026-11-21 are reconstructed from the git log, the module
docstring, saved result files and agent notes. See
`ephys/analyses/glm/timeline.md` for the conventions.

## 2026-10-24 · question · unknown
What: A raw-data overview of the GLM sessions for the talk.
Evidence: 6c222a73

## 2026-10-25 · decision · Gabriel
What: Color shows only a concept (Okabe–Ito). A mouse is a marker shape,
never a color.
Evidence: c35d74f8

## 2026-10-25 · decision · unknown
What: The example unit is excited by the StimulusResponse table and by the
model, and is chosen by the per-flash discriminability of its in-port
response.
Evidence: e006afd0

## 2026-10-28 · decision · unknown
What: First-flash raster uses only trials with the whole window in the port.
Evidence: 849d751d, 9d0d3503

## 2026-11-12 · decision · Gabriel
What: Population maps show (rate − rate in the 40 ms before the first flash)
÷ mean rate. No log2.
Why: One reference (pre-stimulus) and one normalization on all slides.
Evidence: 9d202bf8
