# Choice by outcome

Question: before the center exit, does the V1 choice signal follow the
choice or the stimulus?

Entries before 2026-11-21 are reconstructed from the git log, the module
docstring, saved result files and agent notes. See
`ephys/analyses/glm/timeline.md` for the conventions.

## 2026-10-29 · question · Gabriel
What: Train a choice decoder on correct trials and test it on error trials.
In error trials the choice and the category disagree.
Why: One of the follow-ups agreed after the PI-style review of 2026-10-29.
Evidence: 701555d1

## 2026-10-29 · decision · AI
What: Test one window only (500 to 100 ms before the exit), with a
permutation of the error-trial choices.
Why: Neighbouring windows share most spikes, so they are not independent
tests.
Evidence: 327cccda

## 2026-10-29 · finding · AI
What: Error-trial AUC in that window is 0.41 / 0.38 (p = 0.20 / 0.17), with
89 / 50 error trials. Both lean toward the stimulus, but this is not shown.
The measure mixes the rates: error trials are mostly near the boundary.

## 2026-11-07 · finding · AI
What: At a fixed flash rate (choice probability, weighted over rates), the
activity before the exit goes with the choice: 0.57 / 0.62 (p = 0.16 / 0.03).
The error AUC below 0.5 came from the stimulus, which differs between rates.
It does not exclude a choice signal.
Evidence: e65c7dfe

## 2026-11-07 · question · unknown
What: Next: more error trials from new sessions, to test the pre-exit choice
signal in both mice with the matched measure.
