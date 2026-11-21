# Category and choice decoding

Question: can the stimulus category and the choice be read from V1
population activity?

Entries before 2026-11-21 are reconstructed from the git log, the module
docstring, saved result files and agent notes. See
`ephys/analyses/glm/timeline.md` for the conventions.

## 2026-08-24 · question · Gabriel
What: Decode the low- or high-rate category and the choice from the rate of
each unit 0 to 1 s after the first flash.
Evidence: f0f71318

## 2026-08-25 · decision · Gabriel
What: Balance the four category × choice groups in each resample, so one
decoder cannot use the other variable. Shuffle inside each level of the
other variable.
Evidence: 40bcc687

## 2026-10-24 · decision · Gabriel
What: Decoders use single units and multi-units by default (`--units all`).
Single units are an option (`--units sua`).
Why: Single units give the same pattern with a few points lower accuracy.
Multi-units add units and do not change the conclusions.
Evidence: b8cab6cf

## 2026-10-29 · decision · Gabriel
What: Decoder error bars are the SEM over the CV folds, not percentile
intervals.

## 2026-10-29 · finding · AI
What: Without the GLM flash terms the category AUC after the exit falls by
0.09 / 0.16 (GRB006 / GRB064). The category signal is mostly visual. The choice
AUC stays at 1.00 / 0.97 without them.
Evidence: glm/decoding_without_terms.py (that analysis continues this question)
