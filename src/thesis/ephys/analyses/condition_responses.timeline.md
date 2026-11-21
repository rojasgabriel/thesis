# Condition responses

Question: before any decoder, how do single units and the population differ
between the two values of a decoded variable (choice, category)?

Entries before 2026-11-21 are reconstructed from the git log, the module
docstring, saved result files and agent notes. See
`ephys/analyses/glm/timeline.md` for the conventions.

## 2026-10-25 · question · unknown
What: Show the raw data before each decoder.
Evidence: 022489cd

## 2026-10-25 · decision · unknown
What: Heatmap display limit ±2. Values past the limit stay in the statistics;
the json gives their count.
Evidence: 5754329f

## 2026-11-12 · decision · Gabriel
What: 10 ms bins, and "units" in the labels.
Evidence: 9d202bf8

## 2026-11-12 · finding · AI
What: Category, in port only, 0.1 to 0.5 s: 51 of 173 (GRB006) and 33 of 116
(GRB064) units differ (p < 0.05); 47 and 26 of them prefer fast trains.
Choice, 0 to 0.3 s after the exit: 94 of 173 and 62 of 116 units differ.
Evidence: results/condition_responses/{category,choice}.json (saved
2026-11-12)
