# Rate against count

Question: do V1 units follow the flash rate, or only the number of flashes?

Entries before 2026-11-21 are reconstructed from the git log, the module
docstring, saved result files and agent notes. See
`ephys/analyses/glm/timeline.md` for the conventions.

## 2026-10-31 · question · Gabriel
What: Test rate tuning at a matched number of flashes in the window.
Why: A deck comment: the "fast-train preference" needs this control.
Evidence: 5dfbccfe

## 2026-10-31 · finding · AI
What: Matched fast − slow difference ÷ mean rate: GRB006 +0.04 (52 of 173
units shown, p = 0.14); GRB064 20261008 +0.12 (26 of 114, p = 0.0016).
Evidence: results/rate_vs_count/rate_vs_count.json (saved 2026-10-31)

## 2026-10-31 · dead end · AI
What: A GLM trial-rate term does not explain the GRB064 difference beyond the
per-flash terms (3 of 115 units need it).
Evidence: glm/timeline.md, 2026-10-31
