# Decoding time course

Question: when in the trial can the category and the choice be read from
V1?

Entries before 2026-11-21 are reconstructed from the git log, the module
docstring, saved result files and agent notes. See
`ephys/analyses/glm/timeline.md` for the conventions.

## 2026-10-24 · question · unknown
What: The one-window decoders (0 to 1 s) include the turn. Use a sliding
200 ms window, aligned to the first flash and to the exit.
Evidence: 237622a2

## 2026-10-29 · decision · Gabriel
What: The band is the SEM over folds, not a percentile interval.
Evidence: 719ed293

## 2026-10-31 · finding · AI
What: The category signal is clearest after the center exit.
Evidence: kernel_decoding.py docstring
