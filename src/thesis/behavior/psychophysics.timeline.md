# Psychophysics

Question: which flashes do the mice use, when do they leave the port, and
does the last trial change the choice?

Entries before 2026-11-21 are reconstructed from the git log, the module
docstring, saved result files and agent notes. See
`ephys/analyses/glm/timeline.md` for the conventions.

## 2026-10-31 · question · unknown
What: Psychophysical kernel, exit timing and trial history.
Evidence: 94ff442e

## 2026-10-31 · finding · AI
What: All ephys sessions: GRB006 weighs the early flashes most (peak at
420 ms, flat test p < 0.001); GRB064 the later ones (580–900 ms, p = 0.05).
In the recorded session, GRB064 weighs the last bin most (0.77
[0.48, 1.20], p = 0.016); GRB006 is flat (p = 0.54). After a reward GRB064
repeats the side (0.71 [0.28, 1.20], all
sessions; 0.94 [0.50, 1.59], recorded session).
Evidence: results/behavior/psychophysics{,_recorded}.json (saved 2026-10-31)

## 2026-11-12 · decision · unknown
What: Plot the history weights alone (history_weights.py), no new fit.
Evidence: 9d202bf8
