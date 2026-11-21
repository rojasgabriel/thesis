# Decoding transfer

Question: does a category readout learned while the mouse is still work
while it moves?

Entries before 2026-11-21 are reconstructed from the git log, the module
docstring, saved result files and agent notes. See
`ephys/analyses/glm/timeline.md` for the conventions.

## 2026-10-09 · question · AI
What: Proposed: decode the category in still and in moving epochs.
Why: To separate the visual and the movement representations without
decoding from model parts (that is circular).

## 2026-10-24 · decision · unknown
What: Score with AUC. A gain on all activity leaves the AUC the same, so a
drop in transfer means a change of the readout axis. PCA rule fixed before the
run: K = 10 is the test; K = 5 and 20 are a sensitivity check.
Evidence: 237622a2

## 2026-10-28 · finding · AI
What: AUC for train still → test still / test moving, and train moving →
test moving / test still: GRB006 0.66 / 0.66 and 0.87 / 0.66; GRB064
0.66 / 0.56 and 0.76 / 0.60. So in GRB006 the still readout works as well
while the mouse moves; the moving readout is better on moving trials. In
GRB064 the still readout is worse on moving trials. Transfer − within, test
moving: −0.21 [−0.39, −0.05] and −0.20 [−0.39, −0.04]. With PCA K = 10 the drops are smaller and include 0
(−0.09 [−0.22, 0.01], −0.12 [−0.24, 0.04]). With K = 10, GRB064 also
drops from moving to still (−0.15 [−0.29, −0.01]).
Why: From the docstring: without PCA there are 636 / 485 features for 100 / 68 trials, so the
asymmetry can come from overfitting the still readout.
Evidence: results/decoding_transfer/decoding_transfer{,_pca10}.json (saved
2026-10-28)
