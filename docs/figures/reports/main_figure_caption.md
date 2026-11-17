**A Poisson GLM predicts V1 spikes from the visual stimulus, the task, movement,
and spike history.** The model predicts the spike count of each unit in 1-ms
bins. Each group of regressors adds a term to the log of the firing rate, so a
kernel is a change in log rate (Δ log rate): a value of 0.7 is a rate two times
as high. Colors are the same in all panels: visual (blue), task (orange),
motion energy (green), spike history (red), and model prediction (purple).
**a**, The design: each term over one schematic trial, with the time in which
it is on (bars) or the flashes it models (ticks), and its number of columns.
Before the center entry no event term is on; each event kernel is relative to
that time, at the mean of motion energy, spike history, the session drift, and
the previous choice, which are on there too. Panels b
to d are the example unit of GRB006. **b**, One held-out trial of the
example unit (unit 197, 53 spikes/s), against time from the
first flash. From the top: the events (the flashes; the pre-stimulus time, from
the center-port entry to the first flash, in a light color; the fixation, from
the first flash to the center exit; and the exit); the first motion-energy PC
of the video; the term that each group of regressors adds to the log rate; the
term of spike history, in its own row; and the observed spikes. **c**, The
kernels of the example unit. Top, from the left: the flash kernel for a flash
in the fixation that follows another within 50 ms (40 ms gap) and for every
other flash (80+ ms gap); the flash kernel for a flash in the fixation (in port)
and after the exit (dashed), at a gap of 80 ms or more; and the fixation
kernel, which starts at the first flash and ends at the exit. Bottom: the term
after the exit for a right and a left turn (response period plus and minus
turn direction) and the spike-history kernel. **d**, Observed and predicted
firing rate of the example unit over the 59 held-out test trials, in
20-ms bins. Each trial is predicted by a model that was not fit on it. A
time bin is shown while at least half of the trials have not reached the
response port. **e**, Cross-validated D² (deviance explained on held-out
trials, relative to a constant rate) against bits/spike for each of the 287
units of the two mice. Small points are units, in three groups of equal size by
firing rate (a darker gray is a higher rate). In each group, the units are in
bins of equal size by bits/spike; a large marker is the mean of a bin and the
band is ±1 SEM. The histograms are the distributions over units (median D²
0.075; median bits/spike 0.50). **f**, What each regressor explains (in the
order of GRB006): the unique ΔD² (filled; the D² that the model loses when
that regressor is shuffled: inside each trial, or over trials for a variable
of the trial) and the maximal ΔD² (open; the D² of that regressor alone).
Markers are the median over the pooled units with its hierarchical-bootstrap
95% interval (the mice, then the units of each). The top row is the D² of the
full model. The axis is logarithmic and starts at 0.001; a smaller value is
drawn there. The numbers at the right are the fraction of the units of both
mice with a unique ΔD² above zero (one-sided t-test over the 10 folds, p <
0.05; a regressor with no effect gives about 5%). A regressor is in
the model when it passes two tests over the units of each mouse, each with a
Holm correction for the 13 regressors: the unique ΔD² is above zero
(one-sided Wilcoxon signed-rank test over units), and more units need the
regressor than chance gives (one-sided binomial test against 5%;
18 of 172 (GRB006) and 14 of 115 (GRB064) units or more). A dagger marks a regressor that fails in one mouse,
and n.s. one that fails in both. GRB006: every regressor passes both tests. GRB064: every regressor passes both tests.
