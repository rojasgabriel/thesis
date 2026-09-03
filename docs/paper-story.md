# Emerging paper story

Status as of 2026-09-02. This is a working scientific synthesis, not a locked
analysis plan or established claim.

## Central question

How does the transition from stationary evidence sampling to choice execution
alter how V1 represents visual evidence during freely moving decisions, and how
much of that transition is explained by movement?

## Working hypothesis

V1 carries task-relevant sensory evidence during freely moving decisions, while
the transition from evidence sampling into choice execution alters response
gain and population structure. Task-linked movement is a leading explanation,
but held-out decoding, direct movement measurement, and cross-animal
replication are still needed to establish it.

The task provides the organizing comparison: repeated visual pulses can occur
before and after center-port exit within the same trial. This connects the
single-unit, population, decoding, and movement questions more directly than
treating them as separate findings.

## Terminology boundary

The maintained comparison labels the last flash before center exit as
stationary and the first flash between center exit and response-port entry as
movement. This is a task-epoch definition, not a direct measurement of
locomotion. Choice, movement direction, elapsed time, pulse order, adaptation,
and trial phase remain coupled around center exit.

Historical analyses used scalar full-frame video motion energy. That measure
does not identify body parts, movement direction, or task-independent movement.

## Current evidence

### Verified on the current branch

- `notebooks/ephys/population_geometry.ipynb` was executed for GRB006 session
  `20240821_121447` using 168 stable units and 102 trials balanced across three
  stimulus categories and two responses.
  - Signal PC1 explained 31.6% of condition-mean variance.
  - Mean response trajectories separated most clearly late in the trial.
  - Category and exact-rate trajectories overlapped more strongly.
  - The analysis is descriptive, phase-normalized, and not evaluated on
    held-out trials.
- `notebooks/ephys/demixed_pca.ipynb` was executed on the same population.
  - The first three response-related components captured 10.9% of
    condition-mean variance.
  - The first three category-related components captured 3.3%.
  - The leading response component separated left and right responses mainly
    after center exit.
  - The model was fit and evaluated on the same condition means.
- `notebooks/ephys/tutorials/tutorial_stat_vs_move.ipynb` was executed for
  GRB006 using 23 stable, stimulus-excited units and 283 paired trials.
  - It compares the last 15 ms pulse before center exit with the first 15 ms
    pulse after center exit and before response-port entry.
  - Rasters, PETHs, and unit-level peak comparisons are retained.
  - No inferential movement-effect test or numerical population effect estimate
    is retained.
- `notebooks/ephys/tutorials/tutorial_peth.ipynb` demonstrates a clear
  first-stimulus response for one GRB006 unit. It is tutorial evidence, not a
  population result.

### Implemented but not currently verified by retained output

- `src/thesis/ephys/analyses/stim_responses_by_locomotion.py` implements the
  task-epoch response comparison for one GRB006 and one GRB058 session.
  The intended two-session figure is not present in this checkout.
- `src/thesis/ephys/analyses/double_peaks.py` implements the discovery and
  pulse-width control for double-peaked responses. Historical documentation
  reports only two stable double-peak units with both pulse widths, so this
  remains descriptive.
- `src/labdata_plugin/schema.py` implements unit stability and paired
  stimulus-responsiveness criteria. These identify global stimulus responses,
  not formal category, exact-rate, or choice tuning.

### Historical or separate-branch evidence

- Choice-balanced category decoding, exact-rate decoding, rate tuning, and
  encoding-model analyses were developed historically but are not part of the
  current branch with retained result files.
- Focused category and rate decoders currently live on
  `codex/stimulus-decoding`. They should be rerun on the unified event pipeline
  before their outputs are treated as current evidence.
- A historical one-session analysis found asymmetric cross-motion category
  transfer after exact category-by-choice matching and nuisance adjustment.
  This is a useful hypothesis-generating result, not a replicated claim.
- A historical low- versus high-video-motion-energy analysis did not resolve a
  difference in stationary-period choice decoding.

## Competing explanations

### Additive gain

Movement or another post-exit state variable scales firing rates without
changing the sensory representation.

Prediction: decoders transfer well between states, and movement main effects
are sufficient in a cross-validated encoding model.

### State-dependent reformatting

Task-linked movement changes which population dimensions carry stimulus
information.

Prediction: cross-state decoder transfer is impaired or asymmetric, and
sensory-by-movement interaction terms improve held-out neural predictions.

### Task-phase or action confounding

The apparent state effect is produced by choice, elapsed time, pulse order,
adaptation, task events, or correlated movements.

Prediction: matched comparisons and fold-local nuisance controls remove the
response, tuning, and transfer differences.

## Proposed figure arc

1. **Task and behavioral anchor.** Establish task performance, event timing,
   chronic V1 recording, and the within-trial transition from center-port
   occupancy to choice execution.
2. **Single-unit sensory responses.** Show verified flash responses and
   response heterogeneity, then re-establish rate tuning on the current
   pipeline. Keep double peaks as characterization or supplementary material
   unless they explain the main effect.
3. **What V1 carries over time.** Establish held-out category and rate decoding,
   then relate sensory readout to the late response-related structure seen in
   the population analyses.
4. **What changes with movement.** Add aligned kinematics, match flash order and
   elapsed time, and compare within-state decoding with cross-state transfer
   across sessions and animals.
5. **Encoding-model adjudication.** Partition sensory, task-event, choice,
   outcome, and movement effects. Test whether sensory-by-state interactions add
   reproducible held-out predictive value.

## Minimum useful encoding-model comparison

The base model should include:

- visual and tone onsets;
- task events;
- choice and outcome;
- aligned video or kinematic movement features.

The comparison model should add sensory-by-movement or sensory-by-state
interactions. Compare held-out predictive performance per unit while preserving
sessions and animals as the replication units.

## What should not become the paper's spine

- Double-peaked responses;
- descriptive PCA or dPCA alone;
- audio-event recovery;
- choice decoding interpreted as an abstract cognitive signal;
- retired depth, SNR-threshold, selectivity, or Niell-style analyses.

These remain useful controls, methods, or supporting observations.

## Next decisive step

First rerun the focused category and rate decoders on the unified thesis event
pipeline. Pair that with a direct movement signal and matched pulse-order and
elapsed-time controls. Then run a session-by-session replication matrix for:

1. visual-response modulation;
2. within-state and cross-state category and rate decoding;
3. the cross-validated benefit of sensory-by-movement interactions.

That result will distinguish a paper about additive gain from one about
state-dependent reformatting, or show that the apparent effect is explained by
task phase.
