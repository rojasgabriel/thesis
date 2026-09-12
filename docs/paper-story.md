# Emerging paper story

Working synthesis as of 2026-09-02, not a locked plan or claim.

## Question

How does the shift from stationary evidence sampling to choice execution
change V1’s representation of visual evidence in freely moving decisions, and
how much of that is movement?

## Hypothesis

V1 carries task-relevant sensory evidence; leaving the center port changes
response gain and population structure. Task-linked movement is the leading
explanation. Still needed: held-out decoding, direct movement measurement,
cross-animal replication.

The useful comparison is within-trial: visual pulses before vs after center
exit.

## Terminology

Last flash before center exit = stationary. First flash between center exit
and response entry = movement. That is a task epoch, not measured locomotion.
Choice, direction, elapsed time, pulse order, adaptation, and trial phase are
coupled around exit.

Historical “movement” was scalar full-frame video motion energy. It does not
identify body parts, direction, or task-independent movement.

## Evidence

**Verified on this branch**

- `notebooks/ephys/population_geometry.ipynb` — GRB006 `20240821_121447`, 168
  stable units, 102 trials (3 categories × 2 responses). Signal PC1 31.6% of
  condition-mean variance. Trajectories separate most late in the trial;
  category overlap is stronger. No exact stimulus rate. Descriptive,
  phase-normalized, not held-out.
- `notebooks/ephys/demixed_pca.ipynb` — same population. First three
  response components 10.9% of condition-mean variance; first three category
  components 3.3%. Leading response component separates left/right mainly
  after center exit. Fit and evaluated on the same condition means.
- `notebooks/ephys/tutorials/tutorial_stat_vs_move.ipynb` — GRB006, 23
  stable stimulus-excited units, 283 paired trials. Last 15 ms pulse before
  exit vs first 15 ms pulse after exit and before response entry. Rasters,
  PETHs, peak comparisons only; no inferential test.
- `notebooks/ephys/tutorials/tutorial_peth.ipynb` — one GRB006 unit, tutorial
  only.

**Implemented, no retained output here**

- `src/thesis/ephys/analyses/stim_responses_by_locomotion.py` — GRB006 and
  GRB058 task-epoch comparison; two-session figure missing.
- `src/thesis/ephys/analyses/double_peaks.py` — discovery + pulse-width
  control. Historically two stable double-peak units with both widths.
- `src/labdata_plugin/schema.py` — stability and paired stimulus-responsiveness.
  Global flash responses, not category/rate/choice tuning.

**Historical / other branch**

- Category decoding, exact-rate decoding, rate tuning, encoding models: not
  on this branch with result files.
- Focused category and rate decoders: `codex/stimulus-decoding`. Rerun on the
  unified event pipeline before treating as current.
- One-session asymmetric cross-motion category transfer after
  category-by-choice matching: hypothesis-generating only.
- Low vs high video motion energy did not change stationary-period choice
  decoding.

## Competing explanations

**Additive gain.** Post-exit state scales rates, sensory axes stay put.
Prediction: decoders transfer; movement main effects suffice in a
cross-validated encoding model.

**State-dependent reformatting.** Movement changes which dimensions carry
stimulus. Prediction: impaired/asymmetric transfer; sensory×movement terms
improve held-out predictions.

**Task-phase / action confounding.** Choice, time, pulse order, adaptation,
events, or correlated movement. Prediction: matching and fold-local nuisance
controls remove the differences.

## Figure arc

1. Task, timing, chronic V1, center-port → choice.
2. Single-unit flash responses and heterogeneity; re-establish rate tuning.
   Double peaks stay supplementary unless they explain the main effect.
3. Held-out category and rate decoding; relate sensory readout to late
   response-related population structure.
4. Aligned kinematics, matched flash order and elapsed time; within-state
   decoding vs cross-state transfer across sessions and animals.
5. Encoding model: sensory, task events, choice, outcome, movement. Test
   whether sensory×state interactions add held-out value.

Base encoding model: visual and tone onsets, task events, choice and outcome,
aligned video/kinematics. Comparison model adds sensory×movement or
sensory×state. Score held-out per unit; replicate over sessions and animals.

## Not the spine

Double peaks; descriptive PCA/dPCA alone; audio-event recovery; choice
decoding as an abstract cognitive signal; retired depth/SNR/selectivity/Niell
analyses. Fine as controls or methods.

## Next step

Rerun category and rate decoders on the unified event pipeline, with a direct
movement signal and matched pulse-order and elapsed-time controls. Then a
session-by-session matrix: (1) visual-response modulation, (2) within- vs
cross-state category and rate decoding, (3) held-out benefit of
sensory×movement terms. That distinguishes additive gain, reformatting, or
task-phase confounding.
