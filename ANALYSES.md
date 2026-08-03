# Ephys Analysis Guide

Current analysis surface for this repo.

Run from the repo root:

```bash
uv run python scripts/<group>/<script_name>.py
```

Analysis figure scripts write PDF outputs under `figures/`.

## Regenerate Main Figures

```bash
uv run python scripts/analyses/locomotion_peaks_analysis.py
uv run python scripts/analyses/double_peak_responses_across_sessions.py
```

## Canonical Entrypoints

### Locomotion

Primary locomotion analysis:

`scripts/analyses/locomotion_peaks_analysis.py`

Writes:

- `figures/locomotion/condition_peak_from_locomotion_peaks_paired_last_stat_first_move_shared_stat_baseline_no_waveform_split_log.pdf`
- `figures/locomotion/condition_peak_from_locomotion_peaks_paired_last_stat_first_move_shared_stat_baseline_log.pdf` when run with `--split-by-waveform`

Current policy:

- This is the canonical locomotion entrypoint.
- The maintained comparison is paired last stationary vs first movement.
- Each condition keeps its own peak latency inside the response window.
- Peak responses come from `labdata_plugin.analysisschema.LocomotionPeaks`, so populate that computed table before rerunning the figure for new sessions.
- Use `--show` to open an interactive matplotlib window after saving the figure, or `--show --no-save` to inspect without writing a PDF.

### Double-Peak

Canonical parameters live in `src/config/double_peak.py`.

- PETH: `10 ms` bins, kernel disabled
- Baseline: `(-0.04, 0.0)` s
- Response: `(0.03, 0.12)` s; peak search: `(0.0, 0.12)` s
- Selectivity: Wilcoxon + FDR
- Height floor: both peaks must be at least `5 sp/s` above baseline

`compute_population_peth` converts the default `spks.population_peth` output to
`sp/s`. Do not rescale it again downstream.

Analysis scripts:

- `scripts/analyses/grb006_double_peak_example_units.py`
  - output: `figures/double_peak/grb006_examples.pdf`
  - scope: GRB006-only example figure
- `scripts/analyses/double_peak_responses_across_sessions.py`
  - output: `figures/double_peak/dario_story.pdf`
  - scope: collaborator-facing summary figure
- `scripts/analyses/double_peak_units_waveform_profile.py`
  - output: `figures/double_peak/waveform_grid.pdf`
  - scope: firing rate vs spike duration only
- `scripts/analyses/double_peak_responses_by_pulse_width.py`
  - output: `figures/double_peak/pulse_split.pdf`
  - scope: `15 ms` vs `30 ms` pulse-width control

Current counts:

- GRB006 `20240821_121447`: `5 / 189` double-peak units
  - `[217, 570, 579, 694, 720]`
- GRB058 `20260312_134952`: `2 / 142`
  - `[410, 651]`
- GRB058 `20260319_131303`: `1`
  - `[515]`

Interpretation boundary:

- The pulse-width figure rules out a simple pulse-offset explanation for the
  second peak.
- It does not, by itself, prove a specific mechanism.

## Data Path Caveats

### GRB006

GRB006 `20240821_121447` loads from DataJoint for maintained analyses.

- Spike times: good units via `fetch_good_units`
- Trial anchors: `EventMapping` / `DatasetEvents.Digital`
- Behavior summaries: Chipmunk trial rows

### GRB058

GRB058 uses `fetch_good_units`, `fetch_session_events`, and `fetch_trial_metadata`.

`fetch_trial_metadata` mismatch policy:

- `1` trial mismatch: warn and truncate
- larger mismatch: raise

Current expected warnings:

- `20260312_134952`: `OBX=236`, `Chipmunk=235`
- `20260319_131303`: `OBX=219`, `Chipmunk=218`

### Psychophysical kernel (LAB-TASKS-466)

The session-level table is
`labdata_plugin.analysisschema.SessionPsychophysicalKernel`. The distinct
name preserves the existing pooled `behavior_analyses.PsychophysicalKernel`
table and its rows. This table is intentionally ephys-only because its fixed
windows use NIDAQ `EventMapping`; use the pooled table in `behavior_analyses`
for kernels that do not require ephys timing.

Locked exploratory specification:

- primary: `v2_100ms_10bin_center_rate` (flashes before center-port exit)
- window sensitivity: `v2_100ms_10bin_response_rate`
- evidence: Odoemene et al. 2018 Eq. 5; subtract each trial's generative
  `stim_rate_vision`, fit zero-sum temporal contrasts plus the mean-rate
  nuisance regressor
- fit: 100 ms bins, complete-case prefix, 10-fold stratified CV, `C=1`,
  `min_trials_per_bin=50`, seed 0
- report the continuous late/early absolute-weight ratio, CV accuracy, and
  accuracy relative to the fitted-set majority baseline; the categorical
  label is descriptive only

Live GRB006 `20240821_121447` result on 2026-07-28:

| Window | Trials (fit) | Bins fit | CV | Majority | Δ majority | Late/early |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| center exit | 291 (91) | 7 | 0.704 | 0.747 | -0.043 | 1.890 |
| response | 294 (292) | 10 | 0.688 | 0.695 | -0.007 | 0.689 |

The center-window trial set had wait time `0.628 ± 0.077 s` and post-exit
response time `0.676 ± 0.237 s`; the response-window trial set had
`0.622 ± 0.099 s` and `0.676 ± 0.237 s`, respectively.

The early/late conclusion is not stable enough for a claim. The center-exit
label changes from early at `C=0.1` to late at `C=1` and `C=10`; the 50 ms
fit is flat. Neither locked fit beats its fitted-set majority baseline.
Therefore the figure is diagnostic, not evidence that GRB006 is an early or
late integrator, and it should not yet be linked to the V1 category time
course as a directional result.

Regenerate:

```bash
PYTHONPATH=.. python scripts/analyses/psychophysical_kernel_diagnostics.py --populate
```

Writes:

- `figures/psychophysical_kernels/GRB006_20240821_121447_psychophysical_kernel.png`

The expected one-trial alignment warning is `OBX=505`, `Chipmunk=504`;
the loader truncates to 504 before kernel-specific filtering.

## Other Scripts

Still usable, but not part of the main figure surface:

- `scripts/tools/manual_conditioned_psth_browser.py`
  - interactive locomotion PSTH browser
- `scripts/analyses/grb006_first_stimulus_selectivity.py`
  - local first-stim selectivity utility
- `scripts/analyses/grb006_visual_response_adaptation.py`
  - archived local GRB006 adaptation analysis
- `scripts/diagnostics/locomotion_response_snr_distribution.py`
  - reference-only diagnostic for the removed SNR gate

## Retired Surface

Not part of the current analysis interface:

- duplicate older locomotion scripts
- broken pre-canonical timing scripts
- depth-based locomotion figures
- depth-based double-peak figures
- removed duplicate stimulus-scatter entrypoints

## Script Layout

`scripts/` is grouped by scientific use:

- `scripts/analyses/`
  - scripts that answer a scientific question or generate analysis figures
- `scripts/diagnostics/`
  - one-off checks of data quality, sync, or pipeline assumptions
- `scripts/tools/`
  - interactive utilities and browsers
