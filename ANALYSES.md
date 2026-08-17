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

### Category and Choice Decoding

GRB006 category, choice, and stimulus-rate decoding:

`scripts/analyses/category_choice_decoding.py`

Writes:

- figures under `figures/category_choice_decoding/`
- `reports/category_choice_decoding.html`

The analysis uses balanced classes for classifier accuracy. It requires the
lab database, VPN access, and the synchronized trial-event mappings. Use
`--quick` only for a smoke test. Do not use quick-run results for scientific
claims.

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
- session psychophysical-kernel code, now maintained in `behavior_analyses`

## Script Layout

`scripts/` is grouped by scientific use:

- `scripts/analyses/`
  - scripts that answer a scientific question or generate analysis figures
- `scripts/diagnostics/`
  - one-off checks of data quality, sync, or pipeline assumptions
- `scripts/tools/`
  - interactive utilities and browsers
