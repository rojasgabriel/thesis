# Thesis analysis guide

Run analysis modules from the repository root:

```bash
uv run python -m thesis.ephys.<group>.<module_name>
```

Figures go under `figures/`. Set `THESIS_FIGURE_ROOT` to change that root.

## Main figures

```bash
uv run python -m thesis.ephys.analyses.locomotion_peaks_analysis
uv run python -m thesis.ephys.analyses.double_peak_responses_across_sessions
```

### Locomotion

`locomotion_peaks_analysis.py` is the canonical locomotion entrypoint. It reads
session data through `compute_locomotion_peaks` and compares paired
last-stationary and first-movement responses. Each condition keeps its own peak
latency.

Use `--split-by-waveform` for FS/RS summaries, `--show` for an interactive
window, and `--no-save` to inspect without writing a PDF.

The analysis-specific manual review is available with:

```bash
uv run python -m thesis.ephys.analyses.manual_conditioned_psth_browser -a GRB058 -s SESSION
```

### Double peaks

`double_peak_responses_across_sessions.py` is the collaborator-facing summary.
Supporting analyses are:

- `grb006_double_peak_example_units.py` — GRB006 examples
- `double_peak_responses_by_pulse_width.py` — 15 ms versus 30 ms control
- `double_peak_units_waveform_profile.py` — firing rate and spike duration

Canonical parameters live beside the classifier in
`src/thesis/ephys/peak_classification.py`. The analysis uses 10 ms bins
without smoothing, Wilcoxon responsiveness with FDR, and a 5 sp/s minimum height
above baseline for both peaks.

`spks.event_aligned.population_peth` returns counts per bin. Each caller
converts those counts to spikes per second by dividing by its bin width.

The pulse-width control argues against a simple pulse-offset explanation. It
does not establish a mechanism for the second peak.

## Behavior

Cheap analyses derived from stored trials live on the Chipmunk table. Use
`Chipmunk.fit_psychometric(**key)` or
`Chipmunk.fit_psychophysical_kernel(is_nidq=False, **key)`. Kernel fits use
Bpod timing by default; pass `is_nidq=True` to use synchronized hardware
events. Neither fit is stored in a derived table.

## Ephys module roles

- `src/thesis/ephys/analyses/` answers scientific questions and writes figures.
- `src/thesis/ephys/tools/` contains reusable standalone interactive browsers.
- Shared data access and scientific operations live directly under
  `src/thesis/ephys/`.

## Interactive tools

```bash
uv run psth -a GRB058 -s SESSION
uv run unit-stability -a GRB058 -s SESSION
uv run stimulus-responsiveness -a GRB058 -s SESSION
uv run survey-map -f SURVEY_MAP.csv
```

Add `--stability-param-id 0` to a unit-based browser to require units to pass
both its selected quality criteria and stability parameter set 0.
