# Thesis analysis guide

Run analysis modules from the repository root:

```bash
uv run python -m thesis.ephys.<group>.<module_name>
```

Figures go under `figures/`. Set `THESIS_FIGURE_ROOT` to change that root.

## Main figures

```bash
uv run python -m thesis.ephys.analyses.locomotion_peaks_analysis
uv run python -m thesis.ephys.analyses.double_peak_analysis
```

### Locomotion

`locomotion_peaks_analysis.py` is the canonical locomotion entrypoint. It reads
session data through `compute_locomotion_peaks` and compares paired
last-stationary and first-movement responses. Each condition keeps its own peak
latency.

Use `--split-by-waveform` for FS/RS summaries, `--show` for an interactive
window, and `--no-save` to inspect without writing a PDF.

### Double peaks

`double_peak_analysis.py` builds a possible supplemental figure for a future
paper. It asks whether double peaks reflect separate stimulus-onset and
stimulus-offset responses. If so, increasing pulse duration should leave the
first-peak latency unchanged and delay the second peak by the same amount. The
same figure shows whether double-peak units cluster by waveform duration or
recording depth.

The maintained analyses use unit quality criteria 1 and stability parameter set
0. General excited-unit selection comes from `StimulusResponsiveness` parameter
set 0, which uses all first-stimulus events. Peak shape uses 15 ms events only in
the mixed-width sessions. The classifier uses 10 ms bins without smoothing and
a 5 sp/s minimum height above baseline for both peaks.

`spks.event_aligned.population_peth` returns counts per bin. Each caller
converts those counts to spikes per second by dividing by its bin width.

Double-peak units are selected using all 15 ms first pulses. The pulse-duration
comparison then uses only trials without another pulse during the 120 ms peak
search window and measures the response maximum in fixed windows around the
expected peaks. The result is descriptive because only two stable double-peak
units currently have both 15 ms and 30 ms trials.

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
