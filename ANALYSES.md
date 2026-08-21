# Thesis analysis guide

Run scripts from the repository root:

```bash
uv run python scripts/<group>/<script_name>.py
```

Figures go under `figures/`. Set `THESIS_FIGURE_ROOT` to change that root.

## Main figures

```bash
uv run python scripts/analyses/locomotion_peaks_analysis.py
uv run python scripts/analyses/double_peak_responses_across_sessions.py
```

### Locomotion

`locomotion_peaks_analysis.py` is the canonical locomotion entrypoint. It reads
session data through `compute_locomotion_peaks` and compares paired
last-stationary and first-movement responses. Each condition keeps its own peak
latency.

Use `--split-by-waveform` for FS/RS summaries, `--show` for an interactive
window, and `--no-save` to inspect without writing a PDF.

### Double peaks

`double_peak_responses_across_sessions.py` is the collaborator-facing summary.
Supporting analyses are:

- `grb006_double_peak_example_units.py` — GRB006 examples
- `double_peak_responses_by_pulse_width.py` — 15 ms versus 30 ms control
- `double_peak_units_waveform_profile.py` — firing rate and spike duration

Canonical parameters live in `src/thesis/ephys/config/double_peak.py`. The
analysis uses 10 ms bins without smoothing, Wilcoxon selectivity with FDR, and
a 5 sp/s minimum height above baseline for both peaks.

`compute_population_peth` returns spikes per second. Do not scale its output
again.

The pulse-width control argues against a simple pulse-offset explanation. It
does not establish a mechanism for the second peak.

## Behavior library

Reusable behavior code lives under `src/thesis/behavior/`:

- `learning.py` — trial-set summaries
- `psychometrics.py` — psychometric fits
- `kernels.py` — psychophysical-kernel math
- `kernel_timing.py` — Bpod and NIDAQ timing inputs

There is no active behavior schema or schema-backed behavior command-line
interface. A new schema needs separate approval.

## Script roles

- `scripts/analyses/` answers scientific questions and writes figures.
- `scripts/diagnostics/` checks data quality and assumptions.
- `scripts/tools/` contains interactive browsers and utilities.
