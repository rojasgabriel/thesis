"""Is the larger flash response after the exit one gain on all activity, per unit?

Question: flash responses are larger after the center exit (result 1 of
`docs/glm.md`). The pooled gain test (`exit_flash_responses.py`) compares the
growth of the flash peak with the growth of the rate before the flash over
the pooled units. For each unit: is its flash response scaled by the same
gain as its activity before the flash (pure gain), unchanged on a higher
background (additive), or something else? A check before the GLM.

Measures, per unit and period, from the trial mean in 10-ms bins of the
flashes of the gap class 100-200 ms: the rate before the flash (-40 to 0 ms)
and the peak above it: the bin from 30 to 120 ms chosen on one half of the
trials and read on the other half, minus that rate (split half by trial, both
ways, mean over `SPLITS` random splits; the largest bin of a noisy trace is
biased upward, more in the period with fewer flashes, which would push g_f
toward g_b). The halves are of trials, not flashes, so a trial that a
bootstrap draw takes twice falls in one half only.
g_b = rate before after the exit / in port; g_f = peak after the exit / in
port. g_f = g_b is a pure gain; g_f = 1 an unchanged flash response.

- Why not a model with one response shape for both periods (tried on
  2026-10-29): the shape changes after the exit in some units (unit 197 of
  GRB006: in the port a dip at 0-40 ms and a peak at 70-90 ms; after the exit
  a peak at 30-70 ms), so one scaled shape does not fit and its gain has no
  clear meaning. The peak does not need a fixed shape.
- Flashes: completed GLM trials, windows fully inside the valid interval;
  gap class 100-200 ms (at 50-100 ms the window holds the end of the
  response to the flash before and its dip). In port: the flashes before the
  exit; after the exit: the later flashes (the first after the exit is left
  out: it is larger than any gain, `exit_flash_responses.py`).
- Units: the flash-excited units of `flash_excited_units.json` (`peaks.excited_units`).
- Intervals: `BOOTSTRAPS` draws of the trials with replacement (`SPLITS`
  each); a drawn trial brings its flashes of both periods, so the periods
  stay paired, and all units of a mouse use the same draws. Per unit: 95%,
  error bars in both directions. A unit whose in-port peak is ≤ 0 has no
  g_f and is counted.
- Statistics: per mouse, two-sided Wilcoxon signed-rank tests over units of
  g_f - g_b (pure gain at 0) and of g_f - 1 (unchanged response at 0); they
  take each unit's value as exact and the units as independent. With them,
  the 95% interval of the median over units from draws of trials and units
  together (each draw: one trial draw for all units, and the units with
  replacement), which keeps the trial noise and the trials that the units
  share. The sampling unit is the unit. Each mouse is shown and tested
  alone.

The script writes `figures/glm/flash_gain_units.svg` and
`results/glm/flash_gain_units.json`.

    THESIS_CACHE_DIR=figures/cache nice -n 10 uv run python -m \\
        thesis.ephys.analyses.glm.flash_gain_units
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy.stats import wilcoxon

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thesis.ephys.analyses.glm import dmat, exit_flash_responses, peaks
from thesis.ephys.analyses.glm.fit import build_unit_counts
from thesis.ephys.analyses.glm.flash_kernels import SESSIONS
from thesis.ephys.units import fetch_unit_table
from thesis.plotting import (
    CONCEPT_COLORS,
    FIGURE_STYLE,
    MOUSE_MARKERS,
    REFERENCE_COLOR,
    label_panels,
    save_figure,
)

OUT = Path("results/glm")
GAP_INDICES = (1,)  # 100-200 ms
PERIODS = ("in port", "later after exit")
BIN_MS = 10
BEFORE_BINS = 4  # -40 to 0 ms
PEAK_BINS = slice(7, 16)  # 30 to 120 ms
BOOTSTRAPS = 500
SPLITS = 50
SEED = 3
COLOR = CONCEPT_COLORS["visual"]


def cell_counts(counts: np.ndarray, windows: dict, n_bins: int) -> dict:
    """Return per period the counts of each flash in 10-ms bins, (flash, bin), and its trial."""
    out = {}
    for period in PERIODS:
        window = windows.get((period, GAP_INDICES[0]))
        if window is None:
            continue
        values = counts[window]  # (flash, 1-ms bin)
        out[period] = (
            values.reshape(len(values), -1, BIN_MS).sum(-1),
            window[:, 0] // n_bins,
        )
    return out


def resampled(cells: dict, drawn: np.ndarray, n_trials: int) -> dict:
    """Return the cells of a trial draw: each flash as many times as its trial was drawn."""
    times = np.bincount(drawn, minlength=n_trials)
    out = {}
    for period, (counts, trials) in cells.items():
        index = np.repeat(np.arange(len(trials)), times[trials])
        out[period] = (counts[index], trials[index])
    return out


def split_peak(counts: np.ndarray, trials: np.ndarray, rng, splits: int):
    """Return the rate before the flash and the split-half peak above it (spikes/s).

    The peak bin is chosen on the flashes of a random half of the trials and
    its value is read on the other half, then the halves swap; the mean over
    `splits` random splits. The maximum of a noisy trace is biased upward,
    more with fewer flashes; reading it on other trials removes that bias.
    """
    to_rate = 1 / (BIN_MS * dmat.BINWIDTH_S)
    before = counts[:, :BEFORE_BINS].mean() * to_rate
    ids = np.unique(trials)
    peaks = []
    for _ in range(splits):
        first = np.isin(trials, rng.permutation(ids)[: len(ids) // 2])
        for pick, read in ((first, ~first), (~first, first)):
            chosen = counts[pick].mean(0)[PEAK_BINS].argmax()
            trace = counts[read].mean(0) * to_rate
            peaks.append(trace[PEAK_BINS][chosen] - trace[:BEFORE_BINS].mean())
    return float(before), float(np.mean(peaks))


def gains(cells: dict, rng, splits: int = SPLITS) -> np.ndarray:
    """Return (rate before in port, g_b, g_f) from the per-flash counts of each period."""
    values = {period: split_peak(*cells[period], rng, splits) for period in PERIODS}
    (b_port, p_port), (b_after, p_after) = (values[p] for p in PERIODS)
    g_f = p_after / p_port if p_port > 0 else np.nan
    return np.asarray([b_port, b_after / b_port if b_port > 0 else np.nan, g_f])


def session_gains(session: str, excited: list[int]) -> dict:
    """Return the gains of each flash-excited unit of one session, with bootstrap intervals."""
    root = OUT / session
    prepared = dmat._load_windows(root / "stimulus_windows_me.npz")
    trials = dmat._load_selected_trials(prepared)
    windows = exit_flash_responses.events(trials, prepared)
    subject, stamp = session.split("_", 1)
    table = fetch_unit_table(subject, stamp, 1, 0, include_metrics=False).set_index(
        "unit_id"
    )
    rng = np.random.default_rng(SEED)
    n_trials, n_bins = len(trials), len(prepared["relative_centers"])
    # The trials with a flash of either period; one draw serves every unit.
    pool = np.unique(
        np.concatenate(
            [windows[(p, g)][:, 0] // n_bins for p in PERIODS for g in GAP_INDICES]
        )
    )
    trial_draws = [rng.choice(pool, len(pool)) for _ in range(BOOTSTRAPS)]
    out = {"unit_ids": [], "gains": [], "intervals": []}
    unit_draws = []
    for unit in excited:
        counts = build_unit_counts(
            prepared["alignments"], table.loc[unit, "spike_times_s"]
        )
        cells = cell_counts(counts, windows, n_bins)
        if len(cells) != len(PERIODS):
            continue
        values = gains(cells, rng)
        draws = np.asarray(
            [gains(resampled(cells, d, n_trials), rng) for d in trial_draws]
        )
        unit_draws.append(draws)
        out["unit_ids"].append(int(unit))
        out["gains"].append(values.tolist())
        out["intervals"].append(np.nanpercentile(draws, [2.5, 97.5], axis=0).T.tolist())
        print(
            f"{subject} unit {unit}: b {values[0]:.1f} sp/s, g_b {values[1]:.2f}, g_f {values[2]:.2f}",
            flush=True,
        )
    table = np.asarray(out["gains"])
    defined = np.isfinite(table[:, 2])
    g_b, g_f = table[defined, 1], table[defined, 2]
    out["flashes"] = {
        f"{p}, gap {exit_flash_responses.GAP_LABELS[g]} ms": int(len(windows[(p, g)]))
        for p in PERIODS
        for g in GAP_INDICES
        if (p, g) in windows
    }
    out["stats"] = {
        "units": int(len(table)),
        "units_without_g_f": int(np.sum(~defined)),
        "median_g_b": float(np.nanmedian(table[:, 1])),
        "median_g_f": float(np.median(g_f)),
        "median_g_f_minus_g_b": float(np.median(g_f - g_b)),
        "wilcoxon_p_g_f_minus_g_b": float(wilcoxon(g_f - g_b).pvalue),
        "units_g_f_above_g_b": int(np.sum(g_f > g_b)),
        "median_g_f_minus_1": float(np.median(g_f - 1)),
        "wilcoxon_p_g_f_minus_1": float(wilcoxon(g_f - 1).pvalue),
        "units_g_f_above_1": int(np.sum(g_f > 1)),
        **population_intervals(np.asarray(unit_draws), rng),
    }
    out["subject"] = subject
    return out


def population_intervals(unit_draws: np.ndarray, rng) -> dict:
    """Return the 95% intervals of the medians over units, from draws of trials and units.

    `unit_draws` is (unit, trial draw, (b, g_b, g_f)); draw b uses trial draw b
    for every unit and the units with replacement.
    """
    medians = {"g_f_minus_g_b": [], "g_f_minus_1": []}
    units = len(unit_draws)
    for b in range(unit_draws.shape[1]):
        picked = unit_draws[rng.integers(0, units, units), b]
        g_b, g_f = picked[:, 1], picked[:, 2]
        both = np.isfinite(g_b) & np.isfinite(g_f)
        medians["g_f_minus_g_b"].append(np.median(g_f[both] - g_b[both]))
        medians["g_f_minus_1"].append(np.median(g_f[np.isfinite(g_f)] - 1))
    return {
        f"median_{name}_95_trials_and_units": np.percentile(
            values, [2.5, 97.5]
        ).tolist()
        for name, values in medians.items()
    }


def plot(sessions: list[dict]) -> Path:
    with plt.rc_context(FIGURE_STYLE):
        figure, axes = plt.subplots(
            1, len(sessions), figsize=(4.0 * len(sessions), 4.0), squeeze=False
        )
        every = np.concatenate(
            [np.asarray(s["intervals"])[:, 1:].reshape(-1, 2) for s in sessions]
        )
        every = every[np.isfinite(every).all(1)]
        low = max(min(every.min(), 0.0), -1.0)
        high = min(every.max(), 8.0)
        limits = (low - 0.2, high + 0.2)
        for axis, s in zip(axes[0], sessions, strict=True):
            gains = np.asarray(s["gains"])
            intervals = np.asarray(s["intervals"])  # (unit, param, low/high)
            marker = MOUSE_MARKERS.get(s["subject"], "o")
            axis.errorbar(
                gains[:, 1],
                gains[:, 2],
                xerr=np.abs(intervals[:, 1].T - gains[:, 1]),
                yerr=np.abs(intervals[:, 2].T - gains[:, 2]),
                fmt=marker,
                color=COLOR,
                ms=3,
                elinewidth=0.5,
                alpha=0.6,
            )
            axis.plot(
                np.nanmedian(gains[:, 1]),
                np.nanmedian(gains[:, 2]),
                marker=marker,
                ms=8,
                color="white",
                mec=COLOR,
                mew=1.5,
            )
            axis.plot(limits, limits, color=REFERENCE_COLOR, ls="--", lw=0.8)
            axis.axhline(1, color=REFERENCE_COLOR, ls=":", lw=0.8)
            axis.set_xlim(limits)
            axis.set_ylim(limits)
            outside = int(
                np.sum((intervals[:, 1:] < limits[0]) | (intervals[:, 1:] > limits[1]))
            )
            st = s["stats"]
            axis.set_title(
                f"{s['subject']}: {st['units']} flash-excited units\n"
                f"g_f - g_b: median {st['median_g_f_minus_g_b']:+.2f}"
                f" [{st['median_g_f_minus_g_b_95_trials_and_units'][0]:+.2f},"
                f" {st['median_g_f_minus_g_b_95_trials_and_units'][1]:+.2f}]\n"
                f"g_f - 1: median {st['median_g_f_minus_1']:+.2f}"
                f" [{st['median_g_f_minus_1_95_trials_and_units'][0]:+.2f},"
                f" {st['median_g_f_minus_1_95_trials_and_units'][1]:+.2f}]"
                + (f"\n{outside} interval ends outside the axes" if outside else ""),
                fontsize=7,
            )
            axis.text(
                limits[1],
                limits[1],
                "pure gain ",
                ha="right",
                va="top",
                fontsize=6,
                color="0.4",
                rotation=45,
                rotation_mode="anchor",
            )
            axis.text(
                limits[1] - 0.1,
                1.08,
                "same flash response",
                ha="right",
                fontsize=6,
                color="0.4",
            )
            axis.set_xlabel(
                "gain of the rate before the flash, g_b\n(after exit over in port)"
            )
        axes[0, 0].set_ylabel("gain of the flash response, g_f")
        figure.tight_layout()
        label_panels(axes.ravel())
        return save_figure(figure, OUT / "flash_gain_units")


def main() -> None:
    excited = peaks.load_excited_units()
    sessions = [session_gains(s, sorted(excited[s])) for s in SESSIONS]
    (OUT / "flash_gain_units.json").write_text(json.dumps(sessions, indent=1))
    print(
        json.dumps(
            [
                {"subject": s["subject"], "flashes": s["flashes"], **s["stats"]}
                for s in sessions
            ],
            indent=1,
        )
    )
    print(plot(sessions))


if __name__ == "__main__":
    main()
