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
flashes and read on the other half, minus that rate (split half, both ways,
mean over `SPLITS` random splits; the largest bin of a noisy trace is biased
upward, more in the period with fewer flashes, which would push g_f toward
g_b).
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
- Units: the flash-excited units of `raw_video_trust_classification.json`.
- Intervals: `BOOTSTRAPS` resamples of the flashes of each period (one
  split each), 95%,
  error bars in both directions. A unit whose in-port peak is ≤ 0 has no
  g_f and is counted.
- Statistics: per mouse, two-sided Wilcoxon signed-rank tests over units of
  g_f - g_b (pure gain at 0) and of g_f - 1 (unchanged response at 0). The
  sampling unit is the unit. Each mouse is shown and tested alone.

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

from thesis.ephys.analyses.glm import dmat, exit_flash_responses
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
BOOTSTRAPS = 200
SPLITS = 50
SEED = 3
COLOR = CONCEPT_COLORS["visual"]


def cell_counts(counts: np.ndarray, windows: dict) -> dict:
    """Return per (period, gap) the counts of each flash in 10-ms bins, (flash, bin)."""
    out = {}
    for period in PERIODS:
        for gap in GAP_INDICES:
            window = windows.get((period, gap))
            if window is None:
                continue
            values = counts[window]  # (flash, 1-ms bin)
            out[(period, gap)] = values.reshape(len(values), -1, BIN_MS).sum(-1)
    return out


def split_peak(counts: np.ndarray, rng, splits: int) -> tuple[float, float]:
    """Return the rate before the flash and the split-half peak above it (spikes/s).

    The peak bin is chosen on a random half of the flashes and its value is
    read on the other half, then the halves swap; the mean over `splits`
    random splits. The maximum of a noisy trace is biased upward, more with
    fewer flashes; reading it on other flashes removes that bias.
    """
    to_rate = 1 / (BIN_MS * dmat.BINWIDTH_S)
    before = counts[:, :BEFORE_BINS].mean() * to_rate
    peaks = []
    for _ in range(splits):
        order = rng.permutation(len(counts))
        halves = (order[: len(order) // 2], order[len(order) // 2 :])
        for pick, read in (halves, halves[::-1]):
            chosen = counts[pick].mean(0)[PEAK_BINS].argmax()
            trace = counts[read].mean(0) * to_rate
            peaks.append(trace[PEAK_BINS][chosen] - trace[:BEFORE_BINS].mean())
    return float(before), float(np.mean(peaks))


def gains(cells: dict, rng, splits: int = SPLITS) -> np.ndarray:
    """Return (rate before in port, g_b, g_f) from the per-flash counts of each period."""
    values = {
        period: split_peak(cells[(period, GAP_INDICES[0])], rng, splits)
        for period in PERIODS
    }
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
    out = {"unit_ids": [], "gains": [], "intervals": []}
    for unit in excited:
        counts = build_unit_counts(
            prepared["alignments"], table.loc[unit, "spike_times_s"]
        )
        cells = cell_counts(counts, windows)
        if len(cells) != len(PERIODS) * len(GAP_INDICES):
            continue
        values = gains(cells, rng)
        draws = np.asarray(
            [
                gains(
                    {k: v[rng.integers(0, len(v), len(v))] for k, v in cells.items()},
                    rng,
                    1,
                )
                for _ in range(BOOTSTRAPS)
            ]
        )
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
        "median_g_b": float(np.median(table[:, 1])),
        "median_g_f": float(np.median(g_f)),
        "median_g_f_minus_g_b": float(np.median(g_f - g_b)),
        "wilcoxon_p_g_f_minus_g_b": float(wilcoxon(g_f - g_b).pvalue),
        "units_g_f_above_g_b": int(np.sum(g_f > g_b)),
        "median_g_f_minus_1": float(np.median(g_f - 1)),
        "wilcoxon_p_g_f_minus_1": float(wilcoxon(g_f - 1).pvalue),
        "units_g_f_above_1": int(np.sum(g_f > 1)),
    }
    out["subject"] = subject
    return out


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
                f" (p = {st['wilcoxon_p_g_f_minus_g_b']:.2g})\n"
                f"g_f - 1: median {st['median_g_f_minus_1']:+.2f}"
                f" (p = {st['wilcoxon_p_g_f_minus_1']:.2g})"
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
    excited = json.loads((OUT / "raw_video_trust_classification.json").read_text())
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
