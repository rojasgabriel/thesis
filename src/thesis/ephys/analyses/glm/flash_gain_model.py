"""Is the larger flash response after the exit one gain on all activity?

Question: flash responses are larger after the center exit (result 1 of
`docs/glm.md`). For each unit, is that the same gain that multiplies its
activity before the flash (pure gain), only a change of that activity with
the same flash response (additive), or something else? A small model per
unit, before the GLM.

Model, per unit, in the counts of 10-ms bins from -40 to 120 ms around a
flash (identity link, Poisson):

- in port: rate = b + f_g(t), with f_g(t) = 0 before the flash;
- after the exit: rate = g_b · b + g_f · f_g(t).

b is the rate before an in-port flash; f_g(t) is the flash response at each
bin, free, one shape per gap class g (shared by both periods); g_b is the
gain of the activity before the flash after the exit, and g_f the gain of the
flash response. g_f = g_b is a pure gain on all activity; g_f = 1 is an
unchanged flash response on a changed background. The shape is fit with the
gains, so its noise does not bias g_f toward 0.

- Flashes: completed GLM trials, windows fully inside the valid interval;
  gap classes 50-100 and 100-200 ms (`exit_flash_responses.GAPS`); the first
  flash of a trial, short-gap flashes (50 ms or less) and the first flash
  after the exit are left out (the first after the exit is larger than any
  gain, `exit_flash_responses.py`).
- Units: the flash-excited units of `raw_video_trust_classification.json`.
- Fit: maximum likelihood on the summed counts of each period, gap class and
  bin (the counts of single flashes add up under a Poisson model); rates
  below `FLOOR` are set to it. Intervals: `BOOTSTRAPS` resamples of the
  flashes within each period and gap class, 95%; error bars in both
  directions.
- Statistics: per mouse, two-sided Wilcoxon signed-rank tests over units of
  log(g_f / g_b) (pure gain at 0) and of log g_f (unchanged response at 0).
  The sampling unit is the unit. Each mouse is shown and tested alone.

The script writes `figures/glm/flash_gain_model.svg` and
`results/glm/flash_gain_model.json`.

    THESIS_CACHE_DIR=figures/cache nice -n 10 uv run python -m \\
        thesis.ephys.analyses.glm.flash_gain_model
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy.optimize import minimize
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
GAP_INDICES = (0, 1)  # 50-100 and 100-200 ms
PERIODS = ("in port", "later after exit")
BIN_MS = 10
BEFORE_BINS = 4  # -40 to 0 ms
FLOOR = 1e-4  # spikes per 1-ms bin
BOOTSTRAPS = 200
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


def negative_log_likelihood(params: np.ndarray, sums: dict, shape_bins: int) -> float:
    """Poisson negative log likelihood of the summed counts (per 10-ms bin)."""
    b, g_b, g_f = params[:3]
    shapes = params[3:].reshape(len(GAP_INDICES), shape_bins)
    total = 0.0
    for (period, gap), (count, flashes) in sums.items():
        f = np.concatenate([np.zeros(BEFORE_BINS), shapes[GAP_INDICES.index(gap)]])
        if period == PERIODS[0]:
            rate = b + f
        else:
            rate = g_b * b + g_f * f
        rate = np.maximum(rate, FLOOR * BIN_MS)
        total -= float(np.sum(count * np.log(rate) - flashes * rate))
    return total


def fit_gains(cells: dict) -> np.ndarray:
    """Return (b, g_b, g_f) of the model fit to the per-flash counts of each cell."""
    sums = {k: (v.sum(0), len(v)) for k, v in cells.items()}
    shape_bins = next(iter(cells.values())).shape[1] - BEFORE_BINS
    port = [sums[(PERIODS[0], g)] for g in GAP_INDICES]
    b0 = max(
        np.mean([c[:BEFORE_BINS].sum() / (n * BEFORE_BINS) for c, n in port]),
        FLOOR * BIN_MS,
    )
    shapes0 = np.concatenate([c[BEFORE_BINS:] / n - b0 for c, n in port])
    start = np.concatenate([[b0, 1.0, 1.0], shapes0])
    bounds = [(FLOOR * BIN_MS, None), (1e-3, 100), (-100, 100)] + [(None, None)] * len(
        shapes0
    )
    result = minimize(
        negative_log_likelihood,
        start,
        args=(sums, shape_bins),
        method="L-BFGS-B",
        bounds=bounds,
    )
    return result.x[:3]


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
        gains = fit_gains(cells)
        draws = np.asarray(
            [
                fit_gains(
                    {k: v[rng.integers(0, len(v), len(v))] for k, v in cells.items()}
                )
                for _ in range(BOOTSTRAPS)
            ]
        )
        out["unit_ids"].append(int(unit))
        out["gains"].append(gains.tolist())
        out["intervals"].append(np.percentile(draws, [2.5, 97.5], axis=0).T.tolist())
        print(
            f"{subject} unit {unit}: b {gains[0] * 100:.1f} sp/s, g_b {gains[1]:.2f}, g_f {gains[2]:.2f}",
            flush=True,
        )
    gains = np.asarray(out["gains"])
    ratio = np.log(gains[:, 2] / gains[:, 1])
    finite = np.isfinite(ratio) & (gains[:, 2] > 0)
    out["flashes"] = {
        f"{p}, gap {exit_flash_responses.GAP_LABELS[g]} ms": int(len(windows[(p, g)]))
        for p in PERIODS
        for g in GAP_INDICES
        if (p, g) in windows
    }
    out["stats"] = {
        "units": len(gains),
        "median_g_b": float(np.median(gains[:, 1])),
        "median_g_f": float(np.median(gains[:, 2])),
        "units_with_g_f_at_or_below_0": int(np.sum(gains[:, 2] <= 0)),
        "median_log_g_f_over_g_b": float(np.median(ratio[finite])),
        "wilcoxon_p_log_g_f_over_g_b": float(wilcoxon(ratio[finite]).pvalue),
        "median_log_g_f": float(np.median(np.log(gains[finite, 2]))),
        "wilcoxon_p_log_g_f": float(wilcoxon(np.log(gains[finite, 2])).pvalue),
        "units_g_f_above_g_b": int(np.sum(gains[:, 2] > gains[:, 1])),
    }
    out["subject"] = subject
    return out


def plot(sessions: list[dict]) -> Path:
    with plt.rc_context(FIGURE_STYLE):
        figure, axes = plt.subplots(
            1, len(sessions), figsize=(3.4 * len(sessions), 3.4), squeeze=False
        )
        every = np.concatenate(
            [np.asarray(s["intervals"]).reshape(-1, 2) for s in sessions]
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
                np.median(gains[:, 1]),
                np.median(gains[:, 2]),
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
            outside = int(np.sum((intervals < limits[0]) | (intervals > limits[1])))
            st = s["stats"]
            axis.set_title(
                f"{s['subject']}: {st['units']} flash-excited units", fontsize=8
            )
            axis.text(
                0.97,
                0.03,
                f"median g_f / g_b {np.exp(st['median_log_g_f_over_g_b']):.2f}"
                f" (p = {st['wilcoxon_p_log_g_f_over_g_b']:.2g})"
                + (f"\n{outside} interval ends outside the axes" if outside else ""),
                transform=axis.transAxes,
                ha="right",
                fontsize=6,
                color="0.3",
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
                limits[0] + 0.1, 1.05, "same flash response", fontsize=6, color="0.4"
            )
            axis.set_xlabel(
                "gain of the rate before the flash, g_b\n(after exit over in port)"
            )
        axes[0, 0].set_ylabel("gain of the flash response, g_f")
        figure.tight_layout()
        label_panels(axes.ravel())
        return save_figure(figure, OUT / "flash_gain_model")


def main() -> None:
    excited = json.loads((OUT / "raw_video_trust_classification.json").read_text())
    sessions = [session_gains(s, sorted(excited[s])) for s in SESSIONS]
    (OUT / "flash_gain_model.json").write_text(json.dumps(sessions, indent=1))
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
