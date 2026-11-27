"""Estimate how fast V1 flash responses recover from adaptation.

Question: the response to a flash depends on the gap from the flash before.
How long does it take to recover, and at what level does it end? The time
constant of the recovery term of the GLM (`dmat.FLASH_RECOVERY_TAU_S`) is
chosen by prediction (`tau_grid.py`), not from this fit.

Comparison: no model. In-port flashes only (from the first flash to the
center exit), so the task period does not change with the gap.

- Flashes: every in-port flash of the completed trials of the GLM whose
  window lies fully inside the valid interval, grouped by the gap from the
  flash before, rounded to the 40-ms flash slot (`dmat.FLASH_SLOT_S`); gaps of
  `LONGEST_GROUP` slots or more are one group, at their median gap. The first
  flash of a trial (no flash before) is its own group, the recovered level.
- Response: the trial mean in 10-ms bins, the window peak from 30 to 120 ms
  after the flash (`peaks.window_peak`, the mean over ±20 ms around the
  largest bin; since 2026-11-15, the largest bin before) minus the mean from
  -40 to 0 ms, divided by the mean observed rate
  of the unit over all valid bins.
- Units: the flash-excited units of `flash_excited_units.json`
  (`peaks.excited_units`), every session of `flash_kernels.SESSIONS`. The
  figure has a thin line for each mouse (the mean of its units) and the mean
  of the mice with its 95% interval (1.96 SEM over the mice).
- Fit: response = A (1 - c exp(-gap / tau)) to the mean of the mice, weighted
  by the SEM over the mice, with c free (`C_MAX` only keeps the fit finite).
  With c > 1 the curve is 0 at the delay tau ln c and rises after it. The fit
  is without the 40-ms group (the response to the flash before falls into its
  baseline and peak windows). The first flash of a trial enters with a gap of
  `FIRST_GAP_S` (fully recovered). The interval of tau is a hierarchical
  bootstrap (mice, then units); a resample with no fit is left out and
  counted. The report also has the fit of each mouse alone (its units, SEM
  over units). Until 2026-11-26 the fit was on GRB006 only, with GRB064 as
  the check.

The script writes `flash_recovery.svg` (figures/glm) and `flash_recovery.json`
(results/glm).

    THESIS_CACHE_DIR=figures/cache nice -n 10 uv run python -m \\
        thesis.ephys.analyses.glm.flash_recovery
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy.optimize import curve_fit

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thesis.ephys.analyses.glm import dmat
from thesis.ephys.analyses.glm.fit import build_unit_counts
from thesis.ephys.analyses.glm.flash_kernels import SESSIONS
from thesis.ephys.analyses.glm.peaks import LAGS_MS, load_excited_units, window_peak
from thesis.ephys.units import fetch_unit_table
from thesis.plotting import (
    CONCEPT_COLORS,
    FIGURE_STYLE,
    OBSERVED_COLOR,
    save_figure,
)

OUT = Path("results/glm")
LONGEST_GROUP = 7
FIRST_GAP_S = 10.0
C_MAX = 50.0
MIN_FLASHES = 15
BOOTSTRAPS = 1000
CENTERS_S = (LAGS_MS.reshape(-1, 10).mean(-1) + 0.5) / 1000
BASELINE = (CENTERS_S >= -0.04) & (CENTERS_S < 0)


def recovery(gap, A, c, tau):
    """Return the recovered response at each gap (s)."""
    return A * (1 - c * np.exp(-np.asarray(gap, dtype=float) / tau))


def session_peaks(session: str, units: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Return the median gap of each group (s) and the peak of each unit (unit, group)."""
    root = OUT / session
    subject, stamp = session.split("_", 1)
    table = fetch_unit_table(subject, stamp, 1, 0, include_metrics=False).set_index(
        "unit_id"
    )
    prepared = dmat._load_windows(root / "stimulus_windows_me.npz")
    trials = dmat._load_selected_trials(prepared)
    n_bins = len(prepared["relative_centers"])
    valid = dmat._valid_bin_mask(prepared)
    rows_valid = valid.reshape(len(trials), -1)
    windows, gaps_of = {}, {}
    for row, (_, trial) in enumerate(trials.iterrows()):
        times = np.asarray(trial["stim_pulse_times_s"], dtype=float)
        gaps = np.diff(times, prepend=-np.inf)
        for time, gap in zip(times, gaps, strict=True):
            if time >= trial["center_exit_s"]:
                continue
            start = time - prepared["alignments"][row] - prepared["relative_centers"][0]
            columns = int(np.rint(start / dmat.BINWIDTH_S)) + LAGS_MS
            if columns[0] < 0 or columns[-1] >= n_bins:
                continue
            if not rows_valid[row, columns].all():
                continue
            slots = round(gap / dmat.FLASH_SLOT_S) if np.isfinite(gap) else np.inf
            key = min(slots, LONGEST_GROUP) if np.isfinite(slots) else np.inf
            windows.setdefault(key, []).append(row * n_bins + columns)
            gaps_of.setdefault(key, []).append(gap if np.isfinite(gap) else FIRST_GAP_S)
    keys = sorted(k for k in windows if len(windows[k]) >= MIN_FLASHES)
    peaks = []
    for unit_id in units:
        counts = build_unit_counts(
            prepared["alignments"], table.loc[unit_id, "spike_times_s"]
        )
        scale = (counts * valid).sum() / (valid.sum() * dmat.BINWIDTH_S)
        row = []
        for key in keys:
            rate = counts[np.asarray(windows[key])] / dmat.BINWIDTH_S / scale
            trace = rate.mean(0).reshape(-1, 10).mean(-1)
            row.append(window_peak(trace, 1000 * CENTERS_S) - trace[BASELINE].mean())
        peaks.append(row)
    return np.asarray([np.median(gaps_of[k]) for k in keys]), np.asarray(peaks)


def fit(gaps, peaks, use, tau=None) -> tuple[float, float, float]:
    """Fit A, c (and tau) to the unit means of the used groups, weighted by SEM."""
    mean = peaks.mean(0)[use]
    sem = peaks.std(0, ddof=1)[use] / np.sqrt(len(peaks))
    if tau is None:
        values, _ = curve_fit(
            recovery,
            gaps[use],
            mean,
            p0=(2, 0.9, 0.1),
            sigma=sem,
            bounds=([0, 0, 0.005], [50, C_MAX, 2]),
        )
        return tuple(float(v) for v in values)
    values, _ = curve_fit(
        lambda g, A, c: recovery(g, A, c, tau),
        gaps[use],
        mean,
        p0=(2, 0.9),
        sigma=sem,
        bounds=([0, 0], [50, C_MAX]),
    )
    return float(values[0]), float(values[1]), float(tau)


def delay(c: float, tau: float) -> float:
    """Return the gap (s) where the fitted curve is 0: tau ln c, or 0 for c <= 1."""
    return float(tau * np.log(max(c, 1.0)))


def tau_interval(gaps, groups, use) -> tuple[list[float], int]:
    """Return the 95% hierarchical bootstrap interval of tau and the failed fits.

    `groups` has the peaks of each mouse (unit, group). Each draw resamples the
    mice, then the units of each drawn mouse, and fits the means of the drawn
    mice. A draw with no fit (no convergence, or the same mouse in every
    place, so no SEM) is left out; the count is reported.
    """
    rng = np.random.default_rng(0)
    taus = []
    for _ in range(BOOTSTRAPS):
        means = np.asarray(
            [
                groups[m][rng.integers(0, len(groups[m]), len(groups[m]))].mean(0)
                for m in rng.integers(0, len(groups), len(groups))
            ]
        )
        try:
            taus.append(fit(gaps, means, use)[2])
        except (RuntimeError, ValueError):
            continue
    return np.percentile(taus, [2.5, 97.5]).tolist(), BOOTSTRAPS - len(taus)


def main() -> None:
    excited = load_excited_units()
    data = {s.split("_")[0]: session_peaks(s, sorted(excited[s])) for s in SESSIONS}
    groups = [p for _, p in data.values()]
    if len({len(g) for g, _ in data.values()}) != 1:
        raise ValueError("The mice do not have the same gap groups.")
    gaps = np.mean([g for g, _ in data.values()], 0)
    means = np.asarray([p.mean(0) for p in groups])  # (mice, gap groups)
    use = gaps > dmat.short_gap_s()  # without the 40-ms group
    A, c, tau = fit(gaps, means, use)
    interval, failed = tau_interval(gaps, groups, use)
    report = {
        "mice": {
            "A": A,
            "c": c,
            "tau_s": tau,
            "tau_interval_95_s": interval,
            "tau_bootstrap_fits_failed": failed,
            "delay_s": delay(c, tau),
            "mean_gap_s": gaps.tolist(),
            "mean_peak": means.mean(0).tolist(),
        }
    }
    for subject, (g, p) in data.items():
        report[subject] = dict(
            units=len(p),
            median_gap_s=g.tolist(),
            mean_peak=p.mean(0).tolist(),
            sem_peak=(p.std(0, ddof=1) / np.sqrt(len(p))).tolist(),
        )
        try:
            own = fit(g, p, g > dmat.short_gap_s())
            report[subject].update(own_tau_s=own[2], own_delay_s=delay(own[1], own[2]))
        except RuntimeError:
            report[subject].update(own_tau_s=None)
    curve_gaps = np.linspace(0.03, 0.6, 300)
    color = CONCEPT_COLORS["visual"]
    first = gaps >= FIRST_GAP_S
    x = np.where(first, 650, 1000 * gaps)
    mean = means.mean(0)
    half = 1.96 * means.std(0, ddof=1) / np.sqrt(len(means))
    with plt.rc_context(FIGURE_STYLE):
        figure, axis = plt.subplots(figsize=(4.2, 3.0), layout="constrained")
        for row in means:  # a thin line for each mouse
            axis.plot(x[~first], row[~first], color=color, lw=0.8, alpha=0.45)
            axis.plot(x[first], row[first], "s", color=color, ms=3, alpha=0.45)
        for xi, mi, ei, is_first, used in zip(x, mean, half, first, use, strict=True):
            axis.errorbar(
                [xi],
                [mi],
                yerr=[ei],
                fmt="s" if is_first else "o",
                color=color,
                mfc=color if used and not is_first else "white",
                ms=5,
                capsize=2,
                lw=1.2,
                zorder=4,
            )
        axis.plot(
            1000 * curve_gaps,
            recovery(curve_gaps, A, c, tau),
            color=OBSERVED_COLOR,
            lw=1.4,
            zorder=3,
        )
        axis.plot([600, 700], [A, A], color=OBSERVED_COLOR, lw=1.4, ls=":")
        axis.text(
            0.97,
            0.04,
            f"{sum(len(p) for p in groups)} flash-excited units, {len(groups)} mice\n"
            f"fit without 40 ms: τ = {1000 * tau:.0f} ms\n"
            "open circle (40 ms): not fit",
            transform=axis.transAxes,
            fontsize=7,
            color=OBSERVED_COLOR,
            ha="right",
            va="bottom",
        )
        axis.set_xlabel("gap from the flash before (ms)")
        axis.set_xticks(
            [40, 160, 280, 400, 520, 650],
            ["40", "160", "280", "400", "520", "first\nof trial"],
        )
        axis.set_ylabel("flash peak (/ mean rate)")
        axis.set_ylim(bottom=min(0, (mean - half).min() * 1.1))
        print(save_figure(figure, OUT / "flash_recovery"))
    (OUT / "flash_recovery.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report["mice"], indent=1))
    print({s: r.get("own_tau_s") for s, r in report.items() if s != "mice"})


if __name__ == "__main__":
    main()
