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
- Response: the trial mean in 10-ms bins, the peak from 30 to 120 ms after
  the flash minus the mean from -40 to 0 ms, divided by the mean observed rate
  of the unit over all valid bins.
- Units: the flash-excited units of `raw_video_trust_classification.json`.
  The sampling unit is the unit; points are the mean over units with a 95%
  interval (1.96 SEM).
- Fit: response = A (1 - c exp(-gap / tau)), weighted by the SEM of each
  group, with c free (`C_MAX` only keeps the fit finite). With c > 1 the curve
  is 0 at the delay tau ln c and rises after it. Until 2026-10-24 c had a bound
  of 1.5, which set tau to about 120 ms. The fit is on GRB006 only, without
  the 40-ms group (the response to the flash before falls into its baseline
  and peak windows). The first flash of a trial
  enters with a gap of `FIRST_GAP_S` (fully recovered). tau has a bootstrap
  interval over units. GRB064 is the check: tau is fixed from GRB006 and only
  A and c are fit; its own tau is also given.

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
from thesis.ephys.analyses.glm.peaks import LAGS_MS
from thesis.ephys.units import fetch_unit_table
from thesis.plotting import (
    CONCEPT_COLORS,
    FIGURE_STYLE,
    OBSERVED_COLOR,
    label_panels,
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
PEAK = (CENTERS_S >= 0.03) & (CENTERS_S < 0.12)


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
            row.append(trace[PEAK].max() - trace[BASELINE].mean())
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


def tau_interval(gaps, peaks, use) -> list[float]:
    """Return the 95% bootstrap interval of tau over units."""
    rng = np.random.default_rng(0)
    taus = []
    for _ in range(BOOTSTRAPS):
        sample = peaks[rng.integers(0, len(peaks), len(peaks))]
        try:
            taus.append(fit(gaps, sample, use)[2])
        except RuntimeError:
            continue
    return np.percentile(taus, [2.5, 97.5]).tolist()


def main() -> None:
    excited = json.loads((OUT / "raw_video_trust_classification.json").read_text())
    data = {s.split("_")[0]: session_peaks(s, sorted(excited[s])) for s in SESSIONS}
    report = {}
    gaps, peaks = data["GRB006"]
    use = gaps > dmat.short_gap_s()  # without the 40-ms group
    A, c, tau = fit(gaps, peaks, use)
    report["GRB006"] = {
        "A": A,
        "c": c,
        "tau_s": tau,
        "tau_interval_95_s": tau_interval(gaps, peaks, use),
        "delay_s": delay(c, tau),
    }
    gaps64, peaks64 = data["GRB064"]
    use64 = gaps64 > dmat.short_gap_s()
    A64, c64, _ = fit(gaps64, peaks64, use64, tau=tau)
    own = fit(gaps64, peaks64, use64)
    report["GRB064"] = {
        "A_with_GRB006_tau": A64,
        "c_with_GRB006_tau": c64,
        "own_tau_s": own[2],
        "own_delay_s": delay(own[1], own[2]),
        "own_tau_interval_95_s": tau_interval(gaps64, peaks64, use64),
    }
    for subject, (g, p) in data.items():
        report[subject].update(
            units=len(p),
            median_gap_s=g.tolist(),
            mean_peak=p.mean(0).tolist(),
            sem_peak=(p.std(0, ddof=1) / np.sqrt(len(p))).tolist(),
        )
    curve_gaps = np.linspace(0.03, 0.6, 300)
    with plt.rc_context(FIGURE_STYLE):
        figure, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), layout="constrained")
        for axis, subject, (a, cc), note in (
            (
                axes[0],
                "GRB006",
                (A, c),
                f"fit without 40 ms: τ = {1000 * tau:.0f} ms, 0 until {1000 * delay(c, tau):.0f} ms",
            ),
            (
                axes[1],
                "GRB064",
                (A64, c64),
                f"τ fixed from GRB006 ({1000 * tau:.0f} ms)",
            ),
        ):
            g, p = data[subject]
            color = CONCEPT_COLORS["visual"]  # flash peaks; the panel is the mouse
            mean = p.mean(0)
            interval = 1.96 * p.std(0, ddof=1) / np.sqrt(len(p))
            first = g >= FIRST_GAP_S
            x = np.where(first, 650, 1000 * g)
            for xi, mi, ei, is_first, gi in zip(
                x, mean, interval, first, g, strict=True
            ):
                fitted = not is_first and gi > dmat.short_gap_s()
                axis.errorbar(
                    [xi],
                    [mi],
                    yerr=[ei],
                    fmt="s" if is_first else "o",
                    color=color,
                    mfc=color if fitted else "white",
                    ms=4.5,
                    capsize=2,
                    lw=1,
                )
            axis.plot(
                1000 * curve_gaps, recovery(curve_gaps, a, cc, tau), color=color, lw=1.2
            )
            axis.plot([600, 700], [a, a], color=color, lw=1.2, ls=":")
            axis.text(
                0.04,
                0.96,
                f"{subject}, {len(p)} flash-excited units, in port\n{note}",
                transform=axis.transAxes,
                va="top",
                fontsize=7,
                color=OBSERVED_COLOR,
            )
            axis.text(
                0.97,
                0.04,
                "open circle (40 ms): not fit,\nthe flash before overlaps it",
                transform=axis.transAxes,
                fontsize=6,
                color=OBSERVED_COLOR,
                ha="right",
                va="bottom",
            )
            axis.set_xlabel("gap from the flash before (ms)")
            axis.set_xticks(
                [40, 160, 280, 400, 520, 650],
                ["40", "160", "280", "400", "520", "first\nof trial"],
            )
        axes[0].set_ylabel("flash peak (/ mean rate)")
        # One scale for both mice.
        top = max(
            (p.mean(0) + 1.96 * p.std(0, ddof=1) / np.sqrt(len(p))).max()
            for _, p in data.values()
        )
        for axis in axes:
            axis.set_ylim(0, top * 1.25)
        label_panels(axes, x=-0.2, y=1.05)
        print(save_figure(figure, OUT / "flash_recovery"))
    (OUT / "flash_recovery.json").write_text(json.dumps(report, indent=1))
    print(
        json.dumps(
            {s: {k: v for k, v in r.items() if "tau" in k} for s, r in report.items()},
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
