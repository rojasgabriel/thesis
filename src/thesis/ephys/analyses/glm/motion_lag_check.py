"""Check whether lagged motion energy predicts spikes that the same-bin weights miss.

Question: the V1 GLM gives each motion-energy PC one same-bin weight. Does
motion energy at another time predict spikes that this model does not explain?

Comparison: this script makes no new fits. For each unit it correlates a
motion-energy PC at time t − lag with (a) the spike count at t and (b, c) the
residual of the final GLM refit at t (count minus fitted count). Panels a and b
use PC 1, which follows the total motion energy (r = 0.93). Panel c shows PCs 1
to 10, because the later PCs change faster than PC 1. The lags go from −200 to
+200 ms in 10-ms steps. At a positive lag the motion comes before the spikes.

Units: every unit of one fit set of one session (`--units`, `all` by default):
quality criterion set 1, stability set 0, unit 212 excluded.

Bins, alignment, and exclusions: 1-ms bins in windows aligned to the first
measured flash. Spike bins after response entry are excluded. Motion energy is
taken from the same trial window, so a pair that would leave the window is not
used.

Control (panel a): the same correlation with the motion energy of the next
trial. It keeps the average time course of motion energy in a trial and removes
the trial-to-trial variation.

Sampling unit: the unit. Panels a and b show every unit, and the mean and SEM
over units. Panel c shows the mean over units for each PC. All panels have the
same axis range, set by the mean of panel a; single units reach farther.

Reading: a residual correlation as large as the spike-count correlation, at a
lag other than zero, means that lagged motion energy would add to the model.
The correlations are small in absolute terms because single 1-ms bins are
almost all zero. The check does not give a ΔD²; only a fit with lagged columns
does.

    uv run python -m thesis.ephys.analyses.glm.motion_lag_check
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import sem

from thesis.ephys.analyses.glm.dmat import (
    BINWIDTH_S,
    _load_windows,
    _valid_bin_mask,
    build_unit_history,
    training_zscore,
)
from thesis.ephys.analyses.glm.figures import FIGURE_STYLE, _label_panels, _save_figure
from thesis.ephys.analyses.glm.fit import (
    _write_json_atomic,
    build_unit_counts,
    load_unit_records,
)
from thesis.ephys.units import fetch_unit_table

LAGS_MS = np.arange(-200, 201, 10)
# The smallest PC count that a fit can select, so every fit has these PCs.
COMPONENTS = 10
DATA_COLOR = "C1"
CONTROL_COLOR = "0.45"


def lagged_correlation(signal: np.ndarray, motion: np.ndarray) -> np.ndarray:
    """Correlate signal[t] with motion[t − lag] within trials; NaN signal is skipped."""
    values = []
    bins = signal.shape[1]
    for lag in np.rint(LAGS_MS / (1000 * BINWIDTH_S)).astype(int):
        now = slice(max(lag, 0), bins + min(lag, 0))
        then = slice(max(-lag, 0), bins + min(-lag, 0))
        a, b = signal[:, now], motion[:, then]
        keep = np.isfinite(a)
        values.append(np.corrcoef(a[keep], b[keep])[0, 1])
    return np.asarray(values)


def unit_curves(root: Path, unit_set: str) -> dict[str, np.ndarray]:
    """Return the lagged correlations of each unit, for counts and residuals."""
    prepared = _load_windows(root / "stimulus_windows_me.npz")
    with (root / "common_design_me.json").open() as handle:
        base = int(json.load(handle)["base_columns"])
    common = np.load(root / "common_design_me.npy", mmap_mode="r")
    valid = _valid_bin_mask(prepared)
    trials = len(prepared["alignments"])
    motion = [
        np.asarray(common[:, base + component], dtype=float).reshape(trials, -1)
        for component in range(COMPONENTS)
    ]
    other_trial = np.roll(motion[0], 1, axis=0)
    spike_times = fetch_unit_table(
        prepared["metadata"]["subject_name"],
        prepared["metadata"]["session_name"],
        unit_criteria_id=1,
        stability_param_id=0,
        include_metrics=False,
    ).set_index("unit_id")["spike_times_s"]

    curves = {"count": [], "count_control": [], "residual": [], "residual_control": []}
    fits = load_unit_records(
        root / f"{unit_set}_fit_me" / "checkpoints", "unit_*_final.json"
    )
    # One fit set has one PC count, so its units share these design columns.
    columns = base + int(fits[0]["components"])
    design = np.asarray(common[:, :columns])
    for fit in fits:
        spikes = np.asarray(spike_times.loc[int(fit["unit_id"])], dtype=float)
        counts = build_unit_counts(prepared["alignments"], spikes).astype(float)
        history, _, _ = training_zscore(
            build_unit_history(prepared["alignments"], spikes), valid
        )
        coefficients = np.asarray(fit["coefficients"], dtype=float)
        fitted = np.exp(
            fit["intercept"]
            + design @ coefficients[:columns]
            + history @ coefficients[columns:]
        )
        count = np.where(valid, counts, np.nan).reshape(trials, -1)
        residual = np.where(valid, counts - fitted, np.nan).reshape(trials, -1)
        curves["count"].append(lagged_correlation(count, motion[0]))
        curves["count_control"].append(lagged_correlation(count, other_trial))
        curves["residual"].append(
            [lagged_correlation(residual, values) for values in motion]
        )
        curves["residual_control"].append(lagged_correlation(residual, other_trial))
    return {key: np.asarray(values) for key, values in curves.items()}


def _mean_band(axis, values: np.ndarray, color: str) -> None:
    mean, error = values.mean(axis=0), sem(values, axis=0)
    axis.fill_between(
        LAGS_MS, mean - error, mean + error, color=color, alpha=0.3, linewidth=0
    )
    axis.plot(LAGS_MS, mean, color=color, linewidth=1.4)


def plot_lag_check(curves: dict[str, np.ndarray], output: Path) -> Path:
    """Draw the lagged correlations before the model, after it, and for ten PCs."""
    first = curves["residual"][:, 0]
    # One axis range for all panels shows how much of the correlation remains.
    limit = 2.2 * np.abs(curves["count"].mean(axis=0)).max()
    with plt.rc_context(FIGURE_STYLE):
        figure, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), sharex=True)
        axes[0].plot(
            LAGS_MS, curves["count"].T, color=DATA_COLOR, alpha=0.1, linewidth=0.5
        )
        _mean_band(axes[0], curves["count_control"], CONTROL_COLOR)
        _mean_band(axes[0], curves["count"], DATA_COLOR)
        axes[0].text(
            0.03, 0.12, "same trial", color=DATA_COLOR, transform=axes[0].transAxes
        )
        axes[0].text(
            0.03, 0.04, "next trial", color=CONTROL_COLOR, transform=axes[0].transAxes
        )
        axes[0].set_ylabel("correlation of spike count\nwith motion-energy PC 1")

        axes[1].plot(LAGS_MS, first.T, color=DATA_COLOR, alpha=0.1, linewidth=0.5)
        _mean_band(axes[1], first, DATA_COLOR)
        axes[1].set_ylabel("correlation of GLM residual\nwith motion-energy PC 1")

        means = curves["residual"].mean(axis=0)
        for component, mean in enumerate(means):
            axes[2].plot(
                LAGS_MS,
                mean,
                color=DATA_COLOR if component == 0 else "0.5",
                linewidth=1.4 if component == 0 else 0.7,
            )
        axes[2].text(0.03, 0.12, "PC 1", color=DATA_COLOR, transform=axes[2].transAxes)
        axes[2].text(
            0.03,
            0.04,
            f"PCs 2 to {COMPONENTS}",
            color="0.5",
            transform=axes[2].transAxes,
        )
        axes[2].set_ylabel("correlation of GLM residual\nwith one PC (mean over units)")
        for axis in axes:
            axis.axhline(0, color="black", linestyle="--", linewidth=0.5)
            axis.axvline(0, color="0.65", linewidth=0.6, zorder=0)
            axis.set_ylim(-limit, limit)
            axis.set_xlabel("lag of spikes after motion energy (ms)")
        _label_panels(axes, -0.3)
        figure.tight_layout(w_pad=1.5)
        return _save_figure(figure, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("figures/glm/GRB006_20240821_121447"),
        help="Session artifact directory that holds the fit.",
    )
    parser.add_argument(
        "--units",
        default="all",
        help="Fit set to read: all, sample, or sample1 to sample9.",
    )
    args = parser.parse_args()
    curves = unit_curves(args.root, args.units)
    figure = plot_lag_check(curves, args.root / "motion_lag_check")
    near = np.abs(LAGS_MS) <= 100
    count = curves["count"].mean(axis=0)
    residual = curves["residual"].mean(axis=0)
    summary = {
        "units": len(curves["count"]),
        "lags_ms": LAGS_MS.tolist(),
        "count_pc1_mean": count.tolist(),
        "count_pc1_next_trial_mean": curves["count_control"].mean(axis=0).tolist(),
        "residual_mean_by_pc": residual.tolist(),
        # Signs differ between units, so also give the mean size over units.
        # The next-trial value is the size expected with no trial-specific effect.
        "mean_size_within_100_ms": {
            key: float(np.abs(values).mean(axis=0)[near].mean())
            for key, values in (
                ("count_pc1", curves["count"]),
                ("count_pc1_next_trial", curves["count_control"]),
                ("residual_pc1", curves["residual"][:, 0]),
                ("residual_pc1_next_trial", curves["residual_control"]),
            )
        },
        "largest_count_pc1_within_100_ms": float(np.abs(count[near]).max()),
        "largest_residual_within_100_ms_by_pc": np.abs(residual[:, near])
        .max(axis=1)
        .tolist(),
        "largest_residual_within_200_ms_by_pc": np.abs(residual).max(axis=1).tolist(),
    }
    summary["figure"] = str(figure)
    _write_json_atomic(args.root / "motion_lag_check.json", summary)
    shown = {k: v for k, v in summary.items() if "largest" in k or "size" in k}
    print(json.dumps(shown, indent=2))


if __name__ == "__main__":
    main()
