"""Check what a fitted V1 GLM still gets wrong, to justify its regressors.

Question: do the residuals of a fit have structure that a regressor not in the
model would explain? Each check compares observed spike counts with the counts
that the final refit predicts, as the ratio observed / predicted, along one
variable. A ratio near 1 at every value means that the model needs no term for
that variable.

Comparison: no new GLM fits. The script reads one fit set (`--units`) in one
session directory (`--root`). For each check it fits a small Poisson model on
top of the prediction (offset log predicted count) and reports the deviance
that it gains, as a fraction of the unit's null deviance (ΔD²), next to the
same value with the labels shuffled over trials:

- session drift: a cubic spline over trial number; shuffle = trial order.
- time in trial: a cubic spline over the time from the first flash, 0 to 1.5 s.
  It has no shuffle; compare it with the ΔD² of the regressors in the model.
- flash rate: one time-in-trial spline for low-rate (< 12 Hz) and one for
  high-rate (> 12 Hz) trials, minus one shared spline; shuffle = rate labels.
- response side: one scale per response side in each 100 ms window from 300 ms
  before to 600 ms after the withdrawal, minus one common scale; shuffle =
  side labels. Reported for the windows before and after the withdrawal.

Units, trials, alignment (first flash), baseline (constant rate), and
exclusions (bins after response entry) are those of `fit.py`. The sampling unit
is the unit. Every ΔD² is in-sample: the small model is fit and scored on the
same bins, so the shuffle value, not zero, is the reference.

- Panel a: ratio against trial number (deciles of the session).
- Panel b: ratio against time from the first flash, by flash-rate group.
- Panel c: ratio against time from the withdrawal, by response side.
- Panel d: ΔD² of each check for each unit (colored) and its shuffle (gray).
  The axis is logarithmic and starts at `FLOOR`; a smaller value is drawn at
  the floor. The statistics in the JSON use the real values.
Panels a to c show the median and the interquartile range over units.

    uv run python -m thesis.ephys.analyses.glm.residual_checks --units all
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import xlogy
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import SplineTransformer

from thesis.ephys.analyses.glm import cli, dmat
from thesis.ephys.analyses.glm.fit import build_unit_counts, load_unit_records
from thesis.ephys.units import fetch_unit_table
from thesis.plotting import (
    CONCEPT_COLORS,
    FIGURE_STYLE,
    REFERENCE_COLOR,
    label_panels,
    save_figure,
)

BOUNDARY_HZ = 12
SHUFFLES = 30
FLOOR = 1e-5
SESSION_BINS = 10
ONSET_EDGES_S = np.arange(-0.1, 1.001, 0.05)
EXIT_EDGES_S = np.arange(-0.3, 0.601, 0.05)
ENVELOPE_EDGES_S = np.arange(0, 1.501, 0.02)
SIDE_EDGES_S = np.arange(-0.3, 0.601, 0.1)


def _gain(observed: np.ndarray, predicted: np.ndarray, features: np.ndarray) -> float:
    """Deviance gained by exp(features @ w) on top of the predicted counts."""
    keep = predicted > 0
    observed, predicted, features = observed[keep], predicted[keep], features[keep]
    scale = (
        PoissonRegressor(alpha=0, max_iter=2000)
        .fit(features, observed / predicted, sample_weight=predicted)
        .predict(features)
    )
    return float(2 * np.sum(xlogy(observed, scale) - predicted * (scale - 1)))


def _spline(values: np.ndarray, knots: int) -> np.ndarray:
    return SplineTransformer(n_knots=knots, degree=3, include_bias=False).fit_transform(
        np.asarray(values, dtype=float)[:, None]
    )


def _side_gain(observed, predicted, right) -> np.ndarray:
    """Per window: deviance of one scale per side minus one common scale."""
    total = observed.sum(0) / predicted.sum(0)
    gain = np.zeros(observed.shape[1])
    for side in (right, ~right):
        count, expected = observed[side].sum(0), predicted[side].sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            gain += 2 * xlogy(count, count / expected / total)
    return np.nan_to_num(gain)


def _binned(values, index, keep, size) -> np.ndarray:
    """Sum `values` over the kept bins of each trial into `size` clock bins."""
    out = np.zeros((len(values), size))
    for trial in range(len(values)):
        use = keep[trial]
        out[trial] = np.bincount(
            index[trial, use], weights=values[trial, use], minlength=size
        )
    return out


def _ratio(observed, predicted, trials) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return observed[trials].sum(0) / predicted[trials].sum(0)


def run_checks(root: Path, unit_set: str) -> dict:
    """Return the residual curves and the ΔD² of each check for every unit."""
    model = cli.PoissonGLM(root=root)
    prepared = dmat._load_windows(model.windows)
    trials = dmat._load_selected_trials(prepared)
    with model.design.with_suffix(".json").open() as handle:
        metadata = json.load(handle)
    common = np.load(model.design, mmap_mode="r", allow_pickle=False)
    valid = dmat._valid_bin_mask(prepared)
    alignments = prepared["alignments"]
    n_trials = len(alignments)
    grid = (n_trials, len(valid) // n_trials)
    valid_grid = valid.reshape(grid)
    relative = np.broadcast_to(prepared["relative_centers"], grid)
    from_exit = (
        relative - (trials["center_exit_s"].to_numpy(float) - alignments)[:, None]
    )
    rate = trials["visual_stim_rate_hz"].to_numpy(dtype=float)
    low, high = rate < BOUNDARY_HZ, rate > BOUNDARY_HZ
    right = trials["response"].to_numpy() == 1
    rng = np.random.default_rng(0)
    order = np.arange(n_trials)
    decile = order * SESSION_BINS // n_trials

    def clock(times, edges):
        index = np.floor((times - edges[0]) / (edges[1] - edges[0])).astype(int)
        return index, valid_grid & (index >= 0) & (index < len(edges) - 1)

    onset, exit_, envelope, side = (
        clock(relative, ONSET_EDGES_S),
        clock(from_exit, EXIT_EDGES_S),
        clock(relative, ENVELOPE_EDGES_S),
        clock(from_exit, SIDE_EDGES_S),
    )
    session_features = _spline(order, 6)
    envelope_features = _spline(np.arange(len(ENVELOPE_EDGES_S) - 1), 8)
    before = SIDE_EDGES_S[:-1] < 0
    rate_trials = np.flatnonzero(low | high)

    units = fetch_unit_table(
        prepared["metadata"]["subject_name"],
        prepared["metadata"]["session_name"],
        unit_criteria_id=1,
        stability_param_id=0,
        include_metrics=False,
    ).set_index("unit_id")
    records = load_unit_records(
        model.fit_dir(unit_set) / "checkpoints", "unit_*_final.json"
    )
    curves = {
        key: []
        for key in (
            "session",
            "all",
            "low",
            "high",
            "left",
            "right",
            "observed",
            "predicted",
        )
    }
    # Valid 1-ms bins behind each onset-clock bin, for rates.
    exposure = _binned(valid_grid.astype(float), *onset, len(ONSET_EDGES_S) - 1).sum(0)
    checks = []
    for record in records:
        spikes = np.asarray(units.loc[record["unit_id"], "spike_times_s"], dtype=float)
        counts = build_unit_counts(alignments, spikes).astype(float)
        history, _, _ = dmat.training_zscore(
            dmat.build_unit_history(alignments, spikes), valid
        )
        columns = int(metadata["base_columns"]) + int(record["components"])
        weights = np.asarray(record["coefficients"], dtype=float)
        predicted = np.exp(
            record["intercept"]
            + np.asarray(common[:, :columns]) @ weights[:columns]
            + history @ weights[columns:]
        )
        null = counts[valid]
        null = 2 * np.sum(xlogy(null, null / null.mean()) - (null - null.mean()))
        counts, predicted = counts.reshape(grid), predicted.reshape(grid)

        def sums(index_keep, size):
            return (
                _binned(counts, *index_keep, size),
                _binned(predicted, *index_keep, size),
            )

        trial_n = np.where(valid_grid, counts, 0).sum(1)
        trial_m = np.where(valid_grid, predicted, 0).sum(1)
        onset_n, onset_m = sums(onset, len(ONSET_EDGES_S) - 1)
        exit_n, exit_m = sums(exit_, len(EXIT_EDGES_S) - 1)
        env_n, env_m = sums(envelope, len(ENVELOPE_EDGES_S) - 1)
        side_n, side_m = sums(side, len(SIDE_EDGES_S) - 1)

        curves["session"].append(
            np.bincount(decile, trial_n, SESSION_BINS)
            / np.bincount(decile, trial_m, SESSION_BINS)
        )
        curves["all"].append(_ratio(onset_n, onset_m, slice(None)))
        # Rates relative to the mean rate of the unit over all valid bins.
        mean_count = trial_n.sum() / valid.sum()
        curves["observed"].append(onset_n.sum(0) / exposure / mean_count)
        curves["predicted"].append(onset_m.sum(0) / exposure / mean_count)
        curves["low"].append(_ratio(onset_n, onset_m, low))
        curves["high"].append(_ratio(onset_n, onset_m, high))
        curves["left"].append(_ratio(exit_n, exit_m, ~right))
        curves["right"].append(_ratio(exit_n, exit_m, right))

        def rate_gain(first, second):
            both = np.r_[first, second]
            return (
                _gain(env_n[first].sum(0), env_m[first].sum(0), envelope_features)
                + _gain(env_n[second].sum(0), env_m[second].sum(0), envelope_features)
                - _gain(env_n[both].sum(0), env_m[both].sum(0), envelope_features)
            )

        shuffled_rate, shuffled_side, shuffled_session = [], [], []
        for _ in range(SHUFFLES):
            mixed = rng.permutation(rate_trials)
            shuffled_rate.append(rate_gain(mixed[: low.sum()], mixed[low.sum() :]))
            shuffled_side.append(_side_gain(side_n, side_m, rng.permutation(right)))
            mixed = rng.permutation(n_trials)
            shuffled_session.append(
                _gain(trial_n[mixed], trial_m[mixed], session_features)
            )
        side_real = _side_gain(side_n, side_m, right)
        side_null = np.mean(shuffled_side, axis=0)
        checks.append(
            {
                "unit_id": int(record["unit_id"]),
                "session drift": _gain(trial_n, trial_m, session_features) / null,
                "session drift shuffle": float(np.mean(shuffled_session)) / null,
                "time in trial": _gain(env_n.sum(0), env_m.sum(0), envelope_features)
                / null,
                "flash rate": rate_gain(np.flatnonzero(low), np.flatnonzero(high))
                / null,
                "flash rate shuffle": float(np.mean(shuffled_rate)) / null,
                "side, before withdrawal": float(side_real[before].sum()) / null,
                "side, before withdrawal shuffle": float(side_null[before].sum())
                / null,
                "side, after withdrawal": float(side_real[~before].sum()) / null,
                "side, after withdrawal shuffle": float(side_null[~before].sum())
                / null,
            }
        )
        print(f"Checked unit {record['unit_id']}.", flush=True)
    entry = trials["center_entry_s"].to_numpy(float) - alignments
    return {
        "curves": {key: np.asarray(value) for key, value in curves.items()},
        "checks": checks,
        # Median time of center entry and center exit from the first flash.
        "hold_s": [
            float(np.median(entry)),
            float(np.median(trials["center_exit_s"].to_numpy(float) - alignments)),
        ],
    }


def _band(axis, x, values, color, label, label_at=-1, dy=0.0) -> None:
    low, median, high = np.nanpercentile(values, [25, 50, 75], axis=0)
    axis.fill_between(x, low, high, color=color, alpha=0.18, linewidth=0)
    axis.plot(x, median, color=color, linewidth=1.2)
    if label:
        axis.text(
            x[label_at], median[label_at] + dy, f" {label}", color=color, va="center"
        )


def plot(data: dict, output: Path | None, axes=None) -> tuple[Path | None, dict]:
    """Draw the four panels; return the figure path and the summary statistics.

    With `axes`, draw into them and do not label or save (for a combined figure).
    """
    curves, checks = data["curves"], data["checks"]
    names = [key for key in checks[0] if key != "unit_id" and "shuffle" not in key]
    summary = {"units": len(checks)}
    for name in names:
        real = np.asarray([item[name] for item in checks])
        summary[name] = {"median": float(np.median(real))}
        if f"{name} shuffle" in checks[0]:
            null = np.asarray([item[f"{name} shuffle"] for item in checks])
            summary[name] |= {
                "shuffle_median": float(np.median(null)),
                "fraction_units_above_shuffle": float(np.mean(real > null)),
            }
    visual, task = CONCEPT_COLORS["visual"], CONCEPT_COLORS["task"]
    centers = lambda edges: 1000 * (edges[:-1] + edges[1:]) / 2  # noqa: E731
    rng = np.random.default_rng(0)

    with plt.rc_context(FIGURE_STYLE):
        figure = None
        if axes is None:
            figure, axes = plt.subplots(2, 2, figsize=(7.4, 5.4), layout="constrained")
            axes = axes.ravel()
        _band(axes[0], np.arange(1, SESSION_BINS + 1), curves["session"], "black", "")
        axes[0].set_xlabel("trial number (decile of the session)")
        x = centers(ONSET_EDGES_S)
        _band(axes[1], x, curves["low"], visual, "low rate", 4, -0.12)
        _band(axes[1], x, curves["high"], "0.3", "high rate", 4, 0.12)
        axes[1].set_xlabel("time from first flash (ms)")
        x = centers(EXIT_EDGES_S)
        _band(axes[2], x, curves["left"], task, "left", dy=-0.06)
        _band(axes[2], x, curves["right"], "0.3", "right", dy=0.06)
        axes[2].axvline(0, color="black", linewidth=0.45)
        axes[2].set_xlabel("time from withdrawal (ms)")
        axes[2].set_xlim(right=x[-1] + 160)
        for axis in axes[:3]:
            axis.axhline(1, color=REFERENCE_COLOR, linestyle="--", linewidth=0.8)
            axis.set_ylabel("observed / predicted spikes")

        for position, name in enumerate(names):
            real = np.asarray([item[name] for item in checks])
            jitter = rng.uniform(-0.12, 0.12, len(real))
            if f"{name} shuffle" in checks[0]:
                null = np.asarray([item[f"{name} shuffle"] for item in checks])
                axes[3].scatter(
                    position + 0.2 + jitter,
                    np.maximum(null, FLOOR),
                    s=5,
                    color=REFERENCE_COLOR,
                    linewidths=0,
                )
                axes[3].plot(
                    [position + 0.05, position + 0.35],
                    [np.median(null)] * 2,
                    color="0.4",
                    linewidth=1.5,
                )
            axes[3].scatter(
                position - 0.2 + jitter,
                np.maximum(real, FLOOR),
                s=5,
                color="black",
                linewidths=0,
                alpha=0.7,
            )
            axes[3].plot(
                [position - 0.35, position - 0.05],
                [np.median(real)] * 2,
                color="black",
                linewidth=1.5,
            )
        axes[3].set_yscale("log")
        axes[3].set_ylim(bottom=FLOOR)
        axes[3].set_xticks(
            range(len(names)),
            [
                name.replace("withdrawal", "exit")
                .replace(", ", ",\n")
                .replace(" ", "\n", 1)
                for name in names
            ],
            fontsize=7,
        )
        axes[3].set_ylabel(r"$\Delta D^2$ of the missing term (in-sample)")
        axes[3].text(0.02, 0.1, "units", transform=axes[3].transAxes, va="bottom")
        axes[3].text(
            0.02,
            0.03,
            "labels shuffled",
            color="0.4",
            transform=axes[3].transAxes,
            va="bottom",
        )
        if figure is None:
            return None, summary
        label_panels(axes, -0.16)
        return save_figure(figure, output), summary


def save(data: dict, fit_dir: Path, summary: dict | None = None) -> None:
    """Write the curves and the per-unit values of `run_checks` to the fit directory."""
    if summary is None:
        _, summary = plot(data, fit_dir / "residual_checks")
    curves = {key: value.tolist() for key, value in data["curves"].items()}
    (fit_dir / "residual_checks.json").write_text(
        json.dumps(
            {
                "summary": summary,
                "units": data["checks"],
                "curves": curves,
                "hold_s": data["hold_s"],
            },
            indent=1,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=cli.PoissonGLM().root)
    parser.add_argument("--units", type=cli._unit_set, default="all")
    args = parser.parse_args()
    fit_dir = cli.PoissonGLM(root=args.root).fit_dir(args.units)
    data = run_checks(args.root, args.units)
    figure, summary = plot(data, fit_dir / "residual_checks")
    save(data, fit_dir, summary)
    print(json.dumps({**summary, "figure": str(figure)}, indent=2))


if __name__ == "__main__":
    main()
