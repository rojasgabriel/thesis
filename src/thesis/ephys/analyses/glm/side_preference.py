"""Does the fitted turn-direction effect explain the observed side preference?

Compare (right - left) / (right + left) of observed and predicted rates in
the same bins: 100 ms after center exit to response entry, at least 50 ms.
Use the completed trials, first-flash alignment, quality/stability selection,
and exclusions of the main GLM. No sensory-response selection or baseline
subtraction. Positive means a higher rate for right responses.

Predictions use saved whole-trial test folds, with history scaled on each
fold's fitting rows. Compare the full prediction with its turn-direction
contribution removed in log rate, without refitting. Other terms, including
observed spike history, stay fixed. This tests the contribution of the fitted
term, not unique information or a causal effect of turning. The turn term was
added because the raw rates showed this preference, so the agreement is a
check of the fit, not an independent finding. Shared variance
and observed history can retain side preference when the term is removed.

The unit is the sampling unit; comparisons are paired. Pool both mice in
the figure, test within each mouse, and bootstrap mice then units for the
pooled median. Error bars are 95% trial-bootstrap intervals, stratified by
side, with the fitted predictions fixed; they do not include fit uncertainty.
Units with no observed spikes in a test fold have no saved model for it;
use only their scored folds in all three sources. Trial-bootstrap samples
with no observed spikes give an undefined index and are omitted from its
interval. Units share trials, and there are only two mice.

With --raw-video, use the saved all-unit models with 10 motion-energy and
10 raw-video PCs. Keep the same trials and window, and pair the resulting
turn contribution with the saved main-model result by unit ID. This compares
the fitted allocation of side preference, not a unique or causal turn effect.

    THESIS_CACHE_DIR=figures/cache uv run python -m \
        thesis.ephys.analyses.glm.side_preference
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr, wilcoxon

from thesis.ephys.analyses.glm import cli, dmat
from thesis.ephys.analyses.glm.fit import (
    analysis_units,
    build_unit_counts,
    cross_validation_partitions,
    load_unit_records,
)
from thesis.ephys.analyses.glm.flash_kernels import SESSIONS
from thesis.ephys.analyses.glm.results import raw_contribution
from thesis.plotting import (
    FIGURE_STYLE,
    MODEL_COLOR,
    REFERENCE_COLOR,
    figure_path,
    hierarchical_interval,
    pooled_points,
)

AFTER_EXIT_S = 0.1
MIN_WINDOW_S = 0.05
BOOTSTRAPS = 500
SOURCES = ("observed", "full model", "without turn direction")


def preference(totals, seconds, right):
    """Return the side index for each source, using total counts / exposure.

    Unequal exposure: four spikes in one second right, six in three seconds
    left gives rates 4 and 2, hence index 1/3, rather than a count contrast.

    >>> totals = np.array([[4., 6.], [8., 12.]])
    >>> seconds = np.array([1., 3.])
    >>> right = np.array([True, False])
    >>> np.allclose(preference(totals, seconds, right), [1/3, 1/3])
    True
    >>> np.allclose(preference(totals, seconds, ~right), [-1/3, -1/3])
    True
    >>> np.isnan(preference(np.zeros((1, 2)), seconds, right)).all()
    np.True_
    """
    rates = np.stack(
        [totals[..., side].sum(-1) / seconds[side].sum() for side in (right, ~right)]
    )
    return np.divide(
        rates[0] - rates[1],
        rates.sum(0),
        out=np.full_like(rates[0], np.nan),
        where=rates.sum(0) > 0,
    )


def session_preference(root: Path, raw_video: bool = False) -> dict:
    model = cli.PoissonGLM(root=root)
    prepared = dmat._load_windows(model.windows)
    trials = dmat._load_selected_trials(prepared)
    design_path = model.design
    checkpoints = model.fit_dir("all") / "checkpoints"
    if raw_video:
        from thesis.ephys.analyses.glm.raw_video_test import prepare

        design_path = prepare(root)
        checkpoints = root / "raw_video_test/all_fit_raw/checkpoints"
    metadata = json.loads(design_path.with_suffix(".json").read_text())
    common = np.load(design_path, mmap_mode="r", allow_pickle=False)
    turn = [
        i
        for i, name in enumerate(metadata["column_names"])
        if name.startswith(f"{dmat.TURN_DIRECTION}_basis")
    ]
    if len(turn) != dmat.TURN_DIRECTION_BASES:
        raise ValueError("The saved design does not have the current turn basis.")
    n_trials, n_bins = len(trials), len(prepared["relative_centers"])
    valid = dmat._valid_bin_mask(prepared)
    times = prepared["alignments"][:, None] + prepared["relative_centers"]
    window = valid.reshape(n_trials, n_bins) & (
        times >= trials["center_exit_s"].to_numpy()[:, None] + AFTER_EXIT_S
    )
    seconds = window.sum(1) * dmat.BINWIDTH_S
    right = trials["response"].to_numpy() == 1
    partitions = cross_validation_partitions(n_trials, n_bins, valid)
    records = {
        int(item["unit_id"]): item
        for item in load_unit_records(checkpoints, "unit_*_folds.json")
    }
    units = analysis_units(prepared, "all")
    if set(records) != {unit["unit_id"] for unit in units}:
        raise ValueError("Saved folds and eligible units differ.")
    results = []
    for unit in units:
        counts = build_unit_counts(prepared["alignments"], unit["spike_times"])
        raw_history = dmat.build_unit_history(
            prepared["alignments"], unit["spike_times"]
        )
        record = records[unit["unit_id"]]
        width = int(metadata["base_columns"]) + int(record["components"])
        predicted = np.full((2, n_trials * n_bins), np.nan)
        folds = iter(record["folds"])
        for part in partitions:
            if counts[part["test"]].sum() <= 0:
                continue
            fold = next(folds, None)
            if fold is None:
                raise ValueError("Too few saved folds.")
            weights = np.asarray(fold["coefficients"])
            if len(weights) != width + dmat.HISTORY_COLUMNS:
                raise ValueError("Saved coefficients and design differ.")
            history, _, _ = dmat.training_zscore(raw_history, part["fit"])
            rows = part["test"] & window.ravel()
            design = np.asarray(common[rows, :width])
            log_rate = (
                fold["intercept"]
                + design @ weights[:width]
                + history[rows] @ weights[width:]
            )
            predicted[0, rows] = np.exp(log_rate)
            # Zero the raw regressor, not its z-score; retain the centering offset.
            contribution = raw_contribution(design, weights, metadata, turn)
            predicted[1, rows] = np.exp(log_rate - contribution)
        if next(folds, None) is not None:
            raise ValueError("Too many saved folds.")
        predicted = predicted.reshape(2, n_trials, n_bins)
        scored = np.all(np.isfinite(predicted[0]) | ~window, axis=1)
        used = scored & (seconds >= MIN_WINDOW_S)
        if min(np.sum(used & right), np.sum(used & ~right)) < 2:
            raise ValueError("Fewer than two scored trials on one side.")
        totals = np.vstack(
            (
                np.where(window, counts.reshape(n_trials, n_bins), 0).sum(1),
                np.where(window[None], predicted, 0).sum(2),
            )
        )[:, used]
        duration, sides = seconds[used], right[used]
        if not np.isfinite(totals).all():
            raise ValueError("Nonfinite prediction in a scored window.")
        value = preference(totals, duration, sides)
        # Same resamples across units with the same trials.
        rng = np.random.default_rng(0)
        draws = []
        for _ in range(BOOTSTRAPS):
            sample = np.concatenate(
                [
                    rng.choice(np.flatnonzero(side), side.sum(), replace=True)
                    for side in (sides, ~sides)
                ]
            )
            draws.append(preference(totals[:, sample], duration[sample], sides[sample]))
        interval = np.nanpercentile(draws, [2.5, 97.5], axis=0).T
        results.append(
            {
                "unit_id": unit["unit_id"],
                "index": value.tolist(),
                "interval_95": interval.tolist(),
                "trials": int(used.sum()),
                "right_trials": int(sides.sum()),
                "undefined_bootstraps": np.isnan(draws).sum(0).tolist(),
            }
        )
        if len(results) % 25 == 0:
            print(f"{root.name}: {len(results)}/{len(units)} units", flush=True)
    values = np.asarray([item["index"] for item in results])
    error = np.abs(values[:, 1:] - values[:, :1])
    improvement = error[:, 1] - error[:, 0]
    stats = {}
    for column, name in enumerate(SOURCES[1:], 1):
        correlation = spearmanr(values[:, 0], values[:, column])
        stats[name] = {
            "spearman_rho": float(correlation.statistic),
            "spearman_p": float(correlation.pvalue),
            "median_absolute_error": float(np.median(error[:, column - 1])),
        }
    stats["error reduction from turn direction"] = {
        "median": float(np.median(improvement)),
        "units_improved": int((improvement > 0).sum()),
        "wilcoxon_p": (
            float(wilcoxon(improvement).pvalue) if np.any(improvement) else 1.0
        ),
    }
    return {
        "session": root.name,
        "sources": SOURCES,
        "units": results,
        "statistics": stats,
    }


def plot_sessions(sessions: list[dict], output: Path, raw_video: bool = False) -> dict:
    """Draw the saved unit measurements and return the pooled error reduction."""
    arrays = [
        np.asarray([unit["index"] for unit in item["units"]]) for item in sessions
    ]
    improvement = [
        np.abs(a[:, 2] - a[:, 0]) - np.abs(a[:, 1] - a[:, 0]) for a in arrays
    ]
    pooled = hierarchical_interval(improvement)
    with plt.rc_context(FIGURE_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(7, 3.4), layout="constrained")
        for column, ax in enumerate(axes, 1):
            for item, a in zip(sessions, arrays, strict=True):
                interval = np.asarray([u["interval_95"] for u in item["units"]])
                # Draw interval endpoints directly: percentile intervals can exclude the estimate.
                ax.hlines(
                    a[:, column],
                    interval[:, 0, 0],
                    interval[:, 0, 1],
                    color=MODEL_COLOR,
                    alpha=0.12,
                    lw=0.5,
                )
                ax.vlines(
                    a[:, 0],
                    interval[:, column, 0],
                    interval[:, column, 1],
                    color=MODEL_COLOR,
                    alpha=0.12,
                    lw=0.5,
                )
            pooled_points(
                ax,
                [a[:, 0] for a in arrays],
                [a[:, column] for a in arrays],
                [item["session"].split("_")[0] for item in sessions],
                MODEL_COLOR,
            )
            # Put the mouse labels in empty space, with leaders to their medians.
            labels = {item["session"].split("_")[0] for item in sessions}
            for label in ax.texts:
                if label.get_text() in labels:
                    label.set_anncoords("axes fraction")
                    label.set_position(
                        (0.70, 0.18 if label.get_text() == "GRB006" else 0.10)
                    )
            ax.plot([-1, 1], [-1, 1], color=REFERENCE_COLOR, lw=0.8, ls="--")
            ax.axhline(0, color=REFERENCE_COLOR, lw=0.5)
            ax.axvline(0, color=REFERENCE_COLOR, lw=0.5)
            ax.set(
                xlim=(-1, 1),
                ylim=(-1, 1),
                xticks=(-1, -0.5, 0, 0.5, 1),
                yticks=(-1, -0.5, 0, 0.5, 1),
                aspect="equal",
                xlabel="observed side index",
                ylabel="predicted side index",
            )
            ax.text(
                0.03,
                0.90,
                "raw-video model" if raw_video and column == 1 else SOURCES[column],
                color=MODEL_COLOR,
                transform=ax.transAxes,
                va="top",
            )
            ax.text(
                -0.16, 1.06, "ab"[column - 1], transform=ax.transAxes, weight="bold"
            )
        fig.savefig(figure_path(output / "side_preference.svg"), bbox_inches="tight")
        plt.close(fig)
    return pooled


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("results/glm"))
    parser.add_argument("--raw-video", action="store_true")
    args = parser.parse_args()
    sessions = [
        session_preference(args.out / name, args.raw_video) for name in SESSIONS
    ]
    output = args.out / "raw_video_full" if args.raw_video else args.out
    output.mkdir(exist_ok=True)
    pooled = plot_sessions(sessions, output, args.raw_video)
    report = {
        "model": (
            "10 motion-energy + 10 raw-video PCs"
            if args.raw_video
            else "10 motion-energy PCs"
        ),
        "window": "100 ms after exit to response entry; at least 50 ms",
        "prediction": "out of fold; observed spike history; no refit",
        "intervals": "95% side-stratified trial bootstrap; fixed models",
        "pooled_error_reduction": pooled,
        "sessions": sessions,
    }
    if args.raw_video:
        baseline = json.loads((args.out / "side_preference.json").read_text())
        changes = []
        comparisons = []
        for current, prior in zip(sessions, baseline["sessions"], strict=True):
            if current["session"] != prior["session"]:
                raise ValueError("Main and raw-video session orders differ.")
            old = {u["unit_id"]: u for u in prior["units"]}
            if set(old) != {u["unit_id"] for u in current["units"]}:
                raise ValueError("Main and raw-video unit sets differ.")
            delta = []
            for unit in current["units"]:
                before = old[unit["unit_id"]]
                for key in ("trials", "right_trials"):
                    if before[key] != unit[key]:
                        raise ValueError("The paired side-preference trials differ.")
                np.testing.assert_allclose(
                    before["index"][0], unit["index"][0], atol=1e-12
                )
                np.testing.assert_allclose(
                    before["interval_95"][0], unit["interval_95"][0], atol=1e-12
                )
                a, b = np.asarray(before["index"]), np.asarray(unit["index"])
                old_gain = abs(a[2] - a[0]) - abs(a[1] - a[0])
                new_gain = abs(b[2] - b[0]) - abs(b[1] - b[0])
                delta.append(new_gain - old_gain)
            delta = np.asarray(delta)
            changes.append(delta)
            comparisons.append(
                {
                    "session": current["session"],
                    "baseline_statistics": prior["statistics"],
                    "median_paired_change_in_turn_error_reduction": float(
                        np.median(delta)
                    ),
                    "paired_wilcoxon_p": (
                        float(wilcoxon(delta).pvalue) if np.any(delta) else 1.0
                    ),
                }
            )
        report["main_model_comparison"] = comparisons
        report["pooled_change_in_turn_error_reduction"] = hierarchical_interval(changes)
    (output / "side_preference.json").write_text(
        json.dumps(report, indent=2, allow_nan=False)
    )
    for item in sessions:
        print(item["session"], json.dumps(item["statistics"], indent=2))
    print("Pooled error reduction:", pooled)
    print(output / "side_preference.svg")


if __name__ == "__main__":
    main()
