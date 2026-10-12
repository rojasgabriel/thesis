"""Draw the checks of the V1 GLM in one supplementary figure.

Question: can the fitted model be trusted? Five checks, with no new fit.

- Panels a to c (single trials): observed spikes of the example unit on the
  held-out test trials, spikes drawn from the predicted rate of the same
  trials, and both rates in 10-ms bins. The prediction of a trial comes from
  the cross-validation fold that did not train on it. The predicted spikes are
  one Poisson draw (`PREDICTION_SEED`).
- Panels d and e (units that took no part in the design choices): every design
  choice was made on the 20 units of `--units sample`. Panel d: cross-validated
  D² of those selection units and of all other (held-out) units. Panel e: the
  unique ΔD² of each regressor in the two groups. The groups are not paired;
  the mark is a two-sided Mann-Whitney U test.
- Panel f (terms that are not in the model): for each candidate term, the D²
  that it adds to the final refit of a unit (in-sample), and the same with
  shuffled labels (`residual_checks.py`). The comparison is paired over units;
  the mark is a two-sided Wilcoxon signed-rank test.

- Panel g (calibration): is the predicted rate correct where it is low and
  where it is high? For each unit, the held-out 1-ms bins are sorted by the
  rate that the model predicts and cut in `CALIBRATION_GROUPS` groups of equal
  size; the panel shows the observed rate over the predicted rate in each
  group (median and interquartile range over units). Each bin is predicted by
  the cross-validation fold that did not train on its trial. The values are
  saved in `calibration.npz` in the fit directory and not computed again.
- Panels h to j (recovery of planted kernels): `synthetic_check.py` simulates
  units with known kernels and runs the real pipeline on them. The panels show
  the planted and the recovered kernel of the unit that has every effect, for
  the visual flash, the stimulus during hold, and the withdrawal. The hold
  kernel is drawn to 350 ms, the shortest simulated hold.

Box plots are the quartiles with whiskers to 1.5 IQR and no outlier points;
every unit is in the tests. Units, trials, alignment, baseline, and exclusions
are those of `fit.py`. The sampling unit is the unit in panels d to f and the
trial in panels a to c.

    uv run python -m thesis.ephys.analyses.glm.model_checks

Run `glm fit` and `glm unique` with `--units all` and `--units sample`,
`residual_checks --units all`, and `synthetic_check --out
figures/glm/synthetic_check`, first. The script writes `model_checks.svg` and
`model_checks_stats.json` in the session directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu, wilcoxon

from thesis.ephys.analyses.glm import cli, dmat, fit
from thesis.ephys.analyses.glm.design_optimization import _stars
from thesis.ephys.analyses.glm.figures import draw_prediction
from thesis.ephys.analyses.glm.fit import load_unit_records
from thesis.ephys.analyses.glm.main_figure import REGRESSORS
from thesis.ephys.analyses.glm.unique_dev import UNIQUE
from thesis.ephys.units import fetch_unit_table
from thesis.plotting import (
    FIGURE_STYLE,
    MODEL_COLOR,
    OBSERVED_COLOR,
    REFERENCE_COLOR,
    save_figure,
    separate_axes,
)

PREDICTION_SEED = 2008
HELD_OUT_COLOR = "0.55"
# Synthetic unit with every effect, and how far each kernel is drawn (ms).
SYNTHETIC_UNIT = 197
CALIBRATION_GROUPS = 20
SYNTHETIC_KERNELS = {
    "visual flash": 150,
    "stimulus during hold": 350,
    "withdrawal": 100,
}
# Candidate terms of `residual_checks.py`, with their labels.
CHECKS = {
    "session drift": "session\ndrift",
    "time in trial": "time in\ntrial",
    "flash rate": "flash\nrate",
    "side, before withdrawal": "side,\nbefore\nwithdrawal",
    "side, after withdrawal": "side,\nafter\nwithdrawal",
}


def _boxes(axis, groups, positions, colors, fill=True) -> None:
    for values, position, color in zip(groups, positions, colors, strict=True):
        axis.boxplot(
            values,
            positions=[position],
            widths=0.32,
            showfliers=False,
            patch_artist=True,
            medianprops={"color": color, "linewidth": 1.4},
            boxprops={
                "facecolor": color if fill else "white",
                "edgecolor": color,
                "alpha": 0.3 if fill else 1,
                "linewidth": 0 if fill else 0.8,
            },
            whiskerprops={"color": color, "linewidth": 0.8},
            capprops={"color": color, "linewidth": 0.8},
        )


def _mark(axis, position: float, p_value: float) -> None:
    axis.text(
        position,
        1.0,
        _stars(p_value),
        transform=axis.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=7,
    )


def calibration(model: cli.PoissonGLM) -> np.ndarray:
    """Return observed / predicted rate by group of predicted rate (units x groups)."""
    cache = model.fit_dir("all") / "calibration.npz"
    if cache.exists():
        with np.load(cache, allow_pickle=False) as data:
            return data["ratio"]
    prepared = dmat._load_windows(model.windows)
    alignments = prepared["alignments"]
    valid = dmat._valid_bin_mask(prepared)
    metadata = json.loads(model.design.with_suffix(".json").read_text())
    common = np.load(model.design, mmap_mode="r", allow_pickle=False)
    partitions = fit.cross_validation_partitions(
        len(alignments), len(valid) // len(alignments), valid
    )
    units = fetch_unit_table(
        prepared["metadata"]["subject_name"],
        prepared["metadata"]["session_name"],
        unit_criteria_id=1,
        stability_param_id=0,
        include_metrics=False,
    ).set_index("unit_id")
    ratios = []
    for record in load_unit_records(
        model.fit_dir("all") / "checkpoints", "unit_*_folds.json"
    ):
        spikes = np.asarray(units.loc[record["unit_id"], "spike_times_s"], dtype=float)
        counts = fit.build_unit_counts(alignments, spikes)
        history = dmat.build_unit_history(alignments, spikes)
        width = int(metadata["base_columns"]) + int(record["components"])
        # A fold with no test spikes was not scored and has no saved model.
        scored = [part for part in partitions if counts[part["test"]].sum() > 0]
        observed, predicted = [], []
        for part, fold in zip(scored, record["folds"], strict=True):
            scaled, _, _ = dmat.training_zscore(history, part["fit"])
            design = dmat._design_with_history(common, scaled, width, part["test"])
            predicted.append(
                np.exp(fold["intercept"] + design @ np.asarray(fold["coefficients"]))
            )
            observed.append(counts[part["test"]])
        observed, predicted = np.concatenate(observed), np.concatenate(predicted)
        groups = np.array_split(np.argsort(predicted), CALIBRATION_GROUPS)
        ratios.append(
            [observed[group].mean() / predicted[group].mean() for group in groups]
        )
        print(f"Calibration of unit {record['unit_id']}.", flush=True)
    ratio = np.asarray(ratios)
    np.savez(cache, ratio=ratio)
    return ratio


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=cli.PoissonGLM().root)
    parser.add_argument(
        "--synthetic",
        type=Path,
        default=Path("figures/glm/synthetic_check/synthetic_recovery.json"),
        help="Kernels that `synthetic_check` wrote.",
    )
    args = parser.parse_args()
    model = cli.PoissonGLM(root=args.root)
    fit_dir = model.fit_dir("all")
    checkpoints = fit_dir / "checkpoints"
    final = load_unit_records(checkpoints, "unit_*_final.json")
    unique = {
        int(item["unit_id"]): item
        for item in load_unit_records(checkpoints / "unique", "unit_*.json")
    }
    selection = {
        int(item["unit_id"])
        for item in load_unit_records(
            model.fit_dir("sample") / "checkpoints", "unit_*_final.json"
        )
    }
    residual_path = fit_dir / "residual_checks.json"
    data_path = fit_dir / "figure_data.npz"
    if not (final and selection and len(unique) == len(final)):
        raise FileNotFoundError("Run `glm fit` and `glm unique` for all and sample.")
    for path in (residual_path, data_path):
        if not path.exists():
            raise FileNotFoundError(path)
    chosen = np.asarray([int(item["unit_id"]) in selection for item in final])
    deviance = np.asarray(
        [item["cross_validated_deviance_explained"] for item in final]
    )
    residual = json.loads(residual_path.read_text())["units"]
    data = np.load(data_path, allow_pickle=False)
    example = json.loads((fit_dir / "figure_data.json").read_text())["example_unit_id"]
    observed, conditional = data["prediction_observed"], data["conditional"]
    valid = data["test_valid"]
    predicted = np.zeros_like(observed, dtype=int)
    predicted[valid] = np.random.default_rng(PREDICTION_SEED + int(example)).poisson(
        conditional[valid]
    )
    stats = {"example_unit_id": int(example), "selection_units": int(chosen.sum())}
    ratio = calibration(model)

    with plt.rc_context(FIGURE_STYLE):
        figure = plt.figure(figsize=(7.4, 9.2))
        trials = figure.add_gridspec(
            3,
            1,
            left=0.11,
            right=0.5,
            top=0.96,
            bottom=0.30,
            height_ratios=(1, 1, 1.1),
            hspace=0.22,
        )
        right = figure.add_gridspec(
            3,
            1,
            left=0.62,
            right=0.97,
            top=0.96,
            bottom=0.32,
            height_ratios=(0.8, 1.25, 1.1),
            hspace=0.62,
        )
        trial_axes = [figure.add_subplot(trials[0])]
        trial_axes += [
            figure.add_subplot(trials[row], sharex=trial_axes[0]) for row in (1, 2)
        ]
        draw_prediction(
            trial_axes, data["relative_times"], observed, predicted, conditional, valid
        )
        for axis in trial_axes[:2]:
            axis.tick_params(labelbottom=False)

        colors = (OBSERVED_COLOR, HELD_OUT_COLOR)
        full = figure.add_subplot(right[0])
        _boxes(full, (deviance[chosen], deviance[~chosen]), (0, 1), colors)
        full.set_xticks(
            (0, 1),
            (
                f"selection\n({chosen.sum()} units)",
                f"held out\n({(~chosen).sum()} units)",
            ),
        )
        full.set_xlim(-0.6, 1.6)
        full.set_ylabel(r"cross-validated $D^2$")
        p_value = float(mannwhitneyu(deviance[chosen], deviance[~chosen]).pvalue)
        _mark(full, 0.5, p_value)
        stats["cross-validated D2"] = {
            "median_selection": float(np.median(deviance[chosen])),
            "median_held_out": float(np.median(deviance[~chosen])),
            "mann_whitney_p": p_value,
        }

        parts = figure.add_subplot(right[1])
        names = list(REGRESSORS)
        for position, name in enumerate(names):
            values = np.asarray(
                [unique[int(item["unit_id"])]["groups"][name][UNIQUE] for item in final]
            )
            _boxes(
                parts,
                (values[chosen], values[~chosen]),
                (position - 0.19, position + 0.19),
                colors,
            )
            p_value = float(mannwhitneyu(values[chosen], values[~chosen]).pvalue)
            _mark(parts, position, p_value)
            stats[f"unique dD2, {name}"] = {
                "median_selection": float(np.median(values[chosen])),
                "median_held_out": float(np.median(values[~chosen])),
                "mann_whitney_p": p_value,
            }
        parts.axhline(0, color=REFERENCE_COLOR, linestyle="--", linewidth=0.8)
        parts.set_xticks(
            range(len(names)),
            [REGRESSORS[name][0].replace(" ", "\n") for name in names],
            fontsize=5.5,
        )
        for tick, name in zip(parts.get_xticklabels(), names, strict=True):
            tick.set_color(REGRESSORS[name][1])
        parts.set_ylabel(r"unique $\Delta D^2$")
        parts.text(
            0.98,
            0.95,
            "selection",
            color=colors[0],
            transform=parts.transAxes,
            ha="right",
            va="top",
        )
        parts.text(
            0.98,
            0.85,
            "held out",
            color=colors[1],
            transform=parts.transAxes,
            ha="right",
            va="top",
        )

        left_out = figure.add_subplot(right[2])
        for position, name in enumerate(CHECKS):
            real = np.asarray([item[name] for item in residual])
            shuffled = np.asarray([item[f"{name} shuffle"] for item in residual])
            _boxes(
                left_out,
                (real, shuffled),
                (position - 0.19, position + 0.19),
                colors,
            )
            p_value = float(wilcoxon(real, shuffled).pvalue)
            _mark(left_out, position, p_value)
            stats[f"term not in the model, {name}"] = {
                "median": float(np.median(real)),
                "median_shuffled": float(np.median(shuffled)),
                "units_above_shuffled": int(np.sum(real > shuffled)),
                "units": len(real),
                "wilcoxon_p": p_value,
            }
        left_out.set_xticks(range(len(CHECKS)), list(CHECKS.values()), fontsize=5.5)
        left_out.set_ylabel(r"$\Delta D^2$ of a term" + "\nthat is not in the model")
        left_out.text(
            0.55, 0.82, "real", color=colors[0], transform=left_out.transAxes, va="top"
        )
        left_out.text(
            0.55,
            0.72,
            "shuffled labels",
            color=colors[1],
            transform=left_out.transAxes,
            va="top",
        )

        # Planted and recovered kernels of the synthetic unit with every effect.
        recovery = json.loads(args.synthetic.read_text())[str(SYNTHETIC_UNIT)]
        bottom = figure.add_gridspec(
            1, 4, left=0.11, right=0.97, top=0.2, bottom=0.06, wspace=0.55
        )
        calibration_axis = figure.add_subplot(bottom[0])
        centers = (np.arange(CALIBRATION_GROUPS) + 0.5) * 100 / CALIBRATION_GROUPS
        low, median, high = np.nanpercentile(ratio, [25, 50, 75], axis=0)
        calibration_axis.fill_between(
            centers, low, high, color=MODEL_COLOR, alpha=0.25, linewidth=0
        )
        calibration_axis.plot(centers, median, color=MODEL_COLOR)
        calibration_axis.axhline(
            1, color=REFERENCE_COLOR, linestyle="--", linewidth=0.8
        )
        calibration_axis.set_ylim(0.5, 1.5)
        calibration_axis.set_xlabel("predicted rate\n(percentile within unit)")
        calibration_axis.set_ylabel("observed /\npredicted rate")
        stats["calibration, observed / predicted by group"] = {
            "group_percentile": centers.tolist(),
            "median": median.tolist(),
            "fraction_units_above_1": np.mean(ratio > 1, axis=0).tolist(),
        }
        planted_axes = []
        for column, (name, limit_ms) in enumerate(SYNTHETIC_KERNELS.items()):
            axis = figure.add_subplot(bottom[column + 1])
            kernel = recovery[name]
            times = np.asarray(kernel["time_ms"])
            keep = times <= limit_ms
            axis.plot(
                times[keep], np.asarray(kernel["planted"])[keep], color=OBSERVED_COLOR
            )
            axis.plot(
                times[keep], np.asarray(kernel["recovered"])[keep], color=MODEL_COLOR
            )
            axis.axhline(0, color=REFERENCE_COLOR, linestyle="--", linewidth=0.8)
            axis.set_xlabel(f"{name}:\ntime from event (ms)")
            if column == 0:
                axis.set_ylabel(r"$\Delta$ log rate")
            if column == 1:  # the curves are low here, so the labels are clear of them
                for row, (label, color) in enumerate(
                    (("planted", OBSERVED_COLOR), ("recovered", MODEL_COLOR))
                ):
                    axis.text(
                        0.98,
                        0.95 - 0.11 * row,
                        label,
                        color=color,
                        transform=axis.transAxes,
                        ha="right",
                        va="top",
                    )
            planted_axes.append(axis)
        panels = (*trial_axes, full, parts, left_out, calibration_axis, *planted_axes)
        for axis in panels[3:]:
            separate_axes(axis)
        figure.canvas.draw()
        to_figure = figure.transFigure.inverted()
        for letter, axis in zip("abcdefghij", panels, strict=True):
            left = axis.yaxis.get_tightbbox().transformed(to_figure).x0
            figure.text(
                left,
                axis.get_position().y1 + 0.012,
                letter,
                fontweight="bold",
                fontsize=10,
            )
        output = save_figure(figure, args.root / "model_checks")

    output.with_name("model_checks_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))
    print(output)


if __name__ == "__main__":
    main()
