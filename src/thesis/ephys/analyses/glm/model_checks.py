"""Draw the checks of the V1 GLM in one supplementary figure.

Question: can the fitted model be trusted? Three checks, with no new fit.

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

Box plots are the quartiles with whiskers to 1.5 IQR and no outlier points;
every unit is in the tests. Units, trials, alignment, baseline, and exclusions
are those of `fit.py`. The sampling unit is the unit in panels d to f and the
trial in panels a to c.

    uv run python -m thesis.ephys.analyses.glm.model_checks

Run `glm fit` and `glm unique` with `--units all` and `--units sample`, and
`residual_checks --units all`, first. The script writes `model_checks.svg` and
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

from thesis.ephys.analyses.glm import cli
from thesis.ephys.analyses.glm.design_optimization import _stars
from thesis.ephys.analyses.glm.figures import draw_prediction
from thesis.ephys.analyses.glm.fit import load_unit_records
from thesis.ephys.analyses.glm.main_figure import REGRESSORS
from thesis.ephys.analyses.glm.unique_dev import UNIQUE
from thesis.plotting import (
    FIGURE_STYLE,
    OBSERVED_COLOR,
    REFERENCE_COLOR,
    save_figure,
    separate_axes,
)

PREDICTION_SEED = 2008
HELD_OUT_COLOR = "0.55"
# Candidate terms of `residual_checks.py`, with their labels.
CHECKS = {
    "session drift": "session\ndrift",
    "time in trial": "time in\ntrial",
    "flash rate": "flash\nrate",
    "side, before withdrawal": "side, before\nwithdrawal",
    "side, after withdrawal": "side, after\nwithdrawal",
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=cli.PoissonGLM().root)
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

    with plt.rc_context(FIGURE_STYLE):
        figure = plt.figure(figsize=(7.4, 7.6))
        trials = figure.add_gridspec(
            3,
            1,
            left=0.11,
            right=0.5,
            top=0.96,
            bottom=0.07,
            height_ratios=(1, 1, 1.1),
            hspace=0.22,
        )
        right = figure.add_gridspec(
            3,
            1,
            left=0.64,
            right=0.97,
            top=0.96,
            bottom=0.09,
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
            fontsize=6,
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
        left_out.set_xticks(range(len(CHECKS)), list(CHECKS.values()), fontsize=6)
        left_out.set_ylabel(r"$\Delta D^2$ of a term" + "\nthat is not in the model")
        left_out.text(
            0.02, 0.95, "real", color=colors[0], transform=left_out.transAxes, va="top"
        )
        left_out.text(
            0.02,
            0.85,
            "shuffled labels",
            color=colors[1],
            transform=left_out.transAxes,
            va="top",
        )

        panels = (*trial_axes, full, parts, left_out)
        for axis in panels[3:]:
            separate_axes(axis)
        figure.canvas.draw()
        to_figure = figure.transFigure.inverted()
        for letter, axis in zip("abcdef", panels, strict=True):
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
