"""Draw the checks of the V1 GLM in one supplementary figure.

Question: can the fitted model be trusted? Five checks, with no new fit.

- Panels a to c (single trials): observed spikes of the example unit of each
  mouse (left GRB006, right GRB064) on the held-out test trials, spikes drawn
  from the predicted rate of the same trials, and both rates in 10-ms bins. The prediction of a trial comes from
  the cross-validation fold that did not train on it. The predicted spikes are
  one Poisson draw (`PREDICTION_SEED`).
- Panels d and e (units that took no part in the design choices): every design
  choice was made on the 80 units of the 4 sets of 20 units of the grids
  (`SELECTION_SETS`). Panel d: cross-validated D² (on held-out trials, for
  every unit) of the units used for the design choices and of all other
  units (not used). Panel e: the
  unique ΔD² of each regressor in the two groups. The groups are not paired;
  the mark is a two-sided Mann-Whitney U test. The open box is the other
  session of `SESSIONS` (GRB064), an animal that took no part in the design
  choices; it has no test, because a difference between animals has other
  causes.
- Panel f (terms that are not in the model): `candidate_terms.py` adds the
  candidate terms of `candidate_terms.CANDIDATES` to the design together and
  fits all units; the panel shows the unique ΔD² of each candidate in each
  mouse (filled GRB006, open GRB064) and whether it passes the inclusion rule
  of `inclusion.py` (unique ΔD² above 0, two-sided Wilcoxon signed-rank test,
  and more units that need it than chance gives, binomial test; Holm
  correction over the regressors of that design).

- Panel g (calibration): is the predicted rate correct where it is low and
  where it is high? For each unit, the held-out 1-ms bins are sorted by the
  rate that the model predicts and cut in `CALIBRATION_GROUPS` groups of equal
  size; the panel shows the observed rate over the predicted rate in each
  group (median and interquartile range over units). Each bin is predicted by
  the cross-validation fold that did not train on its trial. The values are
  saved in `calibration.npz` in the fit directory and not computed again.
  The dashed line is the median of the other session.
- Panels h to j (recovery of planted kernels): `synthetic_check.py` simulates
  units with known kernels and runs the real pipeline on them. The panels show
  the planted and the recovered kernel of the unit that has every effect, for
  the flash at a gap of 80 ms or more, the fixation, and the withdrawal. The
  fixation kernel is drawn to 350 ms, the shortest simulated fixation. The
  model makes the fixation kernel too small (see `leak_test.py`).

Box plots are the quartiles with whiskers to 1.5 IQR and no outlier points;
every unit is in the tests. Units, trials, alignment, baseline, and exclusions
are those of `fit.py`. The sampling unit is the unit in panels d to f and the
trial in panels a to c.

    uv run python -m thesis.ephys.analyses.glm.model_checks

Run `glm fit` and `glm unique` with `--units all`,
`candidate_terms` for each session, and `synthetic_check --out
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
from scipy.stats import mannwhitneyu

from thesis.ephys.analyses.glm import cli, dmat, fit
from thesis.ephys.analyses.glm.candidate_terms import CANDIDATES
from thesis.ephys.analyses.glm.candidate_terms import ROOT_SUFFIX as CANDIDATE_SUFFIX
from thesis.ephys.analyses.glm.design_optimization import _stars
from thesis.ephys.analyses.glm.figures import draw_prediction
from thesis.ephys.analyses.glm.fit import load_unit_records
from thesis.ephys.analyses.glm.flash_kernels import SESSIONS
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
# The parameterization of each term was chosen on this many sets of 20 units (`--samples`).
SELECTION_SETS = 4
CALIBRATION_GROUPS = 20
SYNTHETIC_KERNELS = {
    "flash, 80+ ms gap": 150,
    "fixation": 350,
    "withdrawal": 100,
}

CAPTION = """\
**Checks of the GLM.** No new fit is made for this figure. **a–c, Single
trials** of an example unit of each mouse (left: {subject} unit {unit}; right:
{other} unit {other_unit}). **a**, Observed spikes on the held-out test trials; light gray is the time after the response entry, which
is not in the model. **b**, Spikes drawn from
the predicted rate of the same trials (one Poisson draw). Each trial is
predicted by a model that was not fit on it, with the observed spike history.
**c**, Observed and predicted rate of those trials in 10-ms bins. **d–e, Units
that took no part in the design choices.** Every D² here is cross-validated:
each trial is predicted by a fit that did not see it. Every design choice was
made on {selection} units of {subject} (used for design choices). **d**,
Cross-validated D² of those units and of the {held_out} other units of
{subject} (not used): median {d2_sel:.3f} and
{d2_out:.3f} (two-sided Mann-Whitney U test, p = {p_d2:.2f}). The open box is
{other} ({other_units} units, median {d2_other:.3f}), a mouse whose units were all
not used for the design choices; it has no test, because a difference between animals
has other causes. **e**, The unique ΔD² of each regressor in the three groups
(Mann-Whitney U test of the two groups of {subject} for each regressor; the
smallest p is {p_unique:.2f}). **f, Terms that are not in the
model.** The candidate terms are added to the design together and every unit
is fit; the panel shows the unique ΔD² of each (filled: {subject}; open:
{other}). Marks: the term passes the inclusion rule in that mouse (unique ΔD²
above 0, two-sided Wilcoxon signed-rank test, and more units that need it than
chance gives, binomial test; Holm correction). {candidates} **g, Calibration.** For each
unit, the held-out 1-ms bins are sorted by the predicted rate and cut in
{groups} groups of equal size. The line is the observed rate over the
predicted rate in each group (median and interquartile range over units of
{subject}; the dashed line is the median of {other}); 1 is a correct
prediction. The model predicts too high a rate where its prediction
is low (lowest ratio {cal_low:.2f}) and too low a rate where it is high
(highest ratio {cal_high:.2f}). **h–j, Recovery of planted kernels.** Spikes
are simulated with known kernels (planted) and the full pipeline is run on
them (recovered), for the flash at a gap of 80 ms or more, the fixation, and
the withdrawal. The recovered fixation kernel is too small: the penalty of the
model gives a part of the level to other terms. In box plots, boxes are the quartiles, whiskers reach 1.5 times
the interquartile range, and outliers are not drawn; every unit is in the
tests (n.s., p ≥ 0.05; \\*, p < 0.05; \\*\\*, p < 0.01; \\*\\*\\*,
p < 0.001).
"""


def _boxes(axis, groups, positions, colors, fill=True, width=0.32) -> None:
    fills = fill if isinstance(fill, tuple) else (fill,) * len(groups)
    for values, position, color, fill in zip(
        groups, positions, colors, fills, strict=True
    ):
        axis.boxplot(
            values,
            positions=[position],
            widths=width,
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
    # The units of the design choices: the sets of 20 units of the grids.
    metadata = dmat._load_windows(model.windows)["metadata"]
    table = fetch_unit_table(
        metadata["subject_name"],
        metadata["session_name"],
        unit_criteria_id=1,
        stability_param_id=0,
        include_metrics=False,
    )
    selection = {
        int(table.iloc[row]["unit_id"])
        for sample in range(SELECTION_SETS)
        for row in fit.analysis_unit_indices(
            table, "sample" + (str(sample) if sample else "")
        )
    }
    data_path = fit_dir / "figure_data.npz"
    if not (final and selection and len(unique) == len(final)):
        raise FileNotFoundError("Run `glm fit` and `glm unique` for all units.")
    if not data_path.exists():
        raise FileNotFoundError(data_path)
    chosen = np.asarray([int(item["unit_id"]) in selection for item in final])
    deviance = np.asarray(
        [item["cross_validated_deviance_explained"] for item in final]
    )
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
    # The animal that took no part in the design choices: all its units are held
    # out. It is drawn next to the two groups of the first animal, with no test,
    # because a difference between animals has other causes.
    other_root = args.root.parent / next(s for s in SESSIONS if s != args.root.name)
    other = cli.PoissonGLM(root=other_root)
    other_subject = other_root.name.split("_")[0]
    subject = args.root.name.split("_")[0]
    subject_name = subject
    other_final = load_unit_records(
        other.fit_dir("all") / "checkpoints", "unit_*_final.json"
    )
    other_unique = load_unit_records(
        other.fit_dir("all") / "checkpoints" / "unique", "unit_*.json"
    )
    other_deviance = np.asarray(
        [item["cross_validated_deviance_explained"] for item in other_final]
    )
    other_ratio = calibration(other)
    other_dir = other.fit_dir("all")
    other_data = np.load(other_dir / "figure_data.npz", allow_pickle=False)
    other_example = json.loads((other_dir / "figure_data.json").read_text())[
        "example_unit_id"
    ]
    stats["example_unit_id_other"] = int(other_example)
    # The candidate test of each mouse: the unique ΔD² of each unit and the
    # inclusion tests (`candidate_terms.py`).
    candidates = {}
    for root, name in ((args.root, subject), (other_root, other_subject)):
        tested = json.loads(
            (root / f"candidate_terms{CANDIDATE_SUFFIX}.json").read_text()
        )
        records = load_unit_records(
            cli.PoissonGLM(root=root.with_name(root.name + CANDIDATE_SUFFIX)).fit_dir(
                "all"
            )
            / "checkpoints"
            / "unique",
            "unit_*.json",
        )
        candidates[name] = {
            term: (
                np.asarray([item["groups"][term][UNIQUE] for item in records]),
                tested[term],
            )
            for term in CANDIDATES.values()
        }

    with plt.rc_context(FIGURE_STYLE):
        figure = plt.figure(figsize=(7.4, 9.2))
        trials = figure.add_gridspec(
            3,
            2,
            left=0.11,
            right=0.5,
            top=0.95,
            bottom=0.30,
            height_ratios=(1, 1, 1.1),
            hspace=0.22,
            wspace=0.12,
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
        trial_axes = [figure.add_subplot(trials[0, 0])]
        trial_axes += [
            figure.add_subplot(trials[row, 0], sharex=trial_axes[0]) for row in (1, 2)
        ]
        draw_prediction(
            trial_axes, data["relative_times"], observed, predicted, conditional, valid
        )
        other_axes = [figure.add_subplot(trials[0, 1])]
        other_axes += [
            figure.add_subplot(trials[row, 1], sharex=other_axes[0]) for row in (1, 2)
        ]
        other_observed = other_data["prediction_observed"]
        other_valid = other_data["test_valid"]
        other_predicted = np.zeros_like(other_observed, dtype=int)
        other_predicted[other_valid] = np.random.default_rng(
            PREDICTION_SEED + int(other_example)
        ).poisson(other_data["conditional"][other_valid])
        draw_prediction(
            other_axes,
            other_data["relative_times"],
            other_observed,
            other_predicted,
            other_data["conditional"],
            other_valid,
        )
        for axis in other_axes:
            axis.set_ylabel("")
        # The labels of the rates are in the right panel only, clear of the data.
        for text in list(trial_axes[2].texts):
            text.remove()
        for axis, name, unit_id in (
            (trial_axes[0], subject_name, example),
            (other_axes[0], other_subject, other_example),
        ):
            axis.set_title(f"{name}, unit {unit_id}", fontsize=6.5, loc="left")
        for axis in (*trial_axes[:2], *other_axes[:2]):
            axis.tick_params(labelbottom=False)

        colors = (OBSERVED_COLOR, HELD_OUT_COLOR)
        full = figure.add_subplot(right[0])
        _boxes(
            full,
            (deviance[chosen], deviance[~chosen], other_deviance),
            (0, 1, 2),
            (*colors, HELD_OUT_COLOR),
            fill=(True, True, False),
        )
        full.set_xticks(
            (0, 1, 2),
            (
                f"{subject}, used for\ndesign choices ({chosen.sum()})",
                f"{subject},\nnot used ({(~chosen).sum()})",
                f"{other_subject},\nnot used ({len(other_final)})",
            ),
            fontsize=6,
        )
        full.set_xlim(-0.6, 2.6)
        full.set_ylabel(r"cross-validated $D^2$")
        p_value = float(mannwhitneyu(deviance[chosen], deviance[~chosen]).pvalue)
        _mark(full, 0.5, p_value)
        stats["cross-validated D2"] = {
            "median_selection": float(np.median(deviance[chosen])),
            "median_held_out": float(np.median(deviance[~chosen])),
            "mann_whitney_p": p_value,
            f"median_{other_subject}": float(np.median(other_deviance)),
        }

        parts = figure.add_subplot(right[1])
        fitted = next(iter(unique.values()))["groups"]
        names = [name for name in REGRESSORS if name in fitted]
        for position, name in enumerate(names):
            values = np.asarray(
                [unique[int(item["unit_id"])]["groups"][name][UNIQUE] for item in final]
            )
            other_values = np.asarray(
                [item["groups"][name][UNIQUE] for item in other_unique]
            )
            _boxes(
                parts,
                (values[chosen], values[~chosen], other_values),
                (position - 0.26, position, position + 0.26),
                (*colors, HELD_OUT_COLOR),
                fill=(True, True, False),
                width=0.22,
            )
            p_value = float(mannwhitneyu(values[chosen], values[~chosen]).pvalue)
            _mark(parts, position - 0.13, p_value)
            stats[f"unique dD2, {name}"] = {
                "median_selection": float(np.median(values[chosen])),
                "median_held_out": float(np.median(values[~chosen])),
                "mann_whitney_p": p_value,
                f"median_{other_subject}": float(np.median(other_values)),
            }
        parts.axhline(0, color=REFERENCE_COLOR, linestyle="--", linewidth=0.8)
        parts.set_xticks(
            range(len(names)),
            [REGRESSORS[name][0] for name in names],
            fontsize=5.5,
            rotation=40,
            ha="right",
            rotation_mode="anchor",
        )
        for tick, name in zip(parts.get_xticklabels(), names, strict=True):
            tick.set_color(REGRESSORS[name][1])
        parts.set_ylabel(r"unique $\Delta D^2$")
        parts.text(
            0.98,
            0.95,
            "used for design choices",
            color=colors[0],
            transform=parts.transAxes,
            ha="right",
            va="top",
        )
        parts.text(
            0.98,
            0.85,
            "not used",
            color=colors[1],
            transform=parts.transAxes,
            ha="right",
            va="top",
        )
        parts.text(
            0.98,
            0.75,
            f"{other_subject} (open)",
            color=HELD_OUT_COLOR,
            transform=parts.transAxes,
            ha="right",
            va="top",
        )

        left_out = figure.add_subplot(right[2])
        terms = list(CANDIDATES.values())
        for position, term in enumerate(terms):
            (values, tested), (other_values, other_tested) = (
                candidates[name][term] for name in (subject, other_subject)
            )
            _boxes(
                left_out,
                (values, other_values),
                (position - 0.19, position + 0.19),
                (OBSERVED_COLOR, HELD_OUT_COLOR),
                fill=(True, False),
            )
            for shift, entry in ((-0.19, tested), (0.19, other_tested)):
                left_out.text(
                    position + shift,
                    1.0,
                    "pass" if entry["passes_both_tests"] else "fail",
                    transform=left_out.get_xaxis_transform(),
                    ha="center",
                    va="bottom",
                    fontsize=5.5,
                )
            stats[f"candidate term, {term}"] = {
                name: {
                    "median_unique": float(np.median(candidates[name][term][0])),
                    **{
                        key: candidates[name][term][1][key]
                        for key in (
                            "units_that_need_it",
                            "fraction_units_that_need_it",
                            "gain_wilcoxon_p_holm",
                            "units_binomial_p_holm",
                            "passes_both_tests",
                        )
                    },
                }
                for name in (subject, other_subject)
            }
        left_out.axhline(0, color=REFERENCE_COLOR, linestyle="--", linewidth=0.8)
        left_out.set_xticks(
            range(len(terms)),
            [REGRESSORS[term][0].replace(" ", "\n", 1) for term in terms],
            fontsize=5.5,
        )
        left_out.set_xlim(-0.6, len(terms) - 0.4)
        left_out.set_ylabel(
            r"unique $\Delta D^2$ of a term" + "\nthat is not in the model"
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
        other_median = np.nanpercentile(other_ratio, 50, axis=0)
        calibration_axis.plot(centers, other_median, color=MODEL_COLOR, ls="--")
        calibration_axis.text(
            0.97,
            0.95,
            f"{args.root.name.split('_')[0]}\n{other_subject} (dashed)",
            color=MODEL_COLOR,
            transform=calibration_axis.transAxes,
            ha="right",
            va="top",
            fontsize=6,
        )
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
            f"median_{other_subject}": other_median.tolist(),
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
    unique = [v for k, v in stats.items() if k.startswith("unique dD2")]
    passed = {
        term: [
            name
            for name, entry in stats[f"candidate term, {term}"].items()
            if entry["passes_both_tests"]
        ]
        for term in CANDIDATES.values()
    }
    candidate_text = (
        "Pass in both mice: "
        + ", ".join(
            REGRESSORS[term][0] for term, names in passed.items() if len(names) == 2
        )
        + ". Fail in both: "
        + ", ".join(REGRESSORS[term][0] for term, names in passed.items() if not names)
        + ". A term that passes goes into the model only after the collinearity and"
        " parameter-recovery checks of a design with it."
    )
    ratios = stats["calibration, observed / predicted by group"]["median"]
    units = len(final)
    output.with_name("model_checks_caption.md").write_text(
        CAPTION.format(
            unit=stats["example_unit_id"],
            selection=stats["selection_units"],
            held_out=units - stats["selection_units"],
            d2_sel=stats["cross-validated D2"]["median_selection"],
            d2_out=stats["cross-validated D2"]["median_held_out"],
            p_d2=stats["cross-validated D2"]["mann_whitney_p"],
            p_unique=min(item["mann_whitney_p"] for item in unique),
            candidates=candidate_text,
            other_unit=stats["example_unit_id_other"],
            groups=CALIBRATION_GROUPS,
            cal_low=min(ratios),
            cal_high=max(ratios),
            subject=args.root.name.split("_")[0],
            other=other_subject,
            other_units=len(other_final),
            d2_other=stats["cross-validated D2"][f"median_{other_subject}"],
        )
    )
    print(json.dumps(stats, indent=1))
    print(output)


if __name__ == "__main__":
    main()
