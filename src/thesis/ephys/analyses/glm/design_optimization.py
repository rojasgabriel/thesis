"""Combine the design tests of the V1 GLM in one figure: optimizing the design matrix.

Question: why does the design matrix have these regressors, with these
parameters? This script makes no fits and no new analysis. It draws results of
the other design modules in one figure and writes the caption next to it. Each
panel answers one question, and the panels follow one regressor at a time. Each
test is the final design with one thing changed, and D² is relative to the
constant-rate model. A dotted circle marks the value that the model uses.

- Visual flash kernel. Panel a: number of bases (150 ms window). Panel b:
  window (8 bases). `flash_kernel_grid.py`.
- Adaptation term. Panel c: no term, a synaptic-depression resource with a
  slow and with a fast recovery, and "a flash in the last 50 ms".
  `flash_kernel_grid.py` and `adaptation_grid.py`.
- Withdrawal. Panel d: width of the withdrawal kernel, without the in-port
  column. Panel e: the ±300 ms kernel, the 0 to 100 ms kernel, the in-port
  column alone, and both (`withdrawal_grid.py`). Panel f: observed firing rate
  and the rate that the final design predicts, against time from the first
  flash; the shaded time is the median time in the center port. Panel g: the
  lower rate in the center port is not stillness (`port_vs_stillness.py`, all
  units, raw rates).
- Motion energy. Panel h: no motion energy and five numbers of PCs
  (`pc_grid.py`). Panel i: each unit's unique ΔD² of motion energy against its
  trial-specific ΔD² (motion energy taken from another trial) in the final
  design; bars are the SEM over folds.
- Panel j (the result): unique ΔD² of each regressor of the final design
  (`glm unique`; median and interquartile range over units).

Panels a to e and h are box plots, over units, of the validation D² with each
choice minus the D² with a reference choice (the one in use; panel d: the
±300 ms kernel), with whiskers to 1.5 IQR and no outlier points. The mark over
a box is a two-sided Wilcoxon signed-rank test of those differences against
zero; the p values, and a Friedman test over all choices of a panel, go to the
caption and to `design_optimization_stats.json`. Firing rates are divided by
the mean rate of that unit over all bins of the analysis. Panel f has the
origin at (0, 0) and default spines, and starts at the first flash; the other
panels use `separate_axes`. Panel i has logarithmic axes that start at
`CONTROL_FLOOR`: a unit with a smaller ΔD² on one axis is drawn at the axis,
and the caption gives the count.
Panel j has a logarithmic axis that starts at `FLOOR`. Each module docstring
gives the comparison, units, baseline, alignment, exclusions, and sampling unit
of its panels.

    uv run python -m thesis.ephys.analyses.glm.design_optimization --units all

Run the other modules first; this script stops if a result file is missing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon

from thesis.ephys.analyses.glm import (
    cli,
    dmat,
    port_vs_stillness,
    residual_checks,
    withdrawal_grid,
)
from thesis.ephys.analyses.glm.fit import load_unit_records
from thesis.plotting import (
    CONCEPT_COLORS,
    FIGURE_STYLE,
    MODEL_COLOR,
    REFERENCE_COLOR,
    mark_selected,
    save_figure,
    separate_axes,
)

FLOOR = 1e-5
# Panel i: a unit with a smaller ΔD² on an axis is drawn at this value.
CONTROL_FLOOR = 1e-4
REMOVED_COLOR = "0.6"
VISUAL, TASK, MOTION = (
    CONCEPT_COLORS["visual"],
    CONCEPT_COLORS["task"],
    CONCEPT_COLORS["motion energy"],
)
# Synaptic-depression models of panel d: (fraction used, recovery in s).
DEPRESSION = {"slow\ndepression": (0.5, 0.2), "fast\ndepression": (0.5, 0.025)}
# Regressors of the final design (panel j): label and color.
FINAL_DESIGN = {
    "history": ("spike history", CONCEPT_COLORS["history"]),
    "visual_flash": ("visual flash", VISUAL),
    "flash_adaptation": ("flash adaptation", VISUAL),
    "in_port": ("in center port", TASK),
    "withdrawal": ("withdrawal", TASK),
    "video": ("motion energy", MOTION),
}
CAPTION = """\
**Optimizing the GLM design matrix.** Each panel tests one choice in the final
design, with all other regressors as in the final design. D² is the deviance
explained on trials that were not used for the fit, relative to a model with a
constant firing rate. In the box plots (**a–e**, **h**), each unit is scored with every option,
and a box shows, over units, the D² with that option minus the D² with the
reference option of the panel, which is at zero (dashed line). The reference is
the option that the model uses (dotted circle); in **d** it is the ±300 ms
kernel. Boxes are the quartiles, whiskers reach 1.5 times the interquartile
range, and outliers are not drawn. The mark over a box is a two-sided Wilcoxon
signed-rank test of those differences against zero (n.s., p ≥ 0.05; \*,
p < 0.05; \*\*, p < 0.01; \*\*\*, p < 0.001); the p value of each option is
given below. A Friedman test compares all options of a panel at once, paired
over units. Where two options do not differ reliably, the model uses the
simpler one.
**a, b, Visual flash kernel** ({sample_units} units). **a**, Number of raised
cosines ({p_a}). **b**, Length of the kernel ({p_b}).
**c, Adaptation**: a second flash kernel for a flash that follows another
flash. No such term, a synaptic-depression resource that recovers slowly
(200 ms) or fast (25 ms), and a term that is 1 for a flash that follows another
within 50 ms (recent flash). The depression model is best when it recovers
faster than the shortest interval between flashes (40 ms), where it is the same
as the recent-flash term ({p_c}).
**d–g, Withdrawal.** **d**, Width of the withdrawal kernel. A kernel shorter
than ±300 ms predicts worse ({p_d}). **e**, A column that is 1 while the animal
is in the center port: the 0 to 100 ms kernel with this column (both) is better
than the kernel alone and than the column alone, and the ±300 ms kernel is not
better than both ({p_e}). **f**, What the column captures: observed firing rate
(black) and the rate that the model with the 0 to 100 ms kernel and the port
column predicts (purple), against time from the first flash (median of
{sample_units} units). Each rate is divided by the mean rate of that unit, so 1
is the average rate of the unit. The rate is lower while the animal is in the
center port (shaded: median time in the port). **g**, This lower rate does not
depend on how much the animal moves: firing rate against the motion energy of
the video frame, in five groups of 20 percentiles, in the center port and on
the way to the side port (median and interquartile range of {port_units}
units).
**h, i, Motion energy.** **h**, No motion energy (0) and five numbers of
motion-energy PCs ({sample_units} units; {p_h}). **i**, Does motion energy predict spikes through the movement of that
trial, or only because movement and firing both change with the time in the
trial? The D² that each unit loses when motion energy is shuffled in time
within the trial, against the loss when it is replaced with the motion energy
of another trial, which has the same average time course (final design,
{final_units} units; bars are the SEM over folds; logarithmic axes; {negative_units} units with a
value below 0.0001 are drawn at the axis). The two losses are equal, so the
movement of that trial is what predicts the spikes.
**j**, The design with the selected parameters: the unique ΔD² of each
regressor (median and interquartile range of {final_units} units).
"""


def _load(path: Path, command: str):
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run `{command}` first.")
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            return {key: data[key] for key in data.files}
    return json.loads(path.read_text())


def _plot_final_design(axis, groups: dict) -> None:
    order = sorted(FINAL_DESIGN, key=lambda name: groups[name]["median"])
    for position, name in enumerate(order):
        color = FINAL_DESIGN[name][1]
        low, median, high = (
            max(groups[name][key], FLOOR) for key in ("q25", "median", "q75")
        )
        axis.plot([low, high], [position] * 2, color=color, linewidth=1.2)
        axis.scatter(median, position, color=color, s=22, zorder=3)
    axis.set_xscale("log")
    axis.set_xlim(1e-4, 1e-1)
    axis.set_yticks(range(len(order)), [FINAL_DESIGN[name][0] for name in order])
    for tick, name in zip(axis.get_yticklabels(), order, strict=True):
        tick.set_color(FINAL_DESIGN[name][1])
    axis.set_xlabel(r"unique $\Delta D^2$")


def _stars(p: float) -> str:
    return "n.s." if p >= 0.05 else "*" * (1 + (p < 0.01) + (p < 0.001))


def _plot_choice(
    axis, scores: dict, color, xlabel, selected=None, reference=None, line=True
) -> dict:
    """Box plots of each choice's D² minus the D² of one reference choice, per unit.

    `scores` maps the label of each choice to the D² of each unit (same unit
    order). The reference (default: `selected`) is at zero; a dotted circle
    marks it when it is the selected choice. Each other choice has a box of its
    per-unit differences from the reference (whiskers to 1.5 IQR, no outlier
    points) and the result of a two-sided Wilcoxon signed-rank test of those
    differences against zero. A line connects the medians. Returns the
    statistics, with a Friedman test over all choices.
    """
    labels = list(scores)
    reference = selected if reference is None else reference
    base = np.asarray(scores[reference])
    positions = np.arange(len(labels))
    medians = []
    omnibus = float(friedmanchisquare(*(scores[label] for label in labels)).pvalue)
    stats = {"friedman_p_all_choices": omnibus}
    axis.axhline(0, color=REFERENCE_COLOR, linestyle="--", linewidth=0.8)
    for position, label in enumerate(labels):
        if label == reference:
            medians.append(0.0)
            continue
        change = np.asarray(scores[label]) - base
        medians.append(float(np.median(change)))
        box = axis.boxplot(
            change,
            positions=[position],
            widths=0.5,
            showfliers=False,
            patch_artist=True,
            medianprops={"color": color, "linewidth": 1.4},
            boxprops={"facecolor": color, "alpha": 0.25, "linewidth": 0},
            whiskerprops={"color": color, "linewidth": 0.8},
            capprops={"color": color, "linewidth": 0.8},
        )
        p_value = float(wilcoxon(change).pvalue)
        stats[str(label).replace("\n", " ")] = {
            "reference": str(reference).replace("\n", " "),
            "median_difference_choice_minus_reference": medians[-1],
            "mean_difference_choice_minus_reference": float(np.mean(change)),
            "units_choice_higher": int(np.sum(change > 0)),
            "units": len(change),
            "wilcoxon_p": p_value,
        }
        axis.annotate(
            _stars(p_value),
            (position, box["whiskers"][1].get_ydata()[1]),
            xytext=(0, 2),
            textcoords="offset points",
            ha="center",
            fontsize=7,
        )
    if line:
        axis.plot(positions, medians, color=color, linewidth=1, zorder=1)
    spot = labels.index(reference)
    axis.scatter(spot, 0, s=26, facecolor="white", edgecolor=color, zorder=3)
    if selected is not None:
        mark_selected(axis, spot, 0)
    axis.set_xticks(positions, labels)
    axis.set_xlim(-0.6, len(labels) - 0.4)
    axis.margins(y=0.12)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(r"change in $D^2$")
    return stats


def _plot_rates(axis, residual: dict) -> None:
    """Observed rate and the rate that the final design predicts, from the first flash."""
    edges = residual_checks.ONSET_EDGES_S
    x = 1000 * (edges[:-1] + edges[1:]) / 2
    shown = x >= 0
    median = lambda curve: np.nanmedian(np.asarray(curve), axis=0)[shown]  # noqa: E731
    leave = 1000 * residual["hold_s"][1]
    axis.axvspan(0, leave, color=TASK, alpha=0.15, linewidth=0)
    axis.text(
        leave / 2,
        0.93,
        "in port",
        color=TASK,
        ha="center",
        transform=axis.get_xaxis_transform(),
    )
    axis.plot(x[shown], median(residual["curves"]["observed"]), color="black")
    axis.plot(x[shown], median(residual["curves"]["predicted"]), color=MODEL_COLOR)
    axis.set_xlim(0, 1000)
    axis.set_ylim(bottom=0)
    top = axis.get_ylim()[1]
    axis.text(1000, 0.3 * top, "observed", color="black", ha="right", va="top")
    axis.text(1000, 0.18 * top, "predicted", color=MODEL_COLOR, ha="right", va="top")
    axis.set_xlabel("time from first flash (ms)")
    axis.set_ylabel("rate / mean rate of the unit")


def _plot_motion_control(axis, records: list[dict]) -> int:
    """Scatter the two ΔD² of motion energy on log axes; return the units at the floor."""
    video = [item["groups"]["video"] for item in records]
    unique, unique_sem, specific, specific_sem = (
        np.asarray([item[key] for item in video])
        for key in (
            "unique_test_deviance_explained",
            "unique_test_deviance_explained_sem",
            "trial_specific_test_deviance_explained",
            "trial_specific_test_deviance_explained_sem",
        )
    )
    negative = int(np.sum((unique < CONTROL_FLOOR) | (specific < CONTROL_FLOOR)))
    high = 2 * max(unique.max(), specific.max())
    axis.plot(
        [CONTROL_FLOOR, high],
        [CONTROL_FLOOR, high],
        color=REFERENCE_COLOR,
        linewidth=0.8,
    )
    x, y = np.maximum(unique, CONTROL_FLOOR), np.maximum(specific, CONTROL_FLOOR)
    axis.errorbar(
        x,
        y,
        xerr=(np.minimum(unique_sem, x - CONTROL_FLOOR), unique_sem),
        yerr=(np.minimum(specific_sem, y - CONTROL_FLOOR), specific_sem),
        fmt="o",
        color=MOTION,
        markersize=2.5,
        elinewidth=0.5,
        alpha=0.8,
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(CONTROL_FLOOR, high)
    axis.set_ylim(CONTROL_FLOOR, high)
    axis.set_xlabel(r"$D^2$ lost: motion energy" + "\nshuffled in the trial")
    axis.set_ylabel(r"$D^2$ lost: motion energy" + "\nfrom another trial")
    return negative


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=cli.PoissonGLM().root)
    parser.add_argument("--units", type=cli._unit_set, default="all")
    args = parser.parse_args()
    root = args.root
    module = "uv run python -m thesis.ephys.analyses.glm"
    per_unit = lambda units: np.asarray([units[u] for u in sorted(units)])  # noqa: E731
    flash = {
        (row["bases"], row["window"], row["second"]): per_unit(row["units"])
        for row in _load(
            root / "flash_kernel_grid" / "flash_kernel_grid.json",
            f"{module}.flash_kernel_grid",
        )
    }
    adaptation = {
        (row["u"], row["tau"]): per_unit(row["units"])
        for row in _load(
            root / "adaptation_grid" / "adaptation_grid.json",
            f"{module}.adaptation_grid",
        )
    }
    port = _load(root / "port_vs_stillness.npz", f"{module}.port_vs_stillness")
    kernel = {
        name: per_unit(units)
        for name, units in _load(
            root / "withdrawal_grid" / "withdrawal_grid.json",
            f"{module}.withdrawal_grid",
        ).items()
    }
    fit_dir = cli.PoissonGLM(root=root).fit_dir(args.units)
    residual = _load(
        fit_dir / "residual_checks.json",
        f"{module}.residual_checks --units {args.units}",
    )
    final_unique = _load(
        fit_dir / "unique_deviance" / "summary.json", f"glm unique --units {args.units}"
    )
    unique_records = load_unit_records(
        fit_dir / "checkpoints" / "unique", "unit_*.json"
    )
    bases, window = dmat.FLASH_BASES, dmat.FLASH_WINDOW_S
    base_counts = sorted({key[0] for key in flash})
    windows = sorted({key[1] for key in flash})
    pcs = {
        count: per_unit(units)
        for count, units in _load(
            root / "pc_grid" / "pc_grid.json", f"{module}.pc_grid"
        ).items()
    }

    with plt.rc_context(FIGURE_STYLE):
        figure = plt.figure(figsize=(7.4, 7.6))

        # Fixed positions: the panels carry labels outside their axes.
        def row(top, bottom, widths, left=0.1, right=0.98, wspace=0.62):
            grid = figure.add_gridspec(
                1,
                len(widths),
                left=left,
                right=right,
                top=top,
                bottom=bottom,
                width_ratios=widths,
                wspace=wspace,
            )
            return [figure.add_subplot(cell) for cell in grid]

        a, b, c = row(0.955, 0.775, (1, 0.8, 1.5), wspace=0.6)
        d, e, f, g = row(0.63, 0.45, (1.0, 1.25, 1.1, 1.6), wspace=0.72)
        h, i = row(0.295, 0.095, (1.15, 0.85), right=0.52, wspace=0.95)
        (j,) = row(0.295, 0.095, (1,), left=0.76)
        spare_figure = plt.figure()
        spare = spare_figure.subplots(1, 2)

        stats = {}
        stats["a flash bases"] = _plot_choice(
            a,
            {count: flash[(count, window, True)] for count in base_counts},
            VISUAL,
            "raised cosines in the flash kernel",
            selected=bases,
        )
        stats["b flash window"] = _plot_choice(
            b,
            {f"{1000 * value:.0f}": flash[(bases, value, True)] for value in windows},
            VISUAL,
            "length of the flash kernel (ms)",
            selected=f"{1000 * window:.0f}",
        )
        second = {"none": flash[(bases, window, False)]}
        second |= {name: adaptation[pair] for name, pair in DEPRESSION.items()}
        second["recent\nflash"] = adaptation[(None, None)]
        stats["c adaptation term"] = _plot_choice(
            c, second, VISUAL, "adaptation term", selected="recent\nflash", line=False
        )
        c.tick_params(axis="x", labelsize=6.5)
        stats["d withdrawal width"] = _plot_choice(
            d,
            {name: kernel[name] for name in ("±300", "±100", "0–100")},
            TASK,
            "withdrawal kernel (ms)",
            reference="±300",
        )
        d.tick_params(axis="x", labelsize=6)
        stats["e in-port column"] = _plot_choice(
            e,
            {
                "±300": kernel["±300"],
                "0–100": kernel["0–100"],
                "port\ncolumn": kernel["in port\nonly"],
                "both": kernel[withdrawal_grid.IN_USE],
            },
            TASK,
            "kernel (ms), port column",
            selected="both",
            line=False,
        )
        e.tick_params(axis="x", labelsize=6)
        _plot_rates(f, residual)
        port_vs_stillness.plot(port, None, axes=(g, *spare), states=(0, 1))
        g.set_ylabel("")  # the same axis as panel f
        g.set_xlabel("motion energy (percentile)")
        plt.close(spare_figure)
        stats["h motion-energy PCs"] = _plot_choice(
            h,
            pcs,
            MOTION,
            "motion-energy PCs",
            selected=str(max(dmat.VIDEO_COMPONENT_COUNTS)),
        )
        h.tick_params(axis="x", labelsize=7)
        negative = _plot_motion_control(i, unique_records)
        _plot_final_design(j, final_unique["groups"])

        panels = (a, b, c, d, e, f, g, h, i, j)
        for axis in panels:
            if axis is not f:  # panel f has the origin at (0, 0)
                separate_axes(axis)
        # Letters at the left edge of each panel, tick labels and axis label
        # included.
        figure.canvas.draw()
        to_figure = figure.transFigure.inverted()
        for letter, axis in zip("abcdefghij", panels, strict=True):
            left = axis.yaxis.get_tightbbox().transformed(to_figure).x0
            top = axis.get_position().y1 + 0.022
            figure.text(left, top, letter, fontweight="bold", fontsize=10)
        output = save_figure(figure, root / "design_optimization")

    output.with_name("design_optimization_stats.json").write_text(
        json.dumps(stats, indent=1, ensure_ascii=False)
    )

    def tests(panel: str) -> str:
        """Write the Friedman p and each Wilcoxon p of one panel as text."""
        values = stats[panel]
        pairs = "; ".join(
            f"{name}, p = {item['wilcoxon_p']:.2g}"
            for name, item in values.items()
            if isinstance(item, dict)
        )
        return f"Friedman p = {values['friedman_p_all_choices']:.2g}; Wilcoxon: {pairs}"

    caption = CAPTION.format(
        p_a=tests("a flash bases"),
        p_b=tests("b flash window"),
        p_c=tests("c adaptation term"),
        p_d=tests("d withdrawal width"),
        p_e=tests("e in-port column"),
        p_h=tests("h motion-energy PCs"),
        sample_units=len(adaptation[(None, None)]),
        port_units=len(port["rate"]),
        final_units=len(unique_records),
        negative_units=negative,
    )
    output.with_name("design_optimization_caption.md").write_text(caption)
    print(output)


if __name__ == "__main__":
    main()
