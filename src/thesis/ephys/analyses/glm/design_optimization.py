"""Combine the design tests of the V1 GLM in one figure: optimizing the design matrix.

Question: why does the design matrix have these regressors, with these
parameters? This script makes no fits and no new analysis. It draws results of
the other design modules in one figure and writes the caption next to it. Each
panel answers one question, and the panels follow one regressor at a time. Each
test is the final design with one thing changed, and D² is relative to the
constant-rate model. An open point at zero marks the value that the model uses.

- Visual flash (row 1). Panel a: the bases of the flash kernel. Panel b:
  number of bases (150 ms window). Panel c: window (8 bases). Panel d:
  adaptation term: none, a synaptic-depression resource with a slow and with a
  fast recovery, and "a flash in the last 50 ms". `flash_kernel_grid.py` and
  `adaptation_grid.py`.
- Task (row 2). Panel e: the bases of the stimulus-during-hold kernel. Panel f:
  no term, a step from the center entry or from the first flash to the center
  exit, that step with a kernel on the first flash, and one kernel from the
  first flash to the center exit with log or linear spacing (`hold_grid.py`).
  Panel g: the bases of the withdrawal kernel. Panel h: no kernel, the 0 to
  100 ms kernel, and wider kernels (`withdrawal_grid.py`).
- Motion energy and spike history (row 3). Motion energy has no bases: each PC
  is one column at the same time bin. Panel i: no motion energy and five
  numbers of PCs (`pc_grid.py`). Panel j: the final design with its motion
  energy shuffled in time within each trial, or taken from another trial
  (`glm unique`). Panel k: the bases of the spike-history kernel, which has no
  test here.

The basis panels (a, e, g, k) are drawn from `dmat.py` with no fit. The test
panels are box plots, over units, of the D² with each choice
minus the D² with the choice in use, with whiskers to 1.5 IQR and no outlier points. D² is on the
validation trials; in panel j it is the mean over the 10 cross-validation
folds. The mark over a box is a two-sided Wilcoxon signed-rank test of those
differences against zero; the p values, and a Friedman test over all choices of
a panel, go to the caption and to `design_optimization_stats.json`. All panels
use `separate_axes`. Each module docstring gives the
comparison, units, baseline, alignment, exclusions, and sampling unit of its
panels.

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

from thesis.ephys.analyses.glm import cli, dmat, hold_grid, withdrawal_grid
from thesis.ephys.analyses.glm.fit import load_unit_records
from thesis.plotting import (
    CONCEPT_COLORS,
    FIGURE_STYLE,
    REFERENCE_COLOR,
    save_figure,
    separate_axes,
)

REMOVED_COLOR = "0.6"
VISUAL, TASK, MOTION = (
    CONCEPT_COLORS["visual"],
    CONCEPT_COLORS["task"],
    CONCEPT_COLORS["motion energy"],
)
# Flash kernel lengths of panel b, from `flash_kernel_grid.py`.
FLASH_WINDOWS_S = (0.1, 0.15, 0.2, 0.25)
# Synaptic-depression models of panel c: (fraction used, recovery in s).
DEPRESSION = {"slow\nresource": (0.5, 0.2), "fast\nresource": (0.5, 0.025)}
CAPTION = """\
**Optimizing the GLM design matrix.** Each panel tests one choice in the final
design, with all other regressors as in the final design. D² is the deviance
explained on trials that were not used for the fit, relative to a model with a
constant firing rate. In the box plots, each unit is scored with every option,
and a box shows, over units, the D² with that option minus the D² with the
reference option of the panel, which is the open point at zero (dashed line).
The reference is always the option that the model uses. Boxes are the quartiles,
whiskers reach 1.5 times the interquartile range, and outliers are not drawn.
The mark over a box is a two-sided Wilcoxon signed-rank test of those
differences against zero (n.s., p ≥ 0.05; \*, p < 0.05; \*\*, p < 0.01;
\*\*\*, p < 0.001); the p value of each option is given below. A Friedman
test compares all options of a panel at once, paired over units. Where two
options do not differ reliably, the model uses the simpler one.
Each row is one group of regressors: the raised cosines that build its kernel
(drawn from the model definition, no fit), then the tests of its choices.
**a–d, Visual flash** ({sample_units} units). **a**, The raised cosines of the
flash kernel. **b**, Number of raised cosines ({p_a}). **c**, Length of the
kernel ({p_b}). **d**, Adaptation: a second flash kernel for a flash that
follows another flash. No such term, a synaptic-depression resource that
recovers slowly (slow resource, 200 ms) or fast (fast resource, 25 ms), and a term that is 1 for a flash that
follows another within 50 ms (recent flash) ({p_c}).
**e–h, Task.** **e**, The raised cosines of the kernel for the stimulus during
fixation: the time from the first flash to the withdrawal from the center port,
where the firing rate is lower while the flashes go on. **f**, The form of that
term. No term; a column that is 1 from the center entry to the withdrawal
(step, entry) or from the first flash to the withdrawal (step, flash); that
last column with a 150 ms kernel on the first flash (step + onset); one kernel
from the first flash to the withdrawal (kernel, the reference); and the same
kernel with linear spacing of its raised cosines ({p_d}). **g**, The raised
cosines of the withdrawal kernel. **h**, No withdrawal kernel, the kernel from
0 to 100 ms after the withdrawal (the reference), and kernels from 100 or
300 ms before to 100 or 300 ms after it ({p_e}).
**i–k, Motion energy and spike history.** Motion energy has no kernel: each PC
is one value at the same time bin. **i**, No motion energy (0) and five numbers
of motion-energy PCs ({sample_units} units; {p_f}). **j**, Does motion energy
predict spikes through the movement of that trial, or only because movement and
firing both change with the time in the trial? The motion energy of the final
design is shuffled in time within each trial, or replaced with the motion
energy of another trial, which has the same average time course ({final_units}
units; {p_g}; shuffled against another trial, p = {p_swap:.2g}). **k**, The
raised cosines of the spike-history kernel (not tested here).
"""


def _load(path: Path, command: str):
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run `{command}` first.")
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            return {key: data[key] for key in data.files}
    return json.loads(path.read_text())


def _stars(p: float) -> str:
    return "n.s." if p >= 0.05 else "*" * (1 + (p < 0.01) + (p < 0.001))


def _plot_choice(
    axis, scores: dict, color, xlabel, selected=None, reference=None, line=True
) -> dict:
    """Box plots of each choice's D² minus the D² of one reference choice, per unit.

    `scores` maps the label of each choice to the D² of each unit (same unit
    order). The reference (default: `selected`) is an open point at zero. Each other choice has a box of its
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
    axis.set_xticks(positions, labels)
    axis.set_xlim(-0.6, len(labels) - 0.4)
    axis.margins(y=0.12)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(r"$\Delta D^2$")
    return stats


def _motion_control_scores(records: list[dict]) -> dict:
    """Return each unit's D² with real, shuffled, and trial-swapped motion energy."""
    complete = np.asarray(
        [item["complete_test_deviance_explained"] for item in records]
    )
    lost = lambda key: np.asarray(  # noqa: E731
        [item["groups"]["video"][key] for item in records]
    )
    return {
        "real": complete,
        "shuffled\nin trial": complete - lost("unique_test_deviance_explained"),
        "another\ntrial": complete - lost("trial_specific_test_deviance_explained"),
    }


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
    kernel = {
        name: per_unit(units)
        for name, units in _load(
            root / "withdrawal_grid" / "withdrawal_grid.json",
            f"{module}.withdrawal_grid",
        ).items()
    }
    hold = {
        # The saved name of the design has the regressor key; the label has its name.
        name.replace("no\nfixation", "no\ninitiation"): per_unit(units)
        for name, units in _load(
            root / "hold_grid" / "hold_grid.json", f"{module}.hold_grid"
        ).items()
    }
    fit_dir = cli.PoissonGLM(root=root).fit_dir(args.units)
    unique_records = load_unit_records(
        fit_dir / "checkpoints" / "unique", "unit_*.json"
    )
    bases, window = dmat.FLASH_BASES, dmat.FLASH_WINDOW_S
    base_counts = sorted({key[0] for key in flash})
    windows = FLASH_WINDOWS_S
    pcs = {
        count: per_unit(units)
        for count, units in _load(
            root / "pc_grid" / "pc_grid.json", f"{module}.pc_grid"
        ).items()
    }

    with plt.rc_context(FIGURE_STYLE):
        figure = plt.figure(figsize=(7.4, 7.2))

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

        # One row for each group of regressors: its bases, then its tests.
        flash_basis, a, b, c = row(0.96, 0.77, (0.8, 0.95, 0.9, 1.7), wspace=0.9)
        hold_basis, d, exit_basis, e = row(
            0.635, 0.445, (0.65, 2.8, 0.5, 1.3), wspace=0.6
        )
        f, g, history_basis = row(0.295, 0.105, (1.6, 1, 1.0), wspace=0.5)
        task_bases = dmat.task_temporal_bases()
        history = dmat.spike_history_basis()
        for axis, basis, color, xlabel, offset_ms in (
            (
                flash_basis,
                task_bases["visual_flash"],
                VISUAL,
                "time from flash (ms)",
                0,
            ),
            (hold_basis, task_bases[dmat.HOLD], TASK, "time from\nfirst flash (ms)", 0),
            (
                exit_basis,
                task_bases["withdrawal"],
                TASK,
                "time from\nwithdrawal (ms)",
                0,
            ),
            # History columns use spike times shifted by one bin: lags 1 to 100 ms.
            (
                history_basis,
                history,
                CONCEPT_COLORS["history"],
                "time since spike (ms)",
                1000 * history.binwidth_s,
            ),
        ):
            axis.plot(
                1000 * basis.basis_time + offset_ms, basis.basis, color=color, lw=0.8
            )
            axis.set_xlabel(xlabel)
            axis.set_ylabel("basis value")

        stats = {}
        stats["flash bases"] = _plot_choice(
            a,
            {count: flash[(count, window, True)] for count in base_counts},
            VISUAL,
            "bases in the flash kernel",
            selected=bases,
        )
        stats["flash window"] = _plot_choice(
            b,
            {f"{1000 * value:.0f}": flash[(bases, value, True)] for value in windows},
            VISUAL,
            "flash kernel length (ms)",
            selected=f"{1000 * window:.0f}",
        )
        second = {
            "none": flash[(bases, window, False)],
            "recent\nflash": adaptation[(None, None)],
        }
        second |= {name: adaptation[pair] for name, pair in DEPRESSION.items()}
        stats["adaptation term"] = _plot_choice(
            c, second, VISUAL, "adaptation term", selected="recent\nflash", line=False
        )
        c.tick_params(axis="x", labelsize=5.5)
        b.tick_params(axis="x", labelsize=6.5)
        stats["stimulus during hold"] = _plot_choice(
            d,
            hold,
            TASK,
            "term for the fixation",
            selected=hold_grid.IN_USE,
            line=False,
        )
        d.tick_params(axis="x", labelsize=5)
        stats["withdrawal"] = _plot_choice(
            e,
            kernel,
            TASK,
            "withdrawal\nkernel (ms)",
            selected=withdrawal_grid.IN_USE,
            line=False,
        )
        e.tick_params(axis="x", labelsize=5.5)
        stats["motion-energy PCs"] = _plot_choice(
            f,
            pcs,
            MOTION,
            "motion-energy PCs",
            selected=str(max(dmat.VIDEO_COMPONENT_COUNTS)),
        )
        f.tick_params(axis="x", labelsize=7)
        control = _motion_control_scores(unique_records)
        stats["motion energy control"] = _plot_choice(
            g, control, MOTION, "motion energy", selected="real", line=False
        )
        g.tick_params(axis="x", labelsize=7)
        swap = float(
            wilcoxon(control["shuffled\nin trial"], control["another\ntrial"]).pvalue
        )
        stats["motion energy control"]["shuffled against another trial, p"] = swap

        panels = (
            flash_basis,
            a,
            b,
            c,
            hold_basis,
            d,
            exit_basis,
            e,
            f,
            g,
            history_basis,
        )
        for axis in panels:
            separate_axes(axis)
        # Letters at the left edge of each panel, tick labels and axis label
        # included.
        figure.canvas.draw()
        to_figure = figure.transFigure.inverted()
        for letter, axis in zip("abcdefghijk", panels, strict=True):
            left = axis.yaxis.get_tightbbox().transformed(to_figure).x0
            top = axis.get_position().y1 + 0.02
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
        p_a=tests("flash bases"),
        p_b=tests("flash window"),
        p_c=tests("adaptation term"),
        p_d=tests("stimulus during hold"),
        p_e=tests("withdrawal"),
        p_f=tests("motion-energy PCs"),
        p_g=tests("motion energy control"),
        p_swap=swap,
        sample_units=len(adaptation[(None, None)]),
        final_units=len(unique_records),
    )
    output.with_name("design_optimization_caption.md").write_text(caption)
    print(output)


if __name__ == "__main__":
    main()
