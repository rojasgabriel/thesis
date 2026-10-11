"""Combine the design tests of the V1 GLM in one figure: optimizing the design matrix.

Question: why does the design matrix have these regressors, with these
parameters, and not others? This script makes no fits and no new analysis. It
draws results of the other design modules in one figure and writes the caption
next to it. Each panel answers one question, and the panels follow one
regressor at a time. A dotted circle marks the value that the model uses.

- Panel a (the starting point): regressors of the full nine-regressor design.
  Median and interquartile range of the unique ΔD² (167 units). Gray
  regressors were removed; the arrow next to a kept regressor names the panels
  that test it.
- Visual flash. Panel b: number of bases of the flash kernel (150 ms window).
  Panel c: window of the flash kernel (8 bases). `flash_kernel_grid.py`.
- Second flash term. Panel d: no second term, a synaptic-depression resource
  with a slow and with a fast recovery, and "a flash in the last 50 ms".
  `flash_kernel_grid.py` and `adaptation_grid.py`.
- Withdrawal. Panel e: width of the withdrawal kernel, and the short kernel
  with a state column for the time in the center port. Panel f: observed and
  predicted firing rate against time from the first flash, for the short
  kernel without and with the in-port column; the shaded time is the median
  time in the center port. Panel g: the lower rate in the center port is not
  stillness (`port_vs_stillness.py`, all units, raw rates). Panels e and f use
  20-unit fits of designs that differ only in these terms (`COMPARISONS`).
- Motion energy. Panel h: number of PCs (full design, 167 units). Panel i:
  each unit's unique ΔD² of motion energy against its trial-specific ΔD²
  (motion energy taken from another trial) in the final design; bars are the
  SEM over folds.
- Panel j (what is left out): the ΔD² that a term not in the final design
  would add (`residual_checks.py`, in-sample; median and interquartile range
  over units), and the median with the labels shuffled over trials (gray
  line). "Time in trial" has no shuffle.

Panels b to e and h show the mean D² over units for each choice (validation
trials; panel e, 10-fold cross-validation). Panels a and j have a logarithmic
axis that starts at `FLOOR`; a smaller value is drawn at the floor. Each module
docstring gives the comparison, units, baseline, alignment, exclusions, and
sampling unit of its panels. The full design is the model at commit `fcd34fa`.
Its fit and the comparison fits are in directories next to the session
directory, with the suffixes in `FULL_SUFFIX` and `COMPARISONS`.

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

from thesis.ephys.analyses.glm import cli, dmat, port_vs_stillness, residual_checks
from thesis.ephys.analyses.glm.fit import load_unit_records
from thesis.plotting import (
    CONCEPT_COLORS,
    FIGURE_STYLE,
    MODEL_COLOR,
    REFERENCE_COLOR,
    mark_selected,
    save_figure,
)

FLOOR = 1e-5
REMOVED_COLOR = "0.6"
VISUAL, TASK, MOTION = (
    CONCEPT_COLORS["visual"],
    CONCEPT_COLORS["task"],
    CONCEPT_COLORS["motion energy"],
)
FULL_SUFFIX = "_full_design"
# 20-unit fits that differ only in the withdrawal kernel and the in-port column.
COMPARISONS = {
    "±300": "_withdrawal_300ms",
    "±100": "_withdrawal_100ms",
    "0–100": "_causal_no_onset",
    "0–100\n+ port": "_port_only",
}
# Synaptic-depression models of panel d: (fraction used, recovery in s).
DEPRESSION = {"slow\ndepression": (0.5, 0.2), "fast\ndepression": (0.5, 0.025)}
# Regressors of the full design: label, color (gray = removed), note.
FULL_DESIGN = {
    "history": ("spike history", CONCEPT_COLORS["history"], "kept"),
    "visual_flash": ("visual flash", VISUAL, "→ b, c"),
    "withdrawal": ("withdrawal, ±300 ms", TASK, "→ e–g"),
    "video": ("motion energy, 25 PCs", MOTION, "→ h, i"),
    "flash_adaptation": ("flash adaptation", VISUAL, "→ d"),
    "flash_pre_post_withdrawal": ("flash: post − pre withdrawal", REMOVED_COLOR, ""),
    "flash_motion_energy": ("flash × motion energy", REMOVED_COLOR, ""),
    "response_entry": ("response entry", REMOVED_COLOR, ""),
    "initiation": ("initiation", REMOVED_COLOR, ""),
}
# Terms that are not in the final design (panel j), in `residual_checks` names.
LEFT_OUT = {
    "time in trial": "time in the trial",
    "session drift": "drift over the session",
    "side, after withdrawal": "response side, after exit",
    "flash rate": "flash rate",
    "side, before withdrawal": "response side, before exit",
}
CAPTION = """\
**Optimizing the GLM design matrix.** A dotted circle marks the value that the
model uses. D² is the deviance explained on trials that were not used for the
fit, as the mean over units.
**a**, The starting point: the unique ΔD² of each regressor of the first,
nine-regressor design (median and interquartile range of {full_units} units).
Gray regressors explained almost nothing and were removed. The arrows name the
panels that test each kept regressor.
**b, c, Visual flash kernel** ({sample_units} units). **b**, D² against the
number of raised cosines; eight are better than six, and more give nothing.
**c**, D² against the length of the kernel; a longer kernel is not reliably
better.
**d, Second flash term.** D² with no second term, with a synaptic-depression
resource that recovers slowly (200 ms) or fast (25 ms), and with a term that is
1 for a flash that follows another within 50 ms (recent flash). A second term is needed. The
depression model is best when it recovers faster than the shortest interval
between flashes (40 ms), where it is the same as the simple term.
**e–g, Withdrawal.** **e**, D² against the width of the withdrawal kernel. A
kernel shorter than ±300 ms predicts worse, unless the model has a column for
the time in the center port. **f**, Why: firing rate against time from the
first flash (median of {sample_units} units; rates are relative to the mean rate of each unit). With the 0 to 100 ms
kernel, the model (dashed) predicts too many spikes
while the animal is in the center port (shaded: median time in the port). With
the in-port column (solid), it follows the observed rate (black). **g**, The
lower rate in the center port is not stillness: firing rate against the motion
energy of the video frame, by the state of the animal (median and
interquartile range of {port_units} units). At a side port, with the same
motion energy and no stimulus, the rate is higher.
**h, i, Motion energy.** **h**, D² against the number of motion-energy PCs in
the first design ({full_units} units). **i**, The D² that each unit loses when
motion energy is shuffled in time within the trial, against the loss when it is
taken from another trial (final design; bars are the SEM over folds). The two
are equal, so motion energy predicts spikes through the movement of that trial
and not through the time in the trial.
**j**, What is left out: the ΔD² that each term would add to the final design
(median and interquartile range of {residual_units} units), and the same value
with the labels shuffled over trials (gray line).
"""


def _load(path: Path, command: str):
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run `{command}` first.")
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            return {key: data[key] for key in data.files}
    return json.loads(path.read_text())


def _plot_full_design(axis, groups: dict) -> None:
    order = sorted(FULL_DESIGN, key=lambda name: groups[name]["median"])
    for position, name in enumerate(order):
        _, color, note = FULL_DESIGN[name]
        low, median, high = (
            max(groups[name][key], FLOOR) for key in ("q25", "median", "q75")
        )
        axis.plot([low, high], [position] * 2, color=color, linewidth=1.2)
        axis.scatter(median, position, color=color, s=22, zorder=3)
        axis.text(high * 1.3, position, note or "removed", color=color, va="center")
    axis.set_xscale("log")
    axis.set_xlim(FLOOR, 3)
    axis.set_xticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
    axis.set_yticks(range(len(order)), [FULL_DESIGN[name][0] for name in order])
    for tick, name in zip(axis.get_yticklabels(), order, strict=True):
        tick.set_color(FULL_DESIGN[name][1])
    axis.set_xlabel(r"unique $\Delta D^2$ in the first design")


def _plot_choice(axis, labels, values, selected, color, xlabel, line=True) -> None:
    """Mean D² for each choice; a dotted circle marks the selected one."""
    positions = np.arange(len(labels))
    axis.plot(
        positions,
        values,
        color=color,
        marker="o",
        markerfacecolor="white",
        linestyle="-" if line else "none",
    )
    mark_selected(axis, selected, values[selected])
    axis.set_xticks(positions, labels)
    axis.set_xlim(-0.5, len(labels) - 0.5)
    axis.margins(y=0.18)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(r"mean $D^2$")


def _cross_validated(root: Path) -> np.ndarray:
    records = load_unit_records(root / "sample_fit_me" / "checkpoints", "*_final.json")
    if not records:
        raise FileNotFoundError(f"No 20-unit fit in {root}; see docs/glm.md.")
    return np.asarray([item["cross_validated_deviance_explained"] for item in records])


def _plot_rates(axis, without: dict, with_port: dict) -> None:
    edges = residual_checks.ONSET_EDGES_S
    x = 1000 * (edges[:-1] + edges[1:]) / 2
    median = lambda curve: np.nanmedian(np.asarray(curve), axis=0)  # noqa: E731
    entry, leave = (1000 * value for value in with_port["hold_s"])
    axis.axvspan(max(entry, x[0]), leave, color=TASK, alpha=0.15, linewidth=0)
    axis.text(
        (max(entry, x[0]) + leave) / 2,
        0.93,
        "in center port",
        color=TASK,
        ha="center",
        transform=axis.get_xaxis_transform(),
    )
    observed = median(with_port["curves"]["observed"])
    axis.plot(x, observed, color="black", linewidth=1.4)
    axis.plot(
        x, median(without["curves"]["predicted"]), color=MODEL_COLOR, linestyle="--"
    )
    axis.plot(x, median(with_port["curves"]["predicted"]), color=MODEL_COLOR)
    axis.set_ylim(bottom=0)
    top = axis.get_ylim()[1]
    for height, text, color in (
        (0.36, "observed", "black"),
        (0.26, "model (dashed)", MODEL_COLOR),
        (0.16, "model + in port", MODEL_COLOR),
    ):
        axis.text(
            x[-1], height * top, text, color=color, ha="right", va="top", fontsize=7
        )
    axis.set_xlabel("time from first flash (ms)")
    axis.set_ylabel("relative firing rate")


def _plot_motion_control(axis, records: list[dict]) -> None:
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
    high = 1.08 * max(np.max(unique + unique_sem), np.max(specific + specific_sem))
    low = min(
        0, 1.08 * min(np.min(unique - unique_sem), np.min(specific - specific_sem))
    )
    axis.plot([low, high], [low, high], color=REFERENCE_COLOR, linewidth=0.8)
    axis.errorbar(
        unique,
        specific,
        xerr=unique_sem,
        yerr=specific_sem,
        fmt="o",
        color=MOTION,
        markersize=2.5,
        elinewidth=0.5,
        alpha=0.8,
    )
    axis.set_xlim(low, high)
    axis.set_ylim(low, high)
    axis.set_xlabel(r"$D^2$ lost: motion energy" + "\nshuffled in the trial")
    axis.set_ylabel(r"$D^2$ lost: motion energy" + "\nfrom another trial")


def _plot_left_out(axis, checks: list[dict]) -> None:
    values = lambda key: np.asarray([item[key] for item in checks])  # noqa: E731
    order = sorted(LEFT_OUT, key=lambda name: np.median(values(name)))
    for position, name in enumerate(order):
        low, median, high = np.maximum(np.percentile(values(name), [25, 50, 75]), FLOOR)
        axis.plot([low, high], [position] * 2, color="black", linewidth=1.2)
        axis.scatter(median, position, color="black", s=22, zorder=3)
        if f"{name} shuffle" in checks[0]:
            shuffled = max(np.median(values(f"{name} shuffle")), FLOOR)
            axis.plot(
                [shuffled] * 2,
                [position - 0.3, position + 0.3],
                color=REFERENCE_COLOR,
                linewidth=2,
            )
    axis.text(
        0.98,
        0.04,
        "shuffled labels",
        color=REFERENCE_COLOR,
        ha="right",
        transform=axis.transAxes,
    )
    axis.set_xscale("log")
    axis.set_xlim(FLOOR, 1e-1)
    axis.set_ylim(-0.7, len(order) - 0.4)
    axis.set_yticks(range(len(order)), [LEFT_OUT[name] for name in order])
    axis.set_xlabel(r"$\Delta D^2$ that the term would add")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=cli.PoissonGLM().root)
    parser.add_argument("--units", type=cli._unit_set, default="all")
    args = parser.parse_args()
    root = args.root
    module = "uv run python -m thesis.ephys.analyses.glm"
    sibling = lambda suffix: root.with_name(root.name + suffix)  # noqa: E731
    full = sibling(FULL_SUFFIX) / "all_fit_me"
    pc_summary = _load(full / "validation_summary.json", "glm fit (full design)")
    full_unique = _load(full / "unique_deviance" / "summary.json", "glm unique")
    mean = lambda row: float(np.mean(list(row["units"].values())))  # noqa: E731
    flash = {
        (row["bases"], row["window"], row["second"]): mean(row)
        for row in _load(
            root / "flash_kernel_grid" / "flash_kernel_grid.json",
            f"{module}.flash_kernel_grid",
        )
    }
    adaptation_rows = _load(
        root / "adaptation_grid" / "adaptation_grid.json", f"{module}.adaptation_grid"
    )
    adaptation = {(row["u"], row["tau"]): mean(row) for row in adaptation_rows}
    port = _load(root / "port_vs_stillness.npz", f"{module}.port_vs_stillness")
    hold = {
        name: _load(
            sibling(COMPARISONS[name]) / "sample_fit_me" / "residual_checks.json",
            f"{module}.residual_checks --units sample --root <that directory>",
        )
        for name in ("0–100", "0–100\n+ port")
    }
    fit_dir = cli.PoissonGLM(root=root).fit_dir(args.units)
    residual = _load(
        fit_dir / "residual_checks.json",
        f"{module}.residual_checks --units {args.units}",
    )
    unique_records = load_unit_records(
        fit_dir / "checkpoints" / "unique", "unit_*.json"
    )
    bases, window = dmat.FLASH_BASES, dmat.FLASH_WINDOW_S
    base_counts = sorted({key[0] for key in flash})
    windows = sorted({key[1] for key in flash})
    pc_scores = pc_summary["mean_validation_deviance_explained"]
    pc_counts = sorted(pc_scores, key=int)

    with plt.rc_context(FIGURE_STYLE):
        figure = plt.figure(figsize=(7.4, 9.6))

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

        (a,) = row(0.955, 0.80, (1,), left=0.36, right=0.8)
        b, c, d = row(0.71, 0.58, (1, 0.8, 1.5))
        e, f, g = row(0.49, 0.345, (1, 1.15, 1.7), wspace=0.5)
        h, i = row(0.25, 0.085, (1, 0.9), right=0.5, wspace=0.95)
        (j,) = row(0.25, 0.085, (1,), left=0.76)
        spare_figure = plt.figure()
        spare = spare_figure.subplots(1, 2)

        _plot_full_design(a, full_unique["groups"])
        _plot_choice(
            b,
            base_counts,
            [flash[(count, window, True)] for count in base_counts],
            base_counts.index(bases),
            VISUAL,
            "raised cosines in the flash kernel",
        )
        _plot_choice(
            c,
            [f"{1000 * value:.0f}" for value in windows],
            [flash[(bases, value, True)] for value in windows],
            windows.index(window),
            VISUAL,
            "length of the flash kernel (ms)",
        )
        second = {"none": flash[(bases, window, False)]}
        second |= {name: adaptation[pair] for name, pair in DEPRESSION.items()}
        second["recent\nflash"] = adaptation[(None, None)]
        _plot_choice(
            d,
            list(second),
            list(second.values()),
            3,
            VISUAL,
            "second flash term",
            False,
        )
        d.tick_params(axis="x", labelsize=7)
        kernel = {
            name: _cross_validated(sibling(s)).mean() for name, s in COMPARISONS.items()
        }
        _plot_choice(
            e,
            list(kernel),
            list(kernel.values()),
            3,
            TASK,
            "withdrawal kernel (ms)",
            False,
        )
        e.tick_params(axis="x", labelsize=7)
        _plot_rates(f, hold["0–100"], hold["0–100\n+ port"])
        port_vs_stillness.plot(port, None, axes=(g, *spare))
        g.set_ylabel("relative firing rate")
        g.set_xlabel("motion energy of the video frame")
        plt.close(spare_figure)
        _plot_choice(
            h,
            pc_counts,
            [pc_scores[count] for count in pc_counts],
            pc_counts.index(str(max(dmat.VIDEO_COMPONENT_COUNTS))),
            MOTION,
            "motion-energy PCs",
        )
        h.tick_params(axis="x", labelsize=7)
        _plot_motion_control(i, unique_records)
        _plot_left_out(j, residual["units"])

        # Letters at one distance from each panel; the panels differ in width.
        wide = {a: 0.27, j: 0.26}
        for letter, axis in zip(
            "abcdefghij", (a, b, c, d, e, f, g, h, i, j), strict=True
        ):
            box = axis.get_position()
            figure.text(
                box.x0 - wide.get(axis, 0.075),
                box.y1 + 0.018,
                letter,
                fontweight="bold",
                fontsize=10,
            )
        # One line above the first panel of each group names its regressor.
        for axis, text, color in (
            (a, "the first design", "black"),
            (b, "visual flash kernel", VISUAL),
            (d, "second flash term", VISUAL),
            (e, "withdrawal", TASK),
            (h, "motion energy", MOTION),
            (j, "what is left out", "black"),
        ):
            box = axis.get_position()
            figure.text(
                box.x0 - wide.get(axis, 0.075) + 0.025,
                box.y1 + 0.0185,
                text,
                color=color,
                fontsize=9,
            )
        output = save_figure(figure, root / "design_optimization")

    caption = CAPTION.format(
        full_units=pc_summary["units"],
        sample_units=len(adaptation_rows[0]["units"]),
        port_units=len(port["rate"]),
        residual_units=len(residual["units"]),
    )
    output.with_name("design_optimization_caption.md").write_text(caption)
    print(output)


if __name__ == "__main__":
    main()
