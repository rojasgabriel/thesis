"""Combine the design tests of the V1 GLM in one figure: optimizing the design matrix.

Question: why does the design matrix have these regressors, with these
parameters, and not others? This script makes no fits and no new analysis. It
draws results of the other design modules in one figure and writes the caption
next to it. The panels follow one regressor at a time, in the order of panel a.

- Panel a (the starting point): regressors of the full nine-regressor design.
  Median and interquartile range of the unique ΔD² (167 units). Gray
  regressors were removed; the arrow next to a kept regressor names the panels
  that test it. The axis is logarithmic and starts at `FLOOR`; a smaller
  quartile is drawn at the floor.
- Visual flash. Panel b: number of bases and window of the flash kernel, with
  and without a second flash term. Panel c: paired comparisons for the three
  choices. Panel d: what the second flash term is: a synaptic-depression
  resource against "a flash in the last 50 ms". `flash_kernel_grid.py` and
  `adaptation_grid.py` (20 units, validation D²).
- Withdrawal. Panel e: cost of a shorter withdrawal kernel (cross-validated
  D² of each unit minus its value with the ±300 ms kernel). Panel f: what the
  wide kernel held: observed over predicted spikes of the model with the short
  kernel, against time from the first flash; the shaded time is the median
  time in the center port. Panel g: the lower rate in the center port is not
  stillness (`port_vs_stillness.py`, all units, raw rates). Panel h: the same
  residual with an in-port state column in the model. Panels e, f, and h use
  20-unit fits of designs that differ only in these terms (`COMPARISONS`).
- Motion energy. Panel i: number of PCs (validation D² of the full design).
  Panel j: each unit's unique ΔD² of motion energy against its trial-specific
  ΔD² (motion energy taken from another trial) in the final design; bars are
  the SEM over folds.
- Panel k (what remains): `residual_checks.py` of the final design.

Each module docstring gives the comparison, units, baseline, alignment,
exclusions, and sampling unit of its panels. Panel e shows differences inside
`PAIRED_LIMIT`; a unit outside it is drawn as a triangle at the edge, and the
median and counts use all units. The full design is the model at commit
`fcd34fa`. Its fit and the comparison fits are in directories next to the
session directory, with the suffixes in `FULL_SUFFIX` and `COMPARISONS`.

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

from thesis.ephys.analyses.glm import (
    adaptation_grid,
    cli,
    flash_kernel_grid,
    port_vs_stillness,
    residual_checks,
)
from thesis.ephys.analyses.glm.fit import load_unit_records
from thesis.plotting import (
    CONCEPT_COLORS,
    FIGURE_STYLE,
    REFERENCE_COLOR,
    save_figure,
)

FLOOR = 1e-5
PAIRED_LIMIT = 0.012
REMOVED_COLOR = "0.6"
VISUAL, TASK, MOTION = (
    CONCEPT_COLORS["visual"],
    CONCEPT_COLORS["task"],
    CONCEPT_COLORS["motion energy"],
)
FULL_SUFFIX = "_full_design"
# 20-unit fits that differ only in the withdrawal kernel and the in-port column.
COMPARISONS = {
    "wide": "_withdrawal_300ms",
    "±100 ms": "_withdrawal_100ms",
    "0 to 100 ms": "_causal_no_onset",
    "in port": "_port_only",
}
# Regressors of the full design: label, color (gray = removed), note.
FULL_DESIGN = {
    "history": ("spike history", CONCEPT_COLORS["history"], "kept"),
    "visual_flash": ("visual flash", VISUAL, "→ b, c"),
    "withdrawal": ("withdrawal, ±300 ms", TASK, "→ e–h"),
    "video": ("motion energy, 25 PCs", MOTION, "→ i, j"),
    "flash_adaptation": ("flash adaptation", VISUAL, "→ c, d"),
    "flash_pre_post_withdrawal": ("flash: post − pre withdrawal", REMOVED_COLOR, ""),
    "flash_motion_energy": ("flash × motion energy", REMOVED_COLOR, ""),
    "response_entry": ("response entry", REMOVED_COLOR, ""),
    "initiation": ("initiation", REMOVED_COLOR, ""),
}
CAPTION = """\
**Optimizing the GLM design matrix.**
**a**, The starting point: median and interquartile range of the unique ΔD² of
each regressor of the full nine-regressor design ({full_units} units). Gray
regressors explained almost nothing and were removed. The arrows name the
panels that test each kept regressor.
**b–d, Visual flash.**
**b**, Mean validation D² against the number of raised cosines in the flash
kernel, for three kernel windows, with (solid) and without (dashed) a second
flash term ({sample_units} units).
**c**, Paired differences in validation D² for the three choices; one point per
unit, black line at the median. Eight bases are better than six, a longer
window is not better, and the second flash term is needed.
**d**, What the second flash term is. A synaptic-depression resource in its
place: mean validation D² against its recovery time constant, for four
fractions of resource used per flash. It is best when it recovers faster than
the shortest flash interval (40 ms), where it is the same as the term in use
(black line): 1 for a flash that follows another within 50 ms.
**e–h, Withdrawal.**
**e**, A withdrawal kernel shorter than ±300 ms lowers the cross-validated D²
(difference for each unit).
**f**, What the wide kernel held: observed over predicted spikes of the model
with the 0 to 100 ms kernel, against time from the first flash (median and
interquartile range of {sample_units} units). The model predicts too many
spikes while the animal is in the center port (shaded: median time in the
port).
**g**, The lower rate in the center port is not stillness: rate against the
motion energy of the video frame, by the state of the animal (median and
interquartile range of {port_units} units). At a side port, with the same
motion energy and no stimulus, the rate is higher.
**h**, The same residual with a state column for the time in the center port
in the model.
**i, j, Motion energy.**
**i**, Mean validation D² against the number of motion-energy PCs in the full
design; the model uses 10.
**j**, Unique ΔD² of motion energy against its trial-specific ΔD² (motion
energy taken from another trial) in the final design; one point per unit, bars
are the SEM over folds. Motion energy predicts spikes through the movement of
that trial and not through the time in the trial.
**k**, What remains: in-sample ΔD² of terms that are not in the final design
(black), and with the labels shuffled over trials (gray;
{residual_units} units).
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
    axis.set_xlabel(r"unique $\Delta D^2$ in the full design")


def _plot_pc_count(axis, summary: dict) -> None:
    scores = summary["mean_validation_deviance_explained"]
    counts = sorted(scores, key=int)
    values = [scores[count] for count in counts]
    axis.plot(range(len(counts)), values, color=MOTION, marker="o", mfc="white")
    axis.scatter(0, values[0], color="black", s=60, zorder=3)
    axis.annotate("in use", (0, values[0]), xytext=(6, -10), textcoords="offset points")
    axis.set_xticks(range(len(counts)), counts)
    axis.set_xlabel("motion-energy PCs")
    axis.set_ylabel(r"mean validation $D^2$")


def _cross_validated(root: Path) -> dict[int, float]:
    records = load_unit_records(root / "sample_fit_me" / "checkpoints", "*_final.json")
    if not records:
        raise FileNotFoundError(f"No 20-unit fit in {root}; see docs/glm.md.")
    return {
        int(item["unit_id"]): item["cross_validated_deviance_explained"]
        for item in records
    }


def _plot_kernel_cost(axis, scores: dict[str, dict]) -> None:
    units = sorted(scores["wide"])
    rng = np.random.default_rng(0)
    names = ("±100 ms", "0 to 100 ms")
    for position, name in enumerate(names):
        change = np.asarray([scores[name][u] - scores["wide"][u] for u in units])
        inside = np.abs(change) <= PAIRED_LIMIT
        x = position + rng.uniform(-0.15, 0.15, len(change))
        axis.scatter(
            x[inside], change[inside], s=9, linewidths=0, color=TASK, alpha=0.7
        )
        axis.scatter(
            x[~inside],
            np.sign(change[~inside]) * PAIRED_LIMIT,
            s=14,
            marker="^",
            color=TASK,
            alpha=0.7,
        )
        axis.plot(
            [position - 0.28, position + 0.28],
            [np.median(change)] * 2,
            color="black",
            linewidth=1.6,
        )
        axis.text(
            position,
            1.01,
            f"{np.sum(change < 0)}/{len(change)}\nlower",
            transform=axis.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=7,
        )
    axis.axhline(0, color=REFERENCE_COLOR, linestyle="--", linewidth=0.8)
    axis.set_ylim(-1.05 * PAIRED_LIMIT, 1.05 * PAIRED_LIMIT)
    axis.set_xlim(-0.6, len(names) - 0.4)
    axis.set_xticks(range(len(names)), ["±100\nms", "0–100\nms"], fontsize=7)
    axis.set_xlabel("withdrawal kernel")
    axis.set_ylabel(r"$D^2$ minus $D^2$ with ±300 ms")


def _plot_hold_residual(axis, residual: dict, note: str) -> None:
    x = (
        1000
        * (residual_checks.ONSET_EDGES_S[:-1] + residual_checks.ONSET_EDGES_S[1:])
        / 2
    )
    low, median, high = np.nanpercentile(
        np.asarray(residual["curves"]["all"]), [25, 50, 75], axis=0
    )
    entry, leave = (1000 * value for value in residual["hold_s"])
    axis.axvspan(max(entry, x[0]), leave, color=TASK, alpha=0.15, linewidth=0)
    axis.text(
        (max(entry, x[0]) + leave) / 2,
        0.9,
        "in center\nport",
        color=TASK,
        ha="center",
        va="center",
        transform=axis.get_xaxis_transform(),
        fontsize=7,
    )
    axis.fill_between(x, low, high, color="black", alpha=0.18, linewidth=0)
    axis.plot(x, median, color="black", linewidth=1.2)
    axis.axhline(1, color=REFERENCE_COLOR, linestyle="--", linewidth=0.8)
    axis.set_ylim(0.4, 1.8)
    axis.set_xlabel("time from first flash (ms)")
    axis.set_ylabel("observed / predicted spikes")
    axis.text(
        0.5, 1.02, note, transform=axis.transAxes, ha="center", va="bottom", fontsize=7
    )


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
    limit = 1.08 * max(np.max(unique + unique_sem), np.max(specific + specific_sem))
    floor = min(
        0, 1.08 * min(np.min(unique - unique_sem), np.min(specific - specific_sem))
    )
    axis.plot([floor, limit], [floor, limit], color=REFERENCE_COLOR, linewidth=0.8)
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
    axis.set_xlim(floor, limit)
    axis.set_ylim(floor, limit)
    axis.set_xlabel(r"unique $\Delta D^2$" + "\n(shuffled in the trial)")
    axis.set_ylabel(r"trial-specific $\Delta D^2$" + "\n(from another trial)")


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
    flash_rows = _load(
        root / "flash_kernel_grid" / "flash_kernel_grid.json",
        f"{module}.flash_kernel_grid",
    )
    adaptation_rows = _load(
        root / "adaptation_grid" / "adaptation_grid.json", f"{module}.adaptation_grid"
    )
    port = _load(root / "port_vs_stillness.npz", f"{module}.port_vs_stillness")
    scores = {name: _cross_validated(sibling(s)) for name, s in COMPARISONS.items()}
    hold = {
        name: _load(
            sibling(COMPARISONS[name]) / "sample_fit_me" / "residual_checks.json",
            f"{module}.residual_checks --units sample --root <that directory>",
        )
        for name in ("0 to 100 ms", "in port")
    }
    fit_dir = cli.PoissonGLM(root=root).fit_dir(args.units)
    residual = _load(
        fit_dir / "residual_checks.json",
        f"{module}.residual_checks --units {args.units}",
    )
    residual_data = {
        "curves": {key: np.asarray(v) for key, v in residual["curves"].items()},
        "checks": residual["units"],
    }
    unique_records = load_unit_records(
        fit_dir / "checkpoints" / "unique", "unit_*.json"
    )
    units = sorted(scores["wide"])
    gain = np.asarray([scores["in port"][u] - scores["wide"][u] for u in units])

    with plt.rc_context(FIGURE_STYLE):
        figure = plt.figure(figsize=(7.4, 11.0))

        # Fixed positions: the panels carry labels outside their axes.
        def row(top, bottom, widths, wspace, left=0.1):
            grid = figure.add_gridspec(
                1,
                len(widths),
                left=left,
                right=0.98,
                top=top,
                bottom=bottom,
                width_ratios=widths,
                wspace=wspace,
            )
            return [figure.add_subplot(cell) for cell in grid]

        a, b = row(0.955, 0.795, (1, 1.1), 0.35, left=0.29)
        c, d = row(0.715, 0.555, (1, 1.15), 0.38)
        e, f, g, h = row(0.455, 0.305, (1.05, 1, 2.0, 1), 0.72)
        i, j, k = row(0.2, 0.05, (1, 0.85, 1.5), 0.6)
        # Panels of the other modules that this figure does not show.
        spare_figure = plt.figure()
        spare = spare_figure.subplots(1, 5)

        _plot_full_design(a, full_unique["groups"])
        flash_kernel_grid.plot_grid(flash_rows, None, axes=(b, c))
        adaptation_grid.plot_grid(adaptation_rows, None, axes=(d, spare[0]))
        _plot_kernel_cost(e, scores)
        _plot_hold_residual(
            f, hold["0 to 100 ms"], "withdrawal 0 to 100 ms,\nno in-port column"
        )
        port_vs_stillness.plot(port, None, axes=(g, spare[1], spare[2]))
        _plot_hold_residual(
            h,
            hold["in port"],
            f"with the in-port column\n$D^2$ against ±300 ms: {np.median(gain):+.4f}",
        )
        h.set_ylabel("")  # the same axis as panel f
        _plot_pc_count(i, pc_summary)
        _plot_motion_control(j, unique_records)
        residual_checks.plot(
            residual_data, None, axes=(spare[3], spare[4], spare[3], k)
        )
        plt.close(spare_figure)

        # Letters at one distance from each panel; the panels differ in width.
        panels = (a, b, c, d, e, f, g, h, i, j, k)
        for letter, axis in zip("abcdefghijk", panels, strict=True):
            box = axis.get_position()
            offset = 0.2 if axis is a else 0.075
            figure.text(
                box.x0 - offset, box.y1 + 0.028, letter, fontweight="bold", fontsize=10
            )
        # One line above the first panel of each group names its regressor.
        for axis, text, color in (
            (a, "the full design", "black"),
            (b, "visual flash", VISUAL),
            (e, "withdrawal", TASK),
            (i, "motion energy", MOTION),
            (k, "what remains", "black"),
        ):
            box = axis.get_position()
            offset = 0.2 if axis is a else 0.075
            figure.text(
                box.x0 - offset + 0.025, box.y1 + 0.0285, text, color=color, fontsize=9
            )
        output = save_figure(figure, root / "design_optimization")

    caption = CAPTION.format(
        full_units=pc_summary["units"],
        sample_units=len(flash_rows[0]["units"]),
        port_units=len(port["rate"]),
        residual_units=len(residual["units"]),
    )
    output.with_name("design_optimization_caption.md").write_text(caption)
    print(output)


if __name__ == "__main__":
    main()
