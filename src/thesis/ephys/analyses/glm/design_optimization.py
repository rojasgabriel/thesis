"""Combine the design tests of the V1 GLM in one figure: optimizing the design matrix.

Question: why does the design matrix have these regressors, with these
parameters, and not others? This script makes no fits and no new analysis. It
draws panels of the other design modules, from their saved results, in one
figure, and writes the caption next to it.

- Panel a: motion-energy PC count. Mean validation D² of the full design
  (167 units); `fit.py` of that design.
- Panel b: regressors of the full design. Median and interquartile range of
  the unique ΔD² (167 units); gray regressors were removed. The axis is
  logarithmic and starts at `FLOOR`; a smaller quartile is drawn at the floor.
- Panels c and d: flash kernel size and the recent-flash term;
  `flash_kernel_grid.py` (20 units).
- Panel e: synaptic-depression model against the recent-flash term;
  `adaptation_grid.py` (20 units).
- Panels f and g: hold suppression against stillness; `port_vs_stillness.py`
  (all units, raw rates).
- Panels h and i: residuals of the final design; `residual_checks.py`
  (`--units`).

Each module docstring gives the comparison, units, baseline, alignment,
exclusions, and sampling unit of its panels. The full design is the
nine-regressor model at commit `fcd34fa`; its fit is in the session directory
with the suffix `_full_design`.

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
from thesis.plotting import CONCEPT_COLORS, FIGURE_STYLE, save_figure

FLOOR = 1e-5
REMOVED_COLOR = "0.6"
# Regressors of the full design: label, color (gray = removed), note.
FULL_DESIGN = {
    "history": ("spike history", CONCEPT_COLORS["history"], ""),
    "visual_flash": ("visual flash", CONCEPT_COLORS["visual"], ""),
    "withdrawal": ("withdrawal, ±300 ms", CONCEPT_COLORS["task"], "now 0–100 ms"),
    "video": ("motion energy, 25 PCs", CONCEPT_COLORS["motion energy"], "now 10"),
    "flash_adaptation": (
        "flash adaptation",
        CONCEPT_COLORS["visual"],
        "now recent flash",
    ),
    "flash_pre_post_withdrawal": ("flash: post − pre withdrawal", REMOVED_COLOR, ""),
    "flash_motion_energy": ("flash × motion energy", REMOVED_COLOR, ""),
    "response_entry": ("response entry", REMOVED_COLOR, ""),
    "initiation": ("initiation", REMOVED_COLOR, ""),
}
CAPTION = """\
**Optimizing the GLM design matrix.**
**a**, Mean validation D² against the number of motion-energy PCs in the full
nine-regressor design ({full_units} units); the model uses 10.
**b**, Median and interquartile range of the unique ΔD² of each regressor of the
full design; gray regressors were removed.
**c**, Mean validation D² against the number of raised cosines in the flash
kernel, for three kernel windows, with (solid) and without (dashed) the
recent-flash term ({sample_units} units).
**d**, Paired differences in validation D² for three changes to the design in
use; one point per unit, black line at the median.
**e**, A synaptic-depression model in place of the recent-flash term: mean
validation D² against its recovery time constant, for four fractions of
resource used per flash; black line, the term in use.
**f**, Firing rate relative to the mean rate of the unit against the motion
energy of the video frame, by the state of the animal (median and interquartile
range of {port_units} units).
**g**, Rate in the center port against rate at the side port in the frames with
the least motion energy; one point per unit, bars are Poisson standard errors.
**h**, Observed over predicted spikes of the final design against time from the
first flash, for low-rate and high-rate trials (median and interquartile range
of {residual_units} units).
**i**, In-sample ΔD² of terms that are not in the model (black), and with the
labels shuffled over trials (gray).
"""


def _load(path: Path, command: str):
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run `{command}` first.")
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            return {key: data[key] for key in data.files}
    return json.loads(path.read_text())


def _plot_pc_count(axis, summary: dict) -> None:
    scores = summary["mean_validation_deviance_explained"]
    counts = sorted(scores, key=int)
    values = [scores[count] for count in counts]
    axis.plot(range(len(counts)), values, color="black", marker="o", mfc="white")
    axis.scatter(0, values[0], color="black", s=60, zorder=3)
    axis.annotate("in use", (0, values[0]), xytext=(6, -10), textcoords="offset points")
    axis.set_xticks(range(len(counts)), counts)
    axis.set_xlabel("motion-energy PCs")
    axis.set_ylabel(r"mean validation $D^2$")


def _plot_full_design(axis, groups: dict) -> None:
    order = sorted(FULL_DESIGN, key=lambda name: groups[name]["median"])
    for position, name in enumerate(order):
        label, color, note = FULL_DESIGN[name]
        low, median, high = (
            max(groups[name][key], FLOOR) for key in ("q25", "median", "q75")
        )
        axis.plot([low, high], [position] * 2, color=color, linewidth=1.2)
        axis.scatter(median, position, color=color, s=22, zorder=3)
        if note or color == REMOVED_COLOR:
            axis.text(
                high * 1.25, position, note or "removed", color=color, va="center"
            )
    axis.set_xscale("log")
    axis.set_xlim(FLOOR, 0.6)
    axis.set_yticks(range(len(order)), [FULL_DESIGN[name][0] for name in order])
    for tick, name in zip(axis.get_yticklabels(), order, strict=True):
        tick.set_color(FULL_DESIGN[name][1])
    axis.set_xlabel(r"unique $\Delta D^2$ in the full design")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=cli.PoissonGLM().root)
    parser.add_argument("--units", type=cli._unit_set, default="all")
    args = parser.parse_args()
    root = args.root
    module = "uv run python -m thesis.ephys.analyses.glm"
    full = root.with_name(f"{root.name}_full_design") / "all_fit_me"
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
    fit_dir = cli.PoissonGLM(root=root).fit_dir(args.units)
    residual = _load(
        fit_dir / "residual_checks.json",
        f"{module}.residual_checks --units {args.units}",
    )
    residual_data = {
        "curves": {key: np.asarray(v) for key, v in residual["curves"].items()},
        "checks": residual["units"],
    }

    with plt.rc_context(FIGURE_STYLE):
        figure = plt.figure(figsize=(7.4, 10.6))
        # Fixed positions: the panels carry labels outside their axes.
        grid = figure.add_gridspec(
            4, 12, left=0.1, right=0.98, top=0.965, bottom=0.05, hspace=0.62, wspace=0.5
        )
        cells = (
            grid[0, 0:4],
            grid[0, 7:12],
            grid[1, 0:6],
            grid[1, 7:12],
            grid[2, 0:6],
            grid[2, 7:12],
            grid[3, 0:3],
            grid[3, 4:7],
            grid[3, 8:12],
        )
        a, b, c, d, e, f, g, h, i = (figure.add_subplot(cell) for cell in cells)
        # Panels of the other modules that this figure does not show.
        spare_figure = plt.figure()
        spare = spare_figure.subplots(1, 4)

        _plot_pc_count(a, pc_summary)
        _plot_full_design(b, full_unique["groups"])
        flash_kernel_grid.plot_grid(flash_rows, None, axes=(c, d))
        adaptation_grid.plot_grid(adaptation_rows, None, axes=(e, spare[0]))
        port_vs_stillness.plot(port, None, axes=(f, g, spare[1]))
        residual_checks.plot(residual_data, None, axes=(spare[2], h, spare[3], i))
        plt.close(spare_figure)
        # Letters at one distance from each panel; the panels differ in width.
        for letter, axis in zip("abcdefghi", (a, b, c, d, e, f, g, h, i), strict=True):
            box = axis.get_position()
            figure.text(
                box.x0 - 0.075,
                box.y1 + 0.014,
                letter,
                fontweight="bold",
                fontsize=10,
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
