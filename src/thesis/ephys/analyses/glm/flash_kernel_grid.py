"""Show how the window and the number of bases of the V1 GLM flash kernel were selected.

Question: how long must the flash kernel be, how many raised cosines does it
need, and does the model need the second flash term (the flash kernel scaled by
a per-flash value) when the kernel is long enough to hold the suppression that
follows a flash?

Comparison: for each design on the grid (`BASES` x `WINDOWS_S`, with and
without the second flash term), build the full design and run the
model-selection stage of `glm fit`: fit on the training trials and score D² on
the validation trials. Nothing else in the model changes. The comparison is
paired: each unit is scored with each design.

Units, alignment, baseline, and exclusions are those of `fit.py`. Units are the
fixed 20-unit sample. The sampling unit is the unit.

- Panel a: mean validation D² over units against the number of bases, one line
  for each window; solid with the second flash term, dashed without it. The
  filled point is the design in `dmat.py`.
- Panel b: for the designs with `PANEL_B_BASES` bases, the validation D² of
  each unit minus its value with the design in `dmat.py` (small points) and the
  median over units (large points).

More bases can only fit the training trials better, so the rule is the smallest
kernel after which the validation D² is flat, not the maximum.

    uv run python -m thesis.ephys.analyses.glm.flash_kernel_grid

A design that has saved scores in `flash_kernel_grid/flash_kernel_grid.json` is
not fit again.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from thesis.ephys.analyses.glm import cli, dmat
from thesis.ephys.analyses.glm.adaptation_grid import score_design
from thesis.plotting import (
    CONCEPT_COLORS,
    FIGURE_STYLE,
    REFERENCE_COLOR,
    label_panels,
    save_figure,
)

BASES = (6, 8, 10, 12)
WINDOWS_S = (0.15, 0.25, 0.35)
PANEL_B_BASES = 10


def run_grid(source: Path, output: Path) -> list[dict]:
    """Score every design that has no saved scores; return all designs."""
    path = output / "flash_kernel_grid.json"
    rows = json.loads(path.read_text()) if path.exists() else []
    done = {(row["bases"], row["window"], row["second"]) for row in rows}
    for second in (True, False):
        for window in WINDOWS_S:
            for bases in BASES:
                if (bases, window, second) in done:
                    continue
                settings = {
                    "FLASH_BASES": bases,
                    "FLASH_WINDOW_S": window,
                    "SECOND_FLASH_TERM": second,
                }
                rows.append(
                    {
                        "bases": bases,
                        "window": window,
                        "second": second,
                        "units": score_design(
                            source,
                            output / "work",
                            f"{bases}_{window}_{second}",
                            settings,
                        ),
                    }
                )
                output.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(rows, indent=1))
                print(f"Scored {settings}.", flush=True)
    return rows


def plot_grid(rows: list[dict], output: Path) -> Path:
    """Plot mean validation D² and each unit's change from the design in use."""
    scores = {
        (row["bases"], row["window"], row["second"]): row["units"] for row in rows
    }
    current = (dmat.FLASH_BASES, dmat.FLASH_WINDOW_S, dmat.SECOND_FLASH_TERM)
    unit_ids = sorted(scores[current])
    values = {
        key: np.asarray([units[unit] for unit in unit_ids])
        for key, units in scores.items()
    }
    shades = dict(zip(WINDOWS_S, np.linspace(0.4, 1.0, len(WINDOWS_S)), strict=True))
    color = CONCEPT_COLORS["visual"]
    rng = np.random.default_rng(0)

    with plt.rc_context(FIGURE_STYLE):
        figure, axes = plt.subplots(
            1, 2, figsize=(7.4, 3.1), layout="constrained", width_ratios=(1.25, 1)
        )
        position = 0
        ticks = []
        for window in WINDOWS_S:
            for second in (True, False):
                means = [values[(bases, window, second)].mean() for bases in BASES]
                axes[0].plot(
                    BASES,
                    means,
                    marker="o",
                    markersize=4,
                    markerfacecolor="white",
                    linestyle="-" if second else "--",
                    color=color,
                    alpha=shades[window],
                )
                if second:
                    axes[0].text(
                        BASES[-1] + 0.3,
                        means[-1],
                        f"{1000 * window:.0f} ms",
                        color=color,
                        alpha=shades[window],
                        va="center",
                    )
                key = (PANEL_B_BASES, window, second)
                change = values[key] - values[current]
                axes[1].scatter(
                    position + rng.uniform(-0.12, 0.12, len(change)),
                    change,
                    s=5,
                    linewidths=0,
                    color=color,
                    alpha=0.5 * shades[window],
                )
                axes[1].scatter(
                    position,
                    np.median(change),
                    s=34,
                    color=color,
                    alpha=shades[window],
                    marker="o" if second else "D",
                    edgecolors="white",
                    zorder=3,
                )
                ticks.append(f"{1000 * window:.0f}\n{'with' if second else 'without'}")
                position += 1
        axes[0].scatter(
            current[0], values[current].mean(), color="black", s=60, zorder=3
        )
        axes[0].set_xticks(BASES)
        axes[0].set_xlim(BASES[0] - 0.5, BASES[-1] + 2.2)
        axes[0].set_xlabel("raised cosines in the flash kernel")
        axes[0].set_ylabel(r"mean validation deviance explained ($D^2$)")
        axes[0].text(
            0.03,
            0.52,
            "solid: with the second flash term\ndashed: without it",
            transform=axes[0].transAxes,
            va="center",
        )
        axes[1].axhline(0, color=REFERENCE_COLOR, linestyle="--", linewidth=0.8)
        axes[1].set_xticks(range(position), ticks, fontsize=7)
        axes[1].set_xlabel(f"window (ms), second flash term; {PANEL_B_BASES} bases")
        axes[1].set_ylabel(r"validation $D^2$ minus the design in use")
        label_panels(axes, -0.2)
        return save_figure(figure, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=cli.PoissonGLM().root)
    args = parser.parse_args()
    output = args.root / "flash_kernel_grid"
    rows = run_grid(args.root, output)
    print(plot_grid(rows, output / "flash_kernel_grid"))


if __name__ == "__main__":
    main()
