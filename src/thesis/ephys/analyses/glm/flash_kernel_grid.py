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
- Panel b: three paired comparisons with the design in `dmat.py`: one fewer
  step of bases, one longer window, and no second flash term. Each point is the
  difference in validation D² of one unit; the black line is the median. The
  axis shows `PANEL_B_LIMIT` in each direction. A unit outside that range is
  drawn as a triangle at the edge; the median and the count above each group
  use all units.

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
# Panel b shows differences inside this range; see the module docstring.
PANEL_B_LIMIT = 0.004


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


def plot_grid(rows: list[dict], output: Path | None, axes=None) -> Path | None:
    """Plot mean validation D² and three paired comparisons with the design in use.

    With `axes`, draw into them and do not label or save (for a combined figure).
    """
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
        figure = None
        if axes is None:
            figure, axes = plt.subplots(
                1, 2, figsize=(7.4, 3.1), layout="constrained", width_ratios=(1.25, 1)
            )
        ends = {}
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
                    ends[window] = means[-1]
        # Label each window at the end of its solid line; keep the labels apart.
        gap = 0.09 * np.ptp(axes[0].get_ylim())
        height = -np.inf
        for window in sorted(ends, key=ends.get):
            height = max(ends[window], height + gap)
            axes[0].text(
                BASES[-1] + 0.3,
                height,
                f"{1000 * window:.0f} ms",
                color=color,
                alpha=shades[window],
                va="center",
            )
        axes[0].scatter(
            current[0], values[current].mean(), color="black", s=60, zorder=3
        )
        axes[0].annotate(
            "design in use",
            (current[0], values[current].mean()),
            xytext=(8, -17),
            textcoords="offset points",
        )
        axes[0].set_xticks(BASES)
        axes[0].set_xlim(BASES[0] - 0.5, BASES[-1] + 2.2)
        axes[0].set_xlabel("raised cosines in the flash kernel")
        axes[0].set_ylabel(r"mean validation $D^2$")
        axes[0].text(
            0,
            1.02,
            "solid: with the recent-flash term; dashed: without it",
            transform=axes[0].transAxes,
            va="bottom",
            fontsize=7,
        )

        bases, window, second = current
        fewer = BASES[max(BASES.index(bases) - 1, 0)]
        longer = WINDOWS_S[min(WINDOWS_S.index(window) + 1, len(WINDOWS_S) - 1)]
        comparisons = (
            (f"{bases} bases\nminus {fewer}", current, (fewer, window, second)),
            (
                f"{1000 * longer:.0f} ms\nminus {1000 * window:.0f}",
                (bases, longer, second),
                current,
            ),
            (
                "recent-flash term:\nwith minus without",
                (bases, window, True),
                (bases, window, False),
            ),
        )
        for position, (_, first, other) in enumerate(comparisons):
            change = values[first] - values[other]
            inside = np.abs(change) <= PANEL_B_LIMIT
            x = position + rng.uniform(-0.15, 0.15, len(change))
            axes[1].scatter(
                x[inside], change[inside], s=9, linewidths=0, color=color, alpha=0.6
            )
            axes[1].scatter(  # a unit outside the axis range, drawn at its edge
                x[~inside],
                np.sign(change[~inside]) * PANEL_B_LIMIT,
                s=14,
                marker="^",
                color=color,
                alpha=0.6,
            )
            axes[1].plot(
                [position - 0.28, position + 0.28],
                [np.median(change)] * 2,
                color="black",
                linewidth=1.6,
            )
            axes[1].text(
                position,
                1.01,
                f"{np.sum(change > 0)} of {len(change)}\nunits higher",
                transform=axes[1].get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=7,
            )
        axes[1].axhline(0, color=REFERENCE_COLOR, linestyle="--", linewidth=0.8)
        axes[1].set_ylim(-1.05 * PANEL_B_LIMIT, 1.05 * PANEL_B_LIMIT)
        axes[1].set_xlim(-0.6, len(comparisons) - 0.4)
        axes[1].set_xticks(
            range(len(comparisons)), [item[0] for item in comparisons], fontsize=7
        )
        axes[1].set_ylabel(r"difference in validation $D^2$")
        if figure is None:
            return None
        label_panels(axes, -0.2, 1.12)
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
