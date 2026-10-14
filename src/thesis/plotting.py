"""Shared figure style: one rc style, one color for each model concept, panel
letters, and SVG output. Every analysis figure uses these."""

import string
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

# Use with `plt.rc_context(FIGURE_STYLE)`.
FIGURE_STYLE = {
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 8,
    "svg.fonttype": "path",
}
# One color for one concept in every figure.
CONCEPT_COLORS = {
    "visual": "C0",
    "task": "C1",
    "motion energy": "C2",
    "history": "C3",
}
OBSERVED_COLOR = "black"
# Not a concept color and not gray: gray marks the excluded bins in the rasters.
MODEL_COLOR = "C4"
REFERENCE_COLOR = "0.65"
# One marker for each animal, so that colors keep their concept. Both animals go
# in one axis; their units are never pooled.
ANIMAL_MARKERS = {"GRB006": "o", "GRB064": "^"}
BOOTSTRAPS = 2000


def median_interval(values, seed: int = 0) -> tuple[float, float, float]:
    """Return the median and its bootstrap 95% interval over the values."""
    values = np.asarray(values, dtype=float)
    picks = np.random.default_rng(seed).integers(
        0, len(values), (BOOTSTRAPS, len(values))
    )
    low, high = np.percentile(np.median(values[picks], axis=1), [2.5, 97.5])
    return float(np.median(values)), float(low), float(high)


def animal_points(axis, x, y, subject: str, color, label: bool = True) -> None:
    """Draw the units of one animal and its median with 95% intervals on both axes.

    Units are small, light markers of the animal's shape. The median of x and
    the median of y are one large marker, with the bootstrap 95% interval of
    each median as error bars, and the name of the animal next to it.
    """
    marker = ANIMAL_MARKERS[subject]
    axis.scatter(x, y, s=9, marker=marker, color=color, alpha=0.3, lw=0, zorder=2)
    (mx, xl, xh), (my, yl, yh) = median_interval(x), median_interval(y)
    axis.errorbar(
        mx,
        my,
        xerr=[[mx - xl], [xh - mx]],
        yerr=[[my - yl], [yh - my]],
        fmt=marker,
        ms=6,
        color=color,
        mec="white",
        mew=0.6,
        elinewidth=1,
        capsize=2,
        zorder=4,
    )
    if label:
        axis.annotate(
            subject,
            (mx, my),
            xytext=(6, -8),
            textcoords="offset points",
            color=color,
            fontsize=6.5,
        )


def label_panels(axes, x: float = -0.14, y: float = 1.05) -> None:
    """Put bold lowercase panel letters above and left of each axis."""
    for letter, axis in zip(string.ascii_lowercase, axes, strict=False):
        if axis is None:  # keeps the letter of a panel that is labeled elsewhere
            continue
        axis.text(
            x,
            y,
            letter,
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )


def save_figure(figure, output: Path) -> Path:
    """Save as SVG next to any older PDF or PNG of the same name, and close."""
    output.parent.mkdir(parents=True, exist_ok=True)
    svg = output.with_suffix(".svg")
    figure.savefig(svg, bbox_inches="tight")
    for stale in (".pdf", ".png"):
        output.with_suffix(stale).unlink(missing_ok=True)
    plt.close(figure)
    return svg


def separate_axes(ax: Axes) -> None:
    """Code provided by Lukas Oesch, Ph.D.."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for axis, spine, (low, high) in (
        (ax.yaxis, "left", ax.get_ylim()),
        (ax.xaxis, "bottom", ax.get_xlim()),
    ):
        ticks = [
            tick for tick in axis.get_majorticklocs() if low <= tick <= high + 1e-3
        ]
        ax.spines[spine].set_bounds(ticks[0], ticks[-1])
        for location, tick in zip(axis.get_minorticklocs(), axis.get_minor_ticks()):
            if not ticks[0] <= location <= ticks[-1]:
                tick.tick1line.set_visible(False)
                tick.tick2line.set_visible(False)
