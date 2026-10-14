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


# Axis of a peak rate (spikes/s), log on both axes. A peak at or below the lower
# limit (a baseline-corrected peak can be 0 or below) is not drawn; it stays in
# the statistics, and the panel gives the count.
PEAK_AXIS_LIMITS = (0.5, 200)


def peak_axes(axis) -> None:
    """Give a peak-against-peak scatter the same log axes, limits, and ticks, and the identity line."""
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(*PEAK_AXIS_LIMITS)
    axis.set_ylim(*PEAK_AXIS_LIMITS)
    axis.set_xticks([1, 10, 100], ["1", "10", "100"])
    axis.set_yticks([1, 10, 100], ["1", "10", "100"])
    axis.plot(
        PEAK_AXIS_LIMITS, PEAK_AXIS_LIMITS, color=REFERENCE_COLOR, lw=0.8, ls="--"
    )
    axis.set_aspect("equal")


def hierarchical_interval(groups, seed: int = 0) -> dict:
    """Return the pooled median with a hierarchical-bootstrap 95% interval and p.

    `groups` holds the values of each mouse. A resample draws the mice with
    replacement, then the units of each drawn mouse with replacement, and takes
    the median of all of them (Saravanan, Berman and Sober 2020). p is the
    two-sided bootstrap p-value of the median against 0. Also returns the
    median of each mouse.
    """
    groups = [np.asarray(values, dtype=float) for values in groups]
    rng = np.random.default_rng(seed)
    medians = np.empty(BOOTSTRAPS)
    for b in range(BOOTSTRAPS):
        picked = rng.integers(0, len(groups), len(groups))
        medians[b] = np.median(
            np.concatenate(
                [
                    groups[g][rng.integers(0, len(groups[g]), len(groups[g]))]
                    for g in picked
                ]
            )
        )
    low, high = np.percentile(medians, [2.5, 97.5])
    p_value = min(1.0, 2 * min(np.mean(medians <= 0), np.mean(medians >= 0)))
    return {
        "median": float(np.median(np.concatenate(groups))),
        "interval_95": [float(low), float(high)],
        "p_bootstrap": float(max(p_value, 1 / BOOTSTRAPS)),
        "median_of_each_mouse": [float(np.median(values)) for values in groups],
    }


def pooled_points(axis, xs, ys, subjects, color, seed: int = 0) -> None:
    """Draw the units of all mice pooled, with the medians of the mice and of all.

    Units are small light dots. The median of each mouse is a small open
    circle with its name. The pooled median is a large dot with the
    hierarchical-bootstrap 95% interval of each median as error bars.
    """
    axis.scatter(
        np.concatenate(xs),
        np.concatenate(ys),
        s=8,
        color=color,
        alpha=0.3,
        lw=0,
        zorder=2,
    )
    # The label of the higher median goes up and the lower one down.
    rank = np.argsort(np.argsort([np.median(y) for y in ys]))
    for k, (x, y, subject) in enumerate(zip(xs, ys, subjects, strict=True)):
        mx, my = np.median(x), np.median(y)
        axis.scatter(mx, my, s=22, facecolor="white", edgecolor=color, lw=1, zorder=3)
        # One line for each mouse, so that close medians keep legible names.
        axis.annotate(
            subject,
            (mx, my),
            xytext=(12, 9 if rank[k] == len(ys) - 1 else -13),
            textcoords="offset points",
            color=color,
            fontsize=5.5,
            arrowprops={"arrowstyle": "-", "color": color, "lw": 0.4},
        )
    axis.text(
        0.03,
        0.97,
        f"{sum(len(x) for x in xs)} units, {len(xs)} mice"
        + (
            f"\n{hidden} not drawn (a peak\nbelow {axis.get_xlim()[0]:g} spikes/s)"
            if (
                hidden := int(
                    sum(
                        np.sum((x <= axis.get_xlim()[0]) | (y <= axis.get_ylim()[0]))
                        for x, y in zip(xs, ys, strict=True)
                    )
                )
            )
            and axis.get_xscale() == "log"
            else ""
        ),
        transform=axis.transAxes,
        va="top",
        fontsize=6,
    )
    fx, fy = hierarchical_interval(xs, seed), hierarchical_interval(ys, seed)
    axis.errorbar(
        fx["median"],
        fy["median"],
        xerr=[
            [fx["median"] - fx["interval_95"][0]],
            [fx["interval_95"][1] - fx["median"]],
        ],
        yerr=[
            [fy["median"] - fy["interval_95"][0]],
            [fy["interval_95"][1] - fy["median"]],
        ],
        fmt="o",
        ms=6,
        color=color,
        mec="white",
        mew=0.6,
        elinewidth=1,
        capsize=2,
        zorder=4,
    )


def pooled_strip(axis, position: float, groups, subjects, color, seed: int = 0) -> dict:
    """Draw one column of pooled units, the median of each mouse, and the pooled median.

    Returns the statistics of `hierarchical_interval`.
    """
    rng = np.random.default_rng(seed)
    values = np.concatenate(groups)
    axis.scatter(
        position + rng.uniform(-0.15, 0.15, len(values)),
        values,
        s=8,
        color=color,
        alpha=0.3,
        lw=0,
        zorder=2,
    )
    stats = hierarchical_interval(groups, seed)
    medians = stats["median_of_each_mouse"]
    # The label of the higher median goes up and the lower one down, so that
    # close medians keep legible names.
    rank = np.argsort(np.argsort(medians))
    for median, subject, place in zip(medians, subjects, rank, strict=True):
        axis.scatter(
            position + 0.25,
            median,
            s=22,
            facecolor="white",
            edgecolor=color,
            lw=1,
            zorder=3,
        )
        axis.annotate(
            subject,
            (position + 0.25, median),
            xytext=(8, 7 if place == len(medians) - 1 else -9),
            textcoords="offset points",
            color=color,
            fontsize=5.5,
            arrowprops={"arrowstyle": "-", "color": color, "lw": 0.4},
        )
    low, high = stats["interval_95"]
    axis.errorbar(
        position,
        stats["median"],
        yerr=[[stats["median"] - low], [high - stats["median"]]],
        fmt="o",
        ms=6,
        color=color,
        mec="white",
        mew=0.6,
        elinewidth=1,
        capsize=2,
        zorder=4,
    )
    return stats


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
