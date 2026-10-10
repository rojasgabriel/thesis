"""Shared plotting style helpers."""

from matplotlib.axes import Axes


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
