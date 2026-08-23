"""Shared plotting style helpers."""

from matplotlib.axes import Axes


def separate_axes(ax: Axes) -> None:
    """Trim spines to the visible ticks, following Lukas Oesch's helper."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ymin, ymax = ax.get_ylim()
    yticks = [tick for tick in ax.get_yticks() if ymin <= tick <= ymax + 1e-3]
    ax.spines["left"].set_bounds(yticks[0], yticks[-1])
    for location, tick in zip(ax.yaxis.get_minorticklocs(), ax.yaxis.get_minor_ticks()):
        if not yticks[0] <= location <= yticks[-1]:
            tick.tick1line.set_visible(False)
            tick.tick2line.set_visible(False)

    xmin, xmax = ax.get_xlim()
    xticks = [tick for tick in ax.get_xticks() if xmin <= tick <= xmax + 1e-3]
    ax.spines["bottom"].set_bounds(xticks[0], xticks[-1])
    for location, tick in zip(ax.xaxis.get_minorticklocs(), ax.xaxis.get_minor_ticks()):
        if not xticks[0] <= location <= xticks[-1]:
            tick.tick1line.set_visible(False)
            tick.tick2line.set_visible(False)
