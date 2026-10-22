"""Helpers of `main_figure.py`: training rates and the prediction score by rate."""

from __future__ import annotations

import matplotlib
import numpy as np
from scipy.stats import sem

matplotlib.use("Agg")

from thesis.ephys.analyses.glm.dmat import (
    BINWIDTH_S,
)

# Firing-rate groups of main-figure panel d, from the lowest rate to the highest.
# Shades of gray: the rate is not a concept of the model, and a darker shade is
# a higher rate.
RATE_GROUP_COLORS = {"low": "0.68", "medium": "0.42", "high": "black"}
RATE_GROUP_BINS = 4


def training_rate_and_deviance(
    results: list[dict], selections: list[dict]
) -> tuple[np.ndarray, np.ndarray]:
    """Pair training firing rate and cross-validated deviance by unit ID."""
    selection_by_id = {int(item["unit_id"]): item for item in selections}
    result_ids = [int(item["unit_id"]) for item in results]
    if len(selection_by_id) != len(selections) or len(set(result_ids)) != len(results):
        raise ValueError("Unit IDs must be unique in fit results.")
    if set(result_ids) != set(selection_by_id):
        raise ValueError("Final and validation results contain different units.")

    training_rate = np.asarray(
        [
            selection_by_id[unit_id]["training_mean_count"] / BINWIDTH_S
            for unit_id in result_ids
        ],
        dtype=float,
    )
    test_deviance = np.asarray(
        [item["cross_validated_deviance_explained"] for item in results],
        dtype=float,
    )
    if not np.isfinite(training_rate).all() or np.any(training_rate < 0):
        raise ValueError("Training firing rates must be finite and nonnegative.")
    if not np.isfinite(test_deviance).all():
        raise ValueError("Cross-validated deviance must be finite.")
    return training_rate, test_deviance


def _plot_bits_deviance_by_rate(
    axis,
    bits: np.ndarray,
    deviance: np.ndarray,
    rate: np.ndarray,
    marker: str = "o",
    linestyle: str = "-",
    label: bool = True,
) -> None:
    """Plot D² against bits/spike, with the units grouped by firing rate.

    Both measures have the same numerator, the log-likelihood gain over the
    constant-rate null. Bits/spike divides it by the spike count and D² by the
    null deviance, so for the same bits/spike a unit with a higher rate has a
    higher D². Units are cut into three groups of equal size by rate (low,
    medium, high).
    In each group the units are cut into `RATE_GROUP_BINS` bins of equal size by
    bits/spike; a marker is the mean of a bin and the band is ±1 SEM of D². The
    small points are the units.
    """
    if len(rate) < 2 * RATE_GROUP_BINS * len(RATE_GROUP_COLORS):
        axis.scatter(bits, deviance, color="black", s=10, linewidths=0)
    else:
        lines = []
        for members in np.array_split(np.argsort(rate), len(RATE_GROUP_COLORS)):
            bins = np.array_split(members[np.argsort(bits[members])], RATE_GROUP_BINS)
            lines.append(
                (
                    members,
                    np.asarray([bits[b].mean() for b in bins]),
                    np.asarray([deviance[b].mean() for b in bins]),
                    np.asarray([sem(deviance[b]) for b in bins]),
                )
            )
        for (members, x, y, error), (name, color) in zip(
            lines, RATE_GROUP_COLORS.items(), strict=True
        ):
            axis.scatter(
                bits[members],
                deviance[members],
                color=color,
                s=6,
                marker=marker,
                alpha=0.5,
                linewidths=0,
            )
            axis.fill_between(
                x, y - error, y + error, color=color, alpha=0.3, linewidth=0
            )
            axis.plot(
                x, y, color=color, marker=marker, ms=3, linewidth=1.2, ls=linestyle
            )
        if label:
            # The lines of two groups can end close together, so the labels
            # are a list in the empty upper left, in the colors of the lines.
            axis.text(
                0.03,
                0.97,
                "mean rate",
                transform=axis.transAxes,
                va="top",
                fontsize=7,
                color="0.3",
            )
            for row, (name, color) in enumerate(reversed(RATE_GROUP_COLORS.items())):
                axis.text(
                    0.03,
                    0.89 - 0.08 * row,
                    name,
                    transform=axis.transAxes,
                    va="top",
                    fontsize=7,
                    color=color,
                )
    axis.set_xlim(left=0)
    axis.set_ylim(bottom=0)
    axis.set_xlabel("test prediction (bits/spike)")
    axis.set_ylabel(r"test deviance explained ($D^2$)")
