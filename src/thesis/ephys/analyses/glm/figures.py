"""Write the figure data and the held-out prediction figure of the V1 GLM.

The prediction figure uses the cross-validation model that held out each plotted
trial. All rates condition on observed spike history. `main_figure.py` draws
the other panels from the saved figure data and uses the helpers here.
"""

from __future__ import annotations

import matplotlib
import numpy as np
from scipy.stats import sem

matplotlib.use("Agg")
from matplotlib.patheffects import withStroke

from thesis.ephys.analyses.glm.dmat import (
    BINWIDTH_S,
)
from thesis.plotting import (
    MODEL_COLOR,
    OBSERVED_COLOR,
)

COUNT_BIN_MS = 10
MIN_RATE_TRIAL_FRACTION = 0.5
# Firing-rate groups of main-figure panel d, from the lowest rate to the highest.
RATE_GROUP_COLORS = {"low": "#a50026", "medium": "#d9a400", "high": "#313695"}
RATE_GROUP_BINS = 4
PREDICTION_SEED = 2008


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


def _raster_events(
    counts: np.ndarray, times: np.ndarray, valid: np.ndarray
) -> list[np.ndarray]:
    return [
        np.repeat(times[keep], row[keep].astype(int, copy=False))
        for row, keep in zip(np.asarray(counts), np.asarray(valid), strict=True)
    ]


def binned_rates(
    values: np.ndarray, valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the unsmoothed mean firing rate in complete 10-ms bins.

    The second value marks the bins to plot: those where at least
    `MIN_RATE_TRIAL_FRACTION` of the trials are valid. Late bins come from the
    few trials with a late response entry, so their rate is not reliable.
    """
    values = np.atleast_2d(np.asarray(values, dtype=float))
    valid = np.atleast_2d(np.asarray(valid, dtype=bool))
    if values.shape != valid.shape or values.shape[1] < COUNT_BIN_MS:
        raise ValueError("Count values and valid rows must share a usable time grid.")
    samples = round(COUNT_BIN_MS / (1000 * BINWIDTH_S))
    bins = values.shape[1] // samples
    values = values[:, : bins * samples].reshape(len(values), bins, samples)
    complete = valid[:, : bins * samples].reshape(len(valid), bins, samples).all(axis=2)
    totals = values.sum(axis=2)
    numerator = np.where(complete, totals, 0).sum(axis=0)
    denominator = complete.sum(axis=0).astype(float)
    mean = np.divide(
        numerator,
        denominator,
        out=np.full(bins, np.nan),
        where=denominator > 0,
    )
    enough = denominator >= MIN_RATE_TRIAL_FRACTION * len(values)
    return mean / (samples * BINWIDTH_S), enough


def _plot_bits_deviance_by_rate(
    axis, bits: np.ndarray, deviance: np.ndarray, rate: np.ndarray
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
        longest = max(x[-1] for _, x, _, _ in lines)
        for (members, x, y, error), (name, color) in zip(
            lines, RATE_GROUP_COLORS.items(), strict=True
        ):
            axis.scatter(
                bits[members],
                deviance[members],
                color=color,
                s=5,
                alpha=0.5,
                linewidths=0,
            )
            axis.fill_between(
                x, y - error, y + error, color=color, alpha=0.3, linewidth=0
            )
            axis.plot(x, y, color=color, marker="o", ms=3, linewidth=1.2)
            # A line that ends left of the others has them on its right, so its
            # label goes above its end.
            right = x[-1] > 0.8 * longest
            axis.annotate(
                f"{name}\nfiring rate",
                (x[-1], y[-1]),
                xytext=(6, 0) if right else (-2, 6),
                textcoords="offset points",
                color=color,
                ha="left" if right else "right",
                va="center" if right else "bottom",
                fontsize=7,
                # A white outline keeps the label legible over single units.
                path_effects=[withStroke(linewidth=2, foreground="white")],
            )
    axis.set_xlim(left=0)
    axis.set_ylim(bottom=0)
    axis.set_xlabel("test prediction (bits/spike)")
    axis.set_ylabel(r"test deviance explained ($D^2$)")


def draw_prediction(
    axes,
    relative_times: np.ndarray,
    observed: np.ndarray,
    predicted: np.ndarray,
    conditional: np.ndarray,
    valid: np.ndarray,
) -> None:
    """Draw held-out observed spikes, predicted spikes, and both rates on three axes."""
    observed = np.asarray(observed)
    predicted = np.asarray(predicted)
    conditional = np.asarray(conditional)
    valid = np.asarray(valid, dtype=bool)
    if not (
        observed.shape == predicted.shape == conditional.shape == valid.shape
        and observed.ndim == 2
    ):
        raise ValueError(
            "Observed, predicted, conditional, and valid grids must match."
        )
    if observed.shape[1] != len(relative_times) or not valid.any():
        raise ValueError("Prediction grids must match a nonempty time axis.")
    if np.any(np.diff(valid.astype(int), axis=1) > 0):
        raise ValueError("Each plotted trial must contain one leading valid interval.")
    trial_count = observed.shape[0]
    shown = valid.any(axis=0)
    edges = (
        relative_times[np.flatnonzero(shown)[0]] - BINWIDTH_S / 2,
        relative_times[np.flatnonzero(shown)[-1]] + BINWIDTH_S / 2,
    )

    for axis, counts, color in zip(
        axes[:2], (observed, predicted), (OBSERVED_COLOR, MODEL_COLOR), strict=True
    ):
        for trial, keep in enumerate(valid, start=1):
            missing = np.flatnonzero(~keep & shown)
            if missing.size:
                axis.fill_betweenx(
                    (trial - 0.4, trial + 0.4),
                    relative_times[missing[0]] - BINWIDTH_S / 2,
                    edges[1],
                    color="0.94",
                    linewidth=0,
                    zorder=0,
                )
        axis.eventplot(
            _raster_events(counts, relative_times, valid),
            colors=color,
            lineoffsets=np.arange(1, trial_count + 1),
            linelengths=0.8,
            linewidths=0.3,
        )
        axis.set_ylim(0.5, trial_count + 0.5)
        axis.set_yticks([1, (trial_count + 1) // 2, trial_count])
        axis.axvline(0, color="0.75", linewidth=0.8, zorder=0)

    samples = round(COUNT_BIN_MS / (1000 * BINWIDTH_S))
    binned_times = (
        relative_times[: len(relative_times) // samples * samples]
        .reshape(-1, samples)
        .mean(axis=1)
    )
    for values, label, color in zip(
        (observed, conditional),
        ("observed", "predicted"),
        (OBSERVED_COLOR, MODEL_COLOR),
        strict=True,
    ):
        rate, keep = binned_rates(values, valid)
        axes[2].plot(
            binned_times[keep],
            rate[keep],
            color=color,
            linewidth=1.15,
            label=label,
            drawstyle="steps-mid",
        )
    for row, (label, color) in enumerate(
        (("observed", OBSERVED_COLOR), ("predicted", MODEL_COLOR))
    ):
        axes[2].text(
            0.98,
            0.97 - 0.09 * row,
            label,
            color=color,
            transform=axes[2].transAxes,
            ha="right",
            va="top",
        )
    axes[2].axvline(0, color="0.75", linewidth=0.8, zorder=0)

    axes[0].set_ylabel("observed\nheld-out trials")
    axes[1].set_ylabel("predicted\nheld-out trials")
    axes[2].set_ylabel(f"firing rate (spikes/s)\n{COUNT_BIN_MS}-ms bins")
    axes[2].set_xlim(*edges)
    axes[2].set_xlabel("time from first flash (s)")
