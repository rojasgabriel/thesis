"""Create figures for the fitted V1 spike-prediction GLM.

The prediction figure uses the cross-validation model that held out each plotted
trial. The model-design example instead decomposes the canonical final refit,
which was fit on every valid bin. All rates condition on observed spike history.
"""

from __future__ import annotations

import json
import string
from pathlib import Path

import matplotlib
import numpy as np
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thesis.ephys.analyses.glm.dmat import BINWIDTH_S, VIDEO_COMPONENT_COUNTS
from thesis.ephys.analyses.glm.fit import load_unit_records
from thesis.ephys.analyses.glm.results import EXAMPLE_UNIT_ID, write_figure_data
from thesis.ephys.analyses.glm.unique_dev import DETAILED_GROUPS

OBSERVED_COLOR = "black"
MODEL_COLOR = "C0"
GROUP_COLORS = {
    "visual": "C0",
    "task": "C1",
    "video": "C2",
    "history": "C3",
}
TASK_LABELS = {
    "visual_flash": "Visual flash",
    "flash_pre_post_withdrawal": "Flash: post - pre withdrawal",
    "flash_adaptation": "Flash: adaptation",
    "flash_first": "First flash",
    "flash_later_pre": "Later flash, pre-withdrawal",
    "flash_later_post": "Later flash, post-withdrawal",
    "initiation": "Initiation",
    "withdrawal": "Withdrawal",
    "response_entry": "Response entry",
}
COUNT_BIN_MS = 10
PREDICTION_SEED = 2008
FIGURE_STYLE = {
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}
# Unique-deviance box plot: task blocks reuse the kernel labels; its color map
# is distinct from the groupwise contribution colors above.
DEVIANCE_LABELS = {
    "full_model": "Full model",
    "task": "Task",
    "video": "Motion energy",
    "history": "Spike history",
    **TASK_LABELS,
}
DEVIANCE_COLORS = {
    "full_model": "black",
    "task": "C0",
    "video": "C1",
    "history": "C2",
}


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


def _label_panels(axes, x: float, y: float = 1.05) -> None:
    """Put bold lowercase panel letters above and left of each axis."""
    for letter, axis in zip(string.ascii_lowercase, axes, strict=False):
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


def _save_figure(figure, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    svg = output.with_suffix(".svg")
    figure.savefig(svg, bbox_inches="tight")
    for stale in (".pdf", ".png"):
        output.with_suffix(stale).unlink(missing_ok=True)
    plt.close(figure)
    return svg


def binned_rates(
    values: np.ndarray, valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the unsmoothed mean firing rate in complete 10-ms bins."""
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
    return mean / (samples * BINWIDTH_S), complete.any(axis=0)


def plot_model_design(
    relative_times: np.ndarray,
    counts: np.ndarray,
    motion_times: np.ndarray,
    motion_scores: np.ndarray,
    events: dict[str, np.ndarray],
    contributions: dict[str, np.ndarray],
    linear_predictor: np.ndarray,
    expected_count: np.ndarray,
    design_metadata: dict,
    output: Path,
) -> Path:
    """Show observed inputs, fitted log-count terms, and counts for one trial."""
    relative_times = np.asarray(relative_times, dtype=float)
    counts = np.asarray(counts, dtype=float)
    motion_times = np.asarray(motion_times, dtype=float)
    motion_scores = np.asarray(motion_scores, dtype=float)
    if relative_times.ndim != 1 or len(relative_times) == 0:
        raise ValueError("Model-design times must be a nonempty vector.")
    if (
        counts.shape != relative_times.shape
        or expected_count.shape != relative_times.shape
    ):
        raise ValueError("Model-design counts, predictions, and times must match.")
    if linear_predictor.shape != relative_times.shape or any(
        values.shape != relative_times.shape for values in contributions.values()
    ):
        raise ValueError("Every fitted contribution must match the model time grid.")
    if motion_scores.ndim != 2 or motion_scores.shape[1] != 3:
        raise ValueError("The model-design figure requires three motion-PC traces.")
    if motion_times.shape != (len(motion_scores),):
        raise ValueError("Motion-PC times and scores must match.")
    if not all(
        np.isfinite(values).all()
        for values in (
            relative_times,
            counts,
            motion_times,
            motion_scores,
            linear_predictor,
            expected_count,
        )
    ):
        raise ValueError("Model-design values must be finite.")
    expected_events = {"visual_flash", "initiation", "withdrawal", "response_entry"}
    if set(events) != expected_events:
        raise ValueError("Model-design task events are incomplete.")

    event_arrays = {
        name: np.asarray(values, dtype=float) for name, values in events.items()
    }
    if any(
        values.ndim != 1 or not np.isfinite(values).all()
        for values in event_arrays.values()
    ):
        raise ValueError("Model-design event times must be finite vectors.")
    times_ms = 1000 * relative_times
    model_left_ms = 1000 * (relative_times[0] - BINWIDTH_S / 2)
    model_right_ms = 1000 * (relative_times[-1] + BINWIDTH_S / 2)
    event_values_ms = 1000 * np.concatenate(list(event_arrays.values()))
    left_ms = min(model_left_ms, float(event_values_ms.min()))
    right_ms = max(model_right_ms, float(event_values_ms.max()))
    spike_times_ms = np.repeat(times_ms, counts.astype(int, copy=False))
    manifest = {item["name"]: item for item in design_metadata["task_manifest"]}
    event_order = (
        "initiation",
        "visual_flash",
        "withdrawal",
        "response_entry",
    )
    event_labels = {
        "initiation": "Initiation",
        "visual_flash": "Visual flashes",
        "withdrawal": "Withdrawal",
        "response_entry": "Response entry",
    }
    event_colors = {
        "visual_flash": GROUP_COLORS["visual"],
        "initiation": GROUP_COLORS["task"],
        "withdrawal": GROUP_COLORS["task"],
        "response_entry": GROUP_COLORS["task"],
    }

    with plt.rc_context(FIGURE_STYLE):
        figure = plt.figure(figsize=(7.4, 9.0))
        outer = figure.add_gridspec(3, 1, height_ratios=(3.0, 4.0, 1.5), hspace=0.34)
        observed_grid = outer[0].subgridspec(
            3, 1, height_ratios=(1.8, 1.15, 0.72), hspace=0.08
        )
        event_axis = figure.add_subplot(observed_grid[0])
        motion_axis = figure.add_subplot(observed_grid[1], sharex=event_axis)
        history_axis = figure.add_subplot(observed_grid[2], sharex=event_axis)
        contribution_grid = outer[1].subgridspec(5, 1, hspace=0.08)
        contribution_axes = [
            figure.add_subplot(contribution_grid[index], sharex=event_axis)
            for index in range(5)
        ]
        count_axis = figure.add_subplot(outer[2], sharex=event_axis)

        positions = np.arange(len(event_order))[::-1]
        for position, name in zip(positions, event_order, strict=True):
            color = event_colors[name]
            kernel_start, kernel_stop = manifest[name]["kernel_range_s"]
            for event_time in event_arrays[name]:
                support_start = max(left_ms, 1000 * (event_time + kernel_start))
                support_stop = min(right_ms, 1000 * (event_time + kernel_stop))
                if name == "initiation":
                    support_stop = min(support_stop, 0.0)
                if support_stop > support_start:
                    event_axis.fill_betweenx(
                        (position - 0.28, position + 0.28),
                        support_start,
                        support_stop,
                        color=color,
                        alpha=0.18,
                        linewidth=0,
                    )
                event_axis.vlines(
                    1000 * event_time,
                    position - 0.34,
                    position + 0.34,
                    color=color,
                    linewidth=0.9,
                )
        event_axis.set_yticks(positions, [event_labels[name] for name in event_order])
        event_axis.set_ylim(-0.65, len(event_order) - 0.35)
        for index, linestyle in enumerate(("-", "--", ":")):
            motion_axis.plot(
                1000 * motion_times,
                motion_scores[:, index],
                color=GROUP_COLORS["video"],
                linewidth=0.8,
                linestyle=linestyle,
                label=f"PC {index + 1}",
            )
        motion_axis.axhline(0, color="black", linewidth=0.45)
        motion_axis.legend(loc="upper left", frameon=False, fontsize=7)
        motion_axis.set_ylabel("Motion-energy\nPC score")

        for spike_time in spike_times_ms:
            start = spike_time + 1000 * BINWIDTH_S
            stop = min(spike_time + 100, right_ms)
            if stop > start:
                history_axis.fill_betweenx(
                    (0.72, 1.28),
                    start,
                    stop,
                    color=GROUP_COLORS["history"],
                    alpha=0.15,
                    linewidth=0,
                )
                history_axis.vlines(
                    start,
                    0.72,
                    1.28,
                    color=GROUP_COLORS["history"],
                    linewidth=0.6,
                )
        history_axis.set_yticks((1,), ("Spike history",))
        history_axis.set_ylim(0.45, 1.55)

        trace_specs = (
            ("visual", "Visual flashes", GROUP_COLORS["visual"]),
            ("task", "Other task events", GROUP_COLORS["task"]),
            ("video", "Motion-energy PCs", GROUP_COLORS["video"]),
            ("history", "Spike history", GROUP_COLORS["history"]),
        )
        for axis, (name, label, color) in zip(
            contribution_axes[:4], trace_specs, strict=True
        ):
            axis.plot(times_ms, contributions[name], color=color, linewidth=0.9)
            axis.axhline(0, color="black", linewidth=0.45, linestyle="--")
            axis.set_ylabel(label, color=color, rotation=0, ha="right", va="center")
        contribution_axes[-1].plot(
            times_ms, linear_predictor, color="black", linewidth=1.0
        )
        contribution_axes[-1].set_ylabel(
            r"Total log count, $\eta(t)$", rotation=0, ha="right", va="center"
        )
        for axis in contribution_axes:
            axis.tick_params(axis="y", labelsize=6)

        predicted_counts = np.random.default_rng(
            PREDICTION_SEED + EXAMPLE_UNIT_ID
        ).poisson(expected_count)
        predicted_spike_times_ms = np.repeat(
            times_ms, predicted_counts.astype(int, copy=False)
        )
        count_axis.eventplot(
            [spike_times_ms, predicted_spike_times_ms],
            colors=[OBSERVED_COLOR, MODEL_COLOR],
            lineoffsets=[1, 0],
            linelengths=0.8,
            linewidths=0.5,
        )
        count_axis.set_yticks([1, 0], ["Observed", "Predicted"])
        count_axis.set_ylim(-0.6, 1.6)
        count_axis.set_xlabel("Time from first measured flash (ms)")

        all_axes = [
            event_axis,
            motion_axis,
            history_axis,
            *contribution_axes,
            count_axis,
        ]
        for axis in all_axes:
            axis.axvline(0, color="0.65", linewidth=0.6, zorder=0)
            axis.set_xlim(left_ms, right_ms)
        for axis in all_axes[:-1]:
            axis.tick_params(axis="x", labelbottom=False)
        _label_panels((event_axis, contribution_axes[0], count_axis), -0.15, 1.08)
        figure.subplots_adjust(left=0.23, right=0.98, bottom=0.07, top=0.97)
        return _save_figure(figure, output)


def plot_population_summary(
    results: list[dict], selections: list[dict], output: Path
) -> tuple[Path, float]:
    """Show camera selection, held-out performance, and its rate relation."""
    selected_components = {item["components"] for item in results}
    if len(selected_components) != 1:
        raise ValueError("All final fits must use one camera-PC count.")
    selected = selected_components.pop()
    component_counts = (0, *VIDEO_COMPONENT_COUNTS)
    validation_deviance = [
        np.mean(
            [
                item["baseline"]["validation"]["deviance_explained"]
                for item in selections
            ]
        ),
        *[
            np.mean(
                [
                    item["plus_video"][str(count)]["validation"]["deviance_explained"]
                    for item in selections
                ]
            )
            for count in VIDEO_COMPONENT_COUNTS
        ],
    ]
    training_rate, deviance = training_rate_and_deviance(results, selections)
    bits = np.asarray([item["cross_validated_bits_per_spike"] for item in results])
    rate_deviance_rho = float(spearmanr(training_rate, deviance).statistic)
    if not np.isfinite(rate_deviance_rho):
        raise ValueError("Training firing rate and test score must vary across units.")

    with plt.rc_context(FIGURE_STYLE):
        figure, axes = plt.subplots(2, 2, figsize=(7.4, 5.6))
        axes = axes.ravel()

        positions = np.arange(len(component_counts))
        axes[0].plot(
            positions,
            validation_deviance,
            color="black",
            marker="o",
            markerfacecolor="white",
        )
        selected_position = component_counts.index(selected)
        axes[0].scatter(
            selected_position,
            validation_deviance[selected_position],
            color="black",
            s=75,
            zorder=3,
        )
        axes[0].set_xticks(positions, component_counts)
        axes[0].set_xlabel("Candidate motion-energy PCs")
        axes[0].set_ylabel(r"Mean validation deviance explained ($D^2$)")

        for axis, values, xlabel, digits in (
            (axes[1], deviance, r"Test deviance explained ($D^2$)", 3),
            (axes[2], bits, "Test prediction (bits/spike)", 3),
        ):
            median = float(np.median(values))
            axis.hist(values, bins=24, color="black", alpha=0.75)
            axis.axvline(0, color="0.65", linestyle="--", linewidth=0.8)
            axis.axvline(median, color="black", linewidth=1.5)
            axis.text(
                0.98,
                0.93,
                f"median = {median:.{digits}f}",
                color="black",
                transform=axis.transAxes,
                ha="right",
                va="top",
            )
            axis.set_xlabel(xlabel)
            axis.set_ylabel("Units")

        axes[3].scatter(training_rate, deviance, color="black", alpha=0.65, s=16)
        axes[3].axhline(0, color="0.65", linestyle="--", linewidth=0.8)
        if np.all(training_rate > 0):
            axes[3].set_xscale("log")
        axes[3].set_xlabel("mean firing rate (spikes/s)")
        axes[3].set_ylabel(r"test deviance explained ($D^2$)")
        axes[3].text(
            0.98,
            0.04,
            f"Spearman ρ = {rate_deviance_rho:.2f}\nn = {len(results)} units",
            color="black",
            transform=axes[3].transAxes,
            ha="right",
            va="bottom",
        )

        _label_panels(axes, -0.14)
        figure.tight_layout(h_pad=2.0, w_pad=2.0)
        return _save_figure(figure, output), rate_deviance_rho


def plot_bits_vs_deviance(results: list[dict], output: Path) -> Path:
    """Scatter each unit's absolute prediction against its explained fraction."""
    bits = np.asarray([item["cross_validated_bits_per_spike"] for item in results])
    deviance = np.asarray(
        [item["cross_validated_deviance_explained"] for item in results]
    )
    rho = float(spearmanr(bits, deviance).statistic)

    with plt.rc_context(FIGURE_STYLE):
        figure, axis = plt.subplots(figsize=(3.7, 3.5))
        axis.scatter(bits, deviance, color="black", alpha=0.65, s=16)
        axis.axhline(0, color="0.65", linestyle="--", linewidth=0.8)
        axis.axvline(0, color="0.65", linestyle="--", linewidth=0.8)
        axis.set_xlabel("Test prediction (bits/spike)")
        axis.set_ylabel(r"Test deviance explained ($D^2$)")
        axis.text(
            0.98,
            0.04,
            f"Spearman ρ = {rho:.2f}\nn = {len(results)} units",
            color="black",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
        )
        figure.tight_layout()
        return _save_figure(figure, output)


def plot_kernel_figure(
    kernels: dict[str, tuple[np.ndarray, np.ndarray]],
    task_names: list[str],
    output: Path,
) -> Path:
    """Show every interpretable task and history kernel for the example unit."""
    names = [*task_names, "history"]
    rows = int(np.ceil(len(names) / 2))

    with plt.rc_context(FIGURE_STYLE):
        figure = plt.figure(figsize=(7.4, 2.25 * rows))
        grid = figure.add_gridspec(rows, 2)
        axes = []
        for index in range(len(names)):
            location = (
                grid[index // 2, :]
                if len(names) % 2 and index == len(names) - 1
                else grid[index // 2, index % 2]
            )
            axes.append(figure.add_subplot(location))
        for axis, name in zip(axes, names, strict=True):
            times_ms, values = kernels[name]
            axis.plot(times_ms, values, color="black", linewidth=1.1)
            axis.axhline(0, color="black", linestyle="--", linewidth=0.5)
            if times_ms[0] <= 0 <= times_ms[-1]:
                axis.axvline(0, color="black", linewidth=0.45)
            axis.set_title(
                "Spike history" if name == "history" else TASK_LABELS[name],
                fontsize=8,
            )
            axis.set_xlabel(
                "Time since spike (ms)" if name == "history" else "Time from event (ms)"
            )
            axis.set_ylabel(
                r"$\Delta$ log rate / spike"
                if name == "history"
                else r"$\Delta$ log rate",
                fontsize=7,
            )
        _label_panels(axes, -0.16)
        task_limit = 1.05 * max(
            np.max(np.abs(kernels[name][1]), initial=0.0) for name in task_names
        )
        if task_limit > 0:
            for axis in axes[: len(task_names)]:
                axis.set_ylim(-task_limit, task_limit)
        figure.tight_layout(h_pad=1.8, w_pad=1.8)
        return _save_figure(figure, output)


def plot_prediction_figure(
    relative_times: np.ndarray,
    observed: np.ndarray,
    predicted: np.ndarray,
    conditional: np.ndarray,
    valid: np.ndarray,
    output: Path,
) -> Path:
    """Compare held-out observed and one-step-predicted spikes for unit 197."""
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

    with plt.rc_context(FIGURE_STYLE):
        figure, axes = plt.subplots(
            3,
            1,
            figsize=(4.5, 5.8),
            sharex=True,
            gridspec_kw={"height_ratios": [1, 1, 1.2]},
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
            ("Observed", "Predicted"),
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
        axes[2].legend(frameon=False)
        axes[2].axvline(0, color="0.75", linewidth=0.8, zorder=0)

        _label_panels(axes, -0.13)
        axes[0].set_ylabel("Observed\nheld-out trials")
        axes[1].set_ylabel("Predicted\nheld-out trials")
        axes[2].set_ylabel(f"Firing rate (spikes/s)\n{COUNT_BIN_MS}-ms bins")
        axes[-1].set_xlim(*edges)
        axes[-1].set_xlabel("Time from first measured flash (s)")
        figure.align_ylabels(axes)
        figure.tight_layout(h_pad=0.8)
        return _save_figure(figure, output)


def _plot_groups(
    axis,
    records: list[dict],
    groups: tuple[str, ...],
    metric: str,
) -> None:
    """Draw one population box plot for the requested deviance measure."""
    values = [
        np.asarray(
            [
                record[metric]
                if group == "full_model"
                else record["groups"][group][metric]
                for record in records
            ]
        )
        for group in groups
    ]
    positions = np.arange(len(groups))
    boxes = axis.boxplot(
        values,
        positions=positions,
        widths=0.58,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 0.8},
        whiskerprops={"color": "0.3", "linewidth": 0.7},
        capprops={"color": "0.3", "linewidth": 0.7},
    )
    for group, patch in zip(groups, boxes["boxes"], strict=True):
        color = DEVIANCE_COLORS.get(group, DEVIANCE_COLORS["task"])
        patch.set(facecolor=color, edgecolor="black", alpha=0.45, linewidth=0.7)
    axis.axhline(0, color="0.35", linestyle="--", linewidth=0.7, zorder=0)
    axis.set_xticks(positions, [DEVIANCE_LABELS[group] for group in groups])
    axis.tick_params(axis="x", labelrotation=48)
    for label, group in zip(axis.get_xticklabels(), groups, strict=True):
        label.set_ha("right")
        label.set_color(DEVIANCE_COLORS.get(group, DEVIANCE_COLORS["task"]))
    axis.margins(x=0.04)


def plot_deviance_explained(records: list[dict], output: Path) -> Path:
    """Plot maximal and unique deviance for each model regressor."""
    with plt.rc_context(FIGURE_STYLE):
        figure, axes = plt.subplots(1, 3, figsize=(10.5, 3.5))
        _plot_groups(
            axes[0], records, ("full_model",), "complete_test_deviance_explained"
        )
        _plot_groups(
            axes[1], records, DETAILED_GROUPS, "maximal_test_deviance_explained"
        )
        _plot_groups(
            axes[2], records, DETAILED_GROUPS, "unique_test_deviance_explained"
        )
        axes[0].set_ylabel(r"Full-model test deviance explained ($D^2$)")
        axes[1].set_ylabel(r"Maximal deviance explained ($\Delta D^2$)")
        axes[2].set_ylabel(r"Unique deviance explained ($\Delta D^2$)")
        _label_panels(axes, -0.12, 1.04)
        figure.subplots_adjust(
            bottom=0.35, left=0.07, right=0.99, top=0.94, wspace=0.34
        )
        return _save_figure(figure, output)


def plot_fit(windows: Path, design: Path, fit_dir: Path) -> None:
    """Write this fit's figures inside its own artifact directory.

    Population panels use the saved per-unit records. The kernel, model-design,
    and prediction panels are for unit 197, and only when that unit was fit.
    """
    figure_dir = fit_dir
    checkpoints = fit_dir / "checkpoints"
    results = load_unit_records(checkpoints, "unit_*_final.json")
    selections = load_unit_records(checkpoints, "unit_*_validation.json")
    if len(results) != len(selections):
        raise ValueError("Final and validation result counts differ.")

    summary, rate_deviance_rho = plot_population_summary(
        results, selections, figure_dir / "summary"
    )
    bits = plot_bits_vs_deviance(results, figure_dir / "bits_vs_deviance")
    written = {"summary": str(summary), "bits_vs_deviance": str(bits)}
    payload = {
        "training_rate_test_deviance_spearman_rho": rate_deviance_rho,
        "figure_dir": str(figure_dir),
        "figures": written,
    }
    if EXAMPLE_UNIT_ID not in {int(item["unit_id"]) for item in results}:
        print(
            f"Example unit {EXAMPLE_UNIT_ID} was not fit; "
            "skipping its kernel, design, and prediction figures.",
            flush=True,
        )
        print(json.dumps(payload, indent=2))
        return

    write_figure_data(windows, design, fit_dir)
    data = np.load(fit_dir / "figure_data.npz", allow_pickle=False)
    with (fit_dir / "figure_data.json").open() as handle:
        meta = json.load(handle)

    written["fitted_kernels"] = str(
        plot_kernel_figure(
            {
                name: (data[f"kern_t__{name}"], data[f"kern_v__{name}"])
                for name in meta["kernel_keys"]
            },
            meta["display_task_names"],
            figure_dir / "fitted_kernels",
        )
    )
    written["model_design"] = str(
        plot_model_design(
            data["model_times"],
            data["model_counts"],
            data["motion_times"],
            data["motion_scores"],
            {
                name: data[f"event__{name}"]
                for name in (
                    "visual_flash",
                    "initiation",
                    "withdrawal",
                    "response_entry",
                )
            },
            {
                name: data[f"contrib__{name}"]
                for name in ("visual", "task", "video", "history")
            },
            data["linear_predictor"],
            data["fitted_count"],
            {"task_manifest": meta["task_manifest"]},
            figure_dir / "model_design",
        )
    )
    prediction_observed = data["prediction_observed"]
    conditional = data["conditional"]
    test_valid = data["test_valid"]
    seed = PREDICTION_SEED + int(meta["example_unit_id"])
    predicted = np.zeros_like(prediction_observed, dtype=int)
    predicted[test_valid] = np.random.default_rng(seed).poisson(conditional[test_valid])
    written["predicted_spike_trains"] = str(
        plot_prediction_figure(
            data["relative_times"],
            prediction_observed,
            predicted,
            conditional,
            test_valid,
            figure_dir / "predicted_spike_trains",
        )
    )
    print(
        json.dumps(
            {
                **payload,
                **{k: v for k, v in meta.items() if not isinstance(v, list)},
                "prediction_observed_spikes": int(
                    prediction_observed[test_valid].sum()
                ),
                "prediction_sampled_spikes": int(predicted[test_valid].sum()),
                "prediction_seed": seed,
            },
            indent=2,
        )
    )
