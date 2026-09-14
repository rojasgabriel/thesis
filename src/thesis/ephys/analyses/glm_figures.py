"""Create figures for the fitted V1 spike-prediction GLM.

The prediction figure uses the cross-validation model that held out each plotted
trial. The model-design example instead decomposes the canonical final refit,
which was fit on every valid bin. All rates condition on observed spike history.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thesis.ephys.analyses.glm import (
    HISTORY_COLUMNS,
    VIDEO_COMPONENT_COUNTS,
    VIDEO_COMPONENTS,
    _load_selected_trials,
    _load_windows,
    _valid_bin_mask,
    build_unit_counts,
    build_unit_history,
    cross_validation_partitions,
    spike_history_basis,
    split_flash_events,
    task_temporal_bases,
    training_zscore,
)
from thesis.ephys.preprocessing.prepare_glm import BINWIDTH_S
from thesis.ephys.units import fetch_unit_table

OBSERVED_COLOR = "black"
MODEL_COLOR = "C0"
GROUP_COLORS = {
    "visual": "C0",
    "task": "C1",
    "video": "C2",
    "history": "C3",
}
TASK_LABELS = {
    "stationary_flash": "Stationary flash",
    "running_flash": "Running flash",
    "center_poke": "Center poke",
    "center_exit": "Center exit",
    "response_entry": "Response entry",
    "response_side": "Response side\n(right − left)",
}
SMOOTHING_MS = 20
PREDICTION_SEED = 2008
EXAMPLE_UNIT_ID = 197
# Rebound by make_figures; read by _save_figure so plot signatures stay small.
FIGURE_FORMATS: tuple[str, ...] = ("pdf", "png")
FIGURE_STYLE = {
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
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


def _save_figure(figure, output: Path) -> tuple[Path | None, Path | None]:
    """Write the figure in the formats FIGURE_FORMATS names, skipping the rest."""
    output.parent.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path | None] = {"pdf": None, "png": None}
    if "pdf" in FIGURE_FORMATS:
        paths["pdf"] = output.with_suffix(".pdf")
        figure.savefig(paths["pdf"], bbox_inches="tight")
    if "png" in FIGURE_FORMATS:
        paths["png"] = output.with_suffix(".png")
        figure.savefig(paths["png"], dpi=300, bbox_inches="tight")
    plt.close(figure)
    return paths["pdf"], paths["png"]


def fitted_trial_contributions(
    design: np.ndarray, result: dict, design_metadata: dict
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, float]:
    """Decompose one fitted trial into exact groupwise log-rate terms."""
    design = np.asarray(design, dtype=float)
    components = int(result["components"])
    common_columns = int(design_metadata["base_columns"]) + components
    coefficients = np.asarray(result["coefficients"], dtype=float)
    expected_columns = common_columns + HISTORY_COLUMNS
    if design.ndim != 2 or design.shape[1] != expected_columns:
        raise ValueError("Trial design and fitted coefficient counts differ.")
    if coefficients.shape != (expected_columns,):
        raise ValueError("Saved coefficient count does not match the trial design.")
    if not np.isfinite(design).all() or not np.isfinite(coefficients).all():
        raise ValueError("Trial design and coefficients must be finite.")

    visual = np.zeros(len(design))
    task = np.zeros(len(design))
    cursor = 0
    for item in design_metadata["task_manifest"]:
        stop = cursor + int(item["columns"])
        target = visual if item["name"].endswith("_flash") else task
        target += design[:, cursor:stop] @ coefficients[cursor:stop]
        cursor = stop
    if cursor != int(design_metadata["base_columns"]):
        raise ValueError("Task manifest and task-column count differ.")
    contributions: dict[str, np.ndarray] = {
        "visual": visual,
        "task": task,
        "video": design[:, cursor:common_columns] @ coefficients[cursor:common_columns],
        "history": design[:, common_columns:] @ coefficients[common_columns:],
    }
    linear_predictor = float(result["intercept"]) + (
        contributions["visual"]
        + contributions["task"]
        + contributions["video"]
        + contributions["history"]
    )
    direct = float(result["intercept"]) + design @ coefficients
    error = float(np.max(np.abs(linear_predictor - direct), initial=0.0))
    if not np.allclose(linear_predictor, direct, rtol=1e-11, atol=1e-12):
        raise ValueError("Group contributions do not reproduce the fitted predictor.")
    with np.errstate(over="ignore"):
        rate = np.exp(linear_predictor) / BINWIDTH_S
    if not np.isfinite(rate).all() or np.any(rate <= 0):
        raise ValueError("Fitted trial rates must be finite and positive.")
    return contributions, linear_predictor, rate, error


def plot_model_design(
    relative_times: np.ndarray,
    counts: np.ndarray,
    motion_times: np.ndarray,
    motion_scores: np.ndarray,
    events: dict[str, np.ndarray],
    response_side: int,
    contributions: dict[str, np.ndarray],
    linear_predictor: np.ndarray,
    rate: np.ndarray,
    design_metadata: dict,
    output: Path,
) -> tuple[Path | None, Path | None]:
    """Show observed inputs, fitted log-rate terms, and rate for one trial."""
    relative_times = np.asarray(relative_times, dtype=float)
    counts = np.asarray(counts, dtype=float)
    motion_times = np.asarray(motion_times, dtype=float)
    motion_scores = np.asarray(motion_scores, dtype=float)
    if relative_times.ndim != 1 or len(relative_times) == 0:
        raise ValueError("Model-design times must be a nonempty vector.")
    if counts.shape != relative_times.shape or rate.shape != relative_times.shape:
        raise ValueError("Model-design counts, rates, and times must match.")
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
            rate,
        )
    ):
        raise ValueError("Model-design values must be finite.")
    expected_events = {
        "stationary_flash",
        "running_flash",
        "center_poke",
        "center_exit",
        "response_entry",
        "response_side",
    }
    if set(events) != expected_events or response_side not in (-1, 1):
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
        "center_poke",
        "stationary_flash",
        "running_flash",
        "center_exit",
        "response_entry",
        "response_side",
    )
    event_labels = {
        "center_poke": "Center poke",
        "stationary_flash": "Stationary flashes",
        "running_flash": "Running flashes",
        "center_exit": "Center exit",
        "response_entry": "Response entry",
        "response_side": "Response side: right"
        if response_side == 1
        else "Response side: left",
    }
    event_colors = {
        "stationary_flash": GROUP_COLORS["visual"],
        "running_flash": GROUP_COLORS["visual"],
        "center_poke": GROUP_COLORS["task"],
        "center_exit": GROUP_COLORS["task"],
        "response_entry": GROUP_COLORS["task"],
        "response_side": GROUP_COLORS["task"],
    }

    with plt.rc_context(FIGURE_STYLE):
        figure = plt.figure(figsize=(7.4, 9.0))
        outer = figure.add_gridspec(3, 1, height_ratios=(3.2, 4.0, 2.0), hspace=0.34)
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
        prediction_grid = outer[2].subgridspec(
            2, 1, height_ratios=(2.0, 0.55), hspace=0.05
        )
        rate_axis = figure.add_subplot(prediction_grid[0], sharex=event_axis)
        spike_axis = figure.add_subplot(prediction_grid[1], sharex=event_axis)

        positions = np.arange(len(event_order))[::-1]
        for position, name in zip(positions, event_order, strict=True):
            color = event_colors[name]
            kernel_start, kernel_stop = manifest[name]["kernel_range_s"]
            for event_time in event_arrays[name]:
                support_start = max(left_ms, 1000 * (event_time + kernel_start))
                support_stop = min(right_ms, 1000 * (event_time + kernel_stop))
                if name == "center_poke":
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
        history_axis.vlines(spike_times_ms, -0.28, 0.28, color="black", linewidth=0.7)
        history_axis.set_yticks((0, 1), ("Observed spikes", "History support"))
        history_axis.set_ylim(-0.55, 1.55)

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
            "Sum + intercept", rotation=0, ha="right", va="center"
        )
        for axis in contribution_axes:
            axis.tick_params(axis="y", labelsize=6)

        rate_axis.plot(times_ms, rate, color="black", linewidth=1.0)
        rate_axis.set_ylabel("Predicted rate\n(spikes/s)")
        spike_axis.vlines(spike_times_ms, 0, 1, color="black", linewidth=0.7)
        spike_axis.set_yticks((0.5,), ("Observed spikes",))
        spike_axis.set_ylim(0, 1)
        spike_axis.set_xlabel("Time from first measured flash (ms)")

        all_axes = [
            event_axis,
            motion_axis,
            history_axis,
            *contribution_axes,
            rate_axis,
            spike_axis,
        ]
        for axis in all_axes:
            axis.axvline(0, color="0.65", linewidth=0.6, zorder=0)
            axis.set_xlim(left_ms, right_ms)
        for axis in all_axes[:-1]:
            axis.tick_params(axis="x", labelbottom=False)
        for letter, axis in zip(
            "abc", (event_axis, contribution_axes[0], rate_axis), strict=True
        ):
            axis.text(
                -0.15,
                1.08,
                letter,
                transform=axis.transAxes,
                fontweight="bold",
                fontsize=10,
                va="bottom",
            )
        figure.subplots_adjust(left=0.23, right=0.98, bottom=0.07, top=0.97)
        return _save_figure(figure, output)


def plot_population_summary(
    results: list[dict], selections: list[dict], output: Path
) -> tuple[Path | None, Path | None, float]:
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
        axes[0].set_ylabel("Mean validation deviance explained")

        for axis, values, xlabel, digits in (
            (axes[1], deviance, "Test deviance explained", 3),
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
        axes[3].set_ylabel("test deviance explained")
        axes[3].text(
            0.04,
            0.94,
            f"Spearman ρ = {rate_deviance_rho:.2f}\nn = {len(results)} units",
            color="black",
            transform=axes[3].transAxes,
            ha="left",
            va="top",
        )

        for letter, axis in zip("abcd", axes, strict=True):
            axis.text(
                -0.14,
                1.05,
                letter,
                transform=axis.transAxes,
                ha="left",
                va="bottom",
                fontweight="bold",
                fontsize=10,
            )
        figure.tight_layout(h_pad=2.0, w_pad=2.0)
        pdf_path, png_path = _save_figure(figure, output)
        return pdf_path, png_path, rate_deviance_rho


def fitted_kernels(
    result: dict, design_metadata: dict, history_scale: np.ndarray
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Reconstruct one unit's task and self-history filters."""
    components = int(result["components"])
    common_columns = int(design_metadata["base_columns"]) + components
    coefficients = np.asarray(result["coefficients"], dtype=float)
    if coefficients.shape != (common_columns + HISTORY_COLUMNS,):
        raise ValueError("Saved coefficient count does not match the selected design.")
    common_scale = np.asarray(design_metadata["training_scale"], dtype=float)[
        :common_columns
    ]
    if common_scale.shape != (common_columns,) or np.any(common_scale <= 0):
        raise ValueError("Common-design scales must be positive and complete.")
    history_scale = np.asarray(history_scale, dtype=float)
    if history_scale.shape != (HISTORY_COLUMNS,) or np.any(history_scale <= 0):
        raise ValueError("Spike-history scales must be positive and complete.")

    weights = coefficients[:common_columns] / common_scale
    kernels: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    bases = task_temporal_bases()
    cursor = 0
    for item in design_metadata["task_manifest"]:
        name = item["name"]
        basis = bases[name]
        stop = cursor + int(item["columns"])
        if stop - cursor != basis.basis.shape[1]:
            raise ValueError(f"Basis size differs from saved metadata for {name}.")
        kernels[name] = (
            1000 * basis.basis_time,
            weights[cursor:stop] @ basis.basis.T,
        )
        cursor = stop
    if cursor != int(design_metadata["task_columns"]):
        raise ValueError("Task manifest and saved task-column count differ.")

    history_basis = spike_history_basis()
    kernels["history"] = (
        1 + 1000 * history_basis.basis_time,
        history_basis.basis @ (coefficients[common_columns:] / history_scale),
    )
    if not all(np.isfinite(values).all() for _, values in kernels.values()):
        raise ValueError("Reconstructed kernels must be finite.")
    return kernels


def task_kernel_display_values(name: str, values: np.ndarray) -> np.ndarray:
    """Convert signed-code filters to the displayed condition difference."""
    values = np.asarray(values)
    return 2 * values if name == "response_side" else values


def plot_kernel_figure(
    kernels: dict[str, tuple[np.ndarray, np.ndarray]],
    task_names: list[str],
    output: Path,
) -> tuple[Path | None, Path | None]:
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
        for letter, axis, name in zip(
            "abcdefghijklmnopqrstuvwxyz", axes, names, strict=False
        ):
            times_ms, values = kernels[name]
            values = task_kernel_display_values(name, values)
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
            axis.text(
                -0.16,
                1.05,
                letter,
                transform=axis.transAxes,
                ha="left",
                va="bottom",
                fontweight="bold",
                fontsize=10,
            )
        figure.tight_layout(h_pad=1.8, w_pad=1.8)
        return _save_figure(figure, output)


def _select_count_typical_trial(observed: np.ndarray) -> int:
    """Select the earliest trial nearest the median observed spike count."""
    if observed.ndim != 2 or len(observed) == 0:
        raise ValueError("Observed spikes must be a nonempty trial-by-bin matrix.")
    totals = observed.sum(axis=1)
    median = float(np.median(totals))
    return min(
        range(len(totals)), key=lambda index: (abs(totals[index] - median), index)
    )


def plot_prediction_figure(
    relative_times: np.ndarray,
    observed: np.ndarray,
    predicted: np.ndarray,
    conditional: np.ndarray,
    valid: np.ndarray,
    output: Path,
) -> tuple[Path | None, Path | None]:
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

        sigma_bins = SMOOTHING_MS / (BINWIDTH_S * 1000)
        denominator = gaussian_filter1d(valid.sum(axis=0).astype(float), sigma_bins)
        rates = [
            np.divide(
                gaussian_filter1d(np.where(valid, values, 0).sum(axis=0), sigma_bins),
                denominator * BINWIDTH_S,
                out=np.full(len(relative_times), np.nan),
                where=denominator > 0,
            )
            for values in (observed, conditional)
        ]
        for rate, label, color in zip(
            rates, ("Observed", "Model"), (OBSERVED_COLOR, MODEL_COLOR), strict=True
        ):
            axes[2].plot(relative_times, rate, color=color, linewidth=1.15, label=label)
        axes[2].legend(frameon=False)
        axes[2].axvline(0, color="0.75", linewidth=0.8, zorder=0)

        for letter, axis in zip("abc", axes, strict=True):
            axis.text(
                -0.13,
                1.05,
                letter,
                transform=axis.transAxes,
                ha="left",
                va="bottom",
                fontweight="bold",
                fontsize=10,
            )
        axes[0].set_ylabel("Observed\nheld-out trials")
        axes[1].set_ylabel("Predicted\nheld-out trials")
        axes[2].set_ylabel(f"{SMOOTHING_MS}-ms smoothed\nmean rate (spikes/s)")
        axes[-1].set_xlim(*edges)
        axes[-1].set_xlabel("Time from first measured flash (s)")
        figure.align_ylabels(axes)
        figure.tight_layout(h_pad=0.8)
        return _save_figure(figure, output)


def make_figures(
    windows: Path,
    design: Path,
    fit_dir: Path,
    output_dir: Path | None = None,
    formats: tuple[str, ...] = ("pdf", "png"),
) -> None:
    """Write every figure for one completed fit directory.

    Figures land in the fit's figures directory unless `output_dir` says
    otherwise. `formats` selects which of pdf and png to write.
    """
    global FIGURE_FORMATS

    if not set(formats) <= {"pdf", "png"} or not formats:
        raise ValueError("Formats must be a non-empty subset of pdf and png.")
    FIGURE_FORMATS = tuple(formats)
    figure_dir = output_dir if output_dir is not None else fit_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = fit_dir / "checkpoints"
    results = []
    for path in sorted(checkpoints.glob("unit_*_final.json")):
        with path.open() as handle:
            results.append(json.load(handle))
    selections = []
    for path in sorted(checkpoints.glob("unit_*_validation.json")):
        with path.open() as handle:
            selections.append(json.load(handle))
    fold_records = []
    for path in sorted(checkpoints.glob("unit_*_folds.json")):
        with path.open() as handle:
            fold_records.append(json.load(handle))
    if len(results) != len(selections) or len(results) != len(fold_records):
        raise ValueError("Final, validation, and fold result counts differ.")
    selected_components = {item["components"] for item in results}
    if len(selected_components) != 1:
        raise ValueError("All final fits must use one camera-PC count.")
    selected_components = int(selected_components.pop())
    if selected_components != VIDEO_COMPONENTS:
        raise ValueError(f"Figures require the fixed {VIDEO_COMPONENTS}-PC model.")
    with design.with_suffix(".json").open() as handle:
        design_metadata = json.load(handle)
    if int(design_metadata.get("video_columns_per_component", 0)) != 1:
        raise ValueError("Figures require one contemporaneous column per video PC.")
    summary_pdf, summary_png, rate_deviance_rho = plot_population_summary(
        results, selections, figure_dir / "summary"
    )
    result_by_id = {int(item["unit_id"]): item for item in results}
    selection_by_id = {int(item["unit_id"]): item for item in selections}
    if len(result_by_id) != len(results) or len(selection_by_id) != len(selections):
        raise ValueError("Fit results must contain unique unit IDs.")
    if EXAMPLE_UNIT_ID not in result_by_id or EXAMPLE_UNIT_ID not in selection_by_id:
        raise ValueError(f"Example unit {EXAMPLE_UNIT_ID} is not in the fitted units.")
    example_result = result_by_id[EXAMPLE_UNIT_ID]
    example_training_rate = (
        float(selection_by_id[EXAMPLE_UNIT_ID]["training_mean_count"]) / BINWIDTH_S
    )
    task_names = [item["name"] for item in design_metadata["task_manifest"]]

    prepared = _load_windows(windows)
    with np.load(windows, allow_pickle=False) as saved:
        relative_times = saved["relative_bin_centers_s"].copy()
        trial_split = saved["trial_split"].copy()
    units = fetch_unit_table(
        prepared["metadata"]["subject_name"],
        prepared["metadata"]["session_name"],
        unit_criteria_id=1,
        stability_param_id=0,
        include_metrics=False,
    )
    test_rows = np.repeat(trial_split == 2, len(relative_times))
    common = np.load(design, mmap_mode="r", allow_pickle=False)[test_rows]
    common_columns = int(design_metadata["base_columns"]) + selected_components
    valid = _valid_bin_mask(prepared)
    trial_count = int(np.count_nonzero(trial_split == 2))
    bin_count = len(relative_times)
    if len(common) != trial_count * bin_count:
        raise ValueError("Test design and response grid differ.")
    test_valid = valid[test_rows].reshape(trial_count, bin_count)
    partitions = cross_validation_partitions(len(trial_split), bin_count, valid)
    fold_by_id = {int(item["unit_id"]): item for item in fold_records}
    if len(fold_by_id) != len(fold_records) or set(fold_by_id) != {
        int(item["unit_id"]) for item in results
    }:
        raise ValueError("Final and fold results must contain the same unique units.")

    def unit_spike_times(selected_unit: int) -> np.ndarray:
        matching_units = units.loc[units["unit_id"] == selected_unit]
        if len(matching_units) != 1:
            raise ValueError(f"Expected one eligible row for unit {selected_unit}.")
        return np.asarray(matching_units.iloc[0]["spike_times_s"], dtype=float)

    example_spikes = unit_spike_times(EXAMPLE_UNIT_ID)
    _, _, kernel_history_scale = training_zscore(
        build_unit_history(prepared["alignments"], example_spikes),
        valid,
    )
    kernels = fitted_kernels(example_result, design_metadata, kernel_history_scale)
    kernel_pdf, kernel_png = plot_kernel_figure(
        kernels,
        task_names,
        figure_dir / "fitted_kernels",
    )
    for old_stem in ("fitted_task_kernels", "fitted_video_kernels"):
        for suffix in (".pdf", ".png"):
            (figure_dir / old_stem).with_suffix(suffix).unlink(missing_ok=True)

    example_counts = build_unit_counts(prepared["alignments"], example_spikes)
    example_history_raw = build_unit_history(prepared["alignments"], example_spikes)
    count_grid = example_counts.reshape(len(trial_split), bin_count)
    history_grid = example_history_raw.reshape(
        len(trial_split), bin_count, HISTORY_COLUMNS
    )
    valid_grid = valid.reshape(len(trial_split), bin_count)
    adjacent_valid = valid_grid[:, 1:] & valid_grid[:, :-1]
    if not np.array_equal(
        history_grid[:, 1:, 0][adjacent_valid],
        count_grid[:, :-1][adjacent_valid],
    ):
        raise ValueError("Spike history is not one bin behind spikes on valid rows.")
    example_history, _, _ = training_zscore(example_history_raw, valid)
    observed = example_counts[test_rows].reshape(trial_count, bin_count)

    def prediction_data() -> tuple[np.ndarray, np.ndarray]:
        conditional = np.full(len(common), np.nan)
        saved_folds = iter(fold_by_id[EXAMPLE_UNIT_ID]["folds"])
        for partition in partitions:
            if example_counts[partition["test"]].sum() <= 0:
                continue
            fold = next(saved_folds, None)
            if fold is None:
                raise ValueError("Saved fold count does not match fitted partitions.")
            coefficients = np.asarray(fold["coefficients"], dtype=float)
            if len(coefficients) != common_columns + HISTORY_COLUMNS:
                raise ValueError(
                    "Saved coefficient count does not match the selected design."
                )
            history, _, _ = training_zscore(example_history_raw, partition["fit"])
            plotted_rows = partition["test"][test_rows]
            linear_predictor = (
                float(fold["intercept"])
                + np.asarray(common[plotted_rows, :common_columns])
                @ coefficients[:common_columns]
                + history[test_rows][plotted_rows] @ coefficients[common_columns:]
            )
            conditional[plotted_rows] = np.exp(linear_predictor)
        if next(saved_folds, None) is not None:
            raise ValueError("Saved fold count does not match the fitted partitions.")
        plotted = test_valid.ravel()
        if not np.isfinite(conditional[plotted]).all() or np.any(
            conditional[plotted] <= 0
        ):
            raise ValueError("Conditional predictions must be finite and positive.")
        return (
            example_counts[test_rows].reshape(trial_count, bin_count),
            conditional.reshape(trial_count, bin_count),
        )

    prediction_observed, conditional = prediction_data()
    typical_trial = _select_count_typical_trial(np.where(test_valid, observed, 0))
    displayed_design = np.column_stack(
        (
            common[:, :common_columns].reshape(trial_count, bin_count, common_columns)[
                typical_trial
            ],
            example_history[test_rows].reshape(trial_count, bin_count, HISTORY_COLUMNS)[
                typical_trial
            ],
        )
    )
    test_trial_numbers = prepared["selected_trial_numbers"][trial_split == 2]
    selected_trial_row = int(np.flatnonzero(trial_split == 2)[typical_trial])
    displayed_rows = test_valid[typical_trial]
    if not displayed_rows.any() or np.any(np.diff(displayed_rows.astype(int)) > 0):
        raise ValueError("The displayed model interval must be one leading block.")
    model_times = relative_times[displayed_rows]
    model_counts = observed[typical_trial, displayed_rows]
    model_design = displayed_design[displayed_rows]
    contributions, linear_predictor, fitted_rate, contribution_error = (
        fitted_trial_contributions(model_design, example_result, design_metadata)
    )

    trials = _load_selected_trials(prepared)
    trial = trials.iloc[selected_trial_row]
    alignment = float(prepared["alignments"][selected_trial_row])
    flashes = np.asarray(trial["stim_pulse_times_s"], dtype=float)
    center_entry = float(trial["center_entry_s"])
    center_exit = float(trial["center_exit_s"])
    response_entry = float(trial["response_port_entry_s"])
    response_time = float(trial["response_port_entry_s"]) - alignment
    stationary_flashes, running_flashes = split_flash_events(
        trials.iloc[[selected_trial_row]]
    )
    modeled_flashes = flashes[(flashes >= center_entry) & (flashes <= response_entry)]
    if (
        len(modeled_flashes) == 0
        or len(stationary_flashes) + len(running_flashes) != len(modeled_flashes)
        or not np.isclose(modeled_flashes[0] - alignment, 0, atol=BINWIDTH_S)
    ):
        raise ValueError("Every displayed flash must have one movement condition.")
    events = {
        "stationary_flash": stationary_flashes - alignment,
        "running_flash": running_flashes - alignment,
        "center_poke": np.asarray([center_entry - alignment]),
        "center_exit": np.asarray([center_exit - alignment]),
        "response_entry": np.asarray([response_time]),
        "response_side": np.asarray([response_time]),
    }
    with np.load(
        windows.with_name("video_me_features.npz"), allow_pickle=False
    ) as data:
        frame_rows = data["frame_trial_row"] == selected_trial_row
        motion_times = data["frame_times_s"][frame_rows] - alignment
        motion_scores = data["scores"][frame_rows, :3]
    if (
        len(motion_times) < 2
        or motion_times[0] > model_times[0] - BINWIDTH_S / 2
        or motion_times[-1] < model_times[-1] + BINWIDTH_S / 2
    ):
        raise ValueError("Motion-PC samples do not span the displayed model interval.")
    design_pdf, design_png = plot_model_design(
        model_times,
        model_counts,
        motion_times,
        motion_scores,
        events,
        int(trial["response"]),
        contributions,
        linear_predictor,
        fitted_rate,
        design_metadata,
        figure_dir / "model_design",
    )
    for suffix in (".pdf", ".png"):
        (figure_dir / "design_matrix_trial").with_suffix(suffix).unlink(missing_ok=True)
    seed = PREDICTION_SEED + EXAMPLE_UNIT_ID
    predicted = np.zeros_like(prediction_observed, dtype=int)
    predicted[test_valid] = np.random.default_rng(seed).poisson(conditional[test_valid])
    pdf_path, png_path = plot_prediction_figure(
        relative_times,
        prediction_observed,
        predicted,
        conditional,
        test_valid,
        figure_dir / "predicted_spike_trains",
    )
    written = {
        "model_design": (design_pdf, design_png),
        "summary": (summary_pdf, summary_png),
        "fitted_kernels": (kernel_pdf, kernel_png),
        "predicted_spike_trains": (pdf_path, png_path),
    }
    print(
        json.dumps(
            {
                "example_unit_id": EXAMPLE_UNIT_ID,
                "example_unit_training_rate_spikes_s": example_training_rate,
                "example_unit_cross_validated_deviance_explained": example_result[
                    "cross_validated_deviance_explained"
                ],
                "test_trials": trial_count,
                "prediction_observed_spikes": int(
                    prediction_observed[test_valid].sum()
                ),
                "prediction_sampled_spikes": int(predicted[test_valid].sum()),
                "prediction_seed": seed,
                "training_rate_test_deviance_spearman_rho": rate_deviance_rho,
                "model_design_trial": int(test_trial_numbers[typical_trial]),
                "model_design_group_sum_max_abs_error": contribution_error,
                "model_design_stationary_flashes": len(stationary_flashes),
                "model_design_running_flashes": len(running_flashes),
                "model_design_valid_bins": int(displayed_rows.sum()),
                "figure_dir": str(figure_dir),
                "formats": list(FIGURE_FORMATS),
                "figures": {
                    name: [str(path) for path in paths if path is not None]
                    for name, paths in written.items()
                },
            },
            indent=2,
        )
    )
