"""Create figures for the fitted V1 spike-prediction GLM.

Scientific comparison
---------------------
For the previously held-out chronological test trials from GRB006 session
20240821_121447, show observed spikes beside one recursive simulation from the
complete sensory, task, audio, history, drift, and video model. Each simulated
row keeps that trial's external covariates fixed. Self-history is updated from
simulated spikes after seeding each trial with its observed pre-window history.

The final panel shows the one-step conditional mean used for held-out deviance
and bits/spike. Unlike the recursive rasters, that prediction conditions on the
observed spike history. Test trials are distinct trials aligned to their first
measured flash, not repeats of one identical stimulus. The displayed mean rates
use 20 ms Gaussian smoothing; all fitting and scoring remain at 1 ms.

The figure set also shows one actual held-out-trial design matrix and the fitted
temporal kernels. Population heatmaps contain every eligible unit in depth order.
Line plots show the same median-performance unit used for the spike-train figure.
Task kernels are changes in log expected rate for one event. Response side is
shown as right minus left, and eventual outcome as rewarded minus error. History
kernels are per preceding spike, and video kernels are per one-bin sample of a
training-standardized camera-PC score.
Heatmap color limits use the pooled 99th absolute percentile within each logical
regressor group; this affects color saturation only and does not remove units.

The example is selected by a fixed rule as the unit whose full-model test
deviance explained is nearest the population median. This post-fit display
choice does not change any fitted model or reported score. No sensory-response
or visual-quality filter is used. The unit is an observation from one session.
The simulations are offline conditional draws, not causal online forecasts,
because the design includes acausal video and peri-event regressors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from thesis.ephys.analyses.v1_glm import (
    HISTORY_COLUMNS,
    VIDEO_BASIS_COLUMNS,
    VIDEO_COMPONENT_COUNTS,
    _contiguous_slices,
    _load_windows,
    build_unit_design,
    spike_history_basis,
    task_temporal_bases,
    video_temporal_basis,
)
from thesis.ephys.preprocessing.prepare_v1_glm import BINWIDTH_S
from thesis.ephys.units import fetch_unit_table

OBSERVED_COLOR = "black"
MODEL_COLOR = "C0"
GROUP_COLORS = {
    "task": "C0",
    "video": "C1",
    "history": "C2",
    "drift": "black",
}
TASK_LABELS = {
    "visual_flash": "Visual flash",
    "center_entry": "Center entry",
    "go_cue_command": "Go cue",
    "center_exit": "Center exit",
    "response_entry": "Response entry",
    "response_side": "Response side\n(right − left)",
    "outcome": "Eventual outcome\n(rewarded − error)",
    "wrong_punishment_command": "Punishment cue",
}
SMOOTHING_MS = 20
SIMULATION_SEED = 2008
FIGURE_STYLE = {
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def select_representative_result(results: list[dict]) -> tuple[dict, float]:
    """Return the full-model result nearest the population median test score."""
    if not results:
        raise ValueError("No fitted unit results were found.")
    scores = np.asarray(
        [item["plus_video"]["test"]["deviance_explained"] for item in results],
        dtype=float,
    )
    if not np.isfinite(scores).all():
        raise ValueError("Full-model test scores must be finite.")
    median = float(np.median(scores))
    result = min(
        results,
        key=lambda item: (
            abs(item["plus_video"]["test"]["deviance_explained"] - median),
            int(item["unit_id"]),
        ),
    )
    return result, median


def training_rate_and_test_deviance(
    results: list[dict], selections: list[dict]
) -> tuple[np.ndarray, np.ndarray]:
    """Pair training firing rate and full-model test score by unit ID."""
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
        [item["plus_video"]["test"]["deviance_explained"] for item in results],
        dtype=float,
    )
    if not np.isfinite(training_rate).all() or np.any(training_rate < 0):
        raise ValueError("Training firing rates must be finite and nonnegative.")
    if not np.isfinite(test_deviance).all():
        raise ValueError("Full-model test scores must be finite.")
    return training_rate, test_deviance


def conditional_prediction(
    common: np.ndarray,
    history: np.ndarray,
    history_mean: np.ndarray,
    history_scale: np.ndarray,
    fitted_model: dict,
) -> np.ndarray:
    """Rebuild the saved one-step prediction that uses observed history."""
    coefficients = np.asarray(fitted_model["coefficients"], dtype=float)
    common_columns = len(coefficients) - HISTORY_COLUMNS
    if common_columns <= 0 or common.shape[1] < common_columns:
        raise ValueError("Saved coefficient count does not match the common design.")
    if history.shape != (common.shape[0], HISTORY_COLUMNS):
        raise ValueError("Common and history rows must match.")
    scaled_history = (history - history_mean) / history_scale
    linear_prediction = (
        float(fitted_model["intercept"])
        + common[:, :common_columns] @ coefficients[:common_columns]
        + scaled_history @ coefficients[common_columns:]
    )
    prediction = np.exp(linear_prediction)
    if not np.isfinite(prediction).all() or np.any(prediction <= 0):
        raise ValueError(
            "Reconstructed conditional predictions must be finite and positive."
        )
    return prediction


def simulation_history_kernel(
    fitted_model: dict, history_scale: np.ndarray
) -> np.ndarray:
    """Collapse scaled history coefficients into one 100-lag spike filter."""
    coefficients = np.asarray(fitted_model["coefficients"], dtype=float)
    history_coefficients = coefficients[-HISTORY_COLUMNS:] / history_scale
    return spike_history_basis().basis @ history_coefficients


def simulate_spike_counts(
    observed_counts: np.ndarray,
    conditional_mean: np.ndarray,
    history_kernel: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw spike counts while replacing observed with simulated self-history."""
    observed = np.asarray(observed_counts, dtype=float)
    conditional = np.asarray(conditional_mean, dtype=float)
    kernel = np.asarray(history_kernel, dtype=float)
    if observed.ndim != 2 or observed.shape != conditional.shape:
        raise ValueError(
            "Observed counts and conditional means must be trial-by-bin arrays."
        )
    if np.any(observed < 0) or not np.isfinite(observed).all():
        raise ValueError("Observed counts must be finite and nonnegative.")
    if np.any(conditional <= 0) or not np.isfinite(conditional).all():
        raise ValueError("Conditional means must be finite and positive.")
    if kernel.ndim != 1 or not np.isfinite(kernel).all():
        raise ValueError("The history kernel must be one finite vector.")

    simulated = np.zeros(observed.shape, dtype=np.int64)
    difference = np.zeros(observed.shape, dtype=float)
    log_conditional = np.log(conditional)
    for bin_index in range(observed.shape[1]):
        lag_count = min(bin_index, len(kernel))
        if lag_count:
            recent_difference = difference[:, bin_index - lag_count : bin_index]
            correction = recent_difference @ kernel[:lag_count][::-1]
        else:
            correction = 0.0
        with np.errstate(over="raise", invalid="raise"):
            try:
                mean = np.exp(log_conditional[:, bin_index] + correction)
            except FloatingPointError as error:
                raise RuntimeError(
                    "Recursive simulation became unstable; no rate clipping was applied."
                ) from error
        if not np.isfinite(mean).all():
            raise RuntimeError(
                "Recursive simulation became unstable; no rate clipping was applied."
            )
        try:
            simulated[:, bin_index] = rng.poisson(mean)
        except ValueError as error:
            raise RuntimeError(
                "Recursive simulation became unstable; no rate clipping was applied."
            ) from error
        difference[:, bin_index] = simulated[:, bin_index] - observed[:, bin_index]
    return simulated


def _raster_events(counts: np.ndarray, times: np.ndarray) -> list[np.ndarray]:
    return [np.repeat(times, row.astype(int, copy=False)) for row in np.asarray(counts)]


def _save_figure(figure, output: Path) -> tuple[Path, Path]:
    output.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output.with_suffix(".pdf")
    png_path = output.with_suffix(".png")
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return pdf_path, png_path


def plot_model_design(
    design_metadata: dict,
    selected_components: int,
    output: Path,
) -> tuple[Path, Path]:
    """Show the complete model structure and temporal support."""
    supports = [
        (
            TASK_LABELS[item["name"]].split("\n")[0],
            1000 * float(item["kernel_range_s"][0]),
            1000 * float(item["kernel_range_s"][1]),
            int(item["columns"]),
            GROUP_COLORS["task"],
        )
        for item in design_metadata["task_manifest"]
    ]
    supports.extend(
        [
            (
                "Video PCs",
                1000 * float(design_metadata["video_basis_range_s"][0]),
                1000 * float(design_metadata["video_basis_range_s"][1]),
                int(design_metadata["video_basis_columns_per_component"]),
                GROUP_COLORS["video"],
            ),
            ("Spike history", -100, -1, HISTORY_COLUMNS, GROUP_COLORS["history"]),
        ]
    )

    with plt.rc_context(FIGURE_STYLE):
        figure = plt.figure(figsize=(7.4, 4.8))
        grid = figure.add_gridspec(2, 1, height_ratios=[1.1, 2.2], hspace=0.55)

        schematic = figure.add_subplot(grid[0])
        schematic.set(xlim=(0, 1), ylim=(0, 1))
        schematic.axis("off")
        source_boxes = (
            (
                0.02,
                0.76,
                (
                    "sensory, task, audio\n"
                    f"{design_metadata['task_columns']} temporal columns"
                ),
                GROUP_COLORS["task"],
            ),
            (
                0.02,
                0.51,
                (
                    f"video-derived movement\n{selected_components} PCs × "
                    f"{design_metadata['video_basis_columns_per_component']} bases"
                ),
                GROUP_COLORS["video"],
            ),
            (
                0.02,
                0.26,
                f"spike history\n{HISTORY_COLUMNS} bases, 1–100 ms",
                GROUP_COLORS["history"],
            ),
            (0.02, 0.01, "session drift\nlinear + quadratic", GROUP_COLORS["drift"]),
        )
        for x, y, label, color in source_boxes:
            box = FancyBboxPatch(
                (x, y),
                0.27,
                0.17,
                boxstyle="round,pad=0.015",
                facecolor=color,
                edgecolor=color,
                alpha=0.2,
                linewidth=1,
            )
            schematic.add_patch(box)
            schematic.text(x + 0.135, y + 0.085, label, ha="center", va="center")
            schematic.annotate(
                "",
                xy=(0.42, 0.5),
                xytext=(x + 0.27, y + 0.085),
                arrowprops={"arrowstyle": "->", "color": "0.45", "linewidth": 0.8},
            )
        model_box = FancyBboxPatch(
            (0.42, 0.34),
            0.25,
            0.32,
            boxstyle="round,pad=0.02",
            facecolor="0.94",
            edgecolor="0.35",
            linewidth=1,
        )
        schematic.add_patch(model_box)
        schematic.text(
            0.545,
            0.5,
            "$\\log \\mu_t = \\beta_0 + X_t\\beta$\nL2-penalized fit",
            ha="center",
            va="center",
        )
        output_box = FancyBboxPatch(
            (0.78, 0.39),
            0.2,
            0.22,
            boxstyle="round,pad=0.02",
            facecolor=MODEL_COLOR,
            edgecolor=MODEL_COLOR,
            alpha=0.2,
            linewidth=1,
        )
        schematic.add_patch(output_box)
        schematic.text(
            0.88,
            0.5,
            "$y_t \\sim$ Poisson($\\mu_t$)\nspike count in 1 ms",
            ha="center",
            va="center",
        )
        schematic.annotate(
            "",
            xy=(0.78, 0.5),
            xytext=(0.67, 0.5),
            arrowprops={"arrowstyle": "->", "color": "0.35", "linewidth": 1},
        )
        schematic.text(
            -0.02,
            1.02,
            "a",
            transform=schematic.transAxes,
            fontweight="bold",
            fontsize=10,
            va="bottom",
        )

        support_axis = figure.add_subplot(grid[1])
        positions = np.arange(len(supports))
        for position, (label, start, stop, columns, color) in enumerate(supports):
            support_axis.plot(
                [start, stop],
                [position, position],
                color=color,
                linewidth=5,
                solid_capstyle="butt",
            )
            support_axis.text(
                320,
                position,
                f"{columns}" + (" / PC" if label == "Video PCs" else ""),
                va="center",
                ha="left",
                color=color,
            )
        support_axis.axvline(0, color="0.7", linewidth=0.8, zorder=0)
        support_axis.set_yticks(positions, [item[0] for item in supports])
        support_axis.invert_yaxis()
        support_axis.set_xlim(-330, 395)
        support_axis.set_xticks(np.arange(-300, 301, 100))
        support_axis.set_xlabel("Time relative to event or current bin (ms)")
        support_axis.text(
            320,
            -0.8,
            "Basis\nfunctions",
            ha="left",
            va="bottom",
            color="0.3",
            fontsize=7,
        )
        support_axis.text(
            -0.07,
            1.02,
            "b",
            transform=support_axis.transAxes,
            fontweight="bold",
            fontsize=10,
            va="bottom",
        )

        return _save_figure(figure, output)


def plot_population_summary(
    results: list[dict], selections: list[dict], output: Path
) -> tuple[Path, Path, float]:
    """Show camera selection, held-out performance, and its rate relation."""
    selected_components = {item["plus_video"]["components"] for item in results}
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
    full_results = [item["plus_video"]["test"] for item in results]
    training_rate, deviance = training_rate_and_test_deviance(results, selections)
    bits = np.asarray([item["bits_per_spike"] for item in full_results])
    rate_deviance_rho = float(spearmanr(training_rate, deviance).statistic)
    if not np.isfinite(rate_deviance_rho):
        raise ValueError("Training firing rate and test score must vary across units.")

    with plt.rc_context(FIGURE_STYLE):
        figure, axes = plt.subplots(2, 2, figsize=(7.4, 5.6))
        axes = axes.ravel()

        positions = np.arange(len(component_counts))
        axes[0].plot(positions, validation_deviance, color="black", marker="o")
        selected_position = component_counts.index(selected)
        axes[0].scatter(
            selected_position,
            validation_deviance[selected_position],
            color="black",
            s=75,
            zorder=3,
        )
        axes[0].text(
            selected_position,
            validation_deviance[selected_position],
            "selected",
            color="black",
            ha="center",
            va="bottom",
        )
        axes[0].set_xticks(positions, component_counts)
        axes[0].set_xlabel("Candidate camera PCs")
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


def fitted_kernel_matrices(
    results: list[dict], selections: list[dict], design_metadata: dict
) -> tuple[list[int], dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Reconstruct fitted filters in the units of the unscaled regressors."""
    ordered = sorted(
        results, key=lambda item: (float(item["depth"]), int(item["unit_id"]))
    )
    selection_by_id = {int(item["unit_id"]): item for item in selections}
    unit_ids = [int(item["unit_id"]) for item in ordered]
    if len(selection_by_id) != len(selections) or set(unit_ids) != set(selection_by_id):
        raise ValueError("Final and validation results contain different units.")

    component_counts = {int(item["plus_video"]["components"]) for item in ordered}
    if len(component_counts) != 1:
        raise ValueError("All final fits must use one camera-PC count.")
    video_components = component_counts.pop()
    if video_components < 3:
        raise ValueError("At least three camera PCs are needed for the kernel figure.")
    common_columns = (
        int(design_metadata["base_columns"]) + VIDEO_BASIS_COLUMNS * video_components
    )
    coefficients = np.asarray(
        [item["plus_video"]["coefficients"] for item in ordered], dtype=float
    )
    if coefficients.shape != (len(ordered), common_columns + HISTORY_COLUMNS):
        raise ValueError("Saved coefficient count does not match the selected design.")

    common_scale = np.asarray(design_metadata["training_scale"], dtype=float)[
        :common_columns
    ]
    if common_scale.shape != (common_columns,) or np.any(common_scale <= 0):
        raise ValueError("Common-design scales must be positive and complete.")
    common_weights = coefficients[:, :common_columns] / common_scale

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
            common_weights[:, cursor:stop] @ basis.basis.T,
        )
        cursor = stop
    if cursor != int(design_metadata["task_columns"]):
        raise ValueError("Task manifest and saved task-column count differ.")

    video_basis = video_temporal_basis()
    video_start = int(design_metadata["base_columns"])
    for component in range(3):
        start = video_start + component * video_basis.basis.shape[1]
        stop = start + video_basis.basis.shape[1]
        kernels[f"video_pc_{component + 1}"] = (
            1000 * video_basis.basis_time,
            common_weights[:, start:stop] @ video_basis.basis.T,
        )

    history_scale = np.asarray(
        [selection_by_id[unit_id]["history_training_scale"] for unit_id in unit_ids],
        dtype=float,
    )
    if history_scale.shape != (len(ordered), HISTORY_COLUMNS) or np.any(
        history_scale <= 0
    ):
        raise ValueError("History scales must be positive and complete.")
    history_weights = coefficients[:, -HISTORY_COLUMNS:] / history_scale
    history_basis = spike_history_basis()
    kernels["self_history"] = (
        1000 * BINWIDTH_S * np.arange(1, len(history_basis.basis) + 1),
        history_weights @ history_basis.basis.T,
    )
    if not all(np.isfinite(values).all() for _, values in kernels.values()):
        raise ValueError("Reconstructed kernels must be finite.")
    return unit_ids, kernels


def _kernel_color_limit(matrices: list[np.ndarray]) -> float:
    absolute = np.concatenate([np.abs(matrix).ravel() for matrix in matrices])
    limit = float(np.quantile(absolute, 0.99))
    if not np.isfinite(limit) or limit <= 0:
        raise ValueError("Kernel values must have a positive finite color range.")
    return limit


def _plot_kernel_heatmap(axis, times_ms: np.ndarray, values: np.ndarray, limit: float):
    image = axis.imshow(
        values,
        aspect="auto",
        interpolation="nearest",
        extent=(times_ms[0], times_ms[-1], len(values), 0),
        vmin=-limit,
        vmax=limit,
    )
    if times_ms[0] <= 0 <= times_ms[-1]:
        axis.axvline(0, color="black", linewidth=0.45)
    axis.set_yticks([])
    return image


def task_kernel_display_values(name: str, values: np.ndarray) -> np.ndarray:
    """Convert signed-code filters to the displayed condition difference."""
    values = np.asarray(values)
    return 2 * values if name in {"response_side", "outcome"} else values


def plot_task_kernels(
    unit_ids: list[int],
    kernels: dict[str, tuple[np.ndarray, np.ndarray]],
    representative_unit_id: int,
    task_names: list[str],
    output: Path,
) -> tuple[Path, Path]:
    """Show population task-filter heatmaps and one representative unit."""
    if len(task_names) != 8:
        raise ValueError("The task-kernel layout expects eight regressors.")
    example_row = unit_ids.index(representative_unit_id)
    displayed = {
        name: (times, task_kernel_display_values(name, values))
        for name, (times, values) in kernels.items()
        if name in task_names
    }
    limit = _kernel_color_limit([displayed[name][1] for name in task_names])

    with plt.rc_context(FIGURE_STYLE):
        figure = plt.figure(figsize=(7.5, 7.8))
        outer = figure.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.7)
        population_grid = outer[0].subgridspec(
            2, 5, width_ratios=[1, 1, 1, 1, 0.055], hspace=0.88, wspace=0.3
        )
        example_grid = outer[1].subgridspec(2, 4, hspace=0.88, wspace=0.3)
        images = []
        for index, name in enumerate(task_names):
            row, column = divmod(index, 4)
            axis = figure.add_subplot(population_grid[row, column])
            times_ms, values = displayed[name]
            images.append(_plot_kernel_heatmap(axis, times_ms, values, limit))
            axis.set_title(TASK_LABELS[name], fontsize=8)
            if column == 0:
                axis.set_ylabel("Units\n(depth order)")
        colorbar_axis = figure.add_subplot(population_grid[:, 4])
        figure.colorbar(
            images[0],
            cax=colorbar_axis,
            label="$\Delta$ log expected rate\n(99th-percentile color limit)",
        )

        for index, name in enumerate(task_names):
            row, column = divmod(index, 4)
            axis = figure.add_subplot(example_grid[row, column])
            times_ms, values = displayed[name]
            axis.plot(times_ms, values[example_row], color="black", linewidth=1)
            axis.axhline(0, color="black", linestyle="--", linewidth=0.5)
            if times_ms[0] <= 0 <= times_ms[-1]:
                axis.axvline(0, color="black", linewidth=0.45)
            axis.set_title(TASK_LABELS[name], fontsize=8)
            if column == 0:
                axis.set_ylabel("$\Delta$ log rate")

        figure.text(0.01, 0.985, "a", fontweight="bold", fontsize=10, va="top")
        figure.text(0.04, 0.985, "All fitted V1 units, sorted by depth", va="top")
        figure.text(0.5, 0.515, "Time from event (ms)", ha="center", va="top")
        figure.text(0.01, 0.475, "b", fontweight="bold", fontsize=10, va="top")
        figure.text(
            0.04,
            0.475,
            f"Median-performance unit {representative_unit_id}",
            va="top",
        )
        figure.text(0.5, 0.005, "Time from event (ms)", ha="center", va="bottom")
        return _save_figure(figure, output)


def plot_history_video_kernels(
    unit_ids: list[int],
    kernels: dict[str, tuple[np.ndarray, np.ndarray]],
    representative_unit_id: int,
    output: Path,
) -> tuple[Path, Path]:
    """Show population and representative history and camera-PC filters."""
    names = ["self_history", "video_pc_1", "video_pc_2", "video_pc_3"]
    titles = ["Spike history", "Video PC 1", "Video PC 2", "Video PC 3"]
    example_row = unit_ids.index(representative_unit_id)
    history_limit = _kernel_color_limit([kernels["self_history"][1]])
    video_limit = _kernel_color_limit([kernels[name][1] for name in names[1:]])

    with plt.rc_context(FIGURE_STYLE):
        figure = plt.figure(figsize=(7.5, 4.0))
        grid = figure.add_gridspec(
            2,
            6,
            width_ratios=[1, 0.06, 1, 1, 1, 0.06],
            height_ratios=[1, 1],
            hspace=0.55,
            wspace=0.42,
        )
        positions = [0, 2, 3, 4]
        images = []
        for index, (name, title, column) in enumerate(
            zip(names, titles, positions, strict=True)
        ):
            axis = figure.add_subplot(grid[0, column])
            times_ms, values = kernels[name]
            limit = history_limit if index == 0 else video_limit
            images.append(_plot_kernel_heatmap(axis, times_ms, values, limit))
            axis.set_title(title)
            axis.set_xlabel("Lag (ms)")
            if index == 0:
                axis.set_ylabel("Units\n(depth order)")
        history_colorbar = figure.add_subplot(grid[0, 1])
        figure.colorbar(images[0], cax=history_colorbar)
        history_colorbar.set_title("$\Delta$ log\nrate", fontsize=7, pad=3)
        video_colorbar = figure.add_subplot(grid[0, 5])
        figure.colorbar(images[1], cax=video_colorbar)
        video_colorbar.set_title("$\Delta$ log\nrate", fontsize=7, pad=3)

        for index, (name, title, column) in enumerate(
            zip(names, titles, positions, strict=True)
        ):
            axis = figure.add_subplot(grid[1, column])
            times_ms, values = kernels[name]
            axis.plot(times_ms, values[example_row], color="black", linewidth=1)
            axis.axhline(0, color="black", linestyle="--", linewidth=0.5)
            if times_ms[0] <= 0 <= times_ms[-1]:
                axis.axvline(0, color="black", linewidth=0.45)
            axis.set_title(title)
            axis.set_xlabel("Lag (ms)")
            if index == 0:
                axis.set_ylabel("$\Delta$ log rate / spike")
            elif index == 1:
                axis.set_ylabel("$\Delta$ log rate / PC sample", fontsize=7)
        for column in (1, 5):
            figure.add_subplot(grid[1, column]).axis("off")

        figure.text(0.01, 0.985, "a", fontweight="bold", fontsize=10, va="top")
        figure.text(0.04, 0.985, "All fitted V1 units, sorted by depth", va="top")
        figure.text(0.01, 0.49, "b", fontweight="bold", fontsize=10, va="top")
        figure.text(
            0.04,
            0.49,
            f"Median-performance unit {representative_unit_id}",
            va="top",
        )
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


def plot_design_matrix_trial(
    relative_times: np.ndarray,
    counts: np.ndarray,
    design: np.ndarray,
    design_metadata: dict,
    selected_components: int,
    unit_id: int,
    trial_number: int,
    output: Path,
) -> tuple[Path, Path]:
    """Show the response vector and exact standardized matrix for one test trial."""
    expected_columns = (
        int(design_metadata["base_columns"])
        + VIDEO_BASIS_COLUMNS * selected_components
        + HISTORY_COLUMNS
    )
    if design.shape != (len(relative_times), expected_columns):
        raise ValueError("Displayed trial design has the wrong shape.")
    if counts.shape != relative_times.shape:
        raise ValueError("Displayed counts and times must match.")
    if not np.isfinite(design).all() or not np.isfinite(counts).all():
        raise ValueError("Displayed trial values must be finite.")

    task_stop = int(design_metadata["task_columns"])
    drift_stop = int(design_metadata["base_columns"])
    video_stop = drift_stop + VIDEO_BASIS_COLUMNS * selected_components
    stops = [0, task_stop, drift_stop, video_stop, expected_columns]
    labels = [
        f"Task ({task_stop})",
        f"Drift ({drift_stop - task_stop})",
        f"Video PCs ({video_stop - drift_stop})",
        f"Spike history ({HISTORY_COLUMNS})",
    ]
    mids = [(start + stop) / 2 for start, stop in zip(stops[:-1], stops[1:])]
    maximum = float(np.max(np.abs(design)))
    if maximum <= 0:
        raise ValueError("Displayed design matrix cannot be all zero.")
    edges = (
        relative_times[0] - BINWIDTH_S / 2,
        relative_times[-1] + BINWIDTH_S / 2,
    )

    with plt.rc_context(FIGURE_STYLE):
        figure = plt.figure(figsize=(7.5, 4.8))
        grid = figure.add_gridspec(
            2,
            2,
            height_ratios=[0.9, 3.1],
            width_ratios=[1, 0.025],
            hspace=0.14,
            wspace=0.08,
        )
        count_axis = figure.add_subplot(grid[0, 0])
        matrix_axis = figure.add_subplot(grid[1, 0], sharex=count_axis)
        colorbar_axis = figure.add_subplot(grid[1, 1])
        figure.add_subplot(grid[0, 1]).axis("off")

        count_axis.step(
            relative_times, counts, where="mid", color="black", linewidth=0.7
        )
        count_axis.set_ylabel("Spike count\nper 1 ms")
        count_axis.set_ylim(-0.05, max(1, int(counts.max())) + 0.25)
        count_axis.set_yticks(range(max(1, int(counts.max())) + 1))
        count_axis.tick_params(axis="x", labelbottom=False)
        count_axis.text(
            0.99,
            1.02,
            f"held-out trial {trial_number}; median-performance unit {unit_id}",
            transform=count_axis.transAxes,
            ha="right",
            va="bottom",
        )

        image = matrix_axis.imshow(
            design.T,
            aspect="auto",
            interpolation="nearest",
            extent=(*edges, expected_columns, 0),
            vmin=-maximum,
            vmax=maximum,
        )
        for stop in stops[1:-1]:
            matrix_axis.axhline(stop, color="black", linewidth=0.6)
        matrix_axis.set_yticks(mids, labels)
        matrix_axis.set_xlabel("Time from first measured flash (s)")
        matrix_axis.set_ylabel("Model columns")
        figure.colorbar(
            image,
            cax=colorbar_axis,
            label="Model input (standard deviations from training mean)",
        )
        matrix_axis.set_xlim(*edges)

        for letter, axis in zip("ab", (count_axis, matrix_axis), strict=True):
            axis.text(
                -0.12,
                1.03,
                letter,
                transform=axis.transAxes,
                fontweight="bold",
                fontsize=10,
                va="bottom",
            )
        return _save_figure(figure, output)


def plot_prediction_figure(
    relative_times: np.ndarray,
    observed: np.ndarray,
    simulation: np.ndarray,
    prediction: np.ndarray,
    result: dict,
    population_median: float,
    output: Path,
) -> tuple[Path, Path]:
    """Write the Pillow-style raster and conditional-rate comparison."""
    raster_data = (observed, simulation)
    raster_labels = (
        "observed held-out spikes",
        "model-predicted spike trains",
    )
    raster_colors = (OBSERVED_COLOR, MODEL_COLOR)
    trial_count = observed.shape[0]

    with plt.rc_context(FIGURE_STYLE):
        figure, axes = plt.subplots(
            3,
            1,
            figsize=(7.1, 6.0),
            sharex=True,
            gridspec_kw={"height_ratios": [1, 1, 1.35], "hspace": 0.28},
        )
        for axis, counts, label, color in zip(
            axes[:2], raster_data, raster_labels, raster_colors, strict=True
        ):
            axis.eventplot(
                _raster_events(counts, relative_times),
                colors=color,
                lineoffsets=np.arange(1, trial_count + 1),
                linelengths=0.8,
                linewidths=0.35,
            )
            axis.set_ylim(0.5, trial_count + 0.5)
            axis.set_yticks([1, (trial_count + 1) // 2, trial_count])
            axis.axvline(0, color="0.75", linewidth=0.8, zorder=0)
            axis.text(
                0.01,
                1.02,
                label,
                color=color,
                transform=axis.transAxes,
                ha="left",
                va="bottom",
                fontweight="bold",
            )
        axes[1].set_ylabel("Test trials")
        axes[0].text(
            0.99,
            1.02,
            (
                f"median-performance unit {int(result['unit_id'])}: "
                f"test $D^2$={result['plus_video']['test']['deviance_explained']:.3f}; "
                f"population median={population_median:.3f}"
            ),
            transform=axes[0].transAxes,
            ha="right",
            va="bottom",
            color="0.3",
            fontsize=7,
        )

        sigma_bins = SMOOTHING_MS / (BINWIDTH_S * 1000)
        rates = [
            gaussian_filter1d(values.mean(axis=0) / BINWIDTH_S, sigma_bins)
            for values in (observed, prediction)
        ]
        rate_axis = axes[2]
        for rate, color in zip(rates, raster_colors, strict=True):
            rate_axis.plot(relative_times, rate, color=color, linewidth=1.25)
        rate_axis.axvline(0, color="0.75", linewidth=0.8, zorder=0)
        rate_axis.set_ylabel("Mean rate (spikes/s)")
        rate_axis.set_xlabel("Time from first measured flash (s)")
        rate_axis.text(
            0.01,
            0.94,
            "one-step prediction (uses observed recent spikes)",
            transform=rate_axis.transAxes,
            ha="left",
            va="top",
            color="0.3",
            fontsize=7,
        )
        for y, label, color in (
            (0.86, "observed", OBSERVED_COLOR),
            (0.77, "complete model", MODEL_COLOR),
        ):
            rate_axis.text(
                0.99,
                y,
                label,
                color=color,
                transform=rate_axis.transAxes,
                ha="right",
                va="top",
                fontweight="bold",
            )
        rate_axis.text(
            0.99,
            0.03,
            f"{SMOOTHING_MS} ms Gaussian smoothing for display only",
            transform=rate_axis.transAxes,
            ha="right",
            va="bottom",
            color="0.4",
            fontsize=7,
        )

        edges = (
            relative_times[0] - BINWIDTH_S / 2,
            relative_times[-1] + BINWIDTH_S / 2,
        )
        axes[-1].set_xlim(*edges)
        for letter, axis in zip("abc", axes, strict=True):
            axis.text(
                -0.075,
                1.02,
                letter,
                transform=axis.transAxes,
                ha="left",
                va="bottom",
                fontweight="bold",
                fontsize=10,
            )
        figure.align_ylabels(axes)
        return _save_figure(figure, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--windows",
        type=Path,
        default=Path("figures/v1_glm/stimulus_windows.npz"),
    )
    parser.add_argument(
        "--design", type=Path, default=Path("figures/v1_glm/common_design.npy")
    )
    parser.add_argument("--fit-dir", type=Path, default=Path("figures/v1_glm/all_fit"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/v1_glm/all_fit/predicted_spike_trains"),
    )
    args = parser.parse_args()

    results = []
    for path in sorted(args.fit_dir.glob("unit_*_test.json")):
        with path.open() as handle:
            results.append(json.load(handle))
    selections = []
    for path in sorted(args.fit_dir.glob("unit_*_validation.json")):
        with path.open() as handle:
            selections.append(json.load(handle))
    if len(results) != len(selections):
        raise ValueError("Final and validation result counts differ.")
    selected_components = {item["plus_video"]["components"] for item in results}
    if len(selected_components) != 1:
        raise ValueError("All final fits must use one camera-PC count.")
    selected_components = int(selected_components.pop())
    with args.design.with_suffix(".json").open() as handle:
        design_metadata = json.load(handle)
    design_pdf, design_png = plot_model_design(
        design_metadata,
        selected_components,
        args.fit_dir / "model_design",
    )
    summary_pdf, summary_png, rate_deviance_rho = plot_population_summary(
        results, selections, args.fit_dir / "summary"
    )
    result, population_median = select_representative_result(results)
    unit_id = int(result["unit_id"])
    selection = next(item for item in selections if int(item["unit_id"]) == unit_id)
    ordered_unit_ids, kernels = fitted_kernel_matrices(
        results, selections, design_metadata
    )
    task_kernel_pdf, task_kernel_png = plot_task_kernels(
        ordered_unit_ids,
        kernels,
        unit_id,
        [item["name"] for item in design_metadata["task_manifest"]],
        args.fit_dir / "fitted_task_kernels",
    )
    history_video_kernel_pdf, history_video_kernel_png = plot_history_video_kernels(
        ordered_unit_ids,
        kernels,
        unit_id,
        args.fit_dir / "fitted_history_video_kernels",
    )

    prepared = _load_windows(args.windows)
    _, _, test_rows = _contiguous_slices(prepared["split"])
    with np.load(args.windows, allow_pickle=False) as windows:
        relative_times = windows["relative_bin_centers_s"].copy()
        trial_split = windows["trial_split"].copy()
    test_alignments = prepared["alignments"][trial_split == 2]
    units = fetch_unit_table(
        prepared["metadata"]["subject_name"],
        prepared["metadata"]["session_name"],
        unit_criteria_id=1,
        stability_param_id=0,
        include_metrics=False,
    )
    matching_units = units.loc[units["unit_id"] == unit_id]
    if len(matching_units) != 1:
        raise ValueError(f"Expected one eligible row for unit {unit_id}.")
    spike_times = np.asarray(matching_units.iloc[0]["spike_times_s"], dtype=float)
    counts, history = build_unit_design(test_alignments, spike_times)

    common = np.load(args.design, mmap_mode="r", allow_pickle=False)[test_rows]
    if len(common) != len(counts):
        raise ValueError("Test design and response rows differ.")
    history_mean = np.asarray(selection["history_training_mean"], dtype=float)
    history_scale = np.asarray(selection["history_training_scale"], dtype=float)
    prediction = conditional_prediction(
        common, history, history_mean, history_scale, result["plus_video"]
    )
    saved = result["plus_video"]["test"]
    if not np.isclose(prediction.sum(), saved["predicted_spikes"], rtol=1e-6):
        raise ValueError("Reconstructed predictions differ from the saved fit.")

    trial_count = len(test_alignments)
    bin_count = len(relative_times)
    observed = counts.reshape(trial_count, bin_count)
    conditional = prediction.reshape(trial_count, bin_count)
    history_scaled = (history - history_mean) / history_scale
    common_columns = len(result["plus_video"]["coefficients"]) - HISTORY_COLUMNS
    typical_trial = _select_count_typical_trial(observed)
    common_trial = common[:, :common_columns].reshape(
        trial_count, bin_count, common_columns
    )[typical_trial]
    history_trial = history_scaled.reshape(trial_count, bin_count, HISTORY_COLUMNS)[
        typical_trial
    ]
    displayed_design = np.column_stack((common_trial, history_trial))
    test_trial_numbers = prepared["selected_trial_numbers"][trial_split == 2]
    design_matrix_pdf, design_matrix_png = plot_design_matrix_trial(
        relative_times,
        observed[typical_trial],
        displayed_design,
        design_metadata,
        selected_components,
        unit_id,
        int(test_trial_numbers[typical_trial]),
        args.fit_dir / "design_matrix_trial",
    )
    simulation = simulate_spike_counts(
        observed,
        conditional,
        simulation_history_kernel(result["plus_video"], history_scale),
        np.random.default_rng(SIMULATION_SEED),
    )
    pdf_path, png_path = plot_prediction_figure(
        relative_times,
        observed,
        simulation,
        conditional,
        result,
        population_median,
        args.output,
    )
    print(
        json.dumps(
            {
                "unit_id": unit_id,
                "selection": "full-model test deviance explained nearest population median",
                "population_median_test_deviance_explained": population_median,
                "unit_test_deviance_explained": result["plus_video"]["test"][
                    "deviance_explained"
                ],
                "test_trials": trial_count,
                "observed_spikes": int(observed.sum()),
                "simulated_spikes": int(simulation.sum()),
                "simulation_seed": SIMULATION_SEED,
                "model_design_pdf": str(design_pdf),
                "model_design_png": str(design_png),
                "summary_pdf": str(summary_pdf),
                "summary_png": str(summary_png),
                "training_rate_test_deviance_spearman_rho": rate_deviance_rho,
                "design_matrix_trial": int(test_trial_numbers[typical_trial]),
                "design_matrix_pdf": str(design_matrix_pdf),
                "design_matrix_png": str(design_matrix_png),
                "task_kernel_pdf": str(task_kernel_pdf),
                "task_kernel_png": str(task_kernel_png),
                "history_video_kernel_pdf": str(history_video_kernel_pdf),
                "history_video_kernel_png": str(history_video_kernel_png),
                "pdf": str(pdf_path),
                "png": str(png_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
