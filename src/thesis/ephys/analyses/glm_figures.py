"""Create figures for the fitted V1 spike-prediction GLM.

Held-out test trials, first-flash aligned. Rasters are recursive samples from
the fitted model: each simulated spike feeds back through that unit's own
history filter, so the raster is a draw from the model's generative process
rather than a one-step-ahead prediction. Each trial starts with an empty
history.
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
    VIDEO_BASIS_COLUMNS,
    VIDEO_COMPONENT_COUNTS,
    _load_windows,
    _split_masks,
    _valid_bin_mask,
    build_unit_counts,
    build_unit_history,
    spike_history_basis,
    task_temporal_bases,
    training_zscore,
    video_temporal_basis,
)
from thesis.ephys.preprocessing.prepare_glm import BINWIDTH_S
from thesis.ephys.units import fetch_unit_table

OBSERVED_COLOR = "black"
MODEL_COLOR = "C0"
GROUP_COLORS = {
    "task": "C0",
    "video": "C1",
}
TASK_LABELS = {
    "visual_flash": "Visual flash",
    "center_poke": "Center poke",
    "center_exit": "Center exit",
    "response_entry": "Response entry",
    "response_side": "Response side\n(right − left)",
}
SMOOTHING_MS = 20
SIMULATION_SEED = 2008
# Rebound by make_figures; read by _save_figure so the six plot functions keep
# their signatures.
FIGURE_FORMATS: tuple[str, ...] = ("pdf", "png")
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
        [item["cross_validated_deviance_explained"] for item in results],
        dtype=float,
    )
    if not np.isfinite(scores).all():
        raise ValueError("Cross-validated deviance must be finite.")
    median = float(np.median(scores))
    result = min(
        results,
        key=lambda item: (
            abs(item["cross_validated_deviance_explained"] - median),
            int(item["unit_id"]),
        ),
    )
    return result, median


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


def history_filter(
    coefficients: np.ndarray, scale: np.ndarray, mean: np.ndarray
) -> tuple[np.ndarray, float]:
    """Collapse the fitted history bases into one filter over past bins.

    The design columns were standardized, so undo that here: the filter carries
    the scaled weights and the centring becomes a constant added to the
    intercept. Entry ``lag`` multiplies the spike count ``lag + 1`` bins back,
    matching the one-bin shift used when the columns were built.
    """
    basis = np.asarray(spike_history_basis().basis, dtype=float)
    if basis.shape[1] != len(coefficients):
        raise ValueError("History coefficients and basis widths differ.")
    weights = np.asarray(coefficients, dtype=float) / np.asarray(scale, dtype=float)
    return basis @ weights, float(-np.sum(weights * np.asarray(mean, dtype=float)))


def simulate_spike_counts(
    base_eta: np.ndarray,
    filter_taps: np.ndarray,
    offset: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample spike trains recursively, feeding each spike back through history.

    ``base_eta`` holds the linear predictor from every non-history column, one
    row per trial. Each trial starts with no history, so the first bins are
    driven by the covariates alone.
    """
    base_eta = np.asarray(base_eta, dtype=float)
    if base_eta.ndim != 2:
        raise ValueError("Base linear predictor must be a trial-by-bin array.")
    if not np.isfinite(base_eta).all():
        raise ValueError("Base linear predictor must be finite.")
    taps = np.asarray(filter_taps, dtype=float)
    n_trials, n_bins = base_eta.shape
    counts = np.zeros((n_trials, n_bins), dtype=np.int64)
    for trial in range(n_trials):
        recent = np.zeros(len(taps))
        for step in range(n_bins):
            rate = np.exp(base_eta[trial, step] + offset + float(recent @ taps))
            draw = rng.poisson(min(rate, 1e6))
            counts[trial, step] = draw
            recent[1:] = recent[:-1]
            recent[0] = draw
    return counts


def _raster_events(counts: np.ndarray, times: np.ndarray) -> list[np.ndarray]:
    return [np.repeat(times, row.astype(int, copy=False)) for row in np.asarray(counts)]


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


def plot_model_design(
    design_metadata: dict, output: Path
) -> tuple[Path | None, Path | None]:
    """Show the temporal support and basis count of every design block."""
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
                "Motion-energy PCs",
                1000 * float(design_metadata["video_basis_range_s"][0]),
                1000 * float(design_metadata["video_basis_range_s"][1]),
                int(design_metadata["video_basis_columns_per_component"]),
                GROUP_COLORS["video"],
            ),
        ]
    )

    with plt.rc_context(FIGURE_STYLE):
        figure, support_axis = plt.subplots(figsize=(7.4, 3.0))
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
                f"{columns}" + (" / PC" if "Motion-energy" in label else ""),
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

    component_counts = {int(item["components"]) for item in ordered}
    if len(component_counts) != 1:
        raise ValueError("All final fits must use one camera-PC count.")
    video_components = component_counts.pop()
    if video_components < 3:
        raise ValueError("At least three camera PCs are needed for the kernel figure.")
    common_columns = (
        int(design_metadata["base_columns"]) + VIDEO_BASIS_COLUMNS * video_components
    )
    coefficients = np.asarray([item["coefficients"] for item in ordered], dtype=float)
    if coefficients.shape != (len(ordered), common_columns):
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
    return 2 * values if name == "response_side" else values


def plot_kernel_figure(
    unit_ids: list[int],
    kernels: dict[str, tuple[np.ndarray, np.ndarray]],
    representative_unit_id: int,
    names: list[str],
    titles: list[str],
    output: Path,
    columns: int,
    xlabel: str,
    example_ylabel: str,
) -> tuple[Path | None, Path | None]:
    """Show population filter heatmaps above the same filters for one unit."""
    example_row = unit_ids.index(representative_unit_id)
    displayed = {
        name: (kernels[name][0], task_kernel_display_values(name, kernels[name][1]))
        for name in names
    }
    limit = _kernel_color_limit([displayed[name][1] for name in names])
    rows = -(-len(names) // columns)

    with plt.rc_context(FIGURE_STYLE):
        figure = plt.figure(figsize=(7.5, 2.0 + 1.9 * 2 * rows))
        outer = figure.add_gridspec(2, 1, hspace=0.7)
        images = []
        for half in (0, 1):
            grid = outer[half].subgridspec(
                rows,
                columns + 1,
                width_ratios=[1] * columns + [0.06],
                wspace=0.3,
                hspace=0.9,
            )
            for index, (name, title) in enumerate(zip(names, titles, strict=True)):
                row, column = divmod(index, columns)
                axis = figure.add_subplot(grid[row, column])
                times_ms, values = displayed[name]
                if half == 0:
                    images.append(_plot_kernel_heatmap(axis, times_ms, values, limit))
                    if column == 0:
                        axis.set_ylabel("Units\n(depth order)")
                else:
                    axis.plot(times_ms, values[example_row], color="black", linewidth=1)
                    axis.axhline(0, color="black", linestyle="--", linewidth=0.5)
                    if times_ms[0] <= 0 <= times_ms[-1]:
                        axis.axvline(0, color="black", linewidth=0.45)
                    if column == 0:
                        axis.set_ylabel(example_ylabel, fontsize=7)
                axis.set_title(title, fontsize=8)
                axis.set_xlabel(xlabel)
            legend_cell = figure.add_subplot(grid[:, columns])
            if half == 0:
                figure.colorbar(
                    images[0],
                    cax=legend_cell,
                    label="$\Delta$ log expected rate\n(99th-percentile color limit)",
                )
            else:
                legend_cell.axis("off")

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
) -> tuple[Path | None, Path | None]:
    """Show the response vector and exact standardized matrix for one test trial."""
    expected_columns = int(design_metadata["base_columns"]) + (
        VIDEO_BASIS_COLUMNS * selected_components
    )
    if design.shape != (len(relative_times), expected_columns):
        raise ValueError("Displayed trial design has the wrong shape.")
    if counts.shape != relative_times.shape:
        raise ValueError("Displayed counts and times must match.")
    if not np.isfinite(design).all() or not np.isfinite(counts).all():
        raise ValueError("Displayed trial values must be finite.")

    task_stop = int(design_metadata["task_columns"])
    video_stop = expected_columns
    stops = [0, task_stop, video_stop]
    labels = [
        f"Task ({task_stop})",
        f"Motion energy ({video_stop - task_stop})",
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


def _interspike_intervals(counts: np.ndarray) -> np.ndarray:
    """Return within-trial interspike intervals in milliseconds."""
    intervals = []
    for row in np.asarray(counts):
        bins = np.repeat(np.arange(len(row)), row.astype(int, copy=False))
        if len(bins) > 1:
            intervals.append(np.diff(bins))
    if not intervals:
        return np.empty(0)
    return np.concatenate(intervals) * BINWIDTH_S * 1000


def plot_generative_checks(
    observed: np.ndarray, simulation: np.ndarray, unit_id: int, output: Path
) -> tuple[Path | None, Path | None]:
    """Compare spiking statistics the model was never fitted to reproduce.

    Held-out trials. The model matched the mean rate by construction, so these
    two panels are the informative ones: the interspike intervals show whether
    the history filter captured refractoriness, and the per-trial counts show
    whether the variability is right rather than only the mean.
    """
    observed_isi, simulated_isi = (
        _interspike_intervals(observed),
        _interspike_intervals(simulation),
    )
    observed_counts = np.asarray(observed).sum(axis=1)
    simulated_counts = np.asarray(simulation).sum(axis=1)

    with plt.rc_context(FIGURE_STYLE):
        figure, axes = plt.subplots(1, 2, figsize=(7.1, 2.9))
        if len(observed_isi) and len(simulated_isi):
            edges = np.logspace(0, np.log10(max(observed_isi.max(), 2)), 40)
            for values, color, label in (
                (observed_isi, OBSERVED_COLOR, "observed"),
                (simulated_isi, MODEL_COLOR, "model"),
            ):
                axes[0].hist(
                    values,
                    bins=edges,
                    density=True,
                    histtype="step",
                    color=color,
                    linewidth=1.2,
                    label=label,
                )
            axes[0].set_xscale("log")
        axes[0].set_xlabel("Interspike interval (ms)")
        axes[0].set_ylabel("Density")

        low = max(int(min(observed_counts.min(), simulated_counts.min())) - 2, 0)
        top = int(max(observed_counts.max(), simulated_counts.max(), 1))
        edges = np.arange(low, top + 2) - 0.5
        for values, color in (
            (observed_counts, OBSERVED_COLOR),
            (simulated_counts, MODEL_COLOR),
        ):
            axes[1].hist(
                values,
                bins=edges,
                density=True,
                histtype="step",
                color=color,
                linewidth=1.2,
            )
        axes[1].set_xlim(edges[0], edges[-1])
        axes[1].set_xlabel("Spikes per held-out trial")
        axes[1].set_ylabel("Density")
        fano = [
            float(values.var() / values.mean()) if values.mean() > 0 else np.nan
            for values in (observed_counts, simulated_counts)
        ]
        axes[1].text(
            0.97,
            0.94,
            f"Fano: observed {fano[0]:.2f}, model {fano[1]:.2f}",
            transform=axes[1].transAxes,
            ha="right",
            va="top",
            color="0.3",
            fontsize=7,
        )
        for y, label, color in (
            (0.94, "observed", OBSERVED_COLOR),
            (0.85, "model", MODEL_COLOR),
        ):
            axes[0].text(
                0.97,
                y,
                label,
                transform=axes[0].transAxes,
                ha="right",
                va="top",
                color=color,
                fontweight="bold",
            )
        for letter, axis in zip("ab", axes, strict=True):
            axis.text(
                -0.12,
                1.04,
                letter,
                transform=axis.transAxes,
                ha="left",
                va="bottom",
                fontweight="bold",
                fontsize=10,
            )
        figure.suptitle("")
        figure.text(
            0.5,
            1.0,
            f"Median-performance unit {unit_id}, simulated through its own history",
            ha="center",
            va="bottom",
            fontsize=7,
            color="0.3",
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
) -> tuple[Path | None, Path | None]:
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
                f"cross-validated $D^2$="
                f"{result['cross_validated_deviance_explained']:.3f}; "
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
            for values in (observed, prediction, simulation)
        ]
        rate_axis = axes[2]
        for rate, color, style in zip(
            rates,
            (OBSERVED_COLOR, MODEL_COLOR, MODEL_COLOR),
            ("-", "-", "--"),
            strict=True,
        ):
            rate_axis.plot(
                relative_times, rate, color=color, linewidth=1.25, linestyle=style
            )
        rate_axis.axvline(0, color="0.75", linewidth=0.8, zorder=0)
        rate_axis.set_ylabel("Mean rate (spikes/s)")
        rate_axis.set_xlabel("Time from first measured flash (s)")
        rate_axis.text(
            0.01,
            0.94,
            "solid: rate given the observed past; dashed: simulated spikes",
            transform=rate_axis.transAxes,
            ha="left",
            va="top",
            color="0.3",
            fontsize=7,
        )
        for y, label, color in (
            (0.86, "observed", OBSERVED_COLOR),
            (0.77, "model", MODEL_COLOR),
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
    if len(results) != len(selections):
        raise ValueError("Final and validation result counts differ.")
    selected_components = {item["components"] for item in results}
    if len(selected_components) != 1:
        raise ValueError("All final fits must use one camera-PC count.")
    selected_components = int(selected_components.pop())
    with design.with_suffix(".json").open() as handle:
        design_metadata = json.load(handle)
    design_pdf, design_png = plot_model_design(
        design_metadata, figure_dir / "model_design"
    )
    summary_pdf, summary_png, rate_deviance_rho = plot_population_summary(
        results, selections, figure_dir / "summary"
    )
    result, population_median = select_representative_result(results)
    unit_id = int(result["unit_id"])
    ordered_unit_ids, kernels = fitted_kernel_matrices(
        results, selections, design_metadata
    )
    task_names = [item["name"] for item in design_metadata["task_manifest"]]
    task_kernel_pdf, task_kernel_png = plot_kernel_figure(
        ordered_unit_ids,
        kernels,
        unit_id,
        task_names,
        [TASK_LABELS[name] for name in task_names],
        figure_dir / "fitted_task_kernels",
        columns=2,
        xlabel="Time from event (ms)",
        example_ylabel="$\\Delta$ log rate",
    )
    video_names = ["video_pc_1", "video_pc_2", "video_pc_3"]
    video_kernel_pdf, video_kernel_png = plot_kernel_figure(
        ordered_unit_ids,
        kernels,
        unit_id,
        video_names,
        ["ME PC 1", "ME PC 2", "ME PC 3"],
        figure_dir / "fitted_video_kernels",
        columns=3,
        xlabel="Lag (ms)",
        example_ylabel="$\\Delta$ log rate / PC sample",
    )

    prepared = _load_windows(windows)
    with np.load(windows, allow_pickle=False) as saved:
        relative_times = saved["relative_bin_centers_s"].copy()
        trial_split = saved["trial_split"].copy()
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
    counts = build_unit_counts(test_alignments, spike_times)

    test_rows = np.repeat(trial_split == 2, len(relative_times))
    common = np.load(design, mmap_mode="r", allow_pickle=False)[test_rows]
    if len(common) != len(counts):
        raise ValueError("Test design and response rows differ.")
    coefficients = np.asarray(result["coefficients"], dtype=float)
    common_columns = len(coefficients) - HISTORY_COLUMNS

    # The history block is per unit, so rebuild it and standardize it the way
    # the fit did: statistics from the train and validation rows only.
    valid = _valid_bin_mask(prepared)
    train, validation, _ = _split_masks(prepared["split"], valid)
    history_all = build_unit_history(prepared["alignments"], spike_times)
    _, history_mean, history_scale = training_zscore(history_all, train | validation)
    history = (history_all[test_rows] - history_mean) / history_scale
    taps, history_offset = history_filter(
        coefficients[common_columns:], history_scale, history_mean
    )

    trial_count = len(test_alignments)
    bin_count = len(relative_times)
    observed = counts.reshape(trial_count, bin_count)
    base_eta = (
        float(result["intercept"])
        + np.asarray(common[:, :common_columns]) @ coefficients[:common_columns]
    ).reshape(trial_count, bin_count)
    # Rate given the observed past, which is what the model predicts one step
    # ahead. The raster below instead feeds its own samples back.
    conditional = np.exp(
        base_eta
        + (history @ coefficients[common_columns:]).reshape(trial_count, bin_count)
    )
    if not np.isfinite(conditional).all() or np.any(conditional <= 0):
        raise ValueError("Conditional predictions must be finite and positive.")
    typical_trial = _select_count_typical_trial(observed)
    displayed_design = common[:, :common_columns].reshape(
        trial_count, bin_count, common_columns
    )[typical_trial]
    test_trial_numbers = prepared["selected_trial_numbers"][trial_split == 2]
    design_matrix_pdf, design_matrix_png = plot_design_matrix_trial(
        relative_times,
        observed[typical_trial],
        displayed_design,
        design_metadata,
        selected_components,
        unit_id,
        int(test_trial_numbers[typical_trial]),
        figure_dir / "design_matrix_trial",
    )
    simulation = simulate_spike_counts(
        base_eta, taps, history_offset, np.random.default_rng(SIMULATION_SEED)
    )
    checks_pdf, checks_png = plot_generative_checks(
        observed, simulation, unit_id, figure_dir / "generative_checks"
    )
    pdf_path, png_path = plot_prediction_figure(
        relative_times,
        observed,
        simulation,
        conditional,
        result,
        population_median,
        figure_dir / "predicted_spike_trains",
    )
    written = {
        "model_design": (design_pdf, design_png),
        "summary": (summary_pdf, summary_png),
        "fitted_task_kernels": (task_kernel_pdf, task_kernel_png),
        "fitted_video_kernels": (video_kernel_pdf, video_kernel_png),
        "design_matrix_trial": (design_matrix_pdf, design_matrix_png),
        "generative_checks": (checks_pdf, checks_png),
        "predicted_spike_trains": (pdf_path, png_path),
    }
    print(
        json.dumps(
            {
                "unit_id": unit_id,
                "selection": "cross-validated deviance nearest the population median",
                "population_median_cross_validated_deviance": population_median,
                "unit_cross_validated_deviance_explained": result[
                    "cross_validated_deviance_explained"
                ],
                "test_trials": trial_count,
                "observed_spikes": int(observed.sum()),
                "simulated_spikes": int(simulation.sum()),
                "simulation_seed": SIMULATION_SEED,
                "training_rate_test_deviance_spearman_rho": rate_deviance_rho,
                "design_matrix_trial": int(test_trial_numbers[typical_trial]),
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
