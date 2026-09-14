"""Create figures for the fitted V1 spike-prediction GLM.

Prediction examples use the cross-validation model that held out each plotted
trial. The model-design example instead decomposes the canonical final refit,
which was fit on every valid bin. All rates condition on observed spike history.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.colors import ListedColormap
from scipy.ndimage import gaussian_filter1d
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thesis.ephys.analyses.glm import (
    CV_FOLDS,
    HISTORY_COLUMNS,
    VIDEO_BASIS_COLUMNS,
    VIDEO_COMPONENT_COUNTS,
    _load_selected_trials,
    _load_windows,
    _valid_bin_mask,
    build_unit_counts,
    build_unit_history,
    cross_validation_partitions,
    spike_history_basis,
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
    "visual_flash": "Visual flash",
    "center_poke": "Center poke",
    "center_exit": "Center exit",
    "response_entry": "Response entry",
    "response_side": "Response side\n(right − left)",
}
OTHER_RESPONSE_GROUPS = (
    "center_poke",
    "center_exit",
    "response_entry",
    "response_side",
    "video",
)
OTHER_RESPONSE_LABELS = {
    "center_poke": "center-poke contribution",
    "center_exit": "center-exit contribution",
    "response_entry": "response-entry contribution",
    "response_side": "response-side contribution",
    "video": "motion-energy contribution",
}
SMOOTHING_MS = 20
PREDICTION_SEED = 2008
DESIGN_RATE_PERCENTILE = 90
# Rebound by make_figures; read by _save_figure so plot signatures stay small.
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


def select_prediction_examples(
    results: list[dict], unique_records: list[dict]
) -> list[dict]:
    """Select distinct best-fit, visual, and non-visual prediction examples."""
    result_by_id = {int(item["unit_id"]): item for item in results}
    unique_by_id = {int(item["unit_id"]): item for item in unique_records}
    if (
        len(result_by_id) != len(results)
        or len(unique_by_id) != len(unique_records)
        or set(result_by_id) != set(unique_by_id)
    ):
        raise ValueError("Final and unique results must contain the same unique units.")
    eligible = [item for item in results if int(item["folds_scored"]) == CV_FOLDS]
    if len(eligible) < 3:
        raise ValueError("Three distinct units are required for prediction examples.")
    full_scores = np.asarray(
        [item["cross_validated_deviance_explained"] for item in eligible], dtype=float
    )
    if not np.isfinite(full_scores).all():
        raise ValueError("Prediction-example scores must be finite.")

    def unique_score(unit_id: int, group: str) -> float:
        score = float(
            unique_by_id[unit_id]["groups"][group]["unique_test_deviance_explained"]
        )
        if not np.isfinite(score):
            raise ValueError("Prediction-example scores must be finite.")
        return score

    best = max(
        eligible,
        key=lambda item: (
            float(item["cross_validated_deviance_explained"]),
            -int(item["unit_id"]),
        ),
    )
    used = {int(best["unit_id"])}
    sensory = max(
        (item for item in eligible if int(item["unit_id"]) not in used),
        key=lambda item: (
            unique_score(int(item["unit_id"]), "visual_flash"),
            -int(item["unit_id"]),
        ),
    )
    used.add(int(sensory["unit_id"]))

    other_candidates = [
        (unique_score(unit_id, group), -unit_id, group, result_by_id[unit_id])
        for unit_id in (int(item["unit_id"]) for item in eligible)
        if unit_id not in used
        for group in OTHER_RESPONSE_GROUPS
    ]
    other_score, _, other_group, other = max(
        other_candidates, key=lambda item: item[:2]
    )
    return [
        {
            "label": "best fit",
            "result": best,
            "selection_group": "full_model",
            "selection_score": float(best["cross_validated_deviance_explained"]),
        },
        {
            "label": "strongest visual-flash contribution",
            "result": sensory,
            "selection_group": "visual_flash",
            "selection_score": unique_score(int(sensory["unit_id"]), "visual_flash"),
        },
        {
            "label": f"strongest {OTHER_RESPONSE_LABELS[other_group]}",
            "result": other,
            "selection_group": other_group,
            "selection_score": other_score,
        },
    ]


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


def select_design_example_result(
    results: list[dict], selections: list[dict]
) -> tuple[dict, float]:
    """Select the unit nearest the requested training-rate percentile."""
    if not results:
        raise ValueError("No fitted unit results were found.")
    training_rate, _ = training_rate_and_deviance(results, selections)
    target = float(np.percentile(training_rate, DESIGN_RATE_PERCENTILE))
    index = min(
        range(len(results)),
        key=lambda position: (
            abs(training_rate[position] - target),
            int(results[position]["unit_id"]),
        ),
    )
    return results[index], float(training_rate[index])


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


def fitted_trial_contributions(
    design: np.ndarray, result: dict, design_metadata: dict
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, float]:
    """Decompose one fitted trial into exact groupwise log-rate terms."""
    design = np.asarray(design, dtype=float)
    components = int(result["components"])
    common_columns = int(design_metadata["base_columns"]) + (
        VIDEO_BASIS_COLUMNS * components
    )
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
        target = visual if item["name"] == "visual_flash" else task
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
    selected_components: int,
    unit_id: int,
    training_rate: float,
    trial_number: int,
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
        "visual_flash",
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
        "visual_flash",
        "center_exit",
        "response_entry",
        "response_side",
    )
    event_labels = {
        "center_poke": "Center poke",
        "visual_flash": "Visual flashes",
        "center_exit": "Center exit",
        "response_entry": "Response entry",
        "response_side": (
            "Response side: right (+1)"
            if response_side == 1
            else "Response side: left (−1)"
        ),
    }
    event_colors = {
        "visual_flash": GROUP_COLORS["visual"],
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
        event_axis.text(
            0.99,
            0.97,
            "shading = temporal support; flash support repeats after every flash",
            transform=event_axis.transAxes,
            ha="right",
            va="top",
            fontsize=7,
            color="0.3",
        )

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
        motion_axis.text(
            0.01,
            0.04,
            f"PCs 1–3 of {selected_components}; continuous filter spans ±200 ms "
            "around every bin",
            transform=motion_axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=7,
            color="0.3",
        )

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
        history_axis.text(
            0.99,
            0.93,
            "history starts 1 ms after each spike",
            transform=history_axis.transAxes,
            ha="right",
            va="top",
            fontsize=7,
            color="0.3",
        )

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
        event_axis.set_title(
            "Observed regressors and temporal support", loc="left", fontsize=9
        )
        contribution_axes[0].set_title(
            "Fitted contribution to log firing rate", loc="left", fontsize=9
        )
        rate_axis.set_title("Prediction", loc="left", fontsize=9)
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
        figure.text(
            0.99,
            0.995,
            f"unit {unit_id}: {training_rate:.1f} spikes/s in training; "
            f"example test trial {trial_number}",
            ha="right",
            va="top",
            color="0.3",
            fontsize=7,
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


def fitted_kernels(
    result: dict, design_metadata: dict, history_scale: np.ndarray
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Reconstruct one unit's task and self-history filters."""
    components = int(result["components"])
    common_columns = int(design_metadata["base_columns"]) + (
        VIDEO_BASIS_COLUMNS * components
    )
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
    representative_unit_id: int,
    task_names: list[str],
    output: Path,
) -> tuple[Path | None, Path | None]:
    """Show every interpretable kernel trace for one representative unit."""
    names = [*task_names, "history"]

    with plt.rc_context(FIGURE_STYLE):
        figure, axes = plt.subplots(3, 2, figsize=(7.4, 7.0))
        for letter, axis, name in zip("abcdef", axes.ravel(), names, strict=True):
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
                "$\Delta$ log rate / spike"
                if name == "history"
                else "$\Delta$ log rate",
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
        figure.text(
            0.99,
            0.995,
            f"Median-performance unit {representative_unit_id}",
            ha="right",
            va="top",
            color="0.3",
            fontsize=7,
        )
        figure.tight_layout(h_pad=1.8, w_pad=1.8, rect=(0, 0, 1, 0.98))
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
    training_rate: float,
    trial_number: int,
    output: Path,
) -> tuple[Path | None, Path | None]:
    """Show the response vector and basis-expanded input for one test trial."""
    task_stop = int(design_metadata["task_columns"])
    common_columns = int(design_metadata["base_columns"]) + (
        VIDEO_BASIS_COLUMNS * selected_components
    )
    expected_columns = common_columns + HISTORY_COLUMNS
    if design.shape != (len(relative_times), expected_columns):
        raise ValueError("Displayed trial design has the wrong shape.")
    if counts.shape != relative_times.shape:
        raise ValueError("Displayed counts and times must match.")
    if not np.isfinite(design).all() or not np.isfinite(counts).all():
        raise ValueError("Displayed trial values must be finite.")

    blocks = []
    cursor = 0
    for item in design_metadata["task_manifest"]:
        stop = cursor + int(item["columns"])
        blocks.append((cursor, stop, TASK_LABELS[item["name"]].replace("\n", " ")))
        cursor = stop
    if cursor != task_stop or int(design_metadata["base_columns"]) != task_stop:
        raise ValueError("Task manifest and task-column count differ.")
    for component in range(selected_components):
        stop = cursor + VIDEO_BASIS_COLUMNS
        blocks.append((cursor, stop, f"ME PC {component + 1}"))
        cursor = stop
    blocks.append((cursor, cursor + HISTORY_COLUMNS, "Spike history (1–100 ms)"))
    cursor += HISTORY_COLUMNS
    if cursor != expected_columns:
        raise ValueError("Regressor labels do not cover the displayed design.")
    if float(np.max(np.abs(design))) <= 0:
        raise ValueError("Displayed design matrix cannot be all zero.")
    mids = [(start + stop) / 2 for start, stop, _ in blocks]
    labels = [label for _, _, label in blocks]
    edges = (
        relative_times[0] - BINWIDTH_S / 2,
        relative_times[-1] + BINWIDTH_S / 2,
    )
    colors = plt.get_cmap()(np.linspace(0, 1, 257))
    colors[len(colors) // 2] = (0, 0, 0, 1)
    colormap = ListedColormap(colors)

    with plt.rc_context(FIGURE_STYLE):
        figure = plt.figure(figsize=(8.0, 8.5))
        grid = figure.add_gridspec(
            2,
            2,
            height_ratios=[0.8, 5.2],
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
            f"unit {unit_id}: {training_rate:.1f} spikes/s in training; "
            f"example test trial {trial_number}",
            transform=count_axis.transAxes,
            ha="right",
            va="bottom",
        )

        image = matrix_axis.imshow(
            design.T,
            aspect="auto",
            interpolation="nearest",
            extent=(*edges, expected_columns, 0),
            cmap=colormap,
            vmin=-1,
            vmax=1,
            rasterized=True,
        )
        matrix_axis.set_yticks(mids, labels)
        matrix_axis.tick_params(axis="y", labelsize=6)
        matrix_axis.set_xlabel("Time from first measured flash (s)")
        matrix_axis.set_ylabel("Regressor")
        figure.colorbar(
            image,
            cax=colorbar_axis,
            ticks=[-1, 0, 1],
            label="Standardized model input\n(display clipped at ±1)",
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
    examples: list[dict],
    output: Path,
) -> tuple[Path | None, Path | None]:
    """Write three held-out one-step spike-prediction examples."""
    if len(examples) != 3:
        raise ValueError("Exactly three prediction examples are required.")

    with plt.rc_context(FIGURE_STYLE):
        figure, axes = plt.subplots(
            3,
            3,
            figsize=(10.5, 5.8),
            sharex=True,
            gridspec_kw={"height_ratios": [1, 1, 1.2]},
        )
        for column, example in enumerate(examples):
            result = example["result"]
            observed = np.asarray(example["observed"])
            predicted = np.asarray(example["predicted"])
            conditional = np.asarray(example["conditional"])
            if observed.shape != predicted.shape or observed.shape != conditional.shape:
                raise ValueError("Observed and predicted example arrays must match.")
            trial_count = observed.shape[0]

            for axis, counts, label, color in zip(
                axes[:2, column],
                (observed, predicted),
                ("observed", "model (given observed past)"),
                (OBSERVED_COLOR, MODEL_COLOR),
                strict=True,
            ):
                axis.eventplot(
                    _raster_events(counts, relative_times),
                    colors=color,
                    lineoffsets=np.arange(1, trial_count + 1),
                    linelengths=0.8,
                    linewidths=0.3,
                )
                axis.set_ylim(0.5, trial_count + 0.5)
                axis.set_yticks([1, (trial_count + 1) // 2, trial_count])
                axis.axvline(0, color="0.75", linewidth=0.8, zorder=0)
                axis.text(
                    0.02,
                    1.02,
                    label,
                    color=color,
                    transform=axis.transAxes,
                    ha="left",
                    va="bottom",
                    fontweight="bold",
                    fontsize=7,
                )

            selection_score = float(example["selection_score"])
            full_score = float(result["cross_validated_deviance_explained"])
            score_text = (
                f"cross-validated $D^2$={full_score:.3f}"
                if example["selection_group"] == "full_model"
                else f"unique $\u0394D^2$={selection_score:.3f}; full $D^2$={full_score:.3f}"
            )
            axes[0, column].set_title(
                f"{example['label']}\nunit {int(result['unit_id'])}; {score_text}",
                loc="left",
                va="bottom",
                fontweight="bold",
                fontsize=8,
                pad=18,
            )

            sigma_bins = SMOOTHING_MS / (BINWIDTH_S * 1000)
            rates = [
                gaussian_filter1d(values.mean(axis=0) / BINWIDTH_S, sigma_bins)
                for values in (observed, conditional)
            ]
            rate_axis = axes[2, column]
            for rate, color in zip(
                rates,
                (OBSERVED_COLOR, MODEL_COLOR),
                strict=True,
            ):
                rate_axis.plot(relative_times, rate, color=color, linewidth=1.15)
            rate_axis.axvline(0, color="0.75", linewidth=0.8, zorder=0)
            for y, label, color in (
                (0.94, "observed", OBSERVED_COLOR),
                (0.83, "model", MODEL_COLOR),
            ):
                rate_axis.text(
                    0.97,
                    y,
                    label,
                    color=color,
                    transform=rate_axis.transAxes,
                    ha="right",
                    va="top",
                    fontweight="bold",
                    fontsize=7,
                )
            rate_axis.text(
                0.97,
                0.03,
                f"{SMOOTHING_MS} ms smoothing",
                transform=rate_axis.transAxes,
                ha="right",
                va="bottom",
                color="0.4",
                fontsize=6.5,
            )

            axes[0, column].text(
                -0.13,
                1.36,
                "abc"[column],
                transform=axes[0, column].transAxes,
                ha="left",
                va="bottom",
                fontweight="bold",
                fontsize=10,
            )
        axes[0, 0].set_ylabel("Test trials")
        axes[1, 0].set_ylabel("Test trials")
        axes[2, 0].set_ylabel("Mean rate (spikes/s)")
        edges = (
            relative_times[0] - BINWIDTH_S / 2,
            relative_times[-1] + BINWIDTH_S / 2,
        )
        axes[-1, 0].set_xlim(*edges)
        figure.supxlabel("Time from first measured flash (s)", y=0.02)
        figure.align_ylabels(axes[:, 0])
        figure.subplots_adjust(
            left=0.07, right=0.99, bottom=0.11, top=0.86, wspace=0.24, hspace=0.36
        )
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
    unique_records = []
    for path in sorted((checkpoints / "unique").glob("unit_*.json")):
        with path.open() as handle:
            unique_records.append(json.load(handle))
    if len(results) != len(selections) or len(results) != len(fold_records):
        raise ValueError("Final, validation, and fold result counts differ.")
    selected_components = {item["components"] for item in results}
    if len(selected_components) != 1:
        raise ValueError("All final fits must use one camera-PC count.")
    selected_components = int(selected_components.pop())
    with design.with_suffix(".json").open() as handle:
        design_metadata = json.load(handle)
    summary_pdf, summary_png, rate_deviance_rho = plot_population_summary(
        results, selections, figure_dir / "summary"
    )
    kernel_result, population_median = select_representative_result(results)
    design_result, design_training_rate = select_design_example_result(
        results, selections
    )
    prediction_examples = select_prediction_examples(results, unique_records)
    kernel_unit_id = int(kernel_result["unit_id"])
    design_unit_id = int(design_result["unit_id"])
    task_names = [item["name"] for item in design_metadata["task_manifest"]]

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
    test_rows = np.repeat(trial_split == 2, len(relative_times))
    common = np.load(design, mmap_mode="r", allow_pickle=False)[test_rows]
    common_columns = int(design_metadata["base_columns"]) + (
        VIDEO_BASIS_COLUMNS * selected_components
    )
    valid = _valid_bin_mask(prepared)
    trial_count = len(test_alignments)
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

    _, _, kernel_history_scale = training_zscore(
        build_unit_history(prepared["alignments"], unit_spike_times(kernel_unit_id)),
        valid,
    )
    kernels = fitted_kernels(kernel_result, design_metadata, kernel_history_scale)
    kernel_pdf, kernel_png = plot_kernel_figure(
        kernels,
        kernel_unit_id,
        task_names,
        figure_dir / "fitted_kernels",
    )
    for old_stem in ("fitted_task_kernels", "fitted_video_kernels"):
        for suffix in (".pdf", ".png"):
            (figure_dir / old_stem).with_suffix(suffix).unlink(missing_ok=True)

    design_spikes = unit_spike_times(design_unit_id)
    design_counts = build_unit_counts(prepared["alignments"], design_spikes)
    design_history_raw = build_unit_history(prepared["alignments"], design_spikes)
    count_grid = design_counts.reshape(len(trial_split), bin_count)
    history_grid = design_history_raw.reshape(
        len(trial_split), bin_count, HISTORY_COLUMNS
    )
    valid_grid = valid.reshape(len(trial_split), bin_count)
    adjacent_valid = valid_grid[:, 1:] & valid_grid[:, :-1]
    if not np.array_equal(
        history_grid[:, 1:, 0][adjacent_valid],
        count_grid[:, :-1][adjacent_valid],
    ):
        raise ValueError("Spike history is not one bin behind spikes on valid rows.")
    design_history, _, _ = training_zscore(design_history_raw, valid)
    observed = design_counts[test_rows].reshape(trial_count, bin_count)

    def prediction_data(selected_result: dict) -> tuple[np.ndarray, np.ndarray]:
        selected_unit = int(selected_result["unit_id"])
        spike_times = unit_spike_times(selected_unit)
        counts = build_unit_counts(prepared["alignments"], spike_times)
        history_all = build_unit_history(prepared["alignments"], spike_times)
        conditional = np.full(len(common), np.nan)
        saved_folds = iter(fold_by_id[selected_unit]["folds"])
        for partition in partitions:
            if counts[partition["test"]].sum() <= 0:
                continue
            fold = next(saved_folds, None)
            if fold is None:
                raise ValueError("Saved fold count does not match fitted partitions.")
            coefficients = np.asarray(fold["coefficients"], dtype=float)
            if len(coefficients) != common_columns + HISTORY_COLUMNS:
                raise ValueError(
                    "Saved coefficient count does not match the selected design."
                )
            history, _, _ = training_zscore(history_all, partition["fit"])
            plotted_rows = partition["all"][test_rows]
            linear_predictor = (
                float(fold["intercept"])
                + np.asarray(common[plotted_rows, :common_columns])
                @ coefficients[:common_columns]
                + history[test_rows][plotted_rows] @ coefficients[common_columns:]
            )
            conditional[plotted_rows] = np.exp(linear_predictor)
        if next(saved_folds, None) is not None:
            raise ValueError("Saved fold count does not match the fitted partitions.")
        if not np.isfinite(conditional).all() or np.any(conditional <= 0):
            raise ValueError("Conditional predictions must be finite and positive.")
        return (
            counts[test_rows].reshape(trial_count, bin_count),
            conditional.reshape(trial_count, bin_count),
        )

    cached_predictions = {
        int(example["result"]["unit_id"]): prediction_data(example["result"])
        for example in prediction_examples
    }
    typical_trial = _select_count_typical_trial(np.where(test_valid, observed, 0))
    displayed_design = np.column_stack(
        (
            common[:, :common_columns].reshape(trial_count, bin_count, common_columns)[
                typical_trial
            ],
            design_history[test_rows].reshape(trial_count, bin_count, HISTORY_COLUMNS)[
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
        fitted_trial_contributions(model_design, design_result, design_metadata)
    )

    trials = _load_selected_trials(prepared)
    trial = trials.iloc[selected_trial_row]
    alignment = float(prepared["alignments"][selected_trial_row])
    flashes = np.asarray(trial["stim_pulse_times_s"], dtype=float) - alignment
    response_time = float(trial["response_port_entry_s"]) - alignment
    if (
        len(flashes) == 0
        or not np.isclose(flashes[0], 0, atol=BINWIDTH_S)
        or np.any(flashes > response_time)
    ):
        raise ValueError("Every displayed flash must be measured before response.")
    events = {
        "visual_flash": flashes,
        "center_poke": np.asarray([float(trial["center_entry_s"]) - alignment]),
        "center_exit": np.asarray([float(trial["center_exit_s"]) - alignment]),
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
        selected_components,
        design_unit_id,
        design_training_rate,
        int(test_trial_numbers[typical_trial]),
        figure_dir / "model_design",
    )
    design_matrix_pdf, design_matrix_png = plot_design_matrix_trial(
        relative_times,
        observed[typical_trial],
        displayed_design,
        design_metadata,
        selected_components,
        design_unit_id,
        design_training_rate,
        int(test_trial_numbers[typical_trial]),
        figure_dir / "design_matrix_trial",
    )
    plotted_examples = []
    prediction_manifest = []
    for example in prediction_examples:
        example_result = example["result"]
        example_unit = int(example_result["unit_id"])
        example_observed, conditional = cached_predictions[example_unit]
        seed = PREDICTION_SEED + example_unit
        predicted = np.random.default_rng(seed).poisson(conditional)
        plotted_examples.append(
            {
                **example,
                "observed": example_observed,
                "conditional": conditional,
                "predicted": predicted,
            }
        )
        prediction_manifest.append(
            {
                "label": example["label"],
                "unit_id": example_unit,
                "selection_group": example["selection_group"],
                "selection_score": float(example["selection_score"]),
                "cross_validated_deviance_explained": float(
                    example_result["cross_validated_deviance_explained"]
                ),
                "observed_spikes": int(example_observed.sum()),
                "conditional_sampled_spikes": int(predicted.sum()),
                "prediction_seed": seed,
            }
        )
    pdf_path, png_path = plot_prediction_figure(
        relative_times,
        plotted_examples,
        figure_dir / "predicted_spike_trains",
    )
    written = {
        "model_design": (design_pdf, design_png),
        "summary": (summary_pdf, summary_png),
        "fitted_kernels": (kernel_pdf, kernel_png),
        "design_matrix_trial": (design_matrix_pdf, design_matrix_png),
        "predicted_spike_trains": (pdf_path, png_path),
    }
    print(
        json.dumps(
            {
                "kernel_unit_id": kernel_unit_id,
                "kernel_selection": (
                    "cross-validated deviance nearest the population median"
                ),
                "population_median_cross_validated_deviance": population_median,
                "kernel_unit_cross_validated_deviance_explained": kernel_result[
                    "cross_validated_deviance_explained"
                ],
                "test_trials": trial_count,
                "prediction_examples": prediction_manifest,
                "training_rate_test_deviance_spearman_rho": rate_deviance_rho,
                "design_matrix_unit_id": design_unit_id,
                "design_matrix_unit_training_rate_spikes_s": design_training_rate,
                "design_matrix_unit_selection": (
                    f"nearest the {DESIGN_RATE_PERCENTILE}th percentile of "
                    "training firing rate"
                ),
                "design_matrix_trial": int(test_trial_numbers[typical_trial]),
                "model_design_group_sum_max_abs_error": contribution_error,
                "model_design_flashes": len(flashes),
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
