"""Compute the derived figure data for the fitted V1 GLM and persist it.

`unique` calls `write_figure_data` to plot the example unit, which is the unit
whose visual flash block explains the most deviance per second. This module reads the
saved per-unit fits, rebuilds that unit's design, reconstructs the
cross-validated predictions from the saved fold coefficients, decomposes the
final refit into groupwise log-rate terms, and reconstructs the kernels.
`figure_data.npz` and `figure_data.json` hold what those panels draw.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from thesis.ephys.analyses.glm.dmat import (
    BINWIDTH_S,
    HISTORY_COLUMNS,
    _load_selected_trials,
    _load_windows,
    _valid_bin_mask,
    build_unit_history,
    flash_events,
    spike_history_basis,
    task_temporal_bases,
    training_zscore,
)
from thesis.ephys.analyses.glm.fit import (
    _write_json_atomic,
    build_unit_counts,
    cross_validation_partitions,
    load_unit_records,
)
from thesis.ephys.units import fetch_unit_table

# The design codes flashes as a common kernel plus a centered synaptic-resource
# term. Each displayed flash condition maps to a resource-level key that selects
# a centered resource value from the design metadata: "first" (raw resource 1, a
# full pool) or "late" (the mean raw resource of adapted flashes, R < 1; equal
# to "first" if no flash is adapted).
EXAMPLE_UNIT_ID = 197
FLASH_COMMON = "visual_flash"
FLASH_ADAPT = "flash_adaptation"
FLASH_CONDITIONS = {"flash_first": "first", "flash_later": "late"}


def fitted_trial_contributions(
    design: np.ndarray, result: dict, design_metadata: dict
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
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
        target = visual if item["group"] == "sensory" else task
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
    linear_predictor = float(result["intercept"]) + sum(contributions.values())
    direct = float(result["intercept"]) + design @ coefficients
    if not np.allclose(linear_predictor, direct, rtol=1e-11, atol=1e-12):
        raise ValueError("Group contributions do not reproduce the fitted predictor.")
    with np.errstate(over="ignore"):
        expected_count = np.exp(linear_predictor)
    if not np.isfinite(expected_count).all() or np.any(expected_count <= 0):
        raise ValueError("Fitted trial counts must be finite and positive.")
    return contributions, linear_predictor, expected_count


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


def flash_condition_display(
    kernels: dict[str, tuple[np.ndarray, np.ndarray]],
    task_names: list[str],
    adapt_levels: dict[str, float],
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], list[str]]:
    """Replace the raw flash regressors with one kernel per flash condition.

    A flash's log-rate contribution is the common kernel plus the adaptation
    kernel scaled by its centered adaptation index. Evaluating that at
    representative levels gives one interpretable kernel per condition.
    """
    flash_blocks = (FLASH_COMMON, FLASH_ADAPT)
    display = {
        name: values for name, values in kernels.items() if name not in flash_blocks
    }
    times, common = kernels[FLASH_COMMON]
    adapt = kernels[FLASH_ADAPT][1]
    for name, level_key in FLASH_CONDITIONS.items():
        values = common + adapt_levels[level_key] * adapt
        if not np.isfinite(values).all():
            raise ValueError("Combined flash kernels must be finite.")
        display[name] = (times, values)
    remaining = [name for name in task_names if name not in flash_blocks]
    return display, [*FLASH_CONDITIONS, *remaining]


def _select_count_typical_trial(observed: np.ndarray) -> int:
    """Select the earliest trial nearest the median observed spike count."""
    if observed.ndim != 2 or len(observed) == 0:
        raise ValueError("Observed spikes must be a nonempty trial-by-bin matrix.")
    totals = observed.sum(axis=1)
    return int(np.argmin(np.abs(totals - np.median(totals))))


def block_deviance_rate(record: dict, fit: dict, group: str) -> float:
    """Return the deviance per second that one block explains uniquely for a unit.

    Unique ΔD² is a fraction of the null deviance of the unit, so its maximum
    selects a unit with few spikes. Null deviance per spike is proportional to
    bits/spike ÷ D², and the rate converts it to deviance per second (in bits).
    """
    if fit["cross_validated_deviance_explained"] <= 0:
        return -np.inf
    return (
        record["groups"][group]["unique_test_deviance_explained"]
        * fit["cross_validated_bits_per_spike"]
        / fit["cross_validated_deviance_explained"]
        * fit["fit_mean_count"]
        / BINWIDTH_S
    )


def example_unit_id(unique_records: list[dict], results: list[dict]) -> int:
    """Return unit 197 if it was fit, else the unit with the most flash deviance.

    Unit 197 of GRB006 20240821_121447 is the chosen example: 55 spikes/s and a
    clear flash response. For a fit without it (a sample, or another session),
    use the unit whose visual flash block explains the most deviance per second.
    """
    final = {int(item["unit_id"]): item for item in results}
    if EXAMPLE_UNIT_ID in final:
        return EXAMPLE_UNIT_ID
    best = max(
        unique_records,
        key=lambda record: block_deviance_rate(
            record, final[int(record["unit_id"])], FLASH_COMMON
        ),
    )
    return int(best["unit_id"])


def write_figure_data(
    windows: Path, design: Path, fit_dir: Path, example_id: int
) -> None:
    """Compute and persist the example-unit figure data for a completed fit."""
    checkpoints = fit_dir / "checkpoints"
    results = load_unit_records(checkpoints, "unit_*_final.json")
    selections = load_unit_records(checkpoints, "unit_*_validation.json")
    fold_records = load_unit_records(checkpoints, "unit_*_folds.json")
    if len(results) != len(selections) or len(results) != len(fold_records):
        raise ValueError("Final, validation, and fold result counts differ.")
    selected_components = {item["components"] for item in results}
    if len(selected_components) != 1:
        raise ValueError("All final fits must use one camera-PC count.")
    selected_components = int(selected_components.pop())
    with design.with_suffix(".json").open() as handle:
        design_metadata = json.load(handle)
    if int(design_metadata.get("video_columns_per_component", 0)) != 1:
        raise ValueError("Figures require one contemporaneous column per video PC.")

    result_by_id = {int(item["unit_id"]): item for item in results}
    selection_by_id = {int(item["unit_id"]): item for item in selections}
    if len(result_by_id) != len(results) or len(selection_by_id) != len(selections):
        raise ValueError("Fit results must contain unique unit IDs.")
    example_result = result_by_id[example_id]
    example_training_rate = (
        float(selection_by_id[example_id]["training_mean_count"]) / BINWIDTH_S
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

    matching_units = units.loc[units["unit_id"] == example_id]
    if len(matching_units) != 1:
        raise ValueError(f"Expected one eligible row for unit {example_id}.")
    example_spikes = np.asarray(matching_units.iloc[0]["spike_times_s"], dtype=float)
    example_history_raw = build_unit_history(prepared["alignments"], example_spikes)
    example_history, _, kernel_history_scale = training_zscore(
        example_history_raw, valid
    )
    kernels = fitted_kernels(example_result, design_metadata, kernel_history_scale)
    adapt_levels = next(
        item["adapt_levels_centered"]
        for item in design_metadata["task_manifest"]
        if item["name"] == "flash_adaptation"
    )
    display_kernels, display_task_names = flash_condition_display(
        kernels, task_names, adapt_levels
    )
    example_counts = build_unit_counts(prepared["alignments"], example_spikes)
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
    observed = example_counts[test_rows].reshape(trial_count, bin_count)

    conditional = np.full(len(common), np.nan)
    saved_folds = iter(fold_by_id[example_id]["folds"])
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
    if not np.isfinite(conditional[plotted]).all() or np.any(conditional[plotted] <= 0):
        raise ValueError("Conditional predictions must be finite and positive.")
    conditional = conditional.reshape(trial_count, bin_count)
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
    contributions, linear_predictor, fitted_count = fitted_trial_contributions(
        model_design, example_result, design_metadata
    )

    trials = _load_selected_trials(prepared)
    trial = trials.iloc[selected_trial_row]
    alignment = float(prepared["alignments"][selected_trial_row])
    flashes = np.asarray(trial["stim_pulse_times_s"], dtype=float)
    center_entry = float(trial["center_entry_s"])
    center_exit = float(trial["center_exit_s"])
    response_entry = float(trial["response_port_entry_s"])
    modeled_flash_times, resource_values = flash_events(
        trials.iloc[[selected_trial_row]]
    )
    modeled_flashes = flashes[(flashes >= center_entry) & (flashes <= response_entry)]
    if (
        len(modeled_flashes) == 0
        or len(modeled_flash_times) != len(modeled_flashes)
        or not np.isclose(modeled_flashes[0] - alignment, 0, atol=BINWIDTH_S)
    ):
        raise ValueError("Every displayed flash must be represented exactly once.")
    events = {
        "visual_flash": modeled_flash_times - alignment,
        "withdrawal": np.asarray([center_exit - alignment]),
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

    arrays = {
        "relative_times": relative_times,
        "prediction_observed": observed,
        "conditional": conditional,
        "test_valid": test_valid,
        "model_times": model_times,
        "model_counts": model_counts,
        "motion_times": motion_times,
        "motion_scores": motion_scores,
        "linear_predictor": linear_predictor,
        "fitted_count": fitted_count,
    }
    for name, values in contributions.items():
        arrays[f"contrib__{name}"] = values
    for name, values in events.items():
        arrays[f"event__{name}"] = values
    kernel_keys = list(display_kernels)
    for name, (kernel_times, kernel_values) in display_kernels.items():
        arrays[f"kern_t__{name}"] = kernel_times
        arrays[f"kern_v__{name}"] = kernel_values

    scalars = {
        "example_unit_id": example_id,
        "example_unit_training_rate_spikes_s": example_training_rate,
        "example_unit_cross_validated_deviance_explained": example_result[
            "cross_validated_deviance_explained"
        ],
        "selected_video_components": selected_components,
        "test_trials": trial_count,
        "model_design_trial": int(test_trial_numbers[typical_trial]),
        "model_design_flashes": len(modeled_flash_times),
        "model_design_first_flashes": int(np.count_nonzero(resource_values == 1.0)),
        "model_design_valid_bins": int(displayed_rows.sum()),
        "display_task_names": list(display_task_names),
        "kernel_keys": kernel_keys,
        "task_manifest": design_metadata["task_manifest"],
    }

    output = fit_dir / "figure_data.npz"
    fit_dir.mkdir(parents=True, exist_ok=True)
    matrix_partial = output.with_name(f"{output.name}.partial")
    # The CLI holds the session lock; discard unfinished writes before retrying.
    matrix_partial.unlink(missing_ok=True)
    with matrix_partial.open("xb") as handle:
        np.savez_compressed(handle, allow_pickle=False, **arrays)
    matrix_partial.replace(output)
    _write_json_atomic(fit_dir / "figure_data.json", scalars)
    print(f"Completed figure data: {output}", flush=True)
