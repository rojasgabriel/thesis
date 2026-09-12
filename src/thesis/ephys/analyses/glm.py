"""Fit a Poisson GLM to identify variables that predict V1 spiking.

Scientific comparison
---------------------
For GRB006 session 20240821_121447, predict V1 spikes from flashes, a
center-poke kernel truncated at the first flash, peri-exit movement, pre-response
choice side, additive video motion-energy PCs, and each unit's own strictly past
spike history. Bins after response entry are excluded. Whole trials are split
randomly 60/20/20. No coupling between units, session drift, go cue, outcome, or
punishment terms. Validation selects the motion-energy PC count. The 12-unit test
set never scores held-out trials.
"""

from __future__ import annotations

import argparse
import copy
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from damn.alignment import compute_spike_count, construct_timebins
from damn.objects.basis_function_objects import RaisedCosineBasis
from damn.objects.design_matrix_objects import DesignMatrix
from damn.objects.regressor_objects import ContinuousRegressor, EventRegressor
from scipy.special import xlogy
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import PoissonRegressor

from thesis.ephys.preprocessing.prepare_glm import (
    BINWIDTH_S,
    POST_S,
    PRE_S,
    training_zscore,
)
from thesis.ephys.trials import build_trial_table
from thesis.ephys.units import fetch_unit_table

VIDEO_COMPONENT_COUNTS = (10, 25, 50, 100, 200)
# Centred on the optimum measured for this design. The DAMN sweep put it at
# 1e-6 with 168 targets, which is about 3e-4 on sklearn's scale for a single
# neuron. The path still extends by a decade when an endpoint wins.
INITIAL_ALPHAS = tuple(np.logspace(-7, 1, 9))
VIDEO_BASIS_COLUMNS = 3
HISTORY_COLUMNS = 10
TEST_UNITS = 12
MAX_ITER = 500
TOL = 1e-7
MAX_ALPHA_EXTENSIONS = 6


def _flatten_events(values) -> np.ndarray:
    arrays = [np.asarray(value, dtype=float) for value in values if len(value)]
    return np.concatenate(arrays) if arrays else np.empty(0)


def task_temporal_bases() -> dict[str, RaisedCosineBasis]:
    """Return the temporal basis used for each sensory or task regressor."""
    flash = RaisedCosineBasis(6, 0, 0.151, BINWIDTH_S)
    # Poke leads the first flash by 54 ms median, 113 ms at most, and the kernel
    # is truncated there, so bases past 90 ms would be empty on every trial.
    poke = RaisedCosineBasis(4, 0, 0.091, BINWIDTH_S)
    peri_exit = RaisedCosineBasis(9, 0.301, 0.301, BINWIDTH_S)
    pre_response = RaisedCosineBasis(6, 0.301, 0, BINWIDTH_S)
    return {
        "visual_flash": flash,
        "center_poke": poke,
        "center_exit": peri_exit,
        "response_side": pre_response,
    }


def build_task_design(
    alignments: np.ndarray, trials
) -> tuple[DesignMatrix, list[dict]]:
    """Build sensory and task regressors with the accepted DAMN interface."""
    bases = task_temporal_bases()
    specifications = [
        (
            "visual_flash",
            _flatten_events(trials["stim_pulse_times_s"]),
            None,
            bases["visual_flash"],
            "sensory",
            "each measured flash",
        ),
        (
            "center_poke",
            trials["center_entry_s"].to_numpy(dtype=float),
            None,
            bases["center_poke"],
            "task",
            "center poke truncated at the first flash",
        ),
        (
            "center_exit",
            trials["center_exit_s"].to_numpy(dtype=float),
            None,
            bases["center_exit"],
            "task",
            "one event per completed trial",
        ),
        (
            "response_side",
            trials["response_port_entry_s"].to_numpy(dtype=float),
            trials["response"].to_numpy(dtype=float),
            bases["response_side"],
            "task",
            "left=-1, right=+1; pre-response only",
        ),
    ]
    design = DesignMatrix(alignments, PRE_S, POST_S, BINWIDTH_S)
    manifest = []
    for name, times, values, basis, group, coding in specifications:
        design.add_regressor(
            EventRegressor(
                name,
                times,
                BINWIDTH_S,
                event_values=values,
                basis_objects=[basis],
                tags=group.split(),
            )
        )
        manifest.append(
            {
                "name": name,
                "group": group,
                "events": len(times),
                "columns": basis.basis.shape[1],
                "kernel_range_s": [
                    float(basis.basis_time[0]),
                    float(basis.basis_time[-1]),
                ],
                "coding": coding,
            }
        )
    design.build_matrix()
    poke = design.regressors["center_poke"]
    rows_per_trial = poke.X.shape[0] // len(alignments)
    centers, _, _ = construct_timebins(PRE_S, POST_S, BINWIDTH_S)
    truncated = np.asarray(poke.X).reshape(len(alignments), rows_per_trial, -1)
    truncated[:, np.asarray(centers) >= 0] = 0
    poke._X = truncated.reshape(-1, poke.X.shape[1])
    values = np.asarray(design.X)
    start = 0
    widths = [regressor.X.shape[1] for regressor in design.regressors.values()]
    for item, columns in zip(manifest, widths, strict=True):
        stop = start + columns
        item["trials_with_nonzero_values"] = int(
            np.any(
                values[:, start:stop].reshape(len(alignments), rows_per_trial, -1),
                axis=(1, 2),
            ).sum()
        )
        start = stop
    return design, manifest


def task_column_names(manifest: list[dict]) -> list[str]:
    """Return stable names for the task-basis columns."""
    return [
        f"{item['name']}_basis_{index + 1:02d}"
        for item in manifest
        for index in range(item["columns"])
    ]


def build_unit_counts(alignments: np.ndarray, spike_times: np.ndarray) -> np.ndarray:
    """Return raw 1 ms spike counts on the GLM grid."""
    counts, _, _ = compute_spike_count(
        alignments, spike_times, PRE_S, POST_S, BINWIDTH_S
    )
    return counts.ravel()


def spike_history_basis() -> RaisedCosineBasis:
    """Return 10 log-spaced functions at strictly past lags 1 through 100 ms."""
    return RaisedCosineBasis(HISTORY_COLUMNS, 0, 0.1, BINWIDTH_S, log_scale=True)


def build_unit_history(alignments: np.ndarray, spike_times: np.ndarray) -> np.ndarray:
    """Return one unit's 10 strictly past self-history columns.

    Shifting the spike times by one bin keeps the filter causal, so the count in
    a bin is never a predictor of itself.
    """
    design = DesignMatrix(alignments, PRE_S, POST_S, BINWIDTH_S)
    design.add_regressor(
        EventRegressor(
            "self_history",
            np.asarray(spike_times, dtype=float) + BINWIDTH_S,
            BINWIDTH_S,
            basis_objects=[spike_history_basis()],
            tags="history",
        )
    )
    design.build_matrix()
    history = np.asarray(design.X)
    if history.shape[1] != HISTORY_COLUMNS:
        raise ValueError("Expected 10 strictly past self-history columns.")
    return history


def _design_with_history(
    common: np.ndarray, history: np.ndarray, common_columns: int
) -> np.ndarray:
    """Append a unit's standardized history block to the shared design."""
    design = np.empty(
        (common.shape[0], common_columns + history.shape[1]), dtype=np.float32
    )
    design[:, :common_columns] = common[:, :common_columns]
    design[:, common_columns:] = history
    return design


def video_temporal_basis() -> RaisedCosineBasis:
    """Return three smooth acausal motion-energy terms spanning -200 to +200 ms."""
    return RaisedCosineBasis(3, 0.201, 0.201, BINWIDTH_S)


def build_video_component_design(
    alignments: np.ndarray,
    frame_times: np.ndarray,
    score: np.ndarray,
    component: int = 0,
) -> np.ndarray:
    """Build one native DAMN continuous regressor with three temporal terms."""
    frame_times = np.asarray(frame_times, dtype=float)
    score = np.asarray(score, dtype=float)
    if frame_times.ndim != 1 or score.ndim != 1 or len(frame_times) != len(score):
        raise ValueError("Video times and one component must be matching vectors.")
    regressor = ContinuousRegressor(
        f"video_pc_{component + 1:03d}",
        frame_times,
        score,
        BINWIDTH_S,
        zscore=False,
        basis_objects=[video_temporal_basis()],
        tags="video",
    )
    regressor.build_regressor(alignments, PRE_S, POST_S)
    if regressor.X.shape[1] != VIDEO_BASIS_COLUMNS:
        raise ValueError("Expected three temporal columns per video component.")
    return regressor.X


def test_unit_indices(n_units: int) -> np.ndarray:
    """Select deterministic indices spaced across depth-sorted eligible units."""
    if n_units < TEST_UNITS:
        raise ValueError("Not enough eligible units for the requested test set.")
    return np.rint(np.linspace(0, n_units - 1, TEST_UNITS)).astype(int)


def poisson_nll(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return mean Poisson negative log likelihood without constant terms."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape or np.any(y_true < 0):
        raise ValueError(
            "Observed and predicted counts must be matching and nonnegative."
        )
    if not np.isfinite(y_pred).all() or np.any(y_pred <= 0):
        raise ValueError("Poisson predictions must be finite and positive.")
    return float(np.mean(y_pred - xlogy(y_true, y_pred)))


def poisson_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, fit_mean_count: float
) -> dict[str, float]:
    """Score predictions against a constant rate learned on the fit rows."""
    if not np.isfinite(fit_mean_count) or fit_mean_count <= 0:
        raise ValueError("The fit-set mean count must be finite and positive.")
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mean_nll = poisson_nll(y_true, y_pred)
    null = np.full_like(y_true, fit_mean_count, dtype=float)
    ll_model = np.sum(xlogy(y_true, y_pred) - y_pred)
    ll_null = np.sum(xlogy(y_true, null) - null)
    saturated = np.sum(xlogy(y_true, y_true) - y_true)
    model_deviance = 2 * (saturated - ll_model)
    null_deviance = 2 * (saturated - ll_null)
    spikes = float(y_true.sum())
    if spikes <= 0 or null_deviance <= 0:
        raise ValueError("Scoring rows must contain spikes and positive null deviance.")
    return {
        "mean_nll": mean_nll,
        "deviance_explained": float(1 - model_deviance / null_deviance),
        "bits_per_spike": float((ll_model - ll_null) / (np.log(2) * spikes)),
        "observed_spikes": spikes,
        "predicted_spikes": float(y_pred.sum()),
        "predicted_count_max": float(y_pred.max()),
        "predicted_count_p999": float(np.quantile(y_pred, 0.999)),
    }


def fit_poisson_at_alpha(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    warm_model: PoissonRegressor | None = None,
) -> PoissonRegressor:
    """Fit one unclipped canonical Poisson GLM and require LBFGS convergence."""
    if warm_model is None:
        model = PoissonRegressor(
            alpha=alpha,
            fit_intercept=True,
            solver="lbfgs",
            max_iter=MAX_ITER,
            tol=TOL,
            warm_start=True,
        )
    else:
        model = warm_model.set_params(alpha=alpha)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(X, y)
    convergence = [
        item for item in caught if issubclass(item.category, ConvergenceWarning)
    ]
    if convergence:
        raise RuntimeError(f"alpha={alpha:g}: {convergence[-1].message}")
    if not np.isfinite(model.coef_).all() or not np.isfinite(model.intercept_):
        raise RuntimeError("Poisson fit returned non-finite coefficients.")
    return model


def fit_poisson_alpha_path(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    *,
    initial_alphas: tuple[float, ...] = INITIAL_ALPHAS,
) -> tuple[PoissonRegressor, dict]:
    """Select L2 strength on validation data and extend endpoint grids."""
    pending = {float(alpha) for alpha in initial_alphas}
    if not pending or min(pending) <= 0:
        raise ValueError("Alpha candidates must be positive.")
    models: dict[float, PoissonRegressor] = {}
    losses: dict[float, float] = {}
    endpoint_plateau = False
    for extension in range(MAX_ALPHA_EXTENSIONS + 1):
        unfitted = pending - models.keys()
        while unfitted:
            errors = {}
            fitted = 0
            for alpha in sorted(unfitted, reverse=True):
                warm_model = None
                if models:
                    nearest = min(models, key=lambda value: abs(np.log(value / alpha)))
                    warm_model = copy.deepcopy(models[nearest])
                try:
                    model = fit_poisson_at_alpha(X_train, y_train, alpha, warm_model)
                except RuntimeError as error:
                    if warm_model is not None:
                        try:
                            model = fit_poisson_at_alpha(X_train, y_train, alpha)
                        except RuntimeError as cold_error:
                            errors[alpha] = cold_error
                            continue
                    else:
                        errors[alpha] = error
                        continue
                losses[alpha] = poisson_nll(y_validation, model.predict(X_validation))
                models[alpha] = model
                fitted += 1
            unfitted = pending - models.keys()
            if unfitted and not fitted:
                detail = "; ".join(str(errors[alpha]) for alpha in sorted(unfitted))
                raise RuntimeError(f"No converged penalty initialization: {detail}")
        ordered = sorted(losses)
        minimum = min(losses.values())
        tied = [
            alpha
            for alpha in ordered
            if np.isclose(losses[alpha], minimum, rtol=1e-10, atol=1e-12)
        ]
        best = max(tied)
        index = ordered.index(best)
        if 0 < index < len(ordered) - 1:
            break
        neighbor = ordered[1] if index == 0 else ordered[-2]
        if np.isclose(losses[best], losses[neighbor], rtol=1e-10, atol=1e-12):
            endpoint_plateau = True
            break
        if extension == MAX_ALPHA_EXTENSIONS:
            raise RuntimeError("Best L2 penalty remains at an extended grid endpoint.")
        pending.add(best / 10 if index == 0 else best * 10)
    path = {
        "alphas": ordered,
        "validation_mean_nll": [losses[alpha] for alpha in ordered],
        "best_alpha": best,
        "best_index": ordered.index(best),
        "endpoint_plateau": endpoint_plateau,
    }
    return models[best], path


def _load_windows(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as windows:
        selected_rows = windows["selected_trial_rows"]
        eligible = windows["eligible_trials"]
        return {
            "metadata": json.loads(str(windows["metadata_json"])),
            "selected_rows": selected_rows.copy(),
            "selected_trial_numbers": windows["trial_num"][selected_rows].copy(),
            "alignments": windows["first_stim_s"][eligible].copy(),
            "split": windows["bin_split"].copy(),
            "bin_times": windows["bin_center_s"].copy(),
            "relative_centers": windows["relative_bin_centers_s"].copy(),
            "response_times": windows["response_entry_s"][eligible].copy(),
        }


def _load_selected_trials(prepared: dict):
    metadata = prepared["metadata"]
    trials = build_trial_table(
        metadata["subject_name"], metadata["session_name"], include_frames=False
    )
    trials = trials.iloc[prepared["selected_rows"]].reset_index(drop=True)
    if not np.array_equal(trials["trial_num"], prepared["selected_trial_numbers"]):
        raise ValueError("Prepared and live trial selections differ.")
    return trials


def _valid_bin_mask(prepared: dict) -> np.ndarray:
    """Keep bins at or before that trial's response entry."""
    limits = prepared["response_times"] - prepared["alignments"]
    return (prepared["relative_centers"][None, :] <= limits[:, None]).ravel()


def _split_masks(
    split: np.ndarray, valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train / validation / test rows after the response-entry mask."""
    split = np.asarray(split)
    if not np.array_equal(np.unique(split), [0, 1, 2]):
        raise ValueError("Rows must contain train, validation, and test labels.")
    return tuple((split == value) & valid for value in (0, 1, 2))


def _write_json_atomic(path: Path, value: dict) -> None:
    partial = path.with_name(f"{path.name}.partial")
    if path.exists() or partial.exists():
        raise FileExistsError(path)
    with partial.open("x") as handle:
        json.dump(value, handle, indent=2)
    partial.replace(path)


def prepare_common_design(windows: Path, video: Path, output: Path) -> None:
    """Build and save the shared task and motion-energy design columns."""
    prepared = _load_windows(windows)
    trials = _load_selected_trials(prepared)
    with np.load(video, allow_pickle=False) as data:
        frame_times = data["frame_times_s"].copy()
        scores = data["scores"].copy()
    if scores.ndim != 2 or scores.shape[1] != max(VIDEO_COMPONENT_COUNTS):
        raise ValueError("Expected exactly 200 video-component score columns.")
    if len(frame_times) != len(scores):
        raise ValueError("Video times and score rows must match.")

    task, task_manifest = build_task_design(prepared["alignments"], trials)
    task_values = task.X
    task_columns = task_values.shape[1]
    valid = _valid_bin_mask(prepared)
    train_rows, _, _ = _split_masks(prepared["split"], valid)
    scaled_base, base_mean, base_scale = training_zscore(task_values, train_rows)
    base_names = task_column_names(task_manifest)
    base_columns = len(base_names)
    total_columns = base_columns + VIDEO_BASIS_COLUMNS * scores.shape[1]

    output.parent.mkdir(parents=True, exist_ok=True)
    matrix_partial = output.with_name(f"{output.name}.partial")
    metadata_path = output.with_suffix(".json")
    metadata_partial = metadata_path.with_name(f"{metadata_path.name}.partial")
    for path in (output, matrix_partial, metadata_path, metadata_partial):
        if path.exists():
            raise FileExistsError(path)
    matrix = np.lib.format.open_memmap(
        matrix_partial,
        mode="w+",
        dtype=np.float32,
        shape=(len(prepared["split"]), total_columns),
    )
    matrix[:, :base_columns] = scaled_base.astype(np.float32)
    del scaled_base, task_values

    video_mean = []
    video_scale = []
    video_names = []
    for component in range(scores.shape[1]):
        block = build_video_component_design(
            prepared["alignments"],
            frame_times,
            scores[:, component],
            component,
        )
        scaled, mean, scale = training_zscore(block, train_rows)
        start = base_columns + VIDEO_BASIS_COLUMNS * component
        matrix[:, start : start + VIDEO_BASIS_COLUMNS] = scaled.astype(np.float32)
        video_mean.extend(mean.tolist())
        video_scale.extend(scale.tolist())
        video_names.extend(
            f"video_pc_{component + 1:03d}_basis_{index + 1:02d}"
            for index in range(VIDEO_BASIS_COLUMNS)
        )
        if (component + 1) % 10 == 0:
            matrix.flush()
            print(
                f"Built {component + 1} of {scores.shape[1]} video components.",
                flush=True,
            )

    basis = video_temporal_basis()
    metadata = {
        "rows": matrix.shape[0],
        "columns": matrix.shape[1],
        "dtype": str(matrix.dtype),
        "task_columns": task_columns,
        "drift_columns": 0,
        "base_columns": base_columns,
        "video_components": scores.shape[1],
        "video_basis_columns_per_component": VIDEO_BASIS_COLUMNS,
        "video_basis_range_s": basis.basis_time[[0, -1]].tolist(),
        "video_basis_peak_s": [
            float(basis.basis_time[np.argmax(basis.basis[:, index])])
            for index in range(VIDEO_BASIS_COLUMNS)
        ],
        "video_component_candidates": list(VIDEO_COMPONENT_COUNTS),
        "column_names": base_names + video_names,
        "task_manifest": task_manifest,
        "training_mean": base_mean.tolist() + video_mean,
        "training_scale": base_scale.tolist() + video_scale,
        "split_row_counts": np.bincount(prepared["split"], minlength=3).tolist(),
    }
    matrix.flush()
    del matrix
    with metadata_partial.open("x") as handle:
        json.dump(metadata, handle, indent=2)
    matrix_partial.replace(output)
    metadata_partial.replace(metadata_path)
    omitted = {"column_names", "task_manifest", "training_mean", "training_scale"}
    print(
        json.dumps(
            {key: metadata[key] for key in metadata if key not in omitted}, indent=2
        )
    )


def _selection_for_unit(
    common: np.ndarray,
    base_columns: int,
    counts: np.ndarray,
    history: np.ndarray,
    train: np.ndarray,
    validation: np.ndarray,
) -> dict:
    fit_mean = float(counts[train].mean())
    if fit_mean <= 0:
        raise ValueError("Each unit must have at least one training spike.")
    history, _, _ = training_zscore(history, train)

    baseline = _design_with_history(common, history, base_columns)
    model, path = fit_poisson_alpha_path(
        baseline[train],
        counts[train],
        baseline[validation],
        counts[validation],
    )
    result = {
        "training_mean_count": fit_mean,
        "baseline": {
            "alpha_path": path,
            "validation": poisson_metrics(
                counts[validation], model.predict(baseline[validation]), fit_mean
            ),
        },
        "plus_video": {},
    }
    del model

    for component_count in VIDEO_COMPONENT_COUNTS:
        common_columns = base_columns + VIDEO_BASIS_COLUMNS * component_count
        design = _design_with_history(common, history, common_columns)
        model, path = fit_poisson_alpha_path(
            design[train],
            counts[train],
            design[validation],
            counts[validation],
        )
        result["plus_video"][str(component_count)] = {
            "alpha_path": path,
            "validation": poisson_metrics(
                counts[validation], model.predict(design[validation]), fit_mean
            ),
        }
        del model
    return result


def _final_fit_for_unit(
    common: np.ndarray,
    base_columns: int,
    counts: np.ndarray,
    history: np.ndarray,
    train: np.ndarray,
    validation: np.ndarray,
    test: np.ndarray,
    selection: dict,
    component_count: int,
) -> dict:
    fit = train | validation
    fit_mean = float(counts[fit].mean())
    history, _, _ = training_zscore(history, fit)
    baseline = _design_with_history(common, history, base_columns)
    baseline_model = fit_poisson_at_alpha(
        baseline[fit],
        counts[fit],
        selection["baseline"]["alpha_path"]["best_alpha"],
    )
    common_columns = base_columns + VIDEO_BASIS_COLUMNS * component_count
    full = _design_with_history(common, history, common_columns)
    full_model = fit_poisson_at_alpha(
        full[fit],
        counts[fit],
        selection["plus_video"][str(component_count)]["alpha_path"]["best_alpha"],
    )
    return {
        "fit_mean_count": fit_mean,
        "baseline": {
            "alpha": selection["baseline"]["alpha_path"]["best_alpha"],
            "intercept": float(baseline_model.intercept_),
            "coefficients": baseline_model.coef_.tolist(),
            "test": poisson_metrics(
                counts[test], baseline_model.predict(baseline[test]), fit_mean
            ),
        },
        "plus_video": {
            "components": component_count,
            "alpha": selection["plus_video"][str(component_count)]["alpha_path"][
                "best_alpha"
            ],
            "intercept": float(full_model.intercept_),
            "coefficients": full_model.coef_.tolist(),
            "test": poisson_metrics(
                counts[test], full_model.predict(full[test]), fit_mean
            ),
        },
    }


def fit_models(windows: Path, design: Path, unit_set: str, output_dir: Path) -> None:
    """Run validation selection; score held-out trials only for the all-unit run."""
    prepared = _load_windows(windows)
    common = np.load(design, mmap_mode="r", allow_pickle=False)
    with design.with_suffix(".json").open() as handle:
        design_metadata = json.load(handle)
    if list(common.shape) != [design_metadata["rows"], design_metadata["columns"]]:
        raise ValueError("Common design matrix and metadata shapes differ.")
    if common.shape[0] != len(prepared["split"]):
        raise ValueError("Common design and prepared response rows differ.")
    valid = _valid_bin_mask(prepared)
    train, validation, test = _split_masks(prepared["split"], valid)
    units = fetch_unit_table(
        prepared["metadata"]["subject_name"],
        prepared["metadata"]["session_name"],
        unit_criteria_id=1,
        stability_param_id=0,
        include_metrics=False,
    )
    unit_rows = (
        test_unit_indices(len(units)) if unit_set == "test" else np.arange(len(units))
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    selections = []
    for position, row in enumerate(unit_rows, start=1):
        unit = units.iloc[row]
        output = output_dir / f"unit_{int(unit['unit_id'])}_validation.json"
        if output.exists():
            with output.open() as handle:
                selection = json.load(handle)
        else:
            spikes = np.asarray(unit["spike_times_s"], dtype=float)
            counts = build_unit_counts(prepared["alignments"], spikes)
            selection = _selection_for_unit(
                common,
                design_metadata["base_columns"],
                counts,
                build_unit_history(prepared["alignments"], spikes),
                train,
                validation,
            )
            selection.update(
                unit_id=int(unit["unit_id"]),
                depth=float(unit["depth"]),
                spikes_in_selection_bins=int(counts[train | validation].sum()),
            )
            _write_json_atomic(output, selection)
        selections.append(selection)
        print(
            f"Validated unit {position} of {len(unit_rows)}: {int(unit['unit_id'])}",
            flush=True,
        )

    mean_validation_deviance = {
        str(count): float(
            np.mean(
                [
                    selection["plus_video"][str(count)]["validation"][
                        "deviance_explained"
                    ]
                    for selection in selections
                ]
            )
        )
        for count in VIDEO_COMPONENT_COUNTS
    }
    selected_components = max(
        VIDEO_COMPONENT_COUNTS,
        key=lambda count: mean_validation_deviance[str(count)],
    )
    summary = {
        "unit_set": unit_set,
        "units": len(unit_rows),
        "test_scored": unit_set == "all",
        "mean_validation_deviance_explained": mean_validation_deviance,
        "selected_video_components": selected_components,
        "fit_rows": {
            "train": int(train.sum()),
            "validation": int(validation.sum()),
            "test": int(test.sum()),
        },
    }

    if unit_set == "all":
        final_results = []
        for position, (row, selection) in enumerate(
            zip(unit_rows, selections, strict=True), start=1
        ):
            unit = units.iloc[row]
            output = output_dir / f"unit_{int(unit['unit_id'])}_test.json"
            if output.exists():
                with output.open() as handle:
                    final = json.load(handle)
            else:
                spikes = np.asarray(unit["spike_times_s"], dtype=float)
                counts = build_unit_counts(prepared["alignments"], spikes)
                final = _final_fit_for_unit(
                    common,
                    design_metadata["base_columns"],
                    counts,
                    build_unit_history(prepared["alignments"], spikes),
                    train,
                    validation,
                    test,
                    selection,
                    selected_components,
                )
                final.update(unit_id=int(unit["unit_id"]), depth=float(unit["depth"]))
                _write_json_atomic(output, final)
            final_results.append(final)
            print(
                f"Tested unit {position} of {len(unit_rows)}: {int(unit['unit_id'])}",
                flush=True,
            )
        differences = np.asarray(
            [
                item["plus_video"]["test"]["bits_per_spike"]
                - item["baseline"]["test"]["bits_per_spike"]
                for item in final_results
            ]
        )
        summary["test_video_delta_bits_per_spike"] = {
            "median": float(np.median(differences)),
            "q25": float(np.quantile(differences, 0.25)),
            "q75": float(np.quantile(differences, 0.75)),
        }

    _write_json_atomic(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser(
        "prepare", help="Windows, motion-energy SVD, and shared design"
    )
    prepare.add_argument("--frame-times", type=Path)
    for name, help_text in (
        ("fit", "Select and fit Poisson models"),
        ("attribute", "Refit shuffled blocks for conditional deviance"),
        ("figures", "Draw every figure for a completed fit"),
    ):
        command = subparsers.add_parser(name, help=help_text)
        command.add_argument("--units", choices=("test", "all"), default="test")
    args = parser.parse_args()
    model = PoissonGLM()
    if args.command == "prepare":
        if args.frame_times is not None:
            model.frame_times = args.frame_times
        model.prepare()
        return
    getattr(model, args.command)(args.units)


@dataclass
class PoissonGLM:
    """Default artifact layout and the commands that produce a fitted GLM."""

    root: Path = Path("figures/glm")
    subject: str = "GRB006"
    session: str = "20240821_121447"
    frame_times: Path = Path("figures/glm/frame_times.npy")

    @property
    def windows(self) -> Path:
        return self.root / "stimulus_windows_me.npz"

    @property
    def video(self) -> Path:
        return self.root / "video_me_features.npz"

    @property
    def design(self) -> Path:
        return self.root / "common_design_me.npy"

    def fit_dir(self, units: str) -> Path:
        return self.root / f"{units}_fit_me"

    def prepare(self) -> None:
        from thesis.ephys.preprocessing.prepare_glm import write_stimulus_windows
        from thesis.ephys.preprocessing.video_svd import write_motion_energy_features

        if self.windows.exists():
            print(f"Skipping {self.windows}; file exists.", flush=True)
        else:
            write_stimulus_windows(
                self.subject, self.session, self.windows, self.frame_times
            )
        if self.video.exists():
            print(f"Skipping {self.video}; file exists.", flush=True)
        else:
            write_motion_energy_features(self.windows, self.video)
        if self.design.exists():
            print(f"Skipping {self.design}; file exists.", flush=True)
        else:
            prepare_common_design(self.windows, self.video, self.design)

    def fit(self, units: str = "test") -> None:
        fit_models(self.windows, self.design, units, self.fit_dir(units))

    def attribute(self, units: str = "test") -> None:
        from thesis.ephys.analyses.glm_attribution import run_attribution

        run_attribution(self.windows, self.design, self.fit_dir(units), units)

    def figures(self, units: str = "all") -> None:
        from thesis.ephys.analyses.glm_figures import make_figures

        make_figures(self.windows, self.design, self.fit_dir(units))


if __name__ == "__main__":
    main()
