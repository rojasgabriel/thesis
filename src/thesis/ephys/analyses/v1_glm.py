"""Fit a Poisson GLM to identify variables that predict V1 spiking.

Scientific comparison
---------------------
For GRB006 session 20240821_121447, compare a sensory/task/history model with
the same model plus back-camera video PCs. The response is raw spike count in
1 ms bins from 99 ms before through 2.539 s after the first measured visual
flash. Trials are the sampling units and remain in chronological 60/20/20
training/validation/test splits. The pilot uses 12 depth-spaced units that pass
quality criterion 1 and stability parameter 0. No sensory-response selection
or baseline subtraction is used.

Measured flashes and task events use raised-cosine kernels. Self-history uses
10 log-spaced raised cosines over strictly past lags from 1 to 100 ms. Linear
and quadratic session-time terms absorb slow firing-rate drift. Each raw-video
PC uses three raised cosines over -200 to +200 ms because video is a behavioral
nuisance group, not a causal regressor. DAMN supplies native event alignment,
continuous resampling, and trial-edge truncation. Scikit-learn supplies the
unclipped log-link Poisson likelihood and L2-penalized LBFGS fit.

All settings are selected on validation trials. The pilot never scores test
responses. The full-unit run selects one video-PC count for the population,
refits coefficients on training plus validation trials, and scores test trials
once. Predictions condition on observed spike history; they are not free
simulations. Coefficients and predictive changes are conditional associations,
not causal effects.
"""

from __future__ import annotations

import argparse
import copy
import json
import warnings
from pathlib import Path

import numpy as np
from damn.alignment import compute_spike_count
from damn.objects.basis_function_objects import RaisedCosineBasis
from damn.objects.design_matrix_objects import DesignMatrix
from damn.objects.regressor_objects import ContinuousRegressor, EventRegressor
from scipy.special import xlogy
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import PoissonRegressor

from thesis.ephys.preprocessing.prepare_v1_glm import BINWIDTH_S, POST_S, PRE_S
from thesis.ephys.trials import build_trial_table
from thesis.ephys.units import fetch_unit_table

VIDEO_COMPONENT_COUNTS = (10, 25, 50, 100, 200)
INITIAL_ALPHAS = tuple(np.logspace(-3, 3, 7))
VIDEO_BASIS_COLUMNS = 3
HISTORY_COLUMNS = 10
MAX_ITER = 500
TOL = 1e-7
MAX_ALPHA_EXTENSIONS = 6


def training_zscore(
    values: np.ndarray, train_rows: np.ndarray | slice
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scale each design column with training rows and reject constants."""
    mean = values[train_rows].mean(axis=0)
    scale = values[train_rows].std(axis=0)
    if not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise ValueError("Design scaling statistics must be finite.")
    if np.any(scale == 0):
        raise ValueError("Every design column must vary in the training rows.")
    scaled = (values - mean) / scale
    if not np.isfinite(scaled).all():
        raise ValueError("Scaled design values must be finite.")
    return scaled, mean, scale


def _flatten_events(values) -> np.ndarray:
    arrays = [np.asarray(value, dtype=float) for value in values if len(value)]
    return np.concatenate(arrays) if arrays else np.empty(0)


def build_task_design(
    alignments: np.ndarray, trials
) -> tuple[DesignMatrix, list[dict]]:
    """Build sensory and task regressors with the accepted DAMN interface."""
    causal = RaisedCosineBasis(6, 0, 0.301, BINWIDTH_S)
    peri = RaisedCosineBasis(9, 0.301, 0.301, BINWIDTH_S)
    pre_response = RaisedCosineBasis(6, 0.301, 0, BINWIDTH_S)
    response_times = trials["response_port_entry_s"].to_numpy(dtype=float)
    specifications = [
        (
            "visual_flash",
            _flatten_events(trials["stim_pulse_times_s"]),
            None,
            causal,
            "sensory",
            "each measured flash",
        ),
        (
            "center_entry",
            trials["center_entry_s"].to_numpy(dtype=float),
            None,
            causal,
            "task",
            "one event per completed trial",
        ),
        (
            "go_cue_command",
            _flatten_events(trials["go_cue_times_s"]),
            None,
            causal,
            "audio task",
            "Bpod command converted to NIDQ time",
        ),
        (
            "center_exit",
            trials["center_exit_s"].to_numpy(dtype=float),
            None,
            peri,
            "task",
            "one event per completed trial",
        ),
        (
            "response_entry",
            response_times,
            None,
            peri,
            "task",
            "common response-entry effect",
        ),
        (
            "response_side",
            response_times,
            trials["response"].to_numpy(dtype=float),
            peri,
            "task",
            "left=-1, right=+1 at response entry",
        ),
        (
            "outcome",
            response_times,
            np.where(trials["rewarded"].to_numpy(dtype=bool), 1.0, -1.0),
            pre_response,
            "task",
            "error=-1, rewarded=+1 at response entry",
        ),
        (
            "wrong_punishment_command",
            _flatten_events(trials["punish_wrong_times_s"]),
            None,
            causal,
            "audio task",
            "Bpod command converted to NIDQ time",
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
    rows_per_trial = design.X.shape[0] // len(alignments)
    for item, regressor in zip(manifest, design.regressors.values(), strict=True):
        item["trials_with_nonzero_values"] = int(
            np.any(
                regressor.X.reshape(len(alignments), rows_per_trial, -1),
                axis=(1, 2),
            ).sum()
        )
    return design, manifest


def task_column_names(manifest: list[dict]) -> list[str]:
    """Return stable names for the task-basis columns."""
    return [
        f"{item['name']}_basis_{index + 1:02d}"
        for item in manifest
        for index in range(item["columns"])
    ]


def build_session_drift(bin_times: np.ndarray) -> np.ndarray:
    """Return linear and quadratic absolute session-time nuisance columns."""
    times = np.asarray(bin_times, dtype=float)
    if times.ndim != 1 or not np.isfinite(times).all():
        raise ValueError("Session bin times must be one finite vector.")
    span = np.ptp(times)
    if span <= 0:
        raise ValueError("Session bin times must span a positive duration.")
    session_fraction = (times - times.min()) / span
    return np.column_stack((session_fraction, session_fraction**2))


def video_temporal_basis() -> RaisedCosineBasis:
    """Return three smooth acausal video terms spanning -200 to +200 ms."""
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


def spike_history_basis() -> RaisedCosineBasis:
    """Return 10 log-spaced functions at strictly past lags 1 through 100 ms."""
    return RaisedCosineBasis(HISTORY_COLUMNS, 0, 0.1, BINWIDTH_S, log_scale=True)


def build_unit_design(
    alignments: np.ndarray, spike_times: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return raw spike counts and 10 native strictly past history columns."""
    counts, _, _ = compute_spike_count(
        alignments, spike_times, PRE_S, POST_S, BINWIDTH_S
    )
    design = DesignMatrix(alignments, PRE_S, POST_S, BINWIDTH_S)
    design.add_regressor(
        EventRegressor(
            "self_history",
            spike_times + BINWIDTH_S,
            BINWIDTH_S,
            basis_objects=[spike_history_basis()],
            tags="history",
        )
    )
    design.build_matrix()
    history = design.X
    if history.shape[1] != HISTORY_COLUMNS:
        raise ValueError("Expected 10 strictly past self-history columns.")
    return counts.ravel(), history


def pilot_indices(n_units: int, pilot_size: int = 12) -> np.ndarray:
    """Select deterministic indices spaced across depth-sorted eligible units."""
    if n_units < pilot_size:
        raise ValueError("Not enough eligible units for the requested pilot.")
    return np.rint(np.linspace(0, n_units - 1, pilot_size)).astype(int)


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
        return {
            "metadata": json.loads(str(windows["metadata_json"])),
            "selected_rows": selected_rows.copy(),
            "selected_trial_numbers": windows["trial_num"][selected_rows].copy(),
            "alignments": windows["first_stim_s"][windows["eligible_trials"]].copy(),
            "split": windows["bin_split"].copy(),
            "bin_times": windows["bin_center_s"].copy(),
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


def _write_json_atomic(path: Path, value: dict) -> None:
    partial = path.with_name(f"{path.name}.partial")
    if path.exists() or partial.exists():
        raise FileExistsError(path)
    with partial.open("x") as handle:
        json.dump(value, handle, indent=2)
    partial.replace(path)


def prepare_common_design(args: argparse.Namespace) -> None:
    """Build and save the shared task, drift, and video design columns."""
    prepared = _load_windows(args.windows)
    trials = _load_selected_trials(prepared)
    with np.load(args.video, allow_pickle=False) as video:
        frame_times = video["frame_times_s"].copy()
        scores = video["scores"].copy()
    if scores.ndim != 2 or scores.shape[1] != max(VIDEO_COMPONENT_COUNTS):
        raise ValueError("Expected exactly 200 video-component score columns.")
    if len(frame_times) != len(scores):
        raise ValueError("Video times and score rows must match.")

    task, task_manifest = build_task_design(prepared["alignments"], trials)
    task_values = task.X
    task_columns = task_values.shape[1]
    drift = build_session_drift(prepared["bin_times"])
    base = np.hstack((task_values, drift))
    train_rows = prepared["split"] == 0
    scaled_base, base_mean, base_scale = training_zscore(base, train_rows)
    base_names = task_column_names(task_manifest) + [
        "session_time_linear",
        "session_time_quadratic",
    ]
    base_columns = len(base_names)
    total_columns = base_columns + VIDEO_BASIS_COLUMNS * scores.shape[1]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    matrix_partial = args.output.with_name(f"{args.output.name}.partial")
    metadata_path = args.output.with_suffix(".json")
    metadata_partial = metadata_path.with_name(f"{metadata_path.name}.partial")
    for path in (args.output, matrix_partial, metadata_path, metadata_partial):
        if path.exists():
            raise FileExistsError(path)
    matrix = np.lib.format.open_memmap(
        matrix_partial,
        mode="w+",
        dtype=np.float32,
        shape=(len(prepared["split"]), total_columns),
    )
    matrix[:, :base_columns] = scaled_base.astype(np.float32)
    del base, scaled_base, task_values

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
        "drift_columns": 2,
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
    matrix_partial.replace(args.output)
    metadata_partial.replace(metadata_path)
    omitted = {"column_names", "task_manifest", "training_mean", "training_scale"}
    print(
        json.dumps(
            {key: metadata[key] for key in metadata if key not in omitted}, indent=2
        )
    )


def _contiguous_slices(split: np.ndarray) -> tuple[slice, slice, slice]:
    """Return contiguous chronological train, validation, and test slices."""
    if not np.array_equal(np.unique(split), [0, 1, 2]) or np.any(np.diff(split) < 0):
        raise ValueError("Rows must contain contiguous chronological 0/1/2 splits.")
    validation_start = int(np.flatnonzero(split == 1)[0])
    test_start = int(np.flatnonzero(split == 2)[0])
    return (
        slice(0, validation_start),
        slice(validation_start, test_start),
        slice(test_start, len(split)),
    )


def _design_with_history(
    common: np.ndarray, history: np.ndarray, common_columns: int
) -> np.ndarray:
    design = np.empty(
        (common.shape[0], common_columns + history.shape[1]), dtype=np.float32
    )
    design[:, :common_columns] = common[:, :common_columns]
    design[:, common_columns:] = history.astype(np.float32)
    return design


def _selection_for_unit(
    common: np.ndarray,
    base_columns: int,
    counts: np.ndarray,
    history: np.ndarray,
    split_slices: tuple[slice, slice, slice],
) -> dict:
    train, validation, _ = split_slices
    history_scaled, history_mean, history_scale = training_zscore(
        history, slice(0, train.stop)
    )
    fit_mean = float(counts[train].mean())
    if fit_mean <= 0:
        raise ValueError("Each unit must have at least one training spike.")

    baseline = _design_with_history(common, history_scaled, base_columns)
    model, path = fit_poisson_alpha_path(
        baseline[train],
        counts[train],
        baseline[validation],
        counts[validation],
    )
    result = {
        "history_training_mean": history_mean.tolist(),
        "history_training_scale": history_scale.tolist(),
        "training_mean_count": fit_mean,
        "baseline": {
            "alpha_path": path,
            "validation": poisson_metrics(
                counts[validation], model.predict(baseline[validation]), fit_mean
            ),
        },
        "plus_video": {},
    }
    del baseline, model

    for component_count in VIDEO_COMPONENT_COUNTS:
        common_columns = base_columns + VIDEO_BASIS_COLUMNS * component_count
        design = _design_with_history(common, history_scaled, common_columns)
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
        del design, model
    return result


def _final_fit_for_unit(
    common: np.ndarray,
    base_columns: int,
    counts: np.ndarray,
    history: np.ndarray,
    split_slices: tuple[slice, slice, slice],
    selection: dict,
    component_count: int,
) -> dict:
    _, validation, test = split_slices
    fit_stop = validation.stop
    history_mean = np.asarray(selection["history_training_mean"])
    history_scale = np.asarray(selection["history_training_scale"])
    history_scaled = (history - history_mean) / history_scale
    fit_mean = float(counts[:fit_stop].mean())

    baseline = _design_with_history(common, history_scaled, base_columns)
    baseline_model = fit_poisson_at_alpha(
        baseline[:fit_stop],
        counts[:fit_stop],
        selection["baseline"]["alpha_path"]["best_alpha"],
    )
    baseline_prediction = baseline_model.predict(baseline[test])

    common_columns = base_columns + VIDEO_BASIS_COLUMNS * component_count
    full = _design_with_history(common, history_scaled, common_columns)
    full_model = fit_poisson_at_alpha(
        full[:fit_stop],
        counts[:fit_stop],
        selection["plus_video"][str(component_count)]["alpha_path"]["best_alpha"],
    )
    full_prediction = full_model.predict(full[test])
    return {
        "fit_mean_count": fit_mean,
        "baseline": {
            "alpha": selection["baseline"]["alpha_path"]["best_alpha"],
            "intercept": float(baseline_model.intercept_),
            "coefficients": baseline_model.coef_.tolist(),
            "test": poisson_metrics(counts[test], baseline_prediction, fit_mean),
        },
        "plus_video": {
            "components": component_count,
            "alpha": selection["plus_video"][str(component_count)]["alpha_path"][
                "best_alpha"
            ],
            "intercept": float(full_model.intercept_),
            "coefficients": full_model.coef_.tolist(),
            "test": poisson_metrics(counts[test], full_prediction, fit_mean),
        },
    }


def fit_models(args: argparse.Namespace) -> None:
    """Run validation selection for pilot/all units and test only the all-unit run."""
    prepared = _load_windows(args.windows)
    common = np.load(args.design, mmap_mode="r", allow_pickle=False)
    with args.design.with_suffix(".json").open() as handle:
        design_metadata = json.load(handle)
    if list(common.shape) != [design_metadata["rows"], design_metadata["columns"]]:
        raise ValueError("Common design matrix and metadata shapes differ.")
    if common.shape[0] != len(prepared["split"]):
        raise ValueError("Common design and prepared response rows differ.")
    split_slices = _contiguous_slices(prepared["split"])
    units = fetch_unit_table(
        prepared["metadata"]["subject_name"],
        prepared["metadata"]["session_name"],
        unit_criteria_id=1,
        stability_param_id=0,
        include_metrics=False,
    )
    unit_rows = (
        pilot_indices(len(units)) if args.units == "pilot" else np.arange(len(units))
    )
    args.output.mkdir(parents=True, exist_ok=True)

    selections = []
    selection_stop = split_slices[1].stop
    rows_per_trial = len(prepared["split"]) // len(prepared["alignments"])
    selection_alignments = prepared["alignments"][: selection_stop // rows_per_trial]
    selection_common = common[:selection_stop]
    for position, row in enumerate(unit_rows, start=1):
        unit = units.iloc[row]
        output = args.output / f"unit_{int(unit['unit_id'])}_validation.json"
        counts, history = build_unit_design(
            selection_alignments, np.asarray(unit["spike_times_s"], dtype=float)
        )
        if output.exists():
            with output.open() as handle:
                selection = json.load(handle)
        else:
            selection = _selection_for_unit(
                selection_common,
                design_metadata["base_columns"],
                counts,
                history,
                split_slices,
            )
            selection.update(
                unit_id=int(unit["unit_id"]),
                depth=float(unit["depth"]),
                spikes_in_selection_bins=int(counts.sum()),
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
        "unit_set": args.units,
        "units": len(unit_rows),
        "test_scored": args.units == "all",
        "mean_validation_deviance_explained": mean_validation_deviance,
        "selected_video_components": selected_components,
    }

    if args.units == "all":
        test_rows = split_slices[2]
        final_results = []
        for position, (row, selection) in enumerate(
            zip(unit_rows, selections, strict=True), start=1
        ):
            unit = units.iloc[row]
            output = args.output / f"unit_{int(unit['unit_id'])}_test.json"
            counts, history = build_unit_design(
                prepared["alignments"], np.asarray(unit["spike_times_s"], dtype=float)
            )
            if output.exists():
                with output.open() as handle:
                    final = json.load(handle)
            else:
                final = _final_fit_for_unit(
                    common,
                    design_metadata["base_columns"],
                    counts,
                    history,
                    split_slices,
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
        summary["test_rows"] = test_rows.stop - test_rows.start

    _write_json_atomic(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    design = subparsers.add_parser("prepare-design", help="Build shared design columns")
    design.add_argument(
        "--windows",
        type=Path,
        default=Path("figures/v1_glm/stimulus_windows.npz"),
    )
    design.add_argument(
        "--video", type=Path, default=Path("figures/v1_glm/video_features.npz")
    )
    design.add_argument(
        "--output", type=Path, default=Path("figures/v1_glm/common_design.npy")
    )
    design.set_defaults(function=prepare_common_design)

    fit = subparsers.add_parser("fit", help="Select and fit Poisson models")
    fit.add_argument(
        "--windows",
        type=Path,
        default=Path("figures/v1_glm/stimulus_windows.npz"),
    )
    fit.add_argument(
        "--design", type=Path, default=Path("figures/v1_glm/common_design.npy")
    )
    fit.add_argument("--units", choices=("pilot", "all"), default="pilot")
    fit.add_argument("--output", type=Path)
    fit.set_defaults(function=fit_models)

    args = parser.parse_args()
    if args.command == "fit" and args.output is None:
        args.output = Path(f"figures/v1_glm/{args.units}_fit")
    args.function(args)


if __name__ == "__main__":
    main()
