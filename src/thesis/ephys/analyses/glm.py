"""Fit a Poisson GLM to identify variables that predict V1 spiking.

Scientific comparison
---------------------
For GRB006 session 20240821_121447, predict V1 spikes from flashes split at
center exit into stationary and running conditions, a center-poke kernel
truncated at the first flash, peri-exit movement, pre-response choice side split
into a side-independent and a contrast kernel, 10 additive video motion-energy
PCs, and each unit's own strictly past spike history. Bins after response entry
are excluded. Whole trials are split randomly 60/20/20. No coupling between
units, session drift, go cue, outcome, or punishment terms. Validation compares
motion-energy PC counts, while the fitted model uses 10 PCs. Held-out performance
is the ten-fold cross-validated deviance, so every trial is scored once. A
20-unit random sample is available for quick runs.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import copy
import fcntl
import json
import os
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from damn.alignment import compute_spike_count, construct_timebins
from damn.objects.basis_function_objects import RaisedCosineBasis
from damn.objects.design_matrix_objects import DesignMatrix
from damn.objects.regressor_objects import EventRegressor
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
VIDEO_COMPONENTS = 10
# The optimum for this design sits near 3e-4. Below about 1e-5 the fit is
# effectively unpenalized and LBFGS stops hitting its iteration limit rather
# than converging, so the grid starts above that. It still extends by a decade
# when an endpoint wins, and a penalty that fails to converge is dropped.
INITIAL_ALPHAS = tuple(np.logspace(-5, 3, 9))
HISTORY_COLUMNS = 10
SAMPLE_UNITS = 20
SAMPLE_SEED = 20260914
# Per-worker state, filled by _init_worker.
_SHARED: dict = {}


# Bump the stage a record belongs to when its computation changes, so saved
# records from older logic are recomputed instead of silently reused. Tying
# this to a commit hash instead invalidates every record on any edit to this
# file, including edits that cannot affect the result.
RECORD_VERSIONS = {
    "selection": 3,
    "folds": 4,
    "final": 3,
    "unique": 4,
}


def code_version(stage: str) -> int:
    """Return the version of the logic that produces one kind of record."""
    return RECORD_VERSIONS[stage]


def _cached(output: Path, stage: str) -> dict | None:
    """Return a saved record only when the current logic produced it."""
    if not output.exists():
        return None
    with output.open() as handle:
        record = json.load(handle)
    if record.get("record_version") != code_version(stage):
        return None
    return record


CV_FOLDS = 10
CV_SEED = 20260913
CV_INNER_FRACTION = 0.25
MAX_ITER = 500
MAX_ALPHA_EXTENSIONS = 6


def _flatten_events(values) -> np.ndarray:
    arrays = [np.asarray(value, dtype=float) for value in values if len(value)]
    return np.concatenate(arrays) if arrays else np.empty(0)


def task_temporal_bases() -> dict[str, RaisedCosineBasis]:
    """Return the temporal basis used for each sensory or task regressor."""
    # Most flashes are 40 ms apart. Linear spacing keeps resolution across the
    # short response window instead of spending log-spaced bases below 10 ms.
    flash = RaisedCosineBasis(6, 0, 0.151, BINWIDTH_S)
    # Poke leads the first flash by 54 ms median, 113 ms at most, and the kernel
    # is truncated there, so bases past 90 ms would be empty on every trial.
    poke = RaisedCosineBasis(4, 0, 0.091, BINWIDTH_S)
    peri_exit = RaisedCosineBasis(9, 0.301, 0.301, BINWIDTH_S)
    pre_response = RaisedCosineBasis(6, 0.301, 0, BINWIDTH_S)
    return {
        "stationary_flash": flash,
        "running_flash": flash,
        "center_poke": poke,
        "center_exit": peri_exit,
        "response_entry": pre_response,
        "response_side": pre_response,
    }


def split_flash_events(trials) -> tuple[np.ndarray, np.ndarray]:
    """Split every modeled flash at center exit for each completed trial."""
    stationary, running = [], []
    for flashes, center_entry, center_exit, response_entry in zip(
        trials["stim_pulse_times_s"],
        trials["center_entry_s"],
        trials["center_exit_s"],
        trials["response_port_entry_s"],
        strict=True,
    ):
        flashes = np.asarray(flashes, dtype=float)
        stationary.append(flashes[(flashes >= center_entry) & (flashes < center_exit)])
        running.append(flashes[(flashes >= center_exit) & (flashes <= response_entry)])
        modeled = flashes[(flashes >= center_entry) & (flashes <= response_entry)]
        if len(stationary[-1]) + len(running[-1]) != len(modeled):
            raise ValueError("Every modeled flash must be classified exactly once.")
    return _flatten_events(stationary), _flatten_events(running)


def build_task_design(
    alignments: np.ndarray, trials
) -> tuple[DesignMatrix, list[dict]]:
    """Build sensory and task regressors with the accepted DAMN interface."""
    bases = task_temporal_bases()
    stationary_flashes, running_flashes = split_flash_events(trials)
    specifications = [
        (
            "stationary_flash",
            stationary_flashes,
            None,
            bases["stationary_flash"],
            "sensory",
            "each measured flash before center exit",
        ),
        (
            "running_flash",
            running_flashes,
            None,
            bases["running_flash"],
            "sensory",
            "each measured flash from center exit through response entry",
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
            "response_entry",
            trials["response_port_entry_s"].to_numpy(dtype=float),
            None,
            bases["response_entry"],
            "task",
            "every response; the side-independent part",
        ),
        (
            "response_side",
            trials["response_port_entry_s"].to_numpy(dtype=float),
            trials["response"].to_numpy(dtype=float),
            bases["response_side"],
            "task",
            "left=-1, right=+1; the choice contrast",
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
    common: np.ndarray,
    history: np.ndarray,
    common_columns: int,
    rows: np.ndarray | None = None,
) -> np.ndarray:
    """Append a unit's standardized history block to the shared design.

    Pass `rows` to build only the rows a fit needs. Materializing all 775,866
    and then slicing costs the full array plus every slice; roughly half those
    rows are masked out after response entry and never scored.
    """
    index = slice(None) if rows is None else rows
    height = common.shape[0] if rows is None else int(np.count_nonzero(rows))
    design = np.empty((height, common_columns + history.shape[1]), dtype=np.float32)
    design[:, :common_columns] = common[index, :common_columns]
    design[:, common_columns:] = history[index]
    return design


def build_video_component_design(
    alignments: np.ndarray,
    frame_times: np.ndarray,
    score: np.ndarray,
) -> np.ndarray:
    """Interpolate one contemporaneous motion-energy PC onto the DAMN grid."""
    frame_times = np.asarray(frame_times, dtype=float)
    score = np.asarray(score, dtype=float)
    if frame_times.ndim != 1 or score.ndim != 1 or len(frame_times) != len(score):
        raise ValueError("Video times and one component must be matching vectors.")
    if (
        not np.isfinite(frame_times).all()
        or not np.isfinite(score).all()
        or np.any(np.diff(frame_times) <= 0)
    ):
        raise ValueError("Video times and scores must be finite and time ordered.")
    alignments = np.asarray(alignments, dtype=float)
    if alignments.ndim != 1 or not np.isfinite(alignments).all():
        raise ValueError("Trial alignments must be a finite vector.")
    centers, _, _ = construct_timebins(PRE_S, POST_S, BINWIDTH_S)
    prediction_times = (alignments[:, None] + centers).ravel()
    return np.interp(prediction_times, frame_times, score, left=0, right=0)[:, None]


def sample_unit_indices(n_units: int) -> np.ndarray:
    """Draw a fixed random sample of units for quick runs, sorted by depth."""
    if n_units < SAMPLE_UNITS:
        raise ValueError("Not enough eligible units for the requested sample.")
    rng = np.random.default_rng(SAMPLE_SEED)
    return np.sort(rng.choice(n_units, size=SAMPLE_UNITS, replace=False))


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
    failed: set[float] = set()
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
                # LBFGS can stop without converging at either extreme: weak
                # penalties hit the iteration limit, strong ones terminate
                # abnormally once the coefficients are driven to zero. Neither
                # says the unit is unfittable, so drop those penalties and keep
                # the ones that worked. Only give up when none converged.
                if not models:
                    detail = "; ".join(str(errors[alpha]) for alpha in sorted(unfitted))
                    raise RuntimeError(f"No converged penalty initialization: {detail}")
                failed |= unfitted
                pending -= unfitted
                unfitted = set()
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
        candidate = best / 10 if index == 0 else best * 10
        if candidate in failed:
            # The grid already reaches as far as this design can be fitted.
            endpoint_plateau = True
            break
        if extension == MAX_ALPHA_EXTENSIONS:
            raise RuntimeError("Best L2 penalty remains at an extended grid endpoint.")
        pending.add(candidate)
    path = {
        "alphas": ordered,
        "validation_mean_nll": [losses[alpha] for alpha in ordered],
        "best_alpha": best,
        "best_index": ordered.index(best),
        "endpoint_plateau": endpoint_plateau,
        "unconverged_alphas": sorted(failed),
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


def _write_json_atomic(path: Path, value: dict, *, overwrite: bool = False) -> None:
    """Write JSON through a partial file so a kill cannot leave a half record.

    Per-unit records refuse to overwrite, which is what makes a run resumable.
    Derived summaries pass overwrite, since they are recomputed from those
    records every time and a stale one would otherwise block the resume.
    """
    partial = path.with_name(f"{path.name}.partial")
    if not overwrite and path.exists():
        raise FileExistsError(path)
    partial.unlink(missing_ok=True)
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
    total_columns = base_columns + scores.shape[1]

    output.parent.mkdir(parents=True, exist_ok=True)
    matrix_partial = output.with_name(f"{output.name}.partial")
    metadata_path = output.with_suffix(".json")
    metadata_partial = metadata_path.with_name(f"{metadata_path.name}.partial")
    for path in (matrix_partial, metadata_partial):
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
        )
        scaled, mean, scale = training_zscore(block, train_rows)
        start = base_columns + component
        matrix[:, start] = scaled[:, 0].astype(np.float32)
        video_mean.extend(mean.tolist())
        video_scale.extend(scale.tolist())
        video_names.append(f"video_pc_{component + 1:03d}")
        if (component + 1) % 10 == 0:
            matrix.flush()
            print(
                f"Built {component + 1} of {scores.shape[1]} video components.",
                flush=True,
            )

    metadata = {
        "rows": matrix.shape[0],
        "columns": matrix.shape[1],
        "dtype": str(matrix.dtype),
        "task_columns": task_columns,
        "drift_columns": 0,
        "base_columns": base_columns,
        "video_components": scores.shape[1],
        "video_columns_per_component": 1,
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


def _common_design_is_current(output: Path) -> bool:
    """Return whether a saved design has the current task and video columns."""
    metadata_path = output.with_suffix(".json")
    if not output.exists() or not metadata_path.exists():
        return False
    try:
        with metadata_path.open() as handle:
            metadata = json.load(handle)
    except (OSError, ValueError):
        return False
    bases = task_temporal_bases()
    manifest = metadata.get("task_manifest", [])
    return (
        metadata.get("video_columns_per_component") == 1
        and [item.get("name") for item in manifest] == list(bases)
        and [item.get("columns") for item in manifest]
        == [basis.basis.shape[1] for basis in bases.values()]
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
        common_columns = base_columns + component_count
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


def _final_task(job: dict) -> dict:
    """Refit one unit on every valid trial and save the canonical model.

    This is the model the figures and the block analysis read. It is not
    scored here: its held-out performance is the cross-validated deviance,
    carried over from the fold records.
    """
    output = Path(job["output"])
    cached = _cached(output, "final")
    if cached is not None:
        return cached
    prepared = _SHARED["prepared"]
    rows = _SHARED["all_valid"]
    spikes = np.asarray(job["spike_times"], dtype=float)
    counts = build_unit_counts(prepared["alignments"], spikes)
    history, _, _ = training_zscore(
        build_unit_history(prepared["alignments"], spikes), rows
    )
    design = _design_with_history(
        _SHARED["common"], history, job["common_columns"], rows
    )
    model = fit_poisson_at_alpha(design, counts[rows], job["alpha"])
    folds = job["cross_validated"]
    record = {
        "record_version": code_version("final"),
        "unit_id": job["unit_id"],
        "depth": job["depth"],
        "components": job["components"],
        "alpha": job["alpha"],
        "fit_rows": int(np.count_nonzero(rows)),
        "fit_mean_count": float(counts[rows].mean()),
        "intercept": float(model.intercept_),
        "coefficients": model.coef_.tolist(),
        "cross_validated_deviance_explained": folds["deviance_explained_mean"],
        "cross_validated_deviance_explained_sem": folds["deviance_explained_sem"],
        "cross_validated_bits_per_spike": folds["bits_per_spike_mean"],
        "folds_scored": folds["folds_scored"],
    }
    _write_json_atomic(output, record, overwrite=True)
    return record


def _init_worker(
    windows: Path,
    design: Path,
    shuffle_seed: int | None,
    common_columns: int | None,
) -> None:
    """Rebuild the shared state once per worker, not once per unit.

    Everything here is a deterministic function of the window file and the
    seeds, so each worker derives it rather than receiving it. Sending the
    fold masks with every job instead would ship about 4 GB across 168 units.
    """
    prepared = _load_windows(windows)
    valid = _valid_bin_mask(prepared)
    train, validation, test = _split_masks(prepared["split"], valid)
    with np.load(windows, allow_pickle=False) as saved:
        trial_split = saved["trial_split"].copy()
    bins_per_trial = len(prepared["split"]) // len(trial_split)
    partitions = cross_validation_partitions(len(trial_split), bins_per_trial, valid)
    common = np.load(design, mmap_mode="r", allow_pickle=False)
    if common_columns is not None:
        common = np.ascontiguousarray(common[:, :common_columns])
    _SHARED.update(
        prepared=prepared,
        common=common,
        train=train,
        validation=validation,
        all_valid=valid,
        partitions=partitions,
    )
    if shuffle_seed is not None:
        from thesis.ephys.analyses.glm_unique import shuffle_permutations

        _SHARED["within_trial"] = shuffle_permutations(
            len(trial_split), bins_per_trial, shuffle_seed, valid=valid
        )


def run_over_units(
    task,
    jobs: list,
    windows: Path,
    design: Path,
    workers: int | None,
    stage: str,
    shuffle_seed: int | None = None,
):
    """Run one task per unit, in parallel when that helps, and yield results.

    Each unit is an independent fit that writes its own record, so a failed
    worker costs that unit rather than the run. BLAS threading barely helps on
    this design (8 threads buy about 1.3x), so workers run single-threaded and
    the parallelism goes across units instead.
    """
    # Leave one core free so the machine stays usable.
    count = min(max(1, workers or (os.cpu_count() or 2) - 1), len(jobs))
    reused = [_cached(Path(job["output"]), stage) is not None for job in jobs]
    shared_columns = {job.get("common_columns") for job in jobs}
    common_columns = shared_columns.pop() if len(shared_columns) == 1 else None
    if count <= 1:
        _init_worker(windows, design, shuffle_seed, common_columns)
        for job, cache_hit in zip(jobs, reused, strict=True):
            try:
                yield task(job), cache_hit
            except Exception as error:  # noqa: BLE001
                print(
                    f"Failed unit {job['unit_id']}: {type(error).__name__}: {error}",
                    flush=True,
                )
                yield None, False
        return
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"
    print(f"Running {len(jobs)} units across {count} workers.", flush=True)
    with futures.ProcessPoolExecutor(
        max_workers=count,
        initializer=_init_worker,
        initargs=(windows, design, shuffle_seed, common_columns),
    ) as pool:
        for job, cache_hit, future in zip(
            jobs, reused, [pool.submit(task, job) for job in jobs], strict=True
        ):
            try:
                yield future.result(), cache_hit
            except Exception as error:  # noqa: BLE001
                # One unit's failure should not discard the rest of the run.
                print(
                    f"Failed unit {job['unit_id']}: {type(error).__name__}: {error}",
                    flush=True,
                )
                yield None, False


def trial_folds(n_trials: int, n_folds: int, seed: int) -> np.ndarray:
    """Assign each trial to one fold, so every trial is tested exactly once."""
    if n_trials < n_folds:
        raise ValueError("Fewer trials than folds.")
    order = np.random.default_rng(seed).permutation(n_trials)
    folds = np.empty(n_trials, dtype=np.int8)
    for index, block in enumerate(np.array_split(order, n_folds)):
        folds[block] = index
    return folds


def cross_validation_partitions(
    n_trials: int, bins_per_trial: int, valid: np.ndarray
) -> list[dict[str, np.ndarray]]:
    """Recreate the exact whole-trial folds used for fitting and prediction."""
    valid = np.asarray(valid, dtype=bool)
    if valid.shape != (n_trials * bins_per_trial,):
        raise ValueError("Validity mask does not match the trial grid.")
    folds = trial_folds(n_trials, CV_FOLDS, CV_SEED)
    fold_rows = np.repeat(folds, bins_per_trial)
    partitions = []
    for fold in range(CV_FOLDS):
        held_out = fold_rows == fold
        rest = np.flatnonzero(folds != fold)
        inner_trials = np.random.default_rng([CV_SEED, fold]).choice(
            rest,
            size=max(1, int(round(CV_INNER_FRACTION * len(rest)))),
            replace=False,
        )
        inner_rows = np.repeat(
            np.isin(np.arange(n_trials), inner_trials), bins_per_trial
        )
        partitions.append(
            {
                "all": held_out,
                "test": held_out & valid,
                "inner": inner_rows & valid,
                "train": ~inner_rows & ~held_out & valid,
                "fit": ~held_out & valid,
            }
        )
    return partitions


def _fold_fit_for_unit(
    common: np.ndarray,
    common_columns: int,
    counts: np.ndarray,
    history: np.ndarray,
    train: np.ndarray,
    inner: np.ndarray,
    test: np.ndarray,
) -> dict:
    """Select the penalty inside the fold, refit on everything but the fold."""
    fit = train | inner
    fit_mean = float(counts[fit].mean())
    if fit_mean <= 0:
        raise ValueError("Each unit must have at least one training spike.")
    history, _, _ = training_zscore(history, fit)
    # Build each block of rows on its own, so the full-height design is never
    # materialized alongside its slices.
    _, path = fit_poisson_alpha_path(
        _design_with_history(common, history, common_columns, train),
        counts[train],
        _design_with_history(common, history, common_columns, inner),
        counts[inner],
    )
    model = fit_poisson_at_alpha(
        _design_with_history(common, history, common_columns, fit),
        counts[fit],
        path["best_alpha"],
    )
    prediction = model.predict(
        _design_with_history(common, history, common_columns, test)
    )
    return {
        "alpha": path["best_alpha"],
        "fit_mean_count": fit_mean,
        "intercept": float(model.intercept_),
        "coefficients": model.coef_.tolist(),
        "test": poisson_metrics(counts[test], prediction, fit_mean),
    }


def _fold_task(job: dict) -> dict:
    """Score one unit across every fold and save its record."""
    output = Path(job["output"])
    cached = _cached(output, "folds")
    if cached is not None:
        return cached
    prepared = _SHARED["prepared"]
    spikes = np.asarray(job["spike_times"], dtype=float)
    counts = build_unit_counts(prepared["alignments"], spikes)
    history = build_unit_history(prepared["alignments"], spikes)
    results, skipped = [], 0
    for part in _SHARED["partitions"]:
        if counts[part["test"]].sum() <= 0:
            # Deviance explained is undefined with no spikes to explain.
            skipped += 1
            continue
        results.append(
            _fold_fit_for_unit(
                _SHARED["common"],
                job["common_columns"],
                counts,
                history,
                part["train"],
                part["inner"],
                part["test"],
            )
        )
    if len(results) < 2:
        raise ValueError(
            f"only {len(results)} test folds contained spikes; at least 2 are required"
        )
    deviance = np.asarray([item["test"]["deviance_explained"] for item in results])
    bits = np.asarray([item["test"]["bits_per_spike"] for item in results])
    record = {
        "record_version": code_version("folds"),
        "unit_id": job["unit_id"],
        "depth": job["depth"],
        "components": job["components"],
        "folds_scored": len(results),
        "folds_without_spikes": skipped,
        "folds": results,
        "deviance_explained_mean": float(deviance.mean()),
        "deviance_explained_sem": float(deviance.std(ddof=1) / np.sqrt(len(deviance))),
        "bits_per_spike_mean": float(bits.mean()),
        "bits_per_spike_sem": float(bits.std(ddof=1) / np.sqrt(len(bits))),
    }
    _write_json_atomic(output, record, overwrite=True)
    return record


def crossvalidate(
    windows: Path,
    design: Path,
    unit_set: str,
    output_dir: Path,
    components: int,
    workers: int | None = None,
) -> None:
    """Score every trial once through k-fold cross-validation over trials.

    Folds are over whole trials, never bins, because bins inside a trial are
    correlated. The motion-energy PC count is fixed beforehand by `fit`, so
    only each unit's penalty is chosen inside a fold.
    """
    prepared = _load_windows(windows)
    with design.with_suffix(".json").open() as handle:
        design_metadata = json.load(handle)
    common_columns = int(design_metadata["base_columns"]) + components
    with np.load(windows, allow_pickle=False) as saved:
        trial_split = saved["trial_split"].copy()

    units = fetch_unit_table(
        prepared["metadata"]["subject_name"],
        prepared["metadata"]["session_name"],
        unit_criteria_id=1,
        stability_param_id=0,
        include_metrics=False,
    )
    unit_rows = (
        sample_unit_indices(len(units))
        if unit_set == "sample"
        else np.arange(len(units))
    )
    checkpoints = output_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    print(
        f"{CV_FOLDS}-fold cross-validation over {len(trial_split)} trials, "
        f"{len(unit_rows)} units, {components} motion-energy PCs.",
        flush=True,
    )

    jobs = [
        {
            "unit_id": int(units.iloc[row]["unit_id"]),
            "depth": float(units.iloc[row]["depth"]),
            "spike_times": np.asarray(units.iloc[row]["spike_times_s"], dtype=float),
            "components": components,
            "common_columns": common_columns,
            "output": str(
                checkpoints / f"unit_{int(units.iloc[row]['unit_id'])}_folds.json"
            ),
        }
        for row in unit_rows
    ]
    records = []
    failures = 0
    for position, (record, reused) in enumerate(
        run_over_units(_fold_task, jobs, windows, design, workers, "folds"), start=1
    ):
        if record is None:
            failures += 1
            continue
        records.append(record)
        action = "Reused" if reused else "Completed"
        print(
            f"{action} cross-validation {position}/{len(jobs)}: "
            f"unit {record['unit_id']}  "
            f"D2 {record['deviance_explained_mean']:.4f} "
            f"+- {record['deviance_explained_sem']:.4f}",
            flush=True,
        )
    if failures:
        raise RuntimeError(
            f"Cross-validation incomplete: {failures}/{len(jobs)} units failed. "
            "Per-unit results already written are safe to reuse; rerun the same command."
        )

    means = np.asarray([item["deviance_explained_mean"] for item in records])
    summary = {
        "folds": CV_FOLDS,
        "fold_seed": CV_SEED,
        "inner_fraction": CV_INNER_FRACTION,
        "unit_set": unit_set,
        "units": len(records),
        "components": components,
        "trials": int(len(trial_split)),
        "cross_validated_deviance_explained": {
            "median": float(np.median(means)),
            "q25": float(np.quantile(means, 0.25)),
            "q75": float(np.quantile(means, 0.75)),
        },
    }
    _write_json_atomic(
        output_dir / "cross_validation_summary.json", summary, overwrite=True
    )
    print(json.dumps(summary, indent=2))


def _selection_task(job: dict) -> dict:
    """Select each model width's penalty for one unit and save the record."""
    output = Path(job["output"])
    cached = _cached(output, "selection")
    if cached is not None:
        return cached
    prepared = _SHARED["prepared"]
    spikes = np.asarray(job["spike_times"], dtype=float)
    counts = build_unit_counts(prepared["alignments"], spikes)
    selection = _selection_for_unit(
        _SHARED["common"],
        job["base_columns"],
        counts,
        build_unit_history(prepared["alignments"], spikes),
        _SHARED["train"],
        _SHARED["validation"],
    )
    selection.update(
        record_version=code_version("selection"),
        unit_id=job["unit_id"],
        depth=job["depth"],
        spikes_in_selection_bins=int(
            counts[_SHARED["train"] | _SHARED["validation"]].sum()
        ),
    )
    _write_json_atomic(output, selection, overwrite=True)
    return selection


def fit_models(
    windows: Path,
    design: Path,
    unit_set: str,
    output_dir: Path,
    workers: int | None = None,
) -> None:
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
        sample_unit_indices(len(units))
        if unit_set == "sample"
        else np.arange(len(units))
    )
    checkpoints = output_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)

    jobs = [
        {
            "unit_id": int(units.iloc[row]["unit_id"]),
            "depth": float(units.iloc[row]["depth"]),
            "spike_times": np.asarray(units.iloc[row]["spike_times_s"], dtype=float),
            "base_columns": design_metadata["base_columns"],
            "output": str(
                checkpoints / f"unit_{int(units.iloc[row]['unit_id'])}_validation.json"
            ),
        }
        for row in unit_rows
    ]
    selections = []
    failures = 0
    for position, (selection, reused) in enumerate(
        run_over_units(_selection_task, jobs, windows, design, workers, "selection"),
        start=1,
    ):
        if selection is None:
            failures += 1
            continue
        selections.append(selection)
        action = "Reused" if reused else "Completed"
        print(
            f"{action} validation {position}/{len(jobs)}: unit {selection['unit_id']}",
            flush=True,
        )
    if failures:
        raise RuntimeError(
            f"Validation incomplete: {failures}/{len(jobs)} units failed. "
            "Per-unit results already written are safe to reuse; rerun the same command."
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
    summary = {
        "unit_set": unit_set,
        "units": len(unit_rows),
        "mean_validation_deviance_explained": mean_validation_deviance,
        "selected_video_components": VIDEO_COMPONENTS,
        "fit_rows": {
            "train": int(train.sum()),
            "validation": int(validation.sum()),
            "test": int(test.sum()),
        },
    }

    _write_json_atomic(output_dir / "validation_summary.json", summary, overwrite=True)
    print(json.dumps(summary, indent=2))


PIPELINE = """\
pipeline
  uv run glm prepare                 trial windows, motion-energy PCs, design
  uv run glm fit --units all         penalty, cross-validation, final refit
  uv run glm unique --units all      unique and maximal deviance per block
  uv run glm figures --units all     kernels and held-out predictions

Artifacts go to figures/glm/<subject>_<session>/. Restart checkpoints stay in
the fit directory's checkpoints/ folder. Use --units sample for a 20-unit
smoke run on any command.
"""


def refit_final(
    windows: Path,
    design: Path,
    unit_set: str,
    output_dir: Path,
    components: int,
    workers: int | None = None,
) -> None:
    """Refit every unit on all valid trials, after cross-validation has run.

    This is the canonical model the figures and the block analysis read. Its
    penalty is the median of the penalties the folds chose, so it depends on
    cross-validation having finished.
    """
    prepared = _load_windows(windows)
    with design.with_suffix(".json").open() as handle:
        design_metadata = json.load(handle)
    units = fetch_unit_table(
        prepared["metadata"]["subject_name"],
        prepared["metadata"]["session_name"],
        unit_criteria_id=1,
        stability_param_id=0,
        include_metrics=False,
    )
    unit_rows = (
        sample_unit_indices(len(units))
        if unit_set == "sample"
        else np.arange(len(units))
    )
    jobs: list[dict] = [
        {
            "unit_id": int(units.iloc[row]["unit_id"]),
            "depth": float(units.iloc[row]["depth"]),
            "spike_times": np.asarray(units.iloc[row]["spike_times_s"], dtype=float),
        }
        for row in unit_rows
    ]
    # Refit each unit on every valid trial, which is the model the figures and
    # the block analysis use. Its held-out score comes from cross-validation
    # rather than from a second, weaker split.
    checkpoints = output_dir / "checkpoints"
    fold_records = {
        int(json.loads(path.read_text())["unit_id"]): json.loads(path.read_text())
        for path in checkpoints.glob("unit_*_folds.json")
    }
    final_jobs = [
        {
            "unit_id": job["unit_id"],
            "depth": job["depth"],
            "spike_times": job["spike_times"],
            "components": components,
            "common_columns": int(design_metadata["base_columns"]) + components,
            "alpha": float(
                np.median([f["alpha"] for f in fold_records[job["unit_id"]]["folds"]])
            ),
            "cross_validated": fold_records[job["unit_id"]],
            "output": str(checkpoints / f"unit_{job['unit_id']}_final.json"),
        }
        for job in jobs
        if job["unit_id"] in fold_records
    ]
    missing = len(jobs) - len(final_jobs)
    if missing:
        raise ValueError(
            f"{missing} units have no cross-validated record; run fit again."
        )
    failures = 0
    for position, (record, reused) in enumerate(
        run_over_units(_final_task, final_jobs, windows, design, workers, "final"),
        start=1,
    ):
        if record is None:
            failures += 1
            continue
        action = "Reused" if reused else "Completed"
        print(
            f"{action} final fit {position}/{len(final_jobs)}: "
            f"unit {record['unit_id']}",
            flush=True,
        )
    if failures:
        raise RuntimeError(
            f"Final refit incomplete: {failures}/{len(final_jobs)} units failed. "
            "Per-unit results already written are safe to reuse; rerun the same command."
        )


@contextmanager
def _artifact_lock(root: Path):
    """Prevent two GLM commands from writing to one session at the same time."""
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".glm.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(
                f"Another GLM command is already running for {root}."
            ) from None
        yield


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="glm",
        description=(
            "Poisson GLM for V1 spiking from flashes, task events, movement, "
            "and each unit's own spike history."
        ),
        epilog=PIPELINE,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="command")
    for name, help_text in (
        ("prepare", "Trial windows, motion-energy PCs, and the shared design"),
        ("fit", "Compare PC counts, cross-validate 10 PCs, then refit"),
        ("unique", "Unique and maximal explained deviance per block"),
        ("figures", "Kernels and held-out predictions"),
    ):
        command = subparsers.add_parser(
            name, help=help_text, description=help_text + "."
        )
        command.add_argument("--subject", default="GRB006", help="Subject name.")
        command.add_argument(
            "--session", default="20240821_121447", help="Session name."
        )
        command.add_argument(
            "--root",
            type=Path,
            help="Artifact directory. Defaults to figures/glm/<subject>_<session>.",
        )
        if name == "prepare":
            command.add_argument(
                "--frame-times",
                type=Path,
                help="Camera frame times. Defaults to frame_times.npy under the root.",
            )
            continue
        command.add_argument(
            "--units",
            choices=("sample", "all"),
            default="sample",
            help="Fit 20 sampled units or every eligible one.",
        )
        command.add_argument(
            "--workers",
            type=int,
            help="Units to fit at once. Defaults to one less than the core count.",
        )
        if name == "figures":
            command.add_argument(
                "--output-dir",
                type=Path,
                help="Write figures here instead of beside the fit.",
            )
            command.add_argument(
                "--format",
                choices=("pdf", "png", "both"),
                default="both",
                help="Which file formats to write.",
            )
    args = parser.parse_args()
    model = PoissonGLM(
        subject=args.subject,
        session=args.session,
        root=args.root or Path("figures/glm") / f"{args.subject}_{args.session}",
    )
    with _artifact_lock(model.root):
        if args.command == "prepare":
            if args.frame_times is not None:
                model.frame_times = args.frame_times
            model.prepare()
            return
        if args.command == "figures":
            model.figures(args.units, args.output_dir, args.format)
            return
        getattr(model, args.command)(args.units, args.workers)


@dataclass
class PoissonGLM:
    """Default artifact layout and the commands that produce a fitted GLM."""

    subject: str = "GRB006"
    session: str = "20240821_121447"
    root: Path = Path("figures/glm/GRB006_20240821_121447")

    @property
    def frame_times(self) -> Path:
        return self._frame_times or self.root / "frame_times.npy"

    @frame_times.setter
    def frame_times(self, value: Path) -> None:
        self._frame_times = value

    _frame_times: Path | None = None

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
            print(f"Reused prepared windows: {self.windows}", flush=True)
        else:
            write_stimulus_windows(
                self.subject, self.session, self.windows, self.frame_times
            )
            print(f"Completed prepared windows: {self.windows}", flush=True)
        if self.video.exists():
            print(f"Reused motion-energy features: {self.video}", flush=True)
        else:
            write_motion_energy_features(self.windows, self.video)
            print(f"Completed motion-energy features: {self.video}", flush=True)
        if _common_design_is_current(self.design):
            print(f"Reused common design: {self.design}", flush=True)
        else:
            prepare_common_design(self.windows, self.video, self.design)
            print(f"Completed common design: {self.design}", flush=True)

    def fit(self, units: str = "sample", workers: int | None = None) -> None:
        """Compare PC counts, cross-validate 10 PCs, then refit on every trial.

        One command keeps the coefficients and comparisons in the same run.
        """
        directory = self.fit_dir(units)
        fit_models(self.windows, self.design, units, directory, workers)
        with (directory / "validation_summary.json").open() as handle:
            components = int(json.load(handle)["selected_video_components"])
        crossvalidate(self.windows, self.design, units, directory, components, workers)
        refit_final(self.windows, self.design, units, directory, components, workers)

    def unique(self, units: str = "sample", workers: int | None = None) -> None:
        from thesis.ephys.analyses.glm_unique import run_unique

        run_unique(self.windows, self.design, self.fit_dir(units), units, workers)

    def figures(
        self,
        units: str = "all",
        output_dir: Path | None = None,
        formats: str = "both",
    ) -> None:
        from thesis.ephys.analyses.glm_figures import make_figures

        make_figures(
            self.windows,
            self.design,
            self.fit_dir(units),
            output_dir,
            ("pdf", "png") if formats == "both" else (formats,),
        )


if __name__ == "__main__":
    main()
