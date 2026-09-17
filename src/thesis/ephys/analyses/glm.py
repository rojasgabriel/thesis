"""Fit a Poisson GLM to identify variables that predict V1 spiking.

Scientific comparison
---------------------
For GRB006 session 20240821_121447, predict V1 spikes from a common flash
kernel, a pre/post-withdrawal contrast, a synaptic-resource flash-adaptation
term, an initiation kernel truncated at the first flash, peri-withdrawal
movement, response entry,
validation-selected additive video motion-energy PCs, and each unit's own
strictly past spike history. Bins after response entry are excluded. Whole
trials are split randomly 60/20/20. No response-side, coupling, session-drift,
go-cue, outcome, or punishment terms. Unit 212 is excluded because its fitted
spike-history term extrapolates unstably across held-out folds. Held-out
performance is the ten-fold cross-validated deviance, so every trial is scored
once. A 20-unit random sample is available for quick runs.
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
EXCLUDED_UNIT_IDS = {212: "unstable out-of-fold spike-history extrapolation"}
# The optimum for this design sits near 3e-4, so a fixed 1e-6..1e4 grid brackets
# it with a decade of margin on each side. Penalties that fail to converge (weak
# ones hit the iteration limit, strong ones collapse to zero) are dropped per
# unit; widen the grid if a unit's best penalty lands on an endpoint.
ALPHAS = tuple(np.logspace(-6, 4, 11))
HISTORY_COLUMNS = 10
SAMPLE_UNITS = 20
SAMPLE_SEED = 20260914
# Short-term synaptic-depression parameters for the flash-adaptation covariate:
# each flash spends fraction ADAPT_U of the available resource, which recovers
# toward full with time constant ADAPT_TAU_RECOVER_S between flashes.
ADAPT_U = 0.5
ADAPT_TAU_RECOVER_S = 0.2
# Per-worker state, filled by _init_worker.
_SHARED: dict = {}


def _cached(output: Path) -> dict | None:
    """Return a saved per-unit record if one exists, so a killed run resumes.

    Reuse is purely by presence: a record is trusted whenever its file exists.
    Change an input (the design, tau, the fit logic) and the old records are no
    longer valid, so delete the fit directory's checkpoints before rerunning.
    """
    if not output.exists():
        return None
    with output.open() as handle:
        return json.load(handle)


CV_FOLDS = 10
CV_SEED = 20260913
CV_INNER_FRACTION = 0.25
MAX_ITER = 500


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
        "visual_flash": flash,
        "flash_pre_post_withdrawal": flash,
        "flash_adaptation": flash,
        "initiation": poke,
        "withdrawal": peri_exit,
        "response_entry": pre_response,
    }


def flash_resource(modeled: np.ndarray, u: float, tau_recover_s: float) -> np.ndarray:
    """Available synaptic resource at each flash, from short-term depression.

    A resource pool starts full (R = 1). Each flash spends fraction `u` of what
    is available, then the pool recovers toward 1 with time constant
    `tau_recover_s` before the next flash. Returns R just before each flash: the
    first flash sees a full pool (R = 1, the strongest response) and later
    flashes in a dense train see a depleted pool (graded adaptation).

    This is the Tsodyks-Markram depression model (Tsodyks & Markram, 1997, PNAS),
    driven by stimulus flashes.
    """
    resource = np.ones(len(modeled))
    available = 1.0
    for position in range(len(modeled)):
        resource[position] = available
        available *= 1 - u
        if position + 1 < len(modeled):
            gap = modeled[position + 1] - modeled[position]
            available = 1 - (1 - available) * np.exp(-gap / tau_recover_s)
    return resource


def flash_events(trials) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return modeled flashes, the pre/post state contrast, and the raw resource.

    The resource is the available-pool trace from `flash_resource` (R = 1 at the
    first flash, lower for adapted later flashes). `build_task_design` centers it.
    """
    times, state_values, resource_values = [], [], []
    for flashes, center_entry, center_exit, response_entry in zip(
        trials["stim_pulse_times_s"],
        trials["center_entry_s"],
        trials["center_exit_s"],
        trials["response_port_entry_s"],
        strict=True,
    ):
        flashes = np.asarray(flashes, dtype=float)
        modeled = flashes[(flashes >= center_entry) & (flashes <= response_entry)]
        if not len(modeled):
            raise ValueError("Every modeled trial must contain at least one flash.")
        times.append(modeled)
        state_values.append(np.where(modeled < center_exit, -0.5, 0.5))
        resource_values.append(flash_resource(modeled, ADAPT_U, ADAPT_TAU_RECOVER_S))
    return (
        _flatten_events(times),
        _flatten_events(state_values),
        _flatten_events(resource_values),
    )


def build_task_design(
    alignments: np.ndarray, trials
) -> tuple[DesignMatrix, list[dict]]:
    """Build sensory and task regressors with the accepted DAMN interface."""
    bases = task_temporal_bases()
    flashes, state_values, resource_raw = flash_events(trials)
    # Center the resource so the common flash kernel stays the average response.
    # A first flash sees a full pool (R = 1); adapted later flashes see less.
    # Store two representative centered levels for the kernel figure.
    adapt_center = float(resource_raw.mean()) if len(resource_raw) else 0.0
    adapt_values = resource_raw - adapt_center
    later = resource_raw[resource_raw < 1.0]
    adapt_levels = {
        "first": 1.0 - adapt_center,
        "late": (float(later.mean()) - adapt_center)
        if len(later)
        else 1.0 - adapt_center,
    }
    specifications = [
        (
            "visual_flash",
            flashes,
            None,
            bases["visual_flash"],
            "sensory",
            "each measured flash from initiation through response entry",
        ),
        (
            "flash_pre_post_withdrawal",
            flashes,
            state_values,
            bases["flash_pre_post_withdrawal"],
            "sensory",
            "pre-withdrawal=-0.5, post-withdrawal=+0.5",
        ),
        (
            "flash_adaptation",
            flashes,
            adapt_values,
            bases["flash_adaptation"],
            "sensory",
            f"centered depression resource, u={ADAPT_U}, "
            f"tau_recover={ADAPT_TAU_RECOVER_S * 1000:.0f} ms",
        ),
        (
            "initiation",
            trials["center_entry_s"].to_numpy(dtype=float),
            None,
            bases["initiation"],
            "task",
            "initiation truncated at the first flash",
        ),
        (
            "withdrawal",
            trials["center_exit_s"].to_numpy(dtype=float),
            None,
            bases["withdrawal"],
            "task",
            "one withdrawal per completed trial",
        ),
        (
            "response_entry",
            trials["response_port_entry_s"].to_numpy(dtype=float),
            None,
            bases["response_entry"],
            "task",
            "every response; the side-independent part",
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
    initiation = design.regressors["initiation"]
    rows_per_trial = initiation.X.shape[0] // len(alignments)
    centers, _, _ = construct_timebins(PRE_S, POST_S, BINWIDTH_S)
    truncated = np.asarray(initiation.X).reshape(len(alignments), rows_per_trial, -1)
    truncated[:, np.asarray(centers) >= 0] = 0
    initiation._X = truncated.reshape(-1, initiation.X.shape[1])
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
    for item in manifest:
        if item["name"] == "flash_adaptation":
            item["adapt_u"] = ADAPT_U
            item["adapt_tau_recover_s"] = ADAPT_TAU_RECOVER_S
            item["adapt_levels_centered"] = adapt_levels
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


def analysis_unit_indices(units, unit_set: str) -> np.ndarray:
    """Return requested eligible rows after explicit technical exclusions."""
    unit_ids = units["unit_id"].to_numpy(dtype=int)
    keep = np.flatnonzero(~np.isin(unit_ids, list(EXCLUDED_UNIT_IDS)))
    excluded = sorted(set(unit_ids) & set(EXCLUDED_UNIT_IDS))
    for unit_id in excluded:
        print(
            f"Excluded unit {unit_id}: {EXCLUDED_UNIT_IDS[unit_id]}.",
            flush=True,
        )
    return keep[sample_unit_indices(len(keep))] if unit_set == "sample" else keep


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
    alphas: tuple[float, ...] = ALPHAS,
) -> tuple[PoissonRegressor, dict]:
    """Select the L2 penalty with the lowest validation loss on a fixed grid.

    Penalties are fit from strong to weak, each warm-started from the nearest
    already-fit model, with a cold retry when the warm start fails. A penalty
    that still fails to converge is dropped (weak ones hit the iteration limit,
    strong ones collapse to zero coefficients); only a unit with no converged
    penalty raises. Validation-loss ties break toward the stronger penalty.
    """
    candidates = sorted({float(alpha) for alpha in alphas}, reverse=True)
    if not candidates or candidates[-1] <= 0:
        raise ValueError("Alpha candidates must be positive.")
    models: dict[float, PoissonRegressor] = {}
    losses: dict[float, float] = {}
    for alpha in candidates:
        warm_model = None
        if models:
            nearest = min(models, key=lambda value: abs(np.log(value / alpha)))
            warm_model = copy.deepcopy(models[nearest])
        try:
            model = fit_poisson_at_alpha(X_train, y_train, alpha, warm_model)
        except RuntimeError:
            if warm_model is None:
                continue
            try:
                model = fit_poisson_at_alpha(X_train, y_train, alpha)
            except RuntimeError:
                continue
        models[alpha] = model
        losses[alpha] = poisson_nll(y_validation, model.predict(X_validation))
    if not models:
        raise RuntimeError("No penalty on the grid produced a converged fit.")
    ordered = sorted(losses)
    minimum = min(losses.values())
    tied = [
        alpha
        for alpha in ordered
        if np.isclose(losses[alpha], minimum, rtol=1e-10, atol=1e-12)
    ]
    best = max(tied)
    path = {
        "alphas": ordered,
        "validation_mean_nll": [losses[alpha] for alpha in ordered],
        "best_alpha": best,
        "best_index": ordered.index(best),
        "best_at_endpoint": best in (ordered[0], ordered[-1]),
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
    """Write JSON through a partial file so a kill cannot leave a half record.

    Resumability comes from `_cached` skipping units whose record already
    exists, so this always replaces the target rather than guarding against it.
    """
    partial = path.with_name(f"{path.name}.partial")
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
        "adapt_u": ADAPT_U,
        "adapt_tau_recover_s": ADAPT_TAU_RECOVER_S,
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
        and metadata.get("adapt_u") == ADAPT_U
        and metadata.get("adapt_tau_recover_s") == ADAPT_TAU_RECOVER_S
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
    """Refit one unit on every valid trial; return the canonical model record.

    This is the model the figures and the block analysis read. It is not
    scored here: its held-out performance is the cross-validated deviance,
    carried over from the fold records.
    """
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


def _run_one(task, job) -> tuple[dict, bool]:
    """Reuse a saved per-unit record, or compute one, persist it, and return it.

    Returns (record, reused). Reuse is by file presence (`_cached`), so a rerun
    skips finished units, while a failed unit writes nothing and is retried.
    """
    output = Path(job["output"])
    cached = _cached(output)
    if cached is not None:
        return cached, True
    record = task(job)
    _write_json_atomic(output, record)
    return record, False


def run_over_units(
    task,
    jobs: list,
    windows: Path,
    design: Path,
    workers: int | None,
    label: str,
    shuffle_seed: int | None = None,
):
    """Run one task per unit, in parallel when that helps, yielding (record, reused).

    Each unit reuses or writes its own record, so a failed worker costs that
    unit rather than the run. BLAS threading barely helps on this design (8
    threads buy about 1.3x), so workers run single-threaded and the parallelism
    goes across units instead.
    """
    # Leave one core free so the machine stays usable.
    count = min(max(1, workers or (os.cpu_count() or 2) - 1), len(jobs))
    shared_columns = {job.get("common_columns") for job in jobs}
    common_columns = shared_columns.pop() if len(shared_columns) == 1 else None

    def guarded(job, produce):
        try:
            return produce()
        except Exception as error:  # noqa: BLE001
            # One unit's failure should not discard the rest of the run.
            print(
                f"Failed {label} for unit {job['unit_id']}: "
                f"{type(error).__name__}: {error}",
                flush=True,
            )
            return None, False

    if count <= 1:
        _init_worker(windows, design, shuffle_seed, common_columns)
        for job in jobs:
            yield guarded(job, lambda job=job: _run_one(task, job))
        return
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"
    print(f"Running {label} for {len(jobs)} units across {count} workers.", flush=True)
    with futures.ProcessPoolExecutor(
        max_workers=count,
        initializer=_init_worker,
        initargs=(windows, design, shuffle_seed, common_columns),
    ) as pool:
        submitted = [pool.submit(_run_one, task, job) for job in jobs]
        for job, future in zip(jobs, submitted, strict=True):
            yield guarded(job, future.result)


def collect_records(
    label: str,
    task,
    jobs: list,
    windows: Path,
    design: Path,
    workers: int | None,
    *,
    shuffle_seed: int | None = None,
    detail=None,
) -> list[dict]:
    """Drive one task over every unit, print progress, and return the records.

    Successful records are already persisted per unit, so this raises if any
    unit failed and a rerun resumes from what finished. `detail(record)` adds a
    per-unit progress suffix when given.
    """
    records: list[dict] = []
    failures = 0
    for position, (record, reused) in enumerate(
        run_over_units(task, jobs, windows, design, workers, label, shuffle_seed),
        start=1,
    ):
        if record is None:
            failures += 1
            continue
        records.append(record)
        action = "Reused" if reused else "Completed"
        suffix = f"  {detail(record)}" if detail else ""
        print(
            f"{action} {label} {position}/{len(jobs)}: unit {record['unit_id']}{suffix}",
            flush=True,
        )
    if failures:
        raise RuntimeError(
            f"{label} incomplete: {failures}/{len(jobs)} units failed. "
            "Per-unit results already written are safe to reuse; rerun the same command."
        )
    return records


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
    """Score one unit across every fold and return its record."""
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
    unit_rows = analysis_unit_indices(units, unit_set)
    checkpoints = output_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    print(
        f"Starting {CV_FOLDS}-fold cross-validation over {len(trial_split)} trials, "
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
    records = collect_records(
        "cross-validation",
        _fold_task,
        jobs,
        windows,
        design,
        workers,
        detail=lambda record: (
            f"D2 {record['deviance_explained_mean']:.4f} "
            f"+- {record['deviance_explained_sem']:.4f}"
        ),
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
    _write_json_atomic(output_dir / "cross_validation_summary.json", summary)
    print(json.dumps(summary, indent=2))


def _selection_task(job: dict) -> dict:
    """Select each model width's penalty for one unit and return the record."""
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
        unit_id=job["unit_id"],
        depth=job["depth"],
        spikes_in_selection_bins=int(
            counts[_SHARED["train"] | _SHARED["validation"]].sum()
        ),
    )
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
    unit_rows = analysis_unit_indices(units, unit_set)
    checkpoints = output_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    print(
        "Starting model-selection fits: training/validation split, "
        "alpha selection, and motion-energy PC comparison.",
        flush=True,
    )

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
    selections = collect_records(
        "model selection", _selection_task, jobs, windows, design, workers
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
    selected_video_components = max(
        VIDEO_COMPONENT_COUNTS,
        key=lambda count: mean_validation_deviance[str(count)],
    )
    summary = {
        "unit_set": unit_set,
        "units": len(unit_rows),
        "excluded_unit_ids": sorted(EXCLUDED_UNIT_IDS),
        "mean_validation_deviance_explained": mean_validation_deviance,
        "selected_video_components": selected_video_components,
        "video_component_selection_rule": "highest mean validation D2",
        "fit_rows": {
            "train": int(train.sum()),
            "validation": int(validation.sum()),
            "test": int(test.sum()),
        },
    }

    _write_json_atomic(output_dir / "validation_summary.json", summary)
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
    unit_rows = analysis_unit_indices(units, unit_set)
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
    print(
        f"Starting final full-data fits: {len(final_jobs)} units, "
        f"{components} motion-energy PCs.",
        flush=True,
    )
    collect_records(
        "final full-data fit", _final_task, final_jobs, windows, design, workers
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
        if name in {"fit", "unique"}:
            command.add_argument(
                "--workers",
                type=int,
                help="Units to fit at once. Defaults to one less than the core count.",
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
            model.figures(args.units)
            return
        getattr(model, args.command)(args.units, args.workers)


@dataclass
class PoissonGLM:
    """Default artifact layout and the commands that produce a fitted GLM."""

    subject: str = "GRB006"
    session: str = "20240821_121447"
    root: Path = Path("figures/glm/GRB006_20240821_121447")
    frame_times: Path | None = None

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
                self.subject,
                self.session,
                self.windows,
                self.frame_times or self.root / "frame_times.npy",
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
    ) -> None:
        from thesis.ephys.analyses.glm_figures import make_figures

        make_figures(
            self.windows,
            self.design,
            self.fit_dir(units),
        )


if __name__ == "__main__":
    main()
