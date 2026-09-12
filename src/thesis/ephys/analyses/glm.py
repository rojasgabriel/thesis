"""Fit a Poisson GLM to identify variables that predict V1 spiking.

Scientific comparison
---------------------
For GRB006 session 20240821_121447, predict V1 spikes from flashes, a
center-poke kernel truncated at the first flash, peri-exit movement, pre-response
choice side, and additive video motion-energy PCs. Bins after response entry are
excluded. Whole trials are split randomly 60/20/20. No spike history, session
drift, go cue, outcome, or punishment terms. Validation selects the motion-energy
PC count. The 12-unit test set never scores held-out trials.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from damn import fit as damn_fit
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
# Trimmed from logspace(-12, 3, 16) after the first all-unit run: validation
# loss was flat below 1e-7 and saturated above 1e-2, so eleven of sixteen
# points never won. select_alpha_per_unit warns if any unit lands on an edge.
ALPHA_GRID = tuple(np.logspace(-9, -1, 9))
VIDEO_BASIS_COLUMNS = 3
TEST_UNITS = 12
MAX_ITER = 500
TOL = 1e-7
DAMN_MAX_EPOCHS = 100
DAMN_PATIENCE = 10
FIT_SEED = 20260911


# DAMN averages its data loss over targets but sums the L2 penalty over them,
# so one alpha bites N times harder when N units are fit together. Verified
# against sklearn to ~1e-5 in the coefficients for N in 1, 2, 4, 8.
def sklearn_equivalent_alpha(alpha: float, n_targets: int) -> float:
    """Return the sklearn penalty matching DAMN's alpha for an N-target fit."""
    return 2.0 * n_targets * alpha


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


def resolve_device():
    """Return DAMN's accelerator, requiring the MPS-capable checkout."""
    resolver = getattr(damn_fit, "_resolve_device", None)
    if resolver is None:
        raise RuntimeError(
            "The active DAMN install has no device support. Run with "
            "PYTHONPATH=/Users/gabriel/lib/damn-mls-implementation."
        )
    return resolver(None)


def build_counts_matrix(alignments: np.ndarray, units) -> np.ndarray:
    """Stack every unit's 1 ms counts into one bins-by-units response matrix."""
    columns = [
        build_unit_counts(alignments, np.asarray(spikes, dtype=float))
        for spikes in units["spike_times_s"]
    ]
    counts = np.column_stack(columns).astype(np.float32)
    if counts.ndim != 2 or counts.shape[1] != len(units):
        raise ValueError("Counts matrix must have one column per unit.")
    return counts


def damn_rate(design: np.ndarray, weights: np.ndarray, intercept: np.ndarray):
    """Return DAMN's clamped conditional rate and the count of clamped bins."""
    eta = design @ weights + intercept
    clamped = int(np.count_nonzero(eta > damn_fit.CLAMP))
    return np.exp(np.minimum(eta, damn_fit.CLAMP)), clamped


def fit_damn(
    X: np.ndarray,
    Y: np.ndarray,
    val_inds: np.ndarray | None,
    alpha: float | np.ndarray,
    device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Fit every target at once on the shared design; return W, b, per-target loss."""
    torch.manual_seed(FIT_SEED)
    chatter = io.StringIO()
    with contextlib.redirect_stdout(chatter):
        weights, intercept, _, _, _, _, _, val_loss = damn_fit.fit_poisson_glm_lbfgs(
            np.ascontiguousarray(X, dtype=np.float32),
            np.ascontiguousarray(Y, dtype=np.float32),
            alpha=alpha,
            max_epochs=DAMN_MAX_EPOCHS,
            early_stopping="train",
            patience=DAMN_PATIENCE,
            tol=TOL,
            print_every=10**6,
            seed=FIT_SEED,
            device=device,
            per_target_loss=True,
            val_inds=val_inds,
        )
    if not np.isfinite(weights).all() or not np.isfinite(intercept).all():
        raise RuntimeError("DAMN returned non-finite parameters.")
    epochs = re.search(r"at epoch (\d+)", chatter.getvalue())
    return (
        weights,
        intercept,
        val_loss,
        int(epochs.group(1)) if epochs else DAMN_MAX_EPOCHS,
    )


def select_alpha_per_unit(
    X: np.ndarray, Y: np.ndarray, val_inds: np.ndarray, device
) -> tuple[np.ndarray, np.ndarray]:
    """Walk the penalty grid once and take each unit's best validation loss."""
    print(
        f"       {'alpha':>9}  {'mean val loss':>14}  {'best':>5}  {'epochs':>6}  {'time':>6}"
    )
    losses = []
    for alpha in ALPHA_GRID:
        start = time.perf_counter()
        _, _, val_loss, epochs = fit_damn(X, Y, val_inds, float(alpha), device)
        if val_loss is None:
            raise RuntimeError("DAMN returned no per-target validation loss.")
        losses.append(np.asarray(val_loss, dtype=float))
        mean = losses[-1].mean()
        marker = "*" if mean <= min(item.mean() for item in losses) else ""
        print(
            f"       {alpha:>9.0e}  {mean:>14.3f}  {marker:>5}  {epochs:>6}  "
            f"{time.perf_counter() - start:>5.0f}s",
            flush=True,
        )
    grid = np.asarray(ALPHA_GRID, dtype=float)
    stacked = np.stack(losses)
    best = np.argmin(stacked, axis=0)
    edge = int(np.count_nonzero((best == 0) | (best == len(grid) - 1)))
    if edge:
        print(
            f"       WARNING: {edge} of {len(best)} units chose a grid endpoint; "
            "widen ALPHA_GRID.",
            flush=True,
        )
    return grid[best], stacked[best, np.arange(len(best))]


def fit_models(windows: Path, design: Path, unit_set: str, output_dir: Path) -> None:
    """Fit all units together with DAMN; score held-out trials for the all-unit run."""
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
    units = units.iloc[
        test_unit_indices(len(units)) if unit_set == "test" else np.arange(len(units))
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device()
    print(
        f"Fitting {len(units)} units together on {device}: "
        f"{int(train.sum()):,} train and {int(validation.sum()):,} validation rows.",
        flush=True,
    )

    counts = build_counts_matrix(prepared["alignments"], units)
    if np.any(counts[train].sum(axis=0) <= 0):
        raise ValueError("Every unit must have at least one training spike.")
    base_columns = int(design_metadata["base_columns"])

    # One design for every unit, so alpha selection walks the grid once per block.
    fit_rows = np.flatnonzero(train | validation)
    val_inds = np.flatnonzero(np.isin(fit_rows, np.flatnonzero(validation)))
    train_mean = counts[train].mean(axis=0)

    widths = _model_widths(base_columns)
    selection: dict[str, dict] = {}
    started = time.perf_counter()
    for position, (width_name, columns) in enumerate(widths, start=1):
        label = "baseline" if width_name == "baseline" else f"{width_name} video PCs"
        print(
            f"\n[{position}/{len(widths)}] {label} - {columns} columns, "
            f"{len(ALPHA_GRID)} penalties",
            flush=True,
        )
        X = np.asarray(common[np.ix_(fit_rows, np.arange(columns))])
        alpha, _ = select_alpha_per_unit(X, counts[fit_rows], val_inds, device)
        weights, intercept, _, _ = fit_damn(
            X[np.isin(np.arange(len(fit_rows)), val_inds, invert=True)],
            counts[train],
            None,
            alpha,
            device,
        )
        rate, clamped = damn_rate(
            np.asarray(common[np.ix_(np.flatnonzero(validation), np.arange(columns))]),
            weights,
            intercept,
        )
        selection[width_name] = {
            "columns": columns,
            "alpha": alpha.tolist(),
            "clamped_validation_bins": clamped,
            "validation": [
                poisson_metrics(
                    counts[validation][:, unit], rate[:, unit], float(train_mean[unit])
                )
                for unit in range(len(units))
            ],
        }
        median_deviance = float(
            np.median(
                [
                    item["deviance_explained"]
                    for item in selection[width_name]["validation"]
                ]
            )
        )
        print(
            f"       -> median alpha {np.median(alpha):.0e}, "
            f"median validation D2 {median_deviance:.4f}, "
            f"{clamped} clamped bins, {(time.perf_counter() - started) / 60:.1f} min elapsed",
            flush=True,
        )
        del X

    mean_validation_deviance = {
        str(count): float(
            np.mean(
                [
                    item["deviance_explained"]
                    for item in selection[str(count)]["validation"]
                ]
            )
        )
        for count in VIDEO_COMPONENT_COUNTS
    }
    selected_components = max(
        VIDEO_COMPONENT_COUNTS, key=lambda count: mean_validation_deviance[str(count)]
    )
    summary: dict = {
        "backend": "damn",
        "device": str(device),
        "unit_set": unit_set,
        "units": len(units),
        "test_scored": unit_set == "all",
        "alpha_grid": list(ALPHA_GRID),
        "mean_validation_deviance_explained": mean_validation_deviance,
        "selected_video_components": selected_components,
        "fit_rows": {
            "train": int(train.sum()),
            "validation": int(validation.sum()),
            "test": int(test.sum()),
        },
    }

    for position, unit in enumerate(units.itertuples(index=False)):
        record = {
            "unit_id": int(unit.unit_id),
            "depth": float(unit.depth),
            "training_mean_count": float(train_mean[position]),
            "spikes_in_selection_bins": int(
                counts[train | validation][:, position].sum()
            ),
            "baseline": {
                "alpha": selection["baseline"]["alpha"][position],
                "validation": selection["baseline"]["validation"][position],
            },
            "plus_video": {
                str(count): {
                    "alpha": selection[str(count)]["alpha"][position],
                    "validation": selection[str(count)]["validation"][position],
                }
                for count in VIDEO_COMPONENT_COUNTS
            },
        }
        _write_json_atomic(
            output_dir / f"unit_{int(unit.unit_id)}_validation.json", record
        )

    if unit_set == "all":
        fit = train | validation
        fit_mean = counts[fit].mean(axis=0)
        finals = {}
        for width_name, columns in (
            ("baseline", base_columns),
            (
                "plus_video",
                base_columns + VIDEO_BASIS_COLUMNS * selected_components,
            ),
        ):
            key = "baseline" if width_name == "baseline" else str(selected_components)
            X = np.asarray(common[np.ix_(np.flatnonzero(fit), np.arange(columns))])
            weights, intercept, _, _ = fit_damn(
                X, counts[fit], None, np.asarray(selection[key]["alpha"]), device
            )
            rate, clamped = damn_rate(
                np.asarray(common[np.ix_(np.flatnonzero(test), np.arange(columns))]),
                weights,
                intercept,
            )
            finals[width_name] = (
                weights,
                intercept,
                rate,
                clamped,
                selection[key]["alpha"],
            )
            del X

        for position, unit in enumerate(units.itertuples(index=False)):
            record = {
                "unit_id": int(unit.unit_id),
                "depth": float(unit.depth),
                "fit_mean_count": float(fit_mean[position]),
            }
            for width_name, (
                weights,
                intercept,
                rate,
                clamped,
                alpha,
            ) in finals.items():
                block = {
                    "alpha": alpha[position],
                    "clamped_test_bins": clamped,
                    "intercept": float(intercept[position]),
                    "coefficients": weights[:, position].tolist(),
                    "test": poisson_metrics(
                        counts[test][:, position],
                        rate[:, position],
                        float(fit_mean[position]),
                    ),
                }
                if width_name == "plus_video":
                    block["components"] = selected_components
                record[width_name] = block
            _write_json_atomic(
                output_dir / f"unit_{int(unit.unit_id)}_test.json", record
            )

        differences = np.asarray(
            [
                poisson_metrics(
                    counts[test][:, position],
                    finals["plus_video"][2][:, position],
                    float(fit_mean[position]),
                )["bits_per_spike"]
                - poisson_metrics(
                    counts[test][:, position],
                    finals["baseline"][2][:, position],
                    float(fit_mean[position]),
                )["bits_per_spike"]
                for position in range(len(units))
            ]
        )
        summary["test_video_delta_bits_per_spike"] = {
            "median": float(np.median(differences)),
            "q25": float(np.quantile(differences, 0.25)),
            "q75": float(np.quantile(differences, 0.75)),
        }

    _write_json_atomic(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


def _model_widths(base_columns: int) -> list[tuple[str, int]]:
    """Return the baseline and each motion-energy width as (name, column count)."""
    return [("baseline", base_columns)] + [
        (str(count), base_columns + VIDEO_BASIS_COLUMNS * count)
        for count in VIDEO_COMPONENT_COUNTS
    ]


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
        ("check", "Compare the DAMN fit against sklearn (throwaway)"),
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

    def check(self, units: str = "test") -> None:
        from thesis.ephys.analyses.glm_check import run_check

        run_check(self.windows, self.design, self.root / "damn_sklearn_check.json")

    def figures(self, units: str = "all") -> None:
        from thesis.ephys.analyses.glm_figures import make_figures

        make_figures(self.windows, self.design, self.fit_dir(units))


if __name__ == "__main__":
    main()
