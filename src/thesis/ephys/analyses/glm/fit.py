"""Fit a Poisson GLM to identify variables that predict V1 spiking.

Scientific comparison
---------------------
For GRB006 session 20240821_121447, predict V1 spikes from five regressors: a
common flash kernel, a synaptic-resource flash-adaptation term, a
causal post-withdrawal kernel (0 to 100 ms), the first 10 additive video motion-energy PCs, and the
strictly past spike history of each unit. All terms are additive in log rate.
The model has no flash × movement interaction, initiation, response-entry,
response-side, coupling, session-drift, go-cue, outcome, or punishment terms.
The full-design fit that tested and removed four more regressors is in
`docs/glm.md`.

Units: each unit must pass quality criterion set 1 (`unit_criteria_id=1`) and
stability set 0 (`stability_param_id=0`). No sensory-response filter is
applied. Unit 212 is excluded because its fitted spike-history term
extrapolates unstably across held-out folds. `--units sample` uses a fixed
random sample (seed 20260914) of 20 of the remaining units.

Trials and alignment: use completed left or right trials without early
withdrawal. Each trial must have a measured first flash and finite
center-entry, center-exit, and response-entry times. Each window is aligned to
the first measured flash. It covers -99.5 ms to +2539.5 ms in 1-ms bins. Bins
after response entry are excluded.

Baseline: the null model is a constant rate. This rate is the mean count in the
fit rows. In model selection, the fit rows are the training rows. In a
cross-validation fold, the fit rows are all valid rows outside the held-out
fold. D² and bits/spike compare the model with this null.

Split and sampling unit: whole trials are split randomly 60/20/20. The
motion-energy PC count is fixed at 10; the training and validation rows also
score the model without motion energy. Held-out performance is the mean
D² over ten whole-trial cross-validation folds. In each fold, the L2 penalty is
chosen on a random 25% of the remaining trials. A fold with no test spikes for
a unit is skipped. A unit with fewer than two scored folds stops the run. The
sampling unit is the unit. Population results are the median and IQR over units
of each unit's fold mean.
"""

from __future__ import annotations

import concurrent.futures as futures
import copy
import json
import os
import warnings
from pathlib import Path

import numpy as np
from damn.alignment import compute_spike_count
from scipy.special import xlogy
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import PoissonRegressor

from thesis.ephys.analyses.glm.dmat import (
    BINWIDTH_S,
    POST_S,
    PRE_S,
    VIDEO_COMPONENT_COUNTS,
    _design_with_history,
    _load_windows,
    _split_masks,
    _valid_bin_mask,
    build_unit_history,
    training_zscore,
)
from thesis.ephys.units import fetch_unit_table

EXCLUDED_UNIT_IDS = {212: "unstable out-of-fold spike-history extrapolation"}


def load_unit_records(directory: Path, pattern: str) -> list[dict]:
    """Read per-unit JSON records, skipping units excluded from the analysis."""
    records = []
    for path in sorted(directory.glob(pattern)):
        with path.open() as handle:
            record = json.load(handle)
        if int(record["unit_id"]) not in EXCLUDED_UNIT_IDS:
            records.append(record)
    return records


# The optimum for this design sits near 3e-4, so a fixed 1e-6..1e4 grid brackets
# it with a decade of margin on each side. Penalties that fail to converge (weak
# ones hit the iteration limit, strong ones collapse to zero) are dropped per
# unit; widen the grid if a unit's best penalty lands on an endpoint.
ALPHAS = tuple(np.logspace(-6, 4, 11))
SAMPLE_UNITS = 20
SAMPLE_SEED = 20260914
# Per-worker state, filled by _init_worker.
_SHARED: dict = {}
CV_FOLDS = 10
CV_SEED = 20260913
CV_INNER_FRACTION = 0.25
MAX_ITER = 500


def build_unit_counts(alignments: np.ndarray, spike_times: np.ndarray) -> np.ndarray:
    """Return raw 1 ms spike counts on the GLM grid."""
    counts, _, _ = compute_spike_count(
        alignments, spike_times, PRE_S, POST_S, BINWIDTH_S
    )
    return counts.ravel()


def analysis_unit_indices(units, unit_set: str) -> np.ndarray:
    """Return eligible rows after exclusions; `sample` keeps 20 fixed, by depth.

    `sampleN` (N >= 1) is replicate N: 20 other units that are in neither
    `sample` nor another replicate.
    """
    unit_ids = units["unit_id"].to_numpy(dtype=int)
    for unit_id in sorted(set(unit_ids) & set(EXCLUDED_UNIT_IDS)):
        print(f"Excluded unit {unit_id}: {EXCLUDED_UNIT_IDS[unit_id]}.", flush=True)
    keep = np.flatnonzero(~np.isin(unit_ids, list(EXCLUDED_UNIT_IDS)))
    if unit_set == "all":
        return keep
    replicate = int(unit_set.removeprefix("sample") or 0)
    rng = np.random.default_rng(SAMPLE_SEED)
    sample = rng.choice(len(keep), size=SAMPLE_UNITS, replace=False)
    if replicate:
        rest = rng.permutation(np.setdiff1d(np.arange(len(keep)), sample))
        sample = rest[(replicate - 1) * SAMPLE_UNITS : replicate * SAMPLE_UNITS]
    if len(sample) < SAMPLE_UNITS:
        raise ValueError("Not enough eligible units for the requested sample.")
    return keep[np.sort(sample)]


def analysis_units(prepared: dict, unit_set: str) -> list[dict]:
    """Return id, depth, and spike times of each requested eligible unit."""
    units = fetch_unit_table(
        prepared["metadata"]["subject_name"],
        prepared["metadata"]["session_name"],
        unit_criteria_id=1,
        stability_param_id=0,
        include_metrics=False,
    )
    return [
        {
            "unit_id": int(units.iloc[row]["unit_id"]),
            "depth": float(units.iloc[row]["depth"]),
            "spike_times": np.asarray(units.iloc[row]["spike_times_s"], dtype=float),
        }
        for row in analysis_unit_indices(units, unit_set)
    ]


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
) -> tuple[PoissonRegressor, dict]:
    """Select the L2 penalty with the lowest validation loss on a fixed grid.

    Penalties are fit from strong to weak, each warm-started from the nearest
    already-fit model, with a cold retry when the warm start fails. A penalty
    that still fails to converge is dropped (weak ones hit the iteration limit,
    strong ones collapse to zero coefficients); only a unit with no converged
    penalty raises. Validation-loss ties break toward the stronger penalty.
    """
    models: dict[float, PoissonRegressor] = {}
    losses: dict[float, float] = {}
    for alpha in sorted(map(float, ALPHAS), reverse=True):
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


def _write_json_atomic(path: Path, value: dict) -> None:
    """Write JSON through a partial file so a kill cannot leave a half record.

    Resumability comes from `_run_one` skipping units whose record already
    exists, so this always replaces the target rather than guarding against it.
    """
    partial = path.with_name(f"{path.name}.partial")
    partial.unlink(missing_ok=True)
    with partial.open("x") as handle:
        json.dump(value, handle, indent=2)
    partial.replace(path)


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
    # Zero video PCs is the baseline model.
    fits = {}
    for count in (0, *VIDEO_COMPONENT_COUNTS):
        design = _design_with_history(common, history, base_columns + count)
        model, path = fit_poisson_alpha_path(
            design[train], counts[train], design[validation], counts[validation]
        )
        validation_metrics = poisson_metrics(
            counts[validation], model.predict(design[validation]), fit_mean
        )
        fits[count] = {"alpha_path": path, "validation": validation_metrics}
    return {
        "training_mean_count": fit_mean,
        "baseline": fits.pop(0),
        "plus_video": {str(count): fit for count, fit in fits.items()},
    }


def _final_task(job: dict) -> dict:
    """Refit one unit on every valid trial; return the canonical model record.

    The kernel and model-design figures use the coefficients of this model.
    The unique-deviance analysis does not use it; it refits each fold at the
    penalty that fold chose. This model is not scored here. Its held-out
    scores are the cross-validated values from the fold records.
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
    n_trials = len(prepared["alignments"])
    bins_per_trial = len(prepared["relative_centers"])
    partitions = cross_validation_partitions(n_trials, bins_per_trial, valid)
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
        from thesis.ephys.analyses.glm.unique_dev import (
            shuffle_permutations,
            trial_swap_permutation,
        )

        _SHARED["within_trial"] = shuffle_permutations(
            n_trials, bins_per_trial, shuffle_seed, valid=valid
        )
        _SHARED["between_trials"] = trial_swap_permutation(
            n_trials, bins_per_trial, shuffle_seed
        )


def _run_one(task, job) -> tuple[dict, bool]:
    """Reuse a saved per-unit record, or compute one, persist it, and return it.

    Returns (record, reused). Reuse is purely by file presence, so a rerun skips
    finished units and a failed unit writes nothing and is retried. Change an
    input (design, tau, fit logic) and delete the checkpoints before rerunning.
    """
    output = Path(job["output"])
    if output.exists():
        return json.loads(output.read_text()), True
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
    n_trials = len(prepared["alignments"])
    checkpoints = output_dir / "checkpoints"
    jobs = [
        {
            **unit,
            "components": components,
            "common_columns": common_columns,
            "output": str(checkpoints / f"unit_{unit['unit_id']}_folds.json"),
        }
        for unit in analysis_units(prepared, unit_set)
    ]
    checkpoints.mkdir(parents=True, exist_ok=True)
    print(
        f"Starting {CV_FOLDS}-fold cross-validation over {n_trials} trials, "
        f"{len(jobs)} units, {components} motion-energy PCs.",
        flush=True,
    )
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
        "trials": n_trials,
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
    """Select the motion-energy PC count on the validation trials.

    For each unit, fit the baseline model and the model for each motion-energy
    PC count on the training trials. Select each penalty on the validation
    trials. Then select the PC count with the highest mean validation D² across
    units. Write the result to validation_summary.json. This step does not
    score the test trials. Cross-validation (`crossvalidate`) gives the
    held-out scores.
    """
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
    checkpoints = output_dir / "checkpoints"
    jobs = [
        {
            **unit,
            "base_columns": design_metadata["base_columns"],
            "output": str(checkpoints / f"unit_{unit['unit_id']}_validation.json"),
        }
        for unit in analysis_units(prepared, unit_set)
    ]
    checkpoints.mkdir(parents=True, exist_ok=True)
    print(
        "Starting model-selection fits: training/validation split, "
        "alpha selection, and motion-energy PC comparison.",
        flush=True,
    )
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
        "units": len(jobs),
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


def refit_final(
    windows: Path,
    design: Path,
    unit_set: str,
    output_dir: Path,
    components: int,
    workers: int | None = None,
) -> None:
    """Refit every unit on all valid trials, after cross-validation has run.

    The kernel and model-design figures use this model. The unique-deviance
    analysis does not use it. Its penalty is the median of the penalties the
    folds chose, so cross-validation must finish first. Its held-out score
    comes from cross-validation, not from a second, weaker split.
    """
    prepared = _load_windows(windows)
    with design.with_suffix(".json").open() as handle:
        design_metadata = json.load(handle)
    units = analysis_units(prepared, unit_set)
    checkpoints = output_dir / "checkpoints"
    fold_records = {
        int(record["unit_id"]): record
        for record in load_unit_records(checkpoints, "unit_*_folds.json")
    }
    missing = sum(unit["unit_id"] not in fold_records for unit in units)
    if missing:
        raise ValueError(
            f"{missing} units have no cross-validated record; run fit again."
        )
    final_jobs = [
        {
            **unit,
            "components": components,
            "common_columns": int(design_metadata["base_columns"]) + components,
            "alpha": float(
                np.median([f["alpha"] for f in fold_records[unit["unit_id"]]["folds"]])
            ),
            "cross_validated": fold_records[unit["unit_id"]],
            "output": str(checkpoints / f"unit_{unit['unit_id']}_final.json"),
        }
        for unit in units
    ]
    print(
        f"Starting final full-data fits: {len(final_jobs)} units, "
        f"{components} motion-energy PCs.",
        flush=True,
    )
    collect_records(
        "final full-data fit", _final_task, final_jobs, windows, design, workers
    )
    from thesis.ephys.analyses.glm.figures import plot_fit

    plot_fit(output_dir)
