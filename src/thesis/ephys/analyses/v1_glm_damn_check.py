"""Compare DAMN MPS and sklearn Poisson fits on validation trials.

Scientific comparison
---------------------
For the 12 depth-spaced V1 test units, use DAMN's native per-target alpha
selection on random-split training rows with its internal random 5% split.
Then refit the accepted 25 motion-energy-PC model on all training rows with DAMN
and sklearn. The matched sklearn penalty is twice DAMN's penalty because the two
objectives differ by a factor of two in the L2 term. Compare coefficients and
predictions only on validation rows. Test trials are not scored.

Both fits use the same standardized shared design and Poisson log link. DAMN
clips the linear predictor only above 8. The output reports how often that bound
is active. This is a backend check, not a new model-selection or biological
analysis.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from damn import fit as damn_fit

from thesis.ephys.analyses.v1_glm import (
    VIDEO_BASIS_COLUMNS,
    PoissonGLM,
    _load_windows,
    _split_masks,
    _valid_bin_mask,
    build_unit_counts,
    fit_poisson_at_alpha,
    poisson_metrics,
    test_unit_indices,
)
from thesis.ephys.units import fetch_unit_table

VIDEO_COMPONENTS = 25
SEED = 20260911
ALPHA_GRID = np.logspace(-12, 3, 16)


def _damn_prediction(
    design: np.ndarray, weights: np.ndarray, intercept: np.ndarray
) -> tuple[np.ndarray, int]:
    eta = design @ weights + intercept
    clipped = int(np.count_nonzero(eta > damn_fit.CLAMP))
    return np.exp(np.minimum(eta, damn_fit.CLAMP))[:, 0], clipped


def run_check(args: argparse.Namespace) -> None:
    """Run the fixed-design backend comparison and write one JSON artifact."""
    if args.output.exists():
        raise FileExistsError(args.output)
    if not hasattr(damn_fit, "_resolve_device"):
        raise RuntimeError("The active DAMN checkout does not include MPS support.")
    device = damn_fit._resolve_device(args.device)

    prepared = _load_windows(args.windows)
    common = np.load(args.design, mmap_mode="r", allow_pickle=False)
    with args.design.with_suffix(".json").open() as handle:
        metadata = json.load(handle)
    valid = _valid_bin_mask(prepared)
    train, validation, _ = _split_masks(prepared["split"], valid)

    units = fetch_unit_table(
        prepared["metadata"]["subject_name"],
        prepared["metadata"]["session_name"],
        unit_criteria_id=1,
        stability_param_id=0,
        include_metrics=False,
    )
    units = units.iloc[test_unit_indices(len(units))]
    columns = int(metadata["base_columns"]) + (VIDEO_BASIS_COLUMNS * VIDEO_COMPONENTS)
    records = []
    for position, unit in enumerate(units.itertuples(index=False), start=1):
        unit_id = int(unit.unit_id)
        with (args.fit_dir / f"unit_{unit_id}_validation.json").open() as handle:
            selection = json.load(handle)
        saved = selection["plus_video"][str(VIDEO_COMPONENTS)]
        counts = build_unit_counts(
            prepared["alignments"], np.asarray(unit.spike_times_s, dtype=float)
        )
        design = np.asarray(common[:, :columns])
        fit_mean = float(counts[train].mean())

        np.random.seed(SEED + unit_id)
        torch.manual_seed(SEED + unit_id)
        start = time.perf_counter()
        _, _, selected_alpha, _ = damn_fit.fit_poisson_glm_best_alpha_per_target(
            design[train],
            counts[train, None],
            warm_start=True,
            val_fraction=0.05,
            alpha_grid=ALPHA_GRID,
            max_epochs=100,
            early_stopping="train",
            patience=10,
            tol=1e-7,
            print_every=1000,
            device=device,
        )
        alpha_selection_seconds = time.perf_counter() - start
        damn_alpha = float(selected_alpha[0])
        sklearn_alpha = 2 * damn_alpha

        start = time.perf_counter()
        sklearn_model = fit_poisson_at_alpha(
            design[train], counts[train], sklearn_alpha
        )
        sklearn_seconds = time.perf_counter() - start
        sklearn_prediction = sklearn_model.predict(design[validation])

        torch.manual_seed(SEED + unit_id)
        start = time.perf_counter()
        weights, intercept, train_loss, *_ = damn_fit.fit_poisson_glm_lbfgs(
            design[train],
            counts[train, None],
            alpha=damn_alpha,
            max_epochs=100,
            lbfgs_max_iter=20,
            early_stopping="train",
            patience=10,
            tol=1e-7,
            print_every=1000,
            seed=SEED + unit_id,
            device=device,
        )
        damn_seconds = time.perf_counter() - start
        damn_prediction, clipped_validation = _damn_prediction(
            design[validation], weights, intercept
        )
        _, clipped_training = _damn_prediction(design[train], weights, intercept)

        sklearn_metrics = poisson_metrics(
            counts[validation], sklearn_prediction, fit_mean
        )
        damn_metrics = poisson_metrics(counts[validation], damn_prediction, fit_mean)
        record = {
            "unit_id": unit_id,
            "saved_sklearn_alpha": float(saved["alpha_path"]["best_alpha"]),
            "sklearn_alpha": sklearn_alpha,
            "damn_alpha": damn_alpha,
            "alpha_selection_seconds": alpha_selection_seconds,
            "sklearn_seconds": sklearn_seconds,
            "damn_seconds": damn_seconds,
            "damn_epochs": len(train_loss),
            "coefficient_max_abs_difference": float(
                np.max(np.abs(sklearn_model.coef_ - weights[:, 0]))
            ),
            "intercept_abs_difference": float(
                abs(float(sklearn_model.intercept_) - float(intercept[0]))
            ),
            "prediction_correlation": float(
                np.corrcoef(sklearn_prediction, damn_prediction)[0, 1]
            ),
            "prediction_max_abs_difference": float(
                np.max(np.abs(sklearn_prediction - damn_prediction))
            ),
            "clipped_training_bins": clipped_training,
            "clipped_validation_bins": clipped_validation,
            "sklearn_validation": sklearn_metrics,
            "damn_validation": damn_metrics,
            "deviance_explained_difference": (
                damn_metrics["deviance_explained"]
                - sklearn_metrics["deviance_explained"]
            ),
            "bits_per_spike_difference": (
                damn_metrics["bits_per_spike"] - sklearn_metrics["bits_per_spike"]
            ),
        }
        records.append(record)
        print(
            f"Checked unit {position} of {len(units)}: {unit_id}; "
            f"validation ΔD²={record['deviance_explained_difference']:.2g}",
            flush=True,
        )

    result = {
        "comparison": "DAMN versus sklearn on random-trial validation rows",
        "device": str(device),
        "units": len(records),
        "video_components": VIDEO_COMPONENTS,
        "test_scored": False,
        "alpha_selection": "DAMN per target on a random 5% of training bins",
        "alpha_grid": ALPHA_GRID.tolist(),
        "damn_clamp": damn_fit.CLAMP,
        "sklearn_alpha_equals_damn_alpha_times": 2,
        "selected_damn_alphas": [record["damn_alpha"] for record in records],
        "median_alpha_selection_seconds": float(
            np.median([record["alpha_selection_seconds"] for record in records])
        ),
        "median_sklearn_seconds": float(
            np.median([record["sklearn_seconds"] for record in records])
        ),
        "median_damn_seconds": float(
            np.median([record["damn_seconds"] for record in records])
        ),
        "maximum_abs_deviance_explained_difference": float(
            np.max(
                np.abs([record["deviance_explained_difference"] for record in records])
            )
        ),
        "minimum_prediction_correlation": float(
            np.min([record["prediction_correlation"] for record in records])
        ),
        "clipped_training_bins": int(
            np.sum([record["clipped_training_bins"] for record in records])
        ),
        "clipped_validation_bins": int(
            np.sum([record["clipped_validation_bins"] for record in records])
        ),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "records"}, indent=2
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    model = PoissonGLM()
    parser.add_argument("--windows", type=Path, default=model.windows)
    parser.add_argument("--design", type=Path, default=model.design)
    parser.add_argument("--fit-dir", type=Path, default=model.fit_dir("test"))
    parser.add_argument("--device", default="mps")
    parser.add_argument(
        "--output",
        type=Path,
        default=model.root / "damn_mps_check_me.json",
    )
    run_check(parser.parse_args())


if __name__ == "__main__":
    main()
