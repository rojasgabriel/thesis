"""Compare the DAMN fitter against sklearn on one shared design.

Scientific comparison
---------------------
Not an analysis. Fit the 12 depth-spaced test units at the accepted 25
motion-energy-PC width, once with DAMN at its selected per-unit penalty and
once with sklearn at twice that penalty, on identical standardized training
rows. Score both on the trial-held-out validation rows only; test trials are
never touched.

This exists to catch wiring mistakes in the DAMN switch that would be silent
and plausible-looking: a penalty passed at the wrong scale, a transposed
response matrix, the wrong standardization rows, or an ignored eta clamp.

The matched sklearn penalty is 2 * n_units * the DAMN penalty, not 2 *: DAMN
averages its data loss over targets while summing the L2 term over them.
Delete it once the fit is accepted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from thesis.ephys.analyses.glm import (
    VIDEO_BASIS_COLUMNS,
    PoissonGLM,
    _load_windows,
    _split_masks,
    _valid_bin_mask,
    build_counts_matrix,
    damn_rate,
    fit_damn,
    fit_poisson_at_alpha,
    poisson_metrics,
    resolve_device,
    select_alpha_per_unit,
    sklearn_equivalent_alpha,
    test_unit_indices,
)
from thesis.ephys.units import fetch_unit_table

VIDEO_COMPONENTS = 25


def run_check(windows: Path, design_path: Path, output: Path) -> None:
    """Fit both backends on the same rows and report where they disagree."""
    if output.exists():
        raise FileExistsError(output)
    prepared = _load_windows(windows)
    common = np.load(design_path, mmap_mode="r", allow_pickle=False)
    with design_path.with_suffix(".json").open() as handle:
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
    columns = int(metadata["base_columns"]) + VIDEO_BASIS_COLUMNS * VIDEO_COMPONENTS
    device = resolve_device()

    counts = build_counts_matrix(prepared["alignments"], units)
    train_rows = np.flatnonzero(train)
    fit_rows = np.flatnonzero(train | validation)
    val_inds = np.flatnonzero(np.isin(fit_rows, np.flatnonzero(validation)))
    column_index = np.arange(columns)

    alpha, _ = select_alpha_per_unit(
        np.asarray(common[np.ix_(fit_rows, column_index)]),
        counts[fit_rows],
        val_inds,
        device,
    )
    X_train = np.asarray(common[np.ix_(train_rows, column_index)])
    X_validation = np.asarray(common[np.ix_(np.flatnonzero(validation), column_index)])
    weights, intercept, _ = fit_damn(X_train, counts[train], None, alpha, device)
    damn_validation, clamped = damn_rate(X_validation, weights, intercept)

    records: list[dict] = []
    for position, unit in enumerate(units.itertuples(index=False)):
        training_mean = float(counts[train][:, position].mean())
        sklearn_model = fit_poisson_at_alpha(
            X_train,
            counts[train][:, position],
            sklearn_equivalent_alpha(float(alpha[position]), len(units)),
        )
        sklearn_validation = sklearn_model.predict(X_validation)
        damn_metrics = poisson_metrics(
            counts[validation][:, position],
            damn_validation[:, position],
            training_mean,
        )
        sklearn_metrics = poisson_metrics(
            counts[validation][:, position], sklearn_validation, training_mean
        )
        records.append(
            {
                "unit_id": int(unit.unit_id),
                "damn_alpha": float(alpha[position]),
                "sklearn_alpha": sklearn_equivalent_alpha(
                    float(alpha[position]), len(units)
                ),
                "coefficient_max_abs_difference": float(
                    np.max(np.abs(sklearn_model.coef_ - weights[:, position]))
                ),
                "intercept_abs_difference": float(
                    abs(float(sklearn_model.intercept_) - float(intercept[position]))
                ),
                "prediction_correlation": float(
                    np.corrcoef(sklearn_validation, damn_validation[:, position])[0, 1]
                ),
                "deviance_explained_difference": (
                    damn_metrics["deviance_explained"]
                    - sklearn_metrics["deviance_explained"]
                ),
                "damn_validation": damn_metrics,
                "sklearn_validation": sklearn_metrics,
            }
        )
        print(
            f"Checked unit {position + 1} of {len(units)}: "
            f"corr {records[-1]['prediction_correlation']:.6f}",
            flush=True,
        )

    correlations = [item["prediction_correlation"] for item in records]
    summary: dict = {
        "comparison": "DAMN versus sklearn on trial-held-out validation rows",
        "device": str(device),
        "video_components": VIDEO_COMPONENTS,
        "design_columns": columns,
        "units": len(records),
        "clamped_validation_bins": clamped,
        "min_prediction_correlation": float(np.min(correlations)),
        "max_coefficient_difference": float(
            np.max([item["coefficient_max_abs_difference"] for item in records])
        ),
        "max_abs_deviance_explained_difference": float(
            np.max([abs(item["deviance_explained_difference"]) for item in records])
        ),
        "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    model = PoissonGLM()
    parser.add_argument("--windows", type=Path, default=model.windows)
    parser.add_argument("--design", type=Path, default=model.design)
    parser.add_argument(
        "--output", type=Path, default=model.root / "damn_sklearn_check.json"
    )
    args = parser.parse_args()
    run_check(args.windows, args.design, args.output)


if __name__ == "__main__":
    main()
