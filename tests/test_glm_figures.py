import numpy as np

from thesis.ephys.analyses.glm import (
    CV_FOLDS,
    HISTORY_COLUMNS,
    VIDEO_BASIS_COLUMNS,
    build_unit_counts,
    build_unit_history,
    cross_validation_partitions,
    spike_history_basis,
    task_temporal_bases,
)
from thesis.ephys.analyses.glm_figures import (
    fitted_kernels,
    select_design_example_result,
    select_prediction_examples,
)


def test_prediction_examples_are_distinct_and_out_of_fold() -> None:
    results = [
        {
            "unit_id": unit_id,
            "folds_scored": CV_FOLDS - (unit_id == 4),
            "cross_validated_deviance_explained": score,
        }
        for unit_id, score in ((1, 0.5), (2, 0.3), (3, 0.2), (4, 0.9))
    ]
    groups = (
        "visual_flash",
        "center_poke",
        "center_exit",
        "response_entry",
        "response_side",
        "video",
    )
    unique_records = []
    for unit_id in range(1, 5):
        scores = {group: 0.01 for group in groups}
        scores["visual_flash"] = {1: 0.9, 2: 0.8, 3: 0.1, 4: 1.0}[unit_id]
        scores["response_entry"] = {1: 0.1, 2: 0.2, 3: 0.7, 4: 1.0}[unit_id]
        unique_records.append(
            {
                "unit_id": unit_id,
                "groups": {
                    group: {"unique_test_deviance_explained": score}
                    for group, score in scores.items()
                },
            }
        )

    examples = select_prediction_examples(results, unique_records)
    assert [item["result"]["unit_id"] for item in examples] == [1, 2, 3]
    assert [item["selection_group"] for item in examples] == [
        "full_model",
        "visual_flash",
        "response_entry",
    ]

    partitions = cross_validation_partitions(20, 2, np.ones(40, dtype=bool))
    held_out = np.stack([partition["all"] for partition in partitions])
    assert np.all(held_out.sum(axis=0) == 1)
    assert all(
        not np.any(partition["fit"] & partition["all"]) for partition in partitions
    )


def test_fitted_kernels_reverse_design_standardization() -> None:
    bases = task_temporal_bases()
    manifest = [
        {"name": name, "columns": basis.basis.shape[1]} for name, basis in bases.items()
    ]
    task_columns = sum(basis.basis.shape[1] for basis in bases.values())
    components = 2
    common_columns = task_columns + VIDEO_BASIS_COLUMNS * components
    coefficients = np.arange(1, common_columns + HISTORY_COLUMNS + 1, dtype=float)
    common_scale = np.full(common_columns, 2.0)
    history_scale = np.full(HISTORY_COLUMNS, 4.0)

    kernels = fitted_kernels(
        {"components": components, "coefficients": coefficients},
        {
            "base_columns": task_columns,
            "task_columns": task_columns,
            "task_manifest": manifest,
            "training_scale": common_scale,
        },
        history_scale,
    )

    first = manifest[0]
    first_stop = first["columns"]
    np.testing.assert_allclose(
        kernels[first["name"]][1],
        bases[first["name"]].basis @ (coefficients[:first_stop] / 2),
    )
    np.testing.assert_allclose(
        kernels["history"][1],
        spike_history_basis().basis @ (coefficients[common_columns:] / 4),
    )


def test_design_example_uses_training_rate_percentile() -> None:
    results = [
        {"unit_id": unit_id, "cross_validated_deviance_explained": 0.1}
        for unit_id in (1, 2, 3, 4, 5)
    ]
    selections = [
        {"unit_id": unit_id, "training_mean_count": rate * 0.001}
        for unit_id, rate in zip((1, 2, 3, 4, 5), (1, 2, 3, 4, 20), strict=True)
    ]
    result, rate = select_design_example_result(results, selections)
    assert result["unit_id"] == 5
    assert rate == 20


def test_spike_history_starts_one_bin_after_each_spike() -> None:
    alignments = np.array([10.0])
    spikes = alignments[0] + np.array([0.0001, 0.05051, 0.1009])
    counts = build_unit_counts(alignments, spikes)
    history = build_unit_history(alignments, spikes)
    np.testing.assert_array_equal(history[1:, 0], counts[:-1])


if __name__ == "__main__":
    test_prediction_examples_are_distinct_and_out_of_fold()
    test_fitted_kernels_reverse_design_standardization()
    test_design_example_uses_training_rate_percentile()
    test_spike_history_starts_one_bin_after_each_spike()
