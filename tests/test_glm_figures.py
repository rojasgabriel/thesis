import numpy as np

from thesis.ephys.analyses.glm import CV_FOLDS, cross_validation_partitions
from thesis.ephys.analyses.glm_figures import select_prediction_examples


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


if __name__ == "__main__":
    test_prediction_examples_are_distinct_and_out_of_fold()
