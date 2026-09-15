import numpy as np
from damn.alignment import construct_timebins

from thesis.ephys.analyses.glm import (
    HISTORY_COLUMNS,
    build_unit_counts,
    build_unit_history,
    build_video_component_design,
    cross_validation_partitions,
    flash_events,
    spike_history_basis,
    task_temporal_bases,
)
from thesis.ephys.analyses.glm_figures import (
    binned_rates,
    fitted_kernels,
    fitted_trial_contributions,
)
from thesis.ephys.analyses.glm_unique import block_slices
from thesis.ephys.preprocessing.prepare_glm import BINWIDTH_S, POST_S, PRE_S


def test_cross_validation_holds_out_each_trial_once() -> None:
    partitions = cross_validation_partitions(20, 2, np.ones(40, dtype=bool))
    held_out = np.stack([partition["all"] for partition in partitions])
    assert np.all(held_out.sum(axis=0) == 1)
    assert all(
        not np.any(partition["fit"] & partition["all"]) for partition in partitions
    )


def test_flash_interactions_use_symmetric_contrasts() -> None:
    flashes, state, first = flash_events(
        {
            "stim_pulse_times_s": [[1.0, 1.1, 1.2, 1.3], [2.0, 2.2]],
            "center_entry_s": [0.9, 1.9],
            "center_exit_s": [1.2, 2.1],
            "response_port_entry_s": [1.3, 2.3],
        }
    )
    np.testing.assert_array_equal(flashes, [1.0, 1.1, 1.2, 1.3, 2.0, 2.2])
    np.testing.assert_array_equal(state, [-0.5, -0.5, 0.5, 0.5, -0.5, 0.5])
    np.testing.assert_array_equal(first, [0.5, -0.5, -0.5, -0.5, 0.5, -0.5])


def test_video_pc_is_one_contemporaneous_column() -> None:
    alignments = np.array([10.0])
    relative_times, _, _ = construct_timebins(PRE_S, POST_S, BINWIDTH_S)
    frame_times = alignments[0] + np.array([-0.2, 0.0, 0.2, 3.0])
    scores = np.array([-1.0, 0.0, 1.0, 2.0])
    design = build_video_component_design(alignments, frame_times, scores)
    expected = np.interp(alignments[0] + relative_times, frame_times, scores)
    assert design.shape == (len(relative_times), 1)
    np.testing.assert_allclose(design[:, 0], expected)


def test_fitted_kernels_reverse_design_standardization() -> None:
    bases = task_temporal_bases()
    manifest = [
        {"name": name, "columns": basis.basis.shape[1]} for name, basis in bases.items()
    ]
    task_columns = sum(basis.basis.shape[1] for basis in bases.values())
    components = 2
    common_columns = task_columns + components
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


def test_fitted_trial_contributions_reproduce_the_linear_predictor() -> None:
    manifest = [
        {"name": "visual_flash", "group": "sensory", "columns": 2},
        {"name": "flash_pre_post_withdrawal", "group": "sensory", "columns": 2},
        {"name": "withdrawal", "group": "task", "columns": 1},
    ]
    common_columns = 6
    design = (
        np.arange(4 * (common_columns + HISTORY_COLUMNS), dtype=float).reshape(4, -1)
        / 10
    )
    coefficients = np.linspace(-0.2, 0.3, design.shape[1])
    result = {
        "components": 1,
        "intercept": -2.0,
        "coefficients": coefficients,
    }
    contributions, linear_predictor, expected_count, error = fitted_trial_contributions(
        design,
        result,
        {"base_columns": 5, "task_manifest": manifest},
    )

    np.testing.assert_allclose(
        contributions["visual"], design[:, :4] @ coefficients[:4]
    )
    np.testing.assert_allclose(contributions["task"], design[:, 4] * coefficients[4])
    np.testing.assert_allclose(linear_predictor, -2 + design @ coefficients)
    np.testing.assert_allclose(expected_count, np.exp(linear_predictor))
    assert error < 1e-12


def test_unique_motion_block_has_one_column_per_pc() -> None:
    manifest = [
        {"name": "visual_flash", "columns": 2},
        {"name": "flash_pre_post_withdrawal", "columns": 2},
        {"name": "withdrawal", "columns": 1},
    ]
    groups = block_slices(
        {"task_columns": 5, "base_columns": 5, "task_manifest": manifest}, 10
    )
    assert groups["video"] == slice(5, 15)
    assert groups["history"] == slice(15, 15 + HISTORY_COLUMNS)


def test_final_design_has_72_columns_with_25_video_pcs() -> None:
    bases = task_temporal_bases()
    assert list(bases) == [
        "visual_flash",
        "flash_pre_post_withdrawal",
        "flash_first_later",
        "initiation",
        "withdrawal",
        "response_entry",
    ]
    task_columns = sum(basis.basis.shape[1] for basis in bases.values())
    assert task_columns == 37
    assert task_columns + 25 + HISTORY_COLUMNS == 72


def test_binned_rates_convert_counts_to_spikes_per_second() -> None:
    values = np.ones((2, 30))
    rate, valid = binned_rates(values, np.ones_like(values, dtype=bool))
    np.testing.assert_allclose(rate, 1000)
    np.testing.assert_array_equal(valid, True)


def test_spike_history_starts_one_bin_after_each_spike() -> None:
    alignments = np.array([10.0])
    spikes = alignments[0] + np.array([0.0001, 0.05051, 0.1009])
    counts = build_unit_counts(alignments, spikes)
    history = build_unit_history(alignments, spikes)
    np.testing.assert_array_equal(history[1:, 0], counts[:-1])


if __name__ == "__main__":
    test_cross_validation_holds_out_each_trial_once()
    test_flash_interactions_use_symmetric_contrasts()
    test_video_pc_is_one_contemporaneous_column()
    test_fitted_kernels_reverse_design_standardization()
    test_fitted_trial_contributions_reproduce_the_linear_predictor()
    test_unique_motion_block_has_one_column_per_pc()
    test_final_design_has_72_columns_with_25_video_pcs()
    test_binned_rates_convert_counts_to_spikes_per_second()
    test_spike_history_starts_one_bin_after_each_spike()
