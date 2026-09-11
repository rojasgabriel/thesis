"""Small check for invalid video timing being rejected before model fitting."""

import unittest

import numpy as np
import pandas as pd

from thesis.ephys.analyses.v1_glm import (
    build_session_drift,
    build_task_design,
    build_unit_design,
    build_video_component_design,
    fit_poisson_at_alpha,
    pilot_indices,
    poisson_metrics,
    spike_history_basis,
    video_temporal_basis,
)
from thesis.ephys.analyses.v1_glm import (
    training_zscore as design_training_zscore,
)
from thesis.ephys.analyses.v1_glm_attribution import (
    apply_row_permutation,
    attribution_slices,
    shuffle_permutations,
)
from thesis.ephys.analyses.v1_glm_prediction import (
    select_representative_result,
    simulate_spike_counts,
    task_kernel_display_values,
    training_rate_and_test_deviance,
)
from thesis.ephys.preprocessing.audit_camera_pulses import select_falling_edges
from thesis.ephys.preprocessing.prepare_v1_glm import (
    trial_bins,
    validate_frame_times,
)
from thesis.ephys.preprocessing.video_svd import training_zscore


class V1GlmTest(unittest.TestCase):
    def test_signed_task_kernels_show_condition_differences(self):
        values = np.array([[-0.2, 0.3]])
        np.testing.assert_array_equal(
            task_kernel_display_values("response_side", values), 2 * values
        )
        np.testing.assert_array_equal(
            task_kernel_display_values("outcome", values), 2 * values
        )
        self.assertIs(task_kernel_display_values("visual_flash", values), values)

    def test_attribution_groups_keep_basis_columns_together(self):
        metadata = {
            "task_columns": 5,
            "base_columns": 7,
            "task_manifest": [
                {"name": "first", "columns": 2},
                {"name": "second", "columns": 3},
            ],
        }
        groups = attribution_slices(metadata, video_components=2)
        self.assertEqual(groups["task"], slice(0, 5))
        self.assertEqual(groups["first"], slice(0, 2))
        self.assertEqual(groups["second"], slice(2, 5))
        self.assertEqual(groups["drift"], slice(5, 7))
        self.assertEqual(groups["video"], slice(7, 13))
        self.assertEqual(groups["history"], slice(13, 23))

    def test_attribution_shuffles_respect_trial_and_split_boundaries(self):
        trial_split = np.array([0, 0, 1, 1])
        within, whole = shuffle_permutations(trial_split, bins_per_trial=5, seed=4)
        row_grid = np.arange(20).reshape(4, 5)
        within_grid = within.reshape(4, 5)
        np.testing.assert_array_equal(
            within_grid // 5, np.repeat(np.arange(4)[:, None], 5, axis=1)
        )
        np.testing.assert_array_equal(np.sort(within_grid, axis=1), row_grid)

        whole_grid = whole.reshape(4, 5)
        source_trials = whole_grid[:, 0] // 5
        np.testing.assert_array_equal(trial_split[source_trials], trial_split)
        self.assertTrue(np.all(source_trials != np.arange(4)))
        np.testing.assert_array_equal(
            whole_grid % 5, np.repeat(np.arange(5)[None, :], 4, axis=0)
        )

        source = np.arange(40).reshape(20, 2)
        destination = np.empty_like(source)
        apply_row_permutation(destination, source, within, chunk_rows=3)
        np.testing.assert_array_equal(destination, source[within])

    def test_representative_unit_is_nearest_population_median(self):
        results = [
            {
                "unit_id": unit_id,
                "plus_video": {"test": {"deviance_explained": score}},
            }
            for unit_id, score in [(30, 0.1), (20, 0.2), (10, 0.3)]
        ]
        selected, median = select_representative_result(results)
        self.assertEqual(selected["unit_id"], 20)
        self.assertAlmostEqual(median, 0.2)

    def test_training_rate_is_paired_by_unit_and_keeps_zero_rate(self):
        results = [
            {
                "unit_id": 2,
                "plus_video": {"test": {"deviance_explained": 0.2}},
            },
            {
                "unit_id": 1,
                "plus_video": {"test": {"deviance_explained": -0.1}},
            },
        ]
        selections = [
            {"unit_id": 1, "training_mean_count": 0.0},
            {"unit_id": 2, "training_mean_count": 0.002},
        ]
        rate, deviance = training_rate_and_test_deviance(results, selections)
        np.testing.assert_array_equal(rate, [2.0, 0.0])
        np.testing.assert_array_equal(deviance, [0.2, -0.1])

    def test_recursive_simulation_replaces_observed_history(self):
        class ZeroCountGenerator:
            def __init__(self):
                self.means = []

            def poisson(self, mean):
                self.means.append(np.asarray(mean).copy())
                return np.zeros_like(mean, dtype=int)

        generator = ZeroCountGenerator()
        simulated = simulate_spike_counts(
            np.array([[1, 0, 0]]),
            np.ones((1, 3)),
            np.array([np.log(2)]),
            generator,
        )
        np.testing.assert_array_equal(simulated, 0)
        np.testing.assert_allclose(np.concatenate(generator.means), [1, 0.5, 1])

    def test_task_design_has_explicit_additive_contrasts(self):
        alignments = np.arange(5, dtype=float) * 4
        trials = pd.DataFrame(
            {
                "stim_pulse_times_s": [[time, time + 0.1] for time in alignments],
                "center_entry_s": alignments - 0.05,
                "go_cue_times_s": [[time + 0.5] for time in alignments],
                "center_exit_s": alignments + 0.6,
                "response_port_entry_s": alignments + 1,
                "response": [-1, 1, -1, 1, -1],
                "rewarded": [True, False, True, False, True],
                "punish_wrong_times_s": [
                    [] if rewarded else [time + 1]
                    for rewarded, time in zip(
                        [True, False, True, False, True], alignments, strict=True
                    )
                ],
            }
        )
        design, manifest = build_task_design(alignments, trials)
        self.assertEqual(design.X.shape, (5 * 2639, 57))
        self.assertEqual(
            [item["name"] for item in manifest][-4:],
            [
                "response_entry",
                "response_side",
                "outcome",
                "wrong_punishment_command",
            ],
        )
        np.testing.assert_array_equal(
            design.regressors["response_side"].event_values, trials["response"]
        )
        np.testing.assert_array_equal(
            design.regressors["outcome"].event_values, [1, -1, 1, -1, 1]
        )

    def test_design_scaling_and_pilot_selection(self):
        values = np.array([[1.0, 10.0], [3.0, 14.0], [101.0, 110.0]])
        scaled, mean, scale = design_training_zscore(
            values, np.array([True, True, False])
        )
        np.testing.assert_array_equal(mean, [2.0, 12.0])
        np.testing.assert_array_equal(scale, [1.0, 2.0])
        np.testing.assert_array_equal(scaled[:2], [[-1.0, -1.0], [1.0, 1.0]])
        np.testing.assert_array_equal(
            pilot_indices(168), [0, 15, 30, 46, 61, 76, 91, 106, 121, 137, 152, 167]
        )

    def test_spike_history_basis_is_strictly_past(self):
        basis = spike_history_basis()
        self.assertEqual(basis.basis.shape, (100, 10))
        self.assertTrue(np.all(np.any(basis.basis != 0, axis=0)))
        np.testing.assert_allclose(basis.basis_time[[0, -1]], [0, 0.099])
        _, history = build_unit_design(np.array([1.0, 5.0]), np.array([1.0]))
        alignment_bin = 99
        np.testing.assert_array_equal(history[alignment_bin], 0)
        self.assertTrue(np.any(history[alignment_bin + 1] != 0))

    def test_session_drift_has_linear_and_quadratic_terms(self):
        drift = build_session_drift(np.array([10.0, 15.0, 20.0]))
        np.testing.assert_allclose(drift[:, 0], [0, 0.5, 1])
        np.testing.assert_allclose(drift[:, 1], [0, 0.25, 1])

    def test_video_has_three_smooth_acausal_terms(self):
        basis = video_temporal_basis()
        self.assertEqual(basis.basis.shape, (401, 3))
        np.testing.assert_allclose(basis.basis_time[[0, -1]], [-0.2, 0.2])
        peak_times = [
            basis.basis_time[np.argmax(basis.basis[:, column])] for column in range(3)
        ]
        np.testing.assert_allclose(peak_times, [-0.2, 0, 0.2])

        alignments = np.array([1.0, 5.0])
        frame_times = np.arange(0.5, 8.0, 1 / 60)
        score = np.sin(frame_times)
        design = build_video_component_design(alignments, frame_times, score)
        self.assertEqual(design.shape, (2 * 2639, 3))
        self.assertTrue(np.isfinite(design).all())

    def test_poisson_metrics_use_fit_set_null(self):
        observed = np.array([0, 1, 0, 1])
        predicted = np.full(4, 0.5)
        metrics = poisson_metrics(observed, predicted, fit_mean_count=0.5)
        self.assertAlmostEqual(metrics["deviance_explained"], 0)
        self.assertAlmostEqual(metrics["bits_per_spike"], 0)

    def test_poisson_fitter_has_no_damn_lower_rate_floor(self):
        design = np.zeros((10_000, 1), dtype=np.float32)
        counts = np.zeros(10_000, dtype=np.float32)
        counts[0] = 1
        model = fit_poisson_at_alpha(design, counts, alpha=1)
        prediction = model.predict(design[:1])[0]
        self.assertAlmostEqual(prediction, counts.mean(), places=7)
        self.assertLess(prediction, np.exp(-8))

    def test_short_camera_pulse_is_removed_only_for_exact_frame_count(self):
        falling = np.array([0.0, 1.0, 2.0])
        rising = falling + np.array([0.01, 0.00001, 0.01])
        selected, retained = select_falling_edges(falling, rising, 2)
        np.testing.assert_array_equal(selected, [0.0, 2.0])
        np.testing.assert_array_equal(retained, [True, False, True])
        with self.assertRaises(ValueError):
            select_falling_edges(falling, rising, 3)

    def test_equal_stimulus_windows(self):
        from damn.alignment import generate_master_alignment_bin_times

        alignments = np.arange(5) * 4.0
        bins = trial_bins(alignments)
        np.testing.assert_array_equal(np.bincount(bins["bin_trial_row"]), 2639)
        np.testing.assert_array_equal(
            bins["bin_center_s"],
            generate_master_alignment_bin_times(alignments, 0.1, 2.54, 0.001),
        )
        np.testing.assert_allclose(bins["window_start_s"], alignments - 0.0995)
        np.testing.assert_allclose(bins["window_stop_s"], alignments + 2.5395)
        for row in range(5):
            self.assertEqual(
                len(np.unique(bins["bin_split"][bins["bin_trial_row"] == row])), 1
            )
        for invalid in [np.arange(5) * 0.5, alignments[::-1], [0, 2, 4, 6, np.nan]]:
            with self.assertRaises(ValueError):
                trial_bins(np.array(invalid))

    def test_frame_timing_validation(self):
        np.testing.assert_array_equal(validate_frame_times(np.array([1, 2]), 2), [1, 2])
        for times, count in [([1, 1], 2), ([2, 1], 2), ([1, np.nan], 2), ([1, 2], 3)]:
            with self.assertRaises(ValueError):
                validate_frame_times(np.array(times), count)

    def test_video_scaling_uses_training_rows_only(self):
        values = np.array([[1.0, 10.0], [3.0, 14.0], [101.0, 110.0]])
        scaled, mean, scale = training_zscore(values, np.array([True, True, False]))
        np.testing.assert_array_equal(mean, [2.0, 12.0])
        np.testing.assert_array_equal(scale, [1.0, 2.0])
        np.testing.assert_array_equal(scaled[:2], [[-1.0, -1.0], [1.0, 1.0]])


if __name__ == "__main__":
    unittest.main()
