"""Unit tests for psychophysical-kernel design matrix and fit."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from ephys.src.utils.psychophysical_kernel import (
    build_residual_rate_matrix,
    code_choice_right,
    extract_trial_kernel_inputs,
    fit_psychophysical_kernel,
    interpret_kernel_weights,
)


class ChoiceCodingTests(unittest.TestCase):
    def test_right_is_class_one_left_is_class_zero(self):
        coded, mask = code_choice_right([1, -1, 0, 1, np.nan])
        np.testing.assert_array_equal(mask, [True, True, False, True, False])
        np.testing.assert_array_equal(coded, [1, 0, 1])


class ResidualMatrixTests(unittest.TestCase):
    def test_unseen_late_bins_are_nan_not_zero(self):
        # Trial sees flashes only in the first 150 ms of a 100 ms × 3 bin grid.
        stim_times = [np.array([1.05, 1.12])]
        first_stim = [1.0]
        observation_end = [1.15]
        responses = [1]

        residual, choices, n_observed, centers = build_residual_rate_matrix(
            stim_times,
            first_stim,
            observation_end,
            responses,
            timebins=3,
            bin_width_s=0.1,
            max_rate_hz=20.0,
        )

        self.assertEqual(residual.shape, (1, 3))
        np.testing.assert_array_equal(choices, [1])
        self.assertTrue(np.isfinite(residual[0, 0]))
        self.assertTrue(np.isfinite(residual[0, 1]))
        self.assertTrue(np.isnan(residual[0, 2]))
        self.assertNotEqual(residual[0, 2], 0.0)
        np.testing.assert_array_equal(n_observed, [1, 1, 0])
        np.testing.assert_allclose(centers, [0.05, 0.15, 0.25])

    def test_zero_evidence_is_distinct_from_missing(self):
        # A fully observed empty bin should be count-expected (= negative), not NaN.
        stim_times = [np.array([0.0])]
        residual, _, n_observed, _ = build_residual_rate_matrix(
            stim_times,
            [0.0],
            [0.3],
            [1],
            timebins=3,
            bin_width_s=0.1,
            max_rate_hz=20.0,
        )
        self.assertTrue(np.isfinite(residual[0, 1]))
        self.assertLess(residual[0, 1], 0.0)
        np.testing.assert_array_equal(n_observed, [1, 1, 1])

    def test_event_order_aligned_to_first_stim(self):
        stim_times = [np.array([10.0, 10.05, 10.15])]
        residual, _, _, centers = build_residual_rate_matrix(
            stim_times,
            [10.0],
            [10.25],
            [-1],
            timebins=2,
            bin_width_s=0.1,
            max_rate_hz=20.0,
        )
        # Bin0 [0,0.1): two flashes -> 2 - 2 = 0; bin1 [0.1,0.2): one flash -> 1-2=-1
        np.testing.assert_allclose(residual[0], [0.0, -1.0])
        np.testing.assert_allclose(centers, [0.05, 0.15])


class ObservationWindowTests(unittest.TestCase):
    def _one_trial_fixture(self):
        # CP entry 0.1, first flash 0.2, CP exit 0.5, movement flash 0.7, RP 0.9
        align_ev = {
            "stim_ev": np.array([0.2, 0.35, 0.7]),
            "center_port": np.array([0.1]),
            "center_port_exit": np.array([0.5]),
            "left_port": np.array([]),
            "right_port": np.array([0.9]),
            "trial_start": np.array([0.0, 2.0]),
        }
        trial_df = pd.DataFrame(
            {
                "t_sync": [0.0, 2.0],
                "t_react": [0.5, np.nan],
                "response": [1, 0],
            }
        )
        return align_ev, trial_df

    def test_center_exit_excludes_movement_flashes(self):
        align_ev, trial_df = self._one_trial_fixture()
        inputs = extract_trial_kernel_inputs(
            align_ev, trial_df, observation_window="center_exit"
        )
        self.assertEqual(inputs["n_trials"], 1)
        np.testing.assert_allclose(inputs["stim_times_per_trial"][0], [0.2, 0.35])
        self.assertAlmostEqual(float(inputs["observation_end_times"][0]), 0.5)

    def test_response_window_includes_movement_flashes(self):
        align_ev, trial_df = self._one_trial_fixture()
        inputs = extract_trial_kernel_inputs(
            align_ev, trial_df, observation_window="response"
        )
        self.assertEqual(inputs["n_trials"], 1)
        np.testing.assert_allclose(inputs["stim_times_per_trial"][0], [0.2, 0.35, 0.7])
        self.assertAlmostEqual(float(inputs["observation_end_times"][0]), 0.9)


class FitDeterminismTests(unittest.TestCase):
    def _synthetic_fixture(self, n_trials=120, timebins=4, seed=0):
        rng = np.random.default_rng(seed)
        residual = np.full((n_trials, timebins), np.nan)
        choices = np.zeros(n_trials, dtype=int)
        for i in range(n_trials):
            # Early-integrator: only bin 0 predicts choice.
            early = rng.normal(0.0, 1.0)
            choices[i] = int(early > 0)
            residual[i, 0] = early
            residual[i, 1] = rng.normal(0.0, 0.2)
            # Later bins observed on only half the trials.
            if i < n_trials // 2:
                residual[i, 2] = rng.normal(0.0, 0.2)
                residual[i, 3] = rng.normal(0.0, 0.2)
        n_observed = np.sum(np.isfinite(residual), axis=0).astype(int)
        return residual, choices, n_observed

    def test_fit_is_deterministic_for_fixed_random_state(self):
        residual, choices, n_observed = self._synthetic_fixture()
        first = fit_psychophysical_kernel(
            residual,
            choices,
            n_observed_per_bin=n_observed,
            cv_splits=5,
            random_state=7,
            min_trials_per_bin=40,
        )
        second = fit_psychophysical_kernel(
            residual,
            choices,
            n_observed_per_bin=n_observed,
            cv_splits=5,
            random_state=7,
            min_trials_per_bin=40,
        )
        self.assertTrue(first["fit_converged"])
        np.testing.assert_allclose(first["weights_mean"], second["weights_mean"])
        np.testing.assert_allclose(first["scores"], second["scores"])
        # Incomplete late bins contribute fewer trials than early bins.
        self.assertLess(int(n_observed[3]), int(n_observed[0]))
        self.assertTrue(np.any(~np.isfinite(residual[:, 3])))

    def test_interpretation_labels_early_integrator(self):
        weights = np.array([0.8, 0.6, 0.1, 0.05])
        n_observed = np.array([100, 100, 100, 100])
        label = interpret_kernel_weights(
            weights, n_observed, min_trials_per_bin=50, ratio_threshold=1.5
        )
        self.assertEqual(label, "early_integrator")


if __name__ == "__main__":
    unittest.main()
