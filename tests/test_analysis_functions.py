from __future__ import annotations

from pathlib import Path
import sys
import types
import unittest

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))


class PsychometricTests(unittest.TestCase):
    def test_package_adapter_uses_labdata_rightward_response_code(self):
        fake_module = types.ModuleType("fit_psychometric")

        def cumulative_gaussian(alpha, beta, gamma, lapse, x):
            x = np.asarray(x, dtype=float)
            return gamma + (1 - gamma - lapse) / (1 + np.exp(-beta * (x - alpha)))

        def fit_psychometric(stim_values, response_values, min_required_stim_values):
            stims = np.unique(stim_values)
            p_side = np.array(
                [np.mean(response_values[stim_values == stim]) for stim in stims]
            )
            n_obs = np.array([np.sum(stim_values == stim) for stim in stims])
            n_side = p_side * n_obs
            if len(stims) < min_required_stim_values:
                fit_params = None
            else:
                fit_params = np.array([12.0, 0.2, p_side[0], 1 - p_side[-1]])
            return {
                "stims": stims,
                "p_side": p_side,
                "p_side_ci": np.column_stack([p_side, p_side]),
                "n_side": n_side,
                "n_obs": n_obs,
                "fit_params": fit_params,
                "fit": object(),
                "function": cumulative_gaussian,
            }

        fake_module.cumulative_gaussian = cumulative_gaussian
        fake_module.fit_psychometric = fit_psychometric
        sys.modules["fit_psychometric"] = fake_module

        from behavior_analyses.psychometrics import fit_psychometric_labdata

        stims = np.repeat(np.array([4, 6, 8, 12, 16, 20], dtype=float), 30)
        responses = np.where(stims > 12, 1, -1)
        responses[stims == 12] = np.tile([1, -1], 15)

        fit = fit_psychometric_labdata(stims, responses, min_choices=20)

        self.assertIsNotNone(fit)
        self.assertEqual(fit["stims"].tolist(), [4, 6, 8, 12, 16, 20])
        np.testing.assert_allclose(fit["p_right"], [0, 0, 0, 0.5, 1, 1])
        self.assertEqual(len(fit["fit_params"]), 4)


class LearningTests(unittest.TestCase):
    def test_summarize_trialset_counts_choice_and_means(self):
        from behavior_analyses.learning import summarize_trialset

        summary = summarize_trialset(
            {
                "n_trials": 4,
                "performance": 0.75,
                "performance_easy": 1.0,
                "response_values": np.array([1, -1, 0, np.nan]),
                "correct_values": np.array([1, 1, 0, np.nan]),
                "initiation_times": np.array([0.2, 0.3, np.nan, 0.5]),
                "reaction_times": np.array([0.1, np.nan, 0.2, 0.3]),
                "intensity_values": np.array([4, 8, 8, np.nan]),
            }
        )

        self.assertEqual(summary["n_trials"], 4)
        self.assertEqual(summary["n_with_choice"], 2)
        self.assertEqual(summary["n_correct"], 2)
        self.assertAlmostEqual(summary["mean_initiation_time"], 1.0 / 3.0)
        np.testing.assert_allclose(summary["stim_values"], [4, 8])


class KernelTests(unittest.TestCase):
    def test_kernel_keeps_unobserved_late_bins_as_nan(self):
        from behavior_analyses.kernels import build_residual_rate_matrix

        residual, choices, n_observed, centers, expected = build_residual_rate_matrix(
            [np.array([1.0, 1.05, 1.12])],
            [1.0],
            [1.15],
            [1],
            [20.0],
            timebins=3,
            bin_width_s=0.1,
        )

        np.testing.assert_array_equal(choices, [1])
        np.testing.assert_array_equal(n_observed, [1, 1, 0])
        np.testing.assert_allclose(centers, [0.05, 0.15, 0.25])
        self.assertTrue(np.isnan(residual[0, 2]))
        self.assertTrue(np.isnan(expected[0, 2]))

    def test_kernel_fit_is_deterministic(self):
        from behavior_analyses.kernels import fit_psychophysical_kernel

        rng = np.random.default_rng(4)
        residual = rng.normal(size=(120, 4))
        choices = (residual[:, 0] > 0).astype(int)
        residual[60:, 3] = np.nan
        n_observed = np.sum(np.isfinite(residual), axis=0)
        expected = np.where(np.isfinite(residual), 2.0, np.nan)

        first = fit_psychophysical_kernel(
            residual,
            choices,
            expected_counts=expected,
            n_observed_per_bin=n_observed,
            cv_splits=5,
            random_state=7,
            min_trials_per_bin=50,
        )
        second = fit_psychophysical_kernel(
            residual,
            choices,
            expected_counts=expected,
            n_observed_per_bin=n_observed,
            cv_splits=5,
            random_state=7,
            min_trials_per_bin=50,
        )

        self.assertTrue(first["fit_converged"])
        self.assertEqual(first["n_bins_fit"], 4)
        np.testing.assert_allclose(first["weights_mean"], second["weights_mean"])
        np.testing.assert_allclose(first["scores"], second["scores"])


if __name__ == "__main__":
    unittest.main()
