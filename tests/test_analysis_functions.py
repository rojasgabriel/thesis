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
    def test_kernel_design_matrix_skips_no_choice_and_short_stim_trials(self):
        from behavior_analyses.kernels import build_residual_rate_matrix

        stim_events = [
            np.array([0.0, 0.1, 0.2]),
            np.array([0.0]),
            np.array([0.0, 0.2, 0.4]),
        ]
        responses = np.array([1, -1, 0])

        x, y = build_residual_rate_matrix(stim_events, responses, timebins=2)

        self.assertEqual(x.shape, (1, 2))
        np.testing.assert_array_equal(y, [1])


if __name__ == "__main__":
    unittest.main()
