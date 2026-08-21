import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from thesis.ephys.utils import (
    analysis_conditioned_stim,
    analysis_peak_counts,
    analysis_peth,
)
from thesis.ephys.utils.analysis_peth import compute_population_peth


class ComputePopulationPethTests(unittest.TestCase):
    def test_population_peth_converts_counts_to_spikes_per_second(self):
        peth_counts = np.array([[[0.0, 1.0, 2.0]]])
        bin_edges = np.array([-0.01, 0.0, 0.01, 0.02])

        with patch.object(
            analysis_peth,
            "population_peth",
            return_value=(peth_counts, bin_edges, np.array([0])),
        ):
            peth, returned_edges, bin_centers = compute_population_peth(
                [np.array([0.0])],
                np.array([0.0]),
                pre_seconds=0.01,
                post_seconds=0.02,
                binwidth_ms=10,
            )

        np.testing.assert_allclose(peth, peth_counts / 0.01)
        np.testing.assert_allclose(returned_edges, bin_edges)
        np.testing.assert_allclose(bin_centers, np.array([-0.005, 0.005, 0.015]))

    def test_population_peth_rejects_implausible_rates(self):
        peth_counts = np.array([[[11.0]]])
        bin_edges = np.array([0.0, 0.01])

        with patch.object(
            analysis_peth,
            "population_peth",
            return_value=(peth_counts, bin_edges, np.array([0])),
        ):
            with self.assertRaises(AssertionError):
                compute_population_peth(
                    [np.array([0.0])],
                    np.array([0.0]),
                    pre_seconds=0.0,
                    post_seconds=0.01,
                    binwidth_ms=10,
                )

    def test_population_peth_does_not_double_scale_when_spks_returns_rates(self):
        # Some spks versions return sp/s already (e.g. 1 spike in 10 ms -> 100 sp/s).
        peth_rates = np.array([[[0.0, 100.0]]])
        bin_edges = np.array([0.0, 0.01, 0.02])

        with patch.object(
            analysis_peth,
            "population_peth",
            return_value=(peth_rates, bin_edges, np.array([0])),
        ):
            peth, _, _ = compute_population_peth(
                [np.array([0.0])],
                np.array([0.0]),
                pre_seconds=0.0,
                post_seconds=0.02,
                binwidth_ms=10,
            )

        np.testing.assert_allclose(peth, peth_rates)


class ConditionedStimAnchorTests(unittest.TestCase):
    def test_build_trial_stim_classification_uses_center_port_exit_events(self):
        align_ev = {
            "stim_ev_15ms": np.array([0.2, 0.6]),
            "center_port": np.array([0.1]),
            "center_port_exit": np.array([0.5]),
            "left_port": np.array([]),
            "right_port": np.array([0.9]),
            "trial_start": np.array([0.0, 1.0]),
        }
        trial_df = pd.DataFrame(
            {
                "t_sync": [0.0],
                "t_react": [0.7],
                "response": [1],
            }
        )

        trial_ts = analysis_conditioned_stim.build_trial_stim_classification(
            align_ev, trial_df
        )

        self.assertEqual(len(trial_ts), 1)
        np.testing.assert_allclose(trial_ts["cp_exit_obx"], [0.5])
        self.assertEqual(trial_ts["stationary_stims"].iloc[0], [0.2])
        self.assertEqual(trial_ts["movement_stims"].iloc[0], [0.6])

    def test_extract_conditioned_stim_anchors_uses_paired_trials_only_for_pairs(self):
        trial_ts = pd.DataFrame(
            {
                "trial_idx": [0, 1, 2],
                "stationary_stims": [[0.1, 0.2], [], [2.1, 2.2, 2.3]],
                "movement_stims": [[0.5], [1.5], []],
            }
        )

        anchors = analysis_conditioned_stim.extract_conditioned_stim_anchors(trial_ts)

        np.testing.assert_allclose(anchors["first_stationary_all"], [0.1, 2.1])
        np.testing.assert_allclose(anchors["paired_last_stationary"], [0.2])
        np.testing.assert_allclose(anchors["paired_first_movement"], [0.5])
        np.testing.assert_array_equal(anchors["paired_trial_idx"], [0])


class ClassifyPeakCountTests(unittest.TestCase):
    def test_classify_peak_count_detects_clear_single_peak(self):
        bin_centers = np.arange(-0.1, 0.151, 0.01)
        # One unit, multiple identical trials.
        baseline = 5.0
        peak = 8.0 * np.exp(-0.5 * ((bin_centers - 0.05) / 0.01) ** 2)
        mean_trace = baseline + peak
        peth = np.tile(mean_trace, (1, 5, 1))

        df = analysis_peak_counts.classify_peak_count(
            peth,
            bin_centers,
            unit_ids=[123],
            search_window=(0.0, 0.12),
            baseline_window=(-0.1, 0.0),
            min_prominence_frac=0.25,
            min_prominence_abs=1.0,
            min_distance_ms=20.0,
            binwidth_ms=10.0,
            mode="peaks",
        )

        self.assertEqual(int(df.loc[0, "n_peaks"]), 1)
        self.assertTrue(0.0 <= df.loc[0, "peak_times"][0] <= 0.12)

    def test_classify_peak_count_does_not_invent_peak_on_monotonic_ramp(self):
        bin_centers = np.arange(-0.1, 0.151, 0.01)
        baseline = 5.0
        ramp = np.clip((bin_centers - 0.0) / 0.12, 0, 1) * 2.0  # +2 sp/s by 0.12 s
        mean_trace = baseline + ramp
        peth = np.tile(mean_trace, (1, 5, 1))

        df = analysis_peak_counts.classify_peak_count(
            peth,
            bin_centers,
            unit_ids=[123],
            search_window=(0.0, 0.12),
            baseline_window=(-0.1, 0.0),
            min_prominence_frac=0.25,
            min_prominence_abs=1.0,
            min_distance_ms=20.0,
            binwidth_ms=10.0,
            mode="peaks",
        )

        self.assertEqual(int(df.loc[0, "n_peaks"]), 0)


if __name__ == "__main__":
    unittest.main()
