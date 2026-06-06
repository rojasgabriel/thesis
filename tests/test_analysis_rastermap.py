import unittest

import numpy as np
import pandas as pd

from ephys.src.utils.analysis_rastermap import (
    build_continuous_spike_count_matrix,
    build_trial_window_spike_count_matrix,
    choose_rastermap_cluster_count,
    event_positions_in_concatenated_bins,
    event_response_matrix,
    fit_rastermap,
    hardware_trial_event_time_table,
    heatmap_for_result,
    iti_windows_from_hardware_events,
    RastermapResult,
    trial_baseline_event_response_matrix,
    trial_event_time_table_from_metadata,
    trial_event_times_from_metadata,
    trial_windows_from_metadata,
)


class ContinuousSpikeMatrixTests(unittest.TestCase):
    def test_build_continuous_spike_count_matrix_preserves_unit_order_and_counts(self):
        spike_times_by_unit = {
            20: np.array([0.05, 0.11, 0.19, 0.21, np.nan]),
            10: np.array([0.0, 0.1, 0.299]),
        }

        matrix = build_continuous_spike_count_matrix(
            spike_times_by_unit,
            bin_ms=100,
            t_stop_s=0.3,
        )

        np.testing.assert_array_equal(matrix.unit_ids, np.array([20, 10]))
        np.testing.assert_allclose(matrix.bin_edges_s, np.array([0.0, 0.1, 0.2, 0.3]))
        np.testing.assert_array_equal(
            matrix.spike_counts,
            np.array(
                [
                    [1, 2, 1],
                    [1, 1, 1],
                ],
                dtype=np.float32,
            ),
        )

    def test_build_continuous_spike_count_matrix_rejects_empty_units(self):
        with self.assertRaises(ValueError):
            build_continuous_spike_count_matrix({})

    def test_trial_windows_from_metadata_maps_initiation_to_response_to_obx_clock(self):
        trial_df = pd.DataFrame(
            {
                "t_sync": [0.0, 10.0, 30.0],
                "trial_start_ts": [100.0, 110.0, 130.0],
                "t_initiate": [1.0, 11.0, np.nan],
                "t_response": [4.0, 14.0, 24.0],
            }
        )

        windows = trial_windows_from_metadata(trial_df)

        np.testing.assert_allclose(windows, np.array([[101.0, 104.0], [111.0, 114.0]]))

    def test_trial_event_times_from_metadata_maps_task_events_to_obx_clock(self):
        trial_df = pd.DataFrame(
            {
                "t_sync": [0.0, 10.0, 30.0],
                "trial_start_ts": [100.0, 110.0, 130.0],
                "t_initiate": [1.0, 11.0, np.nan],
                "t_stim": [1.5, 11.5, np.nan],
                "t_gocue": [2.0, 12.0, np.nan],
                "t_react": [3.0, 13.0, np.nan],
                "t_response": [4.0, 14.0, np.nan],
            }
        )

        event_times = trial_event_times_from_metadata(trial_df)

        np.testing.assert_allclose(event_times["t_initiate"], [101.0, 111.0])
        np.testing.assert_allclose(event_times["t_stim"], [101.5, 111.5])
        np.testing.assert_allclose(event_times["t_react"], [103.0, 113.0])
        np.testing.assert_allclose(event_times["t_response"], [104.0, 114.0])

    def test_trial_event_time_table_from_metadata_preserves_trial_axis(self):
        trial_df = pd.DataFrame(
            {
                "t_sync": [0.0, 10.0, 30.0],
                "trial_start_ts": [100.0, 110.0, 130.0],
                "t_initiate": [1.0, 11.0, 21.0],
                "t_stim": [1.5, np.nan, 21.5],
                "t_gocue": [2.0, 12.0, 22.0],
                "t_react": [3.0, 13.0, 23.0],
                "t_response": [4.0, 14.0, 24.0],
            }
        )

        event_times = trial_event_time_table_from_metadata(trial_df)

        np.testing.assert_allclose(event_times["t_initiate"], [101.0, 111.0, 121.0])
        np.testing.assert_allclose(
            event_times["t_stim"], [101.5, np.nan, 121.5], equal_nan=True
        )

    def test_event_response_matrix_returns_post_minus_baseline_rates(self):
        spike_times_by_unit = {
            20: np.array([0.85, 1.01, 1.05, 2.15]),
            10: np.array([0.81, 1.15, 1.17, 1.19]),
        }
        event_times_by_name = {"t_stim": np.array([1.0])}

        event_names, response = event_response_matrix(
            spike_times_by_unit,
            event_times_by_name,
            response_window_s=(0.0, 0.2),
            baseline_window_s=(-0.2, 0.0),
        )

        self.assertEqual(event_names, ("t_stim",))
        np.testing.assert_allclose(response[:, 0], [5.0, 10.0])

    def test_trial_baseline_event_response_uses_pre_initiation_baseline_for_all_events(
        self,
    ):
        spike_times_by_unit = {
            20: np.array([0.95, 1.51, 1.55, 2.05]),
            10: np.array([0.91, 0.95, 1.52, 2.01, 2.03]),
        }
        event_times_by_name = {
            "t_initiate": np.array([1.0]),
            "t_stim": np.array([1.5]),
            "t_react": np.array([2.0]),
            "t_response": np.array([2.0]),
        }

        event_names, response = trial_baseline_event_response_matrix(
            spike_times_by_unit,
            event_times_by_name,
            response_window_s=(0.0, 0.2),
            baseline_window_s=(-0.1, 0.0),
        )

        self.assertEqual(event_names, ("t_initiate", "t_stim", "t_react", "t_response"))
        np.testing.assert_allclose(response[0], [-10.0, 0.0, -5.0, -5.0])
        np.testing.assert_allclose(response[1], [-20.0, -15.0, -10.0, -10.0])

    def test_build_trial_window_spike_count_matrix_concatenates_trial_windows(self):
        spike_times_by_unit = {
            20: np.array([0.05, 0.11, 1.05, 1.25, 1.45, 2.0]),
            10: np.array([0.15, 1.15, 1.35]),
        }
        trial_windows_s = np.array([[0.0, 0.25], [1.0, 1.4]])

        matrix = build_trial_window_spike_count_matrix(
            spike_times_by_unit,
            trial_windows_s,
            bin_ms=100,
        )

        np.testing.assert_array_equal(matrix.unit_ids, np.array([20, 10]))
        np.testing.assert_allclose(
            matrix.bin_edges_s, np.array([0.0, 0.1, 0.2, 0.25, 0.35, 0.45, 0.55, 0.65])
        )
        np.testing.assert_array_equal(
            matrix.trial_idx_by_bin,
            np.array([0, 0, 0, 1, 1, 1, 1]),
        )
        np.testing.assert_array_equal(
            matrix.spike_counts,
            np.array(
                [
                    [1, 1, 0, 1, 0, 1, 0],
                    [0, 1, 0, 0, 1, 0, 1],
                ],
                dtype=np.float32,
            ),
        )

    def test_event_positions_in_concatenated_bins_maps_events_to_stitched_axis(self):
        matrix = build_trial_window_spike_count_matrix(
            {20: np.array([0.05, 1.05])},
            np.array([[0.0, 0.25], [1.0, 1.3]]),
            bin_ms=100,
        )
        event_times_by_name = {
            "t_initiate": np.array([0.0, 1.0]),
            "t_stim": np.array([0.15, 1.25]),
        }

        positions = event_positions_in_concatenated_bins(matrix, event_times_by_name)

        np.testing.assert_allclose(positions["t_initiate"], [0.0, 3.0])
        np.testing.assert_allclose(positions["t_stim"], [1.5, 5.5])

    def test_iti_windows_from_hardware_events_uses_choice_to_next_fixation(self):
        trial_df = pd.DataFrame({"trial_start_ts": [0.0, 10.0, 20.0]})
        align_ev = {
            "center_port": np.array([1.0, 11.0, 21.0]),
            "center_port_exit": np.array([3.0, 13.0, 23.0]),
            "first_stim_ev_15ms": np.array([2.0, 12.0, 22.0]),
            "left_port": np.array([4.0, 24.0]),
            "right_port": np.array([14.0]),
        }

        event_times = hardware_trial_event_time_table(trial_df, align_ev)
        windows = iti_windows_from_hardware_events(trial_df, align_ev)

        np.testing.assert_allclose(event_times["choice"], [4.0, 14.0, 24.0])
        np.testing.assert_allclose(windows, np.array([[4.0, 11.0], [14.0, 21.0]]))


class RastermapFitTests(unittest.TestCase):
    def test_choose_rastermap_cluster_count_clamps_below_unit_count(self):
        self.assertEqual(choose_rastermap_cluster_count(12), 11)
        self.assertEqual(choose_rastermap_cluster_count(100), 100)
        self.assertEqual(choose_rastermap_cluster_count(189), 100)

    def test_choose_rastermap_cluster_count_rejects_too_few_units(self):
        with self.assertRaises(ValueError):
            choose_rastermap_cluster_count(9)

    def test_fit_rastermap_uses_expected_parameters_without_real_fit(self):
        class FakeRastermap:
            last_instance = None

            def __init__(self, **kwargs):
                self.kwargs = kwargs
                FakeRastermap.last_instance = self

            def fit(self, data):
                self.data = data
                self.isort = np.arange(data.shape[0] - 1, -1, -1)
                self.embedding = np.arange(data.shape[0])[:, np.newaxis]
                self.X_embedding = data[self.isort]
                return self

        spike_counts = np.arange(60, dtype=float).reshape(12, 5)

        model, n_clusters = fit_rastermap(
            spike_counts,
            rastermap_cls=FakeRastermap,
        )

        self.assertIs(model, FakeRastermap.last_instance)
        self.assertEqual(n_clusters, 11)
        self.assertEqual(
            FakeRastermap.last_instance.kwargs,
            {
                "n_PCs": 200,
                "n_clusters": 11,
                "locality": 0.75,
                "time_lag_window": 5,
            },
        )
        self.assertEqual(FakeRastermap.last_instance.data.dtype, np.float32)
        np.testing.assert_array_equal(model.isort, np.arange(11, -1, -1))

    def test_heatmap_falls_back_to_zscored_sorted_counts(self):
        result = RastermapResult(
            subject="SUBJ",
            session="SESSION",
            unit_ids=np.array([1, 2]),
            depth=np.array([100.0, 200.0]),
            bin_edges_s=np.array([0.0, 0.1, 0.2, 0.3]),
            spike_counts=np.array([[1.0, 2.0, 3.0], [5.0, 5.0, 5.0]]),
            trial_idx_by_bin=None,
            absolute_bin_start_s=None,
            absolute_bin_stop_s=None,
            isort=np.array([1, 0]),
            embedding=np.array([[0.0], [1.0]]),
            x_embedding=None,
            n_clusters=2,
        )

        heatmap = heatmap_for_result(result)

        np.testing.assert_allclose(heatmap[0], np.zeros(3))
        np.testing.assert_allclose(
            heatmap[1],
            np.array([-1.22474487, 0.0, 1.22474487]),
        )


if __name__ == "__main__":
    unittest.main()
