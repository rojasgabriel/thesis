import unittest

import numpy as np
import pandas as pd

from ephys.src.utils.analysis_rate_tuning import (
    add_light_exposure_to_responses,
    add_trial_predictors,
    aggregate_tuning_curves,
    build_task_stimulus_windows,
    compute_light_exposure,
    compute_timecourse_responses,
    compute_trial_responses,
    fit_encoding_models,
    first_event_in_window,
    response_events_for_choice,
    shuffle_fsi_null,
    summarize_units,
    summarize_timecourse_encoding,
)


class RateTuningWindowTests(unittest.TestCase):
    def test_first_event_in_window_uses_half_open_bounds(self):
        events = np.array([0.0, 0.1, 0.2, 0.3])

        self.assertEqual(first_event_in_window(events, 0.1, 0.3), 0.1)
        self.assertEqual(
            first_event_in_window(events, 0.1, 0.3, include_start=False),
            0.2,
        )
        self.assertTrue(np.isnan(first_event_in_window(events, 0.31, 0.5)))

    def test_response_events_for_choice_selects_side_port(self):
        align_ev = {
            "left_port": np.array([1.0]),
            "right_port": np.array([2.0]),
        }

        np.testing.assert_allclose(response_events_for_choice(align_ev, -1), [1.0])
        np.testing.assert_allclose(response_events_for_choice(align_ev, 1), [2.0])
        self.assertEqual(response_events_for_choice(align_ev, 0).size, 0)

    def test_build_task_windows_keeps_only_valid_trials(self):
        align_ev = {
            "first_stim_ev_15ms": np.array([0.2, 1.2, 2.2, 3.2, 4.2]),
            "left_port": np.array([0.8, 4.8]),
            "right_port": np.array([1.8, 2.1, 3.8]),
        }
        trial_df = pd.DataFrame(
            {
                "trial_start_ts": [0.0, 1.0, 2.0, 3.0, 4.0],
                "stim_rate_vision": [4, 8, 12, 22, 20],
                "response": [-1, 1, 1, 1, -1],
                "with_choice": [1, 1, 1, 1, 0],
            },
            index=[10, 11, 12, 13, 14],
        )

        windows = build_task_stimulus_windows(align_ev, trial_df)

        self.assertEqual(windows["trial_idx"].tolist(), [10, 11])
        np.testing.assert_allclose(windows["window_start_s"], [0.2, 1.2])
        np.testing.assert_allclose(windows["window_end_s"], [0.8, 1.8])
        np.testing.assert_allclose(windows["stim_rate_vision"], [4, 8])


class RateTuningResponseTests(unittest.TestCase):
    def test_compute_trial_responses_counts_spikes_in_window(self):
        windows = pd.DataFrame(
            {
                "trial_idx": [0, 1],
                "stim_rate_vision": [4.0, 8.0],
                "response_side": [-1, 1],
                "with_choice": [1, 1],
                "window_start_s": [0.2, 1.0],
                "window_end_s": [0.7, 1.5],
                "window_duration_s": [0.5, 0.5],
            }
        )
        spikes = {
            101: np.array([0.2, 0.4, 0.7, 1.1, 1.4]),
            202: np.array([0.1, 1.6]),
        }

        responses = compute_trial_responses(windows, spikes)

        unit_101 = responses[responses["unit_id"] == 101].sort_values("trial_idx")
        self.assertEqual(unit_101["spike_count"].tolist(), [2, 2])
        np.testing.assert_allclose(unit_101["response_sp_s"], [4.0, 4.0])
        unit_202 = responses[responses["unit_id"] == 202].sort_values("trial_idx")
        self.assertEqual(unit_202["spike_count"].tolist(), [0, 0])
        np.testing.assert_allclose(unit_202["response_sp_s"], [0.0, 0.0])

    def test_aggregate_and_summarize_tuning_curves(self):
        responses = pd.DataFrame(
            {
                "unit_id": [1, 1, 1, 1, 2, 2],
                "stim_rate_vision": [4.0, 4.0, 8.0, 8.0, 4.0, 8.0],
                "response_sp_s": [1.0, 3.0, 5.0, 7.0, 10.0, 4.0],
                "window_duration_s": [0.5, 0.7, 0.5, 0.7, 0.5, 0.5],
            }
        )

        tuning = aggregate_tuning_curves(responses)
        unit_1 = tuning[tuning["unit_id"] == 1].sort_values("stim_rate_vision")
        np.testing.assert_allclose(unit_1["mean_sp_s"], [2.0, 6.0])
        np.testing.assert_allclose(unit_1["median_sp_s"], [2.0, 6.0])
        np.testing.assert_allclose(unit_1["n_trials"], [2, 2])

        summary = summarize_units(tuning)
        unit_1_summary = summary[summary["unit_id"] == 1].iloc[0]
        self.assertEqual(unit_1_summary["preferred_stim_rate"], 8.0)
        self.assertEqual(unit_1_summary["tuning_range_sp_s"], 4.0)
        self.assertEqual(unit_1_summary["frequency_selectivity_index"], 0.5)

        unit_2_summary = summary[summary["unit_id"] == 2].iloc[0]
        self.assertAlmostEqual(
            unit_2_summary["frequency_selectivity_index"],
            6.0 / 14.0,
        )

    def test_shuffle_fsi_null_preserves_responses_and_summarizes_thresholds(self):
        responses = pd.DataFrame(
            {
                "unit_id": [1, 1, 1, 1, 2, 2, 2, 2],
                "trial_idx": [0, 1, 2, 3, 0, 1, 2, 3],
                "stim_rate_vision": [4.0, 4.0, 8.0, 8.0, 4.0, 4.0, 8.0, 8.0],
                "response_sp_s": [1.0, 2.0, 7.0, 8.0, 5.0, 5.0, 5.0, 5.0],
                "window_duration_s": [0.5] * 8,
            }
        )
        observed = summarize_units(aggregate_tuning_curves(responses))

        population_null, unit_null = shuffle_fsi_null(
            responses,
            observed,
            n_shuffles=5,
            seed=1,
        )

        self.assertEqual(len(population_null), 5)
        self.assertEqual(set(unit_null["unit_id"]), {1, 2})
        self.assertIn("shuffle_fsi_p95", unit_null.columns)
        self.assertIn("exceeds_shuffle_p95", unit_null.columns)

    def test_light_exposure_and_spikes_per_flash(self):
        windows = pd.DataFrame(
            {
                "trial_idx": [0, 1],
                "window_start_s": [0.0, 1.0],
                "window_end_s": [0.5, 1.5],
                "window_duration_s": [0.5, 0.5],
            }
        )
        exposure = compute_light_exposure(windows, np.array([0.0, 0.2, 1.6]))
        self.assertEqual(exposure["flash_count"].tolist(), [2, 0])
        np.testing.assert_allclose(exposure["total_light_time_s"], [0.03, 0.0])
        np.testing.assert_allclose(exposure["duty_cycle"], [0.06, 0.0])

        responses = pd.DataFrame(
            {
                "trial_idx": [0, 1],
                "spike_count": [4, 2],
            }
        )
        merged = add_light_exposure_to_responses(responses, exposure)
        self.assertEqual(merged.loc[0, "spikes_per_flash"], 2.0)
        self.assertTrue(np.isnan(merged.loc[1, "spikes_per_flash"]))

    def test_timecourse_response_counts_mask_after_response(self):
        windows = pd.DataFrame(
            {
                "trial_idx": [0],
                "stim_rate_vision": [20.0],
                "response_side": [1],
                "category_boundary": [12.0],
                "window_start_s": [1.0],
                "window_duration_s": [0.25],
            }
        )
        spikes = {10: np.array([1.02, 1.08, 1.12, 1.26])}

        timecourse = compute_timecourse_responses(
            windows,
            spikes,
            np.array([0.0, 0.1, 0.2, 0.3]),
        )

        self.assertEqual(timecourse["bin_start_s"].tolist(), [0.0, 0.1])
        self.assertEqual(timecourse["spike_count"].tolist(), [2, 1])
        np.testing.assert_allclose(timecourse["response_sp_s"], [20.0, 10.0])

    def test_trial_predictors_and_encoding_models(self):
        responses = pd.DataFrame(
            {
                "unit_id": [1] * 8,
                "trial_idx": np.arange(8),
                "stim_rate_vision": [4, 5, 16, 18, 4, 5, 16, 18],
                "category_boundary": [12] * 8,
                "response_side": [-1, -1, 1, 1, -1, -1, 1, 1],
                "response_sp_s": [1, 2, 8, 9, 1.5, 2.5, 8.5, 9.5],
            }
        )

        predicted = add_trial_predictors(responses)
        self.assertEqual(
            predicted["stim_category"].tolist()[:4],
            [
                "low_rate",
                "low_rate",
                "high_rate",
                "high_rate",
            ],
        )

        summary = fit_encoding_models(predicted, n_splits=4)
        self.assertEqual(
            set(summary["model"]),
            {"baseline", "signed_evidence", "category", "choice", "combined"},
        )
        self.assertTrue(summary["cv_r2"].notna().any())

    def test_timecourse_encoding_summary_shapes(self):
        rows = []
        for unit_id in [1, 2]:
            for trial_idx, rate in enumerate([4.0, 5.0, 16.0, 18.0]):
                for bin_start in [0.0, 0.1]:
                    rows.append(
                        {
                            "unit_id": unit_id,
                            "trial_idx": trial_idx,
                            "stim_rate_vision": rate,
                            "response_side": -1 if rate < 12 else 1,
                            "category_boundary": 12.0,
                            "bin_start_s": bin_start,
                            "bin_end_s": bin_start + 0.1,
                            "response_sp_s": rate + unit_id,
                        }
                    )
        timecourse = add_trial_predictors(pd.DataFrame(rows))

        summary, coefficients = summarize_timecourse_encoding(
            timecourse,
            n_splits=2,
        )

        self.assertEqual(len(summary), 2)
        self.assertEqual(len(coefficients), 4)
        self.assertIn("rate_correlation", summary.columns)
        self.assertIn("signed_rate_coefficient", coefficients.columns)


if __name__ == "__main__":
    unittest.main()
