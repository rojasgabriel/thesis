from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np


class KernelTimingTests(unittest.TestCase):
    def setUp(self):
        self.trial_rows = [
            {
                "trial_num": 0,
                "rewarded_modality": "visual",
                "stim_events": np.array([0.2, 0.35, 0.7]),
                "stim_rate_vision": 12.0,
                "response": 1,
                "t_sync": 10.0,
                "t_react": 10.5,
                "t_response": 10.9,
            },
            {
                "trial_num": 1,
                "rewarded_modality": "visual",
                "stim_events": np.array([0.2]),
                "stim_rate_vision": 8.0,
                "response": 0,
                "t_sync": 12.0,
                "t_react": None,
                "t_response": None,
            },
        ]
        self.trialset_keys = [
            {
                "subject_name": "GRB006",
                "session_name": "20240821_121447",
                "dataset_name": "chipmunk",
                "trialset_description": "visual",
            }
        ]

    def test_bpod_window_uses_reaction_or_response_time(self):
        from thesis.behavior.kernel_timing import extract_bpod_kernel_inputs

        center = extract_bpod_kernel_inputs(
            self.trial_rows, "visual", observation_window="center_exit"
        )
        response = extract_bpod_kernel_inputs(
            self.trial_rows, "visual", observation_window="response"
        )

        np.testing.assert_allclose(center["stim_times_per_trial"][0], [0.2, 0.35])
        np.testing.assert_allclose(
            response["stim_times_per_trial"][0], [0.2, 0.35, 0.7]
        )
        np.testing.assert_allclose(center["observation_end_times"], [0.5])
        np.testing.assert_allclose(response["observation_end_times"], [0.9])

    def test_nidq_visual_mapping_requires_complete_valid_events(self):
        from thesis.behavior.kernel_timing import resolve_nidq_event_arrays

        with self.assertRaisesRegex(ValueError, "Missing EventMapping"):
            resolve_nidq_event_arrays(
                [],
                [
                    {
                        "event_name": "visual_stim",
                        "source_dataset_name": "ephys",
                        "source_stream_name": "nidq",
                        "source_event_name": "ai0",
                    }
                ],
                "GRB006",
                "session",
            )

    def test_nidq_window_uses_mapped_flash_and_port_times(self):
        from thesis.behavior.kernel_timing import (
            extract_nidq_kernel_inputs,
            resolve_nidq_event_arrays,
        )

        mapping_rows, event_rows = self._mapped_event_fixture()
        aligned = resolve_nidq_event_arrays(
            event_rows, mapping_rows, "GRB006", "session"
        )
        center = extract_nidq_kernel_inputs(
            aligned,
            self.trial_rows,
            "visual",
            observation_window="center_exit",
        )
        response = extract_nidq_kernel_inputs(
            aligned,
            self.trial_rows,
            "visual",
            observation_window="response",
        )

        np.testing.assert_allclose(center["stim_times_per_trial"][0], [0.2, 0.35])
        np.testing.assert_allclose(
            response["stim_times_per_trial"][0], [0.2, 0.35, 0.7]
        )
        np.testing.assert_allclose(center["observation_end_times"], [0.5])
        np.testing.assert_allclose(response["observation_end_times"], [0.9])

    def test_bpod_and_nidq_match_for_aligned_trial(self):
        from thesis.behavior.kernel_timing import (
            extract_bpod_kernel_inputs,
            extract_nidq_kernel_inputs,
            resolve_nidq_event_arrays,
        )

        mapping_rows, event_rows = self._mapped_event_fixture()
        aligned = resolve_nidq_event_arrays(
            event_rows, mapping_rows, "GRB006", "session"
        )

        for window in ("center_exit", "response"):
            bpod = extract_bpod_kernel_inputs(
                self.trial_rows, "visual", observation_window=window
            )
            nidq = extract_nidq_kernel_inputs(
                aligned,
                self.trial_rows,
                "visual",
                observation_window=window,
            )

            self.assertEqual(bpod["response_values"], nidq["response_values"])
            np.testing.assert_allclose(
                bpod["observation_end_times"], nidq["observation_end_times"]
            )
            np.testing.assert_allclose(
                bpod["stim_times_per_trial"][0], nidq["stim_times_per_trial"][0]
            )

    def test_available_timing_sources_include_nidq_when_mapped(self):
        from thesis.behavior.kernel_timing import available_timing_sources

        mapping_rows, _ = self._mapped_event_fixture()
        with patch(
            "thesis.behavior.kernel_timing._fetch_event_mapping_rows",
            return_value=mapping_rows,
        ):
            sources = available_timing_sources(self.trialset_keys)

        self.assertEqual(sources, ["nidq", "bpod"])

    def test_available_timing_sources_are_bpod_only_without_mapping(self):
        from thesis.behavior.kernel_timing import available_timing_sources

        with patch(
            "thesis.behavior.kernel_timing._fetch_event_mapping_rows",
            return_value=[],
        ):
            sources = available_timing_sources(self.trialset_keys)

        self.assertEqual(sources, ["bpod"])

    def test_available_timing_sources_require_all_nidq_mappings(self):
        from thesis.behavior.kernel_timing import available_timing_sources

        mapping_rows, _ = self._mapped_event_fixture()
        incomplete = [row for row in mapping_rows if row["event_name"] != "right_port"]
        with patch(
            "thesis.behavior.kernel_timing._fetch_event_mapping_rows",
            return_value=incomplete,
        ):
            sources = available_timing_sources(self.trialset_keys)

        self.assertEqual(sources, ["bpod"])

    def test_fetch_pooled_kernel_inputs_requires_requested_timing_source(self):
        from thesis.behavior.kernel_timing import fetch_pooled_kernel_inputs

        with patch(
            "thesis.behavior.kernel_timing.available_timing_sources",
            return_value=["bpod"],
        ):
            with self.assertRaisesRegex(
                ValueError, "cannot supply timing_source='nidq'"
            ):
                fetch_pooled_kernel_inputs(
                    self.trialset_keys,
                    "visual",
                    observation_window="center_exit",
                    timing_source="nidq",
                )

    def test_same_session_yields_distinct_rows_per_timing_source(self):
        from thesis.behavior.kernel_timing import fetch_pooled_kernel_inputs

        mapping_rows, event_rows = self._mapped_event_fixture()

        def fake_trial_rows(_dataset_key):
            return self.trial_rows

        def fake_mapping(_session_key):
            return mapping_rows

        def fake_events(_session_key, _mapping_rows):
            return event_rows

        with (
            patch(
                "thesis.behavior.kernel_timing._fetch_chipmunk_trial_rows",
                side_effect=fake_trial_rows,
            ),
            patch(
                "thesis.behavior.kernel_timing._fetch_event_mapping_rows",
                side_effect=fake_mapping,
            ),
            patch(
                "thesis.behavior.kernel_timing._fetch_mapped_digital_event_rows",
                side_effect=fake_events,
            ),
        ):
            bpod = fetch_pooled_kernel_inputs(
                self.trialset_keys,
                "visual",
                observation_window="center_exit",
                timing_source="bpod",
            )
            nidq = fetch_pooled_kernel_inputs(
                self.trialset_keys,
                "visual",
                observation_window="center_exit",
                timing_source="nidq",
            )

        shared_key = {
            "analysis_set_id": "test_set",
            "subject_name": "GRB006",
            "trialset_description": "visual",
            "kernel_fit_config_id": 0,
        }
        bpod_key = {**shared_key, "timing_source": "bpod"}
        nidq_key = {**shared_key, "timing_source": "nidq"}
        self.assertNotEqual(bpod_key["timing_source"], nidq_key["timing_source"])
        self.assertEqual(
            {field: bpod_key[field] for field in shared_key},
            {field: nidq_key[field] for field in shared_key},
        )
        self.assertEqual(bpod["response_values"], nidq["response_values"])

    @staticmethod
    def _mapped_event_fixture():
        sources = {
            "visual_stim": ("ai0", [0.15, 0.2, 0.35, 0.7, 2.2], None),
            "trial_start": ("line0", [0.0, 2.0], [1, 1]),
            "left_port": ("line1", [2.3, 2.4], [1, 0]),
            "center_port": (
                "line2",
                [0.1, 0.45, 0.499, 0.5, 2.1, 2.5],
                [1, 0, 1, 0, 1, 0],
            ),
            "right_port": ("line3", [0.55, 0.56, 0.9, 1.0], [1, 0, 1, 0]),
        }
        mapping_rows = []
        event_rows = []
        for logical_name, (source_name, timestamps, values) in sources.items():
            mapping_rows.append(
                {
                    "event_name": logical_name,
                    "source_dataset_name": "ephys",
                    "source_stream_name": "nidq",
                    "source_event_name": source_name,
                }
            )
            event_row = {
                "dataset_name": "ephys",
                "stream_name": "nidq",
                "event_name": source_name,
                "event_timestamps": np.asarray(timestamps, dtype=float),
            }
            if values is not None:
                event_row["event_values"] = np.asarray(values)
            event_rows.append(event_row)
        return mapping_rows, event_rows


if __name__ == "__main__":
    unittest.main()
