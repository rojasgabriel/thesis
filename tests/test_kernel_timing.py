from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


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

    def test_bpod_window_uses_reaction_or_response_time(self):
        from behavior_analyses.kernel_timing import extract_bpod_kernel_inputs

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

    def test_nidaq_visual_mapping_requires_complete_valid_events(self):
        from behavior_analyses.kernel_timing import resolve_nidaq_event_arrays

        with self.assertRaisesRegex(ValueError, "Missing EventMapping"):
            resolve_nidaq_event_arrays(
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

    def test_nidaq_window_uses_mapped_flash_and_port_times(self):
        from behavior_analyses.kernel_timing import (
            extract_nidaq_kernel_inputs,
            resolve_nidaq_event_arrays,
        )

        mapping_rows, event_rows = self._mapped_event_fixture()
        aligned = resolve_nidaq_event_arrays(
            event_rows, mapping_rows, "GRB006", "session"
        )
        center = extract_nidaq_kernel_inputs(
            aligned,
            self.trial_rows,
            "visual",
            observation_window="center_exit",
        )
        response = extract_nidaq_kernel_inputs(
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

    def test_bpod_and_nidaq_match_for_aligned_trial(self):
        from behavior_analyses.kernel_timing import (
            extract_bpod_kernel_inputs,
            extract_nidaq_kernel_inputs,
            resolve_nidaq_event_arrays,
        )

        mapping_rows, event_rows = self._mapped_event_fixture()
        aligned = resolve_nidaq_event_arrays(
            event_rows, mapping_rows, "GRB006", "session"
        )

        for window in ("center_exit", "response"):
            bpod = extract_bpod_kernel_inputs(
                self.trial_rows, "visual", observation_window=window
            )
            nidaq = extract_nidaq_kernel_inputs(
                aligned,
                self.trial_rows,
                "visual",
                observation_window=window,
            )

            self.assertEqual(bpod["response_values"], nidaq["response_values"])
            np.testing.assert_allclose(
                bpod["observation_end_times"], nidaq["observation_end_times"]
            )
            np.testing.assert_allclose(
                bpod["stim_times_per_trial"][0], nidaq["stim_times_per_trial"][0]
            )

    def test_combined_provenance_is_mixed(self):
        from behavior_analyses.kernel_timing import (
            combine_kernel_inputs,
            extract_bpod_kernel_inputs,
        )

        first = extract_bpod_kernel_inputs(
            self.trial_rows, "visual", observation_window="center_exit"
        )
        second = extract_bpod_kernel_inputs(
            self.trial_rows, "visual", observation_window="center_exit"
        )
        first["timing_source"] = "bpod"
        second["timing_source"] = "nidaq"

        combined = combine_kernel_inputs([first, second])

        self.assertEqual(combined["timing_source"], "mixed")
        self.assertEqual(len(combined["stim_times_per_trial"]), 2)

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
