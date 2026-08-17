import unittest

import numpy as np
import pandas as pd

from ephys.src.utils.io_chipmunk_trials import (
    align_response_port_entries,
    align_trial_event_timestamps,
)


class TrialEventTimestampTests(unittest.TestCase):
    def setUp(self):
        self.trial_starts = np.array([0.0, 10.0])
        self.responses = pd.Series([-1, 1])
        self.events = {
            "center_port": np.array([1.0, 11.0]),
            "center_port_exit": np.array([2.0, 5.0, 12.0]),
            "left_port": np.array([1.5, 6.0]),
            "right_port": np.array([13.0]),
            "stim_ev_15ms": np.array([3.0, 4.0, 12.5]),
        }

    def test_response_uses_first_selected_port_entry_after_last_center_exit(self):
        response_ts, response_ports = align_response_port_entries(
            self.trial_starts, self.responses, self.events
        )

        np.testing.assert_allclose(response_ts, [6.0, 13.0])
        np.testing.assert_array_equal(response_ports, ["left", "right"])

    def test_trial_events_stay_inside_each_trial(self):
        aligned = align_trial_event_timestamps(
            self.trial_starts, self.responses, self.events
        )

        np.testing.assert_allclose(aligned["first_stim_ts"], [3.0, 12.5])
        np.testing.assert_allclose(aligned["center_port_exit_ts"], [5.0, 12.0])
        np.testing.assert_allclose(aligned["response_ts"], [6.0, 13.0])
        np.testing.assert_allclose(aligned.loc[0, "stim_ts"], [3.0, 4.0])
        np.testing.assert_allclose(aligned.loc[1, "stim_ts"], [12.5])

    def test_missing_center_exit_does_not_assign_a_response(self):
        events = dict(self.events)
        events["center_port_exit"] = np.array([12.0])

        response_ts, _ = align_response_port_entries(
            self.trial_starts, self.responses, events
        )

        self.assertTrue(np.isnan(response_ts[0]))
        self.assertEqual(response_ts[1], 13.0)


if __name__ == "__main__":
    unittest.main()
