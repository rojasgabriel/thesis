"""Initiating entries must survive exit-edge chatter and earlier failed pokes."""

import unittest

import pandas as pd

from thesis.ephys.trials import _select_task_ev_sequence


class TrialEntryTest(unittest.TestCase):
    def test_initiating_entry_not_last_edge(self):
        trial = pd.Series(
            dict(
                response=1,
                left_port_entry_times_s=[],
                right_port_entry_times_s=[3.0],
                center_port_entry_times_s=[0.0, 1.0, 1.999968],
                center_port_exit_times_s=[0.5, 2.0],
            )
        )
        selected = _select_task_ev_sequence(trial, 2.0, 1.0002)
        self.assertEqual(selected["center_entry_s"], 1.0)
        self.assertEqual(selected["center_exit_s"], 2.0)
        self.assertEqual(selected["response_port_entry_s"], 3.0)
