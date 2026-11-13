"""Bpod sync pulses from the Chipmunk labdata plugin pair with the ephys clock.

The inputs are what the plugin stores for a session with 3 saved active trials
and 2 passive replays: sync times 1, 3, 6 | 8, 11 s on the Bpod clock, and pulse
numbers from `_bpod_sync_pulse_numbers` in churchlandlab/chipmunk
(labdata_plugin/utils.py). Moved here from that repository's
tests/test_replay_sync.py so it carries no thesis dependency.
"""

import unittest

import numpy as np

from thesis.ephys.preprocessing.insert_streamsync import _strictly_matched_count

SOURCE = np.array([1.0, 3.0, 6.0, 8.0, 11.0])  # active 1, 3, 6 | replay 8, 11
NUMBERS = np.arange(5)  # _bpod_sync_pulse_numbers(3, 2)
STOPPED_NUMBERS = np.array([0, 1, 2, 4, 5])  # _bpod_sync_pulse_numbers(3, 2, 3)


def _clock(onsets, drift=1.0):
    """OneBox sync line: 100 ms high pulses at the given Bpod times, offset by 20 s."""
    onsets = 20 + drift * np.asarray(onsets, dtype=float)
    return np.column_stack((onsets, onsets + 0.1)).ravel(), np.tile([1, 0], len(onsets))


class ChipmunkSyncContractTest(unittest.TestCase):
    def test_complete_sync_sequence_matches_ephys_edges(self):
        clock, values = _clock(SOURCE)
        self.assertEqual(_strictly_matched_count(SOURCE, NUMBERS, clock, values), 5)
        with self.assertRaises(ValueError):  # replay pulses missing from the source
            _strictly_matched_count(SOURCE[:3], np.arange(3), clock, values)

    def test_stopped_trial_pulse_pairs_with_ephys_clock(self):
        def match(numbers, onsets):
            clock, values = _clock(onsets, drift=1.0001)
            return _strictly_matched_count(
                SOURCE,
                numbers,
                clock,
                values,
                clock_pulse_numbers=numbers,
                source_is_bpod=True,
            )

        # Stop cut trial 4 short at 7 s (pulse sent, not saved); replay then ran.
        self.assertEqual(match(STOPPED_NUMBERS, [1, 3, 6, 7, 8, 11]), 5)
        # Plus the trailing pulse of a replay that Stop also cut short.
        self.assertEqual(match(STOPPED_NUMBERS, [1, 3, 6, 7, 8, 11, 13]), 5)
        # Without the gap, replays would pair one pulse early: rejected, not shifted.
        with self.assertRaises(ValueError):
            match(NUMBERS, [1, 3, 6, 7, 8, 11])
        # A marker without its physical pulse is rejected too.
        with self.assertRaises(ValueError):
            match(STOPPED_NUMBERS, [1, 3, 6, 8, 11])


if __name__ == "__main__":
    unittest.main()
