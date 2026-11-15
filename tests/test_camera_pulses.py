"""Recording-run selection must use camera timing, not excess-pulse counts."""

import unittest

import numpy as np

from thesis.ephys.preprocessing.audit_camera_pulses import align_frame_times


class CameraPulsesTest(unittest.TestCase):
    def test_long_preview_and_restart(self):
        preview = np.arange(33000) * 0.0125
        restart = preview[-1] + 2 + np.arange(1200) * 0.0125
        recorded = restart[-1] + 2 + np.arange(100) * 0.0125
        frames, notes = align_frame_times(
            np.r_[preview, restart, recorded], recorded + 0.004
        )
        np.testing.assert_array_equal(frames, recorded)
        self.assertEqual(notes["pulses_before_the_recording"], 34200)
        self.assertEqual(notes["irregular_places"], [])

    def test_equal_count_runs_use_independent_reference(self):
        run = np.arange(100) * 0.0125
        frames, notes = align_frame_times(np.r_[run, run + 10, run + 20], run + 10)
        np.testing.assert_array_equal(frames, run + 10)
        self.assertEqual(notes["pulses_before_the_recording"], 100)
        self.assertEqual(notes["pulses_after_the_recording"], 100)

    def test_count_alone_is_insufficient(self):
        run = np.arange(100) * 0.0125
        preview = np.arange(3000) * 0.0125
        with self.assertRaisesRegex(ValueError, "frame pulses"):
            align_frame_times(np.r_[preview, run + 40], run + 41)

    def test_legacy_missing_middle_pulse_repair_is_reported(self):
        run = np.arange(100) * 0.0125
        frames, notes = align_frame_times(np.delete(run, 50), run)
        np.testing.assert_allclose(frames, run)
        self.assertEqual(notes["irregular_places"][0]["frames_placed"], 1)

    def test_unexplained_excess_pulse_fails(self):
        run = np.arange(100) * 0.0125
        with self.assertRaisesRegex(ValueError, "frame pulses"):
            align_frame_times(np.r_[run, run[-1] + 0.0125], run)

    def test_missing_reference_fails(self):
        with self.assertRaisesRegex(ValueError, "reference"):
            align_frame_times(np.arange(10), np.array([0, np.nan]))


if __name__ == "__main__":
    unittest.main()
