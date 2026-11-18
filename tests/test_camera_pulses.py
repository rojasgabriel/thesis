"""Recording-run selection must use camera timing, not excess-pulse counts."""

import sys
import unittest
from unittest.mock import patch

import numpy as np
from test_replay_trials import Relation, schema_module

from thesis.ephys import events
from thesis.ephys.preprocessing.audit_camera_pulses import (
    align_frame_times,
    audit_camera_frames,
)


class CameraPulsesTest(unittest.TestCase):
    def audit(self, falling, reference, *, n_frames=None):
        source = dict(
            subject_name="offline",
            session_name="test",
            dataset_name="ephys",
            stream_name="obx",
            event_name="io3",
        )
        times = np.column_stack([falling, falling + 0.002]).ravel()
        modules = schema_module(
            [
                dict(
                    **source,
                    event_timestamps=times,
                    event_values=np.tile([0, 1], len(falling)),
                )
            ]
        )
        modules["labdata.schema"].DatasetVideo = Relation(
            [
                dict(
                    subject_name="offline",
                    session_name="test",
                    video_name="cam0",
                    frame_times=reference - 10,
                    n_frames=len(reference) if n_frames is None else n_frames,
                ),
                dict(
                    subject_name="other",
                    session_name="test",
                    video_name="cam0",
                    frame_times=[],
                    n_frames=0,
                ),
            ]
        )
        with (
            patch.dict(sys.modules, modules),
            patch.object(
                events, "_find_sess_ev_sources", return_value={"frames": source}
            ),
            patch.object(
                events, "bpod_to_clock", return_value=lambda times: times + 10
            ),
        ):
            return audit_camera_frames("offline", "test")

    def test_audit_selects_recorded_falling_edges_in_hardware_clock(self):
        run = 20 + np.arange(100) * 0.0125
        frames, report, arrays = self.audit(np.r_[run - 10, run], run - 0.002)
        np.testing.assert_allclose(frames, run)
        self.assertEqual(report["frame_edge"], "falling")
        self.assertEqual(report["repairs"]["pulses_before_the_recording"], 100)
        np.testing.assert_allclose(arrays["stored_camera_neural_s"], run - 0.002)

    def test_audit_rejects_video_count_mismatch(self):
        run = 20 + np.arange(100) * 0.0125
        with self.assertRaisesRegex(ValueError, "video frame count"):
            self.audit(run, run, n_frames=99)

    def test_audit_rejects_empty_camera_edges(self):
        with self.assertRaisesRegex(ValueError, "camera edges"):
            self.audit(np.array([]), np.arange(100))

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

    def test_first_frame_without_pulse(self):
        frames = np.arange(100) * 0.0125
        reference = frames - 0.002 + 0.003 * np.sin(np.arange(100))
        repaired, notes = align_frame_times(frames[1:], reference)
        np.testing.assert_allclose(repaired, frames, atol=1e-9)
        self.assertEqual(notes["frame_without_pulse"], "first")
        with self.assertRaisesRegex(ValueError, "unambiguous"):
            align_frame_times(frames[1:], reference + 0.006)

    def test_unexplained_excess_pulse_fails(self):
        run = np.arange(100) * 0.0125
        with self.assertRaisesRegex(ValueError, "frame pulses"):
            align_frame_times(np.r_[run, run[-1] + 0.0125], run)

    def test_missing_reference_fails(self):
        with self.assertRaisesRegex(ValueError, "reference"):
            align_frame_times(np.arange(10), np.array([0, np.nan]))


if __name__ == "__main__":
    unittest.main()
