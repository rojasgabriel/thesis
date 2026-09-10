"""Accepted native DAMN equal-window behavior at the pinned commit."""

import unittest

import numpy as np
from damn.alignment import compute_spike_count
from damn.objects.basis_function_objects import DeltaBasis
from damn.objects.design_matrix_objects import DesignMatrix
from damn.objects.regressor_objects import EventRegressor


def native_history(spikes):
    matrix = DesignMatrix(np.array([0.0, 1.0]), 0, 0.005, 0.001)
    matrix.add_regressor(
        EventRegressor(
            "spikes",
            np.asarray(spikes),
            0.001,
            basis_objects=[DeltaBasis(0, 0.004, 0.001)],
        )
    )
    matrix.build_matrix()
    # Select positive-lag columns from the native delta basis; omit zero lag.
    return matrix.X[:, 1:]


class DamnContracts(unittest.TestCase):
    def test_history_before_window_and_no_cross_window_join(self):
        history = native_history([-0.001, 0.0])
        np.testing.assert_array_equal(history[0], [1, 0, 0])
        np.testing.assert_array_equal(history[1], [1, 1, 0])
        np.testing.assert_array_equal(history[5:], 0)

    def test_history_at_exact_window_boundary_is_truncated(self):
        np.testing.assert_array_equal(native_history([-0.003])[0], [0, 0, 0])

    def test_counts_match_the_accepted_native_grid(self):
        from damn.alignment import construct_timebins

        alignments = np.array([2.0, 6.0])
        spikes = np.array([1.901, 1.911, 1.971, 2.031, 2.971, 5.951, 6.031])
        centers, edges, _ = construct_timebins(0.1, 2.54, 0.001)
        counts, labels, _ = compute_spike_count(alignments, spikes, 0.1, 2.54, 0.001)
        np.testing.assert_array_equal(labels, centers)
        self.assertEqual(counts.shape, (2, 2639))
        for i, alignment in enumerate(alignments):
            np.testing.assert_array_equal(
                counts[i], np.histogram(spikes - alignment, edges)[0]
            )

    def test_history_at_first_bin_of_accepted_grid_is_truncated(self):
        matrix = DesignMatrix(np.array([2.0, 6.0]), 0.1, 2.54, 0.001)
        matrix.add_regressor(
            EventRegressor(
                "history",
                np.array([1.897]),
                0.001,
                basis_objects=[DeltaBasis(0, 0.004, 0.001)],
            )
        )
        matrix.build_matrix()
        np.testing.assert_array_equal(matrix.X[0, 1:], [0, 0, 0])

    def test_constant_video_uses_native_edge_truncation(self):
        from damn.alignment import generate_master_alignment_bin_times
        from damn.objects.basis_function_objects import NoBasis
        from damn.objects.regressor_objects import ContinuousRegressor

        alignments = np.array([2.0, 6.0])
        times = generate_master_alignment_bin_times(alignments, 0.1, 2.54, 0.001)
        matrix = DesignMatrix(alignments, 0.1, 2.54, 0.001)
        matrix.add_regressor(
            ContinuousRegressor(
                "video",
                times,
                np.ones(len(times)),
                0.001,
                zscore=False,
                basis_objects=[NoBasis()],
            )
        )
        matrix.build_matrix()
        expected = np.ones((5278, 1))
        expected[[0, 2638, 2639, 5277]] = 0
        np.testing.assert_array_equal(matrix.X, expected)


if __name__ == "__main__":
    unittest.main()
