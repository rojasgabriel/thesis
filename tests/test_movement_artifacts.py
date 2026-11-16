"""A unit is an artifact only when it is silent in the port and its spike is broad."""

import unittest

import numpy as np

from thesis.ephys.preprocessing.movement_artifacts import in_port_ratio, is_artifact


class MovementArtifactsTest(unittest.TestCase):
    windows = np.array([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]])

    def test_steady_unit_has_ratio_one(self):
        spikes = np.arange(0, 100, 0.01)
        self.assertAlmostEqual(in_port_ratio(spikes, self.windows, 100.0), 1.0, 2)

    def test_unit_silent_in_port_has_ratio_zero(self):
        spikes = np.arange(0, 100, 0.01)
        outside = spikes[
            ~(
                (spikes[:, None] >= self.windows[:, 0])
                & (spikes[:, None] < self.windows[:, 1])
            ).any(1)
        ]
        self.assertEqual(in_port_ratio(outside, self.windows, 100.0), 0.0)

    def test_both_rules_needed(self):
        ratio = np.array([0.05, 0.05, 0.8, 0.8])
        fw3m = np.array([0.42, 0.13, 0.42, 0.13])
        np.testing.assert_array_equal(
            is_artifact(ratio, fw3m), [True, False, False, False]
        )

    def test_no_in_port_time_raises(self):
        with self.assertRaises(ValueError):
            in_port_ratio(np.arange(10.0), np.zeros((0, 2)), 10.0)


if __name__ == "__main__":
    unittest.main()
