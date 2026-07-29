"""Tests for Stringer RRR assumption-audit helpers."""

from __future__ import annotations

import unittest

import numpy as np
from ephys.src.utils.stringer_rrr import (
    build_motion_lag_design,
    stimulus_design_from_labels,
)
from ephys.src.utils.stringer_rrr_audits import (
    audit_cv_vs_fulldata_angles,
    audit_stimulus_encoding,
    residualize_columns_by_choice,
)


class ResidualizeTests(unittest.TestCase):
    def test_residualize_removes_choice_mean(self):
        trial_ids = np.array([0, 0, 1, 1, 2, 2, 3, 3])
        choice = np.array([-1, -1, -1, -1, 1, 1, 1, 1], dtype=float)
        # Neuron 0 tracks choice exactly.
        y = np.column_stack(
            [choice.astype(float), np.random.default_rng(0).normal(size=8)]
        )
        resid = residualize_columns_by_choice(y, choice, trial_ids)
        self.assertLess(float(np.var(resid[:, 0])), 1e-12)
        # Second neuron should keep most of its variance (not choice-driven).
        self.assertGreater(float(np.var(resid[:, 1])), 0.1)


class StimulusAuditTests(unittest.TestCase):
    def test_stimulus_audit_returns_recommendation(self):
        rng = np.random.default_rng(0)
        n_trials, n_bins, n_units = 20, 4, 12
        trial_ids = np.repeat(np.arange(n_trials), n_bins)
        bin_idx = np.tile(np.arange(n_bins), n_trials)
        labels_trial = np.array([-1.0, 1.0] * (n_trials // 2))
        stim_labels = labels_trial[trial_ids]
        # Plant a weak category signal in Y.
        y = rng.normal(size=(trial_ids.size, n_units))
        y += 0.3 * stim_labels.reshape(-1, 1) * rng.normal(size=(1, n_units))
        x_beh = rng.normal(size=(trial_ids.size, 5))
        report = audit_stimulus_encoding(
            y,
            x_beh,
            stim_labels,
            trial_ids,
            bin_idx,
            n_time_bases=2,
            ranks=np.arange(1, 4),
            n_splits=4,
        )
        self.assertEqual(report["audit"], "B_stimulus_encoding")
        self.assertIn(
            report["recommendation"],
            {
                "keep_B1_trial_constant_onehot",
                "prefer_B2_category_x_time",
                "investigate_B1_first",
            },
        )
        self.assertEqual(len(report["contrast_table"]), 2)


class AngleAuditTests(unittest.TestCase):
    def test_cv_vs_fulldata_audit_runs(self):
        rng = np.random.default_rng(1)
        n_trials, n_bins, n_units = 16, 3, 10
        trial_ids = np.repeat(np.arange(n_trials), n_bins)
        labels = np.array([-1.0, 1.0] * (n_trials // 2))[trial_ids]
        x_stim = stimulus_design_from_labels(labels)
        x_beh = rng.normal(size=(trial_ids.size, 4))
        y = rng.normal(size=(trial_ids.size, n_units))
        report = audit_cv_vs_fulldata_angles(
            x_stim,
            x_beh,
            y,
            trial_ids,
            stim_rank=1,
            beh_rank=2,
            n_splits=4,
        )
        self.assertEqual(report["audit"], "D_cv_vs_fulldata_angles")
        self.assertEqual(len(report["fold_table"]), 4)
        self.assertIn("recommendation", report)


class MotionLagSmokeTests(unittest.TestCase):
    def test_build_motion_for_confound_path(self):
        motion_energy = np.linspace(1, 10, 20)
        motion_times = np.linspace(0.05, 1.95, 20)
        bin_starts = np.array([0.0, 0.1, 1.0, 1.1])
        bin_stops = np.array([0.1, 0.2, 1.1, 1.2])
        trial_ids = np.array([0, 0, 1, 1])
        x, valid, meta = build_motion_lag_design(
            motion_energy,
            motion_times,
            bin_starts,
            bin_stops,
            trial_ids,
            n_lags=1,
            lag_mode="continuous_time",
            incomplete_policy="drop",
        )
        self.assertEqual(x.shape[1], 2)
        self.assertTrue(meta["uses_prewindow_history"])
        self.assertEqual(valid.shape[0], 4)


if __name__ == "__main__":
    unittest.main()
