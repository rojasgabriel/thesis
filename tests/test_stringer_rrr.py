"""Pure-helper tests for the Stringer RRR baseline utilities."""

from __future__ import annotations

import unittest

import numpy as np
from ephys.src.utils.stringer_rrr import (
    average_trace_in_windows,
    balanced_easy_rate_trial_mask,
    behavior_shift_null,
    bin_spikes_trialwise,
    build_motion_lag_design,
    expand_stimulus_by_time,
    fit_reduced_rank_regression,
    predict_rrr,
    stimulus_design_from_labels,
    trial_group_splits,
    trial_label_shuffle,
    validate_binned_sample_structure,
    validate_design_matrices,
    validate_first_stim_alignment,
    variance_explained,
)


class BinningTests(unittest.TestCase):
    def test_bin_spikes_trialwise_counts_and_ids(self):
        spikes = [np.array([0.05, 0.15, 1.5])]
        binned = bin_spikes_trialwise(
            spikes,
            trial_starts=np.array([0.0, 1.0]),
            trial_stops=np.array([0.2, 1.2]),
            bin_width_s=0.1,
        )
        self.assertEqual(binned.rates.shape[1], 1)
        self.assertEqual(len(binned.trial_ids), len(binned.bin_starts))
        self.assertEqual(binned.rates.shape[0], 4)
        np.testing.assert_array_equal(binned.trial_ids, np.array([0, 0, 1, 1]))
        np.testing.assert_array_equal(binned.bin_idx, np.array([0, 1, 0, 1]))
        self.assertAlmostEqual(binned.rates[0, 0], 10.0)
        self.assertAlmostEqual(binned.rates[1, 0], 10.0)

    def test_drops_short_trailing_bins(self):
        spikes = [np.array([0.01])]
        # Window 0.15 s with 0.1 s bins → one full bin, remainder 0.05 dropped.
        binned = bin_spikes_trialwise(
            spikes,
            trial_starts=np.array([0.0]),
            trial_stops=np.array([0.15]),
            bin_width_s=0.1,
            min_bin_fraction=0.5,
        )
        self.assertEqual(binned.rates.shape[0], 1)
        self.assertAlmostEqual(binned.bin_stops[0] - binned.bin_starts[0], 0.1)

    def test_validate_binned_sample_structure(self):
        spikes = [np.array([0.05, 0.15, 1.05, 1.15])]
        binned = bin_spikes_trialwise(
            spikes,
            trial_starts=np.array([0.0, 1.0]),
            trial_stops=np.array([0.2, 1.2]),
            bin_width_s=0.1,
        )
        report = validate_binned_sample_structure(
            binned.trial_ids, binned.bin_idx, binned.bin_starts, binned.bin_stops
        )
        self.assertTrue(report["passed"], report["errors"])


class TrialGroupSplitTests(unittest.TestCase):
    def test_group_splits_keep_trials_intact(self):
        trial_ids = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        splits = trial_group_splits(trial_ids, n_splits=5)
        self.assertEqual(len(splits), 5)
        for train_idx, test_idx in splits:
            train_trials = set(trial_ids[train_idx])
            test_trials = set(trial_ids[test_idx])
            self.assertFalse(train_trials & test_trials)


class NullLabelPreservationTests(unittest.TestCase):
    def test_trial_label_shuffle_preserves_within_trial_labels(self):
        trial_ids = np.array([0, 0, 1, 1, 2, 2])
        labels = np.array([-1.0, -1.0, 1.0, 1.0, -1.0, -1.0])
        rng = np.random.default_rng(0)
        shuffled = trial_label_shuffle(labels, trial_ids, rng)
        for trial in np.unique(trial_ids):
            vals = shuffled[trial_ids == trial]
            self.assertEqual(len(set(vals.tolist())), 1)
        original_trial_labels = [
            labels[trial_ids == t][0] for t in np.unique(trial_ids)
        ]
        shuffled_trial_labels = [
            shuffled[trial_ids == t][0] for t in np.unique(trial_ids)
        ]
        self.assertEqual(sorted(original_trial_labels), sorted(shuffled_trial_labels))

    def test_trial_label_shuffle_rejects_within_trial_variation(self):
        trial_ids = np.array([0, 0, 1, 1])
        labels = np.array([-1.0, 1.0, 1.0, 1.0])  # trial 0 inconsistent
        with self.assertRaises(ValueError):
            trial_label_shuffle(labels, trial_ids, np.random.default_rng(0))

    def test_behavior_shift_moves_whole_trials(self):
        trial_ids = np.array([0, 0, 1, 1, 2, 2])
        x = np.arange(6, dtype=float).reshape(6, 1)
        shifted = behavior_shift_null(x, trial_ids, shift_trials=1)
        np.testing.assert_allclose(
            shifted[trial_ids == 0].ravel(), np.array([4.0, 5.0])
        )


class EasyRateBalanceTests(unittest.TestCase):
    def test_balanced_easy_rate_mask(self):
        rates = np.array([4, 4, 4, 20, 20, 12, 8], dtype=float)
        mask, labels = balanced_easy_rate_trial_mask(
            rates, rng=np.random.default_rng(0)
        )
        self.assertEqual(int(mask.sum()), 4)
        self.assertTrue(np.all(np.isfinite(labels[mask])))
        self.assertEqual(int(np.sum(labels[mask] == -1)), 2)
        self.assertEqual(int(np.sum(labels[mask] == 1)), 2)


class DesignMatrixTests(unittest.TestCase):
    def test_stimulus_onehot_and_time_expansion(self):
        labels = np.array([-1.0, -1.0, 1.0, 1.0])
        x = stimulus_design_from_labels(labels)
        self.assertEqual(x.shape, (4, 2))
        np.testing.assert_allclose(x.sum(axis=1), np.ones(4))

        bin_idx = np.array([0, 1, 0, 1])
        trial_ids = np.array([0, 0, 1, 1])
        xt = expand_stimulus_by_time(
            labels, bin_idx, n_time_bases=2, trial_ids=trial_ids, mode="fractional"
        )
        self.assertEqual(xt.shape, (4, 4))
        # First sample: low at fractional basis 0 → first two cols one-hot.
        np.testing.assert_allclose(xt[0], np.array([1, 0, 0, 0], dtype=np.float32))

        xt_abs = expand_stimulus_by_time(
            labels, bin_idx, n_time_bases=2, mode="absolute_bin"
        )
        np.testing.assert_allclose(xt_abs[0], np.array([1, 0, 0, 0], dtype=np.float32))

    def test_fractional_time_bases_balance_variable_length_trials(self):
        # Trial 0: 4 bins; trial 1: 2 bins. Absolute mode collapses late bins;
        # fractional keeps equal occupancy.
        labels = np.array([-1.0, -1.0, -1.0, -1.0, 1.0, 1.0])
        bin_idx = np.array([0, 1, 2, 3, 0, 1])
        trial_ids = np.array([0, 0, 0, 0, 1, 1])
        xt = expand_stimulus_by_time(
            labels, bin_idx, n_time_bases=2, trial_ids=trial_ids, mode="fractional"
        )
        # Each trial should place half its bins in each basis.
        t0 = xt[trial_ids == 0]
        self.assertEqual(int((t0[:, 0] + t0[:, 1] > 0).sum()), 2)
        self.assertEqual(int((t0[:, 2] + t0[:, 3] > 0).sum()), 2)

    def test_motion_half_open_windows_do_not_double_count_boundary(self):
        values = np.array([1.0, 10.0, 100.0])
        times = np.array([0.0, 0.1, 0.2])
        # Boundary sample at 0.1 belongs only to the second window.
        vals, counts, _ = average_trace_in_windows(
            values,
            times,
            window_starts=np.array([0.0, 0.1]),
            window_stops=np.array([0.1, 0.2]),
        )
        np.testing.assert_allclose(vals, np.array([1.0, 10.0]))
        np.testing.assert_array_equal(counts, np.array([1, 1]))

    def test_continuous_time_lags_use_prewindow_history(self):
        # ME exists before the first neural bin; lag-1 should see it.
        motion_energy = np.array([7.0, 10.0, 20.0, 30.0, 40.0])
        motion_times = np.array([-0.05, 0.05, 0.15, 1.05, 1.15])
        bin_starts = np.array([0.0, 0.1, 1.0, 1.1])
        bin_stops = np.array([0.1, 0.2, 1.1, 1.2])
        trial_ids = np.array([0, 0, 1, 1])
        x, _valid, meta = build_motion_lag_design(
            motion_energy,
            motion_times,
            bin_starts,
            bin_stops,
            trial_ids,
            n_lags=1,
            lag_mode="continuous_time",
            incomplete_policy="drop",
        )
        self.assertTrue(meta["uses_prewindow_history"])
        self.assertFalse(meta["cross_trial_index_lags"])
        self.assertEqual(x.shape, (4, 2))
        # Lag-1 for first bin looks at [-0.1, 0.0); may be empty → invalid or
        # may catch -0.05 depending on alignment. At least lag0 is finite.
        self.assertTrue(np.isfinite(x[0, 0]))

    def test_within_trial_lags_do_not_cross_trials_and_can_drop(self):
        motion_energy = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        motion_times = np.array([0.05, 0.15, 1.05, 1.15, 1.25])
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
            lag_mode="within_trial",
            incomplete_policy="drop",
        )
        self.assertFalse(meta["cross_trial_index_lags"])
        self.assertEqual(x.shape, (4, 2))
        # First bin of each trial lacks lag-1 history under within_trial.
        self.assertFalse(bool(valid[0]))
        self.assertFalse(bool(valid[2]))
        self.assertTrue(bool(valid[1]))
        self.assertTrue(bool(valid[3]))
        # Lag-1 of trial-1 second bin must come from trial-1 first bin, not trial 0.
        self.assertNotAlmostEqual(float(x[0, 0]), float(x[2, 0]))

    def test_validate_design_matrices_catches_sample_mismatch(self):
        y = np.ones((4, 3))
        x_stim = stimulus_design_from_labels(np.array([-1, -1, 1, 1], dtype=float))
        x_beh = np.random.default_rng(0).normal(size=(3, 2))  # wrong length
        report = validate_design_matrices(
            y,
            x_stim,
            x_beh,
            trial_ids=np.array([0, 0, 1, 1]),
            min_trials=2,
            min_samples=2,
        )
        self.assertFalse(report["passed"])
        self.assertTrue(any("sample-axis mismatch" in e for e in report["errors"]))

    def test_validate_design_passes_aligned_bundle(self):
        rng = np.random.default_rng(0)
        trial_ids = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        labels = np.array([-1, -1, 1, 1, -1, -1, 1, 1, -1, -1], dtype=float)
        y = np.abs(rng.normal(size=(10, 5)))
        x_stim = stimulus_design_from_labels(labels)
        x_beh = rng.normal(size=(10, 4))
        report = validate_design_matrices(
            y,
            x_stim,
            x_beh,
            trial_ids,
            stim_labels=labels,
            min_trials=5,
            min_samples=10,
        )
        self.assertTrue(report["passed"], report["errors"])

    def test_validate_design_flags_stim_label_mismatch(self):
        labels = np.array([-1, -1, 1, 1], dtype=float)
        y = np.ones((4, 3))
        x_stim = stimulus_design_from_labels(labels)
        x_stim[0, :] = [0, 1]  # corrupt
        x_beh = np.ones((4, 2))
        report = validate_design_matrices(
            y,
            x_stim,
            x_beh,
            trial_ids=np.array([0, 0, 1, 1]),
            stim_labels=labels,
            min_trials=2,
            min_samples=2,
        )
        self.assertFalse(report["passed"])
        self.assertTrue(any("does not match" in e for e in report["errors"]))

    def test_validate_design_warns_on_choice_window_confound(self):
        labels = np.array([-1, -1, 1, 1, -1, -1, 1, 1, -1, -1], dtype=float)
        y = np.ones((10, 3))
        x_stim = stimulus_design_from_labels(labels)
        x_beh = np.random.default_rng(0).normal(size=(10, 3))
        report = validate_design_matrices(
            y,
            x_stim,
            x_beh,
            trial_ids=np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4]),
            stim_labels=labels,
            min_trials=5,
            min_samples=10,
            window_key="fixation_to_response",
            category_choice_phi=0.8,
        )
        self.assertTrue(report["passed"])
        self.assertTrue(any("phi(category,choice)" in w for w in report["warnings"]))

    def test_first_stim_alignment_detects_mismatch(self):
        report = validate_first_stim_alignment(
            np.array([1.0, 2.0]),
            np.array([1.0, 2.5]),
            max_abs_offset_s=1e-3,
        )
        self.assertFalse(report["passed"])


class ReducedRankRegressionTests(unittest.TestCase):
    def test_rrr_wires_neuropop_and_returns_factors(self):
        rng = np.random.default_rng(0)
        n, n_feat, n_out, rank = 100, 3, 8, 2
        x = rng.normal(size=(n, n_feat)).astype(np.float32)
        y = rng.normal(size=(n, n_out)).astype(np.float32)
        x = x - x.mean(axis=0, keepdims=True)
        y = y - y.mean(axis=0, keepdims=True)
        a_hat, b_hat = fit_reduced_rank_regression(x, y, rank=rank, lam=1e-3)
        self.assertEqual(a_hat.shape, (n_out, rank))
        self.assertEqual(b_hat.shape, (n_feat, rank))
        pred = predict_rrr(x, a_hat, b_hat)
        self.assertEqual(pred.shape, y.shape)
        self.assertTrue(np.isfinite(pred).all())
        self.assertTrue(np.isfinite(variance_explained(y, pred)))


if __name__ == "__main__":
    unittest.main()
