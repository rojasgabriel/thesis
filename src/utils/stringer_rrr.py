"""Pure helpers for a minimal Stringer-style reduced-rank-regression baseline.

These functions are intentionally free of labdata / video I/O so they can be
unit-tested without the lab DB. The analysis script wires them to live loaders.

Design-matrix contract
----------------------
All matrices share a **sample axis** of trial-concatenated time bins:

* ``Y``: ``(n_samples, n_units)`` — spike rates (sp/s) in each bin
* ``X_stim``: ``(n_samples, n_stim_features)`` — stimulus regressors aligned to bins
* ``X_beh``: ``(n_samples, n_beh_features)`` — behavior regressors aligned to bins
* ``trial_ids``: ``(n_samples,)`` — trial index for GroupKFold (never split a trial)

This stacked ``(trial, bin)`` layout is the task-embedded analogue of Stringer's
continuous-time sample axis. It is **not** the same as:

* trial-averaged vectors (one row per trial), or
* a single long continuous session timeline without trial boundaries.

Lagged behavior features use **continuous-time lookback** on the session ME
trace by default (not index-lags on the stacked axis). Within-trial index lags
are available as an ablation but drop or impute early bins.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import GroupKFold

ArrayF = NDArray[np.floating]


@dataclass(frozen=True)
class RankSelection:
    """Held-out rank choice and the curve used to pick it."""

    selected_rank: int
    ranks: NDArray[np.int_]
    mean_varexp: ArrayF
    sem_varexp: ArrayF
    rule: str


@dataclass(frozen=True)
class BinnedSpikes:
    """Trial-concatenated neural samples with alignment metadata."""

    rates: ArrayF
    trial_ids: NDArray[np.int_]
    bin_idx: NDArray[np.int_]
    bin_starts: ArrayF
    bin_stops: ArrayF
    bin_width_s: float
    n_dropped_short_bins: int


def bin_spikes_trialwise(
    spike_times_per_unit: list[ArrayF],
    trial_starts: ArrayF,
    trial_stops: ArrayF,
    bin_width_s: float = 0.1,
    min_bin_fraction: float = 0.5,
) -> BinnedSpikes:
    """Bin spike counts into fixed-width bins within each trial window.

    Only bins with duration ``>= min_bin_fraction * bin_width_s`` are kept so
    truncated trailing windows do not enter the design as short, high-variance
    rate estimates.

    Spike trains are sorted before counting. Rates are spikes/s.
    """
    if bin_width_s <= 0:
        raise ValueError("bin_width_s must be positive")
    if not 0 < min_bin_fraction <= 1:
        raise ValueError("min_bin_fraction must be in (0, 1]")
    trial_starts = np.asarray(trial_starts, dtype=float)
    trial_stops = np.asarray(trial_stops, dtype=float)
    if trial_starts.shape != trial_stops.shape:
        raise ValueError("trial_starts and trial_stops must have the same shape")

    spike_lists = [
        np.sort(np.asarray(st, dtype=float).ravel()) for st in spike_times_per_unit
    ]
    for unit_idx, spikes in enumerate(spike_lists):
        if spikes.size and not np.all(np.isfinite(spikes)):
            raise ValueError(f"unit {unit_idx} has non-finite spike times")
    n_units = len(spike_lists)
    rate_rows: list[ArrayF] = []
    trial_ids: list[int] = []
    bin_idx: list[int] = []
    starts: list[float] = []
    stops: list[float] = []
    n_dropped_short = 0
    min_dur = float(min_bin_fraction * bin_width_s)

    for trial_i, (start, stop) in enumerate(zip(trial_starts, trial_stops)):
        if not np.isfinite(start) or not np.isfinite(stop) or stop <= start:
            continue
        # Fixed-width edges only; do not force a short trailing remainder bin.
        edges = np.arange(start, stop, bin_width_s, dtype=float)
        if edges.size == 0:
            continue
        if edges[-1] < stop:
            remainder = stop - edges[-1]
            if remainder >= min_dur:
                edges = np.r_[edges, stop]
            # else drop the short trailing fragment
        if edges.size < 2:
            continue
        local_bin = 0
        for left, right in itertools.pairwise(edges):
            duration = float(right - left)
            if duration < min_dur:
                n_dropped_short += 1
                continue
            counts = np.empty(n_units, dtype=float)
            for unit_idx, spikes in enumerate(spike_lists):
                counts[unit_idx] = np.searchsorted(spikes, right) - np.searchsorted(
                    spikes, left
                )
            rate_rows.append(counts / duration)
            trial_ids.append(trial_i)
            bin_idx.append(local_bin)
            starts.append(float(left))
            stops.append(float(right))
            local_bin += 1

    if not rate_rows:
        empty = np.zeros((0, n_units), dtype=float)
        return BinnedSpikes(
            rates=empty,
            trial_ids=np.zeros(0, dtype=int),
            bin_idx=np.zeros(0, dtype=int),
            bin_starts=np.zeros(0),
            bin_stops=np.zeros(0),
            bin_width_s=float(bin_width_s),
            n_dropped_short_bins=n_dropped_short,
        )

    rates = np.vstack(rate_rows)
    if np.any(rates < 0):
        raise ValueError("negative spike rates — spike counting bug")
    return BinnedSpikes(
        rates=rates,
        trial_ids=np.asarray(trial_ids, dtype=int),
        bin_idx=np.asarray(bin_idx, dtype=int),
        bin_starts=np.asarray(starts, dtype=float),
        bin_stops=np.asarray(stops, dtype=float),
        bin_width_s=float(bin_width_s),
        n_dropped_short_bins=n_dropped_short,
    )


def balanced_easy_rate_trial_mask(
    stim_rates: ArrayF,
    low_hz: float = 4.0,
    high_hz: float = 20.0,
    atol: float = 0.1,
    rng: np.random.Generator | None = None,
) -> tuple[NDArray[np.bool_], ArrayF]:
    """Keep balanced 4 vs 20 Hz trials; return mask and ±1 stimulus labels."""
    rates = np.asarray(stim_rates, dtype=float)
    is_low = np.isclose(rates, low_hz, atol=atol)
    is_high = np.isclose(rates, high_hz, atol=atol)
    n_low = int(is_low.sum())
    n_high = int(is_high.sum())
    n_keep = min(n_low, n_high)
    if n_keep == 0:
        labels = np.full(rates.shape, np.nan, dtype=float)
        return np.zeros(rates.shape, dtype=bool), labels

    low_idx = np.flatnonzero(is_low)
    high_idx = np.flatnonzero(is_high)
    if rng is None:
        rng = np.random.default_rng(0)
    if n_low > n_keep:
        low_idx = rng.choice(low_idx, size=n_keep, replace=False)
        low_idx.sort()
    if n_high > n_keep:
        high_idx = rng.choice(high_idx, size=n_keep, replace=False)
        high_idx.sort()
    balanced = np.zeros(rates.shape, dtype=bool)
    balanced[low_idx] = True
    balanced[high_idx] = True
    labels = np.full(rates.shape, np.nan, dtype=float)
    labels[low_idx] = -1.0
    labels[high_idx] = 1.0
    return balanced, labels


def trial_group_splits(
    trial_ids: ArrayF,
    n_splits: int = 5,
) -> list[tuple[NDArray[np.int_], NDArray[np.int_]]]:
    """GroupKFold splits that never put bins from one trial into both folds."""
    trial_ids = np.asarray(trial_ids)
    if trial_ids.size == 0:
        return []
    n_groups = int(np.unique(trial_ids).size)
    n_splits = min(n_splits, n_groups)
    if n_splits < 2:
        raise ValueError("Need at least two trials for group cross-validation")
    splitter = GroupKFold(n_splits=n_splits)
    dummy = np.zeros(trial_ids.shape[0])
    return list(splitter.split(dummy, dummy, groups=trial_ids))


def zscore_columns(x: ArrayF, eps: float = 1e-8) -> ArrayF:
    """Z-score each column; columns with near-zero std become zeros."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x.copy()
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.where(std < eps, 1.0, std)
    out = (x - mean) / std
    out[:, np.squeeze(std) < eps] = 0.0
    return out


def fit_reduced_rank_regression(
    x: ArrayF,
    y: ArrayF,
    rank: int,
    lam: float = 1e-3,
    device: str | None = None,
) -> tuple[ArrayF, ArrayF]:
    """Fit Y ≈ X @ B @ A.T via neuropop's reduced-rank regression.

    Uses ``neuropop.linear_prediction.reduced_rank_regression`` (Stringer /
    Pachitariu). Caller should mean-center or z-score columns first — neuropop
    assumes that.

    ``A`` has shape ``(n_outputs, rank)`` and spans the neural subspace.
    ``B`` has shape ``(n_features, rank)``.
    """
    import torch
    from neuropop.linear_prediction import reduced_rank_regression

    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("x and y must be 2-D")
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must share the sample axis")
    if rank < 1:
        raise ValueError("rank must be >= 1")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("x and y must be finite before RRR")

    # neuropop caps rank at min(n_out, n_samples, n_in) - 1.
    max_rank = min(y.shape[1], x.shape[0], x.shape[1]) - 1
    if max_rank < 1:
        raise ValueError(
            f"neuropop RRR needs at least 2 usable dimensions; got "
            f"n_samples={x.shape[0]}, n_features={x.shape[1]}, n_outputs={y.shape[1]}"
        )
    rank = min(int(rank), max_rank)

    torch_device = torch.device(device or "cpu")
    x_t = torch.from_numpy(x).to(torch_device)
    y_t = torch.from_numpy(y).to(torch_device)
    a_t, b_t = reduced_rank_regression(x_t, y_t, rank=rank, lam=float(lam))
    return a_t.detach().cpu().numpy(), b_t.detach().cpu().numpy()


def predict_rrr(x: ArrayF, a: ArrayF, b: ArrayF) -> ArrayF:
    """Predict Y from X given RRR factors A, B (neuropop convention)."""
    return np.asarray(x, dtype=float) @ b @ a.T


def variance_explained(y_true: ArrayF, y_pred: ArrayF) -> float:
    """Fraction of variance explained (1 - SSE / SST), can be negative."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    resid = y_true - y_pred
    sse = float(np.sum(resid**2))
    sst = float(np.sum((y_true - y_true.mean(axis=0, keepdims=True)) ** 2))
    if sst <= 0:
        return 0.0
    return 1.0 - sse / sst


def _zscore_with_train_stats(
    train: ArrayF, test: ArrayF, eps: float = 1e-8
) -> tuple[ArrayF, ArrayF]:
    """Z-score train/test using train-set mean and std only."""
    train = np.asarray(train, dtype=np.float32)
    test = np.asarray(test, dtype=np.float32)
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std = np.where(std < eps, 1.0, std)
    train_z = (train - mean) / std
    test_z = (test - mean) / std
    zero_cols = np.squeeze(std) < eps
    train_z[:, zero_cols] = 0.0
    test_z[:, zero_cols] = 0.0
    return train_z, test_z


def _pca_fit_transform_train(
    train: ArrayF, test: ArrayF, n_pcs: int
) -> tuple[ArrayF, ArrayF]:
    """PCA fit on train only, then transform train/test (no leakage)."""
    train = np.asarray(train, dtype=np.float32)
    test = np.asarray(test, dtype=np.float32)
    centered = train - train.mean(axis=0, keepdims=True)
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    n_keep = min(int(n_pcs), vt.shape[0], train.shape[1])
    components = vt[:n_keep]
    mean = train.mean(axis=0, keepdims=True)
    return (train - mean) @ components.T, (test - mean) @ components.T


def cross_validated_rrr_curve(
    x: ArrayF,
    y: ArrayF,
    trial_ids: ArrayF,
    ranks: ArrayF | list[int],
    n_splits: int = 5,
    lam: float = 1e-3,
    device: str | None = None,
    n_pcs: int | None = None,
) -> tuple[ArrayF, ArrayF, ArrayF]:
    """Held-out VE vs rank using neuropop RRR and trial-grouped folds.

    For each GroupKFold split, fits neuropop ``linear_prediction`` once at the
    max requested rank with ``allranks=True``, keeping train/test grouped by
    trial. Optional ``n_pcs`` applies PCA **inside each fold** (fit on train
    only) so behavior dimensionality reduction does not leak test information.
    """
    import torch
    from neuropop.linear_prediction import linear_prediction

    ranks = np.asarray(ranks, dtype=int)
    if ranks.size == 0:
        raise ValueError("ranks must be non-empty")
    max_rank = int(np.max(ranks))
    torch_device = torch.device(device or "cpu")
    ve_by_fold: list[list[float]] = [[] for _ in ranks]

    for train_idx, test_idx in trial_group_splits(trial_ids, n_splits=n_splits):
        x_train = np.asarray(x[train_idx], dtype=np.float32)
        x_test = np.asarray(x[test_idx], dtype=np.float32)
        if n_pcs is not None and n_pcs > 0:
            x_train, x_test = _pca_fit_transform_train(x_train, x_test, n_pcs)
        x_train, x_test = _zscore_with_train_stats(x_train, x_test)
        y_train, y_test = _zscore_with_train_stats(y[train_idx], y[test_idx])
        # neuropop's linear_prediction takes full arrays + index sets.
        x_all = np.vstack([x_train, x_test]).astype(np.float32)
        y_all = np.vstack([y_train, y_test]).astype(np.float32)
        itrain = np.arange(x_train.shape[0], dtype=int)
        itest = np.arange(
            x_train.shape[0], x_train.shape[0] + x_test.shape[0], dtype=int
        )
        usable_max = min(y_all.shape[1], x_all.shape[0], x_all.shape[1]) - 1
        if usable_max < 1:
            raise ValueError(
                "neuropop RRR cannot run: need >=2 predictor dimensions after "
                f"encoding (n_features={x_all.shape[1]})"
            )
        fit_rank = min(max_rank, usable_max)
        _pred, ve, _itest, _a, _b, _vf, _cf = linear_prediction(
            x_all,
            y_all,
            rank=fit_rank,
            lam=float(lam),
            allranks=True,
            itrain=itrain,
            itest=itest,
            device=torch_device,
        )
        ve = np.atleast_1d(np.asarray(ve, dtype=float)).reshape(-1)
        for rank_i, rank in enumerate(ranks):
            # ve[r] is variance explained using components 0..r (rank r+1).
            if int(rank) > ve.shape[0]:
                ve_by_fold[rank_i].append(float("nan"))
            else:
                ve_by_fold[rank_i].append(float(ve[int(rank) - 1]))

    mean = np.asarray(
        [np.nanmean(v) if len(v) else np.nan for v in ve_by_fold], dtype=float
    )
    sem = np.asarray(
        [
            np.nanstd(v, ddof=1) / np.sqrt(np.sum(np.isfinite(v)))
            if np.sum(np.isfinite(v)) > 1
            else 0.0
            for v in ve_by_fold
        ],
        dtype=float,
    )
    return ranks, mean, sem


def select_rank_one_se(
    ranks: ArrayF,
    mean_varexp: ArrayF,
    sem_varexp: ArrayF,
) -> RankSelection:
    """Pick the smallest rank within one SEM of the peak held-out VE."""
    ranks = np.asarray(ranks, dtype=int)
    mean_varexp = np.asarray(mean_varexp, dtype=float)
    sem_varexp = np.asarray(sem_varexp, dtype=float)
    if not np.any(np.isfinite(mean_varexp)):
        raise ValueError("mean_varexp has no finite entries")
    best_i = int(np.nanargmax(mean_varexp))
    threshold = mean_varexp[best_i] - sem_varexp[best_i]
    eligible = np.flatnonzero(np.isfinite(mean_varexp) & (mean_varexp >= threshold))
    selected = int(ranks[eligible[0]])
    return RankSelection(
        selected_rank=selected,
        ranks=ranks,
        mean_varexp=mean_varexp,
        sem_varexp=sem_varexp,
        rule="one_se",
    )


def select_rank_knee(
    ranks: ArrayF,
    mean_varexp: ArrayF,
    sem_varexp: ArrayF | None = None,
) -> RankSelection:
    """Pick the knee of the VE-vs-rank curve (max distance from chord)."""
    ranks = np.asarray(ranks, dtype=int)
    mean_varexp = np.asarray(mean_varexp, dtype=float)
    if sem_varexp is None:
        sem_varexp = np.zeros_like(mean_varexp)
    else:
        sem_varexp = np.asarray(sem_varexp, dtype=float)
    if ranks.size == 1:
        selected = int(ranks[0])
    else:
        x = (ranks - ranks.min()).astype(float)
        y = mean_varexp - np.nanmin(mean_varexp)
        x = x / (x.max() if x.max() > 0 else 1.0)
        y_range = np.nanmax(y) if np.isfinite(y).any() else 0.0
        y = y / (y_range if y_range > 0 else 1.0)
        line = np.array([x[-1] - x[0], y[-1] - y[0]])
        line_norm = np.linalg.norm(line)
        if line_norm == 0:
            selected = int(ranks[int(np.nanargmax(mean_varexp))])
        else:
            dists = []
            for xi, yi in zip(x, y):
                vec = np.array([xi - x[0], yi - y[0]])
                dists.append(abs(line[0] * vec[1] - line[1] * vec[0]) / line_norm)
            selected = int(ranks[int(np.argmax(dists))])
    return RankSelection(
        selected_rank=selected,
        ranks=ranks,
        mean_varexp=mean_varexp,
        sem_varexp=sem_varexp,
        rule="knee",
    )


def subspace_basis_from_rrr(a: ArrayF) -> ArrayF:
    """Orthonormalize neural subspace basis columns from RRR factor A."""
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return a.reshape(a.shape[0], 0)
    q, _r = np.linalg.qr(a, mode="reduced")
    return q


def principal_angles_deg(basis_a: ArrayF, basis_b: ArrayF) -> ArrayF:
    """Principal angles (degrees) between two subspaces."""
    from scipy.linalg import subspace_angles

    ua = subspace_basis_from_rrr(basis_a)
    ub = subspace_basis_from_rrr(basis_b)
    if ua.size == 0 or ub.size == 0:
        return np.zeros(0, dtype=float)
    return np.degrees(subspace_angles(ua, ub))


def projection_fraction(source_basis: ArrayF, target_basis: ArrayF) -> float:
    """Fraction of source subspace energy falling in the target subspace."""
    us = subspace_basis_from_rrr(source_basis)
    ut = subspace_basis_from_rrr(target_basis)
    if us.size == 0 or ut.size == 0:
        return float("nan")
    proj = ut @ (ut.T @ us)
    num = float(np.sum(proj**2))
    den = float(np.sum(us**2))
    return num / den if den > 0 else float("nan")


def trial_label_shuffle(
    labels_by_trial: ArrayF,
    trial_ids: ArrayF,
    rng: np.random.Generator,
) -> ArrayF:
    """Shuffle trial-level labels and broadcast to sample rows.

    Sample-level labels are rebuilt from the shuffled trial map so within-trial
    label consistency is preserved (null-label preservation).
    """
    labels_by_trial = np.asarray(labels_by_trial, dtype=float)
    trial_ids = np.asarray(trial_ids, dtype=int)
    unique_trials = np.unique(trial_ids)
    trial_to_label = {}
    for trial in unique_trials:
        sample_i = int(np.flatnonzero(trial_ids == trial)[0])
        if labels_by_trial.shape[0] == trial_ids.shape[0]:
            trial_to_label[int(trial)] = labels_by_trial[sample_i]
        else:
            trial_to_label[int(trial)] = labels_by_trial[int(trial)]
    # Guard: labels must be constant within trial when sample-aligned.
    if labels_by_trial.shape[0] == trial_ids.shape[0]:
        for trial in unique_trials:
            vals = labels_by_trial[trial_ids == trial]
            if not np.allclose(vals, vals[0]):
                raise ValueError(
                    f"stimulus labels vary within trial {int(trial)}; "
                    "trial-constant stimulus encoding required"
                )
    shuffled_trials = unique_trials.copy()
    rng.shuffle(shuffled_trials)
    remap = {
        int(src): trial_to_label[int(dst)]
        for src, dst in zip(unique_trials, shuffled_trials)
    }
    return np.asarray([remap[int(t)] for t in trial_ids], dtype=float)


def behavior_shift_null(
    x_beh: ArrayF,
    trial_ids: ArrayF,
    shift_trials: int,
) -> ArrayF:
    """Circularly reassign whole-trial behavior blocks (variable-length safe)."""
    x_beh = np.asarray(x_beh, dtype=float)
    trial_ids = np.asarray(trial_ids, dtype=int)
    unique_trials = np.unique(trial_ids)
    if unique_trials.size == 0:
        return x_beh.copy()
    shift_trials = int(shift_trials) % unique_trials.size
    trial_to_rows = {int(t): np.flatnonzero(trial_ids == t) for t in unique_trials}
    # Map destination trial -> source trial features truncated/padded in time.
    shifted = np.empty_like(x_beh)
    for i, trial in enumerate(unique_trials):
        src_trial = unique_trials[(i - shift_trials) % unique_trials.size]
        src_rows = trial_to_rows[int(src_trial)]
        dst_rows = trial_to_rows[int(trial)]
        n = min(src_rows.size, dst_rows.size)
        shifted[dst_rows[:n]] = x_beh[src_rows[:n]]
        if dst_rows.size > n:
            # Pad leftover bins by repeating the last available source row.
            shifted[dst_rows[n:]] = x_beh[src_rows[-1]]
    return shifted


def random_subspace_basis(
    n_neurons: int,
    rank: int,
    rng: np.random.Generator,
) -> ArrayF:
    """Rank-matched random orthonormal neural basis (random null)."""
    raw = rng.normal(size=(n_neurons, rank))
    q, _r = np.linalg.qr(raw, mode="reduced")
    return q[:, :rank]


def average_trace_in_windows(
    values: ArrayF,
    times: ArrayF,
    window_starts: ArrayF,
    window_stops: ArrayF,
    *,
    allow_nearest_fill: bool = False,
) -> tuple[ArrayF, NDArray[np.int_], int]:
    """Mean of ``values`` inside each half-open ``[start, stop)`` window.

    Half-open intervals match spike binning so adjacent bins never double-count
    a sample that lands exactly on a boundary.

    By default, empty windows stay NaN (no silent nearest-neighbor fill). Set
    ``allow_nearest_fill=True`` only for diagnostic event-triggered plots.
    """
    values = np.asarray(values, dtype=float)
    times = np.asarray(times, dtype=float)
    starts = np.asarray(window_starts, dtype=float)
    stops = np.asarray(window_stops, dtype=float)
    if times.size and np.any(np.diff(times) < 0):
        raise ValueError("motion/frame times must be sorted ascending")
    out = np.full(starts.shape, np.nan, dtype=float)
    counts = np.zeros(starts.shape, dtype=int)
    nearest_filled = 0
    for index in np.ndindex(starts.shape):
        start = starts[index]
        stop = stops[index]
        if not np.isfinite(start) or not np.isfinite(stop) or stop <= start:
            continue
        # Half-open [start, stop): both edges use side="left".
        on = np.searchsorted(times, start, side="left")
        off = np.searchsorted(times, stop, side="left")
        if off > on:
            out[index] = values[on:off].mean()
            counts[index] = off - on
        elif allow_nearest_fill and times.size:
            center = 0.5 * (start + stop)
            nearest = int(np.clip(np.searchsorted(times, center), 0, len(times) - 1))
            out[index] = values[nearest]
            nearest_filled += 1
    return out, counts, nearest_filled


def build_motion_lag_design(
    motion_energy: ArrayF,
    motion_times: ArrayF,
    bin_starts: ArrayF,
    bin_stops: ArrayF,
    trial_ids: ArrayF,
    n_lags: int = 4,
    *,
    lag_mode: str = "continuous_time",
    incomplete_policy: str = "error",
) -> tuple[ArrayF, NDArray[np.bool_], dict]:
    """Build motion-energy lag features aligned to neural bins.

    Scientific note
    ---------------
    Stringer used PCA over **spatial** motion-energy pixels (face movie). This
    helper uses a **scalar** ME trace (mean |Δframe|) because that is what the
    category-decoding cache provides. That is a related but weaker behavior
    basis — results must not be described as spatial ME-PCs.

    Lag modes
    ---------
    ``continuous_time`` (default, preferred for freely moving tasks)
        Lag *k* at a bin is the mean ME in the time window shifted *k* bin-widths
        earlier on the **session ME trace**. Early bins correctly inherit ITI /
        pre-window motion. This does **not** index-lag the stacked sample axis
        (which would invent cross-trial structure from concatenation order).

    ``within_trial``
        Causal lags using only ME bins inside the same trial. Early bins lack
        history; handle with ``incomplete_policy``.

    Incomplete policies (``within_trial`` only)
    -------------------------------------------
    ``error`` — raise if any lag is missing (forces an explicit choice).
    ``drop`` — mark those sample rows invalid (``valid_mask`` False); no fill.
    ``impute_trial_first`` — fill with the trial's first finite ME (legacy;
    invents temporal structure — avoid for claims).
    """
    trial_ids = np.asarray(trial_ids, dtype=int)
    bin_starts = np.asarray(bin_starts, dtype=float).reshape(-1)
    bin_stops = np.asarray(bin_stops, dtype=float).reshape(-1)
    if bin_starts.shape[0] != trial_ids.shape[0]:
        raise ValueError("bin_starts and trial_ids length mismatch")
    if bin_stops.shape[0] != trial_ids.shape[0]:
        raise ValueError("bin_stops and trial_ids length mismatch")

    n_samples = trial_ids.shape[0]
    n_lags = max(0, int(n_lags))
    lag_features = np.full((n_samples, n_lags + 1), np.nan, dtype=float)
    widths = bin_stops - bin_starts
    median_width = float(np.nanmedian(widths)) if widths.size else 0.1
    if not np.isfinite(median_width) or median_width <= 0:
        median_width = 0.1

    lag0, counts, nearest = average_trace_in_windows(
        motion_energy,
        motion_times,
        bin_starts,
        bin_stops,
        allow_nearest_fill=False,
    )
    lag0 = np.asarray(lag0, dtype=float).reshape(-1)
    lag_features[:, 0] = lag0

    if lag_mode == "continuous_time":
        for lag in range(1, n_lags + 1):
            shift = lag * median_width
            lagged, lag_counts, _ = average_trace_in_windows(
                motion_energy,
                motion_times,
                bin_starts - shift,
                bin_stops - shift,
                allow_nearest_fill=False,
            )
            lag_features[:, lag] = np.asarray(lagged, dtype=float).reshape(-1)
            counts = counts + lag_counts
        construction = "scalar_me_continuous_time_lags"
        cross_trial_index_lags = False
        uses_prewindow_history = True
    elif lag_mode == "within_trial":
        for trial in np.unique(trial_ids):
            rows = np.flatnonzero(trial_ids == trial)
            series = lag0[rows]
            for lag in range(1, n_lags + 1):
                lag_features[rows[lag:], lag] = series[:-lag]
        construction = "scalar_me_within_trial_lags"
        cross_trial_index_lags = False
        uses_prewindow_history = False
    else:
        raise ValueError(
            f"unknown lag_mode={lag_mode!r}; use 'continuous_time' or 'within_trial'"
        )

    n_nan = int(np.isnan(lag_features).sum())
    n_imputed = 0
    valid_mask = np.isfinite(lag_features).all(axis=1)

    if incomplete_policy == "drop":
        pass  # valid_mask already marks complete rows
    elif incomplete_policy == "impute_trial_first":
        for trial in np.unique(trial_ids):
            rows = np.flatnonzero(trial_ids == trial)
            block = lag_features[rows]
            for col in range(block.shape[1]):
                col_vals = block[:, col]
                if np.isnan(col_vals).all():
                    continue
                first = col_vals[np.isfinite(col_vals)][0]
                missing = ~np.isfinite(col_vals)
                n_imputed += int(missing.sum())
                col_vals[missing] = first
                block[:, col] = col_vals
            lag_features[rows] = block
        valid_mask = np.isfinite(lag_features).all(axis=1)
    elif incomplete_policy == "error":
        if not valid_mask.all():
            n_bad = int((~valid_mask).sum())
            raise ValueError(
                f"{n_bad} samples have incomplete ME lags under lag_mode="
                f"{lag_mode!r}. Use continuous_time (default), incomplete_policy="
                "'drop', or (discouraged) 'impute_trial_first'."
            )
    else:
        raise ValueError(f"unknown incomplete_policy={incomplete_policy!r}")

    meta = {
        "construction": construction,
        "lag_mode": lag_mode,
        "incomplete_policy": incomplete_policy,
        "stringer_analogue": (
            "scalar ME lags approximate facial motion energy; NOT spatial ME-PCs"
        ),
        "n_lags": n_lags,
        "n_features": int(lag_features.shape[1]),
        "median_bin_width_s": median_width,
        "n_bins_with_frames": int((np.asarray(counts).reshape(-1) > 0).sum()),
        "n_empty_lag0_windows": int(np.isnan(lag0).sum()),
        "n_nearest_filled": int(nearest),
        "n_nan_before_policy": n_nan,
        "n_imputed_leading_lags": n_imputed,
        "n_invalid_samples": int((~valid_mask).sum()),
        "mean_frames_per_bin": float(np.nanmean(counts)) if np.size(counts) else 0.0,
        "spatial_me_pcs": False,
        "cross_trial_index_lags": cross_trial_index_lags,
        "uses_prewindow_history": uses_prewindow_history,
    }
    if incomplete_policy != "drop" and not np.isfinite(lag_features).all():
        raise ValueError("behavior design still contains NaNs after incomplete policy")
    return lag_features.astype(np.float32), valid_mask, meta


def validate_binned_sample_structure(
    trial_ids: ArrayF,
    bin_idx: ArrayF,
    bin_starts: ArrayF,
    bin_stops: ArrayF,
    *,
    atol: float = 1e-9,
) -> dict:
    """Check trial-concatenated bins are ordered, contiguous, and well-formed."""
    trial_ids = np.asarray(trial_ids, dtype=int)
    bin_idx = np.asarray(bin_idx, dtype=int)
    bin_starts = np.asarray(bin_starts, dtype=float)
    bin_stops = np.asarray(bin_stops, dtype=float)
    report: dict = {"passed": True, "errors": [], "warnings": [], "n_trials": 0}
    n = trial_ids.shape[0]
    if not (bin_idx.shape[0] == bin_starts.shape[0] == bin_stops.shape[0] == n):
        report["errors"].append("bin metadata length mismatch vs trial_ids")
        report["passed"] = False
        return report
    if n == 0:
        report["errors"].append("no binned samples")
        report["passed"] = False
        return report
    if np.any(bin_stops <= bin_starts):
        report["errors"].append("found bins with stop <= start")
    if not np.isfinite(bin_starts).all() or not np.isfinite(bin_stops).all():
        report["errors"].append("bin_starts/stops contain non-finite values")

    widths = bin_stops - bin_starts
    report["bin_width_median_s"] = float(np.median(widths))
    report["bin_width_min_s"] = float(np.min(widths))
    report["bin_width_max_s"] = float(np.max(widths))
    if np.max(widths) - np.min(widths) > 0.51 * report["bin_width_median_s"]:
        report["warnings"].append(
            "bin widths vary substantially; trailing partial bins may remain"
        )

    for trial in np.unique(trial_ids):
        rows = np.flatnonzero(trial_ids == trial)
        # Samples from one trial must be a contiguous block in the stacked axis.
        if np.any(np.diff(rows) != 1):
            report["errors"].append(
                f"trial {int(trial)} samples are not contiguous in the stacked axis"
            )
            continue
        local_idx = bin_idx[rows]
        if local_idx.size and local_idx[0] != 0:
            report["warnings"].append(
                f"trial {int(trial)} does not start at bin_idx 0 "
                f"(starts at {int(local_idx[0])}) — incomplete-lag dropping?"
            )
        if local_idx.size > 1 and np.any(np.diff(local_idx) < 1):
            report["errors"].append(
                f"trial {int(trial)} bin_idx is not strictly increasing"
            )
        starts = bin_starts[rows]
        stops = bin_stops[rows]
        if starts.size > 1 and np.any(np.diff(starts) <= 0):
            report["errors"].append(f"trial {int(trial)} bin_starts not increasing")
        # Contiguity: stop of bin k should equal start of bin k+1.
        if starts.size > 1:
            gaps = starts[1:] - stops[:-1]
            if np.any(np.abs(gaps) > atol) and np.any(gaps < -atol):
                report["errors"].append(
                    f"trial {int(trial)} has overlapping bins (negative gaps)"
                )
            n_gaps = int(np.sum(np.abs(gaps) > atol))
            if n_gaps:
                report["warnings"].append(
                    f"trial {int(trial)} has {n_gaps} non-contiguous bin edges "
                    "(dropped short bins leave gaps — expected if min_bin_fraction "
                    "dropped a middle fragment)"
                )
    report["n_trials"] = int(np.unique(trial_ids).size)
    report["n_samples"] = n
    report["passed"] = len(report["errors"]) == 0
    return report


def validate_design_matrices(
    y: ArrayF,
    x_stim: ArrayF,
    x_beh: ArrayF,
    trial_ids: ArrayF,
    stim_labels: ArrayF | None = None,
    *,
    min_trials: int = 10,
    min_samples: int = 50,
    window_key: str | None = None,
    category_choice_phi: float | None = None,
) -> dict:
    """Assert sample-axis alignment and basic scientific format constraints."""
    y = np.asarray(y, dtype=float)
    x_stim = np.asarray(x_stim, dtype=float)
    x_beh = np.asarray(x_beh, dtype=float)
    trial_ids = np.asarray(trial_ids, dtype=int)
    report: dict = {"passed": True, "errors": [], "warnings": [], "shapes": {}}

    report["shapes"] = {
        "Y": list(y.shape),
        "X_stim": list(x_stim.shape),
        "X_beh": list(x_beh.shape),
        "trial_ids": list(trial_ids.shape),
    }
    n = y.shape[0]
    if y.ndim != 2 or x_stim.ndim != 2 or x_beh.ndim != 2:
        report["errors"].append("Y, X_stim, X_beh must all be 2-D (samples × features)")
    if not (x_stim.shape[0] == x_beh.shape[0] == trial_ids.shape[0] == n):
        report["errors"].append(
            "sample-axis mismatch: "
            f"Y={y.shape[0]}, X_stim={x_stim.shape[0]}, "
            f"X_beh={x_beh.shape[0]}, trial_ids={trial_ids.shape[0]}"
        )
    if n < min_samples:
        report["errors"].append(f"too few samples ({n} < {min_samples})")
    n_trials = int(np.unique(trial_ids).size) if trial_ids.size else 0
    if n_trials < min_trials:
        report["errors"].append(f"too few trials ({n_trials} < {min_trials})")

    for name, arr in (("Y", y), ("X_stim", x_stim), ("X_beh", x_beh)):
        if not np.isfinite(arr).all():
            report["errors"].append(f"{name} contains non-finite values")
    if y.size and np.any(y < 0):
        report["errors"].append("Y has negative rates")

    # Stimulus one-hot / category structure checks.
    if x_stim.shape[1] == 2:
        row_sums = x_stim.sum(axis=1)
        if not np.allclose(row_sums, 1.0):
            report["errors"].append(
                "2-col stimulus design rows must be one-hot (sum to 1)"
            )
        # After mean-centering, effective rank is 1 — expected for binary category.
        report["warnings"].append(
            "binary one-hot stimulus is rank-1 after centering; neuropop max "
            "sensory rank is 1 — this is expected for 4-vs-20 category encoding. "
            "A rank sweep >1 for stimulus is not meaningful until the design has "
            "more independent columns (e.g. category × time bases)."
        )
        report["stim_design_kind"] = "trial_constant_onehot"
    elif x_stim.shape[1] < 2:
        report["errors"].append(
            f"X_stim has {x_stim.shape[1]} columns; neuropop RRR needs >=2 for rank≥1"
        )
    else:
        report["stim_design_kind"] = "expanded"
        # Time-expanded designs: each row should still activate exactly one category
        # within some time basis (row sum == 1 for one-hot × time).
        row_sums = x_stim.sum(axis=1)
        if not np.allclose(row_sums, 1.0):
            report["warnings"].append(
                "expanded X_stim rows do not sum to 1 — unexpected for "
                "category × time one-hot bases"
            )

    if stim_labels is not None:
        stim_labels = np.asarray(stim_labels, dtype=float)
        if stim_labels.shape[0] != n:
            report["errors"].append("stim_labels length != n_samples")
        else:
            for trial in np.unique(trial_ids):
                vals = stim_labels[trial_ids == trial]
                if not np.allclose(vals, vals[0]):
                    report["errors"].append(
                        f"stim_labels vary within trial {int(trial)} — "
                        "trial-constant category encoding required for this analysis"
                    )
            # Encoding ↔ label consistency for 2-col one-hot.
            if x_stim.shape[1] == 2:
                expected = stimulus_design_from_labels(stim_labels, encoding="onehot")
                if not np.allclose(x_stim, expected):
                    report["errors"].append(
                        "X_stim does not match stimulus_design_from_labels(stim_labels)"
                    )
            n_low = int(np.sum(stim_labels < 0))
            n_high = int(np.sum(stim_labels > 0))
            report["n_samples_low"] = n_low
            report["n_samples_high"] = n_high
            if min(n_low, n_high) == 0:
                report["errors"].append("stim_labels contain only one category")
            elif min(n_low, n_high) / max(n_low, n_high) < 0.5:
                report["warnings"].append(
                    f"sample-level category imbalance after binning: "
                    f"low={n_low}, high={n_high} (longer trials overweight a class)"
                )

    # Longer trials contribute more stacked samples — GroupKFold equalizes trials,
    # but VE is still sample-weighted toward long trials.
    if trial_ids.size:
        counts = np.bincount(trial_ids - trial_ids.min())
        report["bins_per_trial_min"] = int(counts.min()) if counts.size else 0
        report["bins_per_trial_max"] = int(counts.max()) if counts.size else 0
        report["bins_per_trial_median"] = float(np.median(counts)) if counts.size else 0
        if counts.size and counts.max() > 3 * max(counts.min(), 1):
            report["warnings"].append(
                "trial lengths differ >3×; held-out VE is sample-weighted toward "
                "longer trials even though GroupKFold balances trial counts"
            )

    # Decision-task confound: category-predictive movement inside the window.
    if (
        window_key in {"stim_to_response", "fixation_to_response"}
        and category_choice_phi is not None
        and np.isfinite(category_choice_phi)
        and abs(float(category_choice_phi)) > 0.2
    ):
        report["warnings"].append(
            f"window={window_key} includes choice-related movement and "
            f"|phi(category,choice)|={abs(float(category_choice_phi)):.2f}. "
            "Trial-constant category regressors can absorb choice-/movement-"
            "related variance into the 'sensory' subspace. Prefer "
            "--window=stim_to_exit for a cleaner sensory estimate, or "
            "residualize choice before claiming sensory–behavior overlap."
        )

    # Column variance: dead predictors.
    for name, arr in (("X_stim", x_stim), ("X_beh", x_beh)):
        if arr.size:
            col_std = arr.std(axis=0)
            n_dead = int(np.sum(col_std < 1e-12))
            if n_dead:
                report["warnings"].append(
                    f"{name} has {n_dead} near-zero-variance columns"
                )

    # Rank diagnostics (before centering).
    if x_stim.size:
        stim_rank = int(np.linalg.matrix_rank(x_stim, tol=1e-6))
        report["x_stim_matrix_rank"] = stim_rank
        report["max_neuropop_stim_rank"] = max_neuropop_rank(
            n, x_stim.shape[1], y.shape[1]
        )
    if x_beh.size:
        beh_rank = int(np.linalg.matrix_rank(x_beh, tol=1e-6))
        report["x_beh_matrix_rank"] = beh_rank
        report["max_neuropop_beh_rank"] = max_neuropop_rank(
            n, x_beh.shape[1], y.shape[1]
        )

    report["n_trials"] = n_trials
    report["n_samples"] = n
    report["passed"] = len(report["errors"]) == 0
    return report


def validate_first_stim_alignment(
    trial_first_stim_ts: ArrayF,
    first_stim_ev_15ms: ArrayF,
    *,
    max_abs_offset_s: float = 1e-3,
) -> dict:
    """Check enriched trial first-stim times match session first_stim_ev_15ms."""
    trial_ts = np.asarray(trial_first_stim_ts, dtype=float)
    session_ts = np.asarray(first_stim_ev_15ms, dtype=float)
    report: dict = {
        "passed": True,
        "errors": [],
        "warnings": [],
        "n_trial_stims": int(np.isfinite(trial_ts).sum()),
        "n_session_first_stims": int(np.isfinite(session_ts).sum()),
    }
    finite_trial = trial_ts[np.isfinite(trial_ts)]
    finite_session = session_ts[np.isfinite(session_ts)]
    if finite_trial.size == 0 or finite_session.size == 0:
        report["warnings"].append("cannot cross-check first-stim alignment (empty)")
        return report
    offsets = []
    unmatched = 0
    for t in finite_trial:
        nearest = finite_session[np.argmin(np.abs(finite_session - t))]
        off = float(abs(nearest - t))
        offsets.append(off)
        if off > max_abs_offset_s:
            unmatched += 1
    report["median_abs_offset_s"] = float(np.median(offsets))
    report["max_abs_offset_s"] = float(np.max(offsets))
    report["n_unmatched"] = unmatched
    if unmatched:
        report["errors"].append(
            f"{unmatched}/{len(offsets)} trial first_stim_ts values differ from "
            f"first_stim_ev_15ms by >{max_abs_offset_s}s"
        )
    report["passed"] = len(report["errors"]) == 0
    return report


def validate_frame_timing(
    frame_times: ArrayF,
    motion_energy: ArrayF | None,
    video_frame_count: int | None,
    event_times: dict[str, ArrayF],
    max_frame_mismatch: int = 1,
) -> dict:
    """Validate video/frame timing against key task events."""
    frame_times = np.asarray(frame_times, dtype=float)
    report: dict = {
        "n_frame_times": int(frame_times.size),
        "n_finite_frame_times": int(np.isfinite(frame_times).sum()),
        "n_nonmonotonic_frames": int(np.sum(np.diff(frame_times) <= 0))
        if frame_times.size > 1
        else 0,
        "video_frame_count": video_frame_count,
        "passed": True,
        "errors": [],
        "warnings": [],
        "event_coverage": {},
    }
    if frame_times.size < 2:
        report["passed"] = False
        report["errors"].append("frame_times must contain at least two timestamps")
        return report
    if report["n_nonmonotonic_frames"] > 0:
        report["passed"] = False
        report["errors"].append("frame_times are not strictly increasing")

    if video_frame_count is not None:
        mismatch = abs(int(video_frame_count) - int(frame_times.size))
        report["video_vs_frame_times_mismatch"] = mismatch
        if mismatch > max_frame_mismatch:
            report["passed"] = False
            report["errors"].append(
                f"video frame count ({video_frame_count}) vs frame_times "
                f"({frame_times.size}) mismatch {mismatch} > {max_frame_mismatch}"
            )

    if motion_energy is not None:
        motion_energy = np.asarray(motion_energy, dtype=float)
        report["n_motion_energy_samples"] = int(motion_energy.size)
        expected = max(frame_times.size - 1, 0)
        me_mismatch = abs(int(motion_energy.size) - expected)
        report["motion_vs_frame_times_mismatch"] = me_mismatch
        if me_mismatch > max_frame_mismatch:
            report["passed"] = False
            report["errors"].append(
                f"motion_energy length ({motion_energy.size}) vs frame_times-1 "
                f"({expected}) mismatch {me_mismatch} > {max_frame_mismatch}"
            )

    t0, t1 = float(frame_times[0]), float(frame_times[-1])
    for name, times in event_times.items():
        times = np.asarray(times, dtype=float)
        finite = times[np.isfinite(times)]
        inside = finite[(finite >= t0) & (finite <= t1)]
        report["event_coverage"][name] = {
            "n_events": int(finite.size),
            "n_inside_video": int(inside.size),
            "fraction_inside_video": float(inside.size / finite.size)
            if finite.size
            else float("nan"),
        }
        if finite.size and inside.size == 0:
            report["warnings"].append(
                f"no {name} events fall inside the video time span"
            )
    return report


def stimulus_design_from_labels(
    labels: ArrayF,
    *,
    encoding: str = "onehot",
) -> ArrayF:
    """Build a stimulus design matrix from trial-constant category labels.

    Parameters
    ----------
    labels
        Sample-aligned ±1 labels (``-1`` low, ``+1`` high).
    encoding
        ``onehot`` (default): 2 columns ``[is_low, is_high]`` — required for
        neuropop rank≥1. After centering this is effectively 1-D.
        ``signed``: single ±1 column (cannot support neuropop rank≥1 alone).

    Analysis meaning
    ----------------
    Trial-constant category encoding asks whether category identity occupies a
    neural subspace **shared across bins within the analysis window**. It does
    **not** model stimulus-driven temporal dynamics (that needs category × time
    bases or event kernels).
    """
    labels = np.asarray(labels, dtype=float).reshape(-1)
    if np.any(~np.isfinite(labels)):
        raise ValueError("stimulus labels contain non-finite values")
    if encoding == "signed":
        return labels.reshape(-1, 1).astype(np.float32)
    if encoding != "onehot":
        raise ValueError(f"unknown stimulus encoding: {encoding}")
    out = np.zeros((labels.shape[0], 2), dtype=np.float32)
    out[labels < 0, 0] = 1.0
    out[labels > 0, 1] = 1.0
    if not np.allclose(out.sum(axis=1), 1.0):
        raise ValueError("labels must be exclusively -1 or +1 for one-hot encoding")
    return out


def expand_stimulus_by_time(
    labels: ArrayF,
    bin_idx: ArrayF,
    n_time_bases: int,
    *,
    trial_ids: ArrayF | None = None,
    mode: str = "fractional",
) -> ArrayF:
    """Category × within-trial time bases (richer sensory design).

    Returns shape ``(n_samples, 2 * n_time_bases)``: for each time basis
    ``k``, columns ``[is_low * 1[bin==k], is_high * 1[bin==k]]``.

    Modes
    -----
    ``fractional`` (default)
        Within each trial, map bin position to equal-occupancy fractions of that
        trial's bins. Variable-length trials then contribute evenly to each
        basis — appropriate for freely moving decision tasks.

    ``absolute_bin``
        Use raw ``bin_idx`` clipped into ``0 .. n_time_bases-1``. Later bins of
        long trials collapse onto the last basis (length-confounded).

    Use this when the scientific question is about **time-resolved** sensory
    subspaces rather than a single trial-constant category direction.
    """
    labels = np.asarray(labels, dtype=float).reshape(-1)
    bin_idx = np.asarray(bin_idx, dtype=int).reshape(-1)
    if labels.shape[0] != bin_idx.shape[0]:
        raise ValueError("labels and bin_idx length mismatch")
    if n_time_bases < 1:
        raise ValueError("n_time_bases must be >= 1")
    onehot = stimulus_design_from_labels(labels, encoding="onehot")

    if mode == "absolute_bin":
        clipped = np.clip(bin_idx, 0, n_time_bases - 1)
    elif mode == "fractional":
        if trial_ids is None:
            raise ValueError("fractional time bases require trial_ids")
        trial_ids = np.asarray(trial_ids, dtype=int).reshape(-1)
        if trial_ids.shape[0] != labels.shape[0]:
            raise ValueError("trial_ids length mismatch")
        clipped = np.zeros(labels.shape[0], dtype=int)
        for trial in np.unique(trial_ids):
            rows = np.flatnonzero(trial_ids == trial)
            n_bins = rows.size
            if n_bins == 1:
                clipped[rows] = 0
            else:
                # Equal-occupancy fractions along the trial's own bin order.
                order = np.argsort(bin_idx[rows], kind="mergesort")
                frac = np.arange(n_bins, dtype=float) / n_bins
                clipped[rows[order]] = np.minimum(
                    (frac * n_time_bases).astype(int), n_time_bases - 1
                )
    else:
        raise ValueError(f"unknown time-basis mode: {mode}")

    out = np.zeros((labels.shape[0], 2 * n_time_bases), dtype=np.float32)
    for k in range(n_time_bases):
        mask = clipped == k
        out[mask, 2 * k : 2 * k + 2] = onehot[mask]
    return out


def max_neuropop_rank(n_samples: int, n_features: int, n_outputs: int) -> int:
    """Largest rank neuropop will fit (``min(...) - 1``)."""
    return max(0, min(n_outputs, n_samples, n_features) - 1)


def choice_category_confound_report(
    stim_labels: ArrayF,
    choices: ArrayF,
    trial_ids: ArrayF,
) -> dict:
    """Summarize category↔choice contingency (important in decision tasks)."""
    stim_labels = np.asarray(stim_labels, dtype=float)
    choices = np.asarray(choices, dtype=float)
    trial_ids = np.asarray(trial_ids, dtype=int)
    # One value per trial.
    rows = []
    for trial in np.unique(trial_ids):
        mask = trial_ids == trial
        rows.append((stim_labels[mask][0], choices[mask][0]))
    s = np.asarray([r[0] for r in rows], dtype=float)
    c = np.asarray([r[1] for r in rows], dtype=float)
    finite = np.isfinite(s) & np.isfinite(c)
    s, c = s[finite], c[finite]
    if s.size < 2:
        return {"n_trials": int(s.size), "phi": float("nan")}
    # Map to binary 0/1.
    s_bin = (s > 0).astype(float)
    c_bin = (c > 0).astype(float)
    s_c = s_bin - s_bin.mean()
    c_c = c_bin - c_bin.mean()
    denom = float(np.sqrt(np.sum(s_c**2) * np.sum(c_c**2)))
    phi = float(np.sum(s_c * c_c) / denom) if denom > 0 else float("nan")
    return {
        "n_trials": int(s.size),
        "phi_category_choice": phi,
        "n_high_right": int(np.sum((s_bin == 1) & (c_bin == 1))),
        "n_high_left": int(np.sum((s_bin == 1) & (c_bin == 0))),
        "n_low_right": int(np.sum((s_bin == 0) & (c_bin == 1))),
        "n_low_left": int(np.sum((s_bin == 0) & (c_bin == 0))),
        "warning": (
            "category and choice are correlated; sensory subspace may partly "
            "reflect choice-/movement-related activity unless residualized"
            if np.isfinite(phi) and abs(phi) > 0.2
            else None
        ),
    }
