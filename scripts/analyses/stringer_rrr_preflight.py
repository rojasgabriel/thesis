"""Preflight + minimal Stringer-style RRR subspace baseline (GRB006).

Builds a reproducible entry point that:

1. Validates video/frame timing against ``frames``, ``first_stim_ev_15ms``,
   center-port exit, and response events.
2. Creates trial-wise 100 ms neural bins and a documented low-dimensional
   motion-energy lag design (scalar ME via continuous-time lookback; not
   spatial ME-PCs).
3. Restricts to balanced 4 vs 20 Hz trials and uses trial-grouped CV.
4. Fits rank-swept stimulus→neural and behavior→neural RRR with
   **neuropop** ``linear_prediction`` / ``reduced_rank_regression`` (CPU),
   selects a held-out one-SE rank (knee secondary), and reports principal
   angles plus projection fractions.
5. Compares against rank-matched random, trial-label-shuffle, and
   behavior-shift nulls.

Design-matrix structure (important)
-----------------------------------
Samples are **trial-concatenated time bins** ``(trial, bin) → rows``. That is
the task-embedded analogue of Stringer's continuous-time axis, but trial
boundaries matter for grouping: GroupKFold never splits a trial, and stimulus
labels are constant within a trial for the default category encoding.

Behavior lags use **continuous-time lookback** on the session ME trace (default),
so early bins inherit real ITI/pre-window motion instead of imputed values or
index-lags across the stacked axis.

Default stimulus encoding is trial-constant 2-col one-hot (4 vs 20 Hz). That
asks whether category occupies a shared neural direction across the window —
not whether sensory dynamics unfold over time. Use ``--stim-time-bases`` for
category × fractional within-trial time bases.

Default analysis window is ``fixation_to_response`` (method-guide first pass).
Windows that include choice movement plus a category↔choice correlation can
leak choice-related variance into the "sensory" subspace — surfaced as a
design warning; use ``--window=stim_to_exit`` for a cleaner sensory estimate.

This is intentionally **not** a full Stringer replication (no SVC spectrum,
no gain analysis, no TIM split).

Requires a local ``neuropop`` checkout as a sibling of this repo (see
``[tool.uv.sources]`` in ``pyproject.toml``).

Run (requires lab DB / VPN)::

    uv run python scripts/analyses/stringer_rrr_preflight.py --no-save
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import types
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if "ephys" not in sys.modules:
    package = types.ModuleType("ephys")
    package.__path__ = [str(REPO_ROOT)]
    sys.modules["ephys"] = package
sys.path.insert(0, str(REPO_ROOT))

from ephys.src.utils.stringer_rrr import (
    average_trace_in_windows,
    balanced_easy_rate_trial_mask,
    behavior_shift_null,
    bin_spikes_trialwise,
    build_motion_lag_design,
    choice_category_confound_report,
    cross_validated_rrr_curve,
    expand_stimulus_by_time,
    fit_reduced_rank_regression,
    max_neuropop_rank,
    principal_angles_deg,
    projection_fraction,
    random_subspace_basis,
    select_rank_knee,
    select_rank_one_se,
    stimulus_design_from_labels,
    subspace_basis_from_rrr,
    trial_label_shuffle,
    validate_binned_sample_structure,
    validate_design_matrices,
    validate_first_stim_alignment,
    validate_frame_timing,
    zscore_columns,
)
from ephys.src.utils.stringer_rrr_events import (
    enrich_trials_with_event_timestamps,
)

SUBJECT = "GRB006"
SESSION = "20240821_121447"
UNIT_CRITERIA_ID = 1
BIN_WIDTH_S = 0.1
N_LAGS = 4
RANK_MAX = 32
N_SPLITS = 5
N_NULLS = 20
RANDOM_STATE = 0
LAM = 1e-3

FIGURE_ROOT = Path(os.environ.get("EPHYS_FIGURE_ROOT", str(REPO_ROOT / "figures")))
FIGURE_DIR = FIGURE_ROOT / "stringer_rrr_preflight"
MOTION_CACHE_DIR = REPO_ROOT / ".cache" / "categorydecoding"

WINDOW_CHOICES = {
    "stim_to_response": "first_stim_ts → response_ts (includes choice movement)",
    "fixation_to_response": "center_port_ts → response_ts (method-guide default)",
    "stim_to_exit": "first_stim_ts → center_port_exit_ts (pre-movement sensory)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Run without writing tidy results or the diagnostic figure.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open an interactive matplotlib window for the diagnostic figure.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Reduce ranks/nulls/splits for a fast smoke path.",
    )
    parser.add_argument(
        "--rank-rule",
        choices=("one_se", "knee"),
        default="one_se",
        help="Primary held-out rank selection rule (default: one_se).",
    )
    parser.add_argument(
        "--window",
        choices=tuple(WINDOW_CHOICES),
        default="fixation_to_response",
        help="Analysis epoch for neural/behavior bins (method-guide default).",
    )
    parser.add_argument(
        "--stim-time-bases",
        type=int,
        default=0,
        help=(
            "If >0, expand stimulus as category × within-trial time bases "
            "(richer sensory design). Default 0 = trial-constant one-hot."
        ),
    )
    parser.add_argument(
        "--stim-time-mode",
        choices=("fractional", "absolute_bin"),
        default="fractional",
        help=(
            "How to assign category × time bases: fractional (equal occupancy "
            "within each trial; default) or absolute_bin (raw bin index, "
            "length-confounded)."
        ),
    )
    parser.add_argument(
        "--motion-pcs",
        type=int,
        default=0,
        help=(
            "If >0, PCA-reduce ME lags inside each CV fold "
            "(train-fit only). Default 0 keeps raw lag features."
        ),
    )
    parser.add_argument(
        "--n-lags",
        type=int,
        default=N_LAGS,
        help="Causal ME lags (default 4).",
    )
    parser.add_argument(
        "--lag-mode",
        choices=("continuous_time", "within_trial"),
        default="continuous_time",
        help=(
            "ME lag construction: continuous_time lookback on the session ME "
            "trace (default), or within_trial index lags."
        ),
    )
    parser.add_argument(
        "--incomplete-lag-policy",
        choices=("error", "drop", "impute_trial_first"),
        default="error",
        help=(
            "How to handle missing within-trial lags. continuous_time usually "
            "needs no special handling; within_trial should use drop or "
            "impute_trial_first (discouraged)."
        ),
    )
    return parser.parse_args()


def discover_motion_cache() -> dict:
    """Locate scalar ME trace caches under .cache/categorydecoding."""
    out: dict = {
        "motion_cache_dir": str(MOTION_CACHE_DIR.relative_to(REPO_ROOT)),
        "dir_exists": MOTION_CACHE_DIR.exists(),
        "trace_candidates": [],
        "early_values_path": None,
        "selected_trace_path": None,
    }
    if not MOTION_CACHE_DIR.exists():
        return out
    early = MOTION_CACHE_DIR / "motion_early_values.npz"
    if early.exists():
        out["early_values_path"] = str(early.relative_to(REPO_ROOT))
    traces = sorted(MOTION_CACHE_DIR.glob("*_raw_motion_energy_trace.npz"))
    out["trace_candidates"] = [str(p.relative_to(REPO_ROOT)) for p in traces]
    preferred = [p for p in traces if SUBJECT in p.name and SESSION in p.name]
    if preferred:
        out["selected_trace_path"] = str(preferred[0].relative_to(REPO_ROOT))
    elif traces:
        out["selected_trace_path"] = str(traces[0].relative_to(REPO_ROOT))
    return out


def load_motion_energy_trace(
    cache_info: dict,
) -> tuple[np.ndarray | None, int | None, str | None]:
    selected = cache_info.get("selected_trace_path")
    if not selected:
        return None, None, None
    path = REPO_ROOT / selected
    with np.load(path, allow_pickle=False) as cache:
        energy = np.asarray(cache["motion_energy"], dtype=float)
        n_frames = (
            int(cache["video_frame_count"]) if "video_frame_count" in cache else None
        )
    return energy, n_frames, selected


def try_load_frame_times(align_ev: dict[str, np.ndarray]) -> tuple[np.ndarray, str]:
    try:
        from labdata.schema import Dataset, DatasetVideo

        dset = Dataset() & f'subject_name = "{SUBJECT}"' & f'session_name = "{SESSION}"'
        rows = (
            DatasetVideo & dset.fetch("subject_name", "session_name", as_dict=True)
        ).fetch(as_dict=True)
        if rows:
            frame_times = np.asarray(rows[0].get("frame_times", []), dtype=float)
            if frame_times.size >= 2:
                return frame_times, "DatasetVideo.frame_times"
    except Exception as exc:  # noqa: BLE001 - live DB may be unavailable
        print(
            f"DatasetVideo.frame_times unavailable ({exc}); using align_ev['frames']."
        )
    frames = np.asarray(align_ev.get("frames", []), dtype=float)
    return frames, "align_ev['frames']"


def resolve_analysis_window(
    trials: pd.DataFrame, window: str
) -> tuple[np.ndarray, np.ndarray, str]:
    if window == "stim_to_response":
        starts = trials["first_stim_ts"].to_numpy(dtype=float)
        stops = trials["response_ts"].to_numpy(dtype=float)
    elif window == "fixation_to_response":
        starts = trials["center_port_ts"].to_numpy(dtype=float)
        stops = trials["response_ts"].to_numpy(dtype=float)
    elif window == "stim_to_exit":
        starts = trials["first_stim_ts"].to_numpy(dtype=float)
        stops = trials["center_port_exit_ts"].to_numpy(dtype=float)
    else:
        raise ValueError(f"unknown window: {window}")
    return starts, stops, WINDOW_CHOICES[window]


def event_triggered_motion_traces(
    motion_energy: np.ndarray,
    motion_times: np.ndarray,
    event_times: np.ndarray,
    pre_s: float = 0.5,
    post_s: float = 1.0,
    bin_s: float = 0.05,
) -> dict:
    edges = np.arange(-pre_s, post_s + bin_s, bin_s)
    centers = 0.5 * (edges[:-1] + edges[1:])
    stacks = []
    for t0 in np.asarray(event_times, dtype=float):
        if not np.isfinite(t0):
            continue
        starts = t0 + edges[:-1]
        stops = t0 + edges[1:]
        vals, _counts, _nearest = average_trace_in_windows(
            motion_energy,
            motion_times,
            starts,
            stops,
            allow_nearest_fill=True,
        )
        stacks.append(vals)
    if not stacks:
        return {
            "bin_centers": centers.tolist(),
            "mean": [float("nan")] * len(centers),
            "sem": [float("nan")] * len(centers),
            "n_events": 0,
        }
    arr = np.vstack(stacks)
    mean = np.nanmean(arr, axis=0)
    sem = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(
        np.sum(np.isfinite(arr), axis=0).clip(min=1)
    )
    return {
        "bin_centers": centers.tolist(),
        "mean": mean.tolist(),
        "sem": sem.tolist(),
        "n_events": int(arr.shape[0]),
    }


def fit_subspace_with_cv(
    x: np.ndarray,
    y: np.ndarray,
    trial_ids: np.ndarray,
    ranks: np.ndarray,
    n_splits: int,
    rank_rule: str,
    n_pcs: int | None = None,
) -> dict:
    # For fold-wise PCA, neuropop rank limit uses post-PCA feature count.
    n_features_for_rank = int(n_pcs) if n_pcs and n_pcs > 0 else x.shape[1]
    usable = max_neuropop_rank(x.shape[0], n_features_for_rank, y.shape[1])
    ranks = np.asarray([r for r in ranks if 1 <= int(r) <= usable], dtype=int)
    if ranks.size == 0:
        raise ValueError(
            f"No usable neuropop ranks for design X{x.shape} → Y{y.shape} "
            f"(max_rank={usable}, n_pcs={n_pcs})"
        )
    ranks_arr, mean_ve, sem_ve = cross_validated_rrr_curve(
        x,
        y,
        trial_ids,
        ranks=ranks,
        n_splits=n_splits,
        lam=LAM,
        n_pcs=n_pcs,
    )
    one_se = select_rank_one_se(ranks_arr, mean_ve, sem_ve)
    knee = select_rank_knee(ranks_arr, mean_ve, sem_ve)
    selected = one_se if rank_rule == "one_se" else knee

    # Full-data refit for reported subspace basis (descriptive; rank from CV).
    x_fit = x
    if n_pcs is not None and n_pcs > 0:
        x_c = x - x.mean(axis=0, keepdims=True)
        _u, _s, vt = np.linalg.svd(x_c, full_matrices=False)
        x_fit = x_c @ vt[: min(int(n_pcs), vt.shape[0])].T
    a, b = fit_reduced_rank_regression(
        zscore_columns(x_fit),
        zscore_columns(y),
        selected.selected_rank,
        lam=LAM,
    )
    basis = subspace_basis_from_rrr(a)
    return {
        "selected_rank": int(selected.selected_rank),
        "rank_rule": selected.rule,
        "one_se_rank": int(one_se.selected_rank),
        "knee_rank": int(knee.selected_rank),
        "ranks": ranks_arr.tolist(),
        "mean_varexp": mean_ve.tolist(),
        "sem_varexp": sem_ve.tolist(),
        "max_neuropop_rank": int(usable),
        "rrr_backend": "neuropop.linear_prediction",
        "n_pcs_foldwise": int(n_pcs) if n_pcs else 0,
        "subspace_fit": "full_data_refit_after_cv_rank",
        "basis": basis,
        "a": a,
        "b": b,
    }


def run_nulls(
    x_stim: np.ndarray,
    x_beh: np.ndarray,
    y: np.ndarray,
    stim_labels: np.ndarray,
    trial_ids: np.ndarray,
    stim_rank: int,
    beh_rank: int,
    n_nulls: int,
    rng: np.random.Generator,
    beh_n_pcs: int | None = None,
) -> dict:
    """Rank-matched random / trial-label-shuffle / behavior-shift nulls."""

    def _prep_beh(x: np.ndarray) -> np.ndarray:
        if beh_n_pcs is None or beh_n_pcs <= 0:
            return zscore_columns(x)
        x_c = x - x.mean(axis=0, keepdims=True)
        _u, _s, vt = np.linalg.svd(x_c, full_matrices=False)
        return zscore_columns(x_c @ vt[: min(int(beh_n_pcs), vt.shape[0])].T)

    y_z = zscore_columns(y)
    stim_basis = subspace_basis_from_rrr(
        fit_reduced_rank_regression(zscore_columns(x_stim), y_z, stim_rank, lam=LAM)[0]
    )
    beh_basis = subspace_basis_from_rrr(
        fit_reduced_rank_regression(_prep_beh(x_beh), y_z, beh_rank, lam=LAM)[0]
    )
    observed_angles = principal_angles_deg(stim_basis, beh_basis)
    observed_stim_in_beh = projection_fraction(stim_basis, beh_basis)
    observed_beh_in_stim = projection_fraction(beh_basis, stim_basis)

    random_min_angles = []
    shuffle_min_angles = []
    shift_min_angles = []
    shuffle_stim_in_beh = []
    shift_stim_in_beh = []

    n_neurons = y.shape[1]
    unique_trials = np.unique(trial_ids)
    for _ in range(n_nulls):
        rand_stim = random_subspace_basis(n_neurons, stim_rank, rng)
        rand_beh = random_subspace_basis(n_neurons, beh_rank, rng)
        random_min_angles.append(
            float(np.min(principal_angles_deg(rand_stim, rand_beh)))
        )

        shuffled_labels = trial_label_shuffle(stim_labels, trial_ids, rng)
        x_stim_null = stimulus_design_from_labels(shuffled_labels)
        a_s, _b_s = fit_reduced_rank_regression(
            zscore_columns(x_stim_null), y_z, stim_rank, lam=LAM
        )
        a_b, _b_b = fit_reduced_rank_regression(
            _prep_beh(x_beh), y_z, beh_rank, lam=LAM
        )
        s_basis = subspace_basis_from_rrr(a_s)
        b_basis = subspace_basis_from_rrr(a_b)
        shuffle_min_angles.append(float(np.min(principal_angles_deg(s_basis, b_basis))))
        shuffle_stim_in_beh.append(float(projection_fraction(s_basis, b_basis)))

        shift = int(rng.integers(1, max(2, unique_trials.size)))
        x_beh_null = behavior_shift_null(x_beh, trial_ids, shift)
        a_s2, _ = fit_reduced_rank_regression(
            zscore_columns(x_stim), y_z, stim_rank, lam=LAM
        )
        a_b2, _ = fit_reduced_rank_regression(
            _prep_beh(x_beh_null), y_z, beh_rank, lam=LAM
        )
        s2 = subspace_basis_from_rrr(a_s2)
        b2 = subspace_basis_from_rrr(a_b2)
        shift_min_angles.append(float(np.min(principal_angles_deg(s2, b2))))
        shift_stim_in_beh.append(float(projection_fraction(s2, b2)))

    return {
        "observed": {
            "principal_angles_deg": observed_angles.tolist(),
            "min_principal_angle_deg": float(np.min(observed_angles))
            if observed_angles.size
            else float("nan"),
            "stim_in_behavior_fraction": float(observed_stim_in_beh),
            "behavior_in_stim_fraction": float(observed_beh_in_stim),
        },
        "random_null_min_angle_deg": random_min_angles,
        "trial_label_shuffle_min_angle_deg": shuffle_min_angles,
        "behavior_shift_min_angle_deg": shift_min_angles,
        "trial_label_shuffle_stim_in_beh": shuffle_stim_in_beh,
        "behavior_shift_stim_in_beh": shift_stim_in_beh,
        "n_nulls": int(n_nulls),
    }


def make_diagnostic_figure(plt, results: dict, no_save: bool, show: bool) -> None:
    fig, axs = plt.subplots(2, 2, figsize=(10, 7))

    stim = results["stimulus_rrr"]
    beh = results["behavior_rrr"]
    axs[0, 0].errorbar(
        stim["ranks"], stim["mean_varexp"], yerr=stim["sem_varexp"], label="stim→neural"
    )
    axs[0, 0].errorbar(
        beh["ranks"], beh["mean_varexp"], yerr=beh["sem_varexp"], label="beh→neural"
    )
    axs[0, 0].axvline(stim["selected_rank"], color="C0", ls="--", lw=1)
    axs[0, 0].axvline(beh["selected_rank"], color="C1", ls="--", lw=1)
    axs[0, 0].set_xlabel("rank")
    axs[0, 0].set_ylabel("held-out variance explained")
    axs[0, 0].legend(frameon=False)
    axs[0, 0].set_title("RRR rank sweeps (neuropop)")

    angles = results["nulls"]["observed"]["principal_angles_deg"]
    axs[0, 1].plot(np.arange(1, len(angles) + 1), angles, marker="o")
    axs[0, 1].axhline(90, color="0.5", ls="--", lw=1)
    axs[0, 1].set_xlabel("principal angle index")
    axs[0, 1].set_ylabel("degrees")
    axs[0, 1].set_title("Stim vs behavior principal angles")

    nulls = results["nulls"]
    axs[1, 0].hist(
        nulls["random_null_min_angle_deg"], bins=12, alpha=0.5, label="random"
    )
    axs[1, 0].hist(
        nulls["trial_label_shuffle_min_angle_deg"],
        bins=12,
        alpha=0.5,
        label="label shuffle",
    )
    axs[1, 0].hist(
        nulls["behavior_shift_min_angle_deg"],
        bins=12,
        alpha=0.5,
        label="beh shift",
    )
    axs[1, 0].axvline(
        nulls["observed"]["min_principal_angle_deg"],
        color="k",
        lw=1.5,
        label="observed",
    )
    axs[1, 0].set_xlabel("min principal angle (deg)")
    axs[1, 0].set_ylabel("null resamples")
    axs[1, 0].legend(frameon=False, fontsize=8)
    axs[1, 0].set_title("Null distributions")

    traces = results["preflight"].get("event_triggered_motion", {})
    for name, color in (
        ("first_stim", "C0"),
        ("center_port_exit", "C1"),
        ("response", "C2"),
    ):
        tr = traces.get(name)
        if not tr:
            continue
        x = np.asarray(tr["bin_centers"])
        y = np.asarray(tr["mean"])
        sem = np.asarray(tr["sem"])
        axs[1, 1].plot(x, y, color=color, label=name)
        axs[1, 1].fill_between(x, y - sem, y + sem, color=color, alpha=0.2)
    axs[1, 1].axvline(0, color="k", lw=0.8)
    axs[1, 1].set_xlabel("time from event (s)")
    axs[1, 1].set_ylabel("motion energy")
    axs[1, 1].legend(frameon=False, fontsize=8)
    axs[1, 1].set_title("Event-triggered motion")

    fig.suptitle(f"{SUBJECT} {SESSION} — Stringer RRR preflight / baseline")
    fig.tight_layout()
    if not no_save:
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURE_DIR / "diagnostic.png", dpi=140)
        print(f"Wrote {FIGURE_DIR / 'diagnostic.png'}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    args = parse_args()
    rank_max = 8 if args.quick else RANK_MAX
    n_splits = 3 if args.quick else N_SPLITS
    n_nulls = 5 if args.quick else N_NULLS
    ranks = np.arange(1, rank_max + 1, dtype=int)
    rng = np.random.default_rng(RANDOM_STATE)
    beh_n_pcs = args.motion_pcs if args.motion_pcs > 0 else None

    if not args.show:
        matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    blockers: list[str] = []
    decisions_for_gabriel = [
        f"behavior epoch: --window={args.window} ({WINDOW_CHOICES[args.window]})",
        "primary bin width: 100 ms (used here) vs 200 ms sensitivity",
        (
            "stimulus basis: trial-constant one-hot"
            if args.stim_time_bases <= 0
            else (
                f"stimulus basis: category × {args.stim_time_bases} time bases "
                f"({args.stim_time_mode})"
            )
        ),
        "TIM / task-aligned movement split: deferred",
        f"rank rule: {args.rank_rule} (knee also reported)",
        "choice / RT / correctness confounds: reported, not residualized",
        (f"behavior features: scalar ME lags via {args.lag_mode} (not spatial ME-PCs)"),
        (
            "binary one-hot stimulus → neuropop sensory rank capped at 1; "
            "use --stim-time-bases for higher-rank sensory designs"
        ),
    ]

    print(f"Loading {SUBJECT} {SESSION} (unit_criteria_id={UNIT_CRITERIA_ID})...")
    try:
        from ephys.src.utils.io_chipmunk_trials import fetch_trial_metadata
        from ephys.src.utils.io_digital_events import fetch_session_events
        from ephys.src.utils.io_session_units import fetch_good_units

        st_per_unit = fetch_good_units(SUBJECT, SESSION, UNIT_CRITERIA_ID)
        align_ev = fetch_session_events(SUBJECT, SESSION)
        trial_df = fetch_trial_metadata(SUBJECT, SESSION, align_ev)
    except Exception as exc:
        detail = str(exc).strip() or type(exc).__name__
        if isinstance(exc, EOFError):
            detail = (
                "DataJoint/labdata prompted for DB credentials but none are "
                "configured in this environment (EOF on getpass)"
            )
        blockers.append(f"lab DB / VPN load failed: {detail}")
        summary = {
            "status": "blocked",
            "blockers": blockers,
            "decisions_for_gabriel": decisions_for_gabriel,
            "motion_cache": discover_motion_cache(),
        }
        print(json.dumps(summary, indent=2))
        raise SystemExit(2) from exc

    if trial_df is None or len(trial_df) == 0:
        blockers.append("fetch_trial_metadata returned no trials")
        print(json.dumps({"status": "blocked", "blockers": blockers}, indent=2))
        raise SystemExit(2)

    trial_df = enrich_trials_with_event_timestamps(trial_df, align_ev)
    choice_trials = trial_df[trial_df["with_choice"] == 1].reset_index(drop=True)
    balanced_mask, stim_labels_trial = balanced_easy_rate_trial_mask(
        choice_trials["stim_rate_vision"].to_numpy(dtype=float),
        rng=rng,
    )
    easy_trials = choice_trials.loc[balanced_mask].reset_index(drop=True)
    stim_labels_trial = stim_labels_trial[balanced_mask]
    print(
        f"{len(st_per_unit)} units; {len(choice_trials)} choice trials; "
        f"{len(easy_trials)} balanced 4-vs-20 Hz trials"
    )

    window_starts, window_stops, window_desc = resolve_analysis_window(
        easy_trials, args.window
    )
    usable = (
        np.isfinite(window_starts)
        & np.isfinite(window_stops)
        & (window_stops > window_starts + BIN_WIDTH_S)
    )
    n_dropped_windows = int((~usable).sum())
    easy_trials = easy_trials.loc[usable].reset_index(drop=True)
    stim_labels_trial = stim_labels_trial[usable]
    window_starts = window_starts[usable]
    window_stops = window_stops[usable]

    # Remap trial indices after filtering so bin trial_ids are 0..n_trials-1.
    unit_ids = list(st_per_unit.keys())
    spike_lists = [np.asarray(st_per_unit[uid], dtype=float) for uid in unit_ids]
    binned = bin_spikes_trialwise(
        spike_lists, window_starts, window_stops, bin_width_s=BIN_WIDTH_S
    )
    bin_structure = validate_binned_sample_structure(
        binned.trial_ids,
        binned.bin_idx,
        binned.bin_starts,
        binned.bin_stops,
    )
    for warning in bin_structure.get("warnings", []):
        print(f"Bin-structure warning: {warning}")
    if not bin_structure["passed"]:
        blockers.extend(bin_structure["errors"])

    y = binned.rates
    trial_ids = binned.trial_ids
    bin_idx = binned.bin_idx
    bin_starts = binned.bin_starts
    bin_stops = binned.bin_stops
    stim_labels = stim_labels_trial[trial_ids]
    if args.stim_time_bases > 0:
        x_stim = expand_stimulus_by_time(
            stim_labels,
            bin_idx,
            args.stim_time_bases,
            trial_ids=trial_ids,
            mode=args.stim_time_mode,
        )
        stim_encoding = (
            f"category_x_time(n_bases={args.stim_time_bases},"
            f"mode={args.stim_time_mode})"
        )
    else:
        x_stim = stimulus_design_from_labels(stim_labels, encoding="onehot")
        stim_encoding = "trial_constant_onehot_4vs20"

    # Choice labels aligned to samples for confound diagnostics.
    choice_trial = easy_trials["response"].to_numpy(dtype=float)
    choice_labels = choice_trial[trial_ids]
    confound = choice_category_confound_report(stim_labels, choice_labels, trial_ids)

    first_stim_check = validate_first_stim_alignment(
        easy_trials["first_stim_ts"].to_numpy(dtype=float),
        np.asarray(align_ev.get("first_stim_ev_15ms", []), dtype=float),
    )
    if not first_stim_check["passed"]:
        blockers.extend(first_stim_check["errors"])
    for warning in first_stim_check.get("warnings", []):
        print(f"First-stim warning: {warning}")

    cache_info = discover_motion_cache()
    motion_energy, video_frame_count, trace_path = load_motion_energy_trace(cache_info)
    frame_times, frame_source = try_load_frame_times(align_ev)

    preflight_events = {
        "frames": np.asarray(align_ev.get("frames", []), dtype=float),
        "first_stim_ev_15ms": np.asarray(
            align_ev.get("first_stim_ev_15ms", []), dtype=float
        ),
        "center_port_exit": easy_trials["center_port_exit_ts"].to_numpy(dtype=float),
        "response": easy_trials["response_ts"].to_numpy(dtype=float),
    }
    timing_report = validate_frame_timing(
        frame_times,
        motion_energy,
        video_frame_count,
        preflight_events,
    )
    timing_report["frame_times_source"] = frame_source
    timing_report["n_usable_easy_trials"] = len(easy_trials)
    timing_report["n_dropped_invalid_windows"] = n_dropped_windows
    timing_report["n_dropped_short_bins"] = binned.n_dropped_short_bins
    timing_report["n_neural_bins"] = int(y.shape[0])
    timing_report["motion_cache"] = cache_info
    timing_report["first_stim_alignment"] = first_stim_check
    timing_report["bin_structure"] = bin_structure

    event_triggered = {}
    x_beh = None
    motion_meta = None
    if motion_energy is None or frame_times.size < 2:
        blockers.append(
            "motion-energy trace and/or frame_times unavailable under "
            ".cache/categorydecoding/ (and DatasetVideo/frame TTLs). "
            "Preflight cannot build the behavior design matrix."
        )
    else:
        n_align = min(len(frame_times), (video_frame_count or len(frame_times)))
        frame_times = frame_times[:n_align]
        motion_energy = motion_energy[: max(n_align - 1, 0)]
        motion_times = frame_times[1 : len(motion_energy) + 1]
        if motion_times.shape[0] != motion_energy.shape[0]:
            blockers.append(
                f"motion_times ({motion_times.shape[0]}) and motion_energy "
                f"({motion_energy.shape[0]}) length mismatch after alignment trim"
            )
        else:
            event_triggered = {
                "first_stim": event_triggered_motion_traces(
                    motion_energy,
                    motion_times,
                    easy_trials["first_stim_ts"].to_numpy(dtype=float),
                ),
                "center_port_exit": event_triggered_motion_traces(
                    motion_energy,
                    motion_times,
                    easy_trials["center_port_exit_ts"].to_numpy(dtype=float),
                ),
                "response": event_triggered_motion_traces(
                    motion_energy,
                    motion_times,
                    easy_trials["response_ts"].to_numpy(dtype=float),
                ),
            }
            incomplete_policy = args.incomplete_lag_policy
            if args.lag_mode == "continuous_time" and incomplete_policy == "error":
                # Continuous-time lookback can still miss ME before video start;
                # drop those edge samples rather than aborting the whole run.
                incomplete_policy = "drop"
            try:
                x_beh, beh_valid, motion_meta = build_motion_lag_design(
                    motion_energy,
                    motion_times,
                    bin_starts,
                    bin_stops,
                    trial_ids,
                    n_lags=args.n_lags,
                    lag_mode=args.lag_mode,
                    incomplete_policy=incomplete_policy,
                )
            except ValueError as exc:
                blockers.append(f"behavior design failed: {exc}")
                x_beh, beh_valid, motion_meta = None, None, None
            if x_beh is not None and beh_valid is not None and not beh_valid.all():
                n_drop = int((~beh_valid).sum())
                print(
                    f"Dropping {n_drop}/{beh_valid.size} samples with incomplete "
                    f"ME lags (lag_mode={args.lag_mode})"
                )
                y = y[beh_valid]
                x_stim = x_stim[beh_valid]
                x_beh = x_beh[beh_valid]
                trial_ids = trial_ids[beh_valid]
                bin_idx = bin_idx[beh_valid]
                bin_starts = bin_starts[beh_valid]
                bin_stops = bin_stops[beh_valid]
                stim_labels = stim_labels[beh_valid]
                choice_labels = choice_labels[beh_valid]
                motion_meta["n_samples_after_lag_filter"] = int(y.shape[0])
                # Recompute confound on the filtered sample set.
                confound = choice_category_confound_report(
                    stim_labels, choice_labels, trial_ids
                )

    if not timing_report["passed"]:
        blockers.extend(timing_report["errors"])

    design_report = None
    if x_beh is not None and y.shape[0] > 0:
        design_report = validate_design_matrices(
            y,
            x_stim,
            x_beh,
            trial_ids,
            stim_labels=stim_labels,
            window_key=args.window,
            category_choice_phi=confound.get("phi_category_choice"),
        )
        if not design_report["passed"]:
            blockers.extend(design_report["errors"])
        for warning in design_report.get("warnings", []):
            print(f"Design warning: {warning}")
        if confound.get("warning"):
            print(f"Confound warning: {confound['warning']}")

    preflight = {
        "timing": timing_report,
        "event_triggered_motion": event_triggered,
        "motion_design_meta": motion_meta,
        "design_validation": design_report,
        "category_choice_confound": confound,
        "trace_path": trace_path,
        "bin_width_s": BIN_WIDTH_S,
        "window": window_desc,
        "window_key": args.window,
        "stimulus_encoding": stim_encoding,
        "sample_structure": "trial_concatenated_bins",
        "lag_mode": args.lag_mode,
        "n_dropped_short_bins": binned.n_dropped_short_bins,
    }

    if blockers:
        summary = {
            "status": "blocked_preflight",
            "blockers": blockers,
            "preflight": preflight,
            "decisions_for_gabriel": decisions_for_gabriel,
            "n_units": len(unit_ids),
            "n_balanced_trials": len(easy_trials),
            "n_bins": int(y.shape[0]),
        }
        print(json.dumps(summary, indent=2, default=str))
        if not args.no_save:
            FIGURE_DIR.mkdir(parents=True, exist_ok=True)
            (FIGURE_DIR / "preflight_blocked.json").write_text(
                json.dumps(summary, indent=2, default=str)
            )
        raise SystemExit(2)

    assert x_beh is not None
    print(
        f"Preflight passed: {y.shape[0]} bins × {y.shape[1]} units; "
        f"X_stim {x_stim.shape}; X_beh {x_beh.shape}; window={args.window}"
    )

    stim_fit = fit_subspace_with_cv(
        x_stim, y, trial_ids, ranks, n_splits, args.rank_rule, n_pcs=None
    )
    beh_fit = fit_subspace_with_cv(
        x_beh, y, trial_ids, ranks, n_splits, args.rank_rule, n_pcs=beh_n_pcs
    )
    nulls = run_nulls(
        x_stim,
        x_beh,
        y,
        stim_labels,
        trial_ids,
        stim_fit["selected_rank"],
        beh_fit["selected_rank"],
        n_nulls,
        rng,
        beh_n_pcs=beh_n_pcs,
    )

    tidy_rows = []
    for name, fit in (("stimulus", stim_fit), ("behavior", beh_fit)):
        for rank, mean_ve, sem_ve in zip(
            fit["ranks"], fit["mean_varexp"], fit["sem_varexp"]
        ):
            tidy_rows.append(
                {
                    "model": name,
                    "rank": rank,
                    "mean_heldout_varexp": mean_ve,
                    "sem_heldout_varexp": sem_ve,
                    "selected_rank": fit["selected_rank"],
                    "rank_rule": fit["rank_rule"],
                }
            )
    tidy = pd.DataFrame(tidy_rows)

    results = {
        "status": "ok",
        "meta": {
            "subject": SUBJECT,
            "session": SESSION,
            "unit_criteria_id": UNIT_CRITERIA_ID,
            "n_units": len(unit_ids),
            "n_balanced_trials": len(easy_trials),
            "n_bins": int(y.shape[0]),
            "bin_width_s": BIN_WIDTH_S,
            "rank_rule": args.rank_rule,
            "rrr_backend": "neuropop.linear_prediction",
            "window": args.window,
            "stimulus_encoding": stim_encoding,
            "motion_pcs_foldwise": int(beh_n_pcs or 0),
            "lag_mode": args.lag_mode,
            "run_mode": "quick" if args.quick else "full",
        },
        "preflight": preflight,
        "stimulus_rrr": {
            k: v for k, v in stim_fit.items() if k not in {"basis", "a", "b"}
        },
        "behavior_rrr": {
            k: v for k, v in beh_fit.items() if k not in {"basis", "a", "b"}
        },
        "overlap": {
            "principal_angles_deg": nulls["observed"]["principal_angles_deg"],
            "stim_in_behavior_fraction": nulls["observed"]["stim_in_behavior_fraction"],
            "behavior_in_stim_fraction": nulls["observed"]["behavior_in_stim_fraction"],
            "note": (
                "angles from full-data subspace refit at CV-selected ranks; "
                "not held-out angle estimates"
            ),
        },
        "nulls": nulls,
        "decisions_for_gabriel": decisions_for_gabriel,
    }

    print(
        "Selected ranks — stim: "
        f"{stim_fit['selected_rank']} (knee={stim_fit['knee_rank']}), "
        f"beh: {beh_fit['selected_rank']} (knee={beh_fit['knee_rank']})"
    )
    print(
        "Min principal angle (deg): "
        f"{nulls['observed']['min_principal_angle_deg']:.2f}; "
        f"stim-in-beh={nulls['observed']['stim_in_behavior_fraction']:.3f}, "
        f"beh-in-stim={nulls['observed']['behavior_in_stim_fraction']:.3f}"
    )

    if not args.no_save:
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        tidy.to_csv(FIGURE_DIR / "rrr_rank_curves.csv", index=False)
        sample_index = pd.DataFrame(
            {
                "trial_id": trial_ids,
                "bin_idx": bin_idx,
                "bin_start": bin_starts,
                "bin_stop": bin_stops,
                "stim_label": stim_labels,
                "choice": choice_labels,
            }
        )
        sample_index.to_csv(FIGURE_DIR / "sample_index.csv", index=False)
        (FIGURE_DIR / "results_summary.json").write_text(
            json.dumps(results, indent=2, default=str)
        )
        null_df = pd.DataFrame(
            {
                "random_min_angle_deg": nulls["random_null_min_angle_deg"],
                "label_shuffle_min_angle_deg": nulls[
                    "trial_label_shuffle_min_angle_deg"
                ],
                "behavior_shift_min_angle_deg": nulls["behavior_shift_min_angle_deg"],
                "label_shuffle_stim_in_beh": nulls["trial_label_shuffle_stim_in_beh"],
                "behavior_shift_stim_in_beh": nulls["behavior_shift_stim_in_beh"],
            }
        )
        null_df.to_csv(FIGURE_DIR / "null_resamples.csv", index=False)
        print(f"Wrote tidy outputs under {FIGURE_DIR}")

    make_diagnostic_figure(plt, results, args.no_save, args.show)
    print(
        json.dumps(
            {
                "status": "ok",
                "meta": results["meta"],
                "overlap": results["overlap"],
                "confound": confound,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
