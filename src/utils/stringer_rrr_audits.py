"""Assumption-audit contrasts for the Stringer RRR pipeline.

These helpers take already-assembled design matrices (same contract as
``stringer_rrr``) and return tidy contrast tables + verdict dicts. They do not
load lab data — the analysis script wires loaders.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .stringer_rrr import (
    choice_category_confound_report,
    cross_validated_rrr_curve,
    expand_stimulus_by_time,
    fit_reduced_rank_regression,
    max_neuropop_rank,
    principal_angles_deg,
    projection_fraction,
    select_rank_knee,
    select_rank_one_se,
    stimulus_design_from_labels,
    subspace_basis_from_rrr,
    trial_group_splits,
    zscore_columns,
)

ArrayF = NDArray[np.floating]


def residualize_columns_by_choice(
    y: ArrayF,
    choice_labels: ArrayF,
    trial_ids: ArrayF,
) -> ArrayF:
    """Remove trial-level choice mean from each neural column (within category-agnostic).

    Fits ``y ~ 1 + choice`` on sample rows using trial-constant ±1 choice, then
    returns residuals. This is a blunt confound control for Audit C — not a
    full behavioral GLM.
    """
    y = np.asarray(y, dtype=float)
    choice = np.asarray(choice_labels, dtype=float).reshape(-1)
    trial_ids = np.asarray(trial_ids, dtype=int)
    if y.shape[0] != choice.shape[0]:
        raise ValueError("y and choice_labels length mismatch")
    # Guard: choice must be constant within trial.
    for trial in np.unique(trial_ids):
        vals = choice[trial_ids == trial]
        if not np.allclose(vals, vals[0], equal_nan=True):
            raise ValueError(f"choice varies within trial {int(trial)}")
    design = np.column_stack([np.ones(y.shape[0]), choice])
    # Least squares per neuron.
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return y - design @ coef


def _select_rank(
    ranks: ArrayF, mean_ve: ArrayF, sem_ve: ArrayF, rule: str
) -> tuple[int, dict]:
    one_se = select_rank_one_se(ranks, mean_ve, sem_ve)
    knee = select_rank_knee(ranks, mean_ve, sem_ve)
    selected = one_se if rule == "one_se" else knee
    return int(selected.selected_rank), {
        "selected_rank": int(selected.selected_rank),
        "one_se_rank": int(one_se.selected_rank),
        "knee_rank": int(knee.selected_rank),
        "rank_rule": selected.rule,
        "ranks": [int(r) for r in ranks],
        "mean_varexp": [float(v) for v in mean_ve],
        "sem_varexp": [float(v) for v in sem_ve],
    }


def fit_cv_and_basis(
    x: ArrayF,
    y: ArrayF,
    trial_ids: ArrayF,
    *,
    ranks: ArrayF | list[int],
    n_splits: int = 5,
    rank_rule: str = "one_se",
    lam: float = 1e-3,
    n_pcs: int | None = None,
) -> dict:
    """Held-out VE curve + full-data subspace basis at the selected rank."""
    ranks = np.asarray(ranks, dtype=int)
    n_feat = int(n_pcs) if n_pcs and n_pcs > 0 else x.shape[1]
    usable = max_neuropop_rank(x.shape[0], n_feat, y.shape[1])
    ranks = np.asarray([r for r in ranks if 1 <= int(r) <= usable], dtype=int)
    if ranks.size == 0:
        raise ValueError(f"no usable ranks (max_neuropop_rank={usable})")
    ranks_arr, mean_ve, sem_ve = cross_validated_rrr_curve(
        x, y, trial_ids, ranks=ranks, n_splits=n_splits, lam=lam, n_pcs=n_pcs
    )
    selected_rank, curve = _select_rank(ranks_arr, mean_ve, sem_ve, rank_rule)
    x_fit = np.asarray(x, dtype=float)
    if n_pcs is not None and n_pcs > 0:
        x_c = x_fit - x_fit.mean(axis=0, keepdims=True)
        _u, _s, vt = np.linalg.svd(x_c, full_matrices=False)
        x_fit = x_c @ vt[: min(int(n_pcs), vt.shape[0])].T
    a, b = fit_reduced_rank_regression(
        zscore_columns(x_fit), zscore_columns(y), selected_rank, lam=lam
    )
    basis = subspace_basis_from_rrr(a)
    return {
        **curve,
        "basis": basis,
        "a": a,
        "b": b,
        "max_neuropop_rank": int(usable),
    }


def overlap_summary(stim_basis: ArrayF, beh_basis: ArrayF) -> dict:
    angles = principal_angles_deg(stim_basis, beh_basis)
    return {
        "principal_angles_deg": angles.tolist(),
        "min_principal_angle_deg": float(np.min(angles))
        if angles.size
        else float("nan"),
        "stim_in_behavior_fraction": float(projection_fraction(stim_basis, beh_basis)),
        "behavior_in_stim_fraction": float(projection_fraction(beh_basis, stim_basis)),
    }


def audit_stimulus_encoding(
    y: ArrayF,
    x_beh: ArrayF,
    stim_labels: ArrayF,
    trial_ids: ArrayF,
    bin_idx: ArrayF,
    *,
    n_time_bases: int = 4,
    ranks: ArrayF | list[int] | None = None,
    n_splits: int = 5,
    rank_rule: str = "one_se",
    lam: float = 1e-3,
) -> dict:
    """Audit B: trial-constant one-hot vs category × fractional time bases."""
    if ranks is None:
        ranks = np.arange(1, 9, dtype=int)
    x_b1 = stimulus_design_from_labels(stim_labels, encoding="onehot")
    x_b2 = expand_stimulus_by_time(
        stim_labels,
        bin_idx,
        n_time_bases,
        trial_ids=trial_ids,
        mode="fractional",
    )
    beh = fit_cv_and_basis(
        x_beh,
        y,
        trial_ids,
        ranks=ranks,
        n_splits=n_splits,
        rank_rule=rank_rule,
        lam=lam,
    )
    rows = []
    fits = {}
    for name, x_stim in (
        ("B1_trial_constant_onehot", x_b1),
        ("B2_category_x_time", x_b2),
    ):
        stim = fit_cv_and_basis(
            x_stim,
            y,
            trial_ids,
            ranks=ranks,
            n_splits=n_splits,
            rank_rule=rank_rule,
            lam=lam,
        )
        ov = overlap_summary(stim["basis"], beh["basis"])
        fits[name] = {"stim": stim, "overlap": ov}
        peak_ve = (
            float(np.nanmax(stim["mean_varexp"]))
            if stim["mean_varexp"]
            else float("nan")
        )
        rows.append(
            {
                "condition": name,
                "n_stim_features": int(x_stim.shape[1]),
                "selected_rank": stim["selected_rank"],
                "peak_heldout_varexp": peak_ve,
                "selected_heldout_varexp": float(
                    stim["mean_varexp"][stim["ranks"].index(stim["selected_rank"])]
                ),
                "min_principal_angle_deg": ov["min_principal_angle_deg"],
                "stim_in_behavior_fraction": ov["stim_in_behavior_fraction"],
            }
        )

    b1_ve = rows[0]["selected_heldout_varexp"]
    b2_ve = rows[1]["selected_heldout_varexp"]
    b1_ang = rows[0]["min_principal_angle_deg"]
    b2_ang = rows[1]["min_principal_angle_deg"]
    ve_gain = b2_ve - b1_ve
    angle_shift = (
        abs(b2_ang - b1_ang)
        if np.isfinite(b1_ang) and np.isfinite(b2_ang)
        else float("nan")
    )

    if not np.isfinite(b1_ve) or b1_ve <= 0:
        recommendation = "investigate_B1_first"
        rationale = "trial-constant sensory VE is non-positive; fix window/encoding before upgrading"
    elif ve_gain > 0.02 or (np.isfinite(angle_shift) and angle_shift > 10):
        recommendation = "prefer_B2_category_x_time"
        rationale = (
            f"time bases add held-out VE (Δ={ve_gain:.3f}) and/or shift angles "
            f"(|Δmin∠|={angle_shift:.1f}°); trial-constant may be too coarse"
        )
    else:
        recommendation = "keep_B1_trial_constant_onehot"
        rationale = (
            f"time bases do not materially change VE (Δ={ve_gain:.3f}) or angles "
            f"(|Δmin∠|={angle_shift:.1f}°); rank-1 category is adequate for first pass"
        )

    return {
        "audit": "B_stimulus_encoding",
        "contrast_table": rows,
        "recommendation": recommendation,
        "verdict": rationale,
        "notes": [
            "B1 sensory rank is capped at 1 after centering by construction",
            "angles use full-data subspace refit at CV-selected ranks (descriptive)",
        ],
        "fits_meta": {
            k: {
                "selected_rank": v["stim"]["selected_rank"],
                "peak_ve": float(np.nanmax(v["stim"]["mean_varexp"])),
            }
            for k, v in fits.items()
        },
    }


def audit_window_confound(
    spike_lists: list[ArrayF],
    trials_meta: dict[str, ArrayF],
    stim_labels_trial: ArrayF,
    motion_energy: ArrayF,
    motion_times: ArrayF,
    *,
    bin_width_s: float = 0.1,
    n_lags: int = 4,
    ranks: ArrayF | list[int] | None = None,
    n_splits: int = 5,
    rank_rule: str = "one_se",
    lam: float = 1e-3,
) -> dict:
    """Audit C: window × residualization contrasts.

    ``trials_meta`` must provide arrays aligned to balanced trials:
    ``first_stim_ts``, ``center_port_ts``, ``center_port_exit_ts``,
    ``response_ts``, ``response`` (choice ±1), optional ``t_react`` / RT.
    """
    from .stringer_rrr import (
        bin_spikes_trialwise,
        build_motion_lag_design,
    )

    if ranks is None:
        ranks = np.arange(1, 9, dtype=int)

    windows = {
        "C1_fixation_to_response": ("center_port_ts", "response_ts"),
        "C2_stim_to_exit": ("first_stim_ts", "center_port_exit_ts"),
        "C4_stim_to_response": ("first_stim_ts", "response_ts"),
    }
    rows = []
    for name, (start_key, stop_key) in windows.items():
        starts = np.asarray(trials_meta[start_key], dtype=float)
        stops = np.asarray(trials_meta[stop_key], dtype=float)
        usable = (
            np.isfinite(starts) & np.isfinite(stops) & (stops > starts + bin_width_s)
        )
        if int(usable.sum()) < 10:
            rows.append(
                {
                    "condition": name,
                    "error": f"too few usable trials ({int(usable.sum())})",
                }
            )
            continue
        binned = bin_spikes_trialwise(
            spike_lists, starts[usable], stops[usable], bin_width_s=bin_width_s
        )
        y = binned.rates
        trial_ids = binned.trial_ids
        labels = np.asarray(stim_labels_trial, dtype=float)[usable][trial_ids]
        choices = np.asarray(trials_meta["response"], dtype=float)[usable][trial_ids]
        x_stim = stimulus_design_from_labels(labels)
        x_beh, valid, _meta = build_motion_lag_design(
            motion_energy,
            motion_times,
            binned.bin_starts,
            binned.bin_stops,
            trial_ids,
            n_lags=n_lags,
            lag_mode="continuous_time",
            incomplete_policy="drop",
        )
        y, x_stim, x_beh, trial_ids, labels, choices = (
            y[valid],
            x_stim[valid],
            x_beh[valid],
            trial_ids[valid],
            labels[valid],
            choices[valid],
        )
        confound = choice_category_confound_report(labels, choices, trial_ids)
        stim = fit_cv_and_basis(
            x_stim,
            y,
            trial_ids,
            ranks=ranks,
            n_splits=n_splits,
            rank_rule=rank_rule,
            lam=lam,
        )
        beh = fit_cv_and_basis(
            x_beh,
            y,
            trial_ids,
            ranks=ranks,
            n_splits=n_splits,
            rank_rule=rank_rule,
            lam=lam,
        )
        ov = overlap_summary(stim["basis"], beh["basis"])
        rows.append(
            {
                "condition": name,
                "residualize_choice": False,
                "n_samples": int(y.shape[0]),
                "n_trials": int(np.unique(trial_ids).size),
                "phi_category_choice": confound.get("phi_category_choice"),
                "stim_selected_rank": stim["selected_rank"],
                "stim_selected_heldout_varexp": float(
                    stim["mean_varexp"][stim["ranks"].index(stim["selected_rank"])]
                ),
                "min_principal_angle_deg": ov["min_principal_angle_deg"],
                "stim_in_behavior_fraction": ov["stim_in_behavior_fraction"],
            }
        )

        # C3 twin on fixation_to_response only.
        if name == "C1_fixation_to_response":
            y_res = residualize_columns_by_choice(y, choices, trial_ids)
            stim_r = fit_cv_and_basis(
                x_stim,
                y_res,
                trial_ids,
                ranks=ranks,
                n_splits=n_splits,
                rank_rule=rank_rule,
                lam=lam,
            )
            beh_r = fit_cv_and_basis(
                x_beh,
                y_res,
                trial_ids,
                ranks=ranks,
                n_splits=n_splits,
                rank_rule=rank_rule,
                lam=lam,
            )
            ov_r = overlap_summary(stim_r["basis"], beh_r["basis"])
            rows.append(
                {
                    "condition": "C3_fixation_to_response_choice_residualized",
                    "residualize_choice": True,
                    "n_samples": int(y_res.shape[0]),
                    "n_trials": int(np.unique(trial_ids).size),
                    "phi_category_choice": confound.get("phi_category_choice"),
                    "stim_selected_rank": stim_r["selected_rank"],
                    "stim_selected_heldout_varexp": float(
                        stim_r["mean_varexp"][
                            stim_r["ranks"].index(stim_r["selected_rank"])
                        ]
                    ),
                    "min_principal_angle_deg": ov_r["min_principal_angle_deg"],
                    "stim_in_behavior_fraction": ov_r["stim_in_behavior_fraction"],
                }
            )

    # Verdict heuristics from available rows.
    by_name = {r["condition"]: r for r in rows if "error" not in r}
    c1 = by_name.get("C1_fixation_to_response")
    c2 = by_name.get("C2_stim_to_exit")
    c3 = by_name.get("C3_fixation_to_response_choice_residualized")
    recommendation = "insufficient_data"
    rationale = "could not compute enough window contrasts"
    if c1 and c2 and c3:
        overlap_only_with_choice = (
            c1["stim_in_behavior_fraction"] - c2["stim_in_behavior_fraction"] > 0.05
        )
        residualization_matters = (
            abs(c1["stim_in_behavior_fraction"] - c3["stim_in_behavior_fraction"])
            > 0.05
            or abs(
                c1["stim_selected_heldout_varexp"] - c3["stim_selected_heldout_varexp"]
            )
            > 0.02
        )
        if residualization_matters or (
            np.isfinite(c1.get("phi_category_choice", np.nan))
            and abs(float(c1["phi_category_choice"])) > 0.2
            and overlap_only_with_choice
        ):
            recommendation = "prefer_stim_to_exit_or_residualize_choice"
            rationale = (
                "choice-inclusive windows and/or residualization change sensory VE "
                "or stim–behavior overlap; do not claim unresidualized "
                "fixation→response sensory subspace without controls"
            )
        else:
            recommendation = "keep_fixation_to_response_unresidualized"
            rationale = (
                "overlap/VE stable across stim_to_exit and choice residualization; "
                "method-guide window is acceptable for first pass"
            )

    return {
        "audit": "C_window_confound",
        "contrast_table": rows,
        "recommendation": recommendation,
        "verdict": rationale,
    }


def audit_cv_vs_fulldata_angles(
    x_stim: ArrayF,
    x_beh: ArrayF,
    y: ArrayF,
    trial_ids: ArrayF,
    *,
    stim_rank: int,
    beh_rank: int,
    n_splits: int = 5,
    lam: float = 1e-3,
) -> dict:
    """Audit D: fold-wise train-fit angles vs full-data refit angles."""
    y_z = zscore_columns(y)
    x_s = zscore_columns(x_stim)
    x_b = zscore_columns(x_beh)
    a_s, _ = fit_reduced_rank_regression(x_s, y_z, stim_rank, lam=lam)
    a_b, _ = fit_reduced_rank_regression(x_b, y_z, beh_rank, lam=lam)
    full = overlap_summary(subspace_basis_from_rrr(a_s), subspace_basis_from_rrr(a_b))

    fold_rows = []
    for fold_i, (train_idx, _test_idx) in enumerate(
        trial_group_splits(trial_ids, n_splits=n_splits)
    ):
        a_st, _ = fit_reduced_rank_regression(
            zscore_columns(x_stim[train_idx]),
            zscore_columns(y[train_idx]),
            stim_rank,
            lam=lam,
        )
        a_bt, _ = fit_reduced_rank_regression(
            zscore_columns(x_beh[train_idx]),
            zscore_columns(y[train_idx]),
            beh_rank,
            lam=lam,
        )
        ov = overlap_summary(
            subspace_basis_from_rrr(a_st), subspace_basis_from_rrr(a_bt)
        )
        fold_rows.append(
            {
                "fold": fold_i,
                "n_train_samples": int(train_idx.size),
                **{k: ov[k] for k in ov if k != "principal_angles_deg"},
                "principal_angles_deg": ov["principal_angles_deg"],
            }
        )

    fold_min = np.asarray(
        [r["min_principal_angle_deg"] for r in fold_rows], dtype=float
    )
    fold_sib = np.asarray(
        [r["stim_in_behavior_fraction"] for r in fold_rows], dtype=float
    )
    full_min = full["min_principal_angle_deg"]
    full_sib = full["stim_in_behavior_fraction"]
    min_spread = float(np.nanstd(fold_min, ddof=1)) if fold_min.size > 1 else 0.0
    sib_spread = float(np.nanstd(fold_sib, ddof=1)) if fold_sib.size > 1 else 0.0
    min_bias = (
        float(abs(full_min - np.nanmean(fold_min)))
        if np.isfinite(full_min) and np.isfinite(np.nanmean(fold_min))
        else float("nan")
    )

    if min_bias > 15 or min_spread > 20:
        recommendation = "require_foldwise_angles_primary"
        rationale = (
            f"full-data min∠ differs from fold mean by {min_bias:.1f}° "
            f"(fold SD={min_spread:.1f}°); report CV angles as primary"
        )
    else:
        recommendation = "full_data_ok_with_cv_primary"
        rationale = (
            f"full-data min∠ within {min_bias:.1f}° of fold mean "
            f"(fold SD={min_spread:.1f}°); keep full-data as figure convenience "
            "with CV angles as primary"
        )

    return {
        "audit": "D_cv_vs_fulldata_angles",
        "full_data": full,
        "fold_table": fold_rows,
        "summary": {
            "fold_min_angle_mean": float(np.nanmean(fold_min)),
            "fold_min_angle_std": min_spread,
            "full_min_angle": full_min,
            "abs_bias_min_angle": min_bias,
            "fold_stim_in_beh_mean": float(np.nanmean(fold_sib)),
            "fold_stim_in_beh_std": sib_spread,
            "full_stim_in_beh": full_sib,
        },
        "recommendation": recommendation,
        "verdict": rationale,
        "stim_rank": int(stim_rank),
        "beh_rank": int(beh_rank),
    }


def audit_behavior_basis_scalar_only(
    y: ArrayF,
    x_beh_scalar: ArrayF,
    x_stim: ArrayF,
    trial_ids: ArrayF,
    *,
    ranks: ArrayF | list[int] | None = None,
    n_splits: int = 5,
    rank_rule: str = "one_se",
    lam: float = 1e-3,
) -> dict:
    """Audit A partial: score scalar ME lags; record spatial ME-PC blocker."""
    if ranks is None:
        ranks = np.arange(1, 9, dtype=int)
    stim = fit_cv_and_basis(
        x_stim,
        y,
        trial_ids,
        ranks=ranks,
        n_splits=n_splits,
        rank_rule=rank_rule,
        lam=lam,
    )
    beh = fit_cv_and_basis(
        x_beh_scalar,
        y,
        trial_ids,
        ranks=ranks,
        n_splits=n_splits,
        rank_rule=rank_rule,
        lam=lam,
    )
    ov = overlap_summary(stim["basis"], beh["basis"])
    return {
        "audit": "A_behavior_basis",
        "status": "blocked_on_spatial_me_pcs",
        "blockers": [
            (
                "spatial ME-PC design (A2) unavailable: no video_svd / spatial ME-PC "
                "pipeline on this branch; cannot decide scalar vs spatial yet"
            )
        ],
        "A1_scalar_me_lags": {
            "selected_rank": beh["selected_rank"],
            "selected_heldout_varexp": float(
                beh["mean_varexp"][beh["ranks"].index(beh["selected_rank"])]
            ),
            "peak_heldout_varexp": float(np.nanmax(beh["mean_varexp"])),
            "overlap_with_stim": ov,
        },
        "recommendation": "blocked_need_spatial_me_pcs",
        "verdict": (
            "A1 scalar ME lags computed, but Audit A cannot recommend keep/upgrade "
            "without A2 spatial ME-PCs"
        ),
    }


def json_safe(obj: Any) -> Any:
    """Convert numpy types for JSON serialization."""
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        return val if np.isfinite(val) else None
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj
