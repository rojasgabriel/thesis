"""Run Stringer RRR assumption audits (A–D) for GRB006.

Orchestrated by LAB-TASKS-463. Each audit writes a tidy JSON contrast + verdict
under ``figures/stringer_rrr_preflight/audits/``. Does not run the claim-facing
analysis — that waits on the design-lock subtask.

Requires lab DB/VPN and ``.cache/categorydecoding/`` motion traces. Audit A
additionally needs spatial ME-PC tooling (absent on this branch) and will
report a partial A1 + blocker for A2.

Example::

    uv run python scripts/analyses/stringer_rrr_assumption_audits.py --audit all --quick
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if "ephys" not in sys.modules:
    package = types.ModuleType("ephys")
    package.__path__ = [str(REPO_ROOT)]
    sys.modules["ephys"] = package
sys.path.insert(0, str(REPO_ROOT))

from ephys.src.utils.stringer_rrr import (
    balanced_easy_rate_trial_mask,
    bin_spikes_trialwise,
    build_motion_lag_design,
    stimulus_design_from_labels,
    validate_design_matrices,
    validate_frame_timing,
)
from ephys.src.utils.stringer_rrr_audits import (
    audit_behavior_basis_scalar_only,
    audit_cv_vs_fulldata_angles,
    audit_stimulus_encoding,
    audit_window_confound,
    fit_cv_and_basis,
    json_safe,
)
from ephys.src.utils.stringer_rrr_events import (
    enrich_trials_with_event_timestamps,
)

_PREFLIGHT_PATH = REPO_ROOT / "scripts" / "analyses" / "stringer_rrr_preflight.py"
_spec = importlib.util.spec_from_file_location(
    "stringer_rrr_preflight", _PREFLIGHT_PATH
)
preflight = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(preflight)

SUBJECT = preflight.SUBJECT
SESSION = preflight.SESSION
UNIT_CRITERIA_ID = preflight.UNIT_CRITERIA_ID
BIN_WIDTH_S = preflight.BIN_WIDTH_S
FIGURE_DIR = preflight.FIGURE_DIR / "audits"
LAM = preflight.LAM
RANDOM_STATE = preflight.RANDOM_STATE
N_LAGS_DEFAULT = preflight.N_LAGS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit",
        choices=("behavior", "stimulus", "confound", "angles", "all"),
        default="all",
        help="Which assumption audit(s) to run.",
    )
    parser.add_argument("--quick", action="store_true", help="Smaller ranks/splits.")
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Print JSON only; do not write audit files.",
    )
    parser.add_argument(
        "--rank-rule",
        choices=("one_se", "knee"),
        default="one_se",
    )
    parser.add_argument("--n-lags", type=int, default=N_LAGS_DEFAULT)
    parser.add_argument("--stim-time-bases", type=int, default=4)
    return parser.parse_args()


def load_analysis_bundle(rng: np.random.Generator) -> dict:
    """Load units/events/trials/motion or raise SystemExit(2) with blockers."""
    blockers: list[str] = []
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
        raise SystemExit(
            json.dumps(
                {
                    "status": "blocked",
                    "blockers": blockers,
                    "motion_cache": preflight.discover_motion_cache(),
                },
                indent=2,
            )
        ) from exc

    if trial_df is None or len(trial_df) == 0:
        raise SystemExit(
            json.dumps(
                {
                    "status": "blocked",
                    "blockers": ["no trials from fetch_trial_metadata"],
                },
                indent=2,
            )
        )

    trial_df = enrich_trials_with_event_timestamps(trial_df, align_ev)
    choice_trials = trial_df[trial_df["with_choice"] == 1].reset_index(drop=True)
    balanced_mask, stim_labels_trial = balanced_easy_rate_trial_mask(
        choice_trials["stim_rate_vision"].to_numpy(dtype=float),
        rng=rng,
    )
    easy = choice_trials.loc[balanced_mask].reset_index(drop=True)
    stim_labels_trial = stim_labels_trial[balanced_mask]

    cache_info = preflight.discover_motion_cache()
    motion_energy, video_frame_count, _trace = preflight.load_motion_energy_trace(
        cache_info
    )
    frame_times, frame_source = preflight.try_load_frame_times(align_ev)
    if motion_energy is None or frame_times.size < 2:
        blockers.append(
            "motion-energy trace and/or frame_times unavailable "
            "(.cache/categorydecoding / DatasetVideo)"
        )
        raise SystemExit(
            json.dumps(
                {
                    "status": "blocked",
                    "blockers": blockers,
                    "motion_cache": cache_info,
                    "frame_source": frame_source,
                },
                indent=2,
            )
        )

    n_align = min(len(frame_times), (video_frame_count or len(frame_times)))
    frame_times = frame_times[:n_align]
    motion_energy = motion_energy[: max(n_align - 1, 0)]
    motion_times = frame_times[1 : len(motion_energy) + 1]

    timing = validate_frame_timing(
        frame_times,
        motion_energy,
        video_frame_count,
        {
            "frames": np.asarray(align_ev.get("frames", []), dtype=float),
            "first_stim_ev_15ms": np.asarray(
                align_ev.get("first_stim_ev_15ms", []), dtype=float
            ),
            "center_port_exit": easy["center_port_exit_ts"].to_numpy(dtype=float),
            "response": easy["response_ts"].to_numpy(dtype=float),
        },
    )
    if not timing["passed"]:
        blockers.extend(timing["errors"])
        raise SystemExit(
            json.dumps(
                {"status": "blocked", "blockers": blockers, "timing": timing}, indent=2
            )
        )

    unit_ids = list(st_per_unit.keys())
    spike_lists = [np.asarray(st_per_unit[uid], dtype=float) for uid in unit_ids]

    # Default analysis window for B/D/A1: fixation_to_response.
    starts = easy["center_port_ts"].to_numpy(dtype=float)
    stops = easy["response_ts"].to_numpy(dtype=float)
    usable = np.isfinite(starts) & np.isfinite(stops) & (stops > starts + BIN_WIDTH_S)
    easy = easy.loc[usable].reset_index(drop=True)
    stim_labels_trial = stim_labels_trial[usable]
    starts, stops = starts[usable], stops[usable]

    binned = bin_spikes_trialwise(spike_lists, starts, stops, bin_width_s=BIN_WIDTH_S)
    y = binned.rates
    trial_ids = binned.trial_ids
    stim_labels = stim_labels_trial[trial_ids]
    x_stim = stimulus_design_from_labels(stim_labels)
    x_beh, valid, motion_meta = build_motion_lag_design(
        motion_energy,
        motion_times,
        binned.bin_starts,
        binned.bin_stops,
        trial_ids,
        n_lags=N_LAGS_DEFAULT,
        lag_mode="continuous_time",
        incomplete_policy="drop",
    )
    y = y[valid]
    x_stim = x_stim[valid]
    x_beh = x_beh[valid]
    trial_ids = trial_ids[valid]
    stim_labels = stim_labels[valid]
    bin_idx = binned.bin_idx[valid]

    design = validate_design_matrices(
        y, x_stim, x_beh, trial_ids, stim_labels=stim_labels
    )
    if not design["passed"]:
        raise SystemExit(
            json.dumps(
                {"status": "blocked", "blockers": design["errors"], "design": design},
                indent=2,
            )
        )

    trials_meta = {
        "first_stim_ts": easy["first_stim_ts"].to_numpy(dtype=float),
        "center_port_ts": easy["center_port_ts"].to_numpy(dtype=float),
        "center_port_exit_ts": easy["center_port_exit_ts"].to_numpy(dtype=float),
        "response_ts": easy["response_ts"].to_numpy(dtype=float),
        "response": easy["response"].to_numpy(dtype=float),
    }

    return {
        "y": y,
        "x_stim": x_stim,
        "x_beh": x_beh,
        "trial_ids": trial_ids,
        "stim_labels": stim_labels,
        "bin_idx": bin_idx,
        "spike_lists": spike_lists,
        "stim_labels_trial": stim_labels_trial,
        "trials_meta": trials_meta,
        "motion_energy": motion_energy,
        "motion_times": motion_times,
        "motion_meta": motion_meta,
        "n_units": len(unit_ids),
        "n_trials": len(easy),
    }


def main() -> None:
    args = parse_args()
    rank_max = 8 if args.quick else 16
    n_splits = 3 if args.quick else 5
    ranks = np.arange(1, rank_max + 1, dtype=int)
    rng = np.random.default_rng(RANDOM_STATE)

    print(f"Loading analysis bundle for audits ({SUBJECT} {SESSION})...")
    bundle = load_analysis_bundle(rng)
    print(
        f"Bundle OK: {bundle['n_trials']} trials, "
        f"{bundle['y'].shape[0]} samples × {bundle['n_units']} units"
    )

    results: dict = {
        "status": "ok",
        "meta": {
            "subject": SUBJECT,
            "session": SESSION,
            "audit_selection": args.audit,
            "rank_rule": args.rank_rule,
            "n_splits": n_splits,
            "run_mode": "quick" if args.quick else "full",
        },
        "audits": {},
        "blockers": [],
    }

    want = {
        "behavior": args.audit in {"behavior", "all"},
        "stimulus": args.audit in {"stimulus", "all"},
        "confound": args.audit in {"confound", "all"},
        "angles": args.audit in {"angles", "all"},
    }

    if want["behavior"]:
        print("Running Audit A (behavior basis; spatial ME-PCs may block)...")
        results["audits"]["A"] = audit_behavior_basis_scalar_only(
            bundle["y"],
            bundle["x_beh"],
            bundle["x_stim"],
            bundle["trial_ids"],
            ranks=ranks,
            n_splits=n_splits,
            rank_rule=args.rank_rule,
            lam=LAM,
        )
        if results["audits"]["A"].get("blockers"):
            results["blockers"].extend(results["audits"]["A"]["blockers"])

    if want["stimulus"]:
        print("Running Audit B (stimulus encoding)...")
        results["audits"]["B"] = audit_stimulus_encoding(
            bundle["y"],
            bundle["x_beh"],
            bundle["stim_labels"],
            bundle["trial_ids"],
            bundle["bin_idx"],
            n_time_bases=args.stim_time_bases,
            ranks=ranks,
            n_splits=n_splits,
            rank_rule=args.rank_rule,
            lam=LAM,
        )

    if want["confound"]:
        print("Running Audit C (window × confound)...")
        results["audits"]["C"] = audit_window_confound(
            bundle["spike_lists"],
            bundle["trials_meta"],
            bundle["stim_labels_trial"],
            bundle["motion_energy"],
            bundle["motion_times"],
            bin_width_s=BIN_WIDTH_S,
            n_lags=args.n_lags,
            ranks=ranks,
            n_splits=n_splits,
            rank_rule=args.rank_rule,
            lam=LAM,
        )

    if want["angles"]:
        print("Running Audit D (CV vs full-data angles)...")
        # Use CV-selected ranks from default designs.
        stim_fit = fit_cv_and_basis(
            bundle["x_stim"],
            bundle["y"],
            bundle["trial_ids"],
            ranks=ranks,
            n_splits=n_splits,
            rank_rule=args.rank_rule,
            lam=LAM,
        )
        beh_fit = fit_cv_and_basis(
            bundle["x_beh"],
            bundle["y"],
            bundle["trial_ids"],
            ranks=ranks,
            n_splits=n_splits,
            rank_rule=args.rank_rule,
            lam=LAM,
        )
        results["audits"]["D"] = audit_cv_vs_fulldata_angles(
            bundle["x_stim"],
            bundle["x_beh"],
            bundle["y"],
            bundle["trial_ids"],
            stim_rank=stim_fit["selected_rank"],
            beh_rank=beh_fit["selected_rank"],
            n_splits=n_splits,
            lam=LAM,
        )

    # Strip large bases from any nested fit objects before save.
    safe = json_safe(results)
    if results["blockers"]:
        safe["status"] = "partial_blocked"

    print(json.dumps({k: safe[k] for k in ("status", "meta", "blockers")}, indent=2))
    for key, audit in safe.get("audits", {}).items():
        print(
            f"Audit {key}: recommendation={audit.get('recommendation')} | "
            f"{audit.get('verdict')}"
        )

    if not args.no_save:
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        out = FIGURE_DIR / f"audits_{args.audit}.json"
        out.write_text(json.dumps(safe, indent=2))
        print(f"Wrote {out}")
        for key, audit in safe.get("audits", {}).items():
            (FIGURE_DIR / f"audit_{key}.json").write_text(json.dumps(audit, indent=2))

    if safe["status"] == "partial_blocked":
        raise SystemExit(2)


if __name__ == "__main__":
    # Avoid unused-import lint when preflight import pulls matplotlib later.
    os.environ.setdefault("MPLBACKEND", "Agg")
    main()
