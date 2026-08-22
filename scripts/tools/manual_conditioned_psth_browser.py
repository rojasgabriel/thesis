"""Standalone interactive browser for last-stationary vs first-movement PSTHs.

Usage:
  python scripts/tools/manual_conditioned_psth_browser.py --subject GRB058 --session 20260312_134952

Controls:
  left/right or j/l : previous/next unit
  [ / ]             : previous/next unit in same cluster (if clusters loaded)
  m                 : jump to next incompletely labeled unit
  e / g / i / n / z : label current unit as excited / suppressed / disinhibition / no-effect / noise
  t                 : label current unit with custom typed effect label
  b                 : clear effect label for current unit
  a / d / w         : set current unit as excited / suppressed / no-effect exemplar
  1 / 2 / 3         : label current unit as single / double / complex peaked
  r / u / v         : label current unit as representative / not representative / clear
  x                 : clear peak-shape label for current unit
  p                 : print current picks to terminal
  c                 : clear picks
  q                 : quit and print final picks
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.widgets import Button
from scipy.stats import sem
from spks.event_aligned import population_peth

from thesis.ephys.utils.analysis_locomotion import (
    extract_paired_stim_anchors,
    load_trial_classification,
)

SHORTCUT_HELP = """\
Navigation
  left/right or j/l : previous/next unit
  [ / ]             : previous/next unit in same cluster (if clusters loaded)
  m                 : jump to next incompletely labeled unit

Effect label (per unit)
  e / g / i / n / z : excited / suppressed / disinhibition / no-effect / noise
  t                 : type a custom label in terminal (saved in state file)
  b                 : clear effect label

Exemplar picks
  a / d / w         : set excited / suppressed / no-effect exemplar

Peak-shape label (per unit)
  1 / 2 / 3         : single / double / complex
  x                 : clear peak-shape label

Representative label (per unit)
  r / u / v         : representative yes / no / clear

Session controls
  p                 : print current picks + labels
  c                 : clear all picks + labels
  click Save PSTH   : export the current PSTH panel only as a PDF
  q                 : quit (saves and prints final state)
"""


def load_browser_data(
    subject: str, session: str, unit_criteria_id: int
) -> tuple[dict[int, np.ndarray], np.ndarray, np.ndarray]:
    from thesis.ephys.utils.io_session_units import fetch_good_units

    st_per_unit = fetch_good_units(subject, session, unit_criteria_id)
    trial_ts = load_trial_classification(subject, session)
    paired_last_stat, paired_first_move = extract_paired_stim_anchors(trial_ts)
    return st_per_unit, paired_last_stat, paired_first_move


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive conditioned PSTH browser")
    parser.add_argument("--subject", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--unit-criteria-id", type=int, default=1)
    parser.add_argument("--pre-seconds", type=float, default=0.04)
    parser.add_argument("--post-seconds", type=float, default=0.15)
    parser.add_argument("--binwidth-ms", type=int, default=10)
    parser.add_argument("--resp-start", type=float, default=0.04)
    parser.add_argument("--resp-end", type=float, default=0.10)
    parser.add_argument("--shared-ylim", action="store_true")
    parser.add_argument(
        "--state-path",
        type=str,
        default=None,
        help="Path to JSON file storing picks/labels (default: figures/manual_psth_labels_<subject>_<session>.json)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Loading data...")
    st_per_unit, paired_last_stat, paired_first_move = load_browser_data(
        args.subject, args.session, args.unit_criteria_id
    )

    unit_ids = list(st_per_unit.keys())
    spike_times = list(st_per_unit.values())

    peth_stat, bin_edges, _ = population_peth(
        all_spike_times=spike_times,
        alignment_times=paired_last_stat,
        pre_seconds=args.pre_seconds,
        post_seconds=args.post_seconds,
        binwidth_ms=args.binwidth_ms,
    )
    peth_stat = peth_stat / (args.binwidth_ms / 1000)
    bc = (bin_edges[:-1] + bin_edges[1:]) / 2
    peth_move, _, _ = population_peth(
        all_spike_times=spike_times,
        alignment_times=paired_first_move,
        pre_seconds=args.pre_seconds,
        post_seconds=args.post_seconds,
        binwidth_ms=args.binwidth_ms,
    )
    peth_move = peth_move / (args.binwidth_ms / 1000)

    if not unit_ids:
        raise RuntimeError("No units found for this session/criteria.")

    resp_mask = (bc >= args.resp_start) & (bc < args.resp_end)
    picks: dict[str, int | None] = {
        "excited": None,
        "suppressed": None,
        "no_effect": None,
    }
    effect_labels: dict[int, str] = {}
    peak_labels: dict[int, str] = {}
    representative_labels: dict[int, bool] = {}
    cluster_labels: dict[int, int] = {}
    clustering_meta: dict[str, object] = {}
    idx = 0

    default_state_path = (
        Path.cwd()
        / "figures"
        / f"manual_psth_labels_{args.subject}_{args.session}.json"
    )
    state_path = Path(args.state_path) if args.state_path else default_state_path
    state_path.parent.mkdir(parents=True, exist_ok=True)
    export_dir = Path.cwd() / "figures"
    export_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(11.8, 5.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[4.3, 1.7])
    ax = fig.add_subplot(gs[0, 0])
    side_gs = gs[0, 1].subgridspec(3, 1, height_ratios=[12, 1.2, 1.2])
    side_ax = fig.add_subplot(side_gs[0, 0])
    jump_ax = fig.add_subplot(side_gs[1, 0])
    save_ax = fig.add_subplot(side_gs[2, 0])
    jump_button = Button(jump_ax, "Next incomplete", color="0.94", hovercolor="0.85")
    save_button = Button(save_ax, "Save PSTH", color="0.94", hovercolor="0.85")
    global_y_hi = float(np.percentile(np.r_[peth_stat, peth_move], 99.5))

    def normalize_effect_label(raw_label: str) -> str:
        label = raw_label.strip().lower()
        label = label.replace("-", "_").replace(" ", "_")
        label = re.sub(r"[^a-z0-9_]", "", label)
        label = re.sub(r"_+", "_", label).strip("_")
        return label

    def save_state() -> None:
        payload = {
            "subject": args.subject,
            "session": args.session,
            "picks": picks,
            "effect_labels": {str(k): v for k, v in effect_labels.items()},
            "peak_labels": {str(k): v for k, v in peak_labels.items()},
            "representative_labels": {
                str(k): bool(v) for k, v in representative_labels.items()
            },
            "cluster_labels": {str(k): int(v) for k, v in cluster_labels.items()},
            "clustering": clustering_meta,
        }
        tmp_path = state_path.with_suffix(state_path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        tmp_path.replace(state_path)

    def load_state() -> None:
        if not state_path.exists():
            return
        try:
            payload = json.loads(state_path.read_text())
        except json.JSONDecodeError:
            print(f"Warning: could not parse state file {state_path}; starting fresh")
            return
        loaded_picks = payload.get("picks", {})
        for key in ("excited", "suppressed", "no_effect"):
            val = loaded_picks.get(key)
            picks[key] = int(val) if val is not None else None
        effect_labels.clear()
        peak_labels.clear()
        representative_labels.clear()
        cluster_labels.clear()
        effect_labels.update(
            {int(k): str(v) for k, v in payload.get("effect_labels", {}).items()}
        )
        peak_labels.update(
            {int(k): str(v) for k, v in payload.get("peak_labels", {}).items()}
        )
        representative_labels.update(
            {
                int(k): bool(v)
                for k, v in payload.get("representative_labels", {}).items()
            }
        )
        cluster_labels.update(
            {int(k): int(v) for k, v in payload.get("cluster_labels", {}).items()}
        )
        clustering_meta.clear()
        raw_meta = payload.get("clustering", {})
        if isinstance(raw_meta, dict):
            clustering_meta.update(raw_meta)
        print(f"Loaded prior labels from {state_path}")

    def print_picks(prefix: str = "Current picks") -> None:
        print(
            f"{prefix}: "
            f"excited={picks['excited']}, suppressed={picks['suppressed']}, no_effect={picks['no_effect']}"
        )
        if effect_labels:
            labels_txt = ", ".join(
                f"{uid}:{lab}" for uid, lab in sorted(effect_labels.items())
            )
            print(f"  effect_labels={{ {labels_txt} }}")
        else:
            print("  effect_labels={}")
        if peak_labels:
            labels_txt = ", ".join(
                f"{uid}:{lab}" for uid, lab in sorted(peak_labels.items())
            )
            print(f"  peak_labels={{ {labels_txt} }}")
        else:
            print("  peak_labels={}")
        if representative_labels:
            labels_txt = ", ".join(
                f"{uid}:{'yes' if lab else 'no'}"
                for uid, lab in sorted(representative_labels.items())
            )
            print(f"  representative={{ {labels_txt} }}")
        else:
            print("  representative={}")
        if cluster_labels:
            labels_txt = ", ".join(
                f"{uid}:C{lab}" for uid, lab in sorted(cluster_labels.items())
            )
            print(f"  cluster_labels={{ {labels_txt} }}")
        else:
            print("  cluster_labels={}")

    def is_fully_labeled(unit_id: int) -> bool:
        return (
            unit_id in effect_labels
            and unit_id in peak_labels
            and unit_id in representative_labels
        )

    def jump_to_next_incomplete() -> None:
        nonlocal idx
        n_units = len(unit_ids)
        for step in range(1, n_units + 1):
            cand_idx = (idx + step) % n_units
            if not is_fully_labeled(unit_ids[cand_idx]):
                idx = cand_idx
                print(f"Jumped to next incomplete unit: {unit_ids[idx]}")
                redraw()
                return
        print("All units are fully labeled (effect + peak + representative).")

    def jump_by_cluster(direction: int) -> None:
        nonlocal idx
        uid = unit_ids[idx]
        current_cluster = cluster_labels.get(uid)
        if current_cluster is None:
            print(f"Unit {uid} has no cluster label.")
            return
        n_units = len(unit_ids)
        for step in range(1, n_units + 1):
            cand_idx = (idx + direction * step) % n_units
            if cluster_labels.get(unit_ids[cand_idx]) == current_cluster:
                idx = cand_idx
                redraw()
                return
        print(f"No other units found in cluster C{current_cluster}.")

    def draw_psth(
        target_ax, ui: int, include_title: bool = True
    ) -> tuple[float, float]:
        uid = unit_ids[ui]
        ps = peth_stat[ui]
        pm = peth_move[ui]
        effect_label = effect_labels.get(uid, "unlabeled")
        peak_label = peak_labels.get(uid, "unlabeled")
        rep_label = (
            "yes"
            if representative_labels.get(uid) is True
            else "no"
            if representative_labels.get(uid) is False
            else "unlabeled"
        )
        cluster_label = cluster_labels.get(uid)

        ms = ps.mean(axis=0)
        mm = pm.mean(axis=0)
        ss = sem(ps, axis=0)
        sm = sem(pm, axis=0)

        stat_resp = float(ps[:, resp_mask].mean())
        move_resp = float(pm[:, resp_mask].mean())
        delta = move_resp - stat_resp

        target_ax.clear()
        target_ax.plot(
            bc,
            ms,
            color="steelblue",
            lw=1.8,
            label=f"last stationary (n={len(paired_last_stat)})",
        )
        target_ax.fill_between(bc, ms - ss, ms + ss, color="steelblue", alpha=0.25)
        target_ax.plot(
            bc,
            mm,
            color="darkorange",
            lw=1.8,
            label=f"first movement (n={len(paired_first_move)})",
        )
        target_ax.fill_between(bc, mm - sm, mm + sm, color="darkorange", alpha=0.25)
        target_ax.axvline(0, color="gray", linestyle="--", lw=0.8)
        target_ax.set_xlabel("Time from stim (s)")
        target_ax.set_ylabel("sp/s")
        if include_title:
            target_ax.set_title(
                f"Unit {uid} ({ui + 1}/{len(unit_ids)})  "
                f"stat={stat_resp:.2f}, move={move_resp:.2f}, Δ={delta:+.2f} sp/s\n"
                f"effect={effect_label}, peak={peak_label}, representative={rep_label}, "
                f"cluster={f'C{cluster_label}' if cluster_label is not None else 'unlabeled'}"
            )
        if args.shared_ylim:
            target_ax.set_ylim(0, max(5.0, global_y_hi))
        target_ax.legend(frameon=False, fontsize=8, loc="upper right")
        return stat_resp, move_resp

    def save_current_psth(_event: object) -> None:
        ui = max(0, min(idx, len(unit_ids) - 1))
        uid = unit_ids[ui]
        save_path = (
            export_dir / f"manual_psth_{args.subject}_{args.session}_unit{uid}.pdf"
        )
        export_fig = Figure(figsize=(6.5, 4.2))
        FigureCanvasAgg(export_fig)
        export_ax = export_fig.add_subplot(1, 1, 1)
        stat_resp, move_resp = draw_psth(export_ax, ui, include_title=False)
        delta = move_resp - stat_resp
        export_ax.set_title(
            f"Unit {uid} | stat={stat_resp:.2f} sp/s, "
            f"move={move_resp:.2f} sp/s, Δ={delta:+.2f} sp/s"
        )
        export_fig.tight_layout()
        export_fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved PSTH PDF: {save_path}")

    def redraw() -> None:
        nonlocal idx
        idx = max(0, min(idx, len(unit_ids) - 1))
        ui = idx
        draw_psth(ax, ui)

        pick_text = (
            f"exc:{picks['excited']}  supp:{picks['suppressed']}  "
            f"no:{picks['no_effect']}  effect_labeled:{len(effect_labels)}  "
            f"peak_labeled:{len(peak_labels)}  rep_labeled:{len(representative_labels)}  "
            f"clustered:{len(cluster_labels)}"
        )
        ax.text(
            0.01,
            0.01,
            pick_text,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.85, ec="0.7"),
        )

        side_ax.clear()
        side_ax.axis("off")
        side_ax.set_title("Shortcuts", fontsize=11, loc="left")
        side_ax.text(
            0.0,
            0.98,
            SHORTCUT_HELP,
            va="top",
            ha="left",
            family="monospace",
            fontsize=8.8,
        )
        if effect_labels:
            side_ax.text(
                0.0,
                0.02,
                f"Seen effect labels: {', '.join(sorted(set(effect_labels.values())))}",
                va="bottom",
                ha="left",
                fontsize=8,
                wrap=True,
            )
        if cluster_labels:
            cids = np.array(list(cluster_labels.values()), dtype=int)
            unique_c, counts = np.unique(cids, return_counts=True)
            cluster_txt = ", ".join(
                f"C{cid}:{cnt}" for cid, cnt in zip(unique_c.tolist(), counts.tolist())
            )
            side_ax.text(
                0.0,
                0.11,
                f"Clusters: {cluster_txt}",
                va="bottom",
                ha="left",
                fontsize=8,
                wrap=True,
            )
            if "selected_k" in clustering_meta:
                side_ax.text(
                    0.0,
                    0.07,
                    f"KMeans k={clustering_meta['selected_k']}",
                    va="bottom",
                    ha="left",
                    fontsize=8,
                )
        fig.canvas.draw_idle()

    def on_key(event: object) -> None:
        nonlocal idx
        key = getattr(event, "key", None)
        if key in ("right", "l"):
            idx += 1
            redraw()
        elif key in ("left", "j"):
            idx -= 1
            redraw()
        elif key in ("[", "bracketleft"):
            jump_by_cluster(-1)
        elif key in ("]", "bracketright"):
            jump_by_cluster(1)
        elif key == "m":
            jump_to_next_incomplete()
        elif key == "e":
            effect_labels[unit_ids[idx]] = "excited"
            save_state()
            print(f"Labeled unit {unit_ids[idx]} as excited")
            redraw()
        elif key == "g":
            effect_labels[unit_ids[idx]] = "suppressed"
            save_state()
            print(f"Labeled unit {unit_ids[idx]} as suppressed")
            redraw()
        elif key == "i":
            effect_labels[unit_ids[idx]] = "disinhibition"
            save_state()
            print(f"Labeled unit {unit_ids[idx]} as disinhibition")
            redraw()
        elif key == "n":
            effect_labels[unit_ids[idx]] = "no_effect"
            save_state()
            print(f"Labeled unit {unit_ids[idx]} as no-effect")
            redraw()
        elif key == "z":
            effect_labels[unit_ids[idx]] = "noise"
            save_state()
            print(f"Labeled unit {unit_ids[idx]} as noise")
            redraw()
        elif key == "t":
            raw_label = input(f"Custom effect label for unit {unit_ids[idx]}: ").strip()
            label = normalize_effect_label(raw_label)
            if not label:
                print("No valid label entered; ignored.")
                return
            effect_labels[unit_ids[idx]] = label
            save_state()
            print(f"Labeled unit {unit_ids[idx]} as {label}")
            redraw()
        elif key == "b":
            effect_labels.pop(unit_ids[idx], None)
            save_state()
            print(f"Cleared effect label for unit {unit_ids[idx]}")
            redraw()
        elif key == "a":
            picks["excited"] = unit_ids[idx]
            save_state()
            print_picks("Set excited exemplar")
            redraw()
        elif key == "d":
            picks["suppressed"] = unit_ids[idx]
            save_state()
            print_picks("Set suppressed exemplar")
            redraw()
        elif key == "w":
            picks["no_effect"] = unit_ids[idx]
            save_state()
            print_picks("Set no-effect exemplar")
            redraw()
        elif key == "1":
            peak_labels[unit_ids[idx]] = "single"
            save_state()
            print(f"Labeled unit {unit_ids[idx]} as single-peaked")
            redraw()
        elif key == "2":
            peak_labels[unit_ids[idx]] = "double"
            save_state()
            print(f"Labeled unit {unit_ids[idx]} as double-peaked")
            redraw()
        elif key == "3":
            peak_labels[unit_ids[idx]] = "complex"
            save_state()
            print(f"Labeled unit {unit_ids[idx]} as complex-peaked")
            redraw()
        elif key == "x":
            peak_labels.pop(unit_ids[idx], None)
            save_state()
            print(f"Cleared peak label for unit {unit_ids[idx]}")
            redraw()
        elif key == "r":
            representative_labels[unit_ids[idx]] = True
            save_state()
            print(f"Labeled unit {unit_ids[idx]} as representative=yes")
            redraw()
        elif key == "u":
            representative_labels[unit_ids[idx]] = False
            save_state()
            print(f"Labeled unit {unit_ids[idx]} as representative=no")
            redraw()
        elif key == "v":
            representative_labels.pop(unit_ids[idx], None)
            save_state()
            print(f"Cleared representative label for unit {unit_ids[idx]}")
            redraw()
        elif key == "c":
            picks["excited"] = None
            picks["suppressed"] = None
            picks["no_effect"] = None
            effect_labels.clear()
            peak_labels.clear()
            representative_labels.clear()
            save_state()
            print_picks("Cleared picks and labels")
            redraw()
        elif key == "p":
            print_picks("Current picks")
        elif key == "q":
            save_state()
            print_picks("Final picks")
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    jump_button.on_clicked(lambda _event: jump_to_next_incomplete())
    save_button.on_clicked(save_current_psth)
    load_state()
    print(
        "Controls: left/right or j/l navigate | [/] same-cluster prev/next | "
        "m next-incomplete | e/g/i/n/z label effect | "
        "t custom-effect-label | b clear-effect-label | "
        "a/d/w set exemplars | 1/2/3 label peak shape | x clear-peak-label | "
        "r/u/v representative yes/no/clear | "
        "p print picks | c clear | q quit"
    )
    print(f"State file: {state_path}")
    redraw()
    plt.tight_layout()
    plt.show()
    print_picks("Final picks")


if __name__ == "__main__":
    main()
