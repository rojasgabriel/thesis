"""Run Rastermap on trial-window continuous spike activity."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import types

import matplotlib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if "ephys" not in sys.modules:
    package = types.ModuleType("ephys")
    package.__path__ = [str(REPO_ROOT)]
    sys.modules["ephys"] = package
sys.path.insert(0, str(REPO_ROOT))

from ephys.src.utils.analysis_rastermap import (  # noqa: E402
    DEFAULT_BIN_MS,
    RastermapResult,
    fit_session_continuous_rastermap,
    heatmap_for_result,
)

DEFAULT_SUBJECT_SESSIONS = [
    ("GRB006", "20240821_121447"),
    ("GRB058", "20260224_152424"),
    ("GRB058", "20260319_131303"),
    ("GRB058", "20260421_160125"),
    ("GRB058", "20260526_124438"),
    ("GRB059", "20260225_154153"),
    ("GRB059", "20260319_142123"),
    ("GRB059", "20260421_134936"),
    ("GRB059", "20260526_140834"),
]
FIGURE_ROOT = Path(
    os.environ.get("EPHYS_FIGURE_ROOT", "/Users/gabriel/lib/ephys/figures")
)
DEFAULT_OUTPUT_DIR = FIGURE_ROOT / "rastermap" / "training_progression"
DEFAULT_FIGURE_NAME = "training_progression_trial_window_rastermap.pdf"
ITI_FIGURE_NAME = "iti_spontaneous_rastermap.pdf"
EVENT_COLORS = {
    "fixation": "#1b9e77",
    "first_stim": "#d95f02",
    "withdrawal": "#e7298a",
    "left_choice": "#66a61e",
    "right_choice": "#e6ab02",
    "t_initiate": "#1b9e77",
    "t_stim": "#d95f02",
    "t_react": "#e7298a",
    "t_response": "#66a61e",
}
EVENT_LABELS = {
    "fixation": "fixation",
    "first_stim": "first stim",
    "withdrawal": "withdrawal",
    "left_choice": "left choice",
    "right_choice": "right choice",
    "t_initiate": "fixation",
    "t_stim": "first stim",
    "t_react": "withdrawal",
    "t_response": "choice",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subject",
        help="Run one subject/session instead of the default SfN comparison sessions.",
    )
    parser.add_argument(
        "--session",
        help="Run one subject/session instead of the default SfN comparison sessions.",
    )
    parser.add_argument(
        "--bin-ms",
        type=float,
        default=DEFAULT_BIN_MS,
        help=f"Continuous spike bin size in milliseconds. Default: {DEFAULT_BIN_MS:g}.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for PDF and NPZ outputs. Default: {DEFAULT_OUTPUT_DIR}.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Run Rastermap and build the figure without writing PDF or NPZ files.",
    )
    parser.add_argument(
        "--full-recording",
        action="store_true",
        help="Use the full recording instead of concatenated trial windows.",
    )
    parser.add_argument(
        "--iti-only",
        action="store_true",
        help="Use ITI windows from choice to next fixation. No event panel is plotted.",
    )
    return parser.parse_args()


def subject_sessions_from_args(args: argparse.Namespace) -> list[tuple[str, str]]:
    if bool(args.subject) != bool(args.session):
        raise ValueError("--subject and --session must be provided together.")
    if args.subject and args.session:
        return [(args.subject, args.session)]
    return DEFAULT_SUBJECT_SESSIONS


def save_result_npz(result: RastermapResult, output_dir: Path) -> Path:
    output_path = output_dir / f"{result.subject}_{result.session}_rastermap.npz"
    payload = {
        "unit_ids": result.unit_ids,
        "depth": result.depth,
        "bin_edges_s": result.bin_edges_s,
        "spike_counts": result.spike_counts,
        "isort": result.isort,
        "embedding": result.embedding,
        "event_names": np.asarray(result.event_names),
    }
    if result.trial_idx_by_bin is not None:
        payload["trial_idx_by_bin"] = result.trial_idx_by_bin
    if result.absolute_bin_start_s is not None:
        payload["absolute_bin_start_s"] = result.absolute_bin_start_s
    if result.absolute_bin_stop_s is not None:
        payload["absolute_bin_stop_s"] = result.absolute_bin_stop_s
    if result.x_embedding is not None:
        payload["X_embedding"] = result.x_embedding
    if result.event_response is not None:
        payload["event_response"] = result.event_response
    if result.event_positions_by_name is not None:
        for event_name, positions in result.event_positions_by_name.items():
            payload[f"event_position_{event_name}"] = positions
    np.savez_compressed(output_path, **payload)
    return output_path


def plot_results(
    results: list[RastermapResult],
    *,
    bin_ms: float,
    full_recording: bool,
    iti_only: bool = False,
):
    from matplotlib import pyplot as plt

    include_event_panel = not full_recording and not iti_only
    ncols = 2 if include_event_panel else 1
    fig, axes = plt.subplots(
        len(results),
        ncols,
        figsize=((13.0 if include_event_panel else 11.0), 4.0 * len(results)),
        squeeze=False,
        constrained_layout=True,
        width_ratios=[8.0, 1.6] if include_event_panel else None,
    )
    for row_index, result in enumerate(results):
        ax = axes[row_index, 0]
        heatmap = heatmap_for_result(result)
        vmax = max(1.0, float(np.nanpercentile(heatmap, 99)))
        image = ax.imshow(
            heatmap,
            aspect="auto",
            cmap="gray_r",
            interpolation="nearest",
            vmin=0,
            vmax=vmax,
        )
        n_bins = result.spike_counts.shape[1]
        duration_min = (n_bins * bin_ms / 1000.0) / 60.0
        if full_recording:
            title_context = "full recording"
        elif iti_only:
            title_context = "ITI windows"
        else:
            title_context = "trial windows"
        ax.set_title(
            f"{result.subject} {result.session}: "
            f"{len(result.unit_ids)} units, {duration_min:.1f} min {title_context}"
        )
        ax.set_xlabel(f"Time bin ({bin_ms:g} ms)")
        ax.set_ylabel("Rastermap-sorted units")
        fig.colorbar(image, ax=ax, fraction=0.025, pad=0.01, label="Activity")
        if not include_event_panel:
            continue

        event_ax = axes[row_index, 1]
        if result.event_response is None or len(result.event_names) == 0:
            event_ax.axis("off")
            continue
        sorted_response = result.event_response[result.isort]
        finite_abs = np.abs(sorted_response[np.isfinite(sorted_response)])
        event_vmax = max(1.0, float(np.nanpercentile(finite_abs, 98)))
        event_image = event_ax.imshow(
            sorted_response,
            aspect="auto",
            cmap="coolwarm",
            interpolation="nearest",
            vmin=-event_vmax,
            vmax=event_vmax,
        )
        event_labels = [
            EVENT_LABELS.get(name, name.removeprefix("t_"))
            for name in result.event_names
        ]
        event_ax.set_xticks(np.arange(len(event_labels)), event_labels, rotation=45)
        event_ax.set_yticks([])
        event_ax.set_title("Event response")
        event_ax.set_xlabel("0-150 ms minus 100 ms pre-initiation")
        fig.colorbar(
            event_image,
            ax=event_ax,
            fraction=0.12,
            pad=0.02,
            label="Delta sp/s",
        )
    return fig


def main() -> None:
    args = parse_args()
    matplotlib.use("Agg")
    subject_sessions = subject_sessions_from_args(args)

    results = []
    for subject, session in subject_sessions:
        print(f"\nRunning Rastermap: {subject} {session}")
        try:
            result = fit_session_continuous_rastermap(
                subject,
                session,
                bin_ms=args.bin_ms,
                trial_window_only=not args.full_recording,
                iti_only=args.iti_only,
            )
        except ValueError as exc:
            print(f"  Skipping: {exc}")
            continue
        print(
            f"  Matrix: {result.spike_counts.shape[0]} units x "
            f"{result.spike_counts.shape[1]} time bins"
        )
        print(f"  n_clusters: {result.n_clusters}")
        results.append(result)
    if not results:
        raise RuntimeError("No sessions produced Rastermap outputs.")

    fig = plot_results(
        results,
        bin_ms=args.bin_ms,
        full_recording=args.full_recording,
        iti_only=args.iti_only,
    )
    if not args.no_save:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        figure_path = args.output_dir / (
            ITI_FIGURE_NAME if args.iti_only else DEFAULT_FIGURE_NAME
        )
        fig.savefig(figure_path, bbox_inches="tight", dpi=300)
        print(f"\nSaved -> {figure_path}")
        for result in results:
            npz_path = save_result_npz(result, args.output_dir)
            print(f"Saved -> {npz_path}")


if __name__ == "__main__":
    main()
