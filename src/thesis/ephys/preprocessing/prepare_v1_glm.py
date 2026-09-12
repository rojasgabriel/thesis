"""Prepare equal first-flash windows for the V1 GLM comparison.

Completed left/right trials, no early withdrawal, ordered center poke, first
flash, exit, and response. Align to the first measured flash. DAMN 1 ms grid
with pre=100 ms and post=2.54 s; later fitting masks bins after response entry.
Split whole trials randomly 60/20/20 with a fixed seed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from damn.alignment import construct_timebins

from thesis.ephys.trials import build_trial_table

PRE_S = 0.1
POST_S = 2.54
BINWIDTH_S = 0.001
SPLIT_SEED = 20260912


def training_zscore(
    values: np.ndarray, train_rows: np.ndarray | slice
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scale each column with training rows and reject constants."""
    mean = values[train_rows].mean(axis=0)
    scale = values[train_rows].std(axis=0)
    if not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise ValueError("Scaling statistics must be finite.")
    if np.any(scale == 0):
        raise ValueError("Every column must vary in the training rows.")
    scaled = (values - mean) / scale
    if not np.isfinite(scaled).all():
        raise ValueError("Scaled values must be finite.")
    return scaled, mean, scale


def validate_frame_times(times: np.ndarray, n_frames: int) -> np.ndarray:
    """Reject missing, duplicate, unordered, or mismatched frame timestamps."""
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or len(times) != n_frames or n_frames < 2:
        raise ValueError("Video frame count and timestamp count must match.")
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Frame timestamps must be finite and strictly increasing.")
    return times


def trial_bins(alignments: np.ndarray, binwidth: float = BINWIDTH_S) -> dict:
    """Use DAMN's native equal grid and a random whole-trial 60/20/20 split."""
    alignments = np.asarray(alignments, dtype=float)
    if alignments.ndim != 1 or len(alignments) < 5:
        raise ValueError("At least five trial alignment times are required.")
    if not np.isfinite(alignments).all():
        raise ValueError("Trial alignment times must be finite.")
    if not np.isfinite(binwidth) or binwidth <= 0:
        raise ValueError("Bin width must be finite and positive.")
    centers, edges, _ = construct_timebins(PRE_S, POST_S, binwidth)
    starts, stops = alignments + edges[0], alignments + edges[-1]
    if np.any(starts[1:] < stops[:-1]):
        raise ValueError("Windows must be ordered and non-overlapping.")
    n_bins = len(centers)
    n_trials = len(starts)
    order = np.random.default_rng(SPLIT_SEED).permutation(n_trials)
    split = np.empty(n_trials, dtype=np.int8)
    n_train = int(0.6 * n_trials)
    n_validation = int(0.8 * n_trials) - n_train
    split[order[:n_train]] = 0
    split[order[n_train : n_train + n_validation]] = 1
    split[order[n_train + n_validation :]] = 2
    absolute_edges = alignments[:, None] + edges
    rows = np.repeat(np.arange(len(starts)), n_bins)
    return dict(
        bin_left_s=absolute_edges[:, :-1].ravel(),
        bin_right_s=absolute_edges[:, 1:].ravel(),
        bin_center_s=(alignments[:, None] + centers).ravel(),
        relative_bin_centers_s=centers,
        relative_bin_edges_s=edges,
        window_start_s=starts,
        window_stop_s=stops,
        bin_trial_row=rows,
        bin_split=split[rows],
        trial_split=split,
    )


def write_stimulus_windows(
    subject: str, session: str, output: Path, frame_times: Path
) -> None:
    """Write first-flash windows, the random split, and the camera-frame map."""
    if output.exists():
        raise FileExistsError(output)
    trials = build_trial_table(subject, session, include_frames=False)
    first = trials["stim_pulse_times_s"].str[0].to_numpy(dtype=float)
    t = trials[["center_entry_s", "center_exit_s", "response_port_entry_s"]].to_numpy(
        dtype=float
    )
    completed = (
        trials["response"].isin((-1, 1)).to_numpy()
        & trials["early_withdrawal"].fillna(1).eq(0).to_numpy()
    )
    eligible = completed & np.isfinite(first) & np.isfinite(t).all(axis=1)
    ordered = (t[:, 0] <= first) & (first < t[:, 1]) & (t[:, 1] < t[:, 2])
    if np.any(eligible & ~ordered):
        raise ValueError(
            f"Resolve event order before selecting trials: {trials.loc[eligible & ~ordered, 'trial_num'].tolist()}"
        )
    bins = trial_bins(first[eligible])
    starts, stops = bins["window_start_s"], bins["window_stop_s"]
    summary = dict(
        subject_name=subject,
        session_name=session,
        alignment_event="first measured flash",
        pre_s=PRE_S,
        post_s=POST_S,
        binwidth_s=BINWIDTH_S,
        eligible_trials=int(eligible.sum()),
        completed_trials_missing_events=int((completed & ~eligible).sum()),
        bins_per_trial=len(bins["relative_bin_centers_s"]),
        grid_source="damn.alignment.construct_timebins",
        actual_window_edges_s=bins["relative_bin_edges_s"][[0, -1]].tolist(),
        split="random whole-trial 60/20/20",
        split_seed=SPLIT_SEED,
        split_labels=["train", "validation", "test"],
        split_trial_counts=np.bincount(bins["trial_split"], minlength=3).tolist(),
        video_aligned=False,
    )
    arrays = dict(
        trial_num=trials["trial_num"].to_numpy(),
        eligible_trials=eligible,
        selected_trial_rows=np.flatnonzero(eligible),
        first_stim_s=first,
        center_entry_s=t[:, 0],
        center_exit_s=t[:, 1],
        response_entry_s=t[:, 2],
        **bins,
    )
    from labdata.schema import DatasetVideo, File

    key = dict(subject_name=subject, session_name=session, video_name="cam0")
    video = (DatasetVideo & key).fetch1()
    paths, missing = (File & (DatasetVideo.File & key)).check_if_files_local()
    if missing or len(paths) != 1 or "BackStereoView" not in str(paths[0]):
        raise ValueError("Expected one local back-view video for cam0.")
    frames = validate_frame_times(
        np.load(frame_times, allow_pickle=False), int(video["n_frames"])
    )
    rows = np.full(len(frames), -1, dtype=int)
    for i, (start, stop) in enumerate(zip(starts, stops, strict=True)):
        lo = np.searchsorted(frames, start, side="right") - 1
        hi = np.searchsorted(frames, stop)
        if lo < 0 or hi >= len(frames):
            raise ValueError(f"Video does not cover window {i}.")
        if np.any(rows[lo : hi + 1] != -1):
            raise ValueError("Video support overlaps between trial windows.")
        rows[lo : hi + 1] = i
    split = np.full(len(frames), -1, dtype=np.int8)
    keep = rows >= 0
    split[keep] = bins["trial_split"][rows[keep]]
    arrays.update(frame_times_s=frames, frame_trial_row=rows, frame_split=split)
    import subprocess

    info = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=nb_frames,width,height",
                "-of",
                "json",
                str(paths[0]),
            ],
            text=True,
        )
    )["streams"][0]
    if int(info["nb_frames"]) != len(frames):
        raise ValueError("Video header and frame timestamp counts differ.")
    summary.update(
        video_aligned=True,
        video_path=str(paths[0]),
        n_frames=len(frames),
        width=info["width"],
        height=info["height"],
        frame_times_source=str(frame_times.resolve()),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        np.savez_compressed(
            handle, allow_pickle=False, metadata_json=json.dumps(summary), **arrays
        )
    print(json.dumps(summary, indent=2))
