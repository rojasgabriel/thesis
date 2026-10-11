"""Build the shared design matrix for the V1 GLM.

This module owns the design side of the GLM, from the raw trial table to the
standardized numeric design. `write_stimulus_windows` selects trials, lays down
the equal first-flash bin grid, fixes the 60/20/20 whole-trial split, and maps
camera frames to trial windows. From that window file the rest of the module
builds every regressor column: the task and sensory regressors (flash kernel,
the short-interval flash term, withdrawal, the center-port state), the contemporaneous motion-energy video
PCs, and
each unit's own strictly past spike-history block. `prepare_common_design`
writes the shared task and video columns to a memmapped `.npy`;
`build_unit_history` and `_design_with_history` add the per-unit history block
at fit time. `fit.py` consumes what is here and never rebuilds the design.

Comparison, unit-selection, baseline, event alignment, exclusions, and sampling
unit are stated in `fit.py`; this module is the deterministic, validated bridge
from trial table plus video to fit-ready inputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from damn.alignment import construct_timebins
from damn.objects.basis_function_objects import RaisedCosineBasis
from damn.objects.design_matrix_objects import DesignMatrix
from damn.objects.regressor_objects import EventRegressor

from thesis.ephys.trials import build_trial_table

PRE_S = 0.1
POST_S = 2.54
BINWIDTH_S = 0.001
SPLIT_SEED = 20260912
# One count. In the first design, 25 and 50 PCs were not reliably better than
# 10 (paired over units), and 100 and 200 were worse; the smallest is used.
VIDEO_COMPONENT_COUNTS = (10,)
# Flash kernel: this long after each flash, with this many bases.
FLASH_WINDOW_S = 0.15
FLASH_BASES = 8
# The second flash term scales the flash kernel by 1 for a flash that follows
# another flash within this time, else 0 (the shortest interval is 40 ms). A
# synaptic-depression resource was tested in its place: validation D² went up as
# its recovery got faster, to the level of this term (`adaptation_grid.py`).
SECOND_FLASH_TERM = True
# A separate kernel for the first flash of a trial, added to the common kernel.
FIRST_FLASH_TERM = False
RECENT_FLASH_S = 0.05
# Causal withdrawal kernel: this long after center exit, with this many bases.
# It is short so that it holds few flashes, and it has no part before the event.
WITHDRAWAL_PRE_S = 0.0
WITHDRAWAL_WINDOW_S = 0.1
WITHDRAWAL_BASES = 4
# `withdrawal_grid.py` turns these off to test the model without each term.
WITHDRAWAL_KERNEL = True
IN_PORT_COLUMN = True
# State column: the animal is in the center port. Units fire less there, with
# the stimulus in view, than at a side port with the same motion energy. Without
# this column, motion energy absorbs part of that time course.
IN_PORT = "in_port"
HISTORY_COLUMNS = 10


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


def trial_bins(alignments: np.ndarray) -> dict:
    """Use DAMN's native equal grid and a random whole-trial 60/20/20 split."""
    alignments = np.asarray(alignments, dtype=float)
    if alignments.ndim != 1 or len(alignments) < 5:
        raise ValueError("At least five trial alignment times are required.")
    if not np.isfinite(alignments).all():
        raise ValueError("Trial alignment times must be finite.")
    centers, edges, _ = construct_timebins(PRE_S, POST_S, BINWIDTH_S)
    starts, stops = alignments + edges[0], alignments + edges[-1]
    if np.any(starts[1:] < stops[:-1]):
        raise ValueError("Windows must be ordered and non-overlapping.")
    n_bins = len(centers)
    n_trials = len(starts)
    order = np.random.default_rng(SPLIT_SEED).permutation(n_trials)
    split = np.full(n_trials, 2, dtype=np.int8)
    split[order[: int(0.8 * n_trials)]] = 1
    split[order[: int(0.6 * n_trials)]] = 0
    rows = np.repeat(np.arange(len(starts)), n_bins)
    return dict(
        bin_center_s=(alignments[:, None] + centers).ravel(),
        relative_bin_centers_s=centers,
        relative_bin_edges_s=edges,
        window_start_s=starts,
        window_stop_s=stops,
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


def task_temporal_bases() -> dict[str, RaisedCosineBasis]:
    """Return the temporal basis used for each sensory or task regressor."""
    # Most flashes are 40 ms apart. Linear spacing keeps resolution across the
    # short response window instead of spending log-spaced bases below 10 ms.
    flash = RaisedCosineBasis(FLASH_BASES, 0, FLASH_WINDOW_S + BINWIDTH_S, BINWIDTH_S)
    withdrawal = RaisedCosineBasis(
        WITHDRAWAL_BASES,
        WITHDRAWAL_PRE_S + BINWIDTH_S * (WITHDRAWAL_PRE_S > 0),
        WITHDRAWAL_WINDOW_S + BINWIDTH_S,
        BINWIDTH_S,
    )
    bases = {
        "visual_flash": flash,
        "flash_adaptation": flash,
        "flash_first": flash,
        "withdrawal": withdrawal,
    }
    if not SECOND_FLASH_TERM:
        del bases["flash_adaptation"]
    if not FIRST_FLASH_TERM:
        del bases["flash_first"]
    if not WITHDRAWAL_KERNEL:
        del bases["withdrawal"]
    return bases


def flash_recent(modeled: np.ndarray) -> np.ndarray:
    """Return 1 for each flash that follows another within `RECENT_FLASH_S`, else 0."""
    recent = np.zeros(len(modeled))
    recent[1:] = np.diff(modeled) <= RECENT_FLASH_S
    return recent


def flash_events(trials) -> tuple[np.ndarray, np.ndarray]:
    """Return modeled flashes and the `flash_recent` value of each.

    `build_task_design` centers the values.
    """
    times, resource_values = [], []
    for flashes, center_entry, response_entry in zip(
        trials["stim_pulse_times_s"],
        trials["center_entry_s"],
        trials["response_port_entry_s"],
        strict=True,
    ):
        flashes = np.asarray(flashes, dtype=float)
        modeled = flashes[(flashes >= center_entry) & (flashes <= response_entry)]
        if not len(modeled):
            raise ValueError("Every modeled trial must contain at least one flash.")
        times.append(modeled)
        resource_values.append(flash_recent(modeled))
    return np.concatenate(times), np.concatenate(resource_values)


def build_task_design(alignments: np.ndarray, trials) -> tuple[np.ndarray, list[dict]]:
    """Build the sensory and task columns: DAMN event kernels and the port state."""
    bases = task_temporal_bases()
    flashes, resource_raw = flash_events(trials)
    # Center the values so the common flash kernel stays the average response.
    # Store two representative centered levels for the kernel figure.
    adapt_center = float(resource_raw.mean())
    adapt_values = resource_raw - adapt_center
    adapt_levels = {"isolated": -adapt_center, "repeat": 1.0 - adapt_center}
    specifications = [
        (
            "visual_flash",
            flashes,
            None,
            "sensory",
            "each measured flash from initiation through response entry",
        ),
        (
            "flash_adaptation",
            flashes,
            adapt_values,
            "sensory",
            f"centered; 1 after a flash in the last {1000 * RECENT_FLASH_S:.0f} ms, "
            "else 0",
        ),
        (
            "flash_first",
            np.asarray(alignments, dtype=float),
            None,
            "sensory",
            "the first flash of each trial, in addition to the common kernel",
        ),
        (
            "withdrawal",
            trials["center_exit_s"].to_numpy(dtype=float),
            None,
            "task",
            "one withdrawal per completed trial",
        ),
    ]
    design = DesignMatrix(alignments, PRE_S, POST_S, BINWIDTH_S)
    manifest = []
    for name, times, values, group, coding in specifications:
        if name not in bases:
            continue
        basis = bases[name]
        design.add_regressor(
            EventRegressor(
                name,
                times,
                BINWIDTH_S,
                event_values=values,
                basis_objects=[basis],
                tags=group.split(),
            )
        )
        manifest.append(
            {
                "name": name,
                "group": group,
                "events": len(times),
                "columns": basis.basis.shape[1],
                "kernel_range_s": [
                    float(basis.basis_time[0]),
                    float(basis.basis_time[-1]),
                ],
                "coding": coding,
            }
        )
        if name == "flash_adaptation":
            manifest[-1]["adapt_levels_centered"] = adapt_levels
    design.build_matrix()
    if not IN_PORT_COLUMN:
        return design.X, manifest
    # One state column with no temporal basis: the animal is in the center port.
    centers, _, _ = construct_timebins(PRE_S, POST_S, BINWIDTH_S)
    times = np.asarray(alignments, dtype=float)[:, None] + centers
    in_port = (times >= trials["center_entry_s"].to_numpy(dtype=float)[:, None]) & (
        times < trials["center_exit_s"].to_numpy(dtype=float)[:, None]
    )
    manifest.append(
        {
            "name": IN_PORT,
            "group": "task",
            "events": len(alignments),
            "columns": 1,
            "coding": "1 from center entry to center exit, else 0",
        }
    )
    return np.column_stack((design.X, in_port.ravel())), manifest


def spike_history_basis() -> RaisedCosineBasis:
    """Return 10 log-spaced functions at strictly past lags 1 through 100 ms."""
    return RaisedCosineBasis(HISTORY_COLUMNS, 0, 0.1, BINWIDTH_S, log_scale=True)


def build_unit_history(alignments: np.ndarray, spike_times: np.ndarray) -> np.ndarray:
    """Return one unit's 10 strictly past self-history columns.

    Shifting the spike times by one bin keeps the filter causal, so the count in
    a bin is never a predictor of itself.
    """
    design = DesignMatrix(alignments, PRE_S, POST_S, BINWIDTH_S)
    design.add_regressor(
        EventRegressor(
            "self_history",
            np.asarray(spike_times, dtype=float) + BINWIDTH_S,
            BINWIDTH_S,
            basis_objects=[spike_history_basis()],
            tags="history",
        )
    )
    design.build_matrix()
    history = np.asarray(design.X)
    if history.shape[1] != HISTORY_COLUMNS:
        raise ValueError("Expected 10 strictly past self-history columns.")
    return history


def _design_with_history(
    common: np.ndarray,
    history: np.ndarray,
    common_columns: int,
    rows: np.ndarray | None = None,
) -> np.ndarray:
    """Append a unit's standardized history block to the shared design.

    Pass `rows` to build only the rows a fit needs. Materializing all 775,866
    and then slicing costs the full array plus every slice; roughly half those
    rows are masked out after response entry and never scored.
    """
    index = slice(None) if rows is None else rows
    height = common.shape[0] if rows is None else int(np.count_nonzero(rows))
    design = np.empty((height, common_columns + history.shape[1]), dtype=np.float32)
    design[:, :common_columns] = common[index, :common_columns]
    design[:, common_columns:] = history[index]
    return design


def _load_windows(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as windows:
        selected_rows = windows["selected_trial_rows"]
        eligible = windows["eligible_trials"]
        return {
            "metadata": json.loads(str(windows["metadata_json"])),
            "selected_rows": selected_rows.copy(),
            "selected_trial_numbers": windows["trial_num"][selected_rows].copy(),
            "alignments": windows["first_stim_s"][eligible].copy(),
            "split": windows["bin_split"].copy(),
            "bin_times": windows["bin_center_s"].copy(),
            "relative_centers": windows["relative_bin_centers_s"].copy(),
            "response_times": windows["response_entry_s"][eligible].copy(),
        }


def _load_selected_trials(prepared: dict):
    metadata = prepared["metadata"]
    trials = build_trial_table(
        metadata["subject_name"], metadata["session_name"], include_frames=False
    )
    trials = trials.iloc[prepared["selected_rows"]].reset_index(drop=True)
    if not np.array_equal(trials["trial_num"], prepared["selected_trial_numbers"]):
        raise ValueError("Prepared and live trial selections differ.")
    return trials


def _valid_bin_mask(prepared: dict) -> np.ndarray:
    """Keep bins at or before that trial's response entry."""
    limits = prepared["response_times"] - prepared["alignments"]
    return (prepared["relative_centers"][None, :] <= limits[:, None]).ravel()


def _split_masks(
    split: np.ndarray, valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train / validation / test rows after the response-entry mask."""
    split = np.asarray(split)
    if not np.array_equal(np.unique(split), [0, 1, 2]):
        raise ValueError("Rows must contain train, validation, and test labels.")
    return tuple((split == value) & valid for value in (0, 1, 2))


def prepare_common_design(windows: Path, video: Path, output: Path) -> None:
    """Build and save the shared task and motion-energy design columns."""
    prepared = _load_windows(windows)
    trials = _load_selected_trials(prepared)
    with np.load(video, allow_pickle=False) as data:
        frame_times = data["frame_times_s"].copy()
        # The video file has more PCs than the model can use; keep the first ones.
        scores = data["scores"][:, : max(VIDEO_COMPONENT_COUNTS)].copy()
    if scores.ndim != 2 or scores.shape[1] != max(VIDEO_COMPONENT_COUNTS):
        raise ValueError("The video file has too few motion-energy PCs.")
    if len(frame_times) != len(scores):
        raise ValueError("Video times and score rows must match.")
    if (
        not np.isfinite(frame_times).all()
        or not np.isfinite(scores).all()
        or np.any(np.diff(frame_times) <= 0)
    ):
        raise ValueError("Video times and scores must be finite and time ordered.")

    task_values, task_manifest = build_task_design(prepared["alignments"], trials)
    task_columns = task_values.shape[1]
    valid = _valid_bin_mask(prepared)
    train_rows, _, _ = _split_masks(prepared["split"], valid)
    scaled_base, base_mean, base_scale = training_zscore(task_values, train_rows)
    base_names = [
        f"{item['name']}_basis_{index + 1:02d}"
        for item in task_manifest
        for index in range(item["columns"])
    ]
    base_columns = len(base_names)
    total_columns = base_columns + scores.shape[1]

    output.parent.mkdir(parents=True, exist_ok=True)
    matrix_partial = output.with_name(f"{output.name}.partial")
    metadata_path = output.with_suffix(".json")
    metadata_partial = metadata_path.with_name(f"{metadata_path.name}.partial")
    for path in (matrix_partial, metadata_partial):
        if path.exists():
            raise FileExistsError(path)
    matrix = np.lib.format.open_memmap(
        matrix_partial,
        mode="w+",
        dtype=np.float32,
        shape=(len(prepared["split"]), total_columns),
    )
    matrix[:, :base_columns] = scaled_base.astype(np.float32)
    del scaled_base, task_values

    # Sample each PC at the bin centers, on the grid rebuilt from the constants
    # like the spike counts and task design, not the reusable windows file.
    centers, _, _ = construct_timebins(PRE_S, POST_S, BINWIDTH_S)
    sample_times = (prepared["alignments"][:, None] + centers).ravel()
    video_mean = []
    video_scale = []
    video_names = []
    for component in range(scores.shape[1]):
        block = np.interp(
            sample_times, frame_times, scores[:, component], left=0, right=0
        )
        scaled, mean, scale = training_zscore(block[:, None], train_rows)
        matrix[:, base_columns + component] = scaled[:, 0].astype(np.float32)
        video_mean.extend(mean.tolist())
        video_scale.extend(scale.tolist())
        video_names.append(f"video_pc_{component + 1:03d}")
        if (component + 1) % 10 == 0:
            matrix.flush()
            print(
                f"Built {component + 1} of {scores.shape[1]} video components.",
                flush=True,
            )

    metadata = {
        "rows": matrix.shape[0],
        "columns": matrix.shape[1],
        "dtype": str(matrix.dtype),
        "task_columns": task_columns,
        "base_columns": base_columns,
        "video_components": scores.shape[1],
        "video_columns_per_component": 1,
        "video_component_candidates": list(VIDEO_COMPONENT_COUNTS),
        "column_names": base_names + video_names,
        "task_manifest": task_manifest,
        "training_mean": base_mean.tolist() + video_mean,
        "training_scale": base_scale.tolist() + video_scale,
        "split_row_counts": np.bincount(prepared["split"], minlength=3).tolist(),
        "recent_flash_s": RECENT_FLASH_S,
    }
    matrix.flush()
    del matrix
    with metadata_partial.open("x") as handle:
        json.dump(metadata, handle, indent=2)
    matrix_partial.replace(output)
    metadata_partial.replace(metadata_path)
    omitted = {"column_names", "task_manifest", "training_mean", "training_scale"}
    print(
        json.dumps(
            {key: metadata[key] for key in metadata if key not in omitted}, indent=2
        )
    )


def _common_design_is_current(output: Path) -> bool:
    """Return whether a saved design has the current task and video columns."""
    metadata_path = output.with_suffix(".json")
    if not output.exists() or not metadata_path.exists():
        return False
    try:
        with metadata_path.open() as handle:
            metadata = json.load(handle)
    except (OSError, ValueError):
        return False
    bases = task_temporal_bases()
    manifest = metadata.get("task_manifest", [])
    return (
        metadata.get("video_columns_per_component") == 1
        and [item.get("name") for item in manifest]
        == [*bases, *([IN_PORT] if IN_PORT_COLUMN else [])]
        and [item.get("columns") for item in manifest]
        == [
            *(basis.basis.shape[1] for basis in bases.values()),
            *([1] * IN_PORT_COLUMN),
        ]
        and metadata.get("recent_flash_s") == RECENT_FLASH_S
    )
