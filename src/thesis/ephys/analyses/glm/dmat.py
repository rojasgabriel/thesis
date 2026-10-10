"""Build the shared design matrix for the V1 GLM.

This module owns the design side of the GLM, from the raw trial table to the
standardized numeric design. `write_stimulus_windows` selects trials, lays down
the equal first-flash bin grid, fixes the 60/20/20 whole-trial split, and maps
camera frames to trial windows. From that window file the rest of the module
builds every regressor column: the common task and sensory regressors (flash
kernel, pre/post-withdrawal contrast, flash-adaptation resource, initiation,
withdrawal, response entry), the contemporaneous motion-energy video PCs, and
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
VIDEO_COMPONENT_COUNTS = (10, 25, 50, 100, 200)
HISTORY_COLUMNS = 10
# Short-term synaptic-depression parameters for the flash-adaptation covariate:
# each flash spends fraction ADAPT_U of the available resource, which recovers
# toward full with time constant ADAPT_TAU_RECOVER_S between flashes.
ADAPT_U = 0.5
ADAPT_TAU_RECOVER_S = 0.2


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
    flash = RaisedCosineBasis(6, 0, 0.151, BINWIDTH_S)
    # Poke leads the first flash by 54 ms median, 113 ms at most, and the kernel
    # is truncated there, so bases past 90 ms would be empty on every trial.
    poke = RaisedCosineBasis(4, 0, 0.091, BINWIDTH_S)
    peri_exit = RaisedCosineBasis(9, 0.301, 0.301, BINWIDTH_S)
    pre_response = RaisedCosineBasis(6, 0.301, 0, BINWIDTH_S)
    return {
        "visual_flash": flash,
        "flash_pre_post_withdrawal": flash,
        "flash_adaptation": flash,
        "flash_motion_energy": flash,
        "initiation": poke,
        "withdrawal": peri_exit,
        "response_entry": pre_response,
    }


def flash_resource(modeled: np.ndarray, u: float, tau_recover_s: float) -> np.ndarray:
    """Available synaptic resource at each flash, from short-term depression.

    A resource pool starts full (R = 1). Each flash spends fraction `u` of what
    is available, then the pool recovers toward 1 with time constant
    `tau_recover_s` before the next flash. Returns R just before each flash: the
    first flash sees a full pool (R = 1, the strongest response) and later
    flashes in a dense train see a depleted pool (graded adaptation).

    This is the Tsodyks-Markram depression model (Tsodyks & Markram, 1997, PNAS),
    driven by stimulus flashes.
    """
    resource = np.ones(len(modeled))
    for position in range(1, len(modeled)):
        recovery = np.exp(-(modeled[position] - modeled[position - 1]) / tau_recover_s)
        resource[position] = 1 - (1 - resource[position - 1] * (1 - u)) * recovery
    return resource


def flash_events(trials) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return modeled flashes, the pre/post state contrast, and the raw resource.

    The resource is the available-pool trace from `flash_resource` (R = 1 at the
    first flash, lower for adapted later flashes). `build_task_design` centers it.
    """
    times, state_values, resource_values = [], [], []
    for flashes, center_entry, center_exit, response_entry in zip(
        trials["stim_pulse_times_s"],
        trials["center_entry_s"],
        trials["center_exit_s"],
        trials["response_port_entry_s"],
        strict=True,
    ):
        flashes = np.asarray(flashes, dtype=float)
        modeled = flashes[(flashes >= center_entry) & (flashes <= response_entry)]
        if not len(modeled):
            raise ValueError("Every modeled trial must contain at least one flash.")
        times.append(modeled)
        state_values.append(np.where(modeled < center_exit, -0.5, 0.5))
        resource_values.append(flash_resource(modeled, ADAPT_U, ADAPT_TAU_RECOVER_S))
    return tuple(np.concatenate(v) for v in (times, state_values, resource_values))


def build_task_design(
    alignments: np.ndarray,
    trials,
    frame_times: np.ndarray,
    motion_energy: np.ndarray,
) -> tuple[DesignMatrix, list[dict]]:
    """Build sensory and task regressors with the accepted DAMN interface.

    `motion_energy` is the total motion energy of the video frame at each of
    `frame_times`. Its value at a flash scales the flash x motion-energy block.
    """
    bases = task_temporal_bases()
    flashes, state_values, resource_raw = flash_events(trials)
    # Center and scale over the modeled flashes: the common flash kernel stays
    # the average response, and this kernel is the change per SD of movement.
    motion_raw = np.interp(flashes, frame_times, motion_energy)
    motion_values = (motion_raw - motion_raw.mean()) / motion_raw.std()
    # Center the resource so the common flash kernel stays the average response.
    # A first flash sees a full pool (R = 1); adapted later flashes see less.
    # Store two representative centered levels for the kernel figure.
    adapt_center = float(resource_raw.mean())
    adapt_values = resource_raw - adapt_center
    later = resource_raw[resource_raw < 1.0]
    adapt_levels = {
        "first": 1.0 - adapt_center,
        "late": (float(later.mean()) - adapt_center)
        if len(later)
        else 1.0 - adapt_center,
    }
    specifications = [
        (
            "visual_flash",
            flashes,
            None,
            "sensory",
            "each measured flash from initiation through response entry",
        ),
        (
            "flash_pre_post_withdrawal",
            flashes,
            state_values,
            "sensory",
            "pre-withdrawal=-0.5, post-withdrawal=+0.5",
        ),
        (
            "flash_adaptation",
            flashes,
            adapt_values,
            "sensory",
            f"centered depression resource, u={ADAPT_U}, "
            f"tau_recover={ADAPT_TAU_RECOVER_S * 1000:.0f} ms",
        ),
        (
            "flash_motion_energy",
            flashes,
            motion_values,
            "sensory",
            "total motion energy at the flash, centered, in SD over modeled flashes",
        ),
        (
            "initiation",
            trials["center_entry_s"].to_numpy(dtype=float),
            None,
            "task",
            "initiation truncated at the first flash",
        ),
        (
            "withdrawal",
            trials["center_exit_s"].to_numpy(dtype=float),
            None,
            "task",
            "one withdrawal per completed trial",
        ),
        (
            "response_entry",
            trials["response_port_entry_s"].to_numpy(dtype=float),
            None,
            "task",
            "every response; the side-independent part",
        ),
    ]
    design = DesignMatrix(alignments, PRE_S, POST_S, BINWIDTH_S)
    manifest = []
    for name, times, values, group, coding in specifications:
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
    initiation = design.regressors["initiation"]
    rows_per_trial = initiation.X.shape[0] // len(alignments)
    centers, _, _ = construct_timebins(PRE_S, POST_S, BINWIDTH_S)
    truncated = np.asarray(initiation.X).reshape(len(alignments), rows_per_trial, -1)
    truncated[:, np.asarray(centers) >= 0] = 0
    initiation._X = truncated.reshape(-1, initiation.X.shape[1])
    return design, manifest


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
        scores = data["scores"].copy()
        motion_energy = data["total"].copy()
    if scores.ndim != 2 or scores.shape[1] != max(VIDEO_COMPONENT_COUNTS):
        raise ValueError("Expected exactly 200 video-component score columns.")
    if len(frame_times) != len(scores):
        raise ValueError("Video times and score rows must match.")
    if (
        not np.isfinite(frame_times).all()
        or not np.isfinite(scores).all()
        or np.any(np.diff(frame_times) <= 0)
    ):
        raise ValueError("Video times and scores must be finite and time ordered.")

    task, task_manifest = build_task_design(
        prepared["alignments"], trials, frame_times, motion_energy
    )
    task_values = task.X
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
        "adapt_u": ADAPT_U,
        "adapt_tau_recover_s": ADAPT_TAU_RECOVER_S,
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
        and [item.get("name") for item in manifest] == list(bases)
        and [item.get("columns") for item in manifest]
        == [basis.basis.shape[1] for basis in bases.values()]
        and metadata.get("adapt_u") == ADAPT_U
        and metadata.get("adapt_tau_recover_s") == ADAPT_TAU_RECOVER_S
    )
