"""Continuous-spike Rastermap helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

DEFAULT_BIN_MS = 100.0
MIN_UNITS_FOR_RASTERMAP = 10
DEFAULT_N_PCS = 200
DEFAULT_LOCALITY = 0.75
DEFAULT_TIME_LAG_WINDOW = 5
DEFAULT_MAX_CLUSTERS = 100
DEFAULT_EVENT_COLUMNS = ("t_initiate", "t_stim", "t_react", "t_response")
RASTERMAP_EVENT_NAMES = (
    "fixation",
    "first_stim",
    "withdrawal",
    "left_choice",
    "right_choice",
)


@dataclass(frozen=True)
class ContinuousSpikeMatrix:
    unit_ids: np.ndarray
    bin_edges_s: np.ndarray
    spike_counts: np.ndarray
    trial_idx_by_bin: np.ndarray | None = None
    absolute_bin_start_s: np.ndarray | None = None
    absolute_bin_stop_s: np.ndarray | None = None


@dataclass(frozen=True)
class RastermapResult:
    subject: str
    session: str
    unit_ids: np.ndarray
    depth: np.ndarray
    bin_edges_s: np.ndarray
    spike_counts: np.ndarray
    trial_idx_by_bin: np.ndarray | None
    absolute_bin_start_s: np.ndarray | None
    absolute_bin_stop_s: np.ndarray | None
    isort: np.ndarray
    embedding: np.ndarray
    x_embedding: np.ndarray | None
    n_clusters: int
    event_names: tuple[str, ...] = ()
    event_response: np.ndarray | None = None
    event_positions_by_name: dict[str, np.ndarray] | None = None


def build_continuous_spike_count_matrix(
    spike_times_by_unit: Mapping[int, np.ndarray],
    *,
    bin_ms: float = DEFAULT_BIN_MS,
    t_start_s: float = 0.0,
    t_stop_s: float | None = None,
) -> ContinuousSpikeMatrix:
    """Bin unit spike times into a neurons-by-time count matrix."""
    if bin_ms <= 0:
        raise ValueError("bin_ms must be positive.")
    if not spike_times_by_unit:
        raise ValueError("No units were provided.")

    bin_s = bin_ms / 1000.0
    unit_ids = np.asarray(list(spike_times_by_unit.keys()), dtype=int)
    spike_arrays = [
        np.asarray(spike_times_by_unit[unit_id], dtype=float) for unit_id in unit_ids
    ]
    finite_spike_arrays = [
        spikes[np.isfinite(spikes)] for spikes in spike_arrays if spikes.size
    ]

    if t_stop_s is None:
        max_spike_times = [
            float(spikes.max()) for spikes in finite_spike_arrays if spikes.size
        ]
        if not max_spike_times:
            raise ValueError("No finite spike times were provided.")
        t_stop_s = max(max_spike_times)

    if t_stop_s <= t_start_s:
        raise ValueError("t_stop_s must be greater than t_start_s.")

    n_bins = int(np.ceil((t_stop_s - t_start_s) / bin_s))
    bin_edges_s = t_start_s + np.arange(n_bins + 1, dtype=float) * bin_s
    if bin_edges_s[-1] < t_stop_s:
        bin_edges_s = np.append(bin_edges_s, bin_edges_s[-1] + bin_s)

    spike_counts = np.zeros((len(unit_ids), len(bin_edges_s) - 1), dtype=np.float32)
    for unit_index, spikes in enumerate(spike_arrays):
        finite_spikes = spikes[np.isfinite(spikes)]
        spike_counts[unit_index], _ = np.histogram(finite_spikes, bins=bin_edges_s)

    return ContinuousSpikeMatrix(
        unit_ids=unit_ids,
        bin_edges_s=bin_edges_s,
        spike_counts=spike_counts,
    )


def trial_windows_from_metadata(trial_df) -> np.ndarray:
    """Return OBX-clock windows from initiation to response/feedback."""
    required_columns = ["t_sync", "t_initiate", "t_response", "trial_start_ts"]
    missing_columns = [
        column for column in required_columns if column not in trial_df.columns
    ]
    if missing_columns:
        raise ValueError(f"trial_df is missing columns: {missing_columns}")

    bpod_sync = trial_df["t_sync"].to_numpy(dtype=float)
    obx_sync = trial_df["trial_start_ts"].to_numpy(dtype=float)
    valid_sync = np.isfinite(bpod_sync) & np.isfinite(obx_sync)
    if valid_sync.sum() < 2:
        raise ValueError("At least two valid sync points are required.")

    t_initiate = trial_df["t_initiate"].to_numpy(dtype=float)
    t_response = trial_df["t_response"].to_numpy(dtype=float)
    finite_window = np.isfinite(t_initiate) & np.isfinite(t_response)
    start_s = np.interp(
        t_initiate[finite_window], bpod_sync[valid_sync], obx_sync[valid_sync]
    )
    stop_s = np.interp(
        t_response[finite_window], bpod_sync[valid_sync], obx_sync[valid_sync]
    )
    valid_window = stop_s > start_s
    return np.column_stack([start_s[valid_window], stop_s[valid_window]])


def trial_event_times_from_metadata(
    trial_df,
    event_columns: tuple[str, ...] = DEFAULT_EVENT_COLUMNS,
) -> dict[str, np.ndarray]:
    required_columns = ["t_sync", "trial_start_ts", *event_columns]
    missing_columns = [
        column for column in required_columns if column not in trial_df.columns
    ]
    if missing_columns:
        raise ValueError(f"trial_df is missing columns: {missing_columns}")

    bpod_sync = trial_df["t_sync"].to_numpy(dtype=float)
    obx_sync = trial_df["trial_start_ts"].to_numpy(dtype=float)
    valid_sync = np.isfinite(bpod_sync) & np.isfinite(obx_sync)
    if valid_sync.sum() < 2:
        raise ValueError("At least two valid sync points are required.")

    event_times = {}
    for column in event_columns:
        bpod_times = trial_df[column].to_numpy(dtype=float)
        finite_events = np.isfinite(bpod_times)
        event_times[column] = np.interp(
            bpod_times[finite_events], bpod_sync[valid_sync], obx_sync[valid_sync]
        )
    return event_times


def trial_event_time_table_from_metadata(
    trial_df,
    event_columns: tuple[str, ...] = DEFAULT_EVENT_COLUMNS,
) -> dict[str, np.ndarray]:
    required_columns = ["t_sync", "trial_start_ts", *event_columns]
    missing_columns = [
        column for column in required_columns if column not in trial_df.columns
    ]
    if missing_columns:
        raise ValueError(f"trial_df is missing columns: {missing_columns}")

    bpod_sync = trial_df["t_sync"].to_numpy(dtype=float)
    obx_sync = trial_df["trial_start_ts"].to_numpy(dtype=float)
    valid_sync = np.isfinite(bpod_sync) & np.isfinite(obx_sync)
    if valid_sync.sum() < 2:
        raise ValueError("At least two valid sync points are required.")

    event_times = {}
    for column in event_columns:
        bpod_times = trial_df[column].to_numpy(dtype=float)
        mapped_times = np.full_like(bpod_times, np.nan, dtype=float)
        finite_events = np.isfinite(bpod_times)
        mapped_times[finite_events] = np.interp(
            bpod_times[finite_events], bpod_sync[valid_sync], obx_sync[valid_sync]
        )
        event_times[column] = mapped_times
    return event_times


def event_response_matrix(
    spike_times_by_unit: Mapping[int, np.ndarray],
    event_times_by_name: Mapping[str, np.ndarray],
    *,
    response_window_s: tuple[float, float] = (0.0, 0.15),
    baseline_window_s: tuple[float, float] = (-0.2, 0.0),
) -> tuple[tuple[str, ...], np.ndarray]:
    if response_window_s[1] <= response_window_s[0]:
        raise ValueError("response_window_s must have positive duration.")
    if baseline_window_s[1] <= baseline_window_s[0]:
        raise ValueError("baseline_window_s must have positive duration.")

    event_names = tuple(event_times_by_name.keys())
    unit_ids = tuple(spike_times_by_unit.keys())
    response = np.full((len(unit_ids), len(event_names)), np.nan, dtype=np.float32)
    response_duration_s = response_window_s[1] - response_window_s[0]
    baseline_duration_s = baseline_window_s[1] - baseline_window_s[0]
    for unit_index, unit_id in enumerate(unit_ids):
        spikes = np.asarray(spike_times_by_unit[unit_id], dtype=float)
        spikes = spikes[np.isfinite(spikes)]
        for event_index, event_name in enumerate(event_names):
            event_times = np.asarray(event_times_by_name[event_name], dtype=float)
            event_times = event_times[np.isfinite(event_times)]
            if event_times.size == 0:
                continue
            response_counts = 0
            baseline_counts = 0
            for event_time in event_times:
                response_counts += np.count_nonzero(
                    (spikes >= event_time + response_window_s[0])
                    & (spikes < event_time + response_window_s[1])
                )
                baseline_counts += np.count_nonzero(
                    (spikes >= event_time + baseline_window_s[0])
                    & (spikes < event_time + baseline_window_s[1])
                )
            response_rate = response_counts / (event_times.size * response_duration_s)
            baseline_rate = baseline_counts / (event_times.size * baseline_duration_s)
            response[unit_index, event_index] = response_rate - baseline_rate
    return event_names, response


def trial_baseline_event_response_matrix(
    spike_times_by_unit: Mapping[int, np.ndarray],
    event_times_by_name: Mapping[str, np.ndarray],
    *,
    baseline_event_name: str = "t_initiate",
    response_window_s: tuple[float, float] = (0.0, 0.15),
    baseline_window_s: tuple[float, float] = (-0.1, 0.0),
) -> tuple[tuple[str, ...], np.ndarray]:
    if baseline_event_name not in event_times_by_name:
        raise ValueError(f"Missing baseline event: {baseline_event_name}")
    if response_window_s[1] <= response_window_s[0]:
        raise ValueError("response_window_s must have positive duration.")
    if baseline_window_s[1] <= baseline_window_s[0]:
        raise ValueError("baseline_window_s must have positive duration.")

    event_names = tuple(event_times_by_name.keys())
    baseline_times = np.asarray(event_times_by_name[baseline_event_name], dtype=float)
    unit_ids = tuple(spike_times_by_unit.keys())
    response = np.full((len(unit_ids), len(event_names)), np.nan, dtype=np.float32)
    response_duration_s = response_window_s[1] - response_window_s[0]
    baseline_duration_s = baseline_window_s[1] - baseline_window_s[0]

    for unit_index, unit_id in enumerate(unit_ids):
        spikes = np.asarray(spike_times_by_unit[unit_id], dtype=float)
        spikes = spikes[np.isfinite(spikes)]
        for event_index, event_name in enumerate(event_names):
            event_times = np.asarray(event_times_by_name[event_name], dtype=float)
            valid_trials = np.isfinite(event_times) & np.isfinite(baseline_times)
            if not np.any(valid_trials):
                continue
            response_counts = 0
            baseline_counts = 0
            for event_time, baseline_time in zip(
                event_times[valid_trials], baseline_times[valid_trials], strict=True
            ):
                response_counts += np.count_nonzero(
                    (spikes >= event_time + response_window_s[0])
                    & (spikes < event_time + response_window_s[1])
                )
                baseline_counts += np.count_nonzero(
                    (spikes >= baseline_time + baseline_window_s[0])
                    & (spikes < baseline_time + baseline_window_s[1])
                )
            n_trials = int(valid_trials.sum())
            response_rate = response_counts / (n_trials * response_duration_s)
            baseline_rate = baseline_counts / (n_trials * baseline_duration_s)
            response[unit_index, event_index] = response_rate - baseline_rate
    return event_names, response


def first_in_window(events: np.ndarray, start_s: float, stop_s: float) -> float:
    events = np.asarray(events, dtype=float)
    matches = events[(events >= start_s) & (events < stop_s)]
    if matches.size == 0:
        return np.nan
    return float(matches[0])


def hardware_trial_event_time_table(trial_df, align_ev: Mapping[str, np.ndarray]):
    required_columns = ["trial_start_ts"]
    missing_columns = [
        column for column in required_columns if column not in trial_df.columns
    ]
    if missing_columns:
        raise ValueError(f"trial_df is missing columns: {missing_columns}")

    trial_starts = trial_df["trial_start_ts"].to_numpy(dtype=float)
    valid_trial_starts = trial_starts[np.isfinite(trial_starts)]
    if valid_trial_starts.size == 0:
        raise ValueError("No finite trial_start_ts values are available.")
    median_trial_s = (
        float(np.nanmedian(np.diff(valid_trial_starts)))
        if valid_trial_starts.size > 1
        else 10.0
    )
    fallback_stop_s = float(valid_trial_starts[-1] + median_trial_s)
    trial_stops = np.r_[trial_starts[1:], fallback_stop_s]

    center_entries = np.asarray(align_ev.get("center_port", []), dtype=float)
    center_exits = np.asarray(align_ev.get("center_port_exit", []), dtype=float)
    first_stims = np.asarray(
        align_ev.get("first_stim_ev_15ms", align_ev.get("first_stim_ev", [])),
        dtype=float,
    )
    left_choices = np.asarray(align_ev.get("left_port", []), dtype=float)
    right_choices = np.asarray(align_ev.get("right_port", []), dtype=float)
    choices = np.sort(np.r_[left_choices, right_choices])

    event_times = {
        name: np.full(len(trial_starts), np.nan) for name in RASTERMAP_EVENT_NAMES
    }
    for trial_index, (start_s, stop_s) in enumerate(
        zip(trial_starts, trial_stops, strict=True)
    ):
        if not np.isfinite(start_s) or not np.isfinite(stop_s) or stop_s <= start_s:
            continue
        fixation = first_in_window(center_entries, start_s, stop_s)
        event_times["fixation"][trial_index] = fixation
        event_times["first_stim"][trial_index] = first_in_window(
            first_stims, start_s, stop_s
        )
        withdrawal_start = fixation if np.isfinite(fixation) else start_s
        withdrawal = first_in_window(center_exits, withdrawal_start, stop_s)
        event_times["withdrawal"][trial_index] = withdrawal
        choice_start = withdrawal if np.isfinite(withdrawal) else withdrawal_start
        event_times["left_choice"][trial_index] = first_in_window(
            left_choices, choice_start, stop_s
        )
        event_times["right_choice"][trial_index] = first_in_window(
            right_choices, choice_start, stop_s
        )
        choice = first_in_window(choices, choice_start, stop_s)
        if np.isfinite(choice):
            event_times["choice"] = event_times.get(
                "choice", np.full(len(trial_starts), np.nan)
            )
            event_times["choice"][trial_index] = choice
    return event_times


def iti_windows_from_hardware_events(trial_df, align_ev: Mapping[str, np.ndarray]):
    event_times = hardware_trial_event_time_table(trial_df, align_ev)
    choice_times = np.asarray(event_times["choice"], dtype=float)
    fixation_times = np.asarray(event_times["fixation"], dtype=float)
    starts = choice_times[:-1]
    stops = fixation_times[1:]
    valid_windows = np.isfinite(starts) & np.isfinite(stops) & (stops > starts)
    if not np.any(valid_windows):
        raise ValueError("No valid ITI windows were found.")
    return np.column_stack([starts[valid_windows], stops[valid_windows]])


def event_positions_in_concatenated_bins(
    matrix: ContinuousSpikeMatrix,
    event_times_by_name: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    if (
        matrix.trial_idx_by_bin is None
        or matrix.absolute_bin_start_s is None
        or matrix.absolute_bin_stop_s is None
    ):
        raise ValueError("Trial-bin metadata are required for event overlays.")

    positions_by_name = {}
    for event_name, event_times in event_times_by_name.items():
        event_times = np.asarray(event_times, dtype=float)
        positions = []
        for trial_index, event_time in enumerate(event_times):
            if not np.isfinite(event_time):
                continue
            candidate_bins = np.flatnonzero(
                (matrix.trial_idx_by_bin == trial_index)
                & (matrix.absolute_bin_start_s <= event_time)
                & (event_time < matrix.absolute_bin_stop_s)
            )
            if candidate_bins.size == 0:
                continue
            bin_index = int(candidate_bins[0])
            bin_duration_s = (
                matrix.absolute_bin_stop_s[bin_index]
                - matrix.absolute_bin_start_s[bin_index]
            )
            if bin_duration_s <= 0:
                positions.append(float(bin_index))
                continue
            fraction = (
                event_time - matrix.absolute_bin_start_s[bin_index]
            ) / bin_duration_s
            positions.append(bin_index + float(fraction))
        positions_by_name[event_name] = np.asarray(positions, dtype=float)
    return positions_by_name


def build_trial_window_spike_count_matrix(
    spike_times_by_unit: Mapping[int, np.ndarray],
    trial_windows_s: np.ndarray,
    *,
    bin_ms: float = DEFAULT_BIN_MS,
) -> ContinuousSpikeMatrix:
    """Bin spikes inside trial windows and concatenate windows in time."""
    if bin_ms <= 0:
        raise ValueError("bin_ms must be positive.")
    if not spike_times_by_unit:
        raise ValueError("No units were provided.")
    trial_windows_s = np.asarray(trial_windows_s, dtype=float)
    if trial_windows_s.ndim != 2 or trial_windows_s.shape[1] != 2:
        raise ValueError("trial_windows_s must be an array with shape (n_trials, 2).")
    finite_windows = trial_windows_s[
        np.isfinite(trial_windows_s).all(axis=1)
        & (trial_windows_s[:, 1] > trial_windows_s[:, 0])
    ]
    if finite_windows.size == 0:
        raise ValueError("No valid trial windows were provided.")

    bin_s = bin_ms / 1000.0
    unit_ids = np.asarray(list(spike_times_by_unit.keys()), dtype=int)
    spike_arrays = [
        np.asarray(spike_times_by_unit[unit_id], dtype=float) for unit_id in unit_ids
    ]

    counts_by_trial = []
    trial_idx_by_bin = []
    absolute_bin_start_s = []
    absolute_bin_stop_s = []
    elapsed_edges = [0.0]
    elapsed_s = 0.0
    for trial_index, (start_s, stop_s) in enumerate(finite_windows):
        duration_s = stop_s - start_s
        n_bins = int(np.ceil(duration_s / bin_s))
        relative_edges = np.arange(n_bins + 1, dtype=float) * bin_s
        relative_edges[-1] = duration_s
        absolute_edges = start_s + relative_edges
        trial_counts = np.zeros((len(unit_ids), n_bins), dtype=np.float32)
        for unit_index, spikes in enumerate(spike_arrays):
            finite_spikes = spikes[np.isfinite(spikes)]
            trial_counts[unit_index], _ = np.histogram(
                finite_spikes, bins=absolute_edges
            )
        counts_by_trial.append(trial_counts)
        trial_idx_by_bin.extend([trial_index] * n_bins)
        absolute_bin_start_s.extend(absolute_edges[:-1])
        absolute_bin_stop_s.extend(absolute_edges[1:])
        elapsed_s += duration_s
        elapsed_edges.extend((elapsed_s - duration_s) + relative_edges[1:])

    return ContinuousSpikeMatrix(
        unit_ids=unit_ids,
        bin_edges_s=np.asarray(elapsed_edges, dtype=float),
        spike_counts=np.concatenate(counts_by_trial, axis=1),
        trial_idx_by_bin=np.asarray(trial_idx_by_bin, dtype=int),
        absolute_bin_start_s=np.asarray(absolute_bin_start_s, dtype=float),
        absolute_bin_stop_s=np.asarray(absolute_bin_stop_s, dtype=float),
    )


def choose_rastermap_cluster_count(
    n_units: int,
    *,
    max_clusters: int = DEFAULT_MAX_CLUSTERS,
) -> int:
    if n_units < MIN_UNITS_FOR_RASTERMAP:
        raise ValueError(
            f"Rastermap needs at least {MIN_UNITS_FOR_RASTERMAP} units; got {n_units}."
        )
    if n_units < max_clusters:
        return max(2, n_units - 1)
    return max_clusters


def fit_rastermap(
    spike_counts: np.ndarray,
    *,
    rastermap_cls=None,
    n_pcs: int = DEFAULT_N_PCS,
    locality: float = DEFAULT_LOCALITY,
    time_lag_window: int = DEFAULT_TIME_LAG_WINDOW,
    max_clusters: int = DEFAULT_MAX_CLUSTERS,
):
    """Fit Rastermap on a neurons-by-time matrix."""
    if spike_counts.ndim != 2:
        raise ValueError("spike_counts must be a 2D neurons-by-time matrix.")
    n_units = spike_counts.shape[0]
    n_clusters = choose_rastermap_cluster_count(n_units, max_clusters=max_clusters)
    if rastermap_cls is None:
        from rastermap import Rastermap

        rastermap_cls = Rastermap
    model = rastermap_cls(
        n_PCs=n_pcs,
        n_clusters=n_clusters,
        locality=locality,
        time_lag_window=time_lag_window,
    ).fit(spike_counts.astype("float32", copy=False))
    return model, n_clusters


def zscore_rows(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    means = values.mean(axis=1, keepdims=True)
    stds = values.std(axis=1, keepdims=True)
    return np.divide(
        values - means,
        stds,
        out=np.zeros_like(values, dtype=float),
        where=stds > 0,
    )


def fit_session_continuous_rastermap(
    subject: str,
    session: str,
    *,
    bin_ms: float = DEFAULT_BIN_MS,
    trial_window_only: bool = True,
    iti_only: bool = False,
    rastermap_cls=None,
) -> RastermapResult:
    from ephys.src.utils.io_session_units import fetch_good_units_with_depth

    spike_times_by_unit, depth_by_unit = fetch_good_units_with_depth(subject, session)
    if trial_window_only or iti_only:
        from ephys.src.utils.io_chipmunk_trials import fetch_trial_metadata
        from ephys.src.utils.io_digital_events import fetch_session_events

        align_ev = fetch_session_events(subject, session)
        trial_df = fetch_trial_metadata(subject, session, align_ev)
        if trial_df is None:
            raise RuntimeError(
                f"Could not load trial metadata for {subject} {session}."
            )
    if iti_only:
        trial_windows_s = iti_windows_from_hardware_events(trial_df, align_ev)
        matrix = build_trial_window_spike_count_matrix(
            spike_times_by_unit,
            trial_windows_s,
            bin_ms=bin_ms,
        )
        event_names = ()
        event_response = None
        event_positions_by_name = None
    elif trial_window_only:
        trial_event_times = hardware_trial_event_time_table(trial_df, align_ev)
        valid_trial_events = (
            np.isfinite(trial_event_times["fixation"])
            & np.isfinite(trial_event_times["choice"])
            & (trial_event_times["choice"] > trial_event_times["fixation"])
        )
        if not np.any(valid_trial_events):
            raise ValueError(
                f"No hardware fixation-to-choice windows found for {subject} {session}."
            )
        trial_event_times = {
            event_name: event_times[valid_trial_events]
            for event_name, event_times in trial_event_times.items()
        }
        trial_windows_s = np.column_stack(
            [trial_event_times["fixation"], trial_event_times["choice"]]
        )
        response_event_times = {
            event_name: trial_event_times[event_name]
            for event_name in RASTERMAP_EVENT_NAMES
        }
        event_names, event_response = trial_baseline_event_response_matrix(
            spike_times_by_unit,
            response_event_times,
            baseline_event_name="fixation",
        )
        matrix = build_trial_window_spike_count_matrix(
            spike_times_by_unit,
            trial_windows_s,
            bin_ms=bin_ms,
        )
        event_positions_by_name = event_positions_in_concatenated_bins(
            matrix,
            response_event_times,
        )
    else:
        event_names = ()
        event_response = None
        event_positions_by_name = None
        matrix = build_continuous_spike_count_matrix(spike_times_by_unit, bin_ms=bin_ms)
    model, n_clusters = fit_rastermap(matrix.spike_counts, rastermap_cls=rastermap_cls)
    isort = np.asarray(model.isort, dtype=int)
    embedding = np.asarray(model.embedding)
    x_embedding = getattr(model, "X_embedding", None)
    if x_embedding is not None:
        x_embedding = np.asarray(x_embedding)
    depth = np.asarray(
        [depth_by_unit[int(unit_id)] for unit_id in matrix.unit_ids],
        dtype=float,
    )
    return RastermapResult(
        subject=subject,
        session=session,
        unit_ids=matrix.unit_ids,
        depth=depth,
        bin_edges_s=matrix.bin_edges_s,
        spike_counts=matrix.spike_counts,
        trial_idx_by_bin=matrix.trial_idx_by_bin,
        absolute_bin_start_s=matrix.absolute_bin_start_s,
        absolute_bin_stop_s=matrix.absolute_bin_stop_s,
        isort=isort,
        embedding=embedding,
        x_embedding=x_embedding,
        n_clusters=n_clusters,
        event_names=event_names,
        event_response=event_response,
        event_positions_by_name=event_positions_by_name,
    )


def heatmap_for_result(result: RastermapResult) -> np.ndarray:
    if result.x_embedding is not None and result.x_embedding.size:
        return np.asarray(result.x_embedding, dtype=float)
    return zscore_rows(result.spike_counts)[result.isort]
