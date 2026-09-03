"""Recover XA1 audio events and insert them as ``obx:io1`` events.

Right now, SpikeGLX takes the analog inputs into the OneBox breakout board and
automatically thresholds them to convert them into digital events. In the
current implementation of labdata, analog inputs in SpikeGLX's obx binary file
are ignored and only the digital inputs are inserted into DatasetEvents.

This code loads the OneBox binary file, digitizes the signal in the audio
channel, and returns the onset and offsets for each audio event detected. These
events are: Go cue, Wrong Choice punishment, and Early Withdrawal punishment. A
separate function (`classify_audio_events`) classifies the events into the
three possible sounds that are presented in the Chipmunk task.

- Gabriel Rojas Bowe, 9/1/26
"""

import argparse
from importlib import import_module
from pathlib import Path

import numpy as np
from spks.spikeglx_utils import load_spikeglx_binary

from thesis.ephys.events import classify_audio_events

AUDIO_CHANNEL = 1
BIN_MS = 10.0
MERGE_GAP_MS = 50.0
MIN_DURATION_MS = 15.0
DEFAULT_THRESHOLD = 200.0


def binned_peak_to_peak(
    data: np.ndarray,
    sample_rate_hz: float,
    channel_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return 10 ms peak-to-peak amplitudes and bin edges for selected channels."""
    samples_per_bin = sample_rate_hz * BIN_MS / 1000.0
    n_bins = int(data.shape[0] // samples_per_bin)
    if not n_bins:
        raise ValueError("Recording is shorter than 10 ms")
    edges = np.rint(np.arange(n_bins + 1) * samples_per_bin).astype(np.int64)

    amplitudes = []
    for first_bin in range(0, n_bins, 5000):
        chunk_edges = edges[first_bin : min(first_bin + 5000, n_bins) + 1]
        samples = np.asarray(data[chunk_edges[0] : chunk_edges[-1], channel_indices])
        bin_onsets = chunk_edges[:-1] - chunk_edges[0]
        amplitudes.append(
            np.maximum.reduceat(samples, bin_onsets, axis=0).astype(np.float32)
            - np.minimum.reduceat(samples, bin_onsets, axis=0).astype(np.float32)
        )
    return np.concatenate(amplitudes), edges


def epochs_from_amplitudes(
    amplitudes: np.ndarray,
    edges: np.ndarray,
    sample_rate_hz: float,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return merged audio epochs from one channel's binned amplitudes."""
    active = amplitudes > threshold
    onsets = np.flatnonzero(active & np.r_[True, ~active[:-1]])
    offsets = np.flatnonzero(active & np.r_[~active[1:], True])
    if not onsets.size:
        return np.array([]), np.array([])

    split = (
        edges[onsets[1:]] - edges[offsets[:-1] + 1]
        > sample_rate_hz * MERGE_GAP_MS / 1000.0
    )
    onsets = edges[onsets[np.r_[True, split]]] / sample_rate_hz
    offsets = edges[offsets[np.r_[split, True]] + 1] / sample_rate_hz
    keep = offsets - onsets >= MIN_DURATION_MS / 1000.0
    return onsets[keep], offsets[keep]


def recover_audio_epochs(
    data: np.ndarray,
    sample_rate_hz: float,
    threshold: float = DEFAULT_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Return audio epochs, threshold, and binned peak-to-peak amplitudes."""
    amplitudes, edges = binned_peak_to_peak(
        data, sample_rate_hz, np.asarray([AUDIO_CHANNEL])
    )
    amplitudes = amplitudes[:, 0]
    if not np.any(amplitudes):
        raise ValueError("XA1 contains no audio signal")
    onsets, offsets = epochs_from_amplitudes(
        amplitudes, edges, sample_rate_hz, threshold
    )
    return onsets, offsets, threshold, amplitudes


def _self_check() -> None:
    data = np.zeros((1000, 2), dtype=np.int16)
    with np.testing.assert_raises_regex(ValueError, "XA1 contains no audio signal"):
        recover_audio_epochs(data, 1000.0)
    for bin_index in (20, 21, 22, 27, 28, 29, 50, 51):
        data[bin_index * 10, AUDIO_CHANNEL] = -150
        data[(bin_index + 1) * 10 - 1, AUDIO_CHANNEL] = 150
    onsets, offsets, _, _ = recover_audio_epochs(data, 1000.0)
    np.testing.assert_allclose(onsets, [0.2, 0.5])
    np.testing.assert_allclose(offsets, [0.30, 0.52])
    amplitudes, edges = binned_peak_to_peak(data, 1000.0, np.arange(2))
    assert amplitudes.shape == (100, 2)
    assert edges.shape == (101,)
    assert not np.any(amplitudes[:, 0])
    assert np.array_equal(amplitudes[:, 1] > DEFAULT_THRESHOLD, amplitudes[:, 1] > 0)

    onsets = np.arange(5.0)
    classified = classify_audio_events(onsets, onsets + [0.03, 0.05, 1.0, 2.0, 0.4])
    assert {name: values.tolist() for name, values in classified.items()} == {
        "audio_stim": [0.0],
        "go_cue": [1.0],
        "punish_wrong": [2.0],
        "punish_early": [3.0],
        "unknown": [4.0],
    }


def bpod_audio_events_in_obx_time(
    dataset_key: dict[str, str],
) -> dict[str, tuple[np.ndarray, float]]:
    """Return expected Bpod sound onsets in the OBX clock."""
    from labdata.schema import DatasetEvents
    from scipy.interpolate import CubicSpline

    Chipmunk = import_module("chipmunk").Chipmunk
    session_key = {
        "subject_name": dataset_key["subject_name"],
        "session_name": dataset_key["session_name"],
    }
    trials = (
        Chipmunk.trial_query(**session_key)
        .proj(
            "t_sync",
            "t_gocue",
            "t_response",
            "t_earlywithdraw",
            "punished",
            "early_withdrawal",
        )
        .fetch(as_dict=True)
    )
    bpod_sync_times = np.asarray(
        [float(trial["t_sync"]) for trial in trials if trial["t_sync"] is not None]
    )
    clock_relation = DatasetEvents.Digital() & {
        **dataset_key,
        "stream_name": "obx",
        "event_name": "io2",
    }
    if len(clock_relation) != 1:
        raise ValueError(f"Expected one OBX io2 row; found {len(clock_relation)}")
    clock = clock_relation.fetch1()
    clock_timestamps = np.asarray(clock["event_timestamps"], dtype=float)
    clock_values = clock["event_values"]
    clock_onsets = (
        clock_timestamps[::2]
        if clock_values is None
        else clock_timestamps[np.asarray(clock_values) == 1]
    )
    if len(clock_onsets) == len(bpod_sync_times) + 1:
        clock_onsets = clock_onsets[:-1]
    if len(bpod_sync_times) != len(clock_onsets) or len(bpod_sync_times) < 2:
        raise ValueError(
            "Cannot align Bpod to OBX: "
            f"Bpod t_sync={len(bpod_sync_times)}, OBX io2={len(clock_onsets)}"
        )
    if np.any(np.diff(bpod_sync_times) <= 0) or np.any(np.diff(clock_onsets) <= 0):
        raise ValueError("Bpod and OBX sync times must increase strictly")
    to_obx_time = CubicSpline(bpod_sync_times, clock_onsets)
    print(f"Aligned {len(bpod_sync_times)} Bpod t_sync pulses to OBX io2")

    event_specs = (
        ("go_cue", "t_gocue", None, 0.25),
        ("punish_wrong", "t_response", "punished", 1.25),
        ("punish_early", "t_earlywithdraw", "early_withdrawal", 2.25),
    )
    events = {}
    for label, time_field, flag_field, maximum_duration_s in event_specs:
        bpod_times = np.asarray(
            [
                float(trial[time_field])
                for trial in trials
                if trial[time_field] is not None
                and (flag_field is None or trial[flag_field])
            ]
        )
        bpod_times = bpod_times[np.isfinite(bpod_times)]
        events[label] = (
            np.asarray(to_obx_time(bpod_times), dtype=float),
            maximum_duration_s,
        )
    return events


def _print_event_aligned_diagnostics(
    amplitudes: np.ndarray,
    events: dict[str, tuple[np.ndarray, float]],
) -> None:
    """Print sound-locked amplitude changes for every analog channel."""
    bin_seconds = BIN_MS / 1000.0
    onset_bins = round(0.25 / bin_seconds)
    print("EVENT-ALIGNED PTP")
    for event_name, (event_times, maximum_duration_s) in events.items():
        centers = np.rint(event_times / bin_seconds).astype(np.int64)
        duration_bins = round(maximum_duration_s / bin_seconds)
        valid = (centers - duration_bins >= 0) & (
            centers + duration_bins < len(amplitudes)
        )
        centers = centers[valid]
        print(
            f"  {event_name}: Bpod={len(event_times)}, "
            f"within_recording={len(centers)}, window={maximum_duration_s:.2f}s"
        )
        if not len(centers):
            continue

        onset_offsets = np.arange(onset_bins)
        duration_offsets = np.arange(duration_bins)
        for channel in range(amplitudes.shape[1]):
            onset = amplitudes[centers[:, None] + onset_offsets, channel]
            onset_baseline = amplitudes[
                centers[:, None] - onset_offsets[::-1] - 1, channel
            ]
            duration = amplitudes[centers[:, None] + duration_offsets, channel]
            duration_baseline = amplitudes[
                centers[:, None] - duration_offsets[::-1] - 1, channel
            ]
            baseline_p99 = float(np.percentile(onset_baseline, 99))
            onset_hits = np.max(onset, axis=1) > baseline_p99
            print(
                f"    XA{channel}: onset_median="
                f"{np.median(onset):.1f}, onset_baseline={np.median(onset_baseline):.1f}, "
                f"onset_delta={np.median(onset) - np.median(onset_baseline):.1f}, "
                f"event_hit_fraction={np.mean(onset_hits):.3f}, "
                f"duration_median={np.median(duration):.1f}, "
                f"duration_baseline={np.median(duration_baseline):.1f}"
            )


def _print_threshold_sweep(
    amplitudes: np.ndarray,
    edges: np.ndarray,
    sample_rate_hz: float,
) -> None:
    """Print recovered duration classes across empirical thresholds."""
    print("THRESHOLD SWEEP")
    for channel in range(amplitudes.shape[1]):
        channel_amplitudes = amplitudes[:, channel]
        if not np.any(channel_amplitudes):
            print(f"  XA{channel}: empty")
            continue
        empirical = np.percentile(
            channel_amplitudes, [50, 75, 90, 95, 97, 99, 99.5, 99.9]
        )
        thresholds = sorted(
            {DEFAULT_THRESHOLD, *(float(value) for value in empirical if value > 0)}
        )
        print(f"  XA{channel}")
        for threshold in thresholds:
            onsets, offsets = epochs_from_amplitudes(
                channel_amplitudes, edges, sample_rate_hz, threshold
            )
            classified = classify_audio_events(onsets, offsets)
            counts = {name: len(values) for name, values in classified.items()}
            unknown_fraction = (
                counts["unknown"] / len(onsets) if len(onsets) else float("nan")
            )
            print(
                f"    threshold={threshold:.1f}, "
                f"active_bins={np.mean(channel_amplitudes > threshold):.4f}, "
                f"epochs={len(onsets)}, classes={counts}, "
                f"unknown_fraction={unknown_fraction:.3f}"
            )


def diagnose_audio_inputs(
    dataset_key: dict[str, str],
    data: np.ndarray,
    metadata: dict,
) -> None:
    """Print enough evidence to classify missing OneBox audio."""
    sample_rate_hz = float(metadata["sRateHz"])
    try:
        analog_channels = int(str(metadata["snsXaDwSy"]).split(",")[0])
    except (KeyError, ValueError) as error:
        raise ValueError("Cannot determine OneBox analog channel count") from error
    if analog_channels > data.shape[1]:
        raise ValueError(
            f"Metadata reports {analog_channels} analog channels, "
            f"but data contain {data.shape[1]} total channels"
        )

    print("AUDIO INPUT DIAGNOSTIC")
    print(f"  dataset={dataset_key}")
    print(
        f"  samples={data.shape[0]}, sample_rate_hz={sample_rate_hz:.6f}, "
        f"duration_s={data.shape[0] / sample_rate_hz:.3f}, "
        f"saved_channels={data.shape[1]}, analog_channels={analog_channels}"
    )
    for field in (
        "snsSaveChanSubset",
        "snsXaDwSy",
        "acqXaDwSy",
        "obAiRangeMin",
        "obAiRangeMax",
        "obMaxInt",
        "userNotes",
    ):
        print(f"  metadata.{field}={metadata.get(field)!r}")

    sample_step = max(1, data.shape[0] // 1_000_000)
    sampled = np.asarray(data[::sample_step, :analog_channels])
    raw_percentiles = np.percentile(sampled, [0, 1, 50, 99, 100], axis=0)
    adc_limit = float(metadata.get("obMaxInt", np.iinfo(sampled.dtype).max + 1))
    print(f"RAW SIGNAL (every {sample_step} samples; n={len(sampled)})")
    for channel in range(analog_channels):
        percentiles = "/".join(f"{value:.1f}" for value in raw_percentiles[:, channel])
        rail_fraction = np.mean(np.abs(sampled[:, channel]) >= adc_limit - 1)
        print(
            f"  XA{channel}: min/p1/median/p99/max={percentiles}, "
            f"std={np.std(sampled[:, channel]):.1f}, "
            f"adc_rail_fraction={rail_fraction:.6f}"
        )

    amplitudes, edges = binned_peak_to_peak(
        data, sample_rate_hz, np.arange(analog_channels)
    )
    amplitude_percentiles = np.percentile(
        amplitudes, [0, 1, 5, 25, 50, 75, 95, 99, 100], axis=0
    )
    print("10 MS PEAK-TO-PEAK")
    for channel in range(analog_channels):
        percentiles = "/".join(
            f"{value:.1f}" for value in amplitude_percentiles[:, channel]
        )
        print(f"  XA{channel}: min/p1/p5/p25/median/p75/p95/p99/max={percentiles}")

    try:
        events = bpod_audio_events_in_obx_time(dataset_key)
    except Exception as error:
        print(f"EVENT ALIGNMENT UNAVAILABLE: {type(error).__name__}: {error}")
    else:
        _print_event_aligned_diagnostics(amplitudes, events)
    _print_threshold_sweep(amplitudes, edges, sample_rate_hz)


def insert_audio_events(
    dataset_key: dict[str, str],
    apply: bool,
    threshold: float,
    diagnose: bool,
) -> None:
    """Recover and optionally insert audio events for one dataset."""
    from labdata.schema import Dataset, DatasetEvents, File

    event_key = {**dataset_key, "stream_name": "obx", "event_name": "io1"}
    if len(DatasetEvents.Digital() & event_key):
        raise RuntimeError(f"DatasetEvents.Digital already contains {event_key}")

    files = (
        File()
        & (Dataset.DataFiles() & dataset_key)
        & ('file_path LIKE "%.obx.bin" OR file_path LIKE "%.obx.meta"')
    )
    file_paths = [Path(path) for path in files.fetch("file_path")]
    binary_files = [path for path in file_paths if path.name.endswith(".obx.bin")]
    metadata_files = [path for path in file_paths if path.name.endswith(".obx.meta")]
    if (
        len(binary_files) != 1
        or len(metadata_files) != 1
        or binary_files[0].with_suffix("") != metadata_files[0].with_suffix("")
    ):
        raise ValueError(
            f"Expected one matching OBX binary and metadata pair: {file_paths}"
        )

    local_files = [Path(path) for path in files.get()]
    obx_bin = next(
        (path for path in local_files if path.name.endswith(".obx.bin")), None
    )
    if obx_bin is None or not obx_bin.with_suffix(".meta").is_file():
        raise FileNotFoundError(
            "LabData did not retrieve a matching local OBX file pair"
        )

    data, metadata = load_spikeglx_binary(obx_bin)
    if diagnose:
        diagnose_audio_inputs(dataset_key, data, metadata)
        return

    onsets, offsets, threshold, amplitudes = recover_audio_epochs(
        data, float(metadata["sRateHz"]), threshold
    )
    classified = classify_audio_events(onsets, offsets)
    counts = {name: len(events) for name, events in classified.items()}
    print(counts)
    print(f"Recovered {len(onsets)} epochs at threshold {threshold:.3f}")
    amplitude_percentiles = np.percentile(
        amplitudes, [0, 1, 5, 25, 50, 75, 95, 99, 100]
    )
    print(
        "10 ms peak-to-peak (min/p1/p5/p25/median/p75/p95/p99/max): "
        + "/".join(f"{value:.1f}" for value in amplitude_percentiles)
    )
    if onsets.size:
        durations_ms = (offsets - onsets) * 1000
        percentiles = np.percentile(durations_ms, [0, 5, 25, 50, 75, 95, 100])
        print(
            "Duration ms (min/p5/p25/median/p75/p95/max): "
            + "/".join(f"{value:.1f}" for value in percentiles)
        )
        print(f"Unknown epochs: {counts['unknown'] / len(onsets):.1%}")
    if not onsets.size or len(classified["unknown"]) / len(onsets) > 0.05:
        raise ValueError("Recovered epochs do not match known task-audio durations")

    if apply:
        DatasetEvents.Digital().insert1(
            {
                **event_key,
                "event_timestamps": np.column_stack((onsets, offsets)).ravel(),
                "event_values": None,
            },
            allow_direct_insert=True,
        )
        print("Inserted obx:io1")
    else:
        print("Dry run; add --apply to insert")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recover XA1 audio events and insert them as obx:io1 events"
    )
    parser.add_argument("subject", nargs="?")
    parser.add_argument("session", nargs="?")
    parser.add_argument("dataset", nargs="?")
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="10 ms peak-to-peak amplitude threshold (default: %(default)s)",
    )
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="print raw, event-aligned, and threshold-sweep diagnostics",
    )
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.threshold <= 0:
        parser.error("--threshold must be positive")
    if args.diagnose and args.apply:
        parser.error("--diagnose and --apply cannot be combined")
    supplied = (args.subject, args.session, args.dataset)
    if any(supplied) and not all(supplied):
        parser.error("provide subject, session, and dataset together, or none")

    _self_check()
    if args.subject is not None:
        assert args.session is not None and args.dataset is not None
        dataset_keys = [
            {
                "subject_name": args.subject,
                "session_name": args.session,
                "dataset_name": args.dataset,
            }
        ]
    else:
        from labdata.schema import Dataset, DatasetEvents, File

        key_fields = ("subject_name", "session_name", "dataset_name")
        candidates = {
            tuple(row[field] for field in key_fields)
            for row in (
                File() * Dataset.DataFiles() & 'file_path LIKE "%.obx.bin"'
            ).fetch(*key_fields, as_dict=True)
        }
        existing = {
            tuple(row[field] for field in key_fields)
            for row in (
                DatasetEvents.Digital() & {"stream_name": "obx", "event_name": "io1"}
            ).fetch(*key_fields, as_dict=True)
        }
        dataset_keys = [
            dict(zip(key_fields, key)) for key in sorted(candidates - existing)
        ]
        print(f"Found {len(dataset_keys)} OBX datasets missing obx:io1")

    failures = []
    for dataset_key in dataset_keys:
        print("\n", dataset_key)
        try:
            insert_audio_events(dataset_key, args.apply, args.threshold, args.diagnose)
        except Exception as error:
            if len(dataset_keys) == 1:
                raise
            failures.append(dataset_key)
            print(f"Skipped: {error}")
    if failures:
        parser.exit(1, f"Skipped {len(failures)} datasets\n")


if __name__ == "__main__":
    main()
