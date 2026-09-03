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


def recover_audio_epochs(
    data: np.ndarray,
    sample_rate_hz: float,
    threshold: float = DEFAULT_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Return audio epochs, threshold, and binned peak-to-peak amplitudes."""
    samples_per_bin = sample_rate_hz * BIN_MS / 1000.0
    n_bins = int(data.shape[0] // samples_per_bin)
    if not n_bins:
        raise ValueError("Recording is shorter than 10 ms")
    edges = np.rint(np.arange(n_bins + 1) * samples_per_bin).astype(np.int64)

    amplitudes = []
    for first_bin in range(0, n_bins, 5000):
        chunk_edges = edges[first_bin : min(first_bin + 5000, n_bins) + 1]
        samples = np.asarray(data[chunk_edges[0] : chunk_edges[-1], AUDIO_CHANNEL])
        bin_onsets = chunk_edges[:-1] - chunk_edges[0]
        amplitudes.append(
            np.maximum.reduceat(samples, bin_onsets).astype(np.float32)
            - np.minimum.reduceat(samples, bin_onsets).astype(np.float32)
        )

    amplitudes = np.concatenate(amplitudes)
    if not np.any(amplitudes):
        raise ValueError("XA1 contains no audio signal")
    active = amplitudes > threshold
    onsets = np.flatnonzero(active & np.r_[True, ~active[:-1]])
    offsets = np.flatnonzero(active & np.r_[~active[1:], True])
    if not onsets.size:
        return np.array([]), np.array([]), threshold, amplitudes

    split = (
        edges[onsets[1:]] - edges[offsets[:-1] + 1]
        > sample_rate_hz * MERGE_GAP_MS / 1000.0
    )
    onsets = edges[onsets[np.r_[True, split]]] / sample_rate_hz
    offsets = edges[offsets[np.r_[split, True]] + 1] / sample_rate_hz
    keep = offsets - onsets >= MIN_DURATION_MS / 1000.0
    return onsets[keep], offsets[keep], threshold, amplitudes


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

    onsets = np.arange(5.0)
    classified = classify_audio_events(onsets, onsets + [0.03, 0.05, 1.0, 2.0, 0.4])
    assert {name: values.tolist() for name, values in classified.items()} == {
        "audio_stim": [0.0],
        "go_cue": [1.0],
        "punish_wrong": [2.0],
        "punish_early": [3.0],
        "unknown": [4.0],
    }


def save_bpod_aligned_audio_plot(
    dataset_key: dict[str, str],
    amplitudes: np.ndarray,
    threshold: float,
    output_path: Path,
) -> None:
    """Plot the XA1 envelope around known Bpod sound-event times."""
    import matplotlib.pyplot as plt
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
    clock = (
        DatasetEvents.Digital()
        & {
            **dataset_key,
            "stream_name": "obx",
            "event_name": "io2",
        }
    ).fetch1()
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
        ("Go cue", "t_gocue", None),
        ("Wrong-choice punishment", "t_response", "punished"),
        ("Early-withdrawal punishment", "t_earlywithdraw", "early_withdrawal"),
    )
    bin_seconds = BIN_MS / 1000
    relative_bins = np.arange(round(-0.5 / bin_seconds), round(2.5 / bin_seconds) + 1)
    relative_times = relative_bins * bin_seconds

    figure, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True, sharey=True)
    for axis, (label, time_field, flag_field) in zip(axes, event_specs, strict=True):
        bpod_times = np.asarray(
            [
                float(trial[time_field])
                for trial in trials
                if trial[time_field] is not None
                and (flag_field is None or trial[flag_field])
            ]
        )
        bpod_times = bpod_times[np.isfinite(bpod_times)]
        obx_times = np.asarray(to_obx_time(bpod_times), dtype=float)
        center_bins = np.rint(obx_times / bin_seconds).astype(np.int64)
        valid = (center_bins + relative_bins[0] >= 0) & (
            center_bins + relative_bins[-1] < len(amplitudes)
        )
        center_bins = center_bins[valid]
        traces = amplitudes[center_bins[:, None] + relative_bins]

        if len(traces):
            shown = np.linspace(0, len(traces) - 1, min(30, len(traces)), dtype=int)
            axis.plot(relative_times, traces[shown].T, color="0.75", alpha=0.35)
            axis.plot(
                relative_times,
                np.median(traces, axis=0),
                color="black",
                linewidth=1.5,
            )
        else:
            axis.text(0.5, 0.5, "No aligned events", transform=axis.transAxes)
        axis.axhline(threshold, color="tab:red", linestyle="--", linewidth=1)
        axis.axvline(0, color="tab:blue", linestyle=":", linewidth=1)
        axis.set_title(f"{label} (n={len(traces)})")
        axis.set_ylabel("10 ms peak-to-peak")
        print(f"{label}: {len(bpod_times)} Bpod events, {len(traces)} plotted")

    axes[-1].set_xlabel("Time from Bpod event (s)")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    print(f"Saved diagnostic plot to {output_path}")


def insert_audio_events(
    dataset_key: dict[str, str],
    apply: bool,
    threshold: float,
    diagnostic_plot: Path | None,
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
    if diagnostic_plot is not None:
        save_bpod_aligned_audio_plot(
            dataset_key, amplitudes, threshold, diagnostic_plot
        )
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
        "--diagnostic-plot",
        type=Path,
        help="save XA1 envelopes aligned to known Bpod sound events",
    )
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.threshold <= 0:
        parser.error("--threshold must be positive")
    supplied = (args.subject, args.session, args.dataset)
    if any(supplied) and not all(supplied):
        parser.error("provide subject, session, and dataset together, or none")
    if args.diagnostic_plot is not None and not all(supplied):
        parser.error("--diagnostic-plot requires subject, session, and dataset")

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
            insert_audio_events(
                dataset_key, args.apply, args.threshold, args.diagnostic_plot
            )
        except Exception as error:
            if len(dataset_keys) == 1:
                raise
            failures.append(dataset_key)
            print(f"Skipped: {error}")
    if failures:
        parser.exit(1, f"Skipped {len(failures)} datasets\n")


if __name__ == "__main__":
    main()
