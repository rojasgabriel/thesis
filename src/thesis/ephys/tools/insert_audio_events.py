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
from pathlib import Path

import numpy as np
from spks.spikeglx_utils import load_spikeglx_binary

from thesis.ephys.events import classify_audio_events

AUDIO_CHANNEL = 1
BIN_MS = 10.0
MERGE_GAP_MS = 50.0
MIN_DURATION_MS = 15.0


def recover_audio_epochs(
    data: np.ndarray, sample_rate_hz: float
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return audio epoch onsets, offsets, and the peak-to-peak threshold."""
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
    threshold = 200.0
    active = amplitudes > threshold
    onsets = np.flatnonzero(active & np.r_[True, ~active[:-1]])
    offsets = np.flatnonzero(active & np.r_[~active[1:], True])
    if not onsets.size:
        return np.array([]), np.array([]), threshold

    split = (
        edges[onsets[1:]] - edges[offsets[:-1] + 1]
        > sample_rate_hz * MERGE_GAP_MS / 1000.0
    )
    onsets = edges[onsets[np.r_[True, split]]] / sample_rate_hz
    offsets = edges[offsets[np.r_[split, True]] + 1] / sample_rate_hz
    keep = offsets - onsets >= MIN_DURATION_MS / 1000.0
    return onsets[keep], offsets[keep], threshold


def _self_check() -> None:
    data = np.zeros((1000, 2), dtype=np.int16)
    with np.testing.assert_raises_regex(ValueError, "XA1 contains no audio signal"):
        recover_audio_epochs(data, 1000.0)
    for bin_index in (20, 21, 22, 27, 28, 29, 50, 51):
        data[bin_index * 10, AUDIO_CHANNEL] = -150
        data[(bin_index + 1) * 10 - 1, AUDIO_CHANNEL] = 150
    onsets, offsets, _ = recover_audio_epochs(data, 1000.0)
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


def insert_audio_events(dataset_key: dict[str, str], apply: bool) -> None:
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
    onsets, offsets, threshold = recover_audio_epochs(data, float(metadata["sRateHz"]))
    classified = classify_audio_events(onsets, offsets)
    if not onsets.size or len(classified["unknown"]) / len(onsets) > 0.05:
        raise ValueError("Recovered epochs do not match known task-audio durations")

    print({name: len(events) for name, events in classified.items()})
    print(f"Recovered {len(onsets)} epochs at threshold {threshold:.3f}")
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
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("subject", nargs="?")
    parser.add_argument("session", nargs="?")
    parser.add_argument("dataset", nargs="?")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
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
            insert_audio_events(dataset_key, args.apply)
        except Exception as error:
            if len(dataset_keys) == 1:
                raise
            failures.append(dataset_key)
            print(f"Skipped: {error}")
    if failures:
        parser.exit(1, f"Skipped {len(failures)} datasets; see errors above.\n")


if __name__ == "__main__":
    main()
