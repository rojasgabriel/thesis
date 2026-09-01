"""Collect XA1 amplitudes and expected Bpod sounds for threshold comparison."""

import argparse
from pathlib import Path

import numpy as np
from spks.spikeglx_utils import load_spikeglx_binary


def bin_audio_amplitudes(data: np.ndarray, sample_rate_hz: float) -> np.ndarray:
    samples_per_bin = sample_rate_hz * 0.01
    n_bins = int(data.shape[0] // samples_per_bin)
    if not n_bins:
        raise ValueError("Recording is shorter than 10 ms")
    edges = np.rint(np.arange(n_bins + 1) * samples_per_bin).astype(np.int64)
    amplitudes = []
    for first_bin in range(0, n_bins, 5000):
        chunk_edges = edges[first_bin : min(first_bin + 5000, n_bins) + 1]
        samples = np.asarray(data[chunk_edges[0] : chunk_edges[-1], 1])
        starts = chunk_edges[:-1] - chunk_edges[0]
        amplitudes.append(
            np.maximum.reduceat(samples, starts).astype(np.float32)
            - np.minimum.reduceat(samples, starts).astype(np.float32)
        )
    return np.concatenate(amplitudes)


def align_bpod_audio_events(
    trials: list[dict], trial_starts: np.ndarray
) -> dict[str, np.ndarray]:
    fields = {
        "go_cue": ("t_gocue", lambda trial: True),
        "punish_wrong": ("t_response", lambda trial: trial["punished"]),
        "punish_early": (
            "t_earlywithdraw",
            lambda trial: trial["early_withdrawal"],
        ),
    }
    aligned = {}
    for name, (field, include) in fields.items():
        event_trials = [
            trial
            for trial in trials
            if include(trial)
            and trial["trial_num"] < len(trial_starts)
            and trial[field] is not None
            and np.isfinite(trial[field])
        ]
        if any(
            trial["t_sync"] is None or not np.isfinite(trial["t_sync"])
            for trial in event_trials
        ):
            raise ValueError(f"A recorded Bpod {name} event has no sync time")
        aligned[name] = np.asarray(
            [
                trial_starts[trial["trial_num"]] + trial[field] - trial["t_sync"]
                for trial in event_trials
            ]
        )
    return aligned


def self_check() -> None:
    data = np.zeros((20, 2), dtype=np.int16)
    data[[0, 9, 10, 19], 1] = [-10, 10, -20, 20]
    np.testing.assert_array_equal(bin_audio_amplitudes(data, 1000), [20, 40])
    trials = [
        {
            "trial_num": 0,
            "t_sync": 10.0,
            "t_gocue": 11.0,
            "t_response": 12.0,
            "t_earlywithdraw": None,
            "punished": 1,
            "early_withdrawal": 0,
        },
        {
            "trial_num": 1,
            "t_sync": 20.0,
            "t_gocue": None,
            "t_response": None,
            "t_earlywithdraw": 21.0,
            "punished": 0,
            "early_withdrawal": 1,
        },
    ]
    expected = align_bpod_audio_events(trials, np.array([100.0, 200.0]))
    np.testing.assert_array_equal(expected["go_cue"], [101.0])
    np.testing.assert_array_equal(expected["punish_wrong"], [102.0])
    np.testing.assert_array_equal(expected["punish_early"], [201.0])
    print("Self-check passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("audio_threshold_data.npz"))
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return

    from labdata.schema import Dataset, DatasetEvents, File
    from chipmunk import Chipmunk
    from labdata.utils import find_local_filepath

    rows = (File() * Dataset.DataFiles() & 'file_path LIKE "%.obx.bin"').fetch(
        "subject_name",
        "session_name",
        "dataset_name",
        "file_path",
        as_dict=True,
    )
    rows = [
        {**row, "local_path": find_local_filepath(row["file_path"])} for row in rows
    ]
    rows = sorted(
        (row for row in rows if row["local_path"]),
        key=lambda row: row["file_path"],
    )
    if not rows:
        raise FileNotFoundError(
            "No registered OBX binaries found in LabData local_paths"
        )
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")

    amplitudes = []
    offsets = [0]
    sample_rates = []
    bpod_datasets = []
    expected = {name: [] for name in ("go_cue", "punish_wrong", "punish_early")}
    expected_offsets = {name: [0] for name in expected}
    included_rows = []
    for row in rows:
        path = Path(row["local_path"])
        restriction = {
            "subject_name": row["subject_name"],
            "session_name": row["session_name"],
        }
        trials = list(
            (Chipmunk.Trial() & restriction).fetch(
                "dataset_name",
                "trial_num",
                "t_sync",
                "t_gocue",
                "t_response",
                "t_earlywithdraw",
                "punished",
                "early_withdrawal",
                as_dict=True,
                order_by="trial_num",
            )
        )
        bpod_dataset = {trial["dataset_name"] for trial in trials}
        if len(bpod_dataset) != 1:
            raise ValueError(f"Expected one Bpod dataset for {restriction}")

        trial_start_query = DatasetEvents.Digital() & {
            **restriction,
            "dataset_name": row["dataset_name"],
            "stream_name": "obx",
            "event_name": "io2",
        }
        if len(trial_start_query) != 1:
            print(f"Skipped {path}: found {len(trial_start_query)} obx:io2 rows")
            continue
        trial_start = trial_start_query.fetch1()
        timestamps = np.asarray(trial_start["event_timestamps"], dtype=float)
        event_values = trial_start.get("event_values")
        trial_starts = (
            timestamps[::2]
            if event_values is None
            else timestamps[np.asarray(event_values) == 1]
        )
        aligned = align_bpod_audio_events(trials, trial_starts)
        data, metadata = load_spikeglx_binary(path)
        sample_rate_hz = float(metadata["sRateHz"])
        values = bin_audio_amplitudes(data, sample_rate_hz)
        amplitudes.append(values.astype(np.uint16))
        offsets.append(offsets[-1] + len(values))
        sample_rates.append(sample_rate_hz)
        bpod_datasets.append(bpod_dataset.pop())
        included_rows.append(row)
        for name, times in aligned.items():
            expected[name].append(times)
            expected_offsets[name].append(expected_offsets[name][-1] + len(times))
        print(
            f"{path}: {len(values):,} bins; "
            + ", ".join(f"{name}={len(times)}" for name, times in aligned.items())
        )

    if not included_rows:
        raise ValueError("No recordings had one usable obx:io2 trial-start row")

    # ponytail: keeps 10 ms summaries in RAM; stream if very large batches need it.
    np.savez_compressed(
        args.output,
        amplitudes=np.concatenate(amplitudes),
        file_offsets=np.asarray(offsets),
        files=np.asarray([row["file_path"] for row in included_rows]),
        subjects=np.asarray([row["subject_name"] for row in included_rows]),
        sessions=np.asarray([row["session_name"] for row in included_rows]),
        datasets=np.asarray([row["dataset_name"] for row in included_rows]),
        bpod_datasets=np.asarray(bpod_datasets),
        sample_rates=np.asarray(sample_rates),
        bin_ms=10.0,
        expected_go_cue_times=np.concatenate(expected["go_cue"]),
        expected_go_cue_offsets=np.asarray(expected_offsets["go_cue"]),
        expected_punish_wrong_times=np.concatenate(expected["punish_wrong"]),
        expected_punish_wrong_offsets=np.asarray(expected_offsets["punish_wrong"]),
        expected_punish_early_times=np.concatenate(expected["punish_early"]),
        expected_punish_early_offsets=np.asarray(expected_offsets["punish_early"]),
    )
    print(f"Saved {len(included_rows)} recordings to {args.output}")


if __name__ == "__main__":
    main()
