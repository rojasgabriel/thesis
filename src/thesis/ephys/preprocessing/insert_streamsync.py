"""Insert validated StreamSync mappings for sorted ephys datasets.

Each mapping converts one source clock to the NIDQ or OneBox clock from the
same ephys dataset. Probe mappings use every available ``imecN`` stream.
Bpod mappings allow the expected extra final hardware trial-start pulse.
"""

import argparse
import warnings

import numpy as np

# Silence the setuptools pkg_resources deprecation notice
warnings.filterwarnings("ignore", category=UserWarning, module="datajoint.plugin")

CLOCK_EVENTS = {
    "nidq": {"probe": "0", "bpod": "2"},
    "obx": {"probe": "6", "bpod": "io2"},
}


def _rising_edges(timestamps: np.ndarray, values: np.ndarray | None) -> np.ndarray:
    return timestamps[::2] if values is None else timestamps[np.asarray(values) == 1]


def _strictly_matched_count(
    source_timestamps: np.ndarray,
    source_values: np.ndarray | None,
    clock_timestamps: np.ndarray,
    clock_values: np.ndarray | None,
    allow_trailing_clock_pulse: bool = False,
) -> int:
    """Return the matched pulse count or raise for an unsafe mapping."""
    source = np.asarray(source_timestamps)
    clock = np.asarray(clock_timestamps)
    if 2 * len(source) - 4 <= len(clock) <= 2 * len(source) + 4:
        clock = _rising_edges(clock, clock_values)
    elif 2 * len(clock) - 4 <= len(source) <= 2 * len(clock) + 4:
        source = _rising_edges(source, source_values)
    if allow_trailing_clock_pulse and len(clock) == len(source) + 1:
        clock = clock[:-1]
    if len(source) != len(clock):
        raise ValueError(
            f"pulse-count mismatch: source={len(source)}, clock={len(clock)}"
        )
    if len(source) < 2:
        raise ValueError(f"need at least two sync pulses, found {len(source)}")
    if np.any(np.diff(source) <= 0) or np.any(np.diff(clock) <= 0):
        raise ValueError("sync timestamps must increase strictly")
    return len(source)


def _self_check() -> None:
    source = np.array([1.0, 2.0, 3.0])
    clock = np.array([10.0, 10.1, 20.0, 20.1, 30.0, 30.1])
    values = np.array([1, 0, 1, 0, 1, 0])
    assert _strictly_matched_count(source, None, clock, values) == 3
    assert (
        _strictly_matched_count(
            source,
            None,
            np.r_[clock, 40.0, 40.1],
            np.r_[values, 1, 0],
            allow_trailing_clock_pulse=True,
        )
        == 3
    )
    with np.testing.assert_raises_regex(ValueError, "pulse-count mismatch"):
        _strictly_matched_count(
            source,
            None,
            np.r_[clock, 40.0, 40.1],
            np.r_[values, 1, 0],
        )


def insert_streamsync(dataset_key: dict[str, str], apply: bool) -> None:
    """Find, validate, and optionally insert mappings for one ephys dataset."""
    from labdata.schema import (
        DatasetEvents,
        EphysRecording,
        SpikeSorting,
        StreamSync,
        dj,
    )

    if not len(EphysRecording() & SpikeSorting() & dataset_key):
        raise ValueError(f"Dataset has no spike sorting: {dataset_key}")

    clock_streams = [
        stream
        for stream in CLOCK_EVENTS
        if len(DatasetEvents() & dataset_key & {"stream_name": stream})
    ]
    if len(clock_streams) != 1:
        raise ValueError(
            f"Expected one NIDQ or OneBox clock for {dataset_key}, found {clock_streams}"
        )
    clock_stream = clock_streams[0]
    session_key = {
        "subject_name": dataset_key["subject_name"],
        "session_name": dataset_key["session_name"],
    }

    candidates: list[tuple[dict[str, str], int, bool]] = []

    def add_mapping(
        source_key: dict[str, str],
        clock_event: str,
        allow_trailing_clock_pulse: bool = False,
    ) -> None:
        existing = StreamSync() & source_key
        if len(existing):
            print(f"Already present: {existing.fetch(as_dict=True)}")
            return
        clock_key = {
            **session_key,
            "dataset_name": dataset_key["dataset_name"],
            "stream_name": clock_stream,
            "event_name": clock_event,
        }
        source_relation = DatasetEvents.Digital() & source_key
        clock_relation = DatasetEvents.Digital() & clock_key
        if len(source_relation) != 1 or len(clock_relation) != 1:
            print(f"Skipped missing event pair: source={source_key}, clock={clock_key}")
            return
        source = source_relation.fetch1()
        clock = clock_relation.fetch1()
        try:
            count = _strictly_matched_count(
                source["event_timestamps"],
                source["event_values"],
                clock["event_timestamps"],
                clock["event_values"],
                allow_trailing_clock_pulse,
            )
        except ValueError as error:
            print(f"Skipped unsafe mapping {source_key}: {error}")
            return
        candidates.append(
            (
                {
                    **source_key,
                    "clock_dataset": clock_key["dataset_name"],
                    "clock_stream": clock_key["stream_name"],
                    "clock_stream_event": clock_key["event_name"],
                },
                count,
                allow_trailing_clock_pulse,
            )
        )

    probe_numbers = sorted(
        (EphysRecording.ProbeSetting() & dataset_key).fetch("probe_num")
    )
    for probe_number in probe_numbers:
        stream_name = f"imec{probe_number}"
        event_names = (
            DatasetEvents.Digital() & dataset_key & {"stream_name": stream_name}
        ).fetch("event_name")
        if len(event_names) != 1:
            print(
                f"Skipped {stream_name}: expected one sync event, "
                f"found {event_names.tolist()}"
            )
            continue
        add_mapping(
            {**dataset_key, "stream_name": stream_name, "event_name": event_names[0]},
            CLOCK_EVENTS[clock_stream]["probe"],
        )

    sorted_ephys = (EphysRecording() & SpikeSorting() & session_key).proj()
    bpod_key = {
        **session_key,
        "dataset_name": "chipmunk",
        "stream_name": "bpod",
        "event_name": "sync",
    }
    if len(sorted_ephys) == 1:
        add_mapping(
            bpod_key,
            CLOCK_EVENTS[clock_stream]["bpod"],
            allow_trailing_clock_pulse=True,
        )
    elif len(sorted_ephys) > 1:
        print(f"Skipped Bpod mapping: multiple sorted ephys datasets for {session_key}")

    if not candidates:
        print("No mappings to insert")
        return
    for row, count, _ in candidates:
        print(f"Validated {count} pulses: {row}")
    if not apply:
        print("Dry run; add --apply to insert")
        return

    with dj.conn().transaction:
        StreamSync().insert([row for row, _, _ in candidates], skip_duplicates=True)
        for row, expected_count, force in candidates:
            source, clock = (StreamSync() & row).get_interp_data(
                force=force, warn=not force
            )
            assert len(source) == len(clock) == expected_count
    print(f"Inserted {len(candidates)} StreamSync mapping(s)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Insert validated StreamSync mappings for a sorted ephys dataset"
    )
    parser.add_argument("subject")
    parser.add_argument("session")
    parser.add_argument("dataset")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    _self_check()
    dataset_key = {
        "subject_name": args.subject,
        "session_name": args.session,
        "dataset_name": args.dataset,
    }
    print(dataset_key)
    insert_streamsync(dataset_key, args.apply)


if __name__ == "__main__":
    main()
