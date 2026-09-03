"""Backfill native OneBox digital events for a legacy ephys dataset.

Older uploads can contain a legacy NIDQ event row that caused LabData to skip
the OneBox digital stream entirely. This command reruns LabData's own OneBox
extractor and inserts only missing ``obx`` events. It leaves ``io1`` absent
because that hardware-thresholded audio channel may be incomplete; recover it
from XA1 with ``insert_audio_events`` instead.
"""

import argparse

import numpy as np

AUDIO_EVENT = "io1"


def insert_obx_events(dataset_key: dict[str, str], apply: bool) -> None:
    """Extract and optionally insert missing native OneBox events."""
    from labdata.schema import Dataset, DatasetEvents, File, dj
    from labdata.rules.ephys import extract_events_from_nidq

    stream_key = {**dataset_key, "stream_name": "obx"}
    files = File() & (Dataset.DataFiles() & dataset_key) & 'file_path LIKE "%.obx.%"'
    file_paths = files.fetch("file_path")
    if not any(str(path).endswith((".obx.bin", ".obx.cbin")) for path in file_paths):
        raise FileNotFoundError(f"No OneBox binary found for {dataset_key}")

    local_files = files.get()
    events, data = extract_events_from_nidq(local_files)
    del data

    event_names = [event["event_name"] for event in events]
    if len(event_names) != len(set(event_names)):
        raise ValueError(f"OneBox extractor returned duplicate events: {event_names}")

    existing = set((DatasetEvents.Digital() & stream_key).fetch("event_name").tolist())
    missing = [
        event
        for event in events
        if event["event_name"] not in existing and event["event_name"] != AUDIO_EVENT
    ]
    counts = {
        event["event_name"]: int(np.count_nonzero(event["event_values"] == 1))
        for event in missing
    }
    print(f"Native OneBox events to insert: {counts}")
    if AUDIO_EVENT in event_names:
        print("Leaving native io1 absent for analog-audio recovery")
    if not missing:
        print("No missing native OneBox events")
        return
    if not apply:
        print("Dry run; add --apply to insert")
        return

    with dj.conn().transaction:
        DatasetEvents().insert1(
            stream_key, skip_duplicates=True, allow_direct_insert=True
        )
        DatasetEvents.Digital().insert(
            [{**stream_key, **event} for event in missing],
            allow_direct_insert=True,
        )
    print(f"Inserted {len(missing)} native OneBox event(s)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill native OneBox digital events for a legacy dataset"
    )
    parser.add_argument("subject")
    parser.add_argument("session")
    parser.add_argument("dataset")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    dataset_key = {
        "subject_name": args.subject,
        "session_name": args.session,
        "dataset_name": args.dataset,
    }
    print(dataset_key)
    insert_obx_events(dataset_key, args.apply)


if __name__ == "__main__":
    main()
