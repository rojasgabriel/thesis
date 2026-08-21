from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from labdata.schema import DatasetEvents

from labdata_plugin.schema import EventMapping

GRB006_KEY = {
    "subject_name": "GRB006",
    "session_name": "20240821_121447",
    "dataset_name": "ephys_g0",
    "stream_name": "nidq",
}

DEFAULT_LOGICAL_EVENT_NAMES = {
    "visual_stim": "io0",
    "trial_start": "io2",
    "frames": "io3",
    "left_port": "io4",
    "center_port": "io5",
    "right_port": "io6",
}

SESSION_MAPPING_SPECS = [
    {
        "subject_name": "GRB006",
        "session_name": "20240821_121447",
        "stream_name": "nidq",
        "source_event_names": {
            "visual_stim": "ai0",
            "trial_start": "2",
            "frames": "1",
            "left_port": "3",
            "center_port": "4",
            "right_port": "5",
        },
    },
    {
        "subject_name": "GRB058",
        "session_name": "20260224_152424",
        "source_event_names": DEFAULT_LOGICAL_EVENT_NAMES,
    },
    {
        "subject_name": "GRB058",
        "session_name": "20260312_134952",
        "source_event_names": DEFAULT_LOGICAL_EVENT_NAMES,
    },
    {
        "subject_name": "GRB058",
        "session_name": "20260319_131303",
        "source_event_names": DEFAULT_LOGICAL_EVENT_NAMES,
    },
]


def fetch_session_rows(subject: str, session: str) -> pd.DataFrame:
    restriction = {"subject_name": subject, "session_name": session}
    rows = pd.DataFrame((DatasetEvents.Digital() & restriction).fetch(as_dict=True))
    if rows.empty:
        raise RuntimeError(
            f"No DatasetEvents.Digital rows found for {subject} {session}"
        )
    return rows


def infer_canonical_stream(rows: pd.DataFrame) -> tuple[str, str]:
    candidates: list[tuple[str, str]] = []
    for stream_name in ("obx", "nidq"):
        mask = (rows["stream_name"] == stream_name) & (rows["event_name"] == "io2")
        if mask.any():
            dataset_name = rows.loc[mask, "dataset_name"].iloc[0]
            candidates.append((str(dataset_name), stream_name))
    if not candidates:
        available = sorted(
            {
                f"{row.dataset_name}:{row.stream_name}:{row.event_name}"
                for row in rows[
                    ["dataset_name", "stream_name", "event_name"]
                ].itertuples(index=False)
            }
        )
        raise RuntimeError(
            "Could not infer a canonical event stream from io2 rows. "
            f"Available rows: {available}"
        )
    return candidates[0]


def load_grb006_visual_onsets() -> np.ndarray:
    relation = DatasetEvents.Digital() & {**GRB006_KEY, "event_name": "ai0"}
    if not len(relation):
        raise RuntimeError("GRB006 ai0 visual-stim row is missing from the DB")
    row = relation.fetch1()
    visual_onsets = np.asarray(row["event_timestamps"], dtype=float)
    visual_onsets = visual_onsets[np.isfinite(visual_onsets)]
    if visual_onsets.size == 0:
        raise RuntimeError("No finite GRB006 visual stim timestamps found in the DB")
    return np.sort(visual_onsets)


def normalize_event_row(row: dict) -> dict:
    return {
        **row,
        "event_timestamps": np.asarray(row["event_timestamps"], dtype=float),
        "event_values": None
        if row.get("event_values") is None
        else np.asarray(row["event_values"]),
    }


def event_count(row: dict) -> int:
    timestamps = row.get("event_timestamps")
    if timestamps is None:
        return 0
    return len(np.asarray(timestamps))


def build_grb006_ai0_row() -> dict:
    visual_onsets = load_grb006_visual_onsets()
    return {
        **GRB006_KEY,
        "event_name": "ai0",
        "event_timestamps": visual_onsets,
        "event_values": np.ones(visual_onsets.shape[0], dtype=np.uint8),
    }


def insert_grb006_ai0_if_missing(
    apply: bool,
    replace_existing_ai0: bool = False,
) -> dict:
    row = build_grb006_ai0_row()
    key = {**GRB006_KEY, "event_name": row["event_name"]}
    relation = DatasetEvents.Digital() & key
    print(f"DB GRB006 ai0 source contains {event_count(row)} visual-stim timestamps")
    if len(relation):
        existing_row = normalize_event_row((relation).fetch1())
        existing_count = event_count(existing_row)
        if not replace_existing_ai0:
            print(
                f"{'Keeping' if apply else 'Would keep'} existing "
                f"{row['dataset_name']}:{row['stream_name']}:{row['event_name']} "
                f"with {existing_count} timestamps "
                f"(local source has {event_count(row)})."
            )
            return existing_row

        print(
            f"{'Replacing' if apply else 'Would replace'} existing "
            f"{row['dataset_name']}:{row['stream_name']}:{row['event_name']} "
            f"with {event_count(row)} timestamps "
            f"(existing row has {existing_count})."
        )
        if apply:
            relation.delete(force=True)
            DatasetEvents.Digital().insert1(row, allow_direct_insert=True)
        return normalize_event_row(row)

    if apply:
        DatasetEvents.Digital().insert1(row, allow_direct_insert=True)
    print(
        f"{'Inserted' if apply else 'Would insert'} "
        f"{row['dataset_name']}:{row['stream_name']}:{row['event_name']} "
        f"with {event_count(row)} timestamps."
    )
    return normalize_event_row(row)


def event_mapping_message(row: dict) -> str:
    return (
        f"EventMapping {row['subject_name']} {row['session_name']} "
        f"{row['event_name']} -> {row['source_stream_name']}:{row['source_event_name']}"
    )


def set_event_mapping_rows(rows: list[dict], apply: bool) -> None:
    for row in rows:
        key = {
            "subject_name": row["subject_name"],
            "session_name": row["session_name"],
            "event_name": row["event_name"],
        }
        relation = EventMapping() & key
        exists = len(relation) > 0
        if apply and not exists:
            EventMapping().insert1(row)
        if apply:
            action = "Kept existing" if exists else "Inserted"
        else:
            action = "Would set"
        print(f"{action} {event_mapping_message(row)}")


def build_event_mapping_rows(
    spec: dict,
    pending_rows: list[dict] | None = None,
) -> list[dict]:
    rows = fetch_session_rows(spec["subject_name"], spec["session_name"]).copy()
    if pending_rows:
        pending_df = pd.DataFrame([normalize_event_row(row) for row in pending_rows])
        rows = pd.concat([rows, pending_df], ignore_index=True)
        rows = rows.drop_duplicates(
            subset=["dataset_name", "stream_name", "event_name"],
            keep="last",
        ).reset_index(drop=True)
    if "stream_name" in spec:
        stream_name = spec["stream_name"]
        dataset_name = spec.get(
            "dataset_name",
            rows.loc[rows["stream_name"] == stream_name, "dataset_name"].iloc[0],
        )
    else:
        dataset_name, stream_name = infer_canonical_stream(rows)
        dataset_name = spec.get("dataset_name", dataset_name)

    mapped_rows: list[dict] = []
    for logical_name, source_event_name in spec["source_event_names"].items():
        mask = (
            (rows["dataset_name"] == dataset_name)
            & (rows["stream_name"] == stream_name)
            & (rows["event_name"] == source_event_name)
        )
        if not mask.any():
            available = sorted(
                {
                    f"{row.dataset_name}:{row.stream_name}:{row.event_name}"
                    for row in rows[
                        ["dataset_name", "stream_name", "event_name"]
                    ].itertuples(index=False)
                }
            )
            raise RuntimeError(
                f"Missing source row for {spec['subject_name']} {spec['session_name']}: "
                f"{dataset_name}:{stream_name}:{source_event_name}. "
                f"Available rows: {available}"
            )
        mapped_rows.append(
            {
                "subject_name": spec["subject_name"],
                "session_name": spec["session_name"],
                "event_name": logical_name,
                "source_dataset_name": dataset_name,
                "source_stream_name": stream_name,
                "source_event_name": source_event_name,
            }
        )
    return mapped_rows


def seed_event_mapping(
    apply: bool,
    replace_existing_ai0: bool = False,
) -> None:
    grb006_ai0_row = insert_grb006_ai0_if_missing(
        apply=apply,
        replace_existing_ai0=replace_existing_ai0,
    )
    for spec in SESSION_MAPPING_SPECS:
        pending_rows = (
            [grb006_ai0_row]
            if (not apply and spec["subject_name"] == "GRB006")
            else None
        )
        set_event_mapping_rows(
            build_event_mapping_rows(spec, pending_rows=pending_rows),
            apply=apply,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seed EventMapping rows and the GRB006 ai0 visual-stim row."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write changes to DataJoint. Without this flag the script only prints the planned actions.",
    )
    parser.add_argument(
        "--replace-existing-ai0",
        action="store_true",
        help="Replace the existing GRB006 nidq:ai0 row if present. Requires --apply to write.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_event_mapping(
        apply=args.apply,
        replace_existing_ai0=args.replace_existing_ai0,
    )


if __name__ == "__main__":
    main()
