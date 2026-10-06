"""Prepare one ingested, spike-sorted ephys session for thesis analyses.

Validate/insert clock sync first, then populate unit metrics, the project's
quality criteria, unit stability, and stimulus responses. Existing results are
kept. Audio recovery from raw OneBox files is a separate optional workflow.
"""

import argparse
import warnings

from thesis.ephys.preprocessing.insert_streamsync import insert_streamsync
from thesis.ephys.preprocessing.populate_unit_counts import CRITERIA

warnings.filterwarnings("ignore", category=UserWarning, module="datajoint.plugin")


def populate_session(subject: str, session: str, apply: bool = False) -> None:
    from labdata.schema import (
        EphysRecording,
        SpikeSorting,
        StreamSync,
        UnitCount,
        UnitCountCriteria,
        UnitMetrics,
    )

    from labdata_plugin.schema import (
        StimulusResponse,
        StimulusResponseParam,
        UnitStability,
        UnitStabilityParam,
    )

    session_key = {"subject_name": subject, "session_name": session}
    sortings = (SpikeSorting & session_key).fetch("KEY")
    if len(sortings) != 1:
        raise ValueError(
            f"Expected one spike sorting for {session_key}, found {len(sortings)}"
        )
    dataset_key = {**session_key, "dataset_name": sortings[0]["dataset_name"]}
    if len(EphysRecording & dataset_key) != 1:
        raise ValueError(f"No ephys recording for {dataset_key}")
    n_units = len(SpikeSorting.Unit & session_key)
    if not n_units:
        raise ValueError(f"No sorted units for {session_key}")

    insert_streamsync(dataset_key, apply)
    if not apply:
        print(f"Sorted units: {n_units}; metrics: {len(UnitMetrics & session_key)}")
        for criteria in CRITERIA:
            key = {**session_key, "unit_criteria_id": criteria["unit_criteria_id"]}
            print(
                f"Criteria {criteria['unit_criteria_id']}: "
                f"counts={len(UnitCount & key)}, "
                f"stability={len(UnitStability & {**key, 'stability_param_id': 0})}, "
                f"responses={len(StimulusResponse & {**key, 'stim_response_param_id': 0})}"
            )
        print("Dry run; add --apply to insert missing results")
        return

    probe_numbers = (EphysRecording.ProbeSetting & dataset_key).fetch("probe_num")
    missing_sync = [
        f"imec{probe}"
        for probe in probe_numbers
        if not len(StreamSync & {**dataset_key, "stream_name": f"imec{probe}"})
    ]
    bpod_key = {**session_key, "dataset_name": "chipmunk", "stream_name": "bpod"}
    if not len(StreamSync & bpod_key):
        missing_sync.append("bpod")
    if missing_sync:
        raise RuntimeError(
            f"Missing StreamSync mappings for {session_key}: {missing_sync}"
        )

    UnitMetrics.populate(session_key)
    if len(UnitMetrics & session_key) != n_units:
        raise RuntimeError(f"UnitMetrics is incomplete for {session_key}")
    print(f"UnitMetrics: {n_units}/{n_units}")

    UnitStabilityParam().insert(UnitStabilityParam.contents, skip_duplicates=True)
    StimulusResponseParam().insert(StimulusResponseParam.contents, skip_duplicates=True)
    for criteria in CRITERIA:
        criteria_key = {"unit_criteria_id": criteria["unit_criteria_id"]}
        existing = (UnitCountCriteria & criteria_key).fetch(as_dict=True)
        if existing and any(
            existing[0][field] != criteria[field]
            for field in ("sua_criteria", "mua_criteria")
        ):
            raise RuntimeError(
                f"Stored quality criteria differ from project criteria: {criteria_key}"
            )
        if not existing:
            UnitCountCriteria.insert1(criteria)

        key = {**session_key, **criteria_key}
        UnitCount.populate(key)
        if len(UnitCount & key) != 1 or len(UnitCount.Unit & key) != n_units:
            raise RuntimeError(f"UnitCount is incomplete for {key}")
        n_passing = len(UnitCount.Unit & key & {"passes": 1})
        print(
            f"UnitCount criteria {criteria['unit_criteria_id']}: "
            f"{n_passing}/{n_units} passing"
        )
        if not n_passing:
            print("No passing units; skipping stability and stimulus response")
            continue

        stability_key = {**key, "stability_param_id": 0}
        UnitStability.populate(stability_key)
        n_stable_rows = len(UnitStability.Unit & stability_key)
        if len(UnitStability & stability_key) != 1 or n_stable_rows != n_passing:
            raise RuntimeError(f"UnitStability is incomplete for {stability_key}")
        print(f"UnitStability: {n_stable_rows}/{n_passing}")

        response_key = {**key, "stim_response_param_id": 0}
        StimulusResponse.populate(response_key)
        n_response_rows = len(StimulusResponse.Unit & response_key)
        if len(StimulusResponse & response_key) != 1 or n_response_rows != n_passing:
            raise RuntimeError(f"StimulusResponse is incomplete for {response_key}")
        excited = len(
            StimulusResponse.Unit & response_key & {"response_type": "excited"}
        )
        print(f"StimulusResponse: {n_response_rows}/{n_passing}; {excited} excited")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Populate a sorted ephys session for thesis analyses"
    )
    parser.add_argument("subject")
    parser.add_argument("session")
    parser.add_argument(
        "--apply", action="store_true", help="Insert missing mappings and results"
    )
    args = parser.parse_args()
    populate_session(args.subject, args.session, args.apply)


if __name__ == "__main__":
    main()
