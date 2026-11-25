"""Prepare one new session for thesis analyses with one command.

    uv run preprocess-session GRB068 20261123_161832          # dry run: status only
    uv run preprocess-session GRB068 20261123_161832 --apply  # run missing steps
    uv run preprocess-session GRB068 20261123_161832 --apply --sbatch  # as a Slurm job

Run it on Hodgkin after the rig uploads (`labdata put ... -r ephys` from
Jupiter, behavior `labdata put` from the rig) are done. Each step checks what
is already in the database and does only what is missing, so it is safe to
rerun. The dry run writes nothing and submits nothing. With `--apply`, the
first step that cannot finish stops the run.

Steps, in order:

1. Behavior: populate `Chipmunk` (task) and then `ChipmunkPassive` (replay).
2. Upload: the session has an `EphysRecording`. Steps that read raw files
   (behavior, audio, sort) first check that the files are on this machine;
   nothing is downloaded from S3 (copy missing files from the rig).
3. Audio, only with `--audio`: recover the OneBox XA1 audio events as
   `obx:io1` (`insert_audio_events`). Optional, because analyses use Bpod
   sound times without it, and inserting it changes the audio times of a
   session that was analyzed without it (GRB064 20261008 has none). A failed
   validation prints a warning and the run continues.
4. Sort: when the session has no spike sorting and no spks compute task,
   submit `labdata run spks` (Slurm, default partition `normal`) with the
   extra arguments of the subject in `SORT_ARGS`, and submit the rest of this
   command as a Slurm job that starts after the sort job
   (`--dependency=afterok`). labdata pipes the sort output to its log, so
   `afterok` also fires after a failed sort; the follow-up job then finds the
   task without a sorting and stops. A task that exists without a sorting
   (running or failed) is reported, never resubmitted.
5. Clock sync: validated `StreamSync` mappings for every probe and Bpod
   (`insert_streamsync`).
6. Unit tables for the newest sorting: `UnitMetrics`, `UnitCount` for each
   criteria set in `CRITERIA`, then `UnitStability` (parameter set 0),
   `StimulusResponse` (parameter set 0) and `MovementArtifact` (parameter set
   0) for each criteria set with passing units.
7. Camera frames: a read-only check that the camera pulses align to the
   saved frames (`audit_camera_pulses.align_video_frames`).

Not done here: GLM prepare and fits (`GLMFit` runs prepare itself, and a
session enters `ThesisSession` only when Gabriel chooses), and unit layers and
pose tracking (not on main yet).
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

from thesis.ephys.preprocessing._database import (
    initialize_tables,
    read_only_database_imports,
)
from thesis.ephys.preprocessing.insert_audio_events import (
    DEFAULT_THRESHOLD,
    insert_audio_events,
)
from thesis.ephys.preprocessing.insert_audio_events import (
    _self_check as _audio_self_check,
)
from thesis.ephys.preprocessing.insert_streamsync import (
    _self_check as _sync_self_check,
)
from thesis.ephys.preprocessing.insert_streamsync import insert_streamsync

# Criteria as stored in the lab database (Joao Couto, 2026-10-24, labdata 0.1.25):
# depth drift is an absolute distance, and noise units get passes = -1.
MUA_CRITERIA = "span_um <= 350 & spike_duration >= 0.05"
CRITERIA = [
    {
        "unit_criteria_id": 0,
        "sua_criteria": "isi_contamination < 0.1 & amplitude_cutoff < 0.1 & spike_duration > 0.1 & spike_amplitude > 50 & presence_ratio > 0.6 & firing_rate > 1",
        "mua_criteria": MUA_CRITERIA,
    },
    {
        "unit_criteria_id": 1,
        "sua_criteria": "isi_contamination < 0.1 & amplitude_cutoff < 0.1 & spike_duration > 0.1 & spike_amplitude > 50 & presence_ratio > 0.6 & firing_rate > 1 & abs(depth_drift_start_to_end) < 6",
        "mua_criteria": MUA_CRITERIA,
    },
    {
        "unit_criteria_id": 2,
        "sua_criteria": "isi_contamination < 0.1 & amplitude_cutoff < 0.01 & spike_duration > 0.1 & spike_amplitude > 50 & presence_ratio > 0.9 & firing_rate > 2 & abs(depth_drift_start_to_end) < 6",
        "mua_criteria": MUA_CRITERIA,
    },
]

# Extra spks arguments per subject. GRB068: channels without signal make junk clusters.
SORT_ARGS = {"GRB068": ["--remove-channels-without-signal", "amplitude"]}


class Blocked(Exception):
    """A step cannot finish; with --apply the run stops here."""


def _remote_files(files) -> list[str]:
    """Return the files of a File query that are not on this machine."""
    return list(files.check_if_files_local()[1]) if len(files) else []


def _slurm_job_ids(output: str) -> list[str]:
    """Return the Slurm job ids in labdata's colored "... to slurm [<id>]" lines."""
    return re.findall(r"to slurm \[(\d+)\]", re.sub(r"\x1b\[[\d;]*m", "", output))


def _step(name: str, status: str) -> None:
    print(f"[{name}] {status}", flush=True)


def submit_job(
    subject: str,
    session: str,
    audio: bool = False,
    threshold: float = DEFAULT_THRESHOLD,
    after: int | None = None,
) -> int:
    """Submit `--apply` for one session as a Slurm job (partition normal)."""
    import os
    import shlex

    from labdata.compute.schedulers import LABDATA_LOG_FOLDER, slurm_submit

    exports = " ".join(
        f"export {name}={shlex.quote(os.environ[name])} &&"
        for name in ("LABDATA_PATH", "PYTHONPATH")
        if name in os.environ
    )
    args = [subject, session, "--apply", "--audio-threshold", str(threshold)]
    if audio:
        args.append("--audio")
    command = (
        f"{exports} cd {shlex.quote(str(Path(__file__).resolve().parents[4]))} && "
        f"{shlex.quote(sys.executable)} -m thesis.ephys.preprocessing.preprocess_session "
        + " ".join(shlex.quote(arg) for arg in args)
    )
    job_id = slurm_submit(
        f"preprocess_{subject}_{session}",
        command,
        ncpuspertask=4,
        partition="normal",
        walltime="12:00:00",
        sbatch_append=(
            f"--dependency=afterok:{after} --kill-on-invalid-dep=yes" if after else ""
        ),
    )
    if job_id is None:
        raise RuntimeError("Slurm did not accept the preprocess job")
    print(f"Submitted Slurm job {job_id}; log in {LABDATA_LOG_FOLDER}", flush=True)
    return int(job_id)


def preprocess_session(
    subject: str,
    session: str,
    apply: bool = False,
    audio: bool = False,
    threshold: float = DEFAULT_THRESHOLD,
) -> None:
    with read_only_database_imports():
        from labdata.schema import (
            ComputeTask,
            Dataset,
            DatasetEvents,
            EphysRecording,
            File,
            SpikeSorting,
            StreamSync,
            UnitCount,
            UnitCountCriteria,
            UnitMetrics,
        )
        from chipmunk import Chipmunk, ChipmunkPassive

        from labdata_plugin.schema import (
            MovementArtifact,
            StimulusResponse,
            UnitStability,
            initialize_schema,
        )

    key = {"subject_name": subject, "session_name": session}

    def blocked(name: str, status: str) -> None:
        _step(name, status)
        if apply:
            raise Blocked(status)

    def files_of(datasets):
        return File() & (Dataset.DataFiles() & datasets)

    # 1. Behavior
    for table, dataset_name in (
        (Chipmunk, "chipmunk"),
        (ChipmunkPassive, "chipmunk_passive"),
    ):
        datasets = Dataset & key & {"dataset_name": dataset_name}
        missing = datasets - table
        name = f"behavior {table.__name__}"
        if not len(datasets):
            _step(name, "no dataset")
        elif not len(missing):
            _step(name, "done")
        elif _remote_files(files_of(missing)):
            blocked(name, "files not local; not downloading")
        elif not apply:
            _step(name, "to populate")
        else:
            table.populate(missing.proj())
            if len(missing - table):
                raise RuntimeError(f"{table.__name__} did not populate {key}")
            _step(name, "populated")

    # 2. Upload
    ephys = (EphysRecording & key).fetch("KEY")
    if len(ephys) != 1:
        blocked(
            "upload",
            f"expected one EphysRecording, found {len(ephys)}; wait for the upload",
        )
        return
    ephys_key = ephys[0]
    _step("upload", f"done ({ephys_key['dataset_name']})")

    # 3. Audio
    obx_files = files_of(Dataset & ephys_key) & 'file_path LIKE "%.obx.%"'
    io1 = (
        DatasetEvents.Digital & ephys_key & {"stream_name": "obx", "event_name": "io1"}
    )
    if not len(obx_files):
        _step("audio", "no OneBox file; Bpod sound times")
    elif len(io1):
        _step("audio", "done")
    elif not audio:
        _step("audio", "no obx:io1; Bpod sound times (add --audio to recover it)")
    elif _remote_files(obx_files):
        blocked("audio", "OneBox files not local; not downloading")
    else:
        try:
            insert_audio_events(ephys_key, apply, threshold)
            _step("audio", "inserted obx:io1" if apply else "to insert")
        except ValueError as error:
            _step(
                "audio", f"WARNING: not inserted ({error}); Bpod sound times are used"
            )

    # 4. Sort
    if not len(SpikeSorting & key):
        tasks = (ComputeTask & ephys_key & {"task_name": "spks"}).fetch(
            "job_id", "task_status", as_dict=True
        )
        if tasks:
            blocked("sort", f"no sorting yet; spks task(s) {tasks}; not resubmitting")
            return
        command = [
            str(Path(sys.executable).parent / "labdata"),
            *("run", "spks", "-a", subject, "-s", session),
        ]
        if subject in SORT_ARGS:
            command += ["--", *SORT_ARGS[subject]]
        remote = _remote_files(files_of(Dataset & ephys_key))
        if remote:
            blocked("sort", f"{len(remote)} ephys files not local, e.g. {remote[0]}")
            return
        if not apply:
            _step("sort", f"to submit: {' '.join(command[1:])}")
            return
        output = subprocess.run(
            command, check=True, stdout=subprocess.PIPE, text=True
        ).stdout
        print(output, end="")
        slurm_ids = _slurm_job_ids(output)
        if len(slurm_ids) != 1:
            _step("sort", "submitted; rerun this command after the sort finishes")
            return
        slurm_id = slurm_ids[0]
        submit_job(subject, session, audio, threshold, after=int(slurm_id))
        _step("sort", f"submitted as Slurm job {slurm_id}; the rest runs after it")
        return
    from thesis.ephys.units import sorting_key

    session_key = sorting_key(subject, session)
    _step("sort", f"done (parameter set {session_key['parameter_set_num']})")
    n_units = len(SpikeSorting.Unit & session_key)
    if not n_units:
        raise Blocked(f"No sorted units for {session_key}")

    # 5. Clock sync
    dataset_key = (SpikeSorting & session_key).fetch1("KEY")
    dataset_key = {
        k: dataset_key[k] for k in ("subject_name", "session_name", "dataset_name")
    }
    if apply:
        initialize_tables(
            StreamSync, UnitMetrics, UnitCountCriteria, UnitCount, UnitCount.Unit
        )
        initialize_schema()
    validated = insert_streamsync(dataset_key, apply)
    probes = (EphysRecording.ProbeSetting & dataset_key).fetch("probe_num")
    sync_keys = [{**dataset_key, "stream_name": f"imec{p}"} for p in probes]
    sync_keys.append({**key, "dataset_name": "chipmunk", "stream_name": "bpod"})
    missing_sync = [
        k["stream_name"]
        for k in sync_keys
        if not (StreamSync().is_declared and len(StreamSync & k))
    ]
    if missing_sync and not apply and validated == len(missing_sync):
        _step("sync", f"to insert {missing_sync}")
        return
    if missing_sync:
        blocked("sync", f"missing {missing_sync}; not validated")
        if not apply:
            return
    else:
        _step("sync", "done")

    # 6. Unit tables
    def count(table, restriction):
        return len(table & restriction) if table.database and table().is_declared else 0

    if not apply:
        _step("units", f"{n_units} sorted; metrics {count(UnitMetrics, session_key)}")
        for criteria in CRITERIA:
            k = {**session_key, "unit_criteria_id": criteria["unit_criteria_id"]}
            _step(
                f"criteria {criteria['unit_criteria_id']}",
                f"counts {count(UnitCount, k)}, "
                f"stability {count(UnitStability, {**k, 'stability_param_id': 0})}, "
                f"responses {count(StimulusResponse, {**k, 'stim_response_param_id': 0})}, "
                f"artifacts {count(MovementArtifact, {**k, 'movement_param_id': 0})}",
            )
    else:
        UnitMetrics.populate(session_key)
        if len(UnitMetrics & session_key) != n_units:
            raise RuntimeError(f"UnitMetrics is incomplete for {session_key}")
        _step("units", f"metrics {n_units}/{n_units}")
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

            k = {**session_key, **criteria_key}
            name = f"criteria {criteria['unit_criteria_id']}"
            UnitCount.populate(k)
            if len(UnitCount & k) != 1 or len(UnitCount.Unit & k) != n_units:
                raise RuntimeError(f"UnitCount is incomplete for {k}")
            n_passing = len(UnitCount.Unit & k & {"passes": 1})
            if not n_passing:
                _step(name, f"0/{n_units} passing; no stability or responses")
                continue

            stability_key = {**k, "stability_param_id": 0}
            UnitStability.populate(stability_key)
            n_stable = len(UnitStability.Unit & stability_key)
            if len(UnitStability & stability_key) != 1 or n_stable != n_passing:
                raise RuntimeError(f"UnitStability is incomplete for {stability_key}")

            response_key = {**k, "stim_response_param_id": 0}
            StimulusResponse.populate(response_key)
            n_responses = len(StimulusResponse.Unit & response_key)
            if len(StimulusResponse & response_key) != 1 or n_responses != n_passing:
                raise RuntimeError(f"StimulusResponse is incomplete for {response_key}")
            excited = len(
                StimulusResponse.Unit & response_key & {"response_type": "excited"}
            )

            artifact_key = {**stability_key, "movement_param_id": 0}
            MovementArtifact.populate(artifact_key)
            if len(MovementArtifact & artifact_key) != 1:
                raise RuntimeError(f"MovementArtifact is incomplete for {artifact_key}")
            n_artifacts = len(MovementArtifact.Unit & artifact_key & {"artifact": 1})
            _step(
                name,
                f"{n_passing}/{n_units} passing, {excited} excited, "
                f"{n_artifacts} movement artifacts",
            )

    # 7. Camera frames
    from thesis.ephys.preprocessing.audit_camera_pulses import align_video_frames

    try:
        _, audit, _ = align_video_frames(subject, session)
        _step(
            "camera",
            f"{audit['frame_count']} frames aligned; repairs {audit['repairs'] or 'none'}",
        )
    except Exception as error:  # report only: analyses that need frames raise again
        _step("camera", f"WARNING: frames not aligned ({error})")
    if not apply:
        print("Dry run; add --apply to run the missing steps")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("subject")
    parser.add_argument("session")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Run the missing steps (writes, submits the sort)",
    )
    parser.add_argument(
        "--sbatch", action="store_true", help="With --apply: run as a Slurm job"
    )
    parser.add_argument(
        "--audio", action="store_true", help="Also recover obx:io1 audio events"
    )
    parser.add_argument(
        "--audio-threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="XA1 10 ms peak-to-peak threshold (default: %(default)s)",
    )
    args = parser.parse_args()
    if args.audio_threshold <= 0:
        parser.error("--audio-threshold must be positive")
    if args.sbatch:
        if not args.apply:
            parser.error("--sbatch needs --apply")
        submit_job(args.subject, args.session, args.audio, args.audio_threshold)
        return
    _audio_self_check()
    _sync_self_check()
    assert _slurm_job_ids("to slurm [\x1b[33m8250\x1b[0m]\n") == ["8250"]
    try:
        preprocess_session(
            args.subject, args.session, args.apply, args.audio, args.audio_threshold
        )
    except Blocked as error:
        parser.exit(1, f"Stopped: {error}\n")


if __name__ == "__main__":
    main()
