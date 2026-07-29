from __future__ import annotations

import argparse

import numpy as np

import _bootstrap  # noqa: F401


ARCHIVE_TABLES = {
    "behavior_session_set": "archive_lab_tasks_479_behavior_session_set",
    "behavior_session_set__session": (
        "archive_lab_tasks_479_behavior_session_set__session"
    ),
    "behavior_session_set__trial_set": (
        "archive_lab_tasks_479_behavior_session_set__trial_set"
    ),
    "behavior_session_set__subject_trial_set": (
        "archive_lab_tasks_479_behavior_session_set__subject_trial_set"
    ),
    "__learning_session_metrics": "archive_lab_tasks_479_learning_session_metrics",
    "__psychometric_session_fit": ("archive_lab_tasks_479_psychometric_session_fit"),
    "__psychometric_subject_fit": ("archive_lab_tasks_479_psychometric_subject_fit"),
    "__psychophysical_kernel": "archive_lab_tasks_479_psychophysical_kernel",
}
TRIALSET_KEY_FIELDS = (
    "subject_name",
    "session_name",
    "dataset_name",
    "trialset_description",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Archive the old tables, activate the locked schema, and copy rows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import datajoint as dj
    import labdata.schema as labdata_schema

    connection = dj.conn()
    database = f"{labdata_schema.dbase_name}_user"
    _validate_table_state(connection, database)
    _print_source_counts(connection, database)
    if not args.apply:
        print("Dry run only. Re-run with --apply after exact live-write approval.")
        return

    rename_sql = "RENAME TABLE " + ", ".join(
        f"`{database}`.`{source}` TO `{database}`.`{archive}`"
        for source, archive in ARCHIVE_TABLES.items()
    )
    connection.query(rename_sql)

    from labdata_plugin.analysisschema import (
        BehaviorAnalysisSet,
        PsychometricFitConfig,
        PsychometricSessionFit,
        PsychometricSubjectFit,
        PsychophysicalKernel,
        PsychophysicalKernelFitConfig,
    )

    PsychometricFitConfig.insert1(("v1", 100, 6, "v1"), skip_duplicates=True)
    PsychophysicalKernelFitConfig.insert1(
        ("v1_10bin_10fold", 10, 10, 0, 20.0, 1.0, "v1"),
        skip_duplicates=True,
    )

    old_master = _archive_table(connection, database, "behavior_session_set")
    BehaviorAnalysisSet.insert(
        [
            {
                "analysis_set_id": row["session_set_id"],
                "analysis_set_name": row["session_set_name"],
                "analysis_set_description": row["session_set_description"],
                "performance_threshold": row["performance_threshold"],
                "min_trials_with_choice": row["min_trials_with_choice"],
                "selection_version": row["analysis_version"],
            }
            for row in old_master.fetch(as_dict=True)
        ],
        skip_duplicates=True,
    )

    old_trialsets = _archive_table(
        connection, database, "behavior_session_set__trial_set"
    )
    BehaviorAnalysisSet.TrialSet.insert(
        [
            {
                "analysis_set_id": row["session_set_id"],
                **{field: row[field] for field in TRIALSET_KEY_FIELDS},
                "include_reason": row["include_reason"],
            }
            for row in old_trialsets.fetch(as_dict=True)
        ],
        skip_duplicates=True,
    )

    old_session_fits = _archive_table(
        connection, database, "__psychometric_session_fit"
    )
    PsychometricSessionFit.insert(
        [
            {
                **{field: row[field] for field in TRIALSET_KEY_FIELDS},
                "psychometric_fit_config_id": "v1",
                "fit_status": "fit",
                "n_choices_fit": int(np.sum(row["n_obs"])),
                **_psychometric_outputs(row),
            }
            for row in _deduplicate_by_fields(
                old_session_fits.fetch(as_dict=True), TRIALSET_KEY_FIELDS
            )
        ],
        skip_duplicates=True,
    )

    old_subject_fits = _archive_table(
        connection, database, "__psychometric_subject_fit"
    )
    PsychometricSubjectFit.insert(
        [
            {
                "analysis_set_id": row["session_set_id"],
                "subject_name": row["subject_name"],
                "trialset_description": row["trialset_description"],
                "psychometric_fit_config_id": "v1",
                "fit_status": "fit",
                "n_choices_fit": int(np.sum(row["n_obs"])),
                **_psychometric_outputs(row),
            }
            for row in old_subject_fits.fetch(as_dict=True)
        ],
        skip_duplicates=True,
    )

    old_kernels = _archive_table(connection, database, "__psychophysical_kernel")
    PsychophysicalKernel.insert(
        [
            {
                "analysis_set_id": row["session_set_id"],
                "subject_name": row["subject_name"],
                "trialset_description": row["trialset_description"],
                "kernel_fit_config_id": "v1_10bin_10fold",
                "fit_status": "fit",
                "n_trials_fit": row["n_trials"],
                **{
                    field: row[field]
                    for field in (
                        "weights",
                        "weights_mean",
                        "weights_error",
                        "scores",
                        "score_mean",
                        "bias",
                        "bias_mean",
                    )
                },
            }
            for row in old_kernels.fetch(as_dict=True)
        ],
        skip_duplicates=True,
    )

    print(f"BehaviorAnalysisSet: {len(BehaviorAnalysisSet())}")
    print(f"BehaviorAnalysisSet.TrialSet: {len(BehaviorAnalysisSet.TrialSet())}")
    print(f"PsychometricSessionFit copied: {len(PsychometricSessionFit())}")
    print(f"PsychometricSubjectFit copied: {len(PsychometricSubjectFit())}")
    print(f"PsychophysicalKernel copied: {len(PsychophysicalKernel())}")


def _validate_table_state(connection, database):
    existing = {
        row[0]
        for row in connection.query(
            "SELECT table_name FROM information_schema.tables "
            f"WHERE table_schema={database!r}"
        ).fetchall()
    }
    missing = set(ARCHIVE_TABLES) - existing
    occupied_archives = set(ARCHIVE_TABLES.values()) & existing
    if missing or occupied_archives:
        raise RuntimeError(
            f"Unsafe table state: missing={sorted(missing)}, "
            f"occupied_archives={sorted(occupied_archives)}"
        )


def _print_source_counts(connection, database):
    for source, archive in ARCHIVE_TABLES.items():
        count = connection.query(
            f"SELECT COUNT(*) FROM `{database}`.`{source}`"
        ).fetchone()[0]
        print(f"{source}: {count} rows -> {archive}")


def _archive_table(connection, database, source):
    import datajoint as dj

    return dj.FreeTable(connection, f"`{database}`.`{ARCHIVE_TABLES[source]}`")


def _deduplicate_by_fields(rows, fields):
    unique = {}
    for row in rows:
        unique.setdefault(tuple(row[field] for field in fields), row)
    return list(unique.values())


def _psychometric_outputs(row):
    return {
        field: row[field]
        for field in (
            "stims",
            "p_right",
            "p_right_ci",
            "n_right",
            "n_obs",
            "bias",
            "sensitivity",
            "guess_rate",
            "lapse_rate",
            "goodness_of_fit",
        )
    }


if __name__ == "__main__":
    main()
