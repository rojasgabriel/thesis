from __future__ import annotations

import argparse
from collections import Counter

import _bootstrap  # noqa: F401


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-set-id", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--description", default="")
    parser.add_argument("--subjects", nargs="+", required=True)
    parser.add_argument("--trialset", default="visual")
    parser.add_argument("--performance-threshold", type=float, default=0.7)
    parser.add_argument("--min-trials-with-choice", type=int, default=200)
    parser.add_argument("--kernel-timebins", type=int, default=10)
    parser.add_argument("--kernel-cv-splits", type=int, default=10)
    parser.add_argument("--kernel-random-state", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from labdata.schema import DecisionTask
    from labdata_plugin.analysisschema import BehaviorSessionSet

    relation = DecisionTask.TrialSet() & {
        "trialset_description": args.trialset,
    }
    relation = relation & [{"subject_name": subject} for subject in args.subjects]
    relation = relation & f"n_with_choice >= {args.min_trials_with_choice}"
    relation = relation & f"performance_easy >= {args.performance_threshold}"
    trialset_rows = relation.fetch("KEY")

    session_rows = _unique_session_rows(trialset_rows, args.session_set_id)
    subject_trialset_rows = [
        {
            "session_set_id": args.session_set_id,
            "subject_name": subject,
            "trialset_description": trialset,
            "n_sessions": count,
        }
        for (subject, trialset), count in Counter(
            (row["subject_name"], row["trialset_description"])
            for row in trialset_rows
        ).items()
    ]
    session_set = {
        "session_set_id": args.session_set_id,
        "session_set_name": args.name,
        "session_set_description": args.description,
        "performance_threshold": args.performance_threshold,
        "min_trials_with_choice": args.min_trials_with_choice,
        "kernel_timebins": args.kernel_timebins,
        "kernel_cv_splits": args.kernel_cv_splits,
        "kernel_random_state": args.kernel_random_state,
        "analysis_version": "v1",
    }

    print(f"Session set: {session_set}")
    print(f"TrialSets: {len(trialset_rows)}")
    print(f"Sessions: {len(session_rows)}")
    print(f"Subject/trialsets: {len(subject_trialset_rows)}")
    if args.dry_run:
        return

    BehaviorSessionSet.insert1(session_set, skip_duplicates=True)
    BehaviorSessionSet.Session.insert(
        session_rows, skip_duplicates=True, ignore_extra_fields=True
    )
    BehaviorSessionSet.TrialSet.insert(
        [
            {
                **row,
                "session_set_id": args.session_set_id,
                "include_reason": "seeded_from_decision_task",
            }
            for row in trialset_rows
        ],
        skip_duplicates=True,
        ignore_extra_fields=True,
    )
    BehaviorSessionSet.SubjectTrialSet.insert(
        subject_trialset_rows, skip_duplicates=True, ignore_extra_fields=True
    )


def _unique_session_rows(trialset_rows, session_set_id):
    keys = {}
    for row in trialset_rows:
        key = (row["subject_name"], row["session_name"])
        keys[key] = {
            "session_set_id": session_set_id,
            "subject_name": row["subject_name"],
            "session_name": row["session_name"],
            "include_reason": "seeded_from_decision_task",
        }
    return list(keys.values())


if __name__ == "__main__":
    main()
