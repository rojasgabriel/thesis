from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-set-id", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--description", default="")
    parser.add_argument("--subjects", nargs="+", required=True)
    parser.add_argument("--trialset", default="visual")
    parser.add_argument("--performance-threshold", type=float, default=0.7)
    parser.add_argument("--min-trials-with-choice", type=int, default=200)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from labdata.schema import DecisionTask
    from labdata_plugin.analysisschema import BehaviorAnalysisSet

    relation = DecisionTask.TrialSet() & {
        "trialset_description": args.trialset,
    }
    relation = relation & [{"subject_name": subject} for subject in args.subjects]
    relation = relation & f"n_with_choice >= {args.min_trials_with_choice}"
    relation = relation & f"performance_easy >= {args.performance_threshold}"
    trialset_rows = relation.fetch("KEY")

    analysis_set = {
        "analysis_set_id": args.analysis_set_id,
        "analysis_set_name": args.name,
        "analysis_set_description": args.description,
        "performance_threshold": args.performance_threshold,
        "min_trials_with_choice": args.min_trials_with_choice,
        "selection_version": "v1",
    }

    print(f"Analysis set: {analysis_set}")
    print(f"TrialSets: {len(trialset_rows)}")
    if args.dry_run:
        return

    BehaviorAnalysisSet.insert1(analysis_set, skip_duplicates=True)
    BehaviorAnalysisSet.TrialSet.insert(
        [
            {
                **row,
                "analysis_set_id": args.analysis_set_id,
                "include_reason": "seeded_from_decision_task",
            }
            for row in trialset_rows
        ],
        skip_duplicates=True,
        ignore_extra_fields=True,
    )


if __name__ == "__main__":
    main()
