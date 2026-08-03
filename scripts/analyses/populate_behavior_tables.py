from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-set-id", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from labdata_plugin.analysisschema import (
        BehaviorAnalysisSet,
        PsychometricSessionFit,
        PsychometricSubjectFit,
        PsychophysicalKernel,
    )

    restriction = {"analysis_set_id": args.analysis_set_id}
    selected_trialsets = BehaviorAnalysisSet.TrialSet() & restriction
    table_restrictions = [
        (PsychometricSessionFit, selected_trialsets),
        (PsychometricSubjectFit, restriction),
        (PsychophysicalKernel, restriction),
    ]
    for table, table_restriction in table_restrictions:
        pending = (table.key_source & table_restriction) - table()
        print(f"{table.__name__}: {len(pending)} pending")
        if not args.dry_run:
            table.populate(table_restriction, display_progress=True)


if __name__ == "__main__":
    main()
