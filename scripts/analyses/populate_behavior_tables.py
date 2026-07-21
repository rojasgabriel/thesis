from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-set-id", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from labdata_plugin.analysisschema import (
        LearningSessionMetrics,
        PsychometricSessionFit,
        PsychometricSubjectFit,
        PsychophysicalKernel,
    )

    restriction = {"session_set_id": args.session_set_id}
    tables = [
        LearningSessionMetrics,
        PsychometricSessionFit,
        PsychometricSubjectFit,
        PsychophysicalKernel,
    ]
    for table in tables:
        pending = (table.key_source & restriction) - table()
        print(f"{table.__name__}: {len(pending)} pending")
        if not args.dry_run:
            table.populate(restriction, display_progress=True)


if __name__ == "__main__":
    main()
