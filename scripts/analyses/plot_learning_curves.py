from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-set-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    import pandas as pd

    from labdata.schema import DecisionTask
    from labdata_plugin.analysisschema import BehaviorAnalysisSet

    selected = BehaviorAnalysisSet.TrialSet() & {
        "analysis_set_id": args.analysis_set_id
    }
    rows = (DecisionTask.TrialSet() & selected).fetch(as_dict=True)
    if not rows:
        raise RuntimeError(
            f"No selected TrialSets for analysis_set_id={args.analysis_set_id}"
        )

    data = pd.DataFrame(rows).sort_values(["subject_name", "session_name"])
    fig, ax = plt.subplots(figsize=(8, 4))
    for subject, subject_df in data.groupby("subject_name"):
        ax.plot(subject_df["performance_easy"].to_numpy(), marker="o", label=subject)
    ax.set_xlabel("Session index")
    ax.set_ylabel("Easy performance")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Easy performance across selected sessions")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")


if __name__ == "__main__":
    main()
