from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-set-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    import numpy as np

    from behavior_analyses.psychometrics import cumulative_gaussian
    from labdata_plugin.analysisschema import PsychometricSubjectFit

    rows = (PsychometricSubjectFit() & {"session_set_id": args.session_set_id}).fetch(
        as_dict=True
    )
    if not rows:
        raise RuntimeError(
            f"No psychometric rows for session_set_id={args.session_set_id}"
        )

    fig, ax = plt.subplots(figsize=(5, 5))
    for row in rows:
        stims = np.asarray(row["stims"], dtype=float)
        params = np.asarray(row["fit_params"], dtype=float)
        p_right = np.asarray(row["p_right"], dtype=float)
        x = np.asarray(sorted(stims), dtype=float)
        ax.plot(x, cumulative_gaussian(*params, x), label=row["subject_name"])
        ax.plot(stims, p_right, "o", ms=4)
    ax.set_xlabel("Stimulus rate relative to boundary (Hz)")
    ax.set_ylabel("P(right choice)")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")


if __name__ == "__main__":
    main()
