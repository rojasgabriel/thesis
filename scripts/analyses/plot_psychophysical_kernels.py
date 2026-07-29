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
    import numpy as np

    from labdata_plugin.analysisschema import PsychophysicalKernel

    rows = (
        PsychophysicalKernel()
        & {"analysis_set_id": args.analysis_set_id, "fit_status": "fit"}
    ).fetch(as_dict=True)
    if not rows:
        raise RuntimeError(
            f"No fitted kernels for analysis_set_id={args.analysis_set_id}"
        )

    fig, ax = plt.subplots(figsize=(5, 4))
    for row in rows:
        weights_mean = np.asarray(row["weights_mean"], dtype=float)
        weights_error = np.asarray(row["weights_error"], dtype=float)
        x = range(len(weights_mean))
        label = f"{row['subject_name']} (n={row['n_trials_fit']})"
        ax.plot(x, weights_mean, label=label)
        ax.fill_between(
            x,
            weights_mean - weights_error,
            weights_mean + weights_error,
            alpha=0.2,
        )
    ax.axhline(0, color="k", alpha=0.3, linestyle="--")
    ax.set_xlabel("Stimulus time bin")
    ax.set_ylabel("Choice weight")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Psychophysical kernels by subject")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")


if __name__ == "__main__":
    main()
