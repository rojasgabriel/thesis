"""Diagnostic plots for the session psychophysical-kernel table.

Reads ``PsychophysicalKernel`` for a subject/session and writes:
- kernel coefficient vs stimulus-event time
- contributing trial counts per bin
- wait-time / response-time summary statistics from the stored row
"""

from __future__ import annotations

import argparse
import os
import sys
import types
from pathlib import Path

import matplotlib

REPO_ROOT = Path(__file__).resolve().parents[2]
if "ephys" not in sys.modules:
    package = types.ModuleType("ephys")
    package.__path__ = [str(REPO_ROOT)]
    sys.modules["ephys"] = package
sys.path.insert(0, str(REPO_ROOT))

FIGURE_ROOT = Path(os.environ.get("EPHYS_FIGURE_ROOT", str(REPO_ROOT / "figures")))
FIGURE_DIR = FIGURE_ROOT / "psychophysical_kernels"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="GRB006")
    parser.add_argument("--session", default="20240821_121447")
    parser.add_argument("--kernel-param-id", default="v1_100ms_10bin")
    parser.add_argument("--populate", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output figure path (default under figures/psychophysical_kernels/).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matplotlib.use("Agg")
    import numpy as np
    from matplotlib import pyplot as plt

    from labdata_plugin.analysisschema import PsychophysicalKernel

    key = {
        "subject_name": args.subject,
        "session_name": args.session,
        "kernel_param_id": args.kernel_param_id,
    }
    if args.populate:
        PsychophysicalKernel.populate(key, display_progress=True)

    relation = PsychophysicalKernel() & key
    if not relation:
        raise RuntimeError(
            f"No PsychophysicalKernel row for {key}. "
            "Re-run with --populate once EventMapping + Chipmunk data are available."
        )
    row = relation.fetch1()

    centers = np.asarray(row["bin_centers_s"], dtype=float)
    weights = np.asarray(row["weights_mean"], dtype=float)
    errors = np.asarray(row["weights_error"], dtype=float)
    n_obs = np.asarray(row["n_observed_per_bin"], dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(6.5, 6.0), sharex=True)
    ax = axes[0]
    ax.plot(centers, weights, color="C0", lw=2)
    finite = np.isfinite(weights) & np.isfinite(errors)
    ax.fill_between(
        centers[finite],
        (weights - errors)[finite],
        (weights + errors)[finite],
        color="C0",
        alpha=0.2,
    )
    ax.axhline(0, color="k", alpha=0.3, linestyle="--", lw=1)
    ax.axvline(0.0, color="0.5", ls=":", lw=1, label="first stim")
    ax.axvline(0.5, color="0.7", ls=":", lw=1, label="0.5 s")
    ax.legend(frameon=False, fontsize=8, loc="best")
    ax.set_ylabel("Choice weight")
    ax.set_title(
        f"{args.subject} {args.session} — {row['interpretation']} "
        f"(n={row['n_trials_fit']}/{row['n_trials']}, score={row['score_mean']:.2f})"
    )

    ax = axes[1]
    ax.bar(centers, n_obs, width=0.08, color="0.55", align="center")
    ax.set_xlabel("Time from first stimulus (s)")
    ax.set_ylabel("Trials contributing (n)")
    ax.set_title(
        "Wait "
        f"{row['wait_time_mean']:.3f}±{row['wait_time_std']:.3f}s; "
        "response "
        f"{row['response_time_mean']:.3f}±{row['response_time_std']:.3f}s"
    )
    fig.tight_layout()

    output = args.output
    if output is None:
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        output = FIGURE_DIR / f"{args.subject}_{args.session}_psychophysical_kernel.png"
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", dpi=150)
    print(f"Wrote {output}")
    print(f"interpretation={row['interpretation']}")
    print(f"n_observed_per_bin={n_obs.astype(int).tolist()}")


if __name__ == "__main__":
    main()
