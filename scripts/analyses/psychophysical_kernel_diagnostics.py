"""Diagnostic plots for the session psychophysical-kernel table.

Reads ``PsychophysicalKernel`` for a subject/session and writes:
- kernel coefficient vs stimulus-event time
- contributing trial counts per bin
- wait-time / response-time summary statistics from the stored row

By default plots both observation windows when present:
``center_exit`` (fixation only) and ``response`` (through response poke).
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

DEFAULT_PARAM_IDS = (
    "v1_100ms_10bin_center",
    "v1_100ms_10bin_response",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="GRB006")
    parser.add_argument("--session", default="20240821_121447")
    parser.add_argument(
        "--kernel-param-id",
        nargs="+",
        default=list(DEFAULT_PARAM_IDS),
        help="One or more PsychophysicalKernelParam ids to plot/populate.",
    )
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

    from labdata_plugin.analysisschema import (
        PsychophysicalKernel,
        PsychophysicalKernelParam,
    )

    if args.populate:
        for param_id in args.kernel_param_id:
            key = {
                "subject_name": args.subject,
                "session_name": args.session,
                "kernel_param_id": param_id,
            }
            PsychophysicalKernel.populate(key, display_progress=True)

    rows = []
    for param_id in args.kernel_param_id:
        key = {
            "subject_name": args.subject,
            "session_name": args.session,
            "kernel_param_id": param_id,
        }
        relation = PsychophysicalKernel() & key
        if not relation:
            print(f"Missing PsychophysicalKernel row for {key}")
            continue
        row = relation.fetch1()
        window = (PsychophysicalKernelParam() & key).fetch1("observation_window")
        row["observation_window"] = window
        rows.append(row)

    if not rows:
        raise RuntimeError(
            f"No PsychophysicalKernel rows for {args.subject} {args.session} "
            f"params={args.kernel_param_id}. Re-run with --populate once "
            "EventMapping + Chipmunk data are available."
        )

    fig, axes = plt.subplots(2, 1, figsize=(6.5, 6.5), sharex=True)
    colors = {"center_exit": "C0", "response": "C3"}
    labels = {
        "center_exit": "until center exit",
        "response": "until response",
    }

    ax = axes[0]
    for row in rows:
        centers = np.asarray(row["bin_centers_s"], dtype=float)
        weights = np.asarray(row["weights_mean"], dtype=float)
        errors = np.asarray(row["weights_error"], dtype=float)
        window = row["observation_window"]
        color = colors.get(window, "C1")
        ax.plot(
            centers,
            weights,
            color=color,
            lw=2,
            label=f"{labels.get(window, window)} ({row['interpretation']})",
        )
        finite = np.isfinite(weights) & np.isfinite(errors)
        ax.fill_between(
            centers[finite],
            (weights - errors)[finite],
            (weights + errors)[finite],
            color=color,
            alpha=0.15,
        )
    ax.axhline(0, color="k", alpha=0.3, linestyle="--", lw=1)
    ax.axvline(0.0, color="0.5", ls=":", lw=1)
    ax.axvline(0.5, color="0.7", ls=":", lw=1)
    ax.legend(frameon=False, fontsize=8, loc="best")
    ax.set_ylabel("Choice weight")
    ax.set_title(f"{args.subject} {args.session}")

    ax = axes[1]
    width = 0.04
    offsets = np.linspace(-width, width, num=max(1, len(rows)))
    for offset, row in zip(offsets, rows):
        centers = np.asarray(row["bin_centers_s"], dtype=float)
        n_obs = np.asarray(row["n_observed_per_bin"], dtype=float)
        window = row["observation_window"]
        color = colors.get(window, "C1")
        ax.bar(
            centers + offset,
            n_obs,
            width=width * 1.6,
            color=color,
            alpha=0.7,
            align="center",
            label=labels.get(window, window),
        )
    ax.set_xlabel("Time from first stimulus (s)")
    ax.set_ylabel("Trials contributing (n)")
    summary = rows[0]
    ax.set_title(
        "Wait "
        f"{summary['wait_time_mean']:.3f}±{summary['wait_time_std']:.3f}s; "
        "response "
        f"{summary['response_time_mean']:.3f}±{summary['response_time_std']:.3f}s"
    )
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()

    output = args.output
    if output is None:
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        output = FIGURE_DIR / f"{args.subject}_{args.session}_psychophysical_kernel.png"
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", dpi=150)
    print(f"Wrote {output}")
    for row in rows:
        n_obs = np.asarray(row["n_observed_per_bin"], dtype=int)
        print(
            f"{row['observation_window']}: interpretation={row['interpretation']} "
            f"n_observed_per_bin={n_obs.tolist()}"
        )


if __name__ == "__main__":
    main()
