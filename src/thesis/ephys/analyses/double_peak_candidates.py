"""Show candidate double-peak responses for one ephys session.

The comparison is the trial-mean response to 15 ms first flashes. Units pass
quality criterion 1, stability parameter 0, and the stored excited
stimulus-response classification. A unit is a candidate only if its trial-mean
response has exactly two peaks from 0 to 120 ms after flash onset. The peaks
are at least 20 ms apart. Each peak has a prominence of at least 1 sp/s and at
least 25% of the maximum baseline-subtracted response in that window. Each peak
is at least 5 sp/s above baseline. Pulses with unknown width are excluded. Some
sessions store only pulse onsets (GRB006). These sessions have no measured
width, so the code sets every pulse to 15 ms.
Spikes align to measured flash onset; the dotted line is the -40 to 0 ms
baseline. The sampling unit for the SEM band is the first flash trial.
This screen does not test whether pulse offset causes the second peak.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from string import ascii_lowercase


def plot_candidates(session_data: dict, output: Path) -> None:
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    responses = list(session_data["unit_responses"].values())
    if not responses:
        raise RuntimeError("No double-peak candidates found for this session")
    if len(responses) > len(ascii_lowercase):
        raise ValueError("Too many candidates for single-letter panel labels")

    nrows = (len(responses) + 2) // 3
    fig, axes = plt.subplots(nrows, 3, figsize=(9, 2.8 * nrows), squeeze=False)
    for index, (ax, response) in enumerate(zip(axes.flat, responses)):
        trials = np.asarray(response["peth_15"])
        times_ms = np.asarray(response["bin_centers"]) * 1000
        mean = trials.mean(axis=0)
        sem = trials.std(axis=0) / np.sqrt(len(trials))
        peaks = response["peak_row_15"]
        ax.axvspan(0, 15, color="#b8d8ed", alpha=0.65, linewidth=0)
        ax.fill_between(
            times_ms, mean - sem, mean + sem, color="0.5", alpha=0.22, linewidth=0
        )
        ax.plot(times_ms, mean, color="0.15", linewidth=1.8)
        ax.scatter(
            np.asarray(peaks["peak_times"]) * 1000,
            peaks["peak_heights"],
            color="#c43c39",
            s=28,
            zorder=3,
        )
        ax.axhline(peaks["baseline"], color="0.65", linestyle=":", linewidth=0.7)
        ax.set_xlim(-100, 150)
        ax.set_ylim(bottom=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(
            0.04,
            0.95,
            f"unit {response['unit_id']}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
        )
        ax.text(
            -0.10,
            1.08,
            ascii_lowercase[index],
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=10,
        )
        ax.set_xlabel("Time from first flash (ms)")
        if index % 3 == 0:
            ax.set_ylabel("Firing rate (sp/s)")
    for ax in list(axes.flat)[len(responses) :]:
        ax.axis("off")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
    print(
        f"{len(responses)} candidates from {responses[0]['n_15ms']} confirmed "
        f"15 ms first flashes: {[r['unit_id'] for r in responses]}"
    )
    print(f"Saved {output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show double-peak candidates for one session"
    )
    parser.add_argument("subject")
    parser.add_argument("session")
    parser.add_argument("--output", type=Path, help="Output SVG path")
    args = parser.parse_args()

    from thesis.ephys.analyses.double_peaks import _collect_session

    output = args.output or (
        Path("figures/double_peak") / f"{args.subject.lower()}_candidates.svg"
    )
    plot_candidates(
        _collect_session(args.subject, args.session, controls=False), output
    )


if __name__ == "__main__":
    main()
