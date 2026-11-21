"""Test whether the GLM kernels of a tracked unit are the same on different days.

Question: is the kernel of a model term a stable property of a unit? Each day
is a separate fit of the V1 GLM, so a kernel that is the same for the same
unit on two days is not noise of one fit. `kernel_reliability.py` asks the
same for two halves of one session; this test needs no second fit.

Comparison: for every pair of fitted days of one mouse and every tracked unit
fitted on both days, the kernel of the unit on the two days (same unit)
against the kernel of the unit on the first day and of each other tracked unit
on the second day (other unit). The other-unit pairs come from the same two
days, so a difference between the days is in both. The comparison is paired
by unit.

- Units: every unit of the final fit (`fit.py`, unit set `all`) that is in a
  UnitMatch track (`<mouse>_ids.csv`, column `track`). Tracks with two units
  in one session are excluded. A unit needs two or more fitted days.
- Kernel of a term: `results.fitted_kernels` on the final fit (Δ log rate).
  A term with one column (initiation, previous choice) is one weight. The
  spike history, the session drift and the video components are not compared
  (the saved fit has no history scale; drift and components are not the same
  regressor on two days). No baseline and no event alignment are involved.
- Reliability of a term: `1 - Σ|same-unit difference|² / Σ|other-unit
  difference|²`, summed over units and pairs of days. 0: a unit is no more
  like itself than like another unit; 1: the same kernel on both days. A wrong
  UnitMatch pair lowers it.
- Shape correlation: Pearson correlation of the two kernels over the time
  points, median over pairs, for same-unit and other-unit pairs.
- The sampling unit is the tracked unit: the 95% interval resamples tracked
  units (2000 draws). One mouse per run.

Figure: (a) the kernels of one tracked unit on each of its days, for the most
and the least reliable kernel (the unit with the median same-unit difference
among the units fitted on the most days); (b) the reliability of each term.

Writes `results/glm/kernel_retest_<mouse>.json` and
`figures/glm/kernel_retest_<mouse>.svg`. Each argument is the fit root of one
day (the directory with `common_design_me.json`):

    uv run python -m thesis.ephys.analyses.glm.kernel_retest GRB064 <root> ...
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from thesis.ephys.analyses.glm import dmat
from thesis.ephys.analyses.glm.fit import load_unit_records
from thesis.ephys.analyses.glm.results import fitted_kernels
from thesis.plotting import (
    BOOTSTRAPS,
    CONCEPT_COLORS,
    FIGURE_STYLE,
    label_panels,
    save_figure,
)

MATCH = Path("results/unit_match")
OUT = Path("results/glm")
SKIPPED = ("history", dmat.DRIFT)
# Label and concept of each term (the colors of the other GLM figures).
TERMS = {
    dmat.FLASH_COMMON: ("flash", "visual"),
    dmat.FLASH_JUST_BEFORE: ("flash × short gap", "visual"),
    dmat.FLASH_AFTER_EXIT: ("flash × after exit", "visual"),
    "flash_recovery": ("flash × recovery", "visual"),
    "flash_side": ("flash × turn side", "visual"),
    "hold": ("fixation", "task"),
    dmat.FIXATION: ("initiation", "task"),
    dmat.PREVIOUS_SIDE: ("previous choice", "task"),
    "response_period": ("response period", "motion energy"),
    "turn_direction": ("turn direction", "motion energy"),
}


def day_kernels(root: Path, tracks: pd.Series) -> tuple[dict, dict]:
    """Return {term: DataFrame (track × time)} and {term: time (ms)} of one day."""
    metadata = json.loads((root / "common_design_me.json").read_text())
    rows, times = {}, {}
    for record in load_unit_records(
        root / "all_fit_me/checkpoints", "unit_*_final.json"
    ):
        unit = int(record["unit_id"])
        if unit not in tracks.index:
            continue
        # The history kernel is skipped, so its scale is not needed.
        kernels = fitted_kernels(record, metadata, np.ones(dmat.HISTORY_COLUMNS))
        for name, (time, values) in kernels.items():
            if name not in SKIPPED:
                rows.setdefault(name, {})[int(tracks[unit])] = values
                times[name] = time
    return {
        n: pd.DataFrame.from_dict(r, orient="index") for n, r in rows.items()
    }, times


def reliability(sums: np.ndarray) -> float:
    """Return 1 - Σ same / Σ other from the (track, 2) sums of squared differences.

    >>> reliability(np.array([[1.0, 4.0], [1.0, 4.0]]))
    0.75
    """
    return float(1 - sums[:, 0].sum() / sums[:, 1].sum())


def compare(days: dict[str, pd.DataFrame], seed: int = 0) -> dict:
    """Compare the kernels of one term between every pair of days."""
    sums: dict[int, np.ndarray] = {}
    same, other = [], []
    sessions = sorted(days)
    for i, first in enumerate(sessions):
        for second in sessions[i + 1 :]:
            shared = days[first].index.intersection(days[second].index)
            if len(shared) < 2:
                continue
            a = days[first].loc[shared].to_numpy()
            b = days[second].loc[shared].to_numpy()
            distance = ((a[:, None] - b[None]) ** 2).sum(-1)  # (unit on a, unit on b)
            off = ~np.eye(len(shared), dtype=bool)
            if np.ptp(a, axis=1).any():  # a kernel, not one weight
                r = np.corrcoef(a, b)[: len(shared), len(shared) :]
                same += list(np.diag(r))
                other += list(r[off])
            for j, track in enumerate(shared):
                entry = sums.setdefault(int(track), np.zeros(2))
                entry += distance[j, j], distance[j, off[j]].mean()
    values = np.asarray(list(sums.values()))
    rng = np.random.default_rng(seed)
    draws = [
        reliability(values[rng.integers(0, len(values), len(values))])
        for _ in range(BOOTSTRAPS)
    ]
    out = {
        "tracked_units": len(values),
        "same_unit_pairs_of_days": len(same) or None,
        "reliability": reliability(values),
        "interval_95": [float(x) for x in np.percentile(draws, [2.5, 97.5])],
    }
    if same:
        out |= {
            "median_shape_r_same_unit": float(np.nanmedian(same)),
            "median_shape_r_other_unit": float(np.nanmedian(other)),
        }
    return out, sums


def example_track(days: dict[str, pd.DataFrame], sums: dict) -> int:
    """Return the unit with the median same-unit difference among the most-fitted."""
    count = pd.concat([d.index.to_series() for d in days.values()]).value_counts()
    most = count[count == count.max()].index
    order = sorted(most, key=lambda t: sums[int(t)][0])
    return int(order[len(order) // 2])


def plot(mouse: str, kernels: dict, times: dict, report: dict, sums: dict) -> Path:
    terms = sorted(report["terms"], key=lambda n: report["terms"][n]["reliability"])
    curves = [n for n in terms if n in times and len(times[n]) > 2]
    shown = (curves[-1], curves[0])  # the most and the least reliable kernel
    track = example_track(kernels[shown[0]], sums[shown[0]])
    with plt.rc_context(FIGURE_STYLE):
        fig = plt.figure(figsize=(7.2, 3.2), layout="constrained")
        left, right = fig.subfigures(1, 2, width_ratios=(1, 1.25))
        axes = left.subplots(2, 1, sharex=True)
        for axis, name in zip(axes, shown):
            label, concept = TERMS[name]
            sessions = [
                s for s in sorted(kernels[name]) if track in kernels[name][s].index
            ]
            for k, session in enumerate(sessions):
                axis.plot(
                    times[name],
                    kernels[name][session].loc[track],
                    color=CONCEPT_COLORS[concept],
                    alpha=0.35 + 0.65 * k / max(len(sessions) - 1, 1),
                    lw=1,
                )
            axis.axhline(0, color="0.65", lw=0.6, ls="--")
            axis.set_ylabel("Δ log rate")
            axis.set_title(
                f"{label} kernel", color=CONCEPT_COLORS[concept], fontsize=8, loc="left"
            )
            axis.set_title(
                f"one tracked unit, {len(sessions)} days (dark = late)"
                if axis is axes[0]
                else "",
                fontsize=7,
                loc="right",
            )
        axes[1].set_xlabel("time from flash (ms)")
        ax = right.subplots()
        for row, name in enumerate(terms):
            entry = report["terms"][name]
            label, concept = TERMS[name]
            color = CONCEPT_COLORS[concept]
            ax.plot(entry["interval_95"], [row, row], color=color, lw=1.2)
            ax.plot(entry["reliability"], row, "o", color=color, ms=5)
        ax.set_yticks(range(len(terms)), [TERMS[n][0] for n in terms])
        for tick, name in zip(ax.get_yticklabels(), terms):
            tick.set_color(CONCEPT_COLORS[TERMS[name][1]])
        ax.set_xlim(0, 1)
        ax.set_xlabel(
            "reliability across days\n(0 = like another unit, 1 = same kernel)"
        )
        ax.text(
            0.02,
            0.98,
            f"{report['tracked_units']} tracked units, {report['days']} days, 1 mouse",
            transform=ax.transAxes,
            va="top",
            fontsize=7,
        )
        label_panels([axes[0], ax], x=-0.2, y=1.08)
        return save_figure(fig, OUT / f"kernel_retest_{mouse}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mouse")
    parser.add_argument("roots", nargs="+", type=Path, help="fit root of each day")
    args = parser.parse_args()
    ids = pd.read_csv(MATCH / f"{args.mouse}_ids.csv", dtype={"session": str})
    ids = ids[
        ~ids.groupby("track")["session"].transform(lambda s: s.duplicated().any())
    ]
    kernels: dict[str, dict[str, pd.DataFrame]] = {}
    times: dict = {}
    for root in args.roots:
        session = re.search(rf"{args.mouse}_(\d+_\d+)", str(root)).group(1)
        tracks = ids[ids["session"] == session].set_index("unit_id")["track"]
        day, day_times = day_kernels(root, tracks)
        times |= day_times
        for name, frame in day.items():
            kernels.setdefault(name, {})[session] = frame
    report = {"mouse": args.mouse, "days": len(args.roots), "terms": {}}
    sums = {}
    for name in kernels:
        report["terms"][name], sums[name] = compare(kernels[name])
    report["tracked_units"] = max(t["tracked_units"] for t in report["terms"].values())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"kernel_retest_{args.mouse}.json").write_text(json.dumps(report, indent=1))
    for name, entry in report["terms"].items():
        low, high = entry["interval_95"]
        print(
            f"{TERMS[name][0]:20s} {entry['reliability']:+.2f} [{low:+.2f}, {high:+.2f}]"
        )
    print(plot(args.mouse, kernels, times, report, sums))


if __name__ == "__main__":
    main()
